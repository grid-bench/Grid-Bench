"""Run a provider over a dataset and record raw plus extracted predictions."""

from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..diagnostics import DIAGNOSTICS_FILE, read_diagnostics
from ..prompt import COMPLETION_PROMPT_VERSION as PROMPT_VERSION
from ..prompt import render_prompt
from ..schema import GoldRecord, Instance, parse_json_document
from .extract import extract_answer
from .providers import (
    DETERMINISTIC_FAILURES,
    FAILURE_KINDS,
    Completion,
    Provider,
    RetryableError,
    make_provider,
)

CONDITION_FIELDS = ("provider", "model", "representation", "prompt_version", "request")


def failure_kind_of(record: Mapping[str, Any]) -> str | None:
    """Failure class of a prediction record, inferred for records written before the field."""

    if record.get("error") is None:
        return None
    kind = record.get("failure_kind")
    if isinstance(kind, str) and kind:
        return kind
    error = str(record["error"])
    if record.get("finish_reason") in ("length", "max_tokens") or "max_tokens" in error:
        return "budget"
    if "no completion within" in error or "timed out" in error.lower():
        return "deadline"
    if error == "empty response":
        return "empty"
    if error.startswith("refusal"):
        return "refusal"
    if error.startswith("status "):
        return "status"
    return "transport"


@dataclass(slots=True)
class Dataset:
    instances: list[Instance]
    gold: dict[str, GoldRecord] = field(default_factory=dict)
    traces: dict[str, dict[str, Any]] = field(default_factory=dict)
    # history-retention diagnostics per rollout id, when ``diagnostics.jsonl``
    # exists next to the manifest (see ``spatial-worlds diagnose``)
    diagnostics: dict[str, dict[str, Any]] = field(default_factory=dict)

    def gold_for(self, instance: Instance) -> GoldRecord | None:
        return self.gold.get(instance.id)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield parse_json_document(line)


def condition_id(
    *,
    provider: object,
    model: object,
    representation: object,
    prompt_version: object,
    request: object,
) -> str:
    """Return the stable identity of one model/prompt/request condition."""

    payload = {
        "provider": provider,
        "model": model,
        "representation": representation,
        "prompt_version": prompt_version,
        "request": request,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def record_condition_id(record: Mapping[str, Any]) -> str | None:
    """Read a condition id, deriving it for pre-id runner records when possible."""

    value = record.get("condition_id")
    if value is not None:
        return value if isinstance(value, str) and value else None
    if all(field in record for field in CONDITION_FIELDS):
        return condition_id(
            provider=record["provider"],
            model=record["model"],
            representation=record["representation"],
            prompt_version=record["prompt_version"],
            request=record["request"],
        )
    return None


def _manifest_files(data: Path) -> tuple[list[Path], list[Path], list[Path]] | None:
    manifest_path = data / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = parse_json_document(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError(f"{manifest_path} must contain a JSON object")
    split = manifest.get("split")
    cells = manifest.get("cells")
    if not isinstance(split, str) or not split or "/" in split or split in {".", ".."}:
        raise ValueError(f"{manifest_path} has an invalid split")
    if not isinstance(cells, list):
        raise ValueError(f"{manifest_path} is missing a cells array")
    families: set[str] = set()
    for index, cell in enumerate(cells):
        if not isinstance(cell, Mapping):
            raise ValueError(f"{manifest_path} cells[{index}] must be an object")
        rollouts = cell.get("rollouts")
        if not isinstance(rollouts, int) or isinstance(rollouts, bool) or rollouts < 0:
            raise ValueError(f"{manifest_path} cells[{index}].rollouts must be nonnegative")
        if rollouts == 0:
            continue
        family = cell.get("family")
        if not isinstance(family, str) or not family or "/" in family or family in {".", ".."}:
            raise ValueError(f"{manifest_path} cells[{index}] has an invalid family")
        families.add(family)
    roots = [data / split / family for family in sorted(families)]
    return (
        [root / "instances.jsonl" for root in roots],
        [root / "gold.jsonl" for root in roots],
        [root / "trace.jsonl" for root in roots],
    )


def load_dataset(
    data: Path | None = None,
    *,
    instances: Sequence[Path] | None = None,
    gold: Sequence[Path] | None = None,
    traces: Sequence[Path] | None = None,
) -> Dataset:
    """Load a dataset directory (``split/family/*.jsonl``) or explicit JSONL files."""

    instance_files = list(instances or [])
    gold_files = list(gold or [])
    trace_files = list(traces or [])
    if data is not None:
        manifest_files = _manifest_files(data)
        if manifest_files is None:
            instance_files.extend(sorted(data.glob("*/*/instances.jsonl")))
            gold_files.extend(sorted(data.glob("*/*/gold.jsonl")))
            trace_files.extend(sorted(data.glob("*/*/trace.jsonl")))
        else:
            manifest_instances, manifest_gold, manifest_traces = manifest_files
            instance_files.extend(manifest_instances)
            gold_files.extend(manifest_gold)
            trace_files.extend(manifest_traces)
    if not instance_files:
        raise ValueError("no instance files found")
    dataset = Dataset(instances=[])
    if data is not None:
        dataset.diagnostics = read_diagnostics(data / DIAGNOSTICS_FILE)
    instance_ids: set[str] = set()
    for path in instance_files:
        for record in iter_jsonl(path):
            item = Instance.from_dict(record)
            if item.id in instance_ids:
                raise ValueError(f"duplicate instance id {item.id!r} (while reading {path})")
            instance_ids.add(item.id)
            dataset.instances.append(item)
    for path in gold_files:
        for record in iter_jsonl(path):
            item = GoldRecord.from_dict(record)
            if item.instance_id in dataset.gold:
                raise ValueError(
                    f"duplicate gold instance_id {item.instance_id!r} (while reading {path})"
                )
            dataset.gold[item.instance_id] = item
    for path in trace_files:
        for record in iter_jsonl(path):
            rollout_id = str(record["rollout_id"])
            if rollout_id in dataset.traces:
                raise ValueError(
                    f"duplicate trace rollout_id {rollout_id!r} (while reading {path})"
                )
            dataset.traces[rollout_id] = record
    instances_by_id = {instance.id: instance for instance in dataset.instances}
    for instance_id, item in dataset.gold.items():
        instance = instances_by_id.get(instance_id)
        if instance is None:
            raise ValueError(f"gold record {instance_id!r} has no matching instance")
        if item.family != instance.family:
            raise ValueError(
                f"gold record {instance_id!r} family {item.family!r} does not match "
                f"instance family {instance.family!r}"
            )
        if item.world != instance.world:
            raise ValueError(f"gold record {instance_id!r} world does not match its instance")
    return dataset


def _rollout_index(instance: Instance) -> int | None:
    """Index of the rollout an instance was cut from (the ``0000`` in its id), if known."""

    prefix = instance.metadata.get("prefix_of")
    if not isinstance(prefix, str):
        return None
    tail = prefix.rsplit("/", 1)[-1]
    return int(tail) if tail.isdigit() else None


_FILTER_KEYS = {
    "family": lambda instance: instance.family,
    "index": _rollout_index,
    "d": lambda instance: instance.dimension,
    "dimension": lambda instance: instance.dimension,
    "T": lambda instance: instance.horizon,
    "horizon": lambda instance: instance.horizon,
    "w": lambda instance: instance.metadata.get("nominal_scan_workload"),
    "tier": lambda instance: instance.metadata.get("tier"),
    "split": lambda instance: instance.metadata.get("split"),
    "level": lambda instance: instance.metadata.get("level"),
}


def parse_filter(text: str | None) -> dict[str, set[str]]:
    filters: dict[str, set[str]] = {}
    if not text:
        return filters
    for part in text.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"filter {part!r} must look like key=value")
        key, value = part.split("=", 1)
        key = key.strip()
        if key not in _FILTER_KEYS:
            raise ValueError(f"unknown filter key {key!r}; expected one of {sorted(_FILTER_KEYS)}")
        filters[key] = {item.strip() for item in value.split("|")}
    return filters


def matches_filter(instance: Instance, filters: Mapping[str, set[str]]) -> bool:
    return all(str(_FILTER_KEYS[key](instance)) in values for key, values in filters.items())


def load_existing(out: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    existing: dict[tuple[str, str, str], dict[str, Any]] = {}
    if out.exists():
        for record in iter_jsonl(out):
            record_condition = record_condition_id(record)
            instance_id = record.get("instance_id")
            prompt_hash = record.get("prompt_sha256")
            if (
                record_condition is not None
                and isinstance(instance_id, str)
                and isinstance(prompt_hash, str)
            ):
                existing[(record_condition, instance_id, prompt_hash)] = record
    return existing


def complete_with_retries(
    provider: Provider,
    prompt: str,
    *,
    instance: Instance,
    gold: GoldRecord | None,
    max_attempts: int,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[Completion, int]:
    attempt = 0
    while True:
        attempt += 1
        try:
            return provider.complete(prompt, instance=instance, gold=gold), attempt
        except RetryableError as error:
            if attempt >= max_attempts:
                return Completion(
                    text=None,
                    error=f"gave up after {attempt} attempts: {error}",
                    failure_kind=error.kind,
                ), attempt
            delay = error.retry_after
            if delay is None:
                delay = min(60.0, 2.0**attempt + random.uniform(0.0, 1.0))
            sleep(delay)


def build_record(
    instance: Instance,
    provider: Provider,
    prompt: str,
    completion: Completion,
    attempts: int,
    *,
    run_id: str,
    representation: str,
    request: Mapping[str, Any],
    condition: str,
) -> dict[str, Any]:
    raw = completion.text
    extraction = extract_answer(raw) if raw is not None else None
    return {
        "instance_id": instance.id,
        "model": provider.model,
        "provider": provider.name,
        "run_id": run_id,
        "condition_id": condition,
        "representation": representation,
        "output_contract": "query" if instance.queries else "cells",
        "prompt_version": PROMPT_VERSION,
        "request": dict(request),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_chars": len(prompt),
        "raw_text": raw,
        "extracted_text": extraction.text if extraction else None,
        "extraction_method": extraction.method if extraction else "failed",
        "extraction_applied": bool(extraction and extraction.applied),
        "usage": dict(completion.usage),
        "finish_reason": completion.finish_reason,
        "latency_s": round(completion.latency_s, 3),
        "attempts": attempts,
        "error": completion.error,
        "failure_kind": completion.failure_kind if completion.error is not None else None,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def parse_retry_kinds(text: str | None) -> frozenset[str] | None:
    """Failure kinds ``--retry-errors`` should re-run; ``None`` means the default set."""

    if text is None or not text.strip():
        return None
    kinds = {item.strip() for item in text.split(",") if item.strip()}
    if "all" in kinds:
        return frozenset(FAILURE_KINDS)
    unknown = kinds - set(FAILURE_KINDS)
    if unknown:
        raise ValueError(
            f"unknown failure kind(s) {sorted(unknown)}; expected {list(FAILURE_KINDS)} or all"
        )
    return frozenset(kinds)


def should_retry(previous: Mapping[str, Any], retry_kinds: frozenset[str] | None) -> bool:
    """Whether an existing failed record is re-run under ``--retry-errors``.

    Budget and deadline records are skipped by declared policy, not because
    a repeated request must fail in the same way; name them explicitly to retry.
    """

    kind = failure_kind_of(previous)
    if retry_kinds is not None:
        return kind is None or kind in retry_kinds
    return kind not in DETERMINISTIC_FAILURES


def run_predictions(
    dataset: Dataset,
    provider: Provider,
    *,
    out: Path,
    representation: str = "cells",
    concurrency: int = 1,
    resume: bool = True,
    retry_errors: bool = False,
    retry_kinds: frozenset[str] | None = None,
    max_attempts: int = 1,
    limit: int | None = None,
    filters: Mapping[str, set[str]] | None = None,
    run_id: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Query ``provider`` on every selected instance, appending records to ``out``."""

    if "study-v1" in out.resolve().parts:
        raise ValueError("study artifacts require 'study formal'; generic output is isolated")

    run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    # The retry policy belongs to the resource regime and hence to the
    # condition identity, next to the provider's own request parameters.
    request = {**provider.request_params(), "max_attempts": int(max_attempts)}
    current_condition = condition_id(
        provider=provider.name,
        model=provider.model,
        representation=representation,
        prompt_version=PROMPT_VERSION,
        request=request,
    )
    existing = load_existing(out) if resume else {}
    selected: list[tuple[Instance, str]] = []
    skipped = 0
    for instance in dataset.instances:
        if filters and not matches_filter(instance, filters):
            continue
        prompt = render_prompt(
            instance, representation=representation, prompt_version=PROMPT_VERSION
        )
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        previous = existing.get((current_condition, instance.id, prompt_hash))
        if previous is not None and (
            previous.get("error") is None
            or not retry_errors
            or not should_retry(previous, retry_kinds)
        ):
            skipped += 1
            continue
        selected.append((instance, prompt))
        if limit is not None and len(selected) >= limit:
            break

    if representation == "grid":
        too_deep = [instance.id for instance, _ in selected if instance.dimension > 2]
        if too_deep:
            raise ValueError(
                f"grid representation requires dimension <= 2 but {len(too_deep)} selected "
                "instances have d > 2; add --filter d=1|2"
            )
    out.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    errors = 0

    def work(item: tuple[Instance, str]) -> dict[str, Any]:
        instance, prompt = item
        completion, attempts = complete_with_retries(
            provider,
            prompt,
            instance=instance,
            gold=dataset.gold_for(instance),
            max_attempts=max_attempts,
        )
        return build_record(
            instance,
            provider,
            prompt,
            completion,
            attempts,
            run_id=run_id,
            representation=representation,
            request=request,
            condition=current_condition,
        )

    with out.open("a", encoding="utf-8") as stream:
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            futures = [pool.submit(work, item) for item in selected]
            for future in as_completed(futures):
                record = future.result()
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                completed += 1
                if record["error"] is not None:
                    errors += 1
                if progress is not None and (completed % 25 == 0 or completed == len(selected)):
                    progress(f"{completed}/{len(selected)} completed, {errors} errors")
    return {
        "run_id": run_id,
        "model": provider.model,
        "provider": provider.name,
        "condition_id": current_condition,
        "selected": len(selected),
        "completed": completed,
        "errors": errors,
        "skipped_existing": skipped,
        "out": str(out),
    }


def parse_params(items: Sequence[str]) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--param {item!r} must look like key=value")
        key, value = item.split("=", 1)
        try:
            extra[key.strip()] = json.loads(value)
        except json.JSONDecodeError:
            extra[key.strip()] = value
    return extra


def run_from_args(args: Any) -> int:
    dataset = load_dataset(args.data, instances=args.instances, gold=args.gold)
    provider = make_provider(
        args.provider,
        args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        max_tokens=args.max_tokens,
        effort=args.effort,
        thinking_budget=args.thinking_budget,
        temperature=args.temperature,
        timeout=args.timeout,
        extra=parse_params(args.param),
    )
    if args.provider == "oracle" and not dataset.gold:
        raise ValueError("the oracle provider needs gold records (--data or --gold)")
    progress = None if args.quiet else (lambda message: print(message, file=sys.stderr))
    summary = run_predictions(
        dataset,
        provider,
        out=args.out,
        representation=args.representation,
        concurrency=args.concurrency,
        retry_errors=args.retry_errors,
        retry_kinds=parse_retry_kinds(args.retry_kinds),
        max_attempts=args.max_attempts,
        limit=args.limit,
        filters=parse_filter(args.filter),
        progress=progress,
    )
    print(json.dumps(summary))
    return 0
