"""Formal collection with frozen campaign masks and resumable execution stages."""

from __future__ import annotations

import argparse
import copy
import json
import re
import signal
import sqlite3
import threading
import time
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..prompt import render_prompt
from .protocol import AttemptResult, consume_attempt, openai_events
from .study_final_bedrock import BedrockGateway, bedrock_events
from .study_final_config import (
    CONFIG_PATH,
    build_manifest,
    formal_scopes,
    lineage_digest,
    load_config,
    model_order,
    paths,
    pending_jobs,
)
from .study_final_gateway import Gateway, reservation
from .study_final_store import FinalStore, FormalRunError, job_key, microdollars, timestamp

SUPERSEDED_SCOPES = (
    "formal128:formal-128k-high-sol-c2-continuation-02",
    "formal128:formal-128k-high-sol-mini-c2-continuation-04",
)
FINISH_REASONS = {
    "stop",
    "length",
    "max_tokens",
    "end_turn",
    "stop_sequence",
    "content_filter",
    "refusal",
}
SAFE_RESPONSE_ID = re.compile(
    r"(?:(?:chatcmpl-|msg_|resp_|req_|cmpl-)[A-Za-z0-9_.:-]{1,120}|[0-9]{1,30})"
)
JOB_IDENTITY_FIELDS = (
    "model",
    "provider",
    "dataset",
    "stage",
    "instance_id",
    "T",
    "prompt_sha256",
    "instance_sha256",
    "gold_sha256",
    "condition_id",
    "request",
)


def add_parser(commands: Any) -> None:
    parser = commands.add_parser("formal", help="run a fixed formal benchmark campaign")
    actions = parser.add_subparsers(dest="formal_command", required=True)
    for name in ("run", "status", "evaluate", "stop"):
        command = actions.add_parser(name)
        command.add_argument("--config", type=Path, default=CONFIG_PATH)
        if name == "run":
            command.add_argument("--execute", action="store_true", help="allow paid model requests")
            command.add_argument(
                "--stage",
                choices=("pilot", "full"),
                help="Sonnet defaults to its serial pilot; full continues after the pilot",
            )
            command.add_argument(
                "--operational-per-model-concurrency",
                type=int,
                help="audited full-stage runtime override; does not change the model condition",
            )
            command.add_argument(
                "--operational-total-concurrency",
                type=int,
                help="audited full-stage runtime override; must accompany the per-model value",
            )


def _write_json(path: Path, value: Any, *, immutable: bool = False) -> None:
    if immutable and path.exists():
        if json.loads(path.read_text()) != value:
            raise FormalRunError(f"immutable {path.name} differs from the active run")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def _result(row: dict[str, Any]) -> AttemptResult | None:
    return AttemptResult(**json.loads(row["result"])) if row.get("result") else None


def is_model_outcome(result: AttemptResult, job: dict[str, Any]) -> bool:
    """A delivered model outcome is never retried, regardless of its contents."""
    expected = set(job["request"]["expected_model"])
    output = result.usage.get("output_tokens")
    return bool(
        result.answer_eligible
        and result.error is None
        and result.protocol_violation is None
        and result.termination in {"stop", "length"}
        and result.finish_reason in FINISH_REASONS
        and isinstance(result.response.get("id"), str)
        and SAFE_RESPONSE_ID.fullmatch(result.response["id"])
        and result.response.get("model") in expected
        and not (isinstance(output, int) and not isinstance(output, bool) and output > 128000)
    )


def is_retryable_technical(result: AttemptResult) -> bool:
    if result.answer_eligible and result.finish_reason in FINISH_REASONS:
        return False
    return bool(
        result.retryable
        or result.termination in {"transport", "deadline"}
        or result.error in {"whole_attempt_deadline", "stream_ended_without_finish_marker"}
    )


def _technical_reason(result: AttemptResult, job: dict[str, Any]) -> str:
    if is_model_outcome(result, job):
        return "model_outcome"
    if result.protocol_violation:
        return str(result.protocol_violation)
    if result.response.get("model") not in set(job["request"]["expected_model"]):
        return "response_model_missing_or_changed"
    if not SAFE_RESPONSE_ID.fullmatch(str(result.response.get("id") or "")):
        return "response_id_missing_or_invalid"
    return str(result.error or result.termination or "technical_failure")


def _matches_job(row: dict[str, Any], job: dict[str, Any]) -> bool:
    saved = json.loads(row["job"])
    return row["job_key"] == job_key(job) and all(
        saved.get(field) == job.get(field) for field in JOB_IDENTITY_FIELDS
    )


def _validate_lineage(
    store: FinalStore, manifest: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, tuple[dict[str, Any], AttemptResult]]]:
    """Validate immutable source attempts and return their unique model outcomes."""
    expected = {job_key(job): job for job in manifest["jobs"]}
    all_rows: list[dict[str, Any]] = []
    outcomes: dict[str, tuple[dict[str, Any], AttemptResult]] = {}
    for source in manifest["lineage"]:
        rows = store.all_attempts(source["scope"])
        if len(rows) != int(source["attempts"]) or lineage_digest(rows) != source["sha256"]:
            raise FormalRunError(f"frozen lineage changed: {source['campaign_id']}")
        source_outcomes = Counter()
        for row in rows:
            job = expected.get(row["job_key"])
            if job is None or not _matches_job(row, job):
                raise FormalRunError("lineage call differs from the frozen formal job")
            result = _result(row)
            if result is None:
                raise FormalRunError("frozen lineage contains an in-flight call")
            if is_model_outcome(result, job):
                if row["job_key"] in outcomes:
                    raise FormalRunError("a formal job has more than one lineage outcome")
                outcomes[row["job_key"]] = (row, result)
                source_outcomes[row["model"]] += 1
        if {model: source_outcomes[model] for model in model_order(manifest)} != source["outcomes"]:
            raise FormalRunError("frozen lineage outcome counts changed")
        all_rows.extend(rows)
    total = Counter(row["model"] for row, _ in outcomes.values())
    if {model: total[model] for model in model_order(manifest)} != manifest["lineage_outcomes"]:
        raise FormalRunError("combined lineage outcomes changed")
    pending_jobs(manifest, set(outcomes))
    for scope in SUPERSEDED_SCOPES:
        if store.all_attempts(scope):
            raise FormalRunError("a superseded formal continuation contains attempts")
    return all_rows, outcomes


def _continuation_state(
    store: FinalStore,
    manifest: dict[str, Any],
    lineage_outcomes: dict[str, tuple[dict[str, Any], AttemptResult]],
) -> tuple[list[dict[str, Any]], dict[str, tuple[dict[str, Any], AttemptResult]]]:
    """Validate every active-continuation call against the frozen remaining mask."""
    expected = {
        job_key(job): job
        for jobs in pending_jobs(manifest, set(lineage_outcomes)).values()
        for job in jobs
    }
    rows = store.all_attempts(manifest["phase"])
    outcomes: dict[str, tuple[dict[str, Any], AttemptResult]] = {}
    for row in rows:
        job = expected.get(row["job_key"])
        if job is None or not _matches_job(row, job):
            raise FormalRunError("continuation contains a call outside the frozen remaining mask")
        result = _result(row)
        if result is not None and is_model_outcome(result, job):
            if row["job_key"] in outcomes:
                raise FormalRunError("a formal job has more than one continuation outcome")
            outcomes[row["job_key"]] = (row, result)
    return rows, outcomes


def _capacity_signal(result: AttemptResult, job: dict[str, Any]) -> str | None:
    """Return an infrastructure-capacity signal; model outcomes never qualify."""
    if is_model_outcome(result, job):
        return None
    error = (result.trace or {}).get("error") or {}
    status = error.get("status")
    code = error.get("code")
    if status == 429:
        return "status_429"
    if isinstance(status, int) and not isinstance(status, bool) and 500 <= status < 600:
        return f"status_{status}"
    if code in {"rate_limit_exceeded", "server_error", "internal_error"}:
        return str(code)
    if (
        result.termination == "transport"
        and result.retryable
        and result.error
        not in {"stream_ended_without_finish_marker", "interrupted_process_before_answer"}
    ):
        return "retryable_transport"
    return None


def _retry_delay(result: AttemptResult, attempts: int) -> float:
    fallback = 10 if attempts == 1 else 30
    return min(300.0, max(float(result.retry_after or 0), fallback))


def _pilot_jobs(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    by_id = {
        (job["model"], job["dataset"], job["instance_id"]): job
        for job in manifest["jobs"]
    }
    models = model_order(manifest)
    return [
        by_id[(item.get("model", models[0]), item["dataset"], item["instance_id"])]
        for item in manifest.get("pilot", {}).get("jobs", [])
    ]


def _execution_jobs(
    manifest: dict[str, Any], execution_stage: str, completed: set[str]
) -> list[dict[str, Any]]:
    pilot = _pilot_jobs(manifest)
    if execution_stage == "pilot":
        if not pilot:
            raise FormalRunError("this campaign has no pilot stage")
        return pilot
    if execution_stage != "full":
        raise FormalRunError("unknown formal execution stage")
    if pilot and not {job_key(job) for job in pilot} <= completed:
        raise FormalRunError("all declared pilot outcomes must be saved before the full stage")
    return manifest["jobs"]


def _dispatch_state(
    rows: list[dict[str, Any]], manifest: dict[str, Any], execution_stage: str | None = None
) -> dict[str, Any]:
    if execution_stage is None:
        execution_stage = "full"
        if manifest.get("pilot") and not any(
            (json.loads(row["job"]).get("collection") or {}).get("execution_stage") == "full"
            for row in rows
        ):
            execution_stage = "pilot"
    progress = Counter()
    capacity_signals: list[str] = []
    promoted_seen = False
    operational_override: dict[str, int | str] | None = None
    jobs = {job_key(job): job for job in manifest["jobs"]}
    for row in rows:
        saved = json.loads(row["job"])
        collection = saved.get("collection") or {}
        saved_override = collection.get("operational_concurrency_override")
        if isinstance(saved_override, dict):
            operational_override = saved_override
        if manifest.get("pilot") and collection.get("execution_stage") == "pilot":
            # Serial pilot outcomes do not establish full-stage capacity.
            continue
        promoted_seen = promoted_seen or collection.get("dispatch_stage") == "promoted"
        result = _result(row)
        job = jobs[row["job_key"]]
        if result is None:
            continue
        if is_model_outcome(result, job):
            progress[row["model"]] += 1
        signal_value = _capacity_signal(result, job)
        if signal_value is not None:
            capacity_signals.append(signal_value)
    threshold = int(manifest["dispatch"]["promotion_outcomes_per_model"])
    if execution_stage == "pilot":
        stage = "pilot"
    elif capacity_signals:
        stage = "fallback" if promoted_seen else "held_initial"
    elif all(progress[model] >= threshold for model in model_order(manifest)):
        stage = "promoted"
    else:
        stage = "initial"
    promoted = stage == "promoted"
    policy = manifest["dispatch"]
    value = {
        "stage": stage,
        "per_model_concurrency": int(
            policy[
                "promoted_per_model_concurrency" if promoted else "initial_per_model_concurrency"
            ]
        ),
        "total_concurrency": int(
            policy["promoted_total_concurrency" if promoted else "initial_total_concurrency"]
        ),
        "promotion_outcomes_per_model": threshold,
        "promotion_progress": {model: progress[model] for model in model_order(manifest)},
        "capacity_signals": sorted(set(capacity_signals)),
    }
    if execution_stage == "full" and operational_override is not None:
        value["per_model_concurrency"] = int(operational_override["per_model_concurrency"])
        value["total_concurrency"] = int(operational_override["total_concurrency"])
        value["operational_concurrency_override"] = operational_override
    if manifest.get("pilot"):
        value["execution_stage"] = execution_stage
    if execution_stage == "pilot":
        value["per_model_concurrency"] = int(manifest["pilot"]["concurrency"])
        value["total_concurrency"] = int(manifest["pilot"]["concurrency"])
    return value


def _recover_inflight(store: FinalStore, scope: str) -> None:
    for row in store.inflight(scope):
        locked = store.locked_result(row["id"])
        if locked is None:
            result = AttemptResult(
                termination="transport",
                error="interrupted_process_before_answer",
                retryable=True,
                answer_eligible=False,
                trace_error="trace_missing",
            )
        else:
            result = AttemptResult(**locked)
        store.finish(row["id"], result.to_dict())


def _status(
    store: FinalStore,
    manifest: dict[str, Any],
    *,
    account: dict[str, Any] | None = None,
) -> dict[str, Any]:
    lineage_rows, lineage_outcomes = _validate_lineage(store, manifest)
    continuation_rows, continuation_outcomes = _continuation_state(
        store, manifest, lineage_outcomes
    )
    outcomes = {**lineage_outcomes, **continuation_outcomes}
    all_rows = [*lineage_rows, *continuation_rows]
    states = store.model_states()
    models = {}
    for model in model_order(manifest):
        model_outcomes = [
            (row, result) for row, result in outcomes.values() if row["model"] == model
        ]
        model_rows = [row for row in all_rows if row["model"] == model]
        continuation_model_rows = [row for row in continuation_rows if row["model"] == model]
        technical = sum(
            result is not None and not is_model_outcome(result, json.loads(row["job"]))
            for row in model_rows
            for result in [_result(row)]
        )
        lineage_completed = int(manifest["lineage_outcomes"][model])
        continuation_completed = sum(
            row["model"] == model for row, _ in continuation_outcomes.values()
        )
        planned = sum(job["model"] == model for job in manifest["jobs"])
        models[model] = {
            "planned": planned,
            "lineage_completed": lineage_completed,
            "continuation_jobs_completed": continuation_completed,
            "completed": len(model_outcomes),
            "remaining": planned - len(model_outcomes),
            "provider_calls": len(model_rows),
            "continuation_provider_calls": len(continuation_model_rows),
            "technical_calls": technical,
            "empty": sum(not result.text.strip() for _, result in model_outcomes),
            "finish_reasons": dict(Counter(result.finish_reason for _, result in model_outcomes)),
            "state": states.get(model),
        }
    inflight_rows = store.inflight(manifest["phase"])
    status = {
        "campaign_id": manifest["campaign_id"],
        "generated_at": timestamp(),
        "collection_protocol": manifest["collection_protocol"],
        "run": store.run_state(),
        "dispatch": _dispatch_state(continuation_rows, manifest),
        "models": models,
        "inflight": len(inflight_rows),
        "inflight_by_model": dict(Counter(row["model"] for row in inflight_rows)),
        "formal_budget_usd": manifest["formal_budget_usd"],
        "formal_accounting": store.accounting(formal_scopes(manifest)),
        "continuation_accounting": store.accounting((manifest["phase"],)),
        "account": account if account is not None else store.latest_account(),
    }
    if manifest.get("pilot"):
        pilot_keys = {job_key(job) for job in _pilot_jobs(manifest)}
        completed = len(pilot_keys & outcomes.keys())
        status["pilot"] = {
            "planned": len(pilot_keys),
            "completed": completed,
            "complete": completed == len(pilot_keys),
        }
    return status


def _publish(out: Path, store: FinalStore, manifest: dict[str, Any]) -> dict[str, Any]:
    status = _status(store, manifest)
    _write_json(out / "status.json", status)
    return status


@dataclass
class _WorkItem:
    job: dict[str, Any]
    attempts: int = 0
    ready_at: float = 0.0


@dataclass
class _ActiveCall:
    item: _WorkItem
    was_retry: bool


class _CreditGuard:
    """Refresh live credit periodically and reserve retry-bounded local exposure."""

    def __init__(
        self,
        gateway: Any,
        store: FinalStore,
        out: Path,
        manifest: dict[str, Any],
        account: dict[str, Any],
    ) -> None:
        self.gateway = gateway
        self.store = store
        self.out = out
        self.manifest = manifest
        self.account = account
        self.checked_at = time.monotonic()
        self.local_retry_bound = 0

    def reset(self, account: dict[str, Any]) -> None:
        self.account = account
        self.checked_at = time.monotonic()
        self.local_retry_bound = 0
        self.store.record_account(account)
        _write_json(self.out / "latest-account.json", account)

    def refresh(self) -> None:
        self.reset(self.gateway.account())

    def authorize(
        self,
        items: list[_WorkItem],
        pricing: dict[str, dict[str, Any]],
    ) -> list[int]:
        refresh_s = float(self.manifest["dispatch"]["account_refresh_s"])
        if time.monotonic() - self.checked_at >= refresh_s:
            self.refresh()
        reservations = [
            reservation(pricing[item.job["model"]], int(self.manifest["resources"]["max_tokens"]))
            for item in items
        ]
        if self.manifest.get("shared_account"):
            # The ledger checks all campaigns on this account atomically when
            # the batch is recorded. A process-local retry hold would count
            # concurrent campaigns inconsistently and pause too early.
            return reservations
        attempts_allowed = int(self.manifest["resources"]["technical_attempts_per_run"])
        retry_bound = sum(
            value * (attempts_allowed - item.attempts)
            for item, value in zip(items, reservations, strict=True)
        )

        def available() -> float:
            return float(self.account["remaining_usd"]) - self.local_retry_bound / 1_000_000

        required = retry_bound / 1_000_000 + float(self.manifest["credit_buffer_usd"])
        if available() < required:
            self.refresh()
        if available() < required:
            raise FormalRunError(
                f"account needs {required:.6f} USD for this retry-bounded dispatch; "
                f"{available():.6f} USD remains after local holds"
            )
        self.local_retry_bound += retry_bound
        return reservations


def _persisted_job(
    job: dict[str, Any],
    pricing: dict[str, Any],
    manifest: dict[str, Any],
    dispatch: dict[str, Any],
) -> dict[str, Any]:
    value = copy.deepcopy(job)
    value["collection"] = {
        "campaign_id": manifest["campaign_id"],
        "protocol": manifest["collection_protocol"],
        "pricing": pricing,
        "dispatch_stage": dispatch["stage"],
        "per_model_concurrency": dispatch["per_model_concurrency"],
        "total_concurrency": dispatch["total_concurrency"],
    }
    if "execution_stage" in dispatch:
        value["collection"]["execution_stage"] = dispatch["execution_stage"]
    if "operational_concurrency_override" in dispatch:
        value["collection"]["operational_concurrency_override"] = dispatch[
            "operational_concurrency_override"
        ]
    return value


def _operational_concurrency_override(
    manifest: dict[str, Any],
    execution_stage: str,
    per_model: int | None,
    total: int | None,
) -> dict[str, int | str] | None:
    if per_model is None and total is None:
        return None
    if per_model is None or total is None:
        raise FormalRunError("both operational concurrency override values are required")
    if execution_stage != "full":
        raise FormalRunError("operational concurrency override is only valid for the full stage")
    if per_model < 1 or total < 1:
        raise FormalRunError("operational concurrency override must be positive")
    models = model_order(manifest)
    policy = manifest["dispatch"]
    if per_model > int(policy["promoted_per_model_concurrency"]):
        raise FormalRunError("operational per-model concurrency exceeds the reviewed maximum")
    if total > int(policy["promoted_total_concurrency"]):
        raise FormalRunError("operational total concurrency exceeds the reviewed maximum")
    if total < per_model or total > per_model * len(models):
        raise FormalRunError("operational total concurrency is inconsistent with the model count")
    return {
        "per_model_concurrency": per_model,
        "total_concurrency": total,
        "reason": "user_authorized_runtime_adjustment",
    }


def _execute_attempt(
    store: FinalStore,
    attempt_id: int,
    job: dict[str, Any],
    prompt: str,
    client: Any,
    manifest: dict[str, Any],
) -> AttemptResult:
    print(
        json.dumps(
            {
                "event": "request_started",
                "attempt_id": attempt_id,
                "model": job["model"],
                "instance_id": job["instance_id"],
            }
        ),
        flush=True,
    )
    def event_factory():
        if job["provider"] == "bedrock":
            return bedrock_events(
                client,
                job["model"],
                prompt,
                job["request"]["parameters"],
                trace=True,
            )
        return openai_events(
            client,
            job["model"],
            prompt,
            job["request"]["parameters"],
            trace=True,
            trace_version=2,
        )
    result = consume_attempt(
        event_factory,
        deadline_s=float(manifest["resources"]["deadline_s"]),
        usage_grace_s=float(manifest["resources"]["usage_grace_s"]),
        on_event=lambda event: store.event(attempt_id, event),
        trace_enabled=True,
    )
    if result.response.get("model") not in set(job["request"]["expected_model"]):
        result.protocol_violation = "response_model_missing_or_changed"
    if not SAFE_RESPONSE_ID.fullmatch(str(result.response.get("id") or "")):
        result.protocol_violation = "response_id_missing_or_invalid"
    output = result.usage.get("output_tokens")
    if isinstance(output, int) and not isinstance(output, bool) and output > 128000:
        result.protocol_violation = "generation_cap_violated"
    store.finish(attempt_id, result.to_dict())
    print(
        json.dumps(
            {
                "event": "request_finished",
                "attempt_id": attempt_id,
                "model": job["model"],
                "finish_reason": result.finish_reason,
                "termination": result.termination,
                "usage": result.usage,
                "empty": not result.text.strip(),
                "outcome": is_model_outcome(result, job),
                "technical_reason": _technical_reason(result, job),
            }
        ),
        flush=True,
    )
    return result


def _run_continuous(
    store: FinalStore,
    gateway: Any,
    manifest: dict[str, Any],
    datasets: dict[str, Any],
    pricing: dict[str, dict[str, Any]],
    clients: dict[str, Any],
    pool: ThreadPoolExecutor,
    out: Path,
    should_stop: threading.Event,
    credit: _CreditGuard,
    *,
    execution_stage: str = "full",
    operational_concurrency_override: dict[str, int | str] | None = None,
) -> None:
    models = model_order(manifest)
    lineage_rows, lineage_outcomes = _validate_lineage(store, manifest)
    del lineage_rows
    continuation_rows, continuation_outcomes = _continuation_state(
        store, manifest, lineage_outcomes
    )
    remaining = pending_jobs(manifest, set(lineage_outcomes))
    selected = _execution_jobs(
        manifest, execution_stage, set(lineage_outcomes) | set(continuation_outcomes)
    )
    allowed = {job_key(job) for jobs in remaining.values() for job in jobs}
    queues = {
        model: deque(
            _WorkItem(job)
            for job in selected
            if job["model"] == model
            and job_key(job) in allowed
            and job_key(job) not in continuation_outcomes
        )
        for model in models
    }
    retry_queues: dict[str, list[_WorkItem]] = {model: [] for model in models}
    retry_active = Counter()
    futures: dict[Future[AttemptResult], _ActiveCall] = {}
    states = store.model_states()
    dispatch_snapshot = _dispatch_state(continuation_rows, manifest, execution_stage)
    if operational_concurrency_override is None:
        saved_override = dispatch_snapshot.get("operational_concurrency_override")
        if isinstance(saved_override, dict):
            operational_concurrency_override = saved_override
    progress = Counter(dispatch_snapshot["promotion_progress"])
    capacity_signals = set(dispatch_snapshot["capacity_signals"])
    ever_promoted = any(
        (json.loads(row["job"]).get("collection") or {}).get("dispatch_stage") == "promoted"
        for row in continuation_rows
    )
    attempts_allowed = int(manifest["resources"]["technical_attempts_per_run"])
    instances = {
        name: {instance.id: instance for instance in dataset.instances}
        for name, dataset in datasets.items()
    }
    cursor = 0
    last_stage: str | None = None
    last_metadata = time.monotonic()
    stopping = False
    pending_error: BaseException | None = None

    def current_dispatch() -> dict[str, Any]:
        threshold = int(manifest["dispatch"]["promotion_outcomes_per_model"])
        if execution_stage == "pilot":
            return {
                **dispatch_snapshot,
                "stage": "pilot",
                "execution_stage": "pilot",
                "per_model_concurrency": int(manifest["pilot"]["concurrency"]),
                "total_concurrency": int(manifest["pilot"]["concurrency"]),
                "capacity_signals": sorted(capacity_signals),
            }
        if capacity_signals:
            stage = "fallback" if ever_promoted else "held_initial"
        elif all(progress[model] >= threshold for model in models):
            stage = "promoted"
        else:
            stage = "initial"
        promoted = stage == "promoted"
        value = {
            "stage": stage,
            "per_model_concurrency": int(
                manifest["dispatch"][
                    "promoted_per_model_concurrency"
                    if promoted
                    else "initial_per_model_concurrency"
                ]
            ),
            "total_concurrency": int(
                manifest["dispatch"][
                    "promoted_total_concurrency" if promoted else "initial_total_concurrency"
                ]
            ),
            "promotion_outcomes_per_model": threshold,
            "promotion_progress": {model: progress[model] for model in models},
            "capacity_signals": sorted(capacity_signals),
        }
        if operational_concurrency_override is not None:
            value["per_model_concurrency"] = int(
                operational_concurrency_override["per_model_concurrency"]
            )
            value["total_concurrency"] = int(
                operational_concurrency_override["total_concurrency"]
            )
            value["operational_concurrency_override"] = operational_concurrency_override
        if manifest.get("pilot"):
            value["execution_stage"] = execution_stage
        return value

    def active_by_model() -> Counter[str]:
        return Counter(call.item.job["model"] for call in futures.values())

    def mark_completed_models() -> None:
        active = active_by_model()
        for model in models:
            if (
                states.get(model, {}).get("status") == "active"
                and not queues[model]
                and not retry_queues[model]
                and retry_active[model] == 0
                and active[model] == 0
            ):
                state = "paused" if execution_stage == "pilot" else "complete"
                reason = "pilot_complete" if execution_stage == "pilot" else None
                store.set_model_state(model, state, reason)
                states[model] = {"status": state, "pause_reason": reason}

    def take_item(model: str, now: float) -> _WorkItem | None:
        if states.get(model, {}).get("status") != "active":
            return None
        ready = [item for item in retry_queues[model] if item.ready_at <= now]
        if ready:
            item = min(ready, key=lambda value: value.ready_at)
            retry_queues[model].remove(item)
            return item
        if retry_queues[model] or retry_active[model]:
            return None
        return queues[model].popleft() if queues[model] else None

    def select_batch(dispatch: dict[str, Any]) -> list[_WorkItem]:
        nonlocal cursor
        selected: list[_WorkItem] = []
        active = active_by_model()
        total_cap = int(dispatch["total_concurrency"])
        per_model_cap = int(dispatch["per_model_concurrency"])
        while len(futures) + len(selected) < total_cap:
            picked = False
            now = time.monotonic()
            for offset in range(len(models)):
                index = (cursor + offset) % len(models)
                model = models[index]
                selected_for_model = sum(item.job["model"] == model for item in selected)
                if active[model] + selected_for_model >= per_model_cap:
                    continue
                item = take_item(model, now)
                if item is None:
                    continue
                selected.append(item)
                cursor = (index + 1) % len(models)
                picked = True
                break
            if not picked:
                break
        return selected

    def record_stage(dispatch: dict[str, Any]) -> None:
        nonlocal last_stage, ever_promoted
        if dispatch["stage"] == "promoted":
            ever_promoted = True
        if dispatch["stage"] != last_stage:
            store.record_dispatch(dispatch)
            print(json.dumps({"event": "dispatch_stage", **dispatch}), flush=True)
            last_stage = str(dispatch["stage"])

    def dispatch_batch(items: list[_WorkItem], dispatch: dict[str, Any]) -> None:
        if not items:
            return
        prompts = [
            render_prompt(
                instances[item.job["dataset"]][item.job["instance_id"]],
                prompt_version=manifest["prompt_version"],
            )
            for item in items
        ]
        reservations = credit.authorize(items, pricing)
        persisted = [
            _persisted_job(item.job, pricing[item.job["model"]], manifest, dispatch)
            for item in items
        ]
        attempt_ids = store.begin_batch(
            manifest["phase"],
            persisted,
            reservations,
            formal_scopes=formal_scopes(manifest),
            formal_cap=microdollars(float(manifest["formal_budget_usd"])),
            per_model_concurrency=int(dispatch["per_model_concurrency"]),
            total_concurrency=int(dispatch["total_concurrency"]),
            shared_account=manifest.get("shared_account"),
            live_account=credit.account if manifest.get("shared_account") else None,
        )
        for attempt_id, item, prompt in zip(attempt_ids, items, prompts, strict=True):
            was_retry = item.attempts > 0
            item.attempts += 1
            if was_retry:
                retry_active[item.job["model"]] += 1
            future = pool.submit(
                _execute_attempt,
                store,
                attempt_id,
                item.job,
                prompt,
                clients[item.job["model"]],
                manifest,
            )
            futures[future] = _ActiveCall(item=item, was_retry=was_retry)
        print(
            json.dumps(
                {
                    "event": "dispatch_batch",
                    "stage": dispatch["stage"],
                    "attempt_ids": attempt_ids,
                    "models": [item.job["model"] for item in items],
                    "inflight": len(futures),
                }
            ),
            flush=True,
        )

    while True:
        run_state = store.run_state()
        if should_stop.is_set() and run_state is not None and run_state["status"] == "running":
            store.set_run_state("paused", "user_stop")
            run_state = store.run_state()
        if run_state is None or run_state["status"] != "running":
            stopping = True

        if not stopping and time.monotonic() - last_metadata >= float(
            manifest["metadata_refresh_s"]
        ):
            metadata = gateway.inspect()
            _append_jsonl(out / "metadata.jsonl", metadata)
            pricing = metadata["pricing"]
            credit.reset(metadata["account"])
            last_metadata = time.monotonic()

        mark_completed_models()
        dispatch = current_dispatch()
        record_stage(dispatch)
        if not stopping:
            batch = select_batch(dispatch)
            if batch:
                dispatch_batch(batch, dispatch)
                _publish(out, store, manifest)

        if futures:
            done, _ = wait(tuple(futures), timeout=1.0, return_when=FIRST_COMPLETED)
            changed = False
            for future in done:
                call = futures.pop(future)
                item = call.item
                model = item.job["model"]
                if call.was_retry:
                    retry_active[model] -= 1
                try:
                    result = future.result()
                except BaseException as error:
                    pending_error = pending_error or error
                    stopping = True
                    current = store.run_state()
                    if current is not None and current["status"] == "running":
                        store.set_run_state("paused", str(error) or type(error).__name__)
                    continue
                changed = True
                signal_value = _capacity_signal(result, item.job)
                if signal_value is not None:
                    capacity_signals.add(signal_value)
                if is_model_outcome(result, item.job):
                    if execution_stage == "full":
                        progress[model] += 1
                elif not stopping and states.get(model, {}).get("status") == "active":
                    if is_retryable_technical(result) and item.attempts < attempts_allowed:
                        item.ready_at = time.monotonic() + _retry_delay(result, item.attempts)
                        retry_queues[model].append(item)
                    else:
                        reason = _technical_reason(result, item.job)
                        store.set_model_state(model, "paused", reason)
                        states[model] = {"status": "paused", "pause_reason": reason}
                        retry_queues[model].clear()
            if changed:
                mark_completed_models()
                _publish(out, store, manifest)
            continue

        if stopping:
            break
        mark_completed_models()
        if all(states.get(model, {}).get("status") == "complete" for model in models):
            store.set_run_state("complete")
            break
        if not any(states.get(model, {}).get("status") == "active" for model in models):
            pilot_complete = execution_stage == "pilot" and all(
                states.get(model, {}).get("pause_reason") == "pilot_complete" for model in models
            )
            store.set_run_state(
                "paused", "pilot_complete" if pilot_complete else "one_or_more_models_paused"
            )
            break
        retry_times = [
            item.ready_at
            for model in models
            for item in retry_queues[model]
            if states.get(model, {}).get("status") == "active"
        ]
        if retry_times:
            should_stop.wait(min(1.0, max(0.0, min(retry_times) - time.monotonic())))
            continue
        # An active model with no in-flight, pending, or retry work is complete;
        # anything else here indicates an internal scheduler invariant failure.
        mark_completed_models()
        if not any(queues[model] for model in models):
            continue
        raise FormalRunError("continuous scheduler could not dispatch pending work")

    if pending_error is not None:
        raise pending_error


def _request_stop(ledger: Path, campaign_id: str) -> dict[str, Any]:
    with sqlite3.connect(f"file:{ledger.resolve()}?mode=rw", uri=True, timeout=30) as db:
        exists = db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='formal_final_runs'"
        ).fetchone()
        if exists is None:
            return {
                "dispatch_stopped": False,
                "campaign_id": campaign_id,
                "inflight_will_drain": False,
            }
        now = timestamp()
        changed = db.execute(
            "UPDATE formal_final_runs SET status='paused',pause_reason='user_stop',updated_at=? "
            "WHERE campaign_id=? AND status='running'",
            (now, campaign_id),
        )
        if changed.rowcount:
            db.execute(
                "INSERT INTO audit(timestamp,action,subject,document) VALUES (?,?,?,?)",
                (
                    now,
                    "formal_final_state",
                    campaign_id,
                    '{"status":"paused","reason":"user_stop"}',
                ),
            )
    return {
        "dispatch_stopped": bool(changed.rowcount),
        "campaign_id": campaign_id,
        "inflight_will_drain": bool(changed.rowcount),
    }


def _run(
    root: Path,
    config: dict[str, Any],
    manifest: dict[str, Any],
    datasets: dict,
    *,
    execution_stage: str = "full",
    operational_concurrency_override: dict[str, int | str] | None = None,
) -> int:
    out, ledger = paths(root, config)
    gateway: Any = (
        BedrockGateway(root, config) if "bedrock" in config["routes"] else Gateway(root, config)
    )
    if any(job["request"]["base_url"].rstrip("/") != gateway.base_url for job in manifest["jobs"]):
        raise FormalRunError("gateway URL differs from the original formal condition")
    metadata = gateway.inspect()
    out.mkdir(parents=True, exist_ok=True)
    _append_jsonl(out / "metadata.jsonl", metadata)
    pricing = metadata["pricing"]
    clients = gateway.clients()
    stop_event = threading.Event()

    def on_signal(_signum: int, _frame: Any) -> None:
        stop_event.set()

    previous_handlers = {
        sig: signal.signal(sig, on_signal) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    pool = ThreadPoolExecutor(
        max_workers=int(manifest["dispatch"]["promoted_total_concurrency"]),
        thread_name_prefix="formal-final",
    )
    try:
        with FinalStore(ledger, manifest["campaign_id"]) as store:
            models = model_order(manifest)
            store.ensure_run(manifest["manifest_sha256"], models)
            if manifest.get("shared_account"):
                shared = manifest["shared_account"]
                store.ensure_account_group(
                    str(shared["group"]),
                    metadata["account"],
                    float(shared["buffer_usd"]),
                )
            _, lineage = _validate_lineage(store, manifest)
            _recover_inflight(store, manifest["phase"])
            _, continuation = _continuation_state(store, manifest, lineage)
            _execution_jobs(manifest, execution_stage, set(lineage) | set(continuation))
            store.record_account(metadata["account"])
            _write_json(out / "latest-account.json", metadata["account"])
            store.set_running(models)
            credit = _CreditGuard(gateway, store, out, manifest, metadata["account"])
            _publish(out, store, manifest)
            try:
                _run_continuous(
                    store,
                    gateway,
                    manifest,
                    datasets,
                    pricing,
                    clients,
                    pool,
                    out,
                    stop_event,
                    credit,
                    execution_stage=execution_stage,
                    operational_concurrency_override=operational_concurrency_override,
                )
            except BaseException as error:
                run_state = store.run_state()
                if run_state is not None and run_state["status"] == "running":
                    store.set_run_state("paused", str(error) or type(error).__name__)
                raise
            finally:
                # An operational error may arrive while other requests are active.
                # Drain them before publishing/re-scoring and releasing the ledger.
                pool.shutdown(wait=True, cancel_futures=False)
                status = _publish(out, store, manifest)
                from .study_final_report import evaluate

                evaluate(out, manifest, datasets, store)
            final_run = status["run"]
            finished = final_run is not None and (
                final_run["status"] == "complete"
                or execution_stage == "pilot"
                and final_run.get("pause_reason") == "pilot_complete"
            )
            return 0 if finished else 2
    finally:
        pool.shutdown(wait=True, cancel_futures=False)
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        for client in clients.values():
            try:
                client.close()
            except Exception:
                pass


def from_args(args: argparse.Namespace) -> int:
    config_path = args.config.resolve()
    root = config_path.parent.parent
    config = load_config(config_path)
    manifest, datasets = build_manifest(root, config)
    out, ledger = paths(root, config)
    command = args.formal_command
    if command == "stop":
        print(json.dumps(_request_stop(ledger, config["campaign_id"]), indent=2))
        return 0
    if command == "run" and not args.execute:
        raise FormalRunError("paid formal execution requires --execute")
    _write_json(out / "manifest.json", manifest, immutable=True)
    if command == "run":
        stage = getattr(args, "stage", None) or ("pilot" if manifest.get("pilot") else "full")
        if stage == "pilot" and not manifest.get("pilot"):
            raise FormalRunError("this campaign has no pilot stage")
        concurrency_override = _operational_concurrency_override(
            manifest,
            stage,
            getattr(args, "operational_per_model_concurrency", None),
            getattr(args, "operational_total_concurrency", None),
        )
        return _run(
            root,
            config,
            manifest,
            datasets,
            execution_stage=stage,
            operational_concurrency_override=concurrency_override,
        )
    with FinalStore(ledger, manifest["campaign_id"], readonly=True) as store:
        _validate_lineage(store, manifest)
        if command == "status":
            result = _status(store, manifest)
        elif command == "evaluate":
            from .study_final_report import evaluate

            result = evaluate(out, manifest, datasets, store)
        else:
            raise AssertionError(command)
    print(json.dumps(result, indent=2))
    return 0
