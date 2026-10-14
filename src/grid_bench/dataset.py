"""Load released or locally generated benchmark instances without a collection ledger."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, overload

from .prompt import PROMPT_VERSION, render_prompt
from .schema import GoldRecord, Instance, parse_json_document
from .simulate import simulate

View = Literal["main", "workload", "all"]
VIEWS = ("main", "workload", "all")


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Read JSON objects, rejecting duplicate keys and nonstandard constants."""
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = parse_json_document(line)
            except ValueError as error:
                raise ValueError(f"{path}:{number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            yield row


@dataclass(frozen=True, slots=True)
class BenchmarkItem:
    """A task, its exact reference, and the prompt to send to a model."""

    public_id: str
    trajectory_id: str
    instance: Instance
    gold: GoldRecord
    prompt: str
    in_main: bool
    in_workload: bool


@dataclass(frozen=True, slots=True)
class Benchmark(Sequence[BenchmarkItem]):
    """An explicit evaluation view with the full source ID set retained."""

    items: tuple[BenchmarkItem, ...]
    all_ids: frozenset[str]
    view: View

    def __len__(self) -> int:
        return len(self.items)

    @overload
    def __getitem__(self, index: int) -> BenchmarkItem: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[BenchmarkItem, ...]: ...

    def __getitem__(self, index: int | slice) -> BenchmarkItem | tuple[BenchmarkItem, ...]:
        return self.items[index]


def _object_hash(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def _item(row: Mapping[str, Any], *, verify_references: bool) -> BenchmarkItem:
    identity = row.get("public_id")
    if not isinstance(identity, str) or not identity.strip():
        raise ValueError("benchmark public_id must be a nonempty string")
    instance = Instance.from_dict(parse_json_document(row["instance_json"]))
    gold = GoldRecord.from_dict(parse_json_document(row["reference_json"]))
    if (gold.instance_id != instance.id or gold.family != instance.family
            or gold.world != instance.world or row.get("instance_id", instance.id) != instance.id):
        raise ValueError(f"Instance/reference identity mismatch: {identity}")
    prompt = row["prompt"]
    if not isinstance(prompt, str) or prompt != render_prompt(instance):
        raise ValueError(f"Official prompt mismatch: {identity}")
    if row.get("prompt_version", PROMPT_VERSION) != PROMPT_VERSION:
        raise ValueError(f"Unexpected prompt version: {identity}")
    expected_hashes = {
        "instance_sha256": _object_hash(instance.to_dict()),
        "gold_sha256": _object_hash(gold.to_dict()),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
    }
    for field, expected in expected_hashes.items():
        if field in row and row[field] != expected:
            raise ValueError(f"{field} mismatch: {identity}")
    for field, expected in {
        "family": instance.family, "dimension": instance.dimension,
        "history_length": instance.horizon, "shape": list(instance.world.shape),
    }.items():
        if field in row and row[field] != expected:
            raise ValueError(f"{field} mismatch: {identity}")
    for field in ("in_main", "in_workload"):
        if not isinstance(row.get(field), bool):
            raise ValueError(f"{field} must be a boolean: {identity}")
    trajectory = row.get("public_trajectory_id")
    if not isinstance(trajectory, str) or not trajectory:
        raise ValueError(f"Missing public_trajectory_id: {identity}")
    if verify_references and simulate(instance).final_state != gold.final_state:
        raise ValueError(f"Reference execution mismatch: {identity}")
    return BenchmarkItem(identity, trajectory, instance, gold, prompt,
                         row["in_main"], row["in_workload"])


def _generated_rows(root: Path) -> Iterator[dict[str, Any]]:
    instance_paths = sorted(root.glob("**/instances.jsonl"))
    if not instance_paths:
        raise ValueError(f"No benchmark.jsonl or generated instances found in {root}")
    for path in instance_paths:
        reference_path = path.with_name("gold.jsonl")
        references: dict[str, GoldRecord] = {}
        for raw in read_jsonl(reference_path):
            reference = GoldRecord.from_dict(raw)
            if reference.instance_id in references:
                raise ValueError(f"Duplicate generated reference ID: {reference.instance_id}")
            references[reference.instance_id] = reference
        seen = set()
        for raw in read_jsonl(path):
            instance = Instance.from_dict(raw)
            if instance.id not in references:
                raise ValueError(f"Missing generated reference: {instance.id}")
            seen.add(instance.id)
            yield {
                "public_id": instance.id,
                "public_trajectory_id": str(instance.metadata.get("prefix_of", instance.id)),
                "instance_json": json.dumps(instance.to_dict()),
                "reference_json": json.dumps(references[instance.id].to_dict()),
                "prompt": render_prompt(instance), "in_main": True, "in_workload": False,
            }
        if set(references) != seen:
            raise ValueError(f"Unmatched generated references in {reference_path}")


def load_benchmark(
    source: str | Path | Iterable[Mapping[str, Any]], *, view: View = "main",
    verify_references: bool = False,
) -> Benchmark:
    """Load a Hugging Face checkout, benchmark JSONL, HF rows, or generated directory.

    ``main`` and ``workload`` use the released membership flags. ``all`` is their
    source union. A locally generated directory is a custom evaluation set: its
    items belong to ``main``/``all`` and are not part of the published study.
    Stored prompts and available hashes are always checked. Reference replay is
    optional because references have already been computed by the simulator.
    """
    if view not in VIEWS:
        raise ValueError(f"Unknown benchmark view: {view!r}; expected one of {VIEWS}")
    rows: Iterable[Mapping[str, Any]]
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_dir():
            candidates = (path / "data" / "benchmark.jsonl", path / "benchmark.jsonl")
            found = next((candidate for candidate in candidates if candidate.is_file()), None)
            rows = read_jsonl(found) if found is not None else _generated_rows(path)
        else:
            rows = read_jsonl(path)
    else:
        rows = source
    items = []
    identities: set[str] = set()
    for row in rows:
        item = _item(row, verify_references=verify_references)
        if item.public_id in identities:
            raise ValueError(f"Duplicate benchmark ID: {item.public_id}")
        identities.add(item.public_id)
        if view == "all" or (view == "main" and item.in_main) or (
            view == "workload" and item.in_workload
        ):
            items.append(item)
    if not items:
        raise ValueError(f"Benchmark view {view!r} is empty")
    return Benchmark(tuple(items), frozenset(identities), view)
