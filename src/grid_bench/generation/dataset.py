"""Dataset construction: rollouts -> nested-prefix instances, gold, traces, manifest."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import platform
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from random import Random
from statistics import fmean
from typing import Any

from ..prompt import LEGACY_PROMPT_VERSION
from ..schema import SCHEMA_VERSION
from ..world import WorldState
from .config import (
    GENERATOR_VERSION,
    QUERY_POLICIES,
    CellSpec,
    DatasetConfig,
    expand_cells,
    infeasibility_reason,
)
from .rollout import GenerationError, Rollout, generate_rollout
from .schedule import core_count, set_count
from .seeds import derive_seed
from .shapes import shape_stats


def _compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def sample_queries(
    rng: Random, initial: WorldState, final: WorldState, k: int, policy: str = "uniform"
) -> tuple[tuple[int, ...], ...]:
    """``k`` distinct query coordinates.

    ``uniform`` draws them without looking at the final state, so the query
    track estimates accuracy on a random cell.  ``half-changed`` picks
    ``(k + 1) // 2`` cells that changed between the two states first and fills
    the rest with unchanged cells; it is a diagnostic with a known
    answer-related sampling prior and is labelled as such in the metadata.
    """

    if policy not in QUERY_POLICIES:
        raise ValueError(f"unknown query policy {policy!r}; expected one of {QUERY_POLICIES}")
    positions = list(initial.positions())
    k = min(k, len(positions))
    if policy == "uniform":
        return tuple(rng.sample(positions, k))
    before = initial.dense_values()
    after = final.dense_values()
    changed = [x for x, a, b in zip(positions, before, after, strict=True) if a != b]
    unchanged = [x for x, a, b in zip(positions, before, after, strict=True) if a == b]
    want_changed = min(len(changed), (k + 1) // 2)
    picked = rng.sample(changed, want_changed)
    remaining = k - len(picked)
    pool = (
        unchanged
        if len(unchanged) >= remaining
        else unchanged + [x for x in changed if x not in picked]
    )
    picked.extend(rng.sample(pool, remaining))
    return tuple(picked)


@dataclass(slots=True)
class CellOutput:
    cell: CellSpec
    instances: list[dict[str, Any]] = field(default_factory=list)
    gold: list[dict[str, Any]] = field(default_factory=list)
    traces: list[dict[str, Any]] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)


def _instance_metadata(
    config: DatasetConfig, cell: CellSpec, rollout: Rollout, rollout_id: str, horizon: int
) -> dict[str, Any]:
    stats = shape_stats(cell.shape)
    metadata: dict[str, Any] = {
        "seed": rollout.seed,
        "dataset_seed": config.dataset_seed,
        "split": config.split,
        "tier": cell.tier,
        "level": cell.level,
        "shape": list(cell.shape),
        "cells": cell.cells,
        "dimension": cell.dimension,
        "horizon": horizon,
        "prefix_of": rollout_id,
        "t_max": cell.t_max,
        "nominal_scan_workload": cell.w,
        "w_kind": cell.w_kind,
        "set_rate": cell.set_rate,
        "noop_rate": cell.noop_rate,
        "core_events": core_count(horizon, cell.set_rate),
        "set_events": set_count(horizon, cell.set_rate),
        "initial_density": rollout.stats["initial_density"],
        "boundary_cell_fraction": round(float(stats["boundary_cell_fraction"]), 6),
        "mean_neighbor_count": round(float(stats["mean_neighbor_count"]), 6),
        "initializer": cell.initializer,
        "set_policy": cell.set_policy,
        "generator_version": GENERATOR_VERSION,
        "prompt_version": LEGACY_PROMPT_VERSION,
    }
    if config.queries:
        metadata["query_policy"] = f"{config.query_policy}:{config.queries}"
    return metadata


def generate_cell(
    cell: CellSpec,
    config: DatasetConfig,
    *,
    indices: Sequence[int] | None = None,
    full_trace: bool = False,
) -> CellOutput:
    """Generate every rollout of one cell and expand it into prefix instances."""

    output = CellOutput(cell=cell)
    reason = infeasibility_reason(cell)
    indices = list(range(config.seeds_per_cell)) if indices is None else list(indices)
    rejections: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []
    rollouts = 0
    failed = 0
    if reason is not None:
        output.report = {
            **cell.to_dict(),
            "key": cell.key,
            "requested": len(indices),
            "rollouts": 0,
            "instances": 0,
            "failed": len(indices),
            "rejections": [],
            "infeasible": True,
            "reason": reason,
        }
        return output
    retain = range(cell.t_max + 1) if full_trace else None
    for index in indices:
        rollout: Rollout | None = None
        for attempt in range(config.rejection.max_attempts):
            seed = derive_seed(
                config.dataset_seed,
                cell.family,
                cell.tier,
                cell.dimension,
                cell.w,
                config.split,
                index,
                attempt,
            )
            try:
                rollout = generate_rollout(
                    cell, seed, attempt=attempt, rejection=config.rejection, retain=retain
                )
                break
            except GenerationError as error:
                rejections.append({"index": index, "attempt": attempt, "reason": str(error)})
        if rollout is None:
            failed += 1
            continue
        rollouts += 1
        rollout_id = cell.rollout_id(config.split, index)
        stats_rows.append(dict(rollout.stats))
        for horizon in cell.t_grid:
            instance_id = f"{rollout_id}/t{horizon:03d}"
            metadata = _instance_metadata(config, cell, rollout, rollout_id, horizon)
            queries: tuple[tuple[int, ...], ...] = ()
            if config.queries:
                query_rng = Random(f"{rollout.seed}|queries|{horizon}")
                queries = sample_queries(
                    query_rng,
                    rollout.initial,
                    rollout.states_at[horizon],
                    config.queries,
                    config.query_policy,
                )
            instance = rollout.instance(horizon, instance_id, metadata, queries)
            output.instances.append(instance.to_dict())
            output.gold.append(rollout.gold(horizon, instance_id).to_dict())
        trace: dict[str, Any] = {
            "rollout_id": rollout_id,
            "family": cell.family,
            "tier": cell.tier,
            "dimension": cell.dimension,
            "shape": list(cell.shape),
            "w": cell.w,
            "seed": rollout.seed,
            "attempt": rollout.attempt,
            "t_grid": list(cell.t_grid),
            "steps": [step.to_dict() for step in rollout.steps],
            "stats": dict(rollout.stats),
        }
        if full_trace:
            trace["states"] = {
                str(t): rollout.states_at[t].to_dict()["cells"] for t in sorted(rollout.states_at)
            }
        output.traces.append(trace)
    summary: dict[str, Any] = {}
    if stats_rows:
        for key in (
            "core_scan_mean",
            "core_w_mean",
            "core_effect_mean",
            "effective_fraction",
            "kind_miss_fraction",
            "density_min",
            "density_max",
            "initial_density",
            "returns_blocked",
        ):
            summary[key] = round(fmean(float(row[key]) for row in stats_rows), 4)
        # Means that exist only when the corresponding events occurred.
        for key in (
            "predicate_false_fraction",
            "core_scan_effective_mean",
            "core_scan_ineffective_mean",
        ):
            values = [float(row[key]) for row in stats_rows if row.get(key) is not None]
            summary[key] = round(fmean(values), 4) if values else None
    output.report = {
        **cell.to_dict(),
        "key": cell.key,
        "requested": len(indices),
        "rollouts": rollouts,
        "instances": len(output.instances),
        "failed": failed,
        "rejections": rejections,
        "infeasible": rollouts == 0,
        "reason": None if rollouts else "all attempts rejected",
        "summary": summary,
    }
    return output


def _worker(args: tuple[CellSpec, DatasetConfig, list[int] | None, bool]) -> CellOutput:
    cell, config, indices, full_trace = args
    return generate_cell(cell, config, indices=indices, full_trace=full_trace)


def _source_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git_state() -> tuple[str | None, bool | None]:
    source_root = _source_root()
    run_options = {
        "cwd": source_root,
        "capture_output": True,
        "text": True,
        "check": False,
        "timeout": 5,
    }
    try:
        top_level = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            **run_options,
        )
        reported_root = top_level.stdout.strip()
        if (
            top_level.returncode != 0
            or not reported_root
            or Path(reported_root).resolve() != source_root
        ):
            return None, None
        commit = subprocess.run(["git", "rev-parse", "HEAD"], **run_options)
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            **run_options,
        )
    except (OSError, subprocess.SubprocessError):
        return None, None
    revision = commit.stdout.strip()
    if commit.returncode != 0 or status.returncode != 0 or not revision:
        return None, None
    return revision, bool(status.stdout.strip())


def _git_commit() -> str | None:
    return _git_state()[0]


def select_cells(
    config: DatasetConfig,
    *,
    families: Iterable[str] | None = None,
    dims: Iterable[int] | None = None,
) -> list[CellSpec]:
    wanted_families = set(families) if families is not None else None
    wanted_dims = set(dims) if dims is not None else None
    return [
        cell
        for cell in expand_cells(config)
        if (wanted_families is None or cell.family in wanted_families)
        and (wanted_dims is None or cell.dimension in wanted_dims)
    ]


def build_dataset(
    config: DatasetConfig,
    out_dir: str | Path,
    *,
    families: Iterable[str] | None = None,
    dims: Iterable[int] | None = None,
    indices: Sequence[int] | None = None,
    workers: int = 1,
    full_trace: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Generate every selected cell and write the dataset directory; returns the manifest."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cells = select_cells(config, families=families, dims=dims)
    if not cells:
        raise ValueError("no benchmark cells selected")
    tasks = [
        (cell, config, list(indices) if indices is not None else None, full_trace) for cell in cells
    ]
    handles: dict[tuple[str, str], Any] = {}

    def handle(family: str, kind: str) -> Any:
        key = (family, kind)
        if key not in handles:
            directory = out / config.split / family
            directory.mkdir(parents=True, exist_ok=True)
            handles[key] = (directory / f"{kind}.jsonl").open("w", encoding="utf-8")
        return handles[key]

    reports: list[dict[str, Any]] = []
    total_instances = 0
    try:
        if workers > 1:
            context = multiprocessing.get_context("spawn")
            with context.Pool(workers) as pool:
                outputs: Iterable[CellOutput] = pool.imap(_worker, tasks)
                for output in outputs:
                    total_instances += _consume(output, handle, reports, progress)
        else:
            for task in tasks:
                total_instances += _consume(_worker(task), handle, reports, progress)
    finally:
        for stream in handles.values():
            stream.close()

    config_json = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
    git_commit, git_dirty = _git_state()
    manifest = {
        "name": config.name,
        "generator_version": GENERATOR_VERSION,
        "schema_version": SCHEMA_VERSION,
        "prompt_version": LEGACY_PROMPT_VERSION,
        "python_version": platform.python_version(),
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "command": sys.argv,
        "config": config.to_dict(),
        "config_sha256": hashlib.sha256(config_json.encode("utf-8")).hexdigest(),
        "split": config.split,
        "full_trace": full_trace,
        "cells": reports,
        "totals": {
            "cells": len(reports),
            "rollouts": sum(r["rollouts"] for r in reports),
            "instances": total_instances,
            "infeasible_cells": sum(1 for r in reports if r["infeasible"]),
            "rejected_attempts": sum(len(r["rejections"]) for r in reports),
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _consume(
    output: CellOutput,
    handle: Callable[[str, str], Any],
    reports: list[dict[str, Any]],
    progress: Callable[[str], None] | None,
) -> int:
    family = output.cell.family
    for record in output.instances:
        handle(family, "instances").write(_compact(record) + "\n")
    for record in output.gold:
        handle(family, "gold").write(_compact(record) + "\n")
    for record in output.traces:
        handle(family, "trace").write(_compact(record) + "\n")
    reports.append(output.report)
    if progress is not None:
        report = output.report
        status = (
            "infeasible: " + str(report["reason"])
            if report["infeasible"]
            else (
                f"{report['rollouts']} rollouts, {report['instances']} instances, "
                f"{len(report['rejections'])} rejected attempts"
            )
        )
        progress(f"{output.cell.key}: {status}")
    return len(output.instances)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]
