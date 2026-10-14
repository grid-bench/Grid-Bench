"""History-retention diagnostics: the omission census and the suffix baseline.

Both diagnostics run on the reference simulator and need no model.  The
omission census asks, for every effective core event in the first half of a
prefix, whether omitting that one event (and replaying the unchanged
remaining events from the true pre-event state) is still visible in the state
at the prefix's horizon.  The suffix baseline is an approximate solver that
executes every ``SET`` event but only the core events among the last ``k``
event positions; its accuracy shows how much of the final answer is fixed by
the recent past alone.  Neither quantity is a lower bound on the computation
a solver needs: they describe how the dynamics preserve or erase history.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .family import Family
from .grid import Grid, hamming
from .registry import get_family
from .schema import GoldRecord, Instance, SchemaError
from .simulate import simulate
from .world import WorldState

DEFAULT_SUFFIX_LENGTHS = (8, 16, 32, 64)
DIAGNOSTICS_FILE = "diagnostics.jsonl"


def suffix_state(instance: Instance, k: int, *, family: Family | None = None) -> WorldState:
    """Final state of the suffix-``k`` baseline for ``instance``.

    Every ``SET`` event is executed; a core event is executed only when its
    position is among the last ``k`` of the prefix.  ``k >= horizon`` executes
    everything and reproduces the gold state.
    """

    if k < 0:
        raise ValueError("suffix length must be nonnegative")
    family = get_family(instance.family) if family is None else family
    grid = Grid.from_state(instance.initial)
    cut = max(0, instance.horizon - k)
    for index, event in enumerate(instance.events):
        if event.op == "SET" or index >= cut:
            family.step_grid(grid, event)
    return grid.to_state()


def erasure_census(
    instance: Instance, horizons: Iterable[int], *, family: Family | None = None
) -> dict[int, tuple[int, int]]:
    """Count the early effective core events whose omission is invisible at each horizon.

    An event at 1-based position ``t <= T // 2`` is eligible when it is a core
    event that changed at least one cell.  It is *erased* when replaying
    events ``t+1 .. T`` from the true state before ``t`` yields exactly the
    true state at ``T``.  Returns ``{T: (eligible, erased)}``.
    """

    family = get_family(instance.family) if family is None else family
    result = simulate(instance, keep_all_states=True, family=family)
    counts: dict[int, tuple[int, int]] = {}
    for horizon in sorted({int(t) for t in horizons}):
        if horizon < 2 or horizon > instance.horizon:
            continue
        reference = Grid.from_state(result.states[horizon]).data
        eligible = 0
        erased = 0
        for index in range(horizon // 2):
            event = instance.events[index]
            if event.op == "SET" or result.steps[index].effect == 0:
                continue
            eligible += 1
            grid = Grid.from_state(result.states[index])
            try:
                for later in instance.events[index + 1 : horizon]:
                    family.step_grid(grid, later)
            except SchemaError:
                # The counterfactual trajectory is not even valid (for example a
                # merge beyond the alphabet): the omission is certainly visible.
                continue
            if hamming(grid.data, reference) == 0:
                erased += 1
        counts[horizon] = (eligible, erased)
    return counts


def _rollout_groups(instances: Sequence[Instance]) -> dict[str, list[Instance]]:
    groups: dict[str, list[Instance]] = defaultdict(list)
    for instance in instances:
        rollout_id = str(instance.metadata.get("prefix_of", instance.id))
        groups[rollout_id].append(instance)
    for group in groups.values():
        group.sort(key=lambda item: item.horizon)
    return groups


def diagnose_rollout(
    prefixes: Sequence[Instance],
    gold: Mapping[str, GoldRecord],
    *,
    suffix_lengths: Sequence[int] = DEFAULT_SUFFIX_LENGTHS,
) -> dict[str, Any]:
    """Diagnostics for one rollout given its nested prefixes (any order)."""

    ordered = sorted(prefixes, key=lambda item: item.horizon)
    longest = ordered[-1]
    family = get_family(longest.family)
    horizons = [item.horizon for item in ordered]
    census = erasure_census(longest, horizons, family=family)
    erasure: dict[str, dict[str, Any]] = {}
    for horizon, (eligible, erased) in census.items():
        erasure[str(horizon)] = {
            "eligible": eligible,
            "erased": erased,
            "fraction": erased / eligible if eligible else None,
        }
    suffix: dict[str, dict[str, dict[str, Any]]] = {}
    for item in ordered:
        if item.horizon == 0:
            continue
        record = gold.get(item.id)
        reference = (
            record.final_state if record is not None else simulate(item, family=family).final_state
        )
        reference_data = Grid.from_state(reference).data
        per_k: dict[str, dict[str, Any]] = {}
        for k in suffix_lengths:
            approximate = Grid.from_state(suffix_state(item, k, family=family)).data
            wrong = hamming(approximate, reference_data)
            per_k[str(k)] = {"wrong_cells": wrong, "em": wrong == 0}
        suffix[str(item.horizon)] = per_k
    metadata = longest.metadata
    return {
        "rollout_id": str(metadata.get("prefix_of", longest.id)),
        "family": longest.family,
        "tier": metadata.get("tier"),
        "dimension": longest.dimension,
        "w": metadata.get("nominal_scan_workload"),
        "t_max": longest.horizon,
        "suffix_lengths": [int(k) for k in suffix_lengths],
        "erasure_by_T": erasure,
        "suffix_by_T": suffix,
    }


def diagnose_dataset(
    instances: Sequence[Instance],
    gold: Mapping[str, GoldRecord],
    *,
    suffix_lengths: Sequence[int] = DEFAULT_SUFFIX_LENGTHS,
) -> list[dict[str, Any]]:
    """One diagnostics row per rollout of the dataset, in rollout-id order."""

    groups = _rollout_groups(instances)
    return [
        diagnose_rollout(groups[rollout_id], gold, suffix_lengths=suffix_lengths)
        for rollout_id in sorted(groups)
    ]


def write_diagnostics(rows: Iterable[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_diagnostics(path: Path) -> dict[str, dict[str, Any]]:
    """Diagnostics rows keyed by rollout id (an absent file gives an empty map)."""

    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["rollout_id"])] = row
    return rows


def summarize_diagnostics(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Per (family, T): pooled erasure fraction and the suffix-``k`` exact-match rates."""

    erasure: dict[tuple[str, int], list[int]] = defaultdict(lambda: [0, 0])
    suffix: dict[tuple[str, int], dict[int, list[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0, 0])
    )
    for row in rows:
        family = str(row["family"])
        for horizon, item in row.get("erasure_by_T", {}).items():
            entry = erasure[(family, int(horizon))]
            entry[0] += int(item["eligible"])
            entry[1] += int(item["erased"])
        for horizon, per_k in row.get("suffix_by_T", {}).items():
            for k, item in per_k.items():
                entry = suffix[(family, int(horizon))][int(k)]
                entry[0] += 1
                entry[1] += int(bool(item["em"]))
                entry[2] += int(item["wrong_cells"])
    keys = sorted(set(erasure) | set(suffix))
    summary: list[dict[str, Any]] = []
    for family, horizon in keys:
        eligible, erased = erasure.get((family, horizon), [0, 0])
        per_k = suffix.get((family, horizon), {})
        summary.append(
            {
                "family": family,
                "T": horizon,
                "rollouts": max([entry[0] for entry in per_k.values()] or [0]),
                "eligible_events": eligible,
                "erased_events": erased,
                "erasure_fraction": erased / eligible if eligible else None,
                "suffix_em": {
                    str(k): entry[1] / entry[0] if entry[0] else None
                    for k, entry in sorted(per_k.items())
                },
                "suffix_wrong_cells": {
                    str(k): entry[2] / entry[0] if entry[0] else None
                    for k, entry in sorted(per_k.items())
                },
            }
        )
    return summary


def render_summary(summary: Sequence[Mapping[str, Any]]) -> str:
    """A plain-text table of :func:`summarize_diagnostics` for the terminal."""

    lengths: list[str] = []
    for item in summary:
        for k in item["suffix_em"]:
            if k not in lengths:
                lengths.append(k)
    lengths.sort(key=int)
    header = ["family", "T", "rollouts", "erased/eligible", *[f"suffix-{k} EM" for k in lengths]]
    lines = ["\t".join(header)]
    for item in summary:
        fraction = item["erasure_fraction"]
        erased = f"{item['erased_events']}/{item['eligible_events']}" + (
            f" ({fraction:.0%})" if fraction is not None else ""
        )
        cells = [
            str(item["family"]),
            str(item["T"]),
            str(item["rollouts"]),
            erased,
            *[
                "-" if item["suffix_em"].get(k) is None else f"{item['suffix_em'][k]:.2f}"
                for k in lengths
            ],
        ]
        lines.append("\t".join(cells))
    return "\n".join(lines)


__all__ = [
    "DEFAULT_SUFFIX_LENGTHS",
    "DIAGNOSTICS_FILE",
    "diagnose_dataset",
    "diagnose_rollout",
    "erasure_census",
    "read_diagnostics",
    "render_summary",
    "suffix_state",
    "summarize_diagnostics",
    "write_diagnostics",
]
