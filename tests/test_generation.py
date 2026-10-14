"""Generator contracts: schedules, seeds, reproducibility, prefixes, datasets."""

from __future__ import annotations

import json
import math
from pathlib import Path
from random import Random
from types import SimpleNamespace

import pytest

import grid_bench.generation.dataset as dataset_module
from grid_bench.generation import (
    FAMILY_DEFAULTS,
    PRESETS,
    CellSpec,
    ConfigError,
    GenerationError,
    build_dataset,
    config_from_dict,
    core_count,
    derive_seed,
    effective_schedule,
    expand_cells,
    generate_rollout,
    infeasibility_reason,
    read_jsonl,
    sample_queries,
    set_count,
    set_schedule,
    shape_for,
    shape_stats,
)
from grid_bench.generation.initial import regions, sample_initial
from grid_bench.registry import family_ids, get_family, validate_instance
from grid_bench.schema import GoldRecord, Instance, WorldSpec
from grid_bench.simulate import simulate

TINY = {
    "name": "tiny",
    "dataset_seed": 7,
    "split": "dev",
    "tier": "S",
    "dims": [1, 2],
    "t_grid": [0, 1, 2, 4, 8],
    "t_max": 8,
    "seeds_per_cell": 2,
}


@pytest.mark.parametrize("set_rate", [0.0, 0.2, 0.25, 0.3, 0.5, 0.75])
def test_set_schedule_is_exact_for_every_prefix(set_rate: float) -> None:
    schedule = set_schedule(64, set_rate)
    if set_rate < 1:
        assert schedule[0] is False
    for horizon in range(1, 65):
        assert sum(schedule[:horizon]) == math.floor(horizon * set_rate + 1e-9)
        assert set_count(horizon, set_rate) == sum(schedule[:horizon])
        assert core_count(horizon, set_rate) == horizon - sum(schedule[:horizon])


def test_effective_schedule_honors_block_quota_first_event_and_prefixes() -> None:
    rng = Random(1)
    plan = effective_schedule(64, 0.25, rng, block=8)
    assert len(plan) == 64
    assert plan[0] is True
    for start in range(0, 64, 8):
        assert sum(1 for x in plan[start : start + 8] if not x) == 2
    assert effective_schedule(5, None, rng) == [True] * 5
    assert effective_schedule(3, 0.9, rng)[0] is True
    for n in (1, 5, 8, 13, 26):
        assert (
            effective_schedule(n, 0.35, Random(9)) == effective_schedule(100, 0.35, Random(9))[:n]
        )
    with pytest.raises(ValueError):
        effective_schedule(4, 1.0, rng)


@pytest.mark.parametrize("family_id", sorted(family_ids()))
def test_shorter_rollout_is_an_exact_prefix_of_the_longer_one(family_id: str) -> None:
    short = _cell(family_id, 2)
    long_config = config_from_dict({**TINY, "dims": [2], "t_max": 24, "t_grid": [0, 8, 24]})
    long = next(c for c in expand_cells(long_config) if c.family == family_id)
    seed = derive_seed(7, family_id, "S", 2, short.w, "dev", 0, 0)
    a = generate_rollout(short, seed)
    b = generate_rollout(long, seed)
    assert a.initial == b.initial
    assert a.events == b.events[: short.t_max]
    assert [s.scan for s in a.steps] == [s.scan for s in b.steps[: short.t_max]]
    assert a.states_at[short.t_max] == b.states_at[8]


def test_config_files_match_the_built_in_presets() -> None:
    root = Path(__file__).resolve().parents[1] / "configs"
    for name, preset in PRESETS.items():
        assert json.loads((root / f"{name}.json").read_text()) == preset


def test_seed_derivation_is_stable_and_distinct() -> None:
    a = derive_seed(1, "metric_paint", "S", 2, 7, "dev", 0, 0)
    assert a == derive_seed(1, "metric_paint", "S", 2, 7, "dev", 0, 0)
    assert a != derive_seed(1, "metric_paint", "S", 2, 7, "dev", 0, 1)
    assert a != derive_seed(1, "metric_paint", "S", 2, 7, "test", 0, 0)
    assert a != derive_seed(2, "metric_paint", "S", 2, 7, "dev", 0, 0)
    assert a == 4643314543683367711


def test_shape_tables_match_the_design() -> None:
    assert shape_for("S", 1) == (81,)
    assert shape_for("S", 2) == (9, 9)
    assert shape_for("S", 3) == (4, 4, 5)
    assert shape_for("S", 4) == (3, 3, 3, 3)
    assert shape_for("M", 5) == (3, 3, 3, 3, 3)
    assert shape_for("L", 4) == (6, 6, 6, 5)
    assert shape_for(150, 4) == (3, 3, 4, 4)
    with pytest.raises(KeyError):
        shape_for("S", 5)
    stats = shape_stats((9, 9))
    assert stats["stencil_scan"] == 369
    assert stats["boundary_cell_fraction"] == pytest.approx(32 / 81)


def test_config_expansion_and_defaults() -> None:
    config = config_from_dict(TINY)
    cells = expand_cells(config)
    assert {cell.family for cell in cells} == set(family_ids())
    assert {cell.dimension for cell in cells} == {1, 2}
    paint = [c for c in cells if c.family == "metric_paint" and c.dimension == 2]
    assert [c.w for c in paint] == FAMILY_DEFAULTS["metric_paint"]["w_targets"]["S"]
    stencil = [c for c in cells if c.family == "local_stencil" and c.dimension == 2]
    assert [c.shape for c in stencil] == [(9, 9)] * 4
    assert [c.w for c in stencil] == [7, 17, 33, 81]
    assert all(c.w_kind == "support" for c in stencil)
    assert [c.whole_world for c in stencil] == [False, False, False, True]
    compress = [c for c in cells if c.family == "aggregating_compression" and c.dimension == 2]
    assert [c.shape for c in compress] == [(9, 9), (12, 13), (16, 16)]
    assert all(c.w_kind == "cells" for c in compress)
    assert compress[0].values == (0, *[2**k for k in range(13)])
    with pytest.raises(ConfigError):
        config_from_dict({**TINY, "bogus": 1})
    with pytest.raises(ConfigError, match="unknown family"):
        config_from_dict({**TINY, "families": ["not_a_family"]})
    with pytest.raises(ConfigError, match="unknown override keys"):
        config_from_dict({**TINY, "families": {"metric_paint": {"w_targtes": [5]}}})
    with pytest.raises(ConfigError, match="unknown rejection keys"):
        config_from_dict({**TINY, "rejection": {"max_atempts": 3}})
    with pytest.raises(ConfigError):
        config_from_dict({**TINY, "t_grid": [0, 9]})
    override = config_from_dict(
        {**TINY, "families": {"metric_paint": {"w_targets": [5], "set_rate": 0.4}}}
    )
    assert override.families[0].w_targets == (5,)
    assert override.families[0].set_rate == 0.4
    for name in PRESETS:
        assert expand_cells(config_from_dict(PRESETS[name]))


def test_infeasibility_reasons() -> None:
    config = config_from_dict(
        {**TINY, "dims": [3], "families": {"occluded_mix": {"w_targets": [12, 200]}}}
    )
    cells = expand_cells(config)
    reasons = {cell.w: infeasibility_reason(cell) for cell in cells}
    assert reasons[12] is None
    # shape (4,4,5) has 80 cells: a region-plus-boundary target of 200 cannot exist
    assert reasons[200] is not None and "exceeds the world" in reasons[200]
    config = config_from_dict(
        {**TINY, "dims": [4], "families": {"local_stencil": {"w_targets": [3, 7, 81]}}}
    )
    reasons = {cell.w: infeasibility_reason(cell) for cell in expand_cells(config)}
    # on [3,3,3,3] a ball has 1 cell (r = 0) or at least 5 cells (r = 1): no ball is near 3
    assert reasons[3] is not None and "STENCIL ball target 3" in reasons[3]
    assert reasons[7] is None
    assert reasons[81] is None  # the tier's nominal N means the whole world


def _cell(family: str, dimension: int = 2, **overrides: object) -> CellSpec:
    config = config_from_dict({**TINY, "dims": [dimension], "families": {family: overrides}})
    return expand_cells(config)[0]


@pytest.mark.parametrize("family_id", sorted(family_ids()))
def test_rollouts_are_reproducible_and_prefix_consistent(family_id: str) -> None:
    cell = _cell(family_id)
    seed = derive_seed(7, family_id, "S", 2, cell.w, "dev", 0, 0)
    first = generate_rollout(cell, seed)
    second = generate_rollout(cell, seed)
    assert first.events == second.events
    assert first.initial == second.initial
    assert first.stats == second.stats
    assert generate_rollout(cell, seed + 1).events != first.events

    assert first.horizon == cell.t_max
    for horizon in cell.t_grid:
        instance = first.instance(horizon, f"x/t{horizon}", {})
        validate_instance(instance)
        result = simulate(instance)
        assert result.final_state == first.states_at[horizon]
        assert result.workloads == first.workloads(horizon)
        assert [s.predicate for s in result.steps] == [s.predicate for s in first.steps[:horizon]]
        gold = first.gold(horizon, instance.id)
        assert gold.final_state == result.final_state
    assert first.events[0].op != "SET"
    sets = [i for i, event in enumerate(first.events, start=1) if event.op == "SET"]
    for horizon in range(1, cell.t_max + 1):
        assert sum(1 for i in sets if i <= horizon) == set_count(horizon, cell.set_rate)


@pytest.mark.parametrize("family_id", sorted(family_ids()))
def test_family_policies_hold_in_generated_events(family_id: str) -> None:
    cell = _cell(family_id)
    rollout = generate_rollout(cell, derive_seed(7, family_id, "S", 2, cell.w, "dev", 1, 0))
    core = [step for step in rollout.steps if step.kind_wanted is not None]
    assert core
    for step in core:
        params = step.event.params
        if family_id == "metric_paint":
            assert params["u"] != params["v"]
        elif family_id == "box_reflection":
            assert params["hi"][params["axis"]] > params["lo"][params["axis"]]
        elif family_id == "obstructed_motion":
            assert params["k"] >= 1
        elif family_id == "occluded_mix":
            assert params["r"] >= 0 and set(params) == {"p", "r"}
            assert step.extras["support"] >= 1
        elif family_id == "component_relabel":
            assert step.effective  # the family offers no ineffective event
        elif family_id == "boundary_layer":
            assert params["s"] in (-1, 1)
        elif family_id == "local_stencil":
            assert params["r"] >= 0 and step.extras["support"] >= 1
    if family_id == "aggregating_compression":
        for step in rollout.steps:
            if step.event.op == "SET" and step.event.params["v"] != 0:
                assert step.event.params["v"] in (1, 2)
    wanted = [s.effective == s.kind_wanted for s in core if not s.kind_miss]
    assert all(wanted)
    if cell.noop_rate is not None:
        effective = sum(1 for s in core if s.effective) / len(core)
        assert abs(effective - (1 - cell.noop_rate)) <= 0.3


def test_rejection_criteria_raise_generation_errors() -> None:
    from grid_bench.generation import RejectionConfig

    cell = _cell("metric_paint", 2)
    # short rollouts receive a zero no-op quota, so only an impossible floor rejects them
    strict_floor = RejectionConfig.from_dict({"effective_floor": {"default": 1.01}})
    with pytest.raises(GenerationError, match="effective fraction"):
        generate_rollout(cell, 3, rejection=strict_floor)
    narrow_band = RejectionConfig.from_dict({"scan_ratio_band": [0.999, 1.001]})
    with pytest.raises(GenerationError, match="mean core workload"):
        generate_rollout(_cell("component_relabel", 2), 3, rejection=narrow_band)
    no_slack = RejectionConfig.from_dict({"density_slack": -0.5})
    with pytest.raises(GenerationError, match="density"):
        generate_rollout(cell, 3, rejection=no_slack)


def test_region_initializer_hits_density_and_labels_neighbours_distinctly() -> None:
    world = WorldSpec((9, 9), (0, 1, 2, 3))
    rng = Random(5)
    grid = regions(rng, world, 0.6, (1, 2, 3), 12)
    assert grid.nonzero_count == round(0.6 * 81)
    for initializer in ("uniform", "mixed_blob", "tiles", "regions"):
        sampled = sample_initial(Random(1), world, 0.5, initializer, 10)
        assert sampled.nonzero_count == round(0.5 * 81)
    with pytest.raises(ValueError):
        sample_initial(Random(1), world, 0.5, "nope", 10)


def test_set_policies_respect_band_and_labels() -> None:
    from grid_bench.generation.setpolicy import POLICIES, propose_set
    from grid_bench.grid import Grid

    world = WorldSpec((5, 5), (0, 1, 2, 3))
    rng = Random(2)
    for name in POLICIES:
        grid = sample_initial(Random(4), world, 0.5, "regions", 6)
        for _ in range(30):
            event = propose_set(name, rng, grid, (0.4, 0.6), (1, 2, 3))
            assert event.op == "SET"
            grid.data[grid.index(event.params["p"])] = event.params["v"]
        assert 0.1 <= grid.density <= 0.9
    full = Grid(world, [1] * 25)
    assert propose_set("thermostat", rng, full, (0.4, 0.6), (1, 2, 3)).params["v"] == 0
    assert propose_set("spawn", rng, full, (0.4, 0.6), (1, 2)).params["v"] == 0
    empty = Grid(world)
    assert propose_set("distinct", rng, empty, (0.4, 0.6), (1, 2, 3)).params["v"] != 0
    assert propose_set("spawn", rng, empty, (0.4, 0.6), (1, 2)).params["v"] in (1, 2)
    with pytest.raises(ValueError):
        propose_set("nope", rng, empty, (0.4, 0.6), (1, 2, 3))


def test_sample_queries_policies() -> None:
    world = WorldSpec((4,), (0, 1))
    from grid_bench.world import WorldState

    initial = WorldState(world, (((0,), 1),))
    final = WorldState(world, (((0,), 1), ((1,), 1), ((2,), 1)))
    # uniform draws never look at the final state
    queries = sample_queries(Random(0), initial, final, 4)
    assert len(queries) == 4 and len(set(queries)) == 4
    assert sample_queries(Random(3), initial, final, 2) == sample_queries(
        Random(3), initial, initial, 2
    )
    # the changed-first diagnostic keeps its old behaviour under its own name
    half = sample_queries(Random(0), initial, final, 4, "half-changed")
    assert len(half) == 4 and len(set(half)) == 4
    two = sample_queries(Random(0), initial, final, 2, "half-changed")
    assert sum(1 for q in two if q in {(1,), (2,)}) == 1
    with pytest.raises(ValueError, match="unknown query policy"):
        sample_queries(Random(0), initial, final, 2, "random")
    config = config_from_dict({**TINY, "queries": 2, "query_policy": "half-changed"})
    assert (
        config.query_policy == "half-changed" and config.to_dict()["query_policy"] == "half-changed"
    )
    with pytest.raises(ConfigError, match="query_policy"):
        config_from_dict({**TINY, "query_policy": "changed-only"})


def test_git_state_is_anchored_to_source_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    source_root = Path(__file__).resolve().parents[1]
    assert dataset_module._source_root() == source_root
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append((command, kwargs))
        if command == ["git", "rev-parse", "--show-toplevel"]:
            return SimpleNamespace(returncode=0, stdout=f"{source_root}\n")
        if command == ["git", "rev-parse", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout="abc123\n")
        assert command == ["git", "status", "--porcelain", "--untracked-files=normal"]
        return SimpleNamespace(returncode=0, stdout=" M tracked.py\n?? new.py\n")

    monkeypatch.setattr(dataset_module.subprocess, "run", fake_run)
    assert dataset_module._git_state() == ("abc123", True)
    assert dataset_module._git_commit() == "abc123"
    assert calls
    assert all(kwargs["cwd"] == source_root for _, kwargs in calls)


def test_git_state_is_unknown_outside_source_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        del command, kwargs
        return SimpleNamespace(returncode=0, stdout="/some/containing/repository\n")

    monkeypatch.setattr(dataset_module.subprocess, "run", fake_run)
    assert dataset_module._git_state() == (None, None)


def test_build_dataset_writes_consistent_files(tmp_path: Path) -> None:
    config = config_from_dict({**TINY, "queries": 2})
    manifest = build_dataset(
        config, tmp_path, families=["metric_paint", "aggregating_compression"], dims=[2]
    )
    assert manifest["totals"]["cells"] == 6
    assert manifest["totals"]["rollouts"] == 12
    assert manifest["config_sha256"]
    assert manifest["git_dirty"] in (True, False, None)
    written_manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert written_manifest["name"] == "tiny"
    assert "git_dirty" in written_manifest
    for family in ("metric_paint", "aggregating_compression"):
        instances = [
            Instance.from_dict(r) for r in read_jsonl(tmp_path / "dev" / family / "instances.jsonl")
        ]
        gold = {
            r["instance_id"]: GoldRecord.from_dict(r)
            for r in read_jsonl(tmp_path / "dev" / family / "gold.jsonl")
        }
        traces = read_jsonl(tmp_path / "dev" / family / "trace.jsonl")
        assert len(instances) == 3 * 2 * len(config.t_grid)
        assert len(traces) == 6
        assert {t["rollout_id"] for t in traces} == {i.metadata["prefix_of"] for i in instances}
        for instance in instances:
            validate_instance(instance)
            assert len(instance.queries) == 2
            assert instance.metadata["query_policy"] == "uniform:2"
            assert instance.id.endswith(f"/t{instance.horizon:03d}")
            assert instance.metadata["horizon"] == instance.horizon
            assert simulate(instance).final_state == gold[instance.id].final_state
            assert len(gold[instance.id].workloads) == instance.horizon
        by_rollout: dict[str, list[Instance]] = {}
        for instance in instances:
            by_rollout.setdefault(instance.metadata["prefix_of"], []).append(instance)
        for group in by_rollout.values():
            group.sort(key=lambda i: i.horizon)
            longest = group[-1]
            for instance in group:
                assert instance.initial == longest.initial
                assert instance.events == longest.events[: instance.horizon]
    again = build_dataset(config, tmp_path / "again", families=["metric_paint"], dims=[2])
    assert again["totals"]["instances"] == 3 * 2 * len(config.t_grid)
    first = (tmp_path / "dev" / "metric_paint" / "instances.jsonl").read_text()
    second = (tmp_path / "again" / "dev" / "metric_paint" / "instances.jsonl").read_text()
    assert first == second


def _relabel_grid(values: list[int]):
    from grid_bench.grid import Grid
    from grid_bench.world import WorldState

    world = WorldSpec((len(values),), (0, 1, 2, 3))
    return Grid.from_state(WorldState(world, tuple(((i,), v) for i, v in enumerate(values) if v)))


def test_generation_cannot_mix_runs_or_overwrite_existing_files(tmp_path: Path) -> None:
    config = config_from_dict({**TINY, "families": {"metric_paint": {"w_targets": [7]}}})
    build_dataset(config, tmp_path)
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    other_split = config_from_dict({**config.source, "split": "other"})
    with pytest.raises(ValueError, match="new or empty"):
        build_dataset(other_split, tmp_path)
    after = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after


@pytest.mark.parametrize("updates,match", [
    ({"split": ".."}, "directory name"),
    ({"split": "."}, "directory name"),
    ({"split": "../outside"}, "directory name"),
    ({"split": "nested\\directory"}, "directory name"),
    ({"t_grid": []}, "at least one prefix"),
    ({"dims": [2, 2]}, "distinct positive dimensions"),
    ({"families": ["metric_paint", "metric_paint"]}, "distinct family"),
    ({"families": {"metric_paint": {"w_targets": [7, 7]}}}, "distinct positive integers"),
])
def test_configuration_rejects_unsafe_or_duplicate_selections(updates: dict, match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        config_from_dict({**TINY, **updates})


@pytest.mark.parametrize("indices", [[], [-1], [0, 0], [True], [0.5]])
def test_invalid_rollout_indices_fail_before_creating_output(tmp_path: Path, indices: list) -> None:
    out = tmp_path / "data"
    with pytest.raises(ValueError, match="rollout indices"):
        build_dataset(config_from_dict(TINY), out, indices=indices)
    assert not out.exists()


@pytest.mark.parametrize("workers", [0, -1])
def test_invalid_worker_count_fails_before_creating_output(tmp_path: Path, workers: int) -> None:
    out = tmp_path / "data"
    with pytest.raises(ValueError, match="workers"):
        build_dataset(config_from_dict(TINY), out, workers=workers)
    assert not out.exists()


def test_relabel_proposer_avoids_labels_that_would_merge_components() -> None:
    proposer = get_family("component_relabel").proposer
    assert proposer is not None
    # The 1-run touches a 2-run; 3 sits apart.  Only 3 avoids a merge and the
    # run is too large to delete (4 * 2 > 6 nonzero cells), so v must be 3.
    grid = _relabel_grid([1, 1, 2, 2, 0, 3, 3])
    for seed in range(50):
        events = proposer(Random(seed), grid, 4, True, 8)
        event = next(e for e in events if e.params["p"][0] in (0, 1))
        assert event.params["v"] == 3
    # Every other label touches the component and it is small enough to
    # delete, so the proposer deletes instead of merging.
    grid = _relabel_grid([2, 1, 3, 0, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2])
    for seed in range(50):
        events = proposer(Random(seed), grid, 3, True, 8)
        event = next(e for e in events if tuple(e.params["p"]) == (1,))
        assert event.params["v"] == 0
    # A merge is chosen only when it cannot be avoided, and the step reports it.
    grid = _relabel_grid([2, 1, 1, 3])
    events = proposer(Random(0), grid, 4, True, 8)
    event = next(e for e in events if e.params["p"][0] in (1, 2))
    assert event.params["v"] in (2, 3)
    result = get_family("component_relabel").step_grid(grid.copy(), event)
    assert result.extras["merge_contacts"] == 1


def test_candidates_within_tolerance_are_chosen_uniformly() -> None:
    from grid_bench.generation.selection import choose, score_candidate

    band = (0.4, 0.6)
    scores = [score_candidate(scan, 60, 0.5, band, tolerance=0.25) for scan in (50, 60, 72, 90)]
    assert scores[0] == scores[1] == scores[2] == 0.0 and scores[3] > 0.0
    picks = {choose(Random(seed), scores) for seed in range(40)}
    assert picks == {0, 1, 2}
    strict = [score_candidate(scan, 60, 0.5, band) for scan in (50, 60, 72)]
    assert strict[1] == 0.0 and strict[0] > 0.0 and strict[2] > 0.0


def test_rollout_stats_split_scan_by_effectiveness() -> None:
    cell = _cell("boundary_layer", 2)
    rollout = generate_rollout(cell, derive_seed(7, "boundary_layer", "S", 2, cell.w, "dev", 0, 0))
    stats = rollout.stats
    core = [s for s in rollout.steps if s.kind_wanted is not None]
    assert stats["core_effect_mean"] == pytest.approx(
        sum(s.effect for s in core) / len(core), abs=1e-3
    )
    effective = [s.scan for s in core if s.effective]
    assert stats["core_scan_effective_mean"] == pytest.approx(
        sum(effective) / len(effective), abs=1e-3
    )
    assert stats["core_scan_ineffective_mean"] is None or stats["core_scan_ineffective_mean"] > 0
