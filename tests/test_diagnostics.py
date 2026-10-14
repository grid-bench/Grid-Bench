"""History-retention diagnostics: omission census, suffix baseline, provider, CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spatial_worlds.cli import main
from spatial_worlds.diagnostics import (
    diagnose_dataset,
    erasure_census,
    read_diagnostics,
    render_summary,
    suffix_state,
    summarize_diagnostics,
    write_diagnostics,
)
from spatial_worlds.eval.providers import SuffixProvider, make_provider, parse_suffix_length
from spatial_worlds.eval.run import load_dataset, run_predictions
from spatial_worlds.generation import build_dataset, config_from_dict
from spatial_worlds.schema import Event, Instance, WorldSpec
from spatial_worlds.simulate import simulate
from spatial_worlds.world import WorldState

TINY = {
    "name": "tiny",
    "dataset_seed": 13,
    "split": "dev",
    "tier": "S",
    "dims": [2],
    "t_grid": [0, 1, 2, 4, 8],
    "t_max": 8,
    "seeds_per_cell": 2,
    "families": {"metric_paint": {"w_targets": [7]}, "box_reflection": {"w_targets": [4]}},
}


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("diag")
    build_dataset(config_from_dict(TINY), path)
    return path


def _line(values: list[int], events: tuple[Event, ...], identifier: str = "line") -> Instance:
    world = WorldSpec((len(values),), (0, 1, 2, 3))
    initial = WorldState(world, tuple(((i,), v) for i, v in enumerate(values) if v))
    return Instance(
        id=identifier, family="metric_paint", world=world, initial=initial, events=events
    )


def test_suffix_state_executes_every_set_and_core_events_in_last_k_positions() -> None:
    instance = _line(
        [1, 1, 0, 2, 2, 0],
        (
            Event("PAINT", {"a": [1], "r": 1, "u": 1, "v": 3}),
            Event("SET", {"p": [5], "v": 1}),
            Event("PAINT", {"a": [4], "r": 1, "u": 2, "v": 1}),
        ),
    )
    gold = simulate(instance).final_state
    assert suffix_state(instance, 3) == gold
    assert suffix_state(instance, 99) == gold
    # k = 0 keeps the SET stream only
    assert list(suffix_state(instance, 0).dense_values()) == [1, 1, 0, 2, 2, 1]
    # k = 1 adds the last core event, which is the last position
    assert list(suffix_state(instance, 1).dense_values()) == [1, 1, 0, 1, 1, 1]
    # k = 2 covers positions 2 and 3: the SET and the last PAINT, still without the first
    assert list(suffix_state(instance, 2).dense_values()) == [1, 1, 0, 1, 1, 1]
    with pytest.raises(ValueError):
        suffix_state(instance, -1)


def test_erasure_census_counts_overwritten_early_events() -> None:
    # the first paint is overwritten by the two that follow: omitting it is invisible
    overwritten = _line(
        [1, 1, 1, 1],
        (
            Event("PAINT", {"a": [0], "r": 1, "u": 1, "v": 2}),
            Event("PAINT", {"a": [0], "r": 3, "u": 2, "v": 3}),
            Event("PAINT", {"a": [0], "r": 3, "u": 1, "v": 3}),
        ),
    )
    assert erasure_census(overwritten, [3]) == {3: (1, 1)}
    # here the first paint survives to the end, and an effect-free event is not eligible
    retained = _line(
        [1, 1, 1, 1],
        (
            Event("PAINT", {"a": [0], "r": 0, "u": 1, "v": 2}),
            Event("PAINT", {"a": [3], "r": 0, "u": 3, "v": 1}),
            Event("SET", {"p": [3], "v": 3}),
            Event("PAINT", {"a": [2], "r": 0, "u": 1, "v": 3}),
        ),
    )
    assert erasure_census(retained, [2, 4, 9]) == {2: (1, 0), 4: (1, 0)}
    assert erasure_census(retained, [1]) == {}


def test_diagnose_dataset_rows_roundtrip_and_summarize(dataset_dir: Path, tmp_path: Path) -> None:
    dataset = load_dataset(dataset_dir)
    rows = diagnose_dataset(dataset.instances, dataset.gold, suffix_lengths=(2, 8))
    rollouts = {str(i.metadata["prefix_of"]) for i in dataset.instances}
    assert [row["rollout_id"] for row in rows] == sorted(rollouts)
    for row in rows:
        assert row["t_max"] == 8 and row["suffix_lengths"] == [2, 8]
        assert set(row["suffix_by_T"]) == {"1", "2", "4", "8"}
        for per_k in row["suffix_by_T"].values():
            assert per_k["8"] == {"wrong_cells": 0, "em": True}  # k >= T executes everything
            assert per_k["2"]["wrong_cells"] >= 0
        assert set(row["erasure_by_T"]) <= {"2", "4", "8"}
        for item in row["erasure_by_T"].values():
            assert 0 <= item["erased"] <= item["eligible"]
    path = tmp_path / "diagnostics.jsonl"
    write_diagnostics(rows, path)
    assert read_diagnostics(path) == {row["rollout_id"]: row for row in rows}
    assert read_diagnostics(tmp_path / "missing.jsonl") == {}
    summary = summarize_diagnostics(rows)
    families = {item["family"] for item in summary}
    assert families == {"metric_paint", "box_reflection"}
    top = next(item for item in summary if item["family"] == "box_reflection" and item["T"] == 8)
    assert top["rollouts"] == 2 and top["suffix_em"]["8"] == 1.0
    assert top["erasure_fraction"] is None or 0 <= top["erasure_fraction"] <= 1
    text = render_summary(summary)
    assert text.splitlines()[0].startswith("family\tT\trollouts\terased/eligible\tsuffix-2 EM")
    assert "box_reflection\t8\t2\t" in text


def test_suffix_provider_is_the_diagnostic_solver(dataset_dir: Path, tmp_path: Path) -> None:
    assert parse_suffix_length(None) == 32 and parse_suffix_length("suffix") == 32
    assert parse_suffix_length("suffix-8") == 8
    with pytest.raises(ValueError):
        parse_suffix_length("gpt-5")
    with pytest.raises(ValueError):
        SuffixProvider("suffix-k")
    provider = make_provider("suffix", "suffix-2")
    assert provider.model == "suffix-2" and provider.request_params() == {"suffix_length": 2}
    dataset = load_dataset(dataset_dir)
    out = tmp_path / "suffix.jsonl"
    summary = run_predictions(dataset, provider, out=out, concurrency=2)
    assert summary["completed"] == len(dataset.instances)
    by_id = {instance.id: instance for instance in dataset.instances}
    for line in out.read_text().splitlines():
        record = json.loads(line)
        instance = by_id[record["instance_id"]]
        expected = WorldState.parse_dict(json.loads(record["raw_text"]), instance.world)[0]
        assert expected == suffix_state(instance, 2)


def test_diagnose_command_writes_next_to_the_manifest(
    dataset_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["diagnose", "--data", str(dataset_dir), "--suffix", "2,8"]) == 0
    captured = capsys.readouterr().out
    assert "suffix-2 EM" in captured and "wrote 4 rollout diagnostics" in captured
    dataset = load_dataset(dataset_dir)
    assert set(dataset.diagnostics) == {str(i.metadata["prefix_of"]) for i in dataset.instances}
    assert all(row["suffix_lengths"] == [2, 8] for row in dataset.diagnostics.values())
