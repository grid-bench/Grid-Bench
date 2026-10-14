from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from grid_bench.dataset import load_benchmark
from grid_bench.prompt import render_prompt
from grid_bench.schema import Instance
from grid_bench.simulate import simulate


def benchmark_row(instance: Instance, *, in_main: bool = True,
                  in_workload: bool = False) -> dict:
    return {
        "public_id": instance.id, "public_trajectory_id": instance.id,
        "instance_json": json.dumps(instance.to_dict()),
        "reference_json": json.dumps(simulate(instance).gold().to_dict()),
        "prompt": render_prompt(instance), "in_main": in_main, "in_workload": in_workload,
    }


def test_load_hf_directory_jsonl_and_rows(tmp_path: Path, small_instance: Instance) -> None:
    row = benchmark_row(small_instance)
    data = tmp_path / "data"
    data.mkdir()
    source = data / "benchmark.jsonl"
    source.write_text(json.dumps(row) + "\n")
    for location in (tmp_path, data, source, [row]):
        loaded = load_benchmark(location, verify_references=True)
        assert len(loaded) == 1
        assert loaded[0].instance == small_instance
        assert loaded[0].prompt == render_prompt(small_instance)
        assert loaded[:1] == loaded.items


def test_views_retain_all_known_ids(small_instance: Instance) -> None:
    first = benchmark_row(small_instance)
    second = benchmark_row(replace(small_instance, id="workload"),
                           in_main=False, in_workload=True)
    assert len(load_benchmark([first, second], view="all")) == 2
    workload = load_benchmark([first, second], view="workload")
    assert len(workload) == 1 and workload[0].public_id == "workload"
    assert workload.all_ids == {small_instance.id, "workload"}


@pytest.mark.parametrize("mutation,match", [
    ({"prompt": "wrong"}, "prompt mismatch"),
    ({"instance_sha256": "wrong"}, "instance_sha256 mismatch"),
    ({"gold_sha256": "wrong"}, "gold_sha256 mismatch"),
    ({"prompt_sha256": "wrong"}, "prompt_sha256 mismatch"),
    ({"instance_id": "wrong"}, "identity mismatch"),
    ({"history_length": 900}, "history_length mismatch"),
    ({"in_main": 1}, "must be a boolean"),
    ({"public_trajectory_id": ""}, "Missing public_trajectory_id"),
])
def test_reject_inconsistent_metadata(small_instance: Instance, mutation: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        load_benchmark([{**benchmark_row(small_instance), **mutation}])


def test_reject_duplicates_empty_and_unknown_view(small_instance: Instance) -> None:
    row = benchmark_row(small_instance)
    with pytest.raises(ValueError, match="Duplicate benchmark"):
        load_benchmark([row, row])
    with pytest.raises(ValueError, match="empty"):
        load_benchmark([])
    with pytest.raises(ValueError, match="empty"):
        load_benchmark([row], view="workload")
    with pytest.raises(ValueError, match="Unknown benchmark view"):
        load_benchmark([row], view="typo")  # type: ignore[arg-type]


def test_reference_replay_detects_changed_target(small_instance: Instance) -> None:
    row = benchmark_row(small_instance)
    target = json.loads(row["reference_json"])
    target["final_state"] = {"cells": []}
    row["reference_json"] = json.dumps(target)
    with pytest.raises(ValueError, match="Reference execution mismatch"):
        load_benchmark([row], verify_references=True)


def test_duplicate_json_keys_fail(tmp_path: Path) -> None:
    source = tmp_path / "benchmark.jsonl"
    source.write_text('{"public_id":"a","public_id":"b"}\n')
    with pytest.raises(ValueError, match="duplicate JSON"):
        load_benchmark(source)
