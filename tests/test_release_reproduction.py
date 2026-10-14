"""The public measurement reader rejects tampering and incomplete result panels."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from grid_bench.analysis import LENGTHS, MODELS, TASKS, group_stats
from grid_bench.prompt import PROMPT_VERSION, render_prompt
from grid_bench.schema import GoldRecord, Instance
from scripts.reproduce_release import (
    TABLE_COUNTS,
    TABLE_KEYS,
    digest,
    reference_index,
    reproduce_tables,
    result_rows,
    validate_table_coverage,
    verify_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def fixture_panel() -> tuple[list[dict], list[dict]]:
    """Create measurement-only records from the bundled simulator example."""
    instance_object = json.loads((ROOT / "examples/metric_paint.json").read_text())
    reference_object = json.loads((ROOT / "examples/metric_paint.gold.json").read_text())
    instance = Instance.from_dict(instance_object)
    gold = GoldRecord.from_dict(reference_object)
    initial, final = instance.initial.as_dict(), gold.final_state.as_dict()
    changed = sum(initial.get(p, 0) != final.get(p, 0) for p in initial.keys() | final.keys())
    prompt = render_prompt(instance, prompt_version=PROMPT_VERSION)
    instance_json, reference_json = json.dumps(instance_object), json.dumps(reference_object)
    identity = {
        "public_id": f"main:{instance.id}", "instance_id": instance.id,
        "source_dataset": "main", "public_trajectory_id": f"main:{instance.id}",
        "rollout_id": instance.id, "cohort": "example", "history_length": instance.horizon,
        "prompt_version": PROMPT_VERSION, "prompt_sha256": digest(prompt.encode()),
    }
    benchmark = [{
        **identity, "instance_json": instance_json, "reference_json": reference_json,
        "instance_sha256": digest(instance_json.encode()),
        "gold_sha256": digest(reference_json.encode()), "prompt": prompt,
        "family": instance.family, "dimension": instance.dimension,
        "shape": list(instance.world.shape), "workload": 0,
        "in_main": True, "in_workload": False,
    }]
    results = []
    for model, valid, wrong in (("exact", True, 0), ("partial", True, 3), ("empty", False, None)):
        results.append({
            **identity, "model_id": model, "model_name": model,
            "em": valid and wrong == 0, "strict_em": False,
            "valid_state": valid, "wrong_cells": wrong,
            "cell_accuracy": (9 - wrong) / 9 if wrong is not None else 0.0,
            "answer_status": "valid" if valid else "empty", "visible_text_empty": not valid,
            "termination": "stop" if valid else "length", "total_cells": 9,
            "changed_cells": changed, "copy_baseline_em": changed == 0,
            "copy_baseline_cell_accuracy": (9 - changed) / 9,
            "zero_baseline_em": not final, "zero_baseline_cell_accuracy": (9 - len(final)) / 9,
            "usage_json": json.dumps({"output_tokens": 50}),
        })
    return benchmark, results


def test_public_measurements_rebuild_statistics_without_answers() -> None:
    benchmark, measurements = fixture_panel()
    joined = result_rows(benchmark, measurements)
    assert all("delivered_text" not in row for row in joined)
    assert group_stats(joined) == {
        "n": 3, "correct": 1, "em": 1 / 3, "strict_correct": 0, "extracted_correct": 1,
        "valid": 2, "valid_wrong": 1, "empty": 1, "length_empty": 1, "other_empty": 0,
        "length_terminated": 1, "invalid_nonempty": 0, "refusal": 0,
        "cell_accuracy_valid_only": (1 + 2 / 3) / 2,
        "wrong_cells_median_valid_incorrect": 3, "output_tokens": 150,
    }


def test_empty_group_has_no_invented_accuracy_or_median() -> None:
    summary = group_stats([])
    assert summary["n"] == 0
    assert summary["em"] is None
    assert summary["cell_accuracy_valid_only"] is None
    assert summary["wrong_cells_median_valid_incorrect"] is None


@pytest.mark.parametrize("field", ["instance_json", "reference_json", "prompt"])
def test_benchmark_record_identity_is_checked(field: str) -> None:
    benchmark, measurements = fixture_panel()
    benchmark[0][field] += " "
    with pytest.raises(ValueError, match="hash mismatch|prompt mismatch"):
        result_rows(benchmark, measurements)


def test_formal_prompt_cannot_fall_back_to_old_default() -> None:
    benchmark, measurements = fixture_panel()
    benchmark[0]["prompt_version"] = "spatial-worlds-prompt/0.3"
    with pytest.raises(ValueError, match="prompt version"):
        result_rows(benchmark, measurements)


@pytest.mark.parametrize("field,value", [
    ("instance_id", "other"), ("public_trajectory_id", "wrong"), ("history_length", 999),
    ("prompt_sha256", "wrong"), ("source_dataset", "other"),
])
def test_measurements_are_joined_by_full_identity(field: str, value: object) -> None:
    benchmark, measurements = fixture_panel()
    measurements[0][field] = value
    with pytest.raises(ValueError, match="Result identity mismatch"):
        result_rows(benchmark, measurements)


@pytest.mark.parametrize("field,value", [
    ("wrong_cells", 8), ("em", False), ("cell_accuracy", 0.5), ("total_cells", 999),
    ("changed_cells", 99), ("zero_baseline_cell_accuracy", 9), ("answer_status", "empty"),
])
def test_inconsistent_measurements_are_rejected(field: str, value: object) -> None:
    benchmark, measurements = fixture_panel()
    measurements[0][field] = value
    with pytest.raises(ValueError, match="Inconsistent"):
        result_rows(benchmark, measurements)


def test_invalid_results_cannot_claim_cells_or_strict_exactness() -> None:
    benchmark, measurements = fixture_panel()
    measurements[2]["wrong_cells"] = 4
    with pytest.raises(ValueError, match="invalid-state"):
        result_rows(benchmark, measurements)
    measurements[2]["wrong_cells"] = None
    measurements[2]["strict_em"] = True
    with pytest.raises(ValueError, match="strict exact-match"):
        result_rows(benchmark, measurements)


def test_duplicate_or_missing_panel_members_are_rejected() -> None:
    benchmark, measurements = fixture_panel()
    with pytest.raises(ValueError, match="Duplicate benchmark"):
        result_rows(benchmark * 2, measurements)
    with pytest.raises(ValueError, match="Duplicate result"):
        result_rows(benchmark, measurements + [measurements[0]])
    extra = copy.deepcopy(benchmark[0])
    extra.update(source_dataset="rq2", public_id=f"rq2:{extra['instance_id']}",
                 public_trajectory_id=f"rq2:{extra['rollout_id']}")
    with pytest.raises(ValueError, match="Incomplete shared"):
        result_rows([*benchmark, extra], measurements)
    with pytest.raises(ValueError, match="Incomplete shared"):
        result_rows(benchmark, [])


@pytest.mark.parametrize("field", ["delivered_text", "predicted_state", "error_coordinates"])
def test_model_answers_are_not_public_measurements(field: str) -> None:
    benchmark, measurements = fixture_panel()
    measurements[0][field] = "unpublished"
    with pytest.raises(ValueError, match="outside the public"):
        result_rows(benchmark, measurements)


def test_reference_execution_detects_tampered_reference() -> None:
    benchmark, _ = fixture_panel()
    reference = json.loads(benchmark[0]["reference_json"])
    reference["final_state"] = {"cells": []}
    benchmark[0]["reference_json"] = json.dumps(reference)
    benchmark[0]["gold_sha256"] = digest(benchmark[0]["reference_json"].encode())
    with pytest.raises(ValueError, match="Reference execution mismatch"):
        reference_index(benchmark)


def manifest_fixture(root: Path) -> None:
    data = b"payload\n"
    (root / "data.jsonl").write_bytes(data)
    (root / "manifest.json").write_text(json.dumps({
        "members": {"data.jsonl": {"bytes": len(data), "sha256": digest(data)}}
    }))


def test_member_inventory_detects_extra_and_changed_files(tmp_path: Path) -> None:
    manifest_fixture(tmp_path)
    verify_manifest(tmp_path)
    (tmp_path / "unexpected.txt").write_text("extra")
    with pytest.raises(ValueError, match="inventory"):
        verify_manifest(tmp_path)
    (tmp_path / "unexpected.txt").unlink()
    (tmp_path / "data.jsonl").write_text("modified")
    with pytest.raises(ValueError, match="checksum"):
        verify_manifest(tmp_path)


def test_member_paths_cannot_escape_release(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps({"members": {"../outside": {}}}))
    with pytest.raises(ValueError, match="Unsafe"):
        verify_manifest(tmp_path)


def test_symlinked_members_are_rejected(tmp_path: Path) -> None:
    manifest_fixture(tmp_path)
    (tmp_path / "data.jsonl").rename(tmp_path / "actual")
    (tmp_path / "data.jsonl").symlink_to(tmp_path / "actual")
    with pytest.raises(ValueError, match="Symlink"):
        verify_manifest(tmp_path)


def test_incomplete_frozen_tables_cannot_report_success() -> None:
    with pytest.raises(ValueError, match="Missing or unexpected frozen tables"):
        reproduce_tables([], {"tables": {}})


def test_missing_or_duplicated_conditions_are_rejected() -> None:
    rows = [
        {"model": m, "family": f, "dimension": d, "T": t, "w": 10,
         "in_main": True, "in_workload": False}
        for m in MODELS for f in TASKS for d in (2, 3, 4) for t in LENGTHS
    ]
    rows.extend(
        {"model": m, "family": "metric_paint", "dimension": 2, "T": t, "w": w,
         "in_main": False, "in_workload": True}
        for m in MODELS for w in range(11) for t in LENGTHS
    )
    tables = {}
    for name, fields in TABLE_KEYS.items():
        if name == "overview":
            keys = {(m, v) for m in MODELS for v in ("main", "main_T8_64")}
        elif name == "resources":
            keys = {(m, "unique_1488") for m in MODELS}
        else:
            membership = "in_workload" if name == "workload" else "in_main"
            keys = {tuple(r[f] for f in fields) for r in rows if r[membership]}
        tables[name] = [dict(zip(fields, key, strict=True)) for key in sorted(keys)]
        assert len(tables[name]) == TABLE_COUNTS[name]
    validate_table_coverage(rows, tables)
    tables["task_length"].pop()
    with pytest.raises(ValueError, match="Missing or duplicate"):
        validate_table_coverage(rows, tables)
    tables["task_length"].append(tables["task_length"][0])
    with pytest.raises(ValueError, match="Missing or duplicate"):
        validate_table_coverage(rows, tables)
