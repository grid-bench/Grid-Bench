"""The portable release reader must reject tampering and incomplete result panels."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.reproduce_release import (
    SCORE_FIELDS,
    digest,
    reproduce_tables,
    rescore,
    validate_table_coverage,
    verify_manifest,
)
from spatial_worlds.eval.aggregate import score_formal_record
from spatial_worlds.prompt import COMPLETION_PROMPT_VERSION, render_prompt
from spatial_worlds.schema import GoldRecord, Instance

ROOT = Path(__file__).resolve().parents[1]


def fixture_panel() -> tuple[list[dict], list[dict]]:
    instance_object = json.loads((ROOT / "examples/metric_paint.json").read_text())
    reference_object = json.loads((ROOT / "examples/metric_paint.gold.json").read_text())
    instance = Instance.from_dict(instance_object)
    gold = GoldRecord.from_dict(reference_object)
    prompt = render_prompt(instance, prompt_version=COMPLETION_PROMPT_VERSION)
    benchmark = [{
        "public_id": "main:example", "instance_id": instance.id, "source_dataset": "main",
        "instance_json": json.dumps(instance_object), "reference_json": json.dumps(reference_object),
        "prompt_version": COMPLETION_PROMPT_VERSION, "prompt": prompt,
        "prompt_sha256": digest(prompt.encode()), "in_main": True, "in_workload": False,
    }]
    outcomes = []
    for model, text in (
        ("wrapped", "Final answer:\n```json\n" + json.dumps(gold.final_state.to_dict()) + "\n```"),
        ("empty", ""),
    ):
        record = {
            "raw_text": text, "model": model, "scoring_version": "spatial-worlds-scoring/1.0",
            "evaluation_protocol": "spatial-worlds-evaluation/1.6", "answer_eligible": True,
            "collection_finalized": True, "protocol_violation": None, "refusal": False,
            "termination": "complete" if text else "length", "finish_reason": "stop",
            "usage": {"output_tokens": 50},
        }
        scored = score_formal_record(record, instance, gold)
        outcomes.append({
            "public_id": "main:example", "model_id": model, "model_name": model,
            "delivered_text": text, "delivered_text_sha256": digest(text.encode()),
            "usage_json": json.dumps(record["usage"]), "visible_text_empty": not text.strip(),
            **{key: record[key] for key in (
                "scoring_version", "evaluation_protocol", "answer_eligible", "collection_finalized",
                "protocol_violation", "refusal", "termination", "finish_reason",
            )},
            **{key: scored[key] for key in SCORE_FIELDS},
        })
    return benchmark, outcomes


def test_release_reextracts_wrapped_answer_and_keeps_empty_failure() -> None:
    benchmark, outcomes = fixture_panel()
    scored = rescore(benchmark, outcomes)
    assert scored[0]["em"] and not scored[0]["strict_em"]
    assert scored[0]["wrong_cells"] == 0
    assert not scored[1]["em"] and not scored[1]["valid_state"]
    assert scored[1]["answer_status"] == "empty"
    assert scored[1]["wrong_cells"] is None


@pytest.mark.parametrize("field,value", [
    ("em", False), ("wrong_cells", 1), ("total_cells", 99),
    ("changed_cells", 999), ("copy_baseline_cell_accuracy", 999.0),
    ("zero_baseline_cell_accuracy", 999.0), ("score_error", "invented"),
])
def test_stored_scores_are_not_trusted(field: str, value: object) -> None:
    benchmark, outcomes = fixture_panel()
    outcomes[0][field] = value
    with pytest.raises(ValueError, match="Re-score mismatch"):
        rescore(benchmark, outcomes)


def test_delivered_text_identity_is_checked() -> None:
    benchmark, outcomes = fixture_panel()
    outcomes[0]["delivered_text"] = ""
    with pytest.raises(ValueError, match="hash mismatch"):
        rescore(benchmark, outcomes)


def test_formal_prompt_cannot_fall_back_to_old_default() -> None:
    benchmark, outcomes = fixture_panel()
    benchmark[0]["prompt_version"] = "spatial-worlds-prompt/0.3"
    with pytest.raises(ValueError, match="prompt version"):
        rescore(benchmark, outcomes)


def test_duplicate_or_missing_panel_members_are_rejected() -> None:
    benchmark, outcomes = fixture_panel()
    with pytest.raises(ValueError, match="Duplicate benchmark"):
        rescore(benchmark * 2, outcomes)
    with pytest.raises(ValueError, match="Duplicate model-instance"):
        rescore(benchmark, outcomes + [outcomes[0]])
    extra = copy.deepcopy(benchmark[0])
    extra["public_id"] = "main:second"
    with pytest.raises(ValueError, match="Incomplete shared"):
        rescore([*benchmark, extra], outcomes)
    with pytest.raises(ValueError, match="Missing outcome"):
        rescore(benchmark, [])


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
    from scripts.reproduce_release import TABLE_COUNTS, TABLE_KEYS
    from spatial_worlds.eval.expanded_analysis_registry import LENGTHS, MODELS, TASKS

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
