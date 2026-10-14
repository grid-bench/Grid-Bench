"""Offline checks for completed-result alignment and failure decomposition."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture(scope="module")
def analysis_module() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts/analyze_completed.py"
    spec = importlib.util.spec_from_file_location("completed_analysis_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def paired_campaigns() -> list[tuple[dict, list[dict]]]:
    """Two complete endpoints with distinct aliases but identical evaluation inputs."""
    campaigns = []
    for model in ("model-a", "model-b"):
        request = {
            "parameters": {"max_completion_tokens": 128000, "reasoning_effort": "high"},
            "expected_model": [model, f"provider/{model}"],
            "base_url": "https://example.invalid/v1",
            "timeout_s": 7200,
            "max_attempts": 1,
            "usage_grace_s": 5.0,
            "n": 1,
            "evaluation_protocol": "spatial-worlds-evaluation/1.6",
            "scoring_version": "spatial-worlds-scoring/1.0",
        }
        jobs = [
            {
                "model": model,
                "dataset": dataset,
                "instance_id": identity,
                "prompt_sha256": f"prompt-{identity}",
                "instance_sha256": f"instance-{identity}",
                "gold_sha256": f"gold-{identity}",
                "condition_id": f"condition-{model}",
                "request": copy.deepcopy(request),
            }
            for dataset, identity in (("main", "shared-main"), ("rq2", "additional-workload"))
        ]
        rows = [
            {**copy.deepcopy(job), "em": True, "prompt_version": "spatial-worlds-prompt/0.4"}
            for job in jobs
        ]
        descriptor = {
            "id": model,
            "provider": "openai",
            "effort": "high",
            "accepted_response_models": [model, f"provider/{model}"],
        }
        campaigns.append(
            ({"jobs": jobs, "models": [descriptor], "workload_reuse_ids": ["shared-main"]}, rows)
        )
    return campaigns


def test_alignment_accepts_complete_paired_masks_and_model_specific_aliases(
    analysis_module: ModuleType, paired_campaigns: list[tuple[dict, list[dict]]]
) -> None:
    paired_campaigns[1][1].reverse()
    analysis_module.verify_alignment(paired_campaigns, expected=2)


@pytest.mark.parametrize("empty_input", [[], [({"jobs": [], "workload_reuse_ids": []}, [])]])
def test_alignment_rejects_empty_input(
    analysis_module: ModuleType, empty_input: list[tuple[dict, list[dict]]]
) -> None:
    with pytest.raises(ValueError, match="empty campaign input"):
        analysis_module.verify_alignment(empty_input, expected=2)


@pytest.mark.parametrize("remove_frozen_job", [False, True])
def test_alignment_rejects_incomplete_rows_or_masks(
    analysis_module: ModuleType,
    paired_campaigns: list[tuple[dict, list[dict]]],
    remove_frozen_job: bool,
) -> None:
    manifest, rows = paired_campaigns[1]
    rows.pop()
    if remove_frozen_job:
        manifest["jobs"].pop()
    with pytest.raises(ValueError, match="declared mask|incomplete model mask"):
        analysis_module.verify_alignment(paired_campaigns, expected=2)


@pytest.mark.parametrize("target", ["job", "row"])
def test_alignment_rejects_duplicates(
    analysis_module: ModuleType,
    paired_campaigns: list[tuple[dict, list[dict]]],
    target: str,
) -> None:
    manifest, rows = paired_campaigns[1]
    collection = manifest["jobs"] if target == "job" else rows
    collection.append(copy.deepcopy(collection[0]))
    with pytest.raises(ValueError, match="duplicate"):
        analysis_module.verify_alignment(paired_campaigns, expected=2)


@pytest.mark.parametrize("field", ["prompt_sha256", "condition_id", "request"])
def test_alignment_rejects_scored_row_drift_from_frozen_job(
    analysis_module: ModuleType,
    paired_campaigns: list[tuple[dict, list[dict]]],
    field: str,
) -> None:
    paired_campaigns[1][1][0][field] = "changed"
    with pytest.raises(ValueError, match=f"scored-row {field}"):
        analysis_module.verify_alignment(paired_campaigns, expected=2)


@pytest.mark.parametrize("field", ["prompt_sha256", "instance_sha256", "gold_sha256"])
def test_alignment_rejects_cross_model_hash_changes(
    analysis_module: ModuleType,
    paired_campaigns: list[tuple[dict, list[dict]]],
    field: str,
) -> None:
    manifest, rows = paired_campaigns[1]
    manifest["jobs"][0][field] = "changed"
    rows[0][field] = "changed"
    with pytest.raises(ValueError, match="identical instances, gold, prompts, settings"):
        analysis_module.verify_alignment(paired_campaigns, expected=2)


@pytest.mark.parametrize(
    ("parameter", "value"),
    [("max_completion_tokens", 65536), ("reasoning_effort", "medium"), ("temperature", 0.3)],
)
def test_alignment_rejects_unapproved_generation_settings(
    analysis_module: ModuleType,
    paired_campaigns: list[tuple[dict, list[dict]]],
    parameter: str,
    value: object,
) -> None:
    for manifest, rows in paired_campaigns:
        for collection in (manifest["jobs"], rows):
            for entry in collection:
                entry["request"]["parameters"][parameter] = value
    with pytest.raises(ValueError, match="unexpected model-visible condition"):
        analysis_module.verify_alignment(paired_campaigns, expected=2)


def test_alignment_rejects_cross_model_timeout_changes(
    analysis_module: ModuleType, paired_campaigns: list[tuple[dict, list[dict]]]
) -> None:
    manifest, rows = paired_campaigns[1]
    for collection in (manifest["jobs"], rows):
        for entry in collection:
            entry["request"]["timeout_s"] = 3600
    with pytest.raises(ValueError, match="unexpected shared evaluation setting"):
        analysis_module.verify_alignment(paired_campaigns, expected=2)


def test_alignment_rejects_prompt_version_changes(
    analysis_module: ModuleType, paired_campaigns: list[tuple[dict, list[dict]]]
) -> None:
    paired_campaigns[1][1][0]["prompt_version"] = "spatial-worlds-prompt/0.3"
    with pytest.raises(ValueError, match="unexpected model-visible condition"):
        analysis_module.verify_alignment(paired_campaigns, expected=2)


def test_alignment_rejects_workload_reuse_drift(
    analysis_module: ModuleType, paired_campaigns: list[tuple[dict, list[dict]]]
) -> None:
    paired_campaigns[1][0]["workload_reuse_ids"] = []
    with pytest.raises(ValueError, match="workload reuse masks differ"):
        analysis_module.verify_alignment(paired_campaigns, expected=2)


def test_alignment_rejects_repeated_model_campaign(
    analysis_module: ModuleType, paired_campaigns: list[tuple[dict, list[dict]]]
) -> None:
    paired_campaigns.append(copy.deepcopy(paired_campaigns[0]))
    with pytest.raises(ValueError, match="model occurs in multiple campaigns"):
        analysis_module.verify_alignment(paired_campaigns, expected=2)


@pytest.mark.parametrize("unscored", [None, 0, 1, "true"])
def test_alignment_rejects_unscored_outcomes(
    analysis_module: ModuleType,
    paired_campaigns: list[tuple[dict, list[dict]]],
    unscored: object,
) -> None:
    paired_campaigns[1][1][0]["em"] = unscored
    with pytest.raises(ValueError, match="unscored outcome"):
        analysis_module.verify_alignment(paired_campaigns, expected=2)


def outcome(
    status: str,
    *,
    finish: str = "stop",
    correct: bool = False,
    strict: bool = False,
    empty: bool = False,
    wrong_cells: int | None = None,
    cell_accuracy: float | None = None,
) -> dict:
    return {
        "answer_status": status,
        "valid_state": status == "valid",
        "em": correct,
        "strict_em": strict,
        "visible_text_empty": empty,
        "finish_reason": finish,
        "termination": "length" if finish in {"length", "max_tokens"} else "stop",
        "wrong_cells": wrong_cells,
        "cell_accuracy": cell_accuracy,
        "usage": {"output_tokens": 128000 if finish in {"length", "max_tokens"} else 100},
    }


def test_group_stats_separates_extraction_and_length_failure_types(
    analysis_module: ModuleType,
) -> None:
    rows = [
        outcome("valid", correct=True, strict=True, wrong_cells=0, cell_accuracy=1.0),
        outcome("valid", correct=True, wrong_cells=0, cell_accuracy=1.0),
        outcome("valid", wrong_cells=2, cell_accuracy=0.75),
        outcome("valid", wrong_cells=6, cell_accuracy=0.25),
        outcome("empty", finish="length", empty=True),
        outcome("empty", empty=True),
        outcome("invalid", finish="length"),
        outcome("invalid"),
    ]
    stats = analysis_module.group_stats(rows)
    assert stats["n"] == 8
    assert stats["correct"] == 2
    assert stats["em"] == 0.25
    assert stats["strict_correct"] == 1
    assert stats["extracted_correct"] == 1
    assert stats["valid"] == 4
    assert stats["valid_wrong"] == 2
    assert stats["length_terminated"] == 2
    assert stats["length_empty"] == 1
    assert stats["other_empty"] == 1
    assert stats["empty"] == 2
    assert stats["invalid_nonempty"] == 2
    assert stats["refusal"] == 0
    assert stats["cell_accuracy_valid_only"] == pytest.approx(0.75)
    assert stats["wrong_cells_median_valid_incorrect"] == 4
    assert stats["output_tokens"] == 256600
    assert sum(stats[key] for key in ("correct", "valid_wrong", "empty", "invalid_nonempty")) == 8


def test_group_stats_retains_all_empty_denominator_without_conditional_accuracy(
    analysis_module: ModuleType,
) -> None:
    stats = analysis_module.group_stats(
        [outcome("empty", finish="length", empty=True), outcome("empty", empty=True)]
    )
    assert stats["n"] == 2
    assert stats["em"] == 0.0
    assert stats["valid"] == stats["valid_wrong"] == 0
    assert stats["cell_accuracy_valid_only"] is None
    assert stats["wrong_cells_median_valid_incorrect"] is None


def test_group_stats_scores_a_complete_length_terminated_answer_on_its_content(
    analysis_module: ModuleType,
) -> None:
    stats = analysis_module.group_stats(
        [
            outcome(
                "valid",
                finish="length",
                correct=True,
                strict=True,
                wrong_cells=0,
                cell_accuracy=1.0,
            )
        ]
    )
    assert stats["em"] == 1.0
    assert stats["length_terminated"] == 1
    assert stats["length_empty"] == stats["empty"] == 0
    assert stats["wrong_cells_median_valid_incorrect"] is None


@pytest.mark.parametrize("native_effort", ["high", "native_always_on"])
def test_alignment_accepts_native_provider_parameters_without_rewriting(
    analysis_module,
    paired_campaigns,
    native_effort,
):
    manifest, rows = paired_campaigns[1]
    descriptor = manifest["models"][0]
    descriptor.update(provider="bedrock", effort=native_effort)
    parameters: dict[str, object] = {"max_tokens": 128000}
    if native_effort == "high":
        parameters["reasoning_effort"] = "high"
    for collection in (manifest["jobs"], rows):
        for entry in collection:
            entry["request"]["parameters"] = dict(parameters)
            entry["request"]["base_url"] = "https://bedrock.example.invalid"
    before = copy.deepcopy(paired_campaigns)
    analysis_module.verify_alignment(paired_campaigns, expected=2)
    assert paired_campaigns == before


def test_alignment_rejects_invented_effort_for_native_always_on(analysis_module, paired_campaigns):
    manifest, rows = paired_campaigns[1]
    manifest["models"][0].update(provider="bedrock", effort="native_always_on")
    for collection in (manifest["jobs"], rows):
        for entry in collection:
            entry["request"]["parameters"] = {"max_tokens": 128000, "reasoning_effort": "high"}
    with pytest.raises(ValueError, match="unexpected model-visible condition"):
        analysis_module.verify_alignment(paired_campaigns, expected=2)


def test_alignment_rejects_changed_expected_aliases(analysis_module, paired_campaigns):
    manifest, rows = paired_campaigns[1]
    for collection in (manifest["jobs"], rows):
        for entry in collection:
            entry["request"]["expected_model"] = ["unreviewed-model"]
    with pytest.raises(ValueError, match="unexpected model-visible condition"):
        analysis_module.verify_alignment(paired_campaigns, expected=2)


def test_native_termination_categories_and_complete_length_answer(analysis_module):
    stats = analysis_module.group_stats(
        [
            outcome("empty", finish="max_tokens", empty=True),
            outcome("empty", finish="end_turn", empty=True),
            outcome("valid", finish="max_tokens", correct=True, wrong_cells=0, cell_accuracy=1),
        ]
    )
    assert (stats["n"], stats["correct"], stats["length_terminated"]) == (3, 1, 2)
    assert (stats["empty"], stats["length_empty"], stats["other_empty"]) == (2, 1, 1)


def test_matched_pairs_keep_declared_and_conditional_denominators(analysis_module, monkeypatch):
    monkeypatch.setattr(analysis_module, "MODELS", ("a", "b"))
    monkeypatch.setattr(analysis_module, "LENGTHS", (8,))
    rows = []
    for model, wrong in (("a", (0, 2, 3, None)), ("b", (0, 8, None, 5))):
        for index, errors in enumerate(wrong):
            rows.append(
                {
                    "model": model,
                    "T": 8,
                    "instance_id": str(index),
                    "valid_state": errors is not None,
                    "em": errors == 0,
                    "wrong_cells": errors,
                    "cell_accuracy": None if errors is None else 1 - errors / 10,
                }
            )
    valid, wrong = analysis_module.matched_pair_diagnostics(rows)
    assert valid["n_total"] == wrong["n_total"] == 4
    assert (valid["n"], wrong["n"]) == (2, 1)
    assert (valid["correct_a"], valid["correct_b"]) == (1, 1)
    assert valid["wrong_cells_median_a"] == 1
    assert wrong["wrong_cells_median_a"] == 2
    assert wrong["wrong_cells_median_b"] == 8
    assert (wrong["a_fewer_wrong"], wrong["equal_wrong"], wrong["b_fewer_wrong"]) == (1, 0, 0)
    rows[-3]["valid_state"] = False
    assert analysis_module.matched_pair_diagnostics(rows)[1]["wrong_cells_median_a"] is None
    rows.pop()
    with pytest.raises(ValueError, match="identical declared masks"):
        analysis_module.matched_pair_diagnostics(rows)
