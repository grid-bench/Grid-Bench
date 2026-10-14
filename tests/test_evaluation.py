from __future__ import annotations

import json
from dataclasses import replace

import pytest
from test_dataset import benchmark_row

from grid_bench.dataset import load_benchmark
from grid_bench.evaluation import evaluate_predictions, score_answer
from grid_bench.schema import Instance
from grid_bench.simulate import simulate


def test_formal_extraction_and_strict_scoring(small_instance: Instance) -> None:
    gold = simulate(small_instance).gold()
    answer = json.dumps(gold.final_state.to_dict())
    exact = score_answer(answer, small_instance, gold)
    assert exact["em"] and exact["strict_em"] and exact["canonical_format"]
    fenced = score_answer(f"```json\n{answer}\n```", small_instance, gold)
    assert fenced["em"] and not fenced["strict_em"]
    assert fenced["extraction_method"] == "fence"
    prose = score_answer(f"Final answer: {answer}", small_instance, gold)
    assert prose["em"] and prose["extraction_method"] == "last_object"
    for bad in (f'{answer}\n{{"cells":', f'{answer}\n{{"cells":unknown}}'):
        failed = score_answer(bad, small_instance, gold)
        assert not failed["em"] and not failed["valid_state"]
        assert failed["extraction_method"] == "failed"
    reordered = gold.final_state.to_dict()
    reordered["cells"].reverse()
    semantic = score_answer(json.dumps(reordered), small_instance, gold)
    assert semantic["em"] and semantic["strict_em"] and not semantic["canonical_format"]


def test_invalid_empty_missing_and_full_denominator(small_instance: Instance) -> None:
    instances = [replace(small_instance, id=f"instance-{i}") for i in range(4)]
    benchmark = load_benchmark([benchmark_row(item) for item in instances])
    predictions = [
        {"public_id": "instance-0", "response": json.dumps(benchmark[0].gold.final_state.to_dict())},
        {"public_id": "instance-1", "response": "not a state"},
        {"public_id": "instance-2", "response": ""},
    ]
    report = evaluate_predictions(benchmark, predictions)
    summary = report["summary"]
    assert summary["em"] == 0.25 and summary["n_instances"] == 4
    assert summary["n_submitted"] == 3
    assert summary["n_missing"] == summary["n_empty"] == summary["n_invalid"] == 1
    assert summary["valid_incorrect"]["n"] == 0
    assert summary["valid_incorrect"]["median_wrong_cells"] is None
    assert report["scores"][-1]["answer_status"] == "missing"
    assert all("response" not in row for row in report["scores"])


def test_duplicate_unknown_and_nonstring_predictions(small_instance: Instance) -> None:
    benchmark = load_benchmark([benchmark_row(small_instance)])
    row = {"public_id": small_instance.id, "response": ""}
    with pytest.raises(ValueError, match="Duplicate prediction"):
        evaluate_predictions(benchmark, [row, row])
    with pytest.raises(ValueError, match="Unknown prediction"):
        evaluate_predictions(benchmark, [{"public_id": "unknown", "response": ""}])
    with pytest.raises(ValueError, match="response must be a string"):
        evaluate_predictions(benchmark, [{"public_id": small_instance.id, "response": None}])


def test_outside_view_predictions_ignored_explicitly(small_instance: Instance) -> None:
    other = replace(small_instance, id="workload")
    benchmark = load_benchmark([
        benchmark_row(small_instance), benchmark_row(other, in_main=False, in_workload=True),
    ])
    report = evaluate_predictions(benchmark, [{"public_id": "workload", "response": ""}])
    assert report["view"] == "main" and report["n_ignored_outside_view"] == 1
    assert report["summary"]["n_missing"] == 1 and report["summary"]["em"] == 0


def test_update_and_retention_partition(small_instance: Instance) -> None:
    benchmark = load_benchmark([benchmark_row(small_instance)])
    gold = benchmark[0].gold
    final = gold.final_state.as_dict()
    initial = small_instance.initial.as_dict()
    changed = [p for p in small_instance.initial.positions() if initial.get(p, 0) != final.get(p, 0)]
    unchanged = [p for p in small_instance.initial.positions() if p not in changed]
    for positions, kind in [([changed[0]], "update_only"), ([unchanged[0]], "retention_only"),
                            ([changed[0], unchanged[0]], "both")]:
        values = dict(final)
        for position in positions:
            values[position] = (final.get(position, 0) + 1) % 4
        response = json.dumps({"cells": [[list(p), v] for p, v in sorted(values.items()) if v]})
        report = evaluate_predictions(benchmark, [{"public_id": small_instance.id, "response": response}])
        row = report["scores"][0]
        assert row["wrong_cells"] == len(positions)
        assert row["wrong_changed_cells"] + row["wrong_unchanged_cells"] == row["wrong_cells"]
        assert report["summary"]["valid_incorrect"][kind] == 1


def test_zero_history_is_scored_not_assumed_correct(small_instance: Instance) -> None:
    instance = replace(small_instance, events=())
    gold = simulate(instance).gold()
    result = score_answer('{"cells":[]}', instance, gold)
    assert result["valid_state"] and not result["em"]
    assert result["changed_cells"] == 0 and result["changed_cell_accuracy"] is None
    assert result["wrong_changed_cells"] == 0


def test_score_rejects_wrong_reference_identity(small_instance: Instance) -> None:
    gold = simulate(replace(small_instance, id="other")).gold()
    with pytest.raises(ValueError, match="identity mismatch"):
        score_answer("", small_instance, gold)
