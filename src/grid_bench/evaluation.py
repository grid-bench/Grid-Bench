"""Official answer scoring and complete-denominator evaluation for Grid-Bench."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from statistics import mean, median
from typing import Any

from .dataset import Benchmark, load_benchmark, read_jsonl
from .eval.extract import SCORING_VERSION, extract_final_answer
from .schema import GoldRecord, Instance, parse_json_document
from .scoring import score_prediction, score_query_prediction
from .world import WorldState

__all__ = ["evaluate_predictions", "load_benchmark", "score_answer"]


def score_answer(response: str, instance: Instance, gold: GoldRecord) -> dict[str, Any]:
    """Score one delivered answer using the published final-answer extraction rule.

    Semantic EM scores the final candidate JSON object, with no repair or fallback
    to an earlier object. Strict EM scores the unmodified response. Invalid or
    empty answers receive zero EM. Update/retention diagnostics are defined only
    for valid returned states and use the final-versus-initial partition.
    """
    if not isinstance(response, str):
        raise TypeError("response must be a string")
    if (instance.id != gold.instance_id or instance.family != gold.family
            or instance.world != gold.world):
        raise ValueError("Instance/reference identity mismatch")
    extraction = extract_final_answer(response)
    extracted = extraction.text or ""
    if instance.queries:
        strict = score_query_prediction(response, gold, instance.queries)
        scored = score_query_prediction(extracted, gold, instance.queries)
        domain = list(instance.queries)
    else:
        strict = score_prediction(response, gold)
        scored = score_prediction(extracted, gold)
        domain = list(instance.initial.positions())
    initial, final = instance.initial.as_dict(), gold.final_state.as_dict()
    changed = [p for p in domain if initial.get(p, 0) != final.get(p, 0)]
    changed_set = set(changed)
    unchanged = [position for position in domain if position not in changed_set]
    changed_accuracy = unchanged_accuracy = None
    wrong_changed = wrong_unchanged = None
    if scored.valid_state:
        payload = parse_json_document(extracted)
        predicted = (
            {tuple(entry[0]): entry[1] for entry in payload["values"]}
            if instance.queries else WorldState.parse_dict(payload, gold.world)[0].as_dict()
        )
        wrong_changed = sum(predicted.get(p, 0) != final.get(p, 0) for p in changed)
        wrong_unchanged = sum(predicted.get(p, 0) != final.get(p, 0) for p in unchanged)
        if changed:
            changed_accuracy = (len(changed) - wrong_changed) / len(changed)
        if unchanged:
            unchanged_accuracy = (len(unchanged) - wrong_unchanged) / len(unchanged)
    return {
        "em": scored.exact_match, "strict_em": strict.exact_match,
        "valid_json": scored.valid_json, "valid_state": scored.valid_state,
        "canonical_format": scored.canonical_format, "wrong_cells": scored.wrong_cells,
        "total_cells": scored.total_cells, "cell_accuracy": scored.cell_accuracy,
        "changed_cells": len(changed), "changed_cell_accuracy": changed_accuracy,
        "unchanged_cell_accuracy": unchanged_accuracy,
        "wrong_changed_cells": wrong_changed, "wrong_unchanged_cells": wrong_unchanged,
        "copy_baseline_em": not changed,
        "copy_baseline_cell_accuracy": len(unchanged) / len(domain),
        "zero_baseline_em": all(final.get(p, 0) == 0 for p in domain),
        "zero_baseline_cell_accuracy": sum(final.get(p, 0) == 0 for p in domain) / len(domain),
        "score_error": scored.error,
        "extraction_method": extraction.method, "extraction_applied": extraction.applied,
        "answer_status": "empty" if not response.strip() else (
            "valid" if scored.valid_state else "invalid"
        ),
    }


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    valid = [row for row in rows if row["valid_state"]]
    incorrect = [row for row in valid if not row["em"]]
    wrong = [row["wrong_cells"] for row in incorrect]
    return {
        "n_instances": count,
        "n_trajectories": len({row["public_trajectory_id"] for row in rows}),
        "n_submitted": sum(not row["missing"] for row in rows),
        "n_missing": sum(row["missing"] for row in rows),
        "n_empty": sum(row["answer_status"] == "empty" for row in rows),
        "n_invalid": sum(row["answer_status"] == "invalid" for row in rows),
        "n_valid": sum(row["valid_state"] for row in rows),
        "n_exact": sum(row["em"] for row in rows),
        "em": sum(row["em"] for row in rows) / count,
        "strict_em": sum(row["strict_em"] for row in rows) / count,
        "valid_state_rate": sum(row["valid_state"] for row in rows) / count,
        "cell_accuracy_valid_only": mean(row["cell_accuracy"] for row in valid) if valid else None,
        "valid_incorrect": {
            "n": len(incorrect),
            "median_wrong_cells": median(wrong) if wrong else None,
            "mean_wrong_cells": mean(wrong) if wrong else None,
            "update_only": sum(row["wrong_changed_cells"] > 0
                               and row["wrong_unchanged_cells"] == 0 for row in incorrect),
            "retention_only": sum(row["wrong_changed_cells"] == 0
                                  and row["wrong_unchanged_cells"] > 0 for row in incorrect),
            "both": sum(row["wrong_changed_cells"] > 0
                        and row["wrong_unchanged_cells"] > 0 for row in incorrect),
        },
    }


def _grouped(rows: Sequence[Mapping[str, Any]], field: str) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for row in rows:
        groups[row[field]].append(row)
    return [{field: key, **_summary(groups[key])} for key in sorted(groups)]


def evaluate_predictions(
    benchmark: Benchmark, predictions: str | Path | Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate one model's ``{public_id, response}`` records on an explicit view.

    Every selected instance contributes to the denominator, including absent
    records. Duplicate IDs and IDs absent from the source benchmark are errors.
    Known source IDs outside the selected view are ignored and counted. Metadata
    may accompany a prediction, but ``response`` must contain delivered text.
    """
    if not isinstance(benchmark, Benchmark) or not benchmark:
        raise ValueError("benchmark must be a nonempty result of load_benchmark()")
    source = read_jsonl(predictions) if isinstance(predictions, (str, Path)) else predictions
    answers: dict[str, str] = {}
    for prediction in source:
        identity = prediction.get("public_id")
        if not isinstance(identity, str) or identity not in benchmark.all_ids:
            raise ValueError(f"Unknown prediction public_id: {identity!r}")
        if identity in answers:
            raise ValueError(f"Duplicate prediction public_id: {identity}")
        if not isinstance(prediction.get("response"), str):
            raise ValueError(f"Prediction response must be a string: {identity}")
        answers[identity] = prediction["response"]
    scores = []
    selected = {item.public_id for item in benchmark}
    for item in benchmark:
        missing = item.public_id not in answers
        scored = score_answer(answers.get(item.public_id, ""), item.instance, item.gold)
        if missing:
            scored["answer_status"] = "missing"
        scores.append({
            "public_id": item.public_id, "public_trajectory_id": item.trajectory_id,
            "family": item.instance.family, "dimension": item.instance.dimension,
            "history_length": item.instance.horizon,
            "workload": item.instance.metadata.get("nominal_scan_workload"),
            "missing": missing, **scored,
        })
    return {
        "format": "grid-bench-evaluation/1", "scoring_version": SCORING_VERSION,
        "view": benchmark.view,
        "n_ignored_outside_view": len(set(answers) - selected),
        "summary": _summary(scores),
        "by_history_length": _grouped(scores, "history_length"),
        "by_family": _grouped(scores, "family"),
        "by_dimension": _grouped(scores, "dimension"),
        "scores": scores,
    }
