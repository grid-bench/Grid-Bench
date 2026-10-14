"""Descriptive summaries and display identities for the released study.

These summaries aggregate recorded measurements; model answers are not part of
this release. Bootstrap intervals are stored in the data package and are not
re-estimated by this module.
"""

from __future__ import annotations

from collections import Counter
from statistics import median

MODELS = (
    "gpt-5.6-sol",
    "claude-sonnet",
    "gpt-5.4-mini",
    "zai.glm-5",
    "moonshotai.kimi-k2.5",
    "minimax.minimax-m2.5",
)
COLORS = dict(
    zip(
        MODELS,
        ("#245C99", "#C77C30", "#3B8D88", "#8C63A8", "#B84A62", "#687A32"),
        strict=True,
    )
)
MARKERS = dict(zip(MODELS, ("o", "s", "^", "D", "P", "X"), strict=True))
LENGTHS = (0, 1, 8, 16, 32, 64)
TASKS = {
    "metric_paint": "PAINT",
    "box_reflection": "REFLECT",
    "local_stencil": "STENCIL",
    "occluded_mix": "MIX",
    "component_relabel": "RELABEL",
    "boundary_layer": "LAYER",
    "obstructed_motion": "MOVE",
    "aggregating_compression": "COMPRESS",
}

def group_stats(rows: list[dict]) -> dict:
    counts = Counter(row["answer_status"] for row in rows)
    valid = [r for r in rows if r["valid_state"]]
    wrong = [r["wrong_cells"] for r in valid if not r["em"]]
    correct = sum(bool(r["em"]) for r in rows)
    empty = sum(r["visible_text_empty"] for r in rows)
    length_empty = sum(r["visible_text_empty"] and r["termination"] == "length" for r in rows)
    return {
        "n": len(rows),
        "correct": correct,
        "em": correct / len(rows) if rows else None,
        "strict_correct": sum(bool(r["strict_em"]) for r in rows),
        "extracted_correct": sum(bool(r["em"]) and not r["strict_em"] for r in rows),
        "valid": len(valid),
        "valid_wrong": len(valid) - correct,
        "empty": empty,
        "length_empty": length_empty,
        "other_empty": empty - length_empty,
        "length_terminated": sum(r["termination"] == "length" for r in rows),
        "invalid_nonempty": counts.get("invalid", 0),
        "refusal": counts.get("refusal", 0),
        "cell_accuracy_valid_only": sum(r["cell_accuracy"] for r in valid) / len(valid)
        if valid
        else None,
        "wrong_cells_median_valid_incorrect": median(wrong) if wrong else None,
        "output_tokens": sum(r["usage"]["output_tokens"] for r in rows),
    }
