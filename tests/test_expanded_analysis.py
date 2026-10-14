"""Checks for eight-trajectory derived evidence without provider calls."""

import importlib
import sys
from pathlib import Path

from spatial_worlds.eval.analysis_registry import LENGTHS, MODELS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
analyze_expanded = importlib.import_module("scripts.analyze_expanded")


def test_prefix_diagnostics_retains_later_success_after_early_failure(monkeypatch) -> None:
    monkeypatch.setattr(analyze_expanded, "MAIN_TRAJECTORIES", 2)
    rows = []
    for model in MODELS:
        for rollout, flags in (("one", (True,) * 6), ("two", (True, True, False, True, False, False))):
            rows.extend(
                {
                    "model": model,
                    "rollout_id": rollout,
                    "T": t,
                    "em": flag,
                }
                for t, flag in zip(LENGTHS, flags, strict=True)
            )
    result = analyze_expanded.prefix_diagnostics(rows)
    assert all(row["wrong_then_correct"] == 1 for row in result["summary"])
    assert all(row["positive_wrong_then_correct"] == 1 for row in result["summary"])
    assert all(
        row["gain"] == 1
        for row in result["transitions"]
        if row["from_T"] == 8 and row["to_T"] == 16
    )
