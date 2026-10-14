from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = PROJECT_ROOT / "examples" / "metric_paint.json"


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    existing_path = env.get("PYTHONPATH")
    source_path = str(PROJECT_ROOT / "src")
    env["PYTHONPATH"] = (
        source_path if not existing_path else os.pathsep.join((source_path, existing_path))
    )
    return subprocess.run(
        [sys.executable, "-m", "spatial_worlds", *args],
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_families_command_lists_all_families() -> None:
    result = run_cli("families")

    assert result.returncode == 0, result.stderr
    assert len([line for line in result.stdout.splitlines() if line.strip()]) == 8
    assert "metric_paint" in result.stdout
    assert "aggregating_compression" in result.stdout
    assert "implemented" in result.stdout

    as_json = run_cli("families", "--json")
    assert as_json.returncode == 0
    assert len(json.loads(as_json.stdout)) == 8


def test_simulate_command_matches_shipped_gold(tmp_path: Path) -> None:
    gold_path = tmp_path / "gold.json"
    trace_path = tmp_path / "trace.json"
    result = run_cli("simulate", str(EXAMPLE), "--gold", str(gold_path), "--trace", str(trace_path))

    assert result.returncode == 0, result.stderr
    shipped = json.loads((PROJECT_ROOT / "examples" / "metric_paint.gold.json").read_text())
    assert json.loads(gold_path.read_text()) == shipped
    trace = json.loads(trace_path.read_text())
    assert len(trace["steps"]) == 3 and set(trace["states"]) == {"0", "1", "2", "3"}


def test_generate_run_evaluate_report_commands(tmp_path: Path) -> None:
    config = tmp_path / "cfg.json"
    config.write_text(
        json.dumps(
            {
                "name": "cli",
                "dataset_seed": 1,
                "split": "dev",
                "tier": "S",
                "dims": [1],
                "t_grid": [0, 1, 2],
                "t_max": 2,
                "seeds_per_cell": 1,
                "families": {"metric_paint": {"w_targets": [7]}},
            }
        )
    )
    data = tmp_path / "data"
    generated = run_cli("generate", "--config", str(config), "--out", str(data), "--quiet")
    assert generated.returncode == 0, generated.stderr
    assert "wrote 3 instances" in generated.stdout
    predictions = tmp_path / "pred.jsonl"
    ran = run_cli(
        "run", "--data", str(data), "--provider", "oracle", "--out", str(predictions), "--quiet"
    )
    assert ran.returncode == 0, ran.stderr
    results = tmp_path / "results"
    evaluated = run_cli(
        "evaluate",
        "--data",
        str(data),
        "--predictions",
        str(predictions),
        "--out",
        str(results),
        "--bootstrap",
        "5",
    )
    assert evaluated.returncode == 0, evaluated.stderr
    reported = run_cli("report", "--results", str(results))
    assert reported.returncode == 0, reported.stderr
    assert (results / "summary.md").exists()
    rendered = run_cli("render-prompt", str(data / "dev" / "metric_paint" / "instances.jsonl"))
    assert rendered.returncode == 2


def test_validate_and_render_example() -> None:
    validation = run_cli("validate-instance", str(EXAMPLE))
    rendered = run_cli("render-prompt", str(EXAMPLE))

    assert validation.returncode == 0, validation.stderr
    assert "valid" in validation.stdout.lower()
    assert rendered.returncode == 0, rendered.stderr
    assert "PAINT" in rendered.stdout
    assert "SET" in rendered.stdout


def test_score_command_reports_exact_match(tmp_path: Path) -> None:
    gold_path = tmp_path / "gold.json"
    prediction_path = tmp_path / "prediction.json"
    gold_path.write_text(
        json.dumps(
            {
                "schema_version": "spatial-worlds/0.2",
                "instance_id": "cli-score",
                "family": "metric_paint",
                "world": {"shape": [2, 2], "values": [0, 1, 2]},
                "final_state": {"cells": [[[0, 0], 1]]},
                "workloads": [],
            }
        ),
        encoding="utf-8",
    )
    prediction_path.write_text('{"cells":[[[0,0],1]]}', encoding="utf-8")

    result = run_cli("score", str(gold_path), str(prediction_path))

    assert result.returncode == 0, result.stderr
    assert '"exact_match"' in result.stdout
    assert "true" in result.stdout.lower()


def test_validate_instance_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    instance_path = tmp_path / "duplicate.json"
    payload = EXAMPLE.read_text(encoding="utf-8")
    instance_path.write_text(payload.replace('"id":', '"id":"shadowed","id":', 1))

    result = run_cli("validate-instance", str(instance_path))

    assert result.returncode == 2
    assert "duplicate" in result.stderr.lower()


def test_validate_instance_rejects_dynamically_illegal_sequence(tmp_path: Path) -> None:
    instance_path = tmp_path / "illegal-compression.json"
    instance_path.write_text(
        json.dumps(
            {
                "schema_version": "spatial-worlds/0.2",
                "id": "illegal-compression",
                "family": "aggregating_compression",
                "world": {"shape": [2], "values": [0, 2, 4]},
                "initial": {"cells": [[[0], 4], [[1], 4]]},
                "events": [{"op": "COMPRESS", "axis": 0, "direction": -1}],
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )

    result = run_cli("validate-instance", str(instance_path))

    assert result.returncode == 2
    assert "outside world.values" in result.stderr
