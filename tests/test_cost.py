"""Observed cost per record, the cost tables, and the projection from a basis."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spatial_worlds.cli import main
from spatial_worlds.eval.aggregate import evaluate, score_records, summarize_cells
from spatial_worlds.eval.cost import (
    Price,
    build_basis,
    format_projection,
    load_prices,
    project_cost,
    record_cost_usd,
    select_instances,
)
from spatial_worlds.eval.providers import OracleProvider
from spatial_worlds.eval.run import load_dataset, run_predictions
from spatial_worlds.generation import build_dataset, config_from_dict

TINY = {
    "name": "tiny-cost",
    "dataset_seed": 11,
    "split": "dev",
    "tier": "S",
    "dims": [2],
    "t_grid": [0, 1, 2, 4, 8],
    "t_max": 8,
    "seeds_per_cell": 2,
    "families": {"metric_paint": {"w_targets": [7]}, "boundary_layer": {"w_targets": [6]}},
}


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data")
    build_dataset(config_from_dict(TINY), path)
    return path


def test_prices_and_record_cost(tmp_path: Path) -> None:
    path = tmp_path / "prices.json"
    path.write_text(
        json.dumps(
            {
                "usd_per_million": {
                    "m": {"input": 5, "output": 30, "note": "ignored"},
                    "cheap": {"input": 0.5, "output": 1.5},
                }
            }
        )
    )
    prices = load_prices(path)
    assert prices["m"] == Price(5.0, 30.0)
    usage = {"input_tokens": 1000, "output_tokens": 2000, "reasoning_tokens": 1500}
    assert record_cost_usd(usage, prices["m"]) == pytest.approx(0.005 + 0.06)
    assert record_cost_usd({"input_tokens": 1000}, prices["m"]) is None
    assert record_cost_usd(
        {"input_tokens": 1000, "output_tokens": 0}, prices["m"]
    ) == pytest.approx(0.005)
    assert record_cost_usd(usage, None) is None
    assert record_cost_usd({}, prices["m"]) is None
    assert record_cost_usd({"input_tokens": None, "output_tokens": None}, prices["m"]) is None
    path.write_text(json.dumps({"usd_per_million": {"m": {"input": "5"}}}))
    with pytest.raises(ValueError):
        load_prices(path)
    path.write_text(json.dumps({"prices": {}}))
    with pytest.raises(ValueError):
        load_prices(path)


def _usage_records(dataset_dir: Path, tmp_path: Path, model: str) -> Path:
    """Oracle predictions relabelled as a priced model with usage that grows with T."""

    dataset = load_dataset(dataset_dir)
    oracle_out = tmp_path / "oracle.jsonl"
    run_predictions(dataset, OracleProvider(), out=oracle_out, concurrency=2)
    horizons = {instance.id: instance.horizon for instance in dataset.instances}
    out = tmp_path / f"{model}.jsonl"
    with out.open("w") as stream:
        for line in oracle_out.read_text().splitlines():
            record = json.loads(line)
            horizon = horizons[record["instance_id"]]
            record = {
                **record,
                "provider": "openai",
                "model": model,
                "usage": {
                    "input_tokens": 1000 + 100 * horizon,
                    "output_tokens": 2000 * (horizon + 1),
                },
                "latency_s": 10.0 * (horizon + 1),
                "attempts": 1,
            }
            record.pop("condition_id", None)
            stream.write(json.dumps(record) + "\n")
    return out


def test_scored_rows_cells_and_report_carry_cost(dataset_dir: Path, tmp_path: Path) -> None:
    dataset = load_dataset(dataset_dir)
    predictions = _usage_records(dataset_dir, tmp_path, "m")
    prices = {"m": Price(5.0, 30.0)}
    records = [json.loads(line) for line in predictions.read_text().splitlines()]
    rows = score_records(dataset, records, prices)
    for row in rows:
        expected = (1000 + 100 * row["T"]) * 5 / 1e6 + 2000 * (row["T"] + 1) * 30 / 1e6
        assert row["cost_usd"] == pytest.approx(expected)
    assert all(r["cost_usd"] is None for r in score_records(dataset, records))
    cells = summarize_cells(rows)
    cell = next(c for c in cells if c["family"] == "metric_paint" and c["T"] == 4)
    assert cell["cost_known_rate"] == 1.0
    assert cell["total_cost_usd"] == pytest.approx(cell["n"] * cell["mean_cost_usd"])

    results = tmp_path / "results"
    prices_path = tmp_path / "prices.json"
    prices_path.write_text(json.dumps({"usd_per_million": {"m": {"input": 5, "output": 30}}}))
    evaluate(dataset, [predictions], results, bootstrap=5, prices=load_prices(prices_path))
    document = json.loads((results / "results.json").read_text())
    costs = document["costs"]
    pooled = [c for c in costs if c["family"] == "all"]
    assert sorted(c["T"] for c in pooled) == [0, 1, 2, 4, 8]
    at_eight = next(c for c in pooled if c["T"] == 8)
    assert at_eight["usage_known"] == at_eight["n"] and at_eight["answered"] == at_eight["n"]
    assert at_eight["mean_output_tokens"] == pytest.approx(18000.0)
    assert at_eight["mean_cost_usd"] == pytest.approx((1800 * 5 + 18000 * 30) / 1e6)
    per_family = [c for c in costs if c["family"] == "metric_paint" and c["T"] == 0]
    assert len(per_family) == 1 and per_family[0]["mean_latency_s"] == pytest.approx(10.0)
    assert (results / "costs.csv").read_text().startswith("condition_id,model,family,T,n")
    summary = (results / "summary.md").read_text()
    assert "### Cost and usage by horizon" in summary
    assert "### Mean cost per instance by family and T (USD)" in summary
    assert "Total observed cost: $" in summary
    # a baseline condition without usage gets no cost section
    oracle_results = tmp_path / "oracle-results"
    evaluate(dataset, [tmp_path / "oracle.jsonl"], oracle_results, bootstrap=5)
    assert "Cost and usage" not in (oracle_results / "summary.md").read_text()


def test_projection_uses_the_most_specific_basis_and_reports_fallbacks(
    dataset_dir: Path,
) -> None:
    dataset = load_dataset(dataset_dir)

    def row(family: str, T: int, output: float, model: str = "m") -> dict[str, object]:
        return {
            "model": model,
            "family": family,
            "T": T,
            "usage": {"input_tokens": 1000.0, "output_tokens": output},
        }

    rows = [
        row("metric_paint", 0, 100.0),
        row("metric_paint", 0, 300.0),
        row("metric_paint", 8, 8000.0),
        row("boundary_layer", 8, 4000.0),
        row("metric_paint", 4, 4000.0, model="other"),
        {"model": "m", "family": "metric_paint", "T": 4, "usage": {}},
    ]
    basis = build_basis(rows, "m")
    assert basis.by_family_T[("metric_paint", 0)].mean_output_tokens == 200.0
    assert basis.by_T[8].mean_output_tokens == 6000.0
    assert basis.overall is not None and basis.overall.n == 4
    assert basis.lookup("metric_paint", 0)[1] == "family_T"
    assert basis.lookup("boundary_layer", 0)[1] == "T"
    assert basis.lookup("metric_paint", 4)[1] == "overall"

    price = Price(5.0, 30.0)
    paint_t0 = select_instances(dataset.instances, ["family=metric_paint,T=0"])
    assert len(paint_t0) == 2
    projection = project_cost(paint_t0, basis, price, model="m")
    assert projection["cost_usd"] == pytest.approx(2 * (1000 * 5 + 200 * 30) / 1e6)
    assert projection["basis_levels"] == {"family_T": 2, "T": 0, "overall": 0, "none": 0}
    doubled = project_cost(paint_t0, basis, price, model="big", multiplier=2.0)
    assert doubled["cost_usd"] == pytest.approx(2 * (1000 * 5 + 400 * 30) / 1e6)
    union = select_instances(
        dataset.instances, ["family=metric_paint,T=0", "family=boundary_layer,T=0", "T=4"]
    )
    assert len(union) == 2 + 2 + 4
    mixed = project_cost(union, basis, price, model="m")
    # paint at T = 0 has its own rows, layer at T = 0 falls back to the horizon,
    # and T = 4 (no usable row) to the overall mean
    assert mixed["basis_levels"] == {"family_T": 2, "T": 2, "overall": 4, "none": 0}
    assert set(mixed["by_T"]) == {0, 4} and set(mixed["by_family"]) == {
        "metric_paint",
        "boundary_layer",
    }
    everything = project_cost(dataset.instances, basis, price, model="m")
    assert everything["basis_levels"]["overall"] > 0 and everything["basis_levels"]["none"] == 0
    line = format_projection(mixed, "stage")
    assert line.startswith("stage: m: 8 instances, $") and "fallbacks: T 2, overall 4" in line
    empty = build_basis(rows, "nobody")
    assert empty.overall is None and empty.lookup("metric_paint", 0) == (None, "none")


def test_cost_command_projects_from_a_scored_run(
    dataset_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    predictions = _usage_records(dataset_dir, tmp_path, "m")
    prices_path = tmp_path / "prices.json"
    prices_path.write_text(
        json.dumps(
            {
                "usd_per_million": {
                    "m": {"input": 5, "output": 30},
                    "big": {"input": 10, "output": 50},
                }
            }
        )
    )
    results = tmp_path / "results"
    assert (
        main(
            [
                "evaluate",
                "--data",
                str(dataset_dir),
                "--predictions",
                str(predictions),
                "--out",
                str(results),
                "--bootstrap",
                "5",
                "--prices",
                str(prices_path),
                "--split-dimensions",
                "2",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert (
        main(
            [
                "cost",
                "--data",
                str(dataset_dir),
                "--filter",
                "family=metric_paint,T=8",
                "--basis",
                str(results / "results.jsonl"),
                "--basis-model",
                "m",
                "--prices",
                str(prices_path),
                "--model",
                "big",
                "--multiplier",
                "1.5",
                "--label",
                "demo",
                "--json",
            ]
        )
        == 0
    )
    projection = json.loads(capsys.readouterr().out)
    assert projection["label"] == "demo" and projection["instances"] == 2
    expected = 2 * (1800 * 10 + 18000 * 1.5 * 50) / 1e6
    assert projection["cost_usd"] == pytest.approx(expected)
    assert projection["basis_levels"]["family_T"] == 2
    with pytest.raises(SystemExit):
        main(["cost", "--data", str(dataset_dir), "--basis", "x", "--prices", "y"])
    assert (
        main(
            [
                "cost",
                "--data",
                str(dataset_dir),
                "--basis",
                str(results / "results.jsonl"),
                "--prices",
                str(prices_path),
                "--model",
                "unpriced",
            ]
        )
        == 2
    )
