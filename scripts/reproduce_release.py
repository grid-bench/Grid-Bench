"""Replay reference states and reconstruct tables from released measurements offline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_completed import group_stats
from spatial_worlds.eval.aggregate import score_formal_record
from spatial_worlds.eval.expanded_analysis_registry import COLORS, LENGTHS, MARKERS, MODELS
from spatial_worlds.prompt import COMPLETION_PROMPT_VERSION, render_prompt
from spatial_worlds.schema import GoldRecord, Instance, parse_json_document
from spatial_worlds.simulate import simulate

SCORE_FIELDS = (
    "em", "strict_em", "valid_json", "valid_state", "canonical_format",
    "wrong_cells", "cell_accuracy", "changed_cell_accuracy", "unchanged_cell_accuracy",
    "answer_status", "extraction_applied", "extraction_method", "analysis_eligible",
    "total_cells", "changed_cells", "copy_baseline_em", "copy_baseline_cell_accuracy",
    "zero_baseline_em", "zero_baseline_cell_accuracy", "score_error",
)
TABLE_KEYS = {
    "overview": ("model", "view"), "resources": ("model", "view"),
    "length": ("model", "T"), "task_length": ("model", "family", "T"),
    "dimension_length": ("model", "dimension", "T"),
    "workload": ("model", "family", "w", "T"),
}
TABLE_COUNTS = dict(overview=12, resources=6, length=36, task_length=288,
                    dimension_length=108, workload=396)


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [parse_json_document(line) for line in path.read_text().splitlines() if line.strip()]


def verify_manifest(root: Path) -> dict[str, Any]:
    """Verify safe, complete member inventory without trusting archive paths."""
    manifest = parse_json_document((root / "manifest.json").read_text())
    members = manifest["members"]
    for name, expected in members.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError(f"Unsafe release member: {name}")
        path = root / name
        if any((root / Path(*relative.parts[:i])).is_symlink()
               for i in range(1, len(relative.parts) + 1)):
            raise ValueError(f"Symlink in release member: {name}")
        payload = path.read_bytes()
        if digest(payload) != expected["sha256"] or len(payload) != expected["bytes"]:
            raise ValueError(f"Release checksum mismatch: {name}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*") if path.is_file()
        and ".git" not in path.relative_to(root).parts
    } - {"manifest.json"}
    if actual != set(members):
        raise ValueError("Release member inventory differs from manifest")
    return manifest


def same_value(first: Any, second: Any) -> bool:
    if isinstance(first, float) and isinstance(second, (float, int)):
        return math.isclose(first, second, rel_tol=1e-12, abs_tol=1e-12)
    return first == second


def reference_index(benchmark: list[dict[str, Any]], *, check_references: bool = True
                    ) -> dict[str, tuple[dict, Instance, GoldRecord]]:
    """Check formal prompts and optionally replay every reference world."""
    index: dict[str, tuple[dict, Instance, GoldRecord]] = {}
    for item in benchmark:
        identity = item["public_id"]
        if identity in index:
            raise ValueError(f"Duplicate benchmark ID: {identity}")
        instance = Instance.from_dict(parse_json_document(item["instance_json"]))
        gold = GoldRecord.from_dict(parse_json_document(item["reference_json"]))
        if instance.id != item["instance_id"] or gold.instance_id != instance.id:
            raise ValueError(f"Instance/reference identity mismatch: {identity}")
        prompt = render_prompt(instance, prompt_version=COMPLETION_PROMPT_VERSION)
        if item["prompt_version"] != COMPLETION_PROMPT_VERSION:
            raise ValueError(f"Unexpected formal prompt version: {identity}")
        if digest(prompt.encode()) != item["prompt_sha256"] or prompt != item["prompt"]:
            raise ValueError(f"Formal prompt mismatch: {identity}")
        if check_references and simulate(instance).final_state != gold.final_state:
            raise ValueError(f"Reference execution mismatch: {identity}")
        index[identity] = item, instance, gold

    return index


def rescore(
    benchmark: list[dict[str, Any]], outcomes: list[dict[str, Any]],
    *, check_references: bool = True,
) -> list[dict[str, Any]]:
    """Check supplied private answers with the official extractor/scorer."""
    index = reference_index(benchmark, check_references=check_references)

    seen: set[tuple[str, str]] = set()
    result: list[dict[str, Any]] = []
    for outcome in outcomes:
        key = outcome["model_id"], outcome["public_id"]
        if key in seen:
            raise ValueError(f"Duplicate model-instance outcome: {key}")
        seen.add(key)
        if key[1] not in index:
            raise ValueError(f"Unknown outcome instance: {key}")
        item, instance, gold = index[key[1]]
        raw = outcome["delivered_text"]
        if digest(raw.encode()) != outcome["delivered_text_sha256"]:
            raise ValueError(f"Delivered-answer hash mismatch: {key}")
        record = {
            "raw_text": raw,
            "model": key[0],
            "dataset": item["source_dataset"],
            "scoring_version": outcome["scoring_version"],
            "evaluation_protocol": outcome["evaluation_protocol"],
            "answer_eligible": outcome["answer_eligible"],
            "collection_finalized": outcome["collection_finalized"],
            "protocol_violation": outcome["protocol_violation"],
            "refusal": outcome["refusal"],
            "termination": outcome["termination"],
            "finish_reason": outcome["finish_reason"],
            "usage": parse_json_document(outcome["usage_json"]),
        }
        scored = score_formal_record(record, instance, gold)
        for field in SCORE_FIELDS:
            if field not in outcome or not same_value(scored[field], outcome[field]):
                raise ValueError(f"Re-score mismatch: {key} {field}")
        scored.update(
            public_id=key[1], in_main=item["in_main"], in_workload=item["in_workload"],
            visible_text_empty=not raw.strip(), model_name=outcome["model_name"],
        )
        if scored["visible_text_empty"] != outcome["visible_text_empty"]:
            raise ValueError(f"Answer-content flag mismatch: {key}")
        result.append(scored)
    if result:
        models = {row["model"] for row in result}
        if seen != {(model, identity) for model in models for identity in index}:
            raise ValueError("Incomplete shared model-instance panel")
    elif benchmark:
        raise ValueError("Missing outcome panel")
    return result


def result_rows(benchmark: list[dict[str, Any]], results: list[dict[str, Any]],
                *, check_references: bool = True) -> list[dict[str, Any]]:
    """Join recorded measurements for aggregation, without claiming answer rescoring."""
    index = reference_index(benchmark, check_references=check_references)
    seen = set()
    rows = []
    for measured in results:
        if {"delivered_text", "raw_text", "prediction"}.intersection(measured):
            raise ValueError("Model answers are outside the public result schema")
        key = measured["model_id"], measured["public_id"]
        if key in seen or key[1] not in index:
            raise ValueError("Duplicate result or unknown instance")
        seen.add(key)
        item, instance, _ = index[key[1]]
        if measured["em"] and (not measured["valid_state"] or measured["wrong_cells"] != 0):
            raise ValueError("Inconsistent exact-match measurement")
        if measured["strict_em"] and not measured["em"]:
            raise ValueError("Inconsistent strict exact-match measurement")
        rows.append({
            **measured, "model": key[0], "dataset": item["source_dataset"],
            "family": instance.family, "dimension": instance.dimension, "T": instance.horizon,
            "w": instance.to_dict()["metadata"].get("nominal_scan_workload", 0),
            "in_main": item["in_main"], "in_workload": item["in_workload"],
            "usage": parse_json_document(measured["usage_json"]),
        })
    models = {row["model"] for row in rows}
    if (not rows and benchmark) or seen != {(m, identity) for m in models for identity in index}:
        raise ValueError("Incomplete shared model-instance result panel")
    return rows


def validate_table_coverage(rows: list[dict], tables: dict[str, list[dict]]) -> None:
    if set(tables) != set(TABLE_KEYS):
        raise ValueError("Missing or unexpected frozen tables")
    for name, fields in TABLE_KEYS.items():
        table = tables[name]
        observed = {tuple(row[field] for field in fields) for row in table}
        if len(table) != TABLE_COUNTS[name] or len(observed) != len(table):
            raise ValueError(f"Missing or duplicate frozen conditions: {name}")
        if name == "overview":
            expected = {(model, view) for model in MODELS for view in ("main", "main_T8_64")}
        elif name == "resources":
            expected = {(model, "unique_1488") for model in MODELS}
        else:
            membership = "in_workload" if name == "workload" else "in_main"
            expected = {
                tuple(row[field] for field in fields) for row in rows if row[membership]
            }
        if observed != expected:
            raise ValueError(f"Frozen condition coverage differs from outcomes: {name}")


def reproduce_tables(rows: list[dict], frozen: dict) -> dict[str, list[dict]]:
    """Recompute descriptive statistics; retain explicitly frozen bootstrap intervals."""
    validate_table_coverage(rows, frozen["tables"])
    output: dict[str, list[dict]] = {}
    for name, expected_rows in frozen["tables"].items():
        output[name] = []
        for expected in expected_rows:
            subset = [row for row in rows if row["model"] == expected["model"]]
            if name == "workload":
                subset = [row for row in subset if row["in_workload"]]
            elif name != "resources":
                subset = [row for row in subset if row["in_main"]]
            if expected.get("view") == "main_T8_64":
                subset = [row for row in subset if row["T"] in (8, 16, 32, 64)]
            for key in ("family", "dimension", "T", "w"):
                if key in expected:
                    subset = [row for row in subset if row[key] == expected[key]]
            computed = group_stats(subset)
            for key, value in computed.items():
                if key not in expected or not same_value(value, expected[key]):
                    raise ValueError(f"Frozen descriptive-statistic mismatch: {name} {key}")
            output[name].append({**expected, **computed})
    return output


def write_tables(tables: dict[str, list[dict]], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        if not rows:
            continue
        with (destination / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def draw_length_plot(rows: list[dict], names: dict[str, str], destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 4.6), layout="constrained")
    for model in MODELS:
        points = sorted((row for row in rows if row["model"] == model), key=lambda r: r["T"])
        xs = list(range(len(points)))
        ax.plot(xs, [row["em"] * 100 for row in points], label=names[model],
                color=COLORS[model], marker=MARKERS[model], linewidth=1.6, markersize=5)
        low = [float("nan") if r["ci95_low"] is None else 100 * r["ci95_low"] for r in points]
        high = [float("nan") if r["ci95_high"] is None else 100 * r["ci95_high"] for r in points]
        ax.fill_between(xs, low, high, color=COLORS[model], alpha=0.10, linewidth=0)
    ax.set(xticks=range(len(LENGTHS)), xticklabels=LENGTHS, ylim=(-2, 102),
           xlabel="Event history length T", ylabel="Exact final-state accuracy (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=8, ncol=2, loc="upper right", frameon=False)
    destination.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination / "history-length.png", dpi=180)
    fig.savefig(destination / "history-length.pdf", metadata={"CreationDate": None})
    plt.close(fig)


def export_cli(benchmark: list[dict], destination: Path) -> None:
    """Write native instance/gold records, preserving source-dataset boundaries."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in benchmark:
        name = item["source_dataset"]
        if name not in {"main", "rq2"}:
            raise ValueError(f"Unknown source dataset: {name}")
        groups[name].append(item)
    for name, items in groups.items():
        folder = destination / name
        folder.mkdir(parents=True, exist_ok=True)
        fields = (("instance_json", "instances.jsonl"), ("reference_json", "gold.jsonl"))
        for field, filename in fields:
            (folder / filename).write_text("".join(item[field] + "\n" for item in items))
    (destination / "README.md").write_text(
        "# Native CLI records\n\n"
        "These files contain only the formal selected instances. Use the explicit\n"
        "`--instances` and `--gold` arguments for each source dataset separately:\n\n"
        "```sh\nspatial-worlds run --instances main/instances.jsonl "
        "--gold main/gold.jsonl --provider oracle --out main-predictions.jsonl\n"
        "spatial-worlds evaluate --instances main/instances.jsonl --gold main/gold.jsonl "
        "--predictions main-predictions.jsonl --out main-results\n```\n\n"
        "Repeat with `rq2` for the additional workload conditions. The workload\n"
        "analysis also reuses main instances; membership is retained in the Hub table.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Local dataset candidate root")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--export-cli", type=Path, help="Also export native CLI records")
    parser.add_argument("--no-plot", action="store_true", help="Run without matplotlib")
    args = parser.parse_args()
    data, out = args.data.resolve(), args.out.resolve()
    if out == data or data in out.parents or out in data.parents:
        parser.error("--out must be separate from the input release")
    if args.export_cli:
        cli = args.export_cli.resolve()
        if cli == data or data in cli.parents or cli in data.parents:
            parser.error("--export-cli must be separate from the input release")
    verify_manifest(data)
    benchmark = read_jsonl(data / "data/benchmark.jsonl")
    results = read_jsonl(data / "data/results.jsonl")
    if len(benchmark) != 1488 or len(results) != 8928:
        raise ValueError("This command expects the frozen 1,488-instance, six-model release")
    if {row["model_id"] for row in results} != set(MODELS):
        raise ValueError("Unexpected model panel")
    rows = result_rows(benchmark, results)
    frozen = parse_json_document((data / "evidence/sources/analysis.json").read_text())
    tables = reproduce_tables(rows, frozen)
    write_tables(tables, out / "tables")
    if not args.no_plot:
        names = {row["model_id"]: row["model_name"] for row in results}
        draw_length_plot(tables["length"], names, out / "figures")
    if args.export_cli:
        export_cli(benchmark, args.export_cli.resolve())
    report = {
        "format": "grid-bench-release-reproduction/2",
        "input_manifest_sha256": digest((data / "manifest.json").read_bytes()),
        "reference_states_reexecuted": len(benchmark),
        "result_records_aggregated": len(rows),
        "delivered_answers_rescored": 0,
        "model_outputs": "Not distributed; individual scores are recorded measurements.",
        "descriptive_table_rows_checked": sum(len(part) for part in tables.values()),
        "score_mismatches": None,
        "table_mismatches": 0,
        "bootstrap_intervals": "Frozen intervals retained; not rerun or independently checked.",
        "plot": None if args.no_plot else "figures/history-length.png",
    }
    (out / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
