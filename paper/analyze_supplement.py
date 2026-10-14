"""Deterministic, read-only-source supplement for the expanded manuscript.

Run with the project environment. --check-only reconstructs every exported
number without writing. Source campaigns are checked against the frozen
six-model analysis before use. No provider client or network is involved.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean

from spatial_worlds.eval.expanded_analysis_registry import (
    ALL_TRAJECTORIES,
    ANALYSIS_DIR,
    LENGTHS,
    MAIN_TRAJECTORIES,
    MODELS,
    PER_MODEL_INSTANCES,
    POSITIVE_PREFIX_REVERSALS,
    PREFIX_REVERSALS,
)
from spatial_worlds.eval.study_final_config import request_parameters

ROOT = Path(__file__).resolve().parents[1]

OUT = ROOT / "paper/data/revision-supplement.json"
SNAPSHOT = ROOT / ANALYSIS_DIR / "analysis.json"
LEDGER = ROOT / "data/study-v1/attempts.sqlite"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encoded(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"


def percentile(values: list[float], q: float) -> float | None:
    """Linear interpolation at (n-1)*q; matches numpy method='linear'."""
    if not 0 <= q <= 1:
        raise ValueError("quantile outside [0,1]")
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def mean(values: list[float]) -> float | None:
    return fmean(values) if values else None


def group(rows: list[dict], fields: tuple[str, ...]) -> list[tuple[dict, list[dict]]]:
    bins = defaultdict(list)
    for row in rows:
        bins[tuple(row[field] for field in fields)].append(row)
    return [(dict(zip(fields, key, strict=True)), bins[key]) for key in sorted(bins)]


def counts(rows: list[dict]) -> dict:
    result = {
        "n": len(rows),
        "correct": sum(r["em"] for r in rows),
        "valid": sum(r["valid_state"] for r in rows),
        "empty": sum(r["visible_text_empty"] for r in rows),
        "length": sum(r["termination"] == "length" for r in rows),
        "strict_correct": sum(r["strict_em"] for r in rows),
    }
    result["valid_wrong"] = result["valid"] - result["correct"]
    result["invalid"] = result["n"] - result["valid"] - result["empty"]
    assert result["invalid"] >= 0
    return result


def error_stats(rows: list[dict]) -> dict:
    wrong = [r for r in rows if r["valid_state"] and not r["em"]]
    values = sorted(r["wrong_cells"] for r in wrong)
    parts = Counter()
    for row in wrong:
        c = row["changed_cell_accuracy"]
        u = row["unchanged_cell_accuracy"]
        changed = c is not None and c < 1
        unchanged = u is not None and u < 1
        assert changed or unchanged
        parts[
            "both" if changed and unchanged else "changed_only" if changed else "unchanged_only"
        ] += 1
    assert sum(parts.values()) == len(wrong)
    return {
        "n": len(rows),
        "valid_wrong_n": len(wrong),
        "wrong_cells": values,
        "median": percentile(values, 0.5),
        "q25": percentile(values, 0.25),
        "q75": percentile(values, 0.75),
        "p90": percentile(values, 0.9),
        "near5": sum(v <= 5 for v in values),
        **{k: parts[k] for k in ("changed_only", "unchanged_only", "both")},
    }


def baseline_stats(rows: list[dict]) -> dict:
    valid = [r for r in rows if r["valid_state"]]
    return {
        "n": len(rows),
        "valid_n": len(valid),
        "model_ca": mean([r["cell_accuracy"] for r in valid]),
        "copy_ca": mean([r["copy_baseline_cell_accuracy"] for r in valid]),
        "zero_ca": mean([r["zero_baseline_cell_accuracy"] for r in valid]),
        "above_copy": sum(r["cell_accuracy"] > r["copy_baseline_cell_accuracy"] for r in valid),
        "changed_n": sum(r["changed_cell_accuracy"] is not None for r in valid),
        "unchanged_n": sum(r["unchanged_cell_accuracy"] is not None for r in valid),
        "changed_ca": mean(
            [r["changed_cell_accuracy"] for r in valid if r["changed_cell_accuracy"] is not None]
        ),
        "unchanged_ca": mean(
            [
                r["unchanged_cell_accuracy"]
                for r in valid
                if r["unchanged_cell_accuracy"] is not None
            ]
        ),
        "copy_correct": sum(r["copy_baseline_em"] for r in rows),
        "zero_correct": sum(r["zero_baseline_em"] for r in rows),
    }


def has_prefix_reversal(flags: list[bool]) -> bool:
    """An exact later independently queried prefix follows an earlier error."""
    return any(flags[i] and not all(flags[:i]) for i in range(1, len(flags)))


def paired_point_diagnostics(rows: list[dict], summaries: list[dict]) -> list[dict]:
    """Attach auditable same-instance coordinates to frozen conditional summaries.

    The masks are reconstructed independently before coordinates are exported;
    plotting consumers never need to filter a provider's raw result records.
    ``rollout_id`` identifies the shared trajectory, while ``instance_id`` also
    identifies its queried prefix. Zero-error points remain in joint_valid.
    """
    index = {(r["model"], r["T"], r["instance_id"]): r for r in rows}
    assert len(index) == len(rows), "Duplicate outcome in paired-point evidence"
    result = []
    identity_fields = ("dataset", "rollout_id", "family", "dimension", "total_cells")
    for summary in summaries:
        a, b, t = summary["model_a"], summary["model_b"], summary["T"]
        left = {key[2]: r for key, r in index.items() if key[:2] == (a, t)}
        right = {key[2]: r for key, r in index.items() if key[:2] == (b, t)}
        assert left.keys() == right.keys(), "Paired-point declared instance masks differ"
        assert len(left) == summary["n_total"]
        valid = sorted(k for k in left if left[k]["valid_state"] and right[k]["valid_state"])
        subset = summary["subset"]
        assert subset in ("joint_valid", "joint_valid_incorrect")
        selected = (
            valid
            if subset == "joint_valid"
            else [k for k in valid if not left[k]["em"] and not right[k]["em"]]
        )
        assert selected == summary["instance_ids"], "Paired-point conditional mask changed"
        assert len(selected) == summary["n"]
        points = []
        for instance_id in selected:
            lrow, rrow = left[instance_id], right[instance_id]
            assert all(lrow[field] == rrow[field] for field in identity_fields), (
                "Paired-point instance metadata differs"
            )
            point = {"instance_id": instance_id, **{f: lrow[f] for f in identity_fields}}
            for side, row in (("a", lrow), ("b", rrow)):
                wrong = row["wrong_cells"]
                assert isinstance(wrong, int) and 0 <= wrong <= row["total_cells"]
                assert row["em"] == (wrong == 0)
                for field in ("wrong_cells", "em", "valid_state", "cell_accuracy"):
                    point[f"{field}_{side}"] = row[field]
            points.append(point)
        for side in ("a", "b"):
            assert sum(p[f"em_{side}"] for p in points) == summary[f"correct_{side}"]
            assert percentile([p[f"wrong_cells_{side}"] for p in points], 0.5) == summary[
                f"wrong_cells_median_{side}"
            ]
            ca = mean([p[f"cell_accuracy_{side}"] for p in points])
            expected = summary[f"cell_accuracy_{side}"]
            assert ca is expected or (
                ca is not None and expected is not None and math.isclose(ca, expected)
            )
        assert sum(p["wrong_cells_a"] < p["wrong_cells_b"] for p in points) == summary[
            "a_fewer_wrong"
        ]
        assert sum(p["wrong_cells_a"] == p["wrong_cells_b"] for p in points) == summary[
            "equal_wrong"
        ]
        assert sum(p["wrong_cells_a"] > p["wrong_cells_b"] for p in points) == summary[
            "b_fewer_wrong"
        ]
        result.append({**summary, "paired_points": points})
    return result


def prefix_stats(rows: list[dict]) -> dict:
    histories, transitions = [], []
    for model in MODELS:
        trajectories = [
            sorted(rs, key=lambda r: r["T"])
            for _, rs in group([r for r in rows if r["model"] == model], ("rollout_id",))
        ]
        assert len(trajectories) == MAIN_TRAJECTORIES
        for trajectory in trajectories:
            assert tuple(r["T"] for r in trajectory) == LENGTHS
            flags = [r["em"] for r in trajectory]
            histories.append(
                {
                    "model": model,
                    "rollout_id": trajectory[0]["rollout_id"],
                    "pattern": "".join(str(int(f)) for f in flags),
                    "wrong_then_correct": has_prefix_reversal(flags),
                    "positive_wrong_then_correct": has_prefix_reversal(flags[1:]),
                }
            )
        for i in range(1, len(LENGTHS)):
            bins = Counter((t[i - 1]["em"], t[i]["em"]) for t in trajectories)
            transitions.append(
                {
                    "model": model,
                    "from_T": LENGTHS[i - 1],
                    "to_T": LENGTHS[i],
                    "n": MAIN_TRAJECTORIES,
                    "both_correct": bins[(True, True)],
                    "gain": bins[(False, True)],
                    "loss": bins[(True, False)],
                    "both_wrong": bins[(False, False)],
                }
            )
    return {
        "histories": histories,
        "transitions": transitions,
        "summary": [
            {
                "model": m,
                "n": MAIN_TRAJECTORIES,
                "wrong_then_correct": sum(
                    h["wrong_then_correct"] for h in histories if h["model"] == m
                ),
                "positive_wrong_then_correct": sum(
                    h["positive_wrong_then_correct"] for h in histories if h["model"] == m
                ),
            }
            for m in MODELS
        ],
    }


def read_jsonl(path: Path, sources: dict) -> list[dict]:
    sources[str(path.relative_to(ROOT))] = sha(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def load_sources() -> tuple[list[dict], dict, dict, dict]:
    snapshot = json.loads(SNAPSHOT.read_text())
    sources = {str(SNAPSHOT.relative_to(ROOT)): sha(SNAPSHOT)}
    rows, jobs = [], {}
    for campaign in snapshot["source_campaigns"]:
        for name, expected in campaign["files"].items():
            path = ROOT / name
            assert sha(path) == expected, f"Frozen source changed: {name}"
            sources[name] = expected
            if name.endswith("results.jsonl"):
                rows.extend(read_jsonl(path, sources))
            elif name.endswith("manifest.json"):
                for job in json.loads(path.read_text())["jobs"]:
                    jobs[(job["model"], job["dataset"], job["instance_id"])] = job
    assert len(rows) == PER_MODEL_INSTANCES * len(MODELS)
    assert len({(r["model"], r["dataset"], r["instance_id"]) for r in rows}) == len(rows)
    assert Counter(r["model"] for r in rows) == Counter(
        dict.fromkeys(MODELS, PER_MODEL_INSTANCES)
    )
    conditions = {item["id"]: item for item in snapshot["model_conditions"]}
    for row in rows:
        job = jobs[(row["model"], row["dataset"], row["instance_id"])]
        for field in ("prompt_sha256", "request", "condition_id"):
            assert row[field] == job[field]
        assert row["request"]["parameters"] == request_parameters(conditions[row["model"]])
        assert row["prompt_version"] == "spatial-worlds-prompt/0.4"
    return rows, jobs, sources, snapshot


def reference_stats(unique: list[dict], sources: dict) -> dict:
    diagnostics = {}
    for dataset in ("main", "rq2"):
        diagnostics.update(
            {
                (dataset, r["rollout_id"]): r
                for r in read_jsonl(ROOT / "data" / dataset / "diagnostics.jsonl", sources)
            }
        )
    retention = []
    for key, rs in group(
        [r for r in unique if r["dataset"] == "main" and r["T"] >= 8], ("family", "T")
    ):
        eligible = erased = 0
        suffix = Counter()
        for row in rs:
            diag = diagnostics[(row["dataset"], row["rollout_id"])]
            erasure = diag["erasure_by_T"][str(row["T"])]
            eligible += erasure["eligible"]
            erased += erasure["erased"]
            for k in (8, 16, 32):
                suffix[k] += diag["suffix_by_T"][str(row["T"])][str(k)]["em"]
        retention.append(
            {
                **key,
                "n": len(rs),
                "eligible": eligible,
                "erased": erased,
                "erasure_rate": erased / eligible if eligible else None,
                **{f"suffix{k}": suffix[k] for k in (8, 16, 32)},
            }
        )
    realized_fields = (
        "mean_core_scan",
        "mean_core_effect",
        "initial_density",
        "final_density",
        "boundary_fraction",
        "n_core",
        "n_set",
    )
    realized = [
        {**key, "n": len(rs), **{field: mean([r[field] for r in rs]) for field in realized_fields}}
        for key, rs in group([r for r in unique if r["T"] == 64], ("family", "dimension", "w"))
    ]
    selected = {
        r["rollout_id"]
        for r in unique
        if r["dataset"] == "main" and r["family"] == "component_relabel" and r["T"] == 64
    }
    traces = read_jsonl(ROOT / "data/main/test/component_relabel/trace.jsonl", sources)
    merge_rows = []
    for trace in traces:
        if trace["rollout_id"] not in selected:
            continue
        steps = [s for s in trace["steps"] if s["t"] <= 64 and s["op"] != "SET"]
        merge_rows.append(
            {
                "rollout_id": trace["rollout_id"],
                "core_n": len(steps),
                "merge_contact_events": sum(
                    s["extras"].get("merge_contacts", 0) > 0 for s in steps
                ),
            }
        )
    assert len(merge_rows) == 24
    assert sum(r["core_n"] for r in merge_rows) == 1248
    return {"retention": retention, "realized": realized, "relabel_merge": merge_rows}


def failure_cases(rows: list[dict], jobs: dict, sources: dict) -> list[dict]:
    from spatial_worlds.eval.extract import extract_final_answer
    from spatial_worlds.eval.study_final_store import digest
    from spatial_worlds.schema import GoldRecord, Instance

    candidates = [
        r
        for r in rows
        if r["dataset"] == "main" and r["dimension"] == 2 and r["valid_state"] and not r["em"]
    ]
    rules = [
        (
            "near_miss",
            "Sol, d=2, T=16, 1--5 wrong cells; first instance ID",
            lambda r: r["model"] == "gpt-5.6-sol" and r["T"] == 16 and r["wrong_cells"] <= 5,
        ),
        (
            "preservation",
            "Mini, d=2, T=1, unchanged-only error; first instance ID",
            lambda r: (
                r["model"] == "gpt-5.4-mini" and r["T"] == 1 and r["changed_cell_accuracy"] == 1
            ),
        ),
        (
            "broad_error",
            "Mini, d=2, T=64, at least half the cells wrong; first instance ID",
            lambda r: (
                r["model"] == "gpt-5.4-mini"
                and r["T"] == 64
                and r["wrong_cells"] >= r["total_cells"] / 2
            ),
        ),
    ]
    cases = []
    with sqlite3.connect(f"file:{LEDGER}?mode=ro", uri=True) as database:
        for name, rule, predicate in rules:
            row = min((r for r in candidates if predicate(r)), key=lambda r: r["instance_id"])
            base = ROOT / "data/main/test" / row["family"]
            instance = next(
                r
                for r in read_jsonl(base / "instances.jsonl", sources)
                if r["id"] == row["instance_id"]
            )
            gold = next(
                r
                for r in read_jsonl(base / "gold.jsonl", sources)
                if r["instance_id"] == row["instance_id"]
            )
            job = jobs[(row["model"], row["dataset"], row["instance_id"])]
            assert digest(Instance.from_dict(instance).to_dict()) == job["instance_sha256"]
            assert digest(GoldRecord.from_dict(gold).to_dict()) == job["gold_sha256"]
            found = []
            for call_id in row["provider_call_ids"]:
                result = json.loads(
                    database.execute(
                        "SELECT result FROM attempts WHERE id=?", (call_id,)
                    ).fetchone()[0]
                )
                if result.get("response") == row["response"]:
                    found.append(result)
            assert len(found) == 1
            answer = found[0]["text"]
            prediction = json.loads(extract_final_answer(answer).text)
            reference = {tuple(p): v for p, v in gold["final_state"]["cells"]}
            predicted = {tuple(p): v for p, v in prediction["cells"]}
            initial = {tuple(p): v for p, v in instance["initial"]["cells"]}
            errors = []
            for p in itertools.product(*(range(n) for n in instance["world"]["shape"])):
                if reference.get(p, 0) != predicted.get(p, 0):
                    errors.append(
                        {
                            "coordinate": list(p),
                            "initial": initial.get(p, 0),
                            "gold": reference.get(p, 0),
                            "prediction": predicted.get(p, 0),
                            "reference_changed": initial.get(p, 0) != reference.get(p, 0),
                        }
                    )
            assert len(errors) == row["wrong_cells"]
            cases.append(
                {
                    "name": name,
                    "selection_rule": rule,
                    "model": row["model"],
                    "instance_id": row["instance_id"],
                    "T": row["T"],
                    "shape": instance["world"]["shape"],
                    "family": row["family"],
                    "wrong_cells": row["wrong_cells"],
                    "initial": instance["initial"],
                    "events": instance["events"],
                    "gold": gold["final_state"],
                    "prediction": prediction,
                    "error_coordinates": errors,
                    "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
                }
            )
    return cases


def reconstruct() -> dict:
    rows, jobs, sources, snapshot = load_sources()
    main = [r for r in rows if r["dataset"] == "main"]
    unique = [r for r in rows if r["model"] == "gpt-5.6-sol"]
    assert len(unique) == PER_MODEL_INSTANCES
    result = {
        "version": "revision-supplement/3.0",
        "scored_outcomes": len(rows),
        "model_order": list(MODELS),
        "model_conditions": snapshot["model_conditions"],
        "unique_instances": PER_MODEL_INSTANCES,
        "unique_trajectories": ALL_TRAJECTORIES,
        "percentile": "linear interpolation at (n-1)*q; P90 q=0.9; no independent-cell inference",
        "task_summary": [
            {**key, **counts(rs)}
            for key, rs in group([r for r in main if r["T"] >= 8], ("family", "model"))
        ],
        "pooled_summary": [
            {**key, **counts(rs)} for key, rs in group([r for r in main if r["T"] >= 8], ("model",))
        ],
        "task_dimension_length": [
            {**key, **counts(rs)} for key, rs in group(main, ("family", "model", "dimension", "T"))
        ],
        "errors": [{**key, **error_stats(rs)} for key, rs in group(main, ("model", "T"))],
        "task_errors_T64": [
            {**key, **error_stats(rs)}
            for key, rs in group([r for r in main if r["T"] == 64], ("family", "model"))
        ],
        "matched_valid_baselines": [
            {**key, **baseline_stats(rs)} for key, rs in group(main, ("model", "T"))
        ],
        "prefix": prefix_stats(main),
        "matched_pairs": paired_point_diagnostics(
            main, snapshot["supporting_diagnostics"]["matched_pairs"]
        ),
        "paired_main_comparisons": snapshot["paired_main_comparisons"],
        "reference": reference_stats(unique, sources),
        "format_resources": [],
        "failure_cases": failure_cases(rows, jobs, sources),
    }
    for key, rs in group(main, ("model", "T")):
        tokens = [r["usage"]["output_tokens"] for r in rs]
        result["format_resources"].append(
            {
                **key,
                **counts(rs),
                "valid_json": sum(r["valid_json"] for r in rs),
                "canonical": sum(r["canonical_format"] for r in rs),
                "extracted_correct": sum(r["em"] and not r["strict_em"] for r in rs),
                "extraction_methods_correct": dict(
                    Counter(r["extraction_method"] for r in rs if r["em"] and not r["strict_em"])
                ),
                "output_tokens_total": sum(tokens),
                "output_tokens_median": percentile(tokens, 0.5),
                "output_tokens_p90": percentile(tokens, 0.9),
                "length_empty": sum(
                    r["visible_text_empty"] and r["termination"] == "length" for r in rs
                ),
                "stop_empty": sum(
                    r["visible_text_empty"] and r["termination"] == "stop" for r in rs
                ),
                "raw_finish_reasons": dict(sorted(Counter(r["finish_reason"] for r in rs).items())),
            }
        )
    assert all(r["n"] == 96 for r in result["task_summary"])
    assert all(r["n"] == 8 for r in result["task_dimension_length"])
    assert len(result["task_dimension_length"]) == len(MODELS) * 8 * 3 * len(LENGTHS)
    assert {
        r["model"]: r["wrong_then_correct"] for r in result["prefix"]["summary"]
    } == PREFIX_REVERSALS
    assert {
        r["model"]: r["positive_wrong_then_correct"] for r in result["prefix"]["summary"]
    } == POSITIVE_PREFIX_REVERSALS
    assert all(sha(ROOT / name) == expected for name, expected in sources.items())
    result["source_sha256"] = sources
    result["generator_sha256"] = sha(Path(__file__))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    result = reconstruct()
    payload = encoded(result)
    if args.check_only:
        assert OUT.read_text() == payload, "Supplement is stale or changed."
    else:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(payload)
    print(
        f"Verified {result['scored_outcomes']:,} frozen outcomes; "
        f"{len(result['task_dimension_length'])} task-dimension-length cells; "
        f"{len(result['failure_cases'])} checked cases."
    )


if __name__ == "__main__":
    main()
