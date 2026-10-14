"""Reproduce the completed six-model analysis offline; never alter source campaigns."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from statistics import median

from spatial_worlds.eval.analysis_registry import (
    ANALYSIS_DIR,
    ANALYSIS_ID,
    CAMPAIGN_CONFIGS,
    COLORS,
    LENGTHS,
    MODELS,
    NAMES,
    TASKS,
)
from spatial_worlds.eval.protocol import FORMAL128_PROTOCOL, AttemptResult
from spatial_worlds.eval.study_final import is_model_outcome
from spatial_worlds.eval.study_final_config import (
    build_manifest,
    formal_scopes,
    load_config,
    request_parameters,
)
from spatial_worlds.eval.study_final_store import FinalStore
from spatial_worlds.eval.study_report import materialize_rows, paired_comparisons, summarize_group

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ANALYSIS_DIR
FIGURES = ROOT / "docs/figures/six-model"
LABELS = NAMES
SCORE_FIELDS = (
    "em",
    "strict_em",
    "valid_json",
    "valid_state",
    "canonical_format",
    "wrong_cells",
    "cell_accuracy",
    "changed_cell_accuracy",
    "unchanged_cell_accuracy",
    "answer_status",
    "finish_reason",
    "termination",
    "usage",
    "response",
    "analysis_eligible",
    "score_error",
    "extraction_applied",
    "extraction_method",
)


def digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row_key(row: dict) -> tuple:
    return row["model"], row["dataset"], row["instance_id"]


def verify_alignment(campaigns: list[tuple[dict, list[dict]]], expected: int = 744) -> None:
    """Reject incomplete/mixed conditions and compare model-independent frozen job hashes."""
    if not campaigns or any(not rows for _, rows in campaigns):
        raise ValueError("empty campaign input")
    reference = None
    reuse = None
    models_seen: set[str] = set()
    for manifest, rows in campaigns:
        model_descriptors = {model["id"]: model for model in manifest["models"]}
        jobs = {row_key(job): job for job in manifest["jobs"]}
        if len(jobs) != len(manifest["jobs"]) or len({row_key(r) for r in rows}) != len(rows):
            raise ValueError("duplicate frozen job or scored row")
        if set(jobs) != {row_key(r) for r in rows}:
            raise ValueError("scored rows differ from the declared mask")
        current_reuse = set(manifest["workload_reuse_ids"])
        if reuse is not None and current_reuse != reuse:
            raise ValueError("workload reuse masks differ")
        reuse = current_reuse
        for model in sorted({r["model"] for r in rows}):
            if model in models_seen:
                raise ValueError("model occurs in multiple campaigns")
            models_seen.add(model)
            descriptor = model_descriptors[model]
            subset = [r for r in rows if r["model"] == model]
            if len(subset) != expected:
                raise ValueError("incomplete model mask")
            mask = {}
            for row in subset:
                job = jobs[row_key(row)]
                if not isinstance(row.get("em"), bool):
                    raise ValueError("unscored outcome")
                for field in ("prompt_sha256", "condition_id", "request"):
                    if row[field] != job[field]:
                        raise ValueError(f"scored-row {field} differs from frozen job")
                if (
                    row["request"]["parameters"] != request_parameters(descriptor)
                    or row["prompt_version"] != "spatial-worlds-prompt/0.4"
                    or row["request"].get("expected_model")
                    != descriptor["accepted_response_models"]
                ):
                    raise ValueError("unexpected model-visible condition")
                request = {
                    k: v
                    for k, v in job["request"].items()
                    if k not in {"expected_model", "base_url", "parameters"}
                }
                if request != {
                    "timeout_s": 7200,
                    "max_attempts": 1,
                    "usage_grace_s": 5.0,
                    "n": 1,
                    "evaluation_protocol": "spatial-worlds-evaluation/1.6",
                    "scoring_version": "spatial-worlds-scoring/1.0",
                }:
                    raise ValueError("unexpected shared evaluation setting")
                request["generation_ceiling"] = 128000
                mask[(job["dataset"], job["instance_id"])] = (
                    job["prompt_sha256"],
                    job["instance_sha256"],
                    job["gold_sha256"],
                    json.dumps(request, sort_keys=True),
                )
            if reference is not None and reference != mask:
                raise ValueError("models do not share identical instances, gold, prompts, settings")
            reference = mask


def read_and_rescore(config_name: str) -> tuple[dict, dict, list[dict], dict]:
    config = load_config(ROOT / "configs" / config_name)
    manifest, datasets = build_manifest(ROOT, config)
    source = ROOT / "data/study-v1" / manifest["campaign_id"]
    saved_manifest = json.loads((source / "manifest.json").read_text())
    if manifest != saved_manifest:
        raise ValueError("rebuilt manifest differs from frozen campaign")
    rows = [json.loads(line) for line in (source / "results.jsonl").read_text().splitlines()]
    with FinalStore(
        ROOT / "data/study-v1/attempts.sqlite", manifest["campaign_id"], readonly=True
    ) as store:
        store.db.execute("BEGIN")
        replayed = materialize_rows(
            manifest,
            datasets,
            store,
            protocol=FORMAL128_PROTOCOL,
            prompt_version=manifest["prompt_version"],
            attempt_scopes=list(formal_scopes(manifest)),
            terminal_predicate=lambda r, j: is_model_outcome(AttemptResult(**r), j),
            finalized_only=True,
        )
        raw_empty = {}
        for scope in formal_scopes(manifest):
            for attempt in store.all_attempts(scope):
                if attempt["result"]:
                    result = AttemptResult(**json.loads(attempt["result"]))
                    job = json.loads(attempt["job"])
                    if is_model_outcome(result, job):
                        raw_empty[row_key(job)] = not result.text.strip()
    lookup = {row_key(row): row for row in replayed}
    for row in rows:
        if row["visible_text_empty"] != raw_empty[row_key(row)]:
            raise ValueError(f"raw empty-answer flag mismatch: {row_key(row)}")
        for field in SCORE_FIELDS:
            if row[field] != lookup[row_key(row)][field]:
                raise ValueError(f"re-score mismatch: {row_key(row)} {field}")
    provenance = {
        "campaign_id": manifest["campaign_id"],
        "rescored_outcomes": len(rows),
        "files": {
            str((source / name).relative_to(ROOT)): digest_file(source / name)
            for name in ("manifest.json", "results.jsonl", "summary.json")
        },
    }
    return manifest, datasets, rows, provenance


def verify_gold_with_oracles(manifest: dict, datasets: dict) -> dict:
    """Check every selected gold via the separate dictionary-based test implementations."""
    path = ROOT / "tests/oracles.py"
    spec = importlib.util.spec_from_file_location("completed_analysis_oracles", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("independent test oracles unavailable")
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    instances = {name: {item.id: item for item in ds.instances} for name, ds in datasets.items()}
    selected = {(job["dataset"], job["instance_id"]) for job in manifest["jobs"]}
    events = 0
    for name, identity in sorted(selected):
        instance = instances[name][identity]
        state = oracle.to_dense(instance.initial)
        for event in instance.events:
            state, _, _ = oracle.apply_oracle(instance.family, instance.world, state, event)
            events += 1
        if oracle.from_dense(instance.world, state) != datasets[name].gold[identity].final_state:
            raise ValueError(f"independent oracle/gold mismatch: {identity}")
    return {
        "selected_states": len(selected),
        "replayed_events": events,
        "mismatches": 0,
        "oracle_sha256": digest_file(path),
        "scope": "separate rule implementations; shares schema and task interpretation",
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


def tables_for(rows: list[dict], reuse: set[str]) -> dict[str, list[dict]]:
    tables: dict[str, list[dict]] = {
        key: []
        for key in (
            "overview",
            "length",
            "task_length",
            "dimension_length",
            "workload",
            "resources",
        )
    }
    for model in MODELS:
        all_rows = [r for r in rows if r["model"] == model]
        main = [r for r in all_rows if r["dataset"] == "main"]
        for view, subset in (("main", main), ("main_T8_64", [r for r in main if r["T"] >= 8])):
            tables["overview"].append({"model": model, "view": view, **group_stats(subset)})
        tables["resources"].append({"model": model, "view": "unique_744", **group_stats(all_rows)})
        for t in LENGTHS:
            at_t = [r for r in main if r["T"] == t]
            summary = summarize_group(at_t)
            ci = summary["ci95"]
            tables["length"].append(
                {
                    "model": model,
                    "T": t,
                    **group_stats(at_t),
                    "ci95_low": ci[0] if ci else None,
                    "ci95_high": ci[1] if ci else None,
                    "ci_warning": summary.get("warning"),
                }
            )
            for family in TASKS:
                tables["task_length"].append(
                    {
                        "model": model,
                        "family": family,
                        "T": t,
                        **group_stats([r for r in at_t if r["family"] == family]),
                    }
                )
            for dimension in (2, 3, 4):
                tables["dimension_length"].append(
                    {
                        "model": model,
                        "dimension": dimension,
                        "T": t,
                        **group_stats([r for r in at_t if r["dimension"] == dimension]),
                    }
                )
        workload = [r for r in all_rows if r["dataset"] == "rq2" or r["instance_id"] in reuse]
        if len(workload) != 264 or len({r["instance_id"] for r in workload}) != 264:
            raise ValueError("workload view must reuse 96 main rows without duplicates")
        for family, w, t in sorted({(r["family"], r["w"], r["T"]) for r in workload}):
            tables["workload"].append(
                {
                    "model": model,
                    "family": family,
                    "w": w,
                    "T": t,
                    **group_stats(
                        [r for r in workload if (r["family"], r["w"], r["T"]) == (family, w, t)]
                    ),
                }
            )
    return tables


def make_figures(tables: dict[str, list[dict]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none", "svg.hashsalt": "spatial-worlds"})
    FIGURES.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        fig.savefig(FIGURES / f"{name}.svg", bbox_inches="tight", metadata={"Date": None})
        fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight", dpi=160)
        plt.close(fig)

    def decorate(ax):
        ax.set_xticks(range(6), LENGTHS)
        ax.set_ylim(-2, 102)
        ax.set_xlabel("Tested history length T (categorical)")
        ax.set_ylabel("Full-world exact match (%)")
        ax.grid(axis="y", alpha=0.2)

    fig, ax = plt.subplots(figsize=(7, 4))
    for model in MODELS:
        points = [r for r in tables["length"] if r["model"] == model]
        ax.plot(
            range(6),
            [100 * r["em"] for r in points],
            "o-",
            label=LABELS[model],
            color=COLORS[model],
        )
        ax.fill_between(
            range(6),
            [100 * r["ci95_low"] if r["ci95_low"] is not None else float("nan") for r in points],
            [100 * r["ci95_high"] if r["ci95_high"] is not None else float("nan") for r in points],
            color=COLORS[model],
            alpha=0.1,
        )
    decorate(ax)
    ax.legend()
    ax.set_title(
        "Main view: 96 outcomes per model and length\nMarginal trajectory-bootstrap 95% intervals"
    )
    fig.tight_layout()
    save(fig, "main-length")

    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5), sharey=True)
    for ax, (family, label) in zip(axes.flat, TASKS.items(), strict=True):
        for model in MODELS:
            points = [
                r for r in tables["task_length"] if r["model"] == model and r["family"] == family
            ]
            ax.plot(
                range(6),
                [100 * r["em"] for r in points],
                "o-",
                label=LABELS[model],
                color=COLORS[model],
                markersize=3,
            )
        decorate(ax)
        ax.set_xlabel("T (categorical)")
        ax.set_title(label + " (n=12 / point)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Task-dependent length profiles — descriptive means, not simultaneous bands")
    fig.tight_layout()
    save(fig, "task-length")

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    axes = axes.ravel()
    outcome_types = (
        ("correct", "Correct", "#2166ac"),
        ("valid_wrong", "Valid, wrong", "#92c5de"),
        ("length_empty", "Length-empty", "#d6604d"),
        ("other_empty", "Other empty", "#f4a582"),
        ("invalid_nonempty", "Invalid, nonempty", "#777777"),
    )
    for ax, model in zip(axes, MODELS, strict=True):
        points = [r for r in tables["length"] if r["model"] == model]
        bottoms = [0.0] * 6
        for key, label, color in outcome_types:
            heights = [100 * r[key] / r["n"] for r in points]
            ax.bar(range(6), heights, bottom=bottoms, label=label, color=color)
            bottoms = [a + b for a, b in zip(bottoms, heights, strict=True)]
        ax.set_xticks(range(6), LENGTHS)
        ax.set_xlabel("T (categorical)")
        ax.set_title(LABELS[model])
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("Share of all 96 outcomes (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8)
    fig.suptitle("Failure decomposition: fixed denominators, no discarded answers")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "failure-types")


def supporting_diagnostics(rows: list[dict]) -> dict:
    """Explicitly post-hoc diagnostics; never replace the complete-mask primary estimates."""
    main = [r for r in rows if r["dataset"] == "main"]
    partitions = []
    for model in MODELS:
        subset = [
            r
            for r in main
            if r["model"] == model and r["T"] == 1 and r["valid_state"] and not r["em"]
        ]
        counts = Counter()
        for row in subset:
            changed = row["changed_cell_accuracy"]
            unchanged = row["unchanged_cell_accuracy"]
            bad_changed = changed is not None and changed < 1
            bad_unchanged = unchanged is not None and unchanged < 1
            if not bad_changed and not bad_unchanged:
                raise ValueError("valid incorrect state has no changed/unchanged cell errors")
            counts[
                "both"
                if bad_changed and bad_unchanged
                else "changed_only"
                if bad_changed
                else "unchanged_only"
            ] += 1
        partitions.append(
            {
                "model": model,
                "n": len(subset),
                **{key: counts[key] for key in ("changed_only", "unchanged_only", "both")},
            }
        )
    t64 = {
        model: {r["instance_id"]: r for r in main if r["model"] == model and r["T"] == 64}
        for model in MODELS
    }
    common_valid = set.intersection(
        *[
            {key for key, r in t64[model].items() if r["valid_state"]}
            for model in ("gpt-5.6-sol", "claude-sonnet")
        ]
    )
    common_stats = [
        {"model": model, **group_stats([t64[model][key] for key in sorted(common_valid)])}
        for model in ("gpt-5.6-sol", "claude-sonnet")
    ]
    exhausted = {
        key
        for key, r in t64["claude-sonnet"].items()
        if r["visible_text_empty"] and r["termination"] == "length"
    }
    exhausted_stats = [
        {"model": model, **group_stats([t64[model][key] for key in sorted(exhausted)])}
        for model in MODELS
    ]
    return {
        "status": "post-hoc descriptive; conditioned masks are not primary scores",
        "one_step_error_partitions": partitions,
        "T64_Sol_Sonnet_jointly_valid": common_stats,
        "T64_Sonnet_length_empty_mask": exhausted_stats,
        "T64_without_RELABEL": [
            {
                "model": model,
                **group_stats(
                    [r for r in t64[model].values() if r["family"] != "component_relabel"]
                ),
            }
            for model in MODELS
        ],
        "matched_pairs": matched_pair_diagnostics(main),
    }


def matched_pair_diagnostics(rows: list[dict]) -> list[dict]:
    """All same-instance conditional comparisons; full-set EM stays separate."""
    result = []
    for t in LENGTHS:
        by_model = {
            model: {r["instance_id"]: r for r in rows if r["model"] == model and r["T"] == t}
            for model in MODELS
        }
        for a, b in combinations(MODELS, 2):
            left, right = by_model[a], by_model[b]
            if set(left) != set(right):
                raise ValueError("matched diagnostics require identical declared masks")
            valid = sorted(k for k in left if left[k]["valid_state"] and right[k]["valid_state"])
            for subset, keys in (
                ("joint_valid", valid),
                (
                    "joint_valid_incorrect",
                    [k for k in valid if not left[k]["em"] and not right[k]["em"]],
                ),
            ):
                part = {
                    "model_a": a,
                    "model_b": b,
                    "T": t,
                    "subset": subset,
                    "n_total": len(left),
                    "n": len(keys),
                    "instance_ids": keys,
                    "a_fewer_wrong": sum(
                        left[k]["wrong_cells"] < right[k]["wrong_cells"] for k in keys
                    ),
                    "equal_wrong": sum(
                        left[k]["wrong_cells"] == right[k]["wrong_cells"] for k in keys
                    ),
                    "b_fewer_wrong": sum(
                        left[k]["wrong_cells"] > right[k]["wrong_cells"] for k in keys
                    ),
                }
                for side, source in (("a", left), ("b", right)):
                    selected = [source[k] for k in keys]
                    part[f"correct_{side}"] = sum(r["em"] for r in selected)
                    part[f"cell_accuracy_{side}"] = (
                        sum(r["cell_accuracy"] for r in selected) / len(selected)
                        if selected
                        else None
                    )
                    part[f"wrong_cells_median_{side}"] = (
                        median(r["wrong_cells"] for r in selected) if selected else None
                    )
                result.append(part)
    return result


def main() -> None:
    loaded = [read_and_rescore(name) for name in CAMPAIGN_CONFIGS]
    verify_alignment([(m, rows) for m, _, rows, _ in loaded])
    oracle_check = verify_gold_with_oracles(loaded[0][0], loaded[0][1])
    rows = [row for _, _, part, _ in loaded for row in part]
    if set(r["model"] for r in rows) != set(MODELS):
        raise ValueError("unexpected model roster")
    tables = tables_for(rows, set(loaded[0][0]["workload_reuse_ids"]))
    historical_snapshot = ROOT / "data/analysis/three-model-20260917/analysis.json"
    historical = json.loads(historical_snapshot.read_text())
    historical_code = OUT / "historical-three-model-code/scripts/analyze_completed.py"
    if digest_file(historical_code) != historical["script_sha256"]:
        raise ValueError("archived three-model analysis code differs from its frozen hash")
    old_models = {r["model"] for r in historical["tables"]["length"]}
    for name, old_table in historical["tables"].items():
        if [r for r in tables[name] if r["model"] in old_models] != old_table:
            raise ValueError(f"historical three-model results changed in {name}")
    paired = {
        str(t): paired_comparisons([r for r in rows if r["dataset"] == "main" and r["T"] == t])
        for t in (32, 64)
    }
    report = {
        "analysis": ANALYSIS_ID,
        "model_order": list(MODELS),
        "model_conditions": [
            {
                **descriptor,
                "campaign_id": manifest["campaign_id"],
                "collection_protocol": manifest["collection_protocol"],
                "request": next(
                    job["request"] for job in manifest["jobs"] if job["model"] == model
                ),
            }
            for model in MODELS
            for manifest, _, _, _ in loaded
            for descriptor in manifest["models"]
            if descriptor["id"] == model
        ],
        "source_campaigns": [p for *_, p in loaded],
        "script_sha256": digest_file(Path(__file__)),
        "registry_sha256": digest_file(ROOT / "src/spatial_worlds/eval/analysis_registry.py"),
        "historical_analysis": {
            "snapshot": str(historical_snapshot.relative_to(ROOT)),
            "snapshot_sha256": digest_file(historical_snapshot),
            "archived_script": str(historical_code.relative_to(ROOT)),
            "script_sha256": digest_file(historical_code),
            "original_model_tables_unchanged": True,
        },
        "scored_outcomes": len(rows),
        "unique_instances": 744,
        "independent_gold_check": oracle_check,
        "tables": tables,
        "paired_main_comparisons": paired,
        "supporting_diagnostics": supporting_diagnostics(rows),
        "notes": [
            "Derived analysis, not a new campaign or scoring condition.",
            "No API calls. Source campaign files and ledger opened read-only.",
            "All outcomes retained; workload reuses 96 existing main rows per model.",
            "Cell accuracy is valid-answer conditional; masks differ across models.",
            "Intervals are marginal; four trajectories per cell limit precision.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "analysis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    for name, table in tables.items():
        with (OUT / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    make_figures(tables)
    for provenance in report["source_campaigns"]:
        for path, expected in provenance["files"].items():
            if digest_file(ROOT / path) != expected:
                raise ValueError("source artifact changed during analysis")
    print(
        json.dumps(
            {
                "output": str(OUT.relative_to(ROOT)),
                "figures": str(FIGURES.relative_to(ROOT)),
                "rescored_outcomes": len(rows),
                "oracle_check": oracle_check,
                "source_artifacts_unchanged": True,
                "paid_calls": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
