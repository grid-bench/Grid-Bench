"""Build the eight-trajectory, six-model analysis without changing source campaigns.

The original four-trajectory snapshot remains frozen. This script accepts only
complete original and expansion campaigns, rescoring all outcomes from the
shared ledger before combining their disjoint masks.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from paper.analyze_supplement import baseline_stats, error_stats, has_prefix_reversal
from scripts.analyze_completed import (
    digest_file,
    group_stats,
    read_and_rescore,
    supporting_diagnostics,
    verify_alignment,
    verify_gold_with_oracles,
)
from spatial_worlds.eval.expanded_analysis_registry import (
    ALL_TRAJECTORIES,
    ANALYSIS_DIR,
    ANALYSIS_ID,
    LENGTHS,
    MAIN_TRAJECTORIES,
    MODELS,
    PER_MODEL_INSTANCES,
    TASKS,
    WORKLOAD_INSTANCES,
)
from spatial_worlds.eval.study_report import paired_comparisons, summarize_group

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ANALYSIS_DIR
SOURCE_PAIRS = (
    ("formal.json", "formal_expand08.json"),
    ("formal_sonnet.json", "formal_sonnet_expand08.json"),
    ("formal_aws.json", "formal_aws_expand08.json"),
    ("formal_minimax.json", "formal_minimax_expand08.json"),
)


def merged_model_conditions(sources: list[tuple[dict, dict, list[dict], dict]]) -> list[dict]:
    """Describe one unchanged request condition backed by two source campaigns."""
    conditions = []
    for model in MODELS:
        matches = [
            (manifest, descriptor)
            for manifest, _, _, _ in sources
            for descriptor in manifest["models"]
            if descriptor["id"] == model
        ]
        if len(matches) != 2:
            raise ValueError(f"expected original and expansion condition for {model}")
        descriptors = [descriptor for _, descriptor in matches]
        if descriptors[0] != descriptors[1]:
            raise ValueError(f"model descriptor changed across cohorts: {model}")
        requests = [
            next(job["request"] for job in manifest["jobs"] if job["model"] == model)
            for manifest, _ in matches
        ]
        if requests[0] != requests[1]:
            raise ValueError(f"request condition changed across cohorts: {model}")
        conditions.append(
            {
                **descriptors[0],
                "campaign_ids": [manifest["campaign_id"] for manifest, _ in matches],
                "collection_protocols": sorted(
                    {manifest["collection_protocol"] for manifest, _ in matches}
                ),
                "request": requests[0],
            }
        )
    return conditions


def _key(row: dict) -> tuple[str, str, str]:
    return row["model"], row["dataset"], row["instance_id"]


def load_complete_sources() -> tuple[
    list[dict], set[str], list[tuple[dict, dict, list[dict], dict]], list[dict]
]:
    originals = [read_and_rescore(old) for old, _ in SOURCE_PAIRS]
    expansions = [read_and_rescore(new) for _, new in SOURCE_PAIRS]
    verify_alignment([(manifest, rows) for manifest, _, rows, _ in originals])
    verify_alignment([(manifest, rows) for manifest, _, rows, _ in expansions])

    merged = []
    for original, expansion in zip(originals, expansions, strict=True):
        old_manifest, _, old_rows, _ = original
        new_manifest, _, new_rows, _ = expansion
        if (
            new_manifest.get("expansion_of") != old_manifest["campaign_id"]
            or set(map(_key, old_rows)) & set(map(_key, new_rows))
        ):
            raise ValueError("expansion is not disjoint from its frozen source")
        combined_manifest = {
            "models": old_manifest["models"],
            "jobs": [*old_manifest["jobs"], *new_manifest["jobs"]],
            "workload_reuse_ids": [
                *old_manifest["workload_reuse_ids"],
                *new_manifest["workload_reuse_ids"],
            ],
        }
        merged.append((combined_manifest, [*old_rows, *new_rows]))
    verify_alignment(merged, expected=PER_MODEL_INSTANCES)
    rows = [row for _, part in merged for row in part]
    reuse = {identity for manifest, _ in merged for identity in manifest["workload_reuse_ids"]}
    if (
        len(rows) != PER_MODEL_INSTANCES * len(MODELS)
        or len({_key(row) for row in rows}) != len(rows)
        or len({(row["dataset"], row["instance_id"]) for row in rows})
        != PER_MODEL_INSTANCES
        or len(reuse) != MAIN_TRAJECTORIES
    ):
        raise ValueError("eight-trajectory combined mask has incorrect cardinality")
    return rows, reuse, [*originals, *expansions], [
        verify_gold_with_oracles(originals[0][0], originals[0][1]),
        verify_gold_with_oracles(expansions[0][0], expansions[0][1]),
    ]


def _group(rows: list[dict], fields: tuple[str, ...]) -> list[tuple[dict, list[dict]]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(row)
    return [
        (dict(zip(fields, key, strict=True)), groups[key]) for key in sorted(groups)
    ]


def prefix_diagnostics(main: list[dict]) -> dict:
    histories = []
    transitions = []
    for model in MODELS:
        trajectories = [
            sorted(group, key=lambda row: row["T"])
            for _, group in _group(
                [row for row in main if row["model"] == model], ("rollout_id",)
            )
        ]
        if len(trajectories) != MAIN_TRAJECTORIES or any(
            tuple(row["T"] for row in trajectory) != LENGTHS for trajectory in trajectories
        ):
            raise ValueError("main prefix grid is incomplete")
        for trajectory in trajectories:
            flags = [row["em"] for row in trajectory]
            histories.append(
                {
                    "model": model,
                    "rollout_id": trajectory[0]["rollout_id"],
                    "pattern": "".join(str(int(flag)) for flag in flags),
                    "wrong_then_correct": has_prefix_reversal(flags),
                    "positive_wrong_then_correct": has_prefix_reversal(flags[1:]),
                }
            )
        for i in range(1, len(LENGTHS)):
            counts = Counter((t[i - 1]["em"], t[i]["em"]) for t in trajectories)
            transitions.append(
                {
                    "model": model,
                    "from_T": LENGTHS[i - 1],
                    "to_T": LENGTHS[i],
                    "n": MAIN_TRAJECTORIES,
                    "both_correct": counts[(True, True)],
                    "gain": counts[(False, True)],
                    "loss": counts[(True, False)],
                    "both_wrong": counts[(False, False)],
                }
            )
    summary = [
        {
            "model": model,
            "n": MAIN_TRAJECTORIES,
            "wrong_then_correct": sum(
                row["wrong_then_correct"] for row in histories if row["model"] == model
            ),
            "positive_wrong_then_correct": sum(
                row["positive_wrong_then_correct"]
                for row in histories
                if row["model"] == model
            ),
        }
        for model in MODELS
    ]
    return {"histories": histories, "transitions": transitions, "summary": summary}


def tables_for_expanded(rows: list[dict], reuse: set[str]) -> dict[str, list[dict]]:
    tables: dict[str, list[dict]] = {
        name: []
        for name in (
            "overview",
            "length",
            "task_length",
            "dimension_length",
            "workload",
            "resources",
        )
    }
    for model in MODELS:
        all_rows = [row for row in rows if row["model"] == model]
        main = [row for row in all_rows if row["dataset"] == "main"]
        for view, subset in (
            ("main", main),
            ("main_T8_64", [row for row in main if row["T"] >= 8]),
        ):
            tables["overview"].append({"model": model, "view": view, **group_stats(subset)})
        tables["resources"].append(
            {"model": model, "view": "unique_1488", **group_stats(all_rows)}
        )
        for t in LENGTHS:
            at_t = [row for row in main if row["T"] == t]
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
                        **group_stats([row for row in at_t if row["family"] == family]),
                    }
                )
            for dimension in (2, 3, 4):
                tables["dimension_length"].append(
                    {
                        "model": model,
                        "dimension": dimension,
                        "T": t,
                        **group_stats([row for row in at_t if row["dimension"] == dimension]),
                    }
                )
        workload = [
            row for row in all_rows if row["dataset"] == "rq2" or row["instance_id"] in reuse
        ]
        if len(workload) != WORKLOAD_INSTANCES or len(
            {row["instance_id"] for row in workload}
        ) != WORKLOAD_INSTANCES:
            raise ValueError("workload view must reuse 192 main rows without duplicates")
        for family, w, t in sorted({(row["family"], row["w"], row["T"]) for row in workload}):
            tables["workload"].append(
                {
                    "model": model,
                    "family": family,
                    "w": w,
                    "T": t,
                    **group_stats(
                        [
                            row
                            for row in workload
                            if (row["family"], row["w"], row["T"]) == (family, w, t)
                        ]
                    ),
                }
            )
    return tables


def build_report() -> dict:
    rows, reuse, sources, oracle_checks = load_complete_sources()
    main = [row for row in rows if row["dataset"] == "main"]
    workload = [
        row for row in rows if row["dataset"] == "rq2" or row["instance_id"] in reuse
    ]
    if (
        len(main) != MAIN_TRAJECTORIES * len(LENGTHS) * len(MODELS)
        or len(workload) != WORKLOAD_INSTANCES * len(MODELS)
        or len({row["rollout_id"] for row in rows}) != ALL_TRAJECTORIES
    ):
        raise ValueError("main or workload denominator is incomplete")
    tables = tables_for_expanded(rows, reuse)
    if any(row["n"] != MAIN_TRAJECTORIES for row in tables["length"]):
        raise ValueError("main length denominator changed")
    if any(row["n"] != 8 for row in tables["workload"]):
        raise ValueError("workload level denominator changed")
    return {
        "analysis": ANALYSIS_ID,
        "model_order": list(MODELS),
        "model_conditions": merged_model_conditions(sources),
        "scored_outcomes": len(rows),
        "unique_instances": PER_MODEL_INSTANCES,
        "unique_trajectories": ALL_TRAJECTORIES,
        "source_campaigns": [provenance for *_, provenance in sources],
        "independent_gold_checks": oracle_checks,
        "script_sha256": digest_file(Path(__file__)),
        "analysis_helpers_sha256": digest_file(ROOT / "scripts/analyze_completed.py"),
        "registry_sha256": digest_file(
            ROOT / "src/spatial_worlds/eval/expanded_analysis_registry.py"
        ),
        "historical_analysis": {
            "snapshot": "data/analysis/six-model-20260920/analysis.json",
            "snapshot_sha256": digest_file(
                ROOT / "data/analysis/six-model-20260920/analysis.json"
            ),
            "original_analysis_unchanged": True,
        },
        "tables": tables,
        "paired_main_comparisons": {
            str(t): paired_comparisons([row for row in main if row["T"] == t])
            for t in (32, 64)
        },
        "supporting_diagnostics": supporting_diagnostics(rows),
        "errors": [
            {**key, **error_stats(group)}
            for key, group in _group(main, ("model", "T"))
        ],
        "matched_valid_baselines": [
            {**key, **baseline_stats(group)}
            for key, group in _group(main, ("model", "T"))
        ],
        "prefix": prefix_diagnostics(main),
        "notes": [
            "All six lengths are independently requested and scored; no prefix imputation.",
            "Original four-trajectory campaigns and analysis remain immutable.",
            "Workload reuses 192 main instances per model; no duplicate provider calls.",
            "Conditional diagnostics are not substitutes for complete-mask exact match.",
        ],
    }


def main() -> None:
    report = build_report()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "analysis.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    for name, table in report["tables"].items():
        with (OUT / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    print(
        json.dumps(
            {
                "output": str(OUT.relative_to(ROOT)),
                "scored_outcomes": report["scored_outcomes"],
                "unique_instances": report["unique_instances"],
                "paid_calls": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
