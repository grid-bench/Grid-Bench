"""Common-mask study reporting; no inference from last-write-wins legacy files."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any, Protocol

from ..prompt import PROMPT_VERSION, render_prompt
from .aggregate import score_formal_record
from .extract import SCORING_VERSION
from .protocol import ProtocolContext
from .run import Dataset, condition_id
from .stats import (
    mean,
    newcombe_difference,
    percentile,
    stratified_cluster_interval,
    wilson_interval,
)


class StudyReader(Protocol):
    def all_attempts(self, scope: str) -> list[dict[str, Any]]: ...

    def locked_result(self, attempt_id: int) -> dict[str, Any] | None: ...

    def paused(self, model: str) -> str | None: ...


def cluster_ci(rows: Sequence[Mapping[str, Any]], field: str = "em") -> dict:
    strata: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get(field) is None:
            return {"ci95": None, "warning": "incomplete_selection"}
        key = json.dumps([row["family"], row["dimension"], row["w"]])
        strata[key][row["rollout_id"]].append(float(row[field]))
    return stratified_cluster_interval(strata)


def _is_formal(manifest: Mapping[str, Any]) -> bool:
    phase = str(manifest["phase"])
    return phase == "formal" or phase.startswith("formal128:")


def summarize_group(rows: Sequence[Mapping[str, Any]], *, cell: bool = False) -> dict:
    scored = [row for row in rows if row.get("em") is not None]
    n, total = len(scored), len(rows)
    k = sum(bool(row["em"]) for row in scored)
    complete = n == total and total > 0
    wrong = sorted(
        float(row["wrong_cells"])
        for row in scored
        if row.get("valid_state")
        and not row["em"]
        and row.get("wrong_cells") is not None
        and row.get("answer_status") != "refusal"
    )
    valid = [
        row for row in scored if row.get("valid_state") and row.get("answer_status") != "refusal"
    ]
    extent_by_termination = {}
    for kind in ("stop", "length", "deadline"):
        values = sorted(
            float(row["wrong_cells"])
            for row in valid
            if not row["em"]
            and row.get("termination") == kind
            and row.get("wrong_cells") is not None
        )
        extent_by_termination[kind] = {
            "n": len(values),
            "median": percentile(values, 50) if values else None,
            "p90": percentile(values, 90) if values else None,
        }
    interval = (
        {"ci95": list(wilson_interval(k, n)), "method": "Wilson_95_marginal"}
        if complete and cell
        else cluster_ci(rows)
    )
    return {
        "expected": total,
        "scored": n,
        "correct": k,
        "complete": complete,
        "em": k / n if complete else None,
        "observed_em_incomplete_only": k / n if n and not complete else None,
        **interval,
        "answer_status_counts": dict(Counter(row["answer_status"] for row in rows)),
        "termination_counts": dict(
            Counter(str(row.get("termination") or "unqueried") for row in rows)
        ),
        "wrong_cells_population": "analysis_eligible_valid_incorrect_nonrefusal",
        "wrong_cells_by_termination": extent_by_termination,
        "wrong_cells_n": len(wrong),
        "wrong_cells_median": percentile(wrong, 50) if wrong else None,
        "wrong_cells_p90": percentile(wrong, 90) if wrong else None,
        "cell_accuracy_valid_only": mean([float(row["cell_accuracy"]) for row in valid]),
        "cell_accuracy_valid_n": len(valid),
        "copy_baseline_em": mean([float(row["copy_baseline_em"]) for row in rows]),
        "zero_baseline_em": mean([float(row["zero_baseline_em"]) for row in rows]),
        "copy_baseline_cell_accuracy": mean(
            [float(row["copy_baseline_cell_accuracy"]) for row in rows]
        ),
        "zero_baseline_cell_accuracy": mean(
            [float(row["zero_baseline_cell_accuracy"]) for row in rows]
        ),
        "changed_cell_accuracy_valid_only": mean(
            [
                float(row["changed_cell_accuracy"])
                for row in valid
                if row["changed_cell_accuracy"] is not None
            ]
        ),
        "unchanged_cell_accuracy_valid_only": mean(
            [
                float(row["unchanged_cell_accuracy"])
                for row in valid
                if row["unchanged_cell_accuracy"] is not None
            ]
        ),
    }


def paired_comparisons(rows: Sequence[Mapping[str, Any]]) -> list[dict]:
    by_model: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_model[row["model"]][row["instance_id"]] = row
    comparisons = []
    for a, b in combinations(sorted(by_model), 2):
        # Compare only models with the same DECLARED mask, not its observed intersection.
        if set(by_model[a]) != set(by_model[b]):
            comparisons.append(
                {
                    "reference": a,
                    "other": b,
                    "complete": False,
                    "reason": "different_declared_masks",
                }
            )
            continue
        left, right = by_model[a], by_model[b]
        if any(left[key]["em"] is None or right[key]["em"] is None for key in left):
            comparisons.append(
                {"reference": a, "other": b, "complete": False, "reason": "incomplete_selection"}
            )
            continue
        differences = [
            {**left[key], "difference": int(right[key]["em"]) - int(left[key]["em"])}
            for key in sorted(left)
        ]
        comparisons.append(
            {
                "reference": a,
                "other": b,
                "complete": True,
                "n": len(differences),
                "wins_other": sum(row["difference"] > 0 for row in differences),
                "losses_other": sum(row["difference"] < 0 for row in differences),
                "ties": sum(row["difference"] == 0 for row in differences),
                "difference_other_minus_reference": mean(
                    [float(row["difference"]) for row in differences]
                ),
                **cluster_ci(differences, "difference"),
            }
        )
    return comparisons


def independent_contrasts(rows: Sequence[Mapping[str, Any]], axis: str) -> list[dict]:
    keys = ["model", "family", "T", "dimension", "w"]
    keys.remove(axis)
    groups: dict[tuple, dict[Any, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[tuple(row[key] for key in keys)][row[axis]].append(row)
    results = []
    for key, variants in sorted(groups.items(), key=str):
        levels = sorted(variants)
        for other in levels[1:]:
            reference = levels[0]
            a, b = variants[reference], variants[other]
            complete = all(row["em"] is not None for row in [*a, *b])
            results.append(
                {
                    **dict(zip(keys, key, strict=True)),
                    "axis": axis,
                    "reference": reference,
                    "other": other,
                    "complete": complete,
                    "n_reference": len(a),
                    "n_other": len(b),
                    "difference": (
                        sum(row["em"] for row in b) / len(b) - sum(row["em"] for row in a) / len(a)
                    )
                    if complete
                    else None,
                    "ci95": newcombe_difference([row["em"] for row in a], [row["em"] for row in b])
                    if complete
                    else None,
                    "method": "Newcombe_Wilson_method_10_no_CC_independent",
                }
            )
    return results


def length_effects(rows: Sequence[Mapping[str, Any]], reference: int, other: int) -> list[dict]:
    """Paired within-trajectory differences; missing prefixes never disappear."""
    grouped: dict[tuple, dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["T"] in {reference, other}:
            grouped[(row["model"], row["rollout_id"])][row["T"]] = row
    effects = []
    for group in grouped.values():
        row = next(iter(group.values()))
        a, b = group.get(reference), group.get(other)
        value = None
        if a is not None and b is not None and a["em"] is not None and b["em"] is not None:
            value = int(b["em"]) - int(a["em"])
        effects.append({**row, "difference": value})
    return effects


def interaction_contrasts(rows: Sequence[Mapping[str, Any]], axis: str) -> list[dict]:
    """Direct difference-of-differences, with paired T and independent d/w clusters.

    No claim is inferred from 'significant here, not significant there'. These
    declared contrasts are descriptive discovery summaries, not a test grid.
    """
    results = []
    for short, long in ((1, 64), (32, 64)):
        effects = length_effects(rows, short, long)
        keys = ["model", "family", "dimension", "w"]
        keys.remove(axis)
        groups: dict[tuple, dict[Any, list[dict]]] = defaultdict(lambda: defaultdict(list))
        for row in effects:
            groups[tuple(row[key] for key in keys)][row[axis]].append(row)
        for key, variants in sorted(groups.items(), key=str):
            levels = sorted(variants)
            for level in levels[1:]:
                a, b = variants[levels[0]], variants[level]
                complete = all(row["difference"] is not None for row in [*a, *b])
                weighted = []
                if complete:
                    # The statistic's mean is mean(delta_other) - mean(delta_ref),
                    # even if the two independent groups differ in sample size.
                    weighted = [
                        {
                            **row,
                            "contrast": sign * (len(a) + len(b)) / len(group) * row["difference"],
                        }
                        for group, sign in ((a, -1), (b, 1))
                        for row in group
                    ]
                interval = (
                    cluster_ci(weighted, "contrast")
                    if complete
                    else {"ci95": None, "warning": "incomplete_selection"}
                )
                results.append(
                    {
                        **dict(zip(keys, key, strict=True)),
                        "axis": axis,
                        "reference": levels[0],
                        "other": level,
                        "T_reference": short,
                        "T_other": long,
                        "complete": complete,
                        "difference_of_differences": mean([row["contrast"] for row in weighted]),
                        **interval,
                    }
                )
    return results


def materialize_rows(
    manifest: Mapping[str, Any],
    datasets: Mapping[str, Dataset],
    store: StudyReader | None,
    *,
    protocol: ProtocolContext,
    prompt_version: str = PROMPT_VERSION,
    attempt_scopes: Sequence[str] | None = None,
    terminal_predicate: Callable[[Mapping[str, Any], Mapping[str, Any]], bool] | None = None,
    finalized_only: bool = False,
) -> list[dict]:
    scopes = list(attempt_scopes) if attempt_scopes is not None else [str(manifest["phase"])]
    attempts = (
        [attempt for scope in scopes for attempt in store.all_attempts(scope)] if store else []
    )
    by_pair: dict[tuple, list[dict]] = defaultdict(list)
    for attempt in attempts:
        job = json.loads(attempt["job"])
        if job.get("request", {}).get("evaluation_protocol") != protocol.version:
            continue  # Historical attempts remain charged, never relabeled as current evidence.
        by_pair[(job["model"], job["instance_id"])].append({**attempt, "job": job})
    rows = []
    instances = {
        name: {item.id: item for item in data.instances} for name, data in datasets.items()
    }
    for selected in manifest["jobs"]:
        key = (selected["model"], selected["instance_id"])
        history = by_pair.pop(key, [])
        terminals = []
        for attempt in history:
            job = attempt["job"]
            expected_condition = condition_id(
                provider=job["provider"],
                model=job["model"],
                representation="cells",
                prompt_version=prompt_version,
                request=job["request"],
            )
            if job["condition_id"] != expected_condition:
                raise ValueError("attempt request condition hash mismatch")
            for field in ("prompt_sha256", "instance_sha256", "gold_sha256"):
                if job.get(field) != selected[field]:
                    raise ValueError(f"{field} mismatch in formal attempt")
            if any(job.get(field) != selected.get(field) for field in ("model", "instance_id")):
                raise ValueError("attempt identity mismatch")
            result = (
                json.loads(attempt["result"])
                if attempt["result"]
                else (store.locked_result(attempt["id"]) if store and not finalized_only else None)
            )
            is_terminal = (
                terminal_predicate(result, job)
                if result and terminal_predicate is not None
                else bool(result and result["answer_eligible"])
            )
            if is_terminal:
                terminals.append((attempt, result))
        if len({attempt["job"]["condition_id"] for attempt in history}) > 1 or len(terminals) > 1:
            raise ValueError("mixed conditions or conflicting terminals in a study cell")
        active = (
            terminals[0]
            if terminals
            else (history[-1], json.loads(history[-1]["result"] or "{}"))
            if history
            else ({}, {})
        )
        attempt, result = active
        request_job = attempt.get("job") or {}
        record = {
            **selected,
            **result,
            "raw_text": result.get("text"),
            "model": selected["model"],
            "provider": request_job.get("provider"),
            "request": request_job.get("request", {}),
            "condition_id": request_job.get("condition_id"),
            "attempts": len(history),
            "collection_finalized": bool(attempt.get("result")),
            "representation": "cells",
            "prompt_version": prompt_version,
            "scoring_version": SCORING_VERSION,
            "evaluation_protocol": protocol.version,
        }
        dataset = datasets[selected["dataset"]]
        instance = instances[selected["dataset"]][selected["instance_id"]]
        record["prompt_chars"] = len(render_prompt(instance, prompt_version=prompt_version))
        rollout = str(instance.metadata.get("prefix_of", instance.id))
        row = score_formal_record(
            record,
            instance,
            dataset.gold[instance.id],
            dataset.traces.get(rollout),
            dataset.diagnostics.get(rollout),
        )
        row["endpoint_pause"] = store.paused(selected["model"]) if store else None
        row["known_usage_cost_upper_usd"] = (
            sum(attempt["known_upper"] or 0 for attempt in history) / 1e6
        )
        row["unknown_or_inflight_reserved_usd"] = (
            sum(attempt["accounted"] for attempt in history if attempt["known_upper"] is None) / 1e6
        )
        rows.append(row)
    if by_pair:
        raise ValueError("ledger contains attempts outside this frozen selection")
    return rows


def build_report(manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict:
    cell_keys = ("model", "family", "dimension", "w", "T")
    grouped: dict[tuple, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in cell_keys)].append(row)
    cells = [
        {**dict(zip(cell_keys, key, strict=True)), **summarize_group(group, cell=True)}
        for key, group in sorted(grouped.items(), key=str)
    ]
    model_order = list(dict.fromkeys(job["model"] for job in manifest["jobs"]))
    views = []
    task_curves = []
    for model in model_order:
        for stage in ("main", "workload") if _is_formal(manifest) else ("preflight",):
            selected = [
                row
                for row in rows
                if row["model"] == model
                and (
                    stage == "preflight"
                    or (stage == "main" and row["dataset"] == "main")
                    or (
                        stage == "workload"
                        and (
                            row["dataset"] == "rq2"
                            or row["instance_id"] in manifest["workload_reuse_ids"]
                        )
                    )
                )
            ]
            if stage == "workload" and not any(row["dataset"] == "rq2" for row in selected):
                continue
            for horizon in sorted({row["T"] for row in selected}):
                at_t = [row for row in selected if row["T"] == horizon]
                views.append({"model": model, "view": stage, "T": horizon, **summarize_group(at_t)})
                for family in sorted({row["family"] for row in at_t}):
                    task_curves.append(
                        {
                            "model": model,
                            "view": stage,
                            "family": family,
                            "T": horizon,
                            **summarize_group([row for row in at_t if row["family"] == family]),
                        }
                    )
    # Dataset descriptors are computed ONCE per instance, not once per model.
    descriptors = {}
    fields = (
        "instance_id",
        "family",
        "dimension",
        "shape",
        "w",
        "w_kind",
        "T",
        "n_core",
        "n_set",
        "mean_core_scan",
        "mean_core_effect",
        "cumulative_core_scan",
        "changed_cells",
        "erasure_fraction",
        "first_effective",
        "copy_baseline_em",
        "copy_baseline_cell_accuracy",
        "zero_baseline_em",
        "zero_baseline_cell_accuracy",
        "initial_density",
        "final_density",
        "boundary_fraction",
        "prompt_chars",
    )
    for row in rows:
        descriptors.setdefault(
            row["instance_id"],
            {
                **{key: row.get(key) for key in fields},
                **{key: value for key, value in row.items() if key.startswith("suffix_")},
            },
        )
    main = [row for row in rows if row["dataset"] == "main"]
    workload_models = {row["model"] for row in rows if row["dataset"] == "rq2"}
    workload = [
        row
        for row in rows
        if row["model"] in workload_models
        and (row["dataset"] == "rq2" or row["instance_id"] in manifest["workload_reuse_ids"])
    ]
    complete = bool(rows) and all(row["em"] is not None for row in rows)
    return {
        "selection_sha256": manifest["selection_sha256"],
        "phase": manifest["phase"],
        "complete": complete,
        "formal_ranking_available": complete
        and _is_formal(manifest)
        and not any(row.get("protocol_violation") or row.get("endpoint_pause") for row in rows),
        "cells": cells,
        "views": views,
        "task_curves": task_curves,
        "paired_model_comparisons": {
            str(t): paired_comparisons([row for row in main if row["T"] == t]) for t in (32, 64)
        },
        "dimension_contrasts": independent_contrasts(main, "dimension"),
        "workload_contrasts": independent_contrasts(workload, "w"),
        "dimension_length_interactions": interaction_contrasts(main, "dimension"),
        "workload_length_interactions": interaction_contrasts(workload, "w"),
        "paired_task_comparisons": {
            f"{family}:T{t}": paired_comparisons(
                [row for row in main if row["family"] == family and row["T"] == t]
            )
            for family in sorted({row["family"] for row in main})
            for t in (32, 64)
        },
        "data_profile": list(descriptors.values()),
        "notes": [
            "EM is the sole primary score; intervals are marginal, not simultaneous bands.",
            "Four trajectories/cell are exploratory, not precise rankings.",
            "Incomplete selections are not ranked; observed subset means are explicitly labeled.",
            "Wrong-cell statistics condition on valid incorrect answers and include n.",
            "No automatic epsilon/mechanism fitting or significance-star grid.",
        ],
    }


def write_report(
    out: Path,
    manifest: Mapping[str, Any],
    report: dict,
    rows: Sequence[Mapping[str, Any]],
    *,
    plots: bool = False,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("selection.json", manifest),
        ("summary.json", report),
        ("data_profile.json", report["data_profile"]),
    ):
        (out / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    (out / "results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    lines = [
        "# Grid-Bench study report",
        "",
        f"Phase: {report['phase']}.",
        "",
        "Complete fixed selection."
        if report["complete"]
        else "INCOMPLETE — no formal ranking; missing observations remain missing.",
        "",
    ]
    lines.extend(report["notes"])
    if report.get("batch_id"):
        lines += [
            "",
            f"Batch: {report['batch_id']}. Whole-study complete: {report['study_complete']}.",
            "Batch completion does not certify completion of the candidate panel.",
        ]
    lines += [
        "",
        "## Overview (T=32 and T=64)",
        "",
        "| Model | View | T | Correct / scored / expected | EM | 95% CI |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["views"]:
        if row["T"] not in (32, 64):
            continue
        em = "—" if row["em"] is None else f"{row['em']:.3f}"
        ci = "—" if row["ci95"] is None else f"[{row['ci95'][0]:.3f}, {row['ci95'][1]:.3f}]"
        lines.append(
            f"| {row['model']} | {row['view']} | {row['T']} | "
            f"{row['correct']} / {row['scored']} / {row['expected']} | {em} | {ci} |"
        )
    lines += [
        "",
        "Full per-task curves, counts, conditional errors and interval warnings are in "
        "`summary.json`; dataset-only diagnostics are in `data_profile.json`.",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines))
    if plots:
        write_plots(out, report["cells"])


def write_plots(out: Path, cells: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for family in sorted({row["family"] for row in cells}):
        group = [row for row in cells if row["family"] == family]
        colors = {
            model: f"C{index % 10}"
            for index, model in enumerate(sorted({row["model"] for row in cells}))
        }
        variants = sorted({(row["dimension"], row["w"]) for row in group})
        fig, axes = plt.subplots(1, len(variants), figsize=(4 * len(variants), 3.5), squeeze=False)
        for ax, (dimension, workload) in zip(axes[0], variants, strict=True):
            for model in sorted({row["model"] for row in group}):
                points = sorted(
                    (
                        row
                        for row in group
                        if row["model"] == model
                        and row["dimension"] == dimension
                        and row["w"] == workload
                        and row["em"] is not None
                    ),
                    key=lambda row: row["T"],
                )
                if points:
                    ax.errorbar(
                        [row["T"] for row in points],
                        [row["em"] for row in points],
                        yerr=[
                            [max(0, row["em"] - row["ci95"][0]) for row in points],
                            [max(0, row["ci95"][1] - row["em"]) for row in points],
                        ],
                        marker="o",
                        color=colors[model],
                        capsize=2,
                        label=model,
                    )
            ax.set(
                title=f"d={dimension}, w={workload}",
                xlabel="Sequence length T",
                ylabel="Semantic EM",
                ylim=(-0.03, 1.03),
            )
        if axes[0][0].get_legend_handles_labels()[0]:
            axes[0][0].legend(fontsize=6)
        fig.suptitle(family + " (marginal Wilson 95% intervals)")
        fig.tight_layout()
        fig.savefig(out / f"curves-{family}.svg")
        plt.close(fig)
