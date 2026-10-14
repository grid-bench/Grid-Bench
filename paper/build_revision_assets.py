"""Build expanded manuscript visuals and tables from frozen offline evidence.

--check-only validates the supplement, all generated tables, and asset hashes.
It never changes source data, existing figure builders, or manuscript prose.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Literal

from analyze_supplement import OUT as SUPPLEMENT
from analyze_supplement import encoded, sha
from build_figures import (
    COLORS,
    DATA,
    LENGTHS,
    MODELS,
    NAMES,
    ROOT,
    TASKS,
    check_tables,
    read_tables,
)
from model_labels import WRAPPED_NAMES, tex_header

from spatial_worlds.eval.expanded_analysis_registry import (
    ALL_TRAJECTORIES,
    CELL_TRAJECTORIES,
    MAIN_TRAJECTORIES,
    PER_MODEL_INSTANCES,
)

OUT = ROOT / "paper/figures"
TABLES = ROOT / "paper/tables"
MANIFEST = ROOT / "paper/data/revision-assets-manifest.json"
WORKLOADS = {
    "metric_paint": (25, 100),
    "box_reflection": (16, 64, 256),
    "boundary_layer": (15, 60, 150),
    "occluded_mix": (6, 12, 30),
}
MAIN_W = dict(zip(TASKS, (100, 64, 10, 12, 100, 60, 30, 256), strict=True))
ERROR_COLORS = {"changed_only": "#C58A3B", "unchanged_only": "#6E9D9B", "both": "#304B63"}
POOLED_LENGTHS = (8, 16, 32, 64)
EXPERIMENT_ASSETS = {
    "tables": [
        f"paper/tables/{stem}.tex"
        for stem in ("revision-coverage", "revision-dimension-summary", "revision-baselines")
    ],
    "figures": [
        f"paper/figures/{stem}.pdf"
        for stem in (
            "length-overview", "revision-rule-matrix", "revision-workload-summary",
            "revision-error-contrast", "revision-error-composition",
        )
    ],
}
APPENDIX_ASSETS = {
    "tables": [
        f"paper/tables/{stem}.tex"
        for stem in (
            "revision-endpoints", "revision-history-summary", "revision-cohorts",
            "revision-prefix-summary", "revision-error-summary",
            "revision-changed-unchanged-2", "revision-matched-baselines-2",
        )
    ],
    "figures": [
        f"paper/figures/{stem}.pdf"
        for stem in (
            "task-length", "revision-dimensions-full", "revision-workloads-full",
            "revision-workloads-full-2", "revision-workloads-full-3",
            "revision-workloads-full-4",
        )
    ],
}


def selected(rows: list[dict], **conditions) -> list[dict]:
    return [r for r in rows if all(r[k] == v for k, v in conditions.items())]


def one(rows: list[dict], **conditions) -> dict:
    result = selected(rows, **conditions)
    assert len(result) == 1, conditions
    return result[0]


def num(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def pct(value: float | None) -> str:
    return "--" if value is None else f"{100 * value:.1f}"


def paired_number(value: float) -> str:
    """Print exact zero without a sign; preserve every nonzero value."""
    return f"{0.0 if value == 0 else value:.2f}"


def exact_percent(row: dict) -> float:
    """Use the recorded denominator, including after an evidence expansion."""
    assert row["n"] > 0 and 0 <= row["correct"] <= row["n"]
    return 100 * row["correct"] / row["n"]


def pooled_workload(rows: list[dict]) -> list[dict]:
    """Pool four complete prefix cells for each model and workload setting."""
    output = []
    for family, levels in WORKLOADS.items():
        for model in MODELS:
            for workload in levels:
                points = [
                    one(rows, model=model, family=family, w=workload, T=length)
                    for length in POOLED_LENGTHS
                ]
                assert all(p["n"] == CELL_TRAJECTORIES for p in points)
                row = {
                    "model": model,
                    "family": family,
                    "w": workload,
                    "n": sum(p["n"] for p in points),
                    "correct": sum(p["correct"] for p in points),
                }
                exact_percent(row)
                output.append(row)
    return output


def cohort_summary(analysis: dict) -> list[list]:
    """Recover disjoint cohort counts from the frozen original and merged views."""
    history = analysis["historical_analysis"]
    path = ROOT / history["snapshot"]
    assert sha(path) == history["snapshot_sha256"]
    original = json.loads(path.read_text())
    output = []
    for model in MODELS:
        points = [
            one(original["tables"]["length"], model=model, T=length)
            for length in POOLED_LENGTHS
        ]
        combined = [
            one(analysis["tables"]["length"], model=model, T=length)
            for length in POOLED_LENGTHS
        ]
        old_n = sum(p["n"] for p in points)
        full_n = sum(p["n"] for p in combined)
        old_correct = sum(p["correct"] for p in points)
        full_correct = sum(p["correct"] for p in combined)
        assert old_n == 384 and full_n == 2 * old_n
        assert 0 <= full_correct - old_correct <= full_n - old_n
        output.append([NAMES[model], old_correct, full_correct - old_correct, full_correct])
    return output


def table(
    stem: str,
    headers: list[str],
    rows: list[list],
    caption: str,
    *,
    wide: bool = True,
    columns: str | None = None,
    label: str | None = None,
    group_header: str | None = None,
    tabcolsep: float = 4,
    placement: str = "t",
) -> str:
    environment = "table*" if wide else "table"
    width = r"\textwidth" if wide else r"\columnwidth"
    if columns is None:
        columns = "l" + r"*{" + str(len(headers) - 1) + r"}{>{\centering\arraybackslash}X}"
    lines = [
        "% Generated by paper/build_revision_assets.py; do not edit manually.",
        rf"\begin{{{environment}}}[{placement}]",
        r"\centering",
        r"\small",
        r"\begingroup",
        rf"\setlength{{\tabcolsep}}{{{tabcolsep:g}pt}}",
        r"\renewcommand{\arraystretch}{1.08}",
        rf"\begin{{tabularx}}{{{width}}}{{{columns}}}",
        r"\toprule",
    ]
    if group_header is not None:
        lines += [r"\tablehead", group_header + r" \\"]
    lines += [r"\tablehead", " & ".join(headers) + r" \\", r"\midrule"]
    for i, row in enumerate(rows):
        assert len(row) == len(headers), (stem, row)
        lines += [
            r"\rowcolor{TableStripe}" if i % 2 == 0 else r"\rowcolor{white}",
            " & ".join(str(v) for v in row) + r" \\",
        ]
    lines += [
        r"\bottomrule",
        r"\end{tabularx}",
        r"\endgroup",
        r"\caption{" + caption + "}",
        r"\label{" + (label or "tab:" + stem) + "}",
        rf"\end{{{environment}}}",
        "",
    ]
    return "\n".join(lines)


def build_tables(data: dict) -> dict[str, str]:
    output = {}

    def add(stem: str, headers: list[str], rows: list[list], caption: str, **kwargs) -> None:
        if stem in {
            "revision-error-extent",
            "revision-error-partitions",
            "revision-matched-baselines",
            "revision-changed-unchanged",
            "revision-format-resources",
            "revision-resource-usage",
        }:
            for index, lengths in enumerate((LENGTHS[:3], LENGTHS[3:])):
                suffix = "-2" if index else ""
                part = [row for row in rows if row[1] in lengths]
                part_options = dict(kwargs)
                if stem == "revision-changed-unchanged" and index == 1:
                    # Keep the late-history diagnostic beside its discussion.
                    part_options["placement"] = "!b"
                output[f"{stem}{suffix}.tex"] = table(
                    stem + suffix,
                    headers,
                    part,
                    caption + " This table covers $T=" + ",".join(map(str, lengths)) + "$.",
                    **part_options,
                )
        elif stem == "revision-task-errors":
            for index in range(2):
                suffix = "-2" if index else ""
                output[f"{stem}{suffix}.tex"] = table(
                    stem + suffix, headers, rows[index * 24 : (index + 1) * 24], caption, **kwargs
                )
        else:
            output[f"{stem}.tex"] = table(stem, headers, rows, caption, **kwargs)

    add(
        "revision-design",
        ["View", "Trajectories", "Instances", "Construction and role"],
        [
            ["Main", 192, 1152, "Eight rules, three dimensions, eight trajectories, six prefixes."],
            ["Additional workload", 56, 336, "Seven additional conditions at $d=2$."],
            ["Workload analysis", 88, 528, "Additional instances plus 192 reused main instances."],
            [
                "Unique evaluation",
                ALL_TRAJECTORIES,
                PER_MODEL_INSTANCES,
                "Main plus additional workload; the same mask per model.",
            ],
        ],
        "Sample accounting per model. Six separately queried prefixes per trajectory are "
        "$T=0,1,8,16,32,64$. Six models yield 8,928 outcomes on 1,488 unique instances "
        "from 248 reference trajectories. The workload view overlaps the main view.",
        columns=r"lrrX",
        label="tab:design",
    )

    analysis = json.loads((DATA / "analysis.json").read_text())
    add(
        "revision-coverage",
        ["Evaluation", r"\shortstack{Instances\\per model}", r"\shortstack{Six-model\\outcomes}"],
        [
            [
                "Main",
                f"{MAIN_TRAJECTORIES * len(LENGTHS):,}",
                f"{MAIN_TRAJECTORIES * len(LENGTHS) * len(MODELS):,}",
            ],
            [
                "Additional workload",
                f"{PER_MODEL_INSTANCES - MAIN_TRAJECTORIES * len(LENGTHS):,}",
                f"{(PER_MODEL_INSTANCES - MAIN_TRAJECTORIES * len(LENGTHS)) * len(MODELS):,}",
            ],
            ["Unique total", "1,488", "8,928"],
        ],
        "Evaluation coverage. Lengths are $T=0,1,8,16,32,64$. The workload analysis "
        "combines seven additional settings with four reused main settings, for 528 "
        "instances per model; reused instances are counted only once in the total.",
        columns=r"Xrr",
        wide=False,
        label="tab:evaluation-coverage",
    )
    add(
        "revision-cohorts",
        ["Model", "Original /384", "Additional /384", "Combined /768"],
        cohort_summary(analysis),
        "Exact main-set answers over $T=8,16,32,64$, separated by collection cohort. "
        "The original and additional cohorts each contain 96 trajectories, with four "
        "prefixes per trajectory. They use disjoint seeds under the same selected "
        "generation and evaluation conditions. The combined column is their sum.",
        wide=False,
    )
    workload = pooled_workload(read_tables()["workload"])
    add(
        "revision-workload-summary",
        [r"\textbf{Task}", "$w$", *[tex_header(m) for m in MODELS]],
        [
            [r"\task{" + TASKS[family] + "}", w]
            + [one(workload, family=family, w=w, model=m)["correct"] for m in MODELS]
            for family, levels in WORKLOADS.items()
            for w in levels
        ],
        "Exact workload answers pooled over $T=8,16,32,64$, out of 32 per cell. "
        "Each setting comprises eight trajectories evaluated at four nested prefixes. "
        "Targets have the family-specific meanings in Figure~"
        r"\ref{fig:workload-summary}; the data package supplies all length-specific counts.",
        columns=r"lr*{6}{>{\centering\arraybackslash}X}",
    )
    condition_map = {condition["id"]: condition for condition in data["model_conditions"]}
    endpoint_rows = []
    for model in MODELS:
        condition = condition_map[model]
        route = condition.get("expected_upstream_model", model)
        parameters = condition["request"]["parameters"]
        displayed_parameters = r"\newline ".join(
            r"\nolinkurl{" + key + r"} = \texttt{" + json.dumps(value) + "}"
            for key, value in parameters.items()
        )
        if condition["effort"] == "native_always_on":
            assert "reasoning_effort" not in parameters
            displayed_parameters += r"\newline Effort absent; native thinking."
        endpoint_rows.append(
            [
                NAMES[model],
                r"\nolinkurl{" + model + "}",
                r"\nolinkurl{" + route + "}",
                displayed_parameters,
            ]
        )
    add(
        "revision-endpoints",
        ["Model", "Requested identity", "Recorded route", "Recorded request parameters"],
        endpoint_rows,
        "Six evaluated models, each with 1,488/1,488 scored instances, prompt 0.4, a "
        "128,000-token generation allowance, and no tools. Parameter keys and values are "
        "copied from the retained request ledger. For Bedrock, ledger "
        r"\texttt{max\_tokens} maps to wire field \texttt{inferenceConfig.maxTokens}; "
        r"gateway requests retain \texttt{max\_completion\_tokens}. MiniMax M2.5 sends no effort "
        "field and uses native always-on thinking. Identifiers and routes follow the "
        "frozen collection metadata.",
        columns=(
            r"l>{\hsize=.8\hsize\linewidth=\hsize\raggedright\arraybackslash}X"
            r">{\hsize=.9\hsize\linewidth=\hsize\raggedright\arraybackslash}X"
            r">{\hsize=1.3\hsize\linewidth=\hsize\raggedright\arraybackslash}X"
        ),
        label="tab:endpoints",
    )
    for length in (32, 64):
        pairs = analysis["paired_main_comparisons"][str(length)]
        pair_rows = []
        for first_index, first in enumerate(MODELS):
            for second in MODELS[first_index + 1 :]:
                r = next(
                    row for row in pairs if {row["reference"], row["other"]} == {first, second}
                )
                sign = 1 if r["other"] == first else -1
                difference = 100 * sign * r["difference_other_minus_reference"]
                interval = r["ci95"]
                if interval is None:
                    interval_text = "--"
                else:
                    lo, hi = interval if sign == 1 else (-interval[1], -interval[0])
                    interval_text = rf"$[{paired_number(100 * lo)},{paired_number(100 * hi)}]$"
                pair_rows.append(
                    [
                        NAMES[first] + " $-$ " + NAMES[second],
                        paired_number(difference),
                        interval_text,
                    ]
                )
        assert len(pair_rows) == 15
        add(
            f"revision-paired-{length}",
            ["Difference", "pp", "95\\% interval"],
            pair_rows,
            f"All 15 paired semantic-EM differences at $T={length}$, in percentage points "
            "on the same 192 main instances. Intervals use 2,000 stratified whole-trajectory "
            "bootstrap samples and are marginal, without multiplicity adjustment. A dash "
            "denotes a degenerate bootstrap interval, not certainty.",
            label="tab:paired" if length == 32 else "tab:paired-64",
        )

    pair_rows = []
    for subset, label in (
        ("joint_valid", "Both valid"),
        ("joint_valid_incorrect", "Both valid, incorrect"),
    ):
        r = one(data["matched_pairs"], model_a=MODELS[4], model_b=MODELS[5], T=8, subset=subset)
        pair_rows.append(
            [
                label,
                r["n"],
                num(r["wrong_cells_median_a"]),
                num(r["wrong_cells_median_b"]),
                pct(r["cell_accuracy_a"]),
                pct(r["cell_accuracy_b"]),
                r["a_fewer_wrong"],
                r["equal_wrong"],
                r["b_fewer_wrong"],
            ]
        )
    add(
        "revision-paired-errors",
        [
            "Shared subset",
            "$n$",
            "Kimi K2.5 med.",
            "MiniMax M2.5 med.",
            "Kimi K2.5 CA",
            "MiniMax M2.5 CA",
            "Kimi K2.5 fewer",
            "Tie",
            "MiniMax M2.5 fewer",
        ],
        pair_rows,
        "Kimi K2.5--MiniMax M2.5 errors at $T=8$ on the same selected instances. "
        "Med. counts wrong cells; CA is mean per-instance cell accuracy (percent). "
        "The last three columns compare wrong-cell counts per instance and sum to $n$. "
        "The first row includes exact answers, whereas the second requires both answers "
        "to be valid and incorrect. Both are outcome-selected subsets of 192 main requests.",
        columns=r"l*{8}{>{\centering\arraybackslash}X}",
    )

    summary = []
    for family, name in TASKS.items():
        summary.append(
            [r"\task{" + name + "}"]
            + [
                f"{one(data['task_summary'], family=family, model=m)['correct']}, "
                f"{one(data['task_summary'], family=family, model=m)['valid']}"
                for m in MODELS
            ]
        )
    summary.append(
        ["All eight rules"]
        + [
            f"{one(data['pooled_summary'], model=m)['correct']}, "
            f"{one(data['pooled_summary'], model=m)['valid']}"
            for m in MODELS
        ]
    )
    add(
        "revision-task-summary",
        [r"\textbf{Rule}", *[tex_header(m) for m in MODELS]],
        summary,
        "Substantive main-set results pooled over $T=8,16,32,64$. "
        "Each cell gives exact-count, valid-count (E, V): the denominator is 96 for each rule "
        "and 768 for the final row. Valid states include exact ones; empty and invalid outputs "
        "remain in the exact-match denominator. The 96 instances derive from 24 trajectories "
        "queried at four nested prefixes, not 96 independent histories. Controls are excluded.",
    )
    baselines = []
    for model in MODELS:
        r = one(data["matched_valid_baselines"], model=model, T=64)
        baselines.append(
            [NAMES[model], f"{r['valid_n']}/{r['n']}",
             pct(r["model_ca"]), pct(r["copy_ca"]), pct(r["zero_ca"])]
        )
    add(
        "revision-baselines",
        [r"\textbf{Model}", "$n/N$", r"\textbf{Pred.}", r"\textbf{Copy}", r"\textbf{Zero}"],
        baselines,
        "Cell accuracy (percent) at $T=64$. $n/N$ gives valid grids, including exact ones, "
        "out of all main requests. "
        "Pred. denotes the model prediction; Copy and Zero denote copy-initial and all-zero. "
        "All three use the same instances within each row; "
        "subsets differ across models.",
        columns=r"lr*{3}{>{\centering\arraybackslash}X}",
        wide=False,
    )
    old = read_tables()
    check_tables(old)
    dimension_rows = [
        [NAMES[model]]
        + [
            one(old["dimension_length"], model=model, T=length, dimension=dimension)["correct"]
            for length in (1, 16)
            for dimension in (2, 3, 4)
        ]
        for model in MODELS
    ]
    add(
        "revision-dimension-summary",
        [r"\textbf{Model}", *[rf"\mbox{{$d{{=}}{d}$}}" for _ in (1, 16) for d in (2, 3, 4)]],
        dimension_rows,
        "Exact answers out of 64 main instances per cell, retaining all failures. "
        "The two lengths show early and intermediate structural contrasts. "
        "Worlds at $d=2,3,4$ contain 256, 252, and 256 cells; geometry and coordinate "
        "representation change together. Complete length curves appear in "
        r"Appendix Figure~\ref{fig:structure}.",
        wide=False,
        group_header=r" & \multicolumn{3}{c}{$T=1$} & \multicolumn{3}{c}{$T=16$}",
        label="tab:dimension-summary",
        tabcolsep=3,
    )
    workload_rows = []
    for family, levels in WORKLOADS.items():
        for w in levels:
            workload_rows.append(
                [r"\task{" + TASKS[family] + "}", w]
                + [
                    f"{one(old['workload'], family=family, w=w, model=m, T=64)['correct']}, "
                    f"{one(old['workload'], family=family, w=w, model=m, T=64)['valid']}"
                    for m in MODELS
                ]
            )
    add(
        "revision-workload-valid",
        [r"\textbf{Rule}", "$w$", *[tex_header(m) for m in MODELS]],
        workload_rows,
        "Workload outcomes at $d=2,T=64$. Every endpoint cell gives "
        "exact-count, valid-count (E, V), both out of eight instances. Valid includes exact. "
        "These two counts distinguish incorrect valid states from invalid or empty outputs "
        "even where all exact counts are zero. $w$ is task-specific; workload cells are "
        "independently generated, and main-condition requests are reused.",
    )
    error_rows, part_rows, baseline_rows, format_rows, resources = [], [], [], [], []
    for model in MODELS:
        for t in LENGTHS:
            r = one(data["errors"], model=model, T=t)
            error_rows.append(
                [
                    NAMES[model],
                    t,
                    r["valid_wrong_n"],
                    num(r["median"]),
                    num(r["p90"]),
                    r["near5"],
                ]
            )
            part_rows.append(
                [
                    NAMES[model],
                    t,
                    r["changed_only"],
                    r["unchanged_only"],
                    r["both"],
                    r["valid_wrong_n"],
                ]
            )
            b = one(data["matched_valid_baselines"], model=model, T=t)
            baseline_rows.append(
                [
                    NAMES[model],
                    t,
                    b["valid_n"],
                    pct(b["model_ca"]),
                    pct(b["copy_ca"]),
                    pct(b["zero_ca"]),
                    b["above_copy"],
                ]
            )
            r = one(data["format_resources"], model=model, T=t)
            format_rows.append(
                [
                    NAMES[model],
                    t,
                    r["correct"],
                    r["strict_correct"],
                    r["valid_json"],
                    r["valid"],
                    r["canonical"],
                ]
            )
            resources.append(
                [
                    NAMES[model],
                    t,
                    num(r["output_tokens_median"]),
                    num(r["output_tokens_p90"]),
                    r["length"],
                    r["length_empty"],
                    r["stop_empty"],
                ]
            )
    add(
        "revision-error-extent",
        ["Model", "$T$", "$n$", "Median", "P90", r"$\leq5$"],
        error_rows,
        "Wrong-cell extent among valid incorrect main answers. $n$ counts those "
        "answers; median and P90 count wrong cells out of 252 or 256. P90 uses linear "
        "interpolation at order position $(n-1)0.9$, not a confidence interval. The final "
        "column counts answers with at most five wrong cells, a descriptive near-miss threshold "
        "that does not change EM. Empty conditional subsets have undefined quantiles, "
        "shown as dashes.",
    )
    add(
        "revision-error-partitions",
        ["Model", "$T$", "$C$ only", "$U$ only", "Both", "$n$"],
        part_rows,
        "Error-location partitions of valid incorrect main answers. $C$ and $U$ "
        "are coordinates whose final reference value differs from or equals its initial value. "
        "The first three counts sum to $n$; empty and invalid answers cannot enter this "
        "spatial diagnostic but remain failures in primary EM. Final-unchanged does not mean "
        "outside operation support.",
    )
    add(
        "revision-matched-baselines",
        ["Model", "$T$", "$n$", "Prediction", "Copy", "Zero", "Above copy"],
        baseline_rows,
        "Cell accuracy (percent) for model predictions, copy-initial, and all-zero. $n$ is "
        "the number of valid main answers; all three predictions in a row use the "
        "same instances. Above copy counts model answers with strictly greater "
        "cell accuracy than copy-initial. Valid-answer subsets can differ across models.",
    )
    add(
        "revision-changed-unchanged",
        ["Model", "$T$", "$n_C$", "$C$ acc. (\\%)", "$n_U$", "$U$ acc. (\\%)"],
        [
            [
                NAMES[m],
                t,
                (r := one(data["matched_valid_baselines"], model=m, T=t))["changed_n"],
                pct(r["changed_ca"]),
                r["unchanged_n"],
                pct(r["unchanged_ca"]),
            ]
            for m in MODELS
            for t in LENGTHS
        ],
        "Changed- and unchanged-cell accuracy for valid main answers. Each value is a "
        "macro-average over the $n_C$ or $n_U$ answers with a nonempty reference subset. "
        "$C$ denotes final-reference-changed cells; $U$ denotes final-reference-unchanged "
        "cells. Accuracy is undefined for an empty contributing subset.",
    )
    add(
        "revision-task-errors",
        ["Rule", "Model", "$n$", "Median", "P90", r"$\leq5$"],
        [
            [
                TASKS[f],
                NAMES[m],
                (r := one(data["task_errors_T64"], family=f, model=m))["valid_wrong_n"],
                num(r["median"]),
                num(r["p90"]),
                r["near5"],
            ]
            for f in TASKS
            for m in MODELS
        ],
        "Rule-specific wrong-cell extent at $T=64$ among valid incorrect main answers. "
        "$n$ counts these answers out of 24 total requests per rule and endpoint. Median "
        "and linearly interpolated P90 are wrong-cell counts; the final column counts "
        "near misses with at most five wrong cells. Dashes indicate no valid incorrect "
        "answer, not zero error or successful delivery.",
    )
    add(
        "revision-format-resources",
        ["Model", "$T$", "EM", "Strict", "JSON", "State", "Ordered"],
        format_rows,
        "Format and correctness counts, each out of all 192 main requests at "
        "the given length. EM is semantic exact match after frozen extraction; Strict applies "
        "the same scorer to the raw response. JSON, State, and Ordered count valid JSON, "
        "valid complete states, and lexicographically ordered valid states. These overlapping "
        "properties are not a disjoint failure partition.",
    )
    add(
        "revision-resource-usage",
        ["Model", "$T$", "Token median", "Token P90", "Length", "Length-empty", "Stop-empty"],
        resources,
        "Provider-reported combined output-token distribution and termination "
        "counts across all 192 main requests per row, including empty and invalid outputs. "
        "P90 uses linear interpolation and is not uncertainty. Length, length-empty, and "
        "stop-empty are distinct counts. Reported output includes reasoning; provider-specific "
        "reasoning subfields do not establish equal private compute.",
    )
    add(
        "revision-prefix-transitions",
        ["Model", "$T$ pair", "Both exact", "Gain", "Loss", "Both wrong"],
        [
            [
                NAMES[r["model"]],
                f"{r['from_T']}--{r['to_T']}",
                r["both_correct"],
                r["gain"],
                r["loss"],
                r["both_wrong"],
            ]
            for r in data["prefix"]["transitions"]
        ],
        "Paired prefix outcomes on the same 192 main trajectories. Gain means incorrect at "
        "the shorter prefix and exact at the longer; Loss is the converse. All four counts "
        "sum to 192. Prefixes are independently queried, not model-state continuations, so "
        "a gain does not identify internal self-correction.",
    )
    for index in range(8):
        families = [list(TASKS)[index]]
        rows = []
        for family in families:
            for model in MODELS:
                for d in (2, 3, 4):
                    points = [
                        one(
                            data["task_dimension_length"],
                            family=family,
                            model=model,
                            dimension=d,
                            T=t,
                        )
                        for t in LENGTHS
                    ]
                    rows.append(
                        [TASKS[family], NAMES[model], d]
                        + [f"{r['correct']}, {r['valid']}" for r in points]
                    )
        add(
            f"revision-detail-{index + 1}",
            ["Rule", "Model", "$d$", *[f"$T={t}$" for t in LENGTHS]],
            rows,
            "Full task--dimension--length counts for "
            + " and ".join(TASKS[f] for f in families)
            + ". Every entry gives exact-count, valid-count (E, V), each out of eight "
            "trajectories at that length. Empty and invalid answers remain in the denominator; "
            "valid includes exact. These are the smallest evaluated main-set cells.",
        )
    for index in range(2):
        families = list(TASKS)[index * 4 : index * 4 + 4]
        rs = [
            r
            for f in families
            for r in sorted(
                selected(data["reference"]["retention"], family=f), key=lambda r: r["T"]
            )
        ]
        stem = "revision-retention" + ("-2" if index else "")
        add(
            stem,
            [
                "Rule",
                "$T$",
                "Erased / eligible",
                "Erased (\\%)",
                "Suffix-8",
                "Suffix-16",
                "Suffix-32",
            ],
            [
                [
                    TASKS[r["family"]],
                    r["T"],
                    f"{r['erased']} / {r['eligible']}",
                    pct(r["erasure_rate"]),
                    r["suffix8"],
                    r["suffix16"],
                    r["suffix32"],
                ]
                for r in rs
            ],
            "Reference-dynamics diagnostics on 24 unique main trajectories per rule, counted "
            "once rather than once per endpoint. Erasure pools eligible effective core events "
            "in the first half whose individual omission is invisible at the final state. "
            "Suffix columns are exact counts out of 24; suffix-$k$ executes every SET and "
            "core events only within the final $k$ event positions. If $k\\geq T$, it executes "
            "the full history and is not a shortened-history baseline.",
        )
    merge = data["reference"]["relabel_merge"]
    add(
        "revision-relabel-merges",
        ["Dimension", "Trajectory index", "Core events", "Merge-contact events"],
        [
            [
                r["rollout_id"].split("/")[2][1:],
                int(r["rollout_id"].split("/")[-1]),
                r["core_n"],
                r["merge_contact_events"],
            ]
            for r in merge
        ],
        f"RELABEL merge-contact audit at $T=64$ on the {len(merge)} selected main trajectories. "
        "The last column counts core events with positive merge-contact diagnostics, not "
        "the number of merged components. There are "
        f"{sum(r['merge_contact_events'] for r in merge)} "
        f"such events among {sum(r['core_n'] for r in merge)} core events and at least one in "
        f"{sum(r['merge_contact_events'] > 0 for r in merge)} of {len(merge)} trajectories.",
    )
    for view in ("main", "workload"):
        rows = []
        for r in data["reference"]["realized"]:
            is_main = r["w"] == MAIN_W[r["family"]]
            if is_main != (view == "main"):
                continue
            rows.append(
                [
                    TASKS[r["family"]],
                    r["dimension"],
                    r["w"],
                    num(r["mean_core_scan"]),
                    num(r["mean_core_effect"]),
                    pct(r["initial_density"]),
                    pct(r["final_density"]),
                    pct(r["boundary_fraction"]),
                ]
            )
        add(
            f"revision-realized-{view}",
            [
                "Rule",
                "$d$",
                "$w$",
                "Reads",
                "Changes",
                "Initial (\\%)",
                "Final (\\%)",
                "Boundary (\\%)",
            ],
            rows,
            "Realized reference properties at $T=64$ for "
            + (
                "the main conditions"
                if view == "main"
                else "the seven additional workload conditions"
            )
            + ". Each row averages eight unique trajectories, without endpoint replication. "
            "Reads and Changes are per-core-event reference reads and changed cells; density "
            "columns are initial and final occupied-cell percentages. Boundary is the "
            "percentage of world cells on a geometric boundary. These are descriptive "
            "properties, not a common measure of model computation.",
        )
    case_rows = [
        [
            r["name"].replace("_", " "),
            NAMES[r["model"]],
            TASKS[r["family"]],
            r["T"],
            r["wrong_cells"],
            r"\texttt{" + r["instance_id"].split("/test/")[1] + "}",
        ]
        for r in data["failure_cases"]
    ]
    add(
        "revision-cases",
        ["Case", "Model", "Rule", "$T$", "Wrong cells", "Index / prefix"],
        case_rows,
        "Deterministically selected $d=2$ main-set examples (256 cells). Within "
        "each declared category choose the lexicographically first instance ID: GPT-5.6 Sol $T=16$ "
        "with one to five wrong cells; GPT-5.4 Mini $T=1$ with errors only on "
        "final-unchanged cells; "
        "GPT-5.4 Mini $T=64$ with at least half the cells wrong. The final column supplies the "
        "trajectory index and prefix suffix. These illustrate error types, not their prevalence.",
    )
    history_rows = []
    for model in MODELS:
        cells = []
        for length in LENGTHS:
            row = one(data["format_resources"], model=model, T=length)
            other = row["invalid"] + row["empty"]
            assert row["correct"] + row["valid_wrong"] == row["valid"]
            assert row["valid"] + other == row["n"] == MAIN_TRAJECTORIES
            assert row["correct"] == one(old["length"], model=model, T=length)["correct"]
            cells.append(f"{row['correct']}/{row['valid_wrong']}/{other}")
        history_rows.append([NAMES[model], *cells])
    add(
        "revision-history-summary",
        ["Model", *[f"$T={t}$" for t in LENGTHS]],
        history_rows,
        "Answer composition at each main history length. Each cell lists "
        "exact / valid incorrect / invalid or empty counts, "
        f"which sum to {MAIN_TRAJECTORIES}. The first two categories form the valid-state "
        "population; the second is used for error-extent and location diagnostics. "
        "All six lengths use separately queried prefixes of the same trajectories.",
        columns=r"l*{6}{>{\centering\arraybackslash}X}",
        tabcolsep=3,
    )
    add(
        "revision-prefix-summary",
        ["Model", "Including $T=0$", "Positive lengths only"],
        [[NAMES[m], (r := one(data["prefix"]["summary"], model=m))["wrong_then_correct"],
          r["positive_wrong_then_correct"]] for m in MODELS],
        "Trajectories with at least one exact later answer after an earlier failed answer, "
        f"out of {MAIN_TRAJECTORIES} per model. The second column includes the reconstruction "
        "control; the third uses only positive lengths. Each prefix is a separate request.",
        wide=False,
    )
    add(
        "revision-error-summary",
        ["Model", "$T$", "$n$", "Median", "P90", r"$\leq5$ cells"],
        [[NAMES[m], t, (r := one(data["errors"], model=m, T=t))["valid_wrong_n"],
          num(r["median"]), num(r["p90"]), r["near5"]]
         for m in MODELS for t in (1, 16, 64)],
        "Error extent at early, intermediate, and long histories, conditional on valid "
        "incorrect main answers. Median and P90 count wrong cells in worlds of 252 or 256 "
        "cells; P90 uses linear interpolation. The last column counts predictions with at "
        "most five wrong cells. Each row has its own subset of size $n$.",
    )
    return output


def render(data: dict) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.text import Text

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.labelsize": 8.2,
            "axes.titlesize": 9,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#77838E",
            "axes.linewidth": 0.6,
            "text.color": "#263746",
            "axes.labelcolor": "#263746",
            "xtick.color": "#445460",
            "ytick.color": "#445460",
            "legend.fontsize": 8.2,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "spatial-worlds-revision-20260917",
        }
    )
    result = {}

    def save(fig, stem, description):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        frame = fig.bbox
        for text in fig.findobj(Text):
            if text.get_visible() and text.get_text():
                b = text.get_window_extent(renderer)
                assert b.x0 >= frame.x0 - 1 and b.x1 <= frame.x1 + 1, (stem, text.get_text())
                assert b.y0 >= frame.y0 - 1 and b.y1 <= frame.y1 + 1, (stem, text.get_text())
                if stem in {
                    "revision-dimensions",
                    "revision-workloads",
                    "revision-errors",
                    "revision-outcomes",
                }:
                    assert text.get_fontsize() >= 8.2, (stem, "undersized text", text.get_text())
        for extension in ("pdf", "svg", "png"):
            path = OUT / f"{stem}.{extension}"
            metadata: dict[str, str | None] = {"Creator": "Spatial Worlds offline supplement"}
            if extension == "pdf":
                metadata.update({"CreationDate": None, "ModDate": None})
            elif extension == "svg":
                metadata["Date"] = None
            fig.savefig(path, dpi=240, metadata=metadata)
            if extension == "svg":
                # Normalize Matplotlib's trailing spaces in multi-line paths.
                svg = path.read_text(encoding="utf-8")
                path.write_text(
                    "\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
                    encoding="utf-8",
                )
            result[str(path.relative_to(ROOT))] = {
                "sha256": sha(path),
                "width_inches": float(fig.get_figwidth()),
                "height_inches": float(fig.get_figheight()),
                "minimum_font_points": min(
                    text.get_fontsize()
                    for text in fig.findobj(Text)
                    if text.get_visible() and text.get_text()
                ),
                "description": description,
            }
        plt.close(fig)

    def decorate(ax, *, ymax=100):
        ax.spines[["right", "top"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#D8E0E7", linewidth=0.55)
        ax.set_ylim(-3, ymax + 3)
        ax.set_xticks(range(6), [str(t) for t in LENGTHS])
        ax.set_yticks([0, ymax / 2, ymax])
        ax.tick_params(length=2.5)

    old = read_tables()
    styles: tuple[tuple[str, Literal["-", "--", ":"]], ...] = (
        ("o", "-"), ("s", "--"), ("^", ":"),
    )

    # Main-text rule summary: one shared scale, rather than six hue-specific
    # scales, makes differences across models as readable as those across rules.
    fig = plt.figure(figsize=(6.3, 2.40))
    ax = fig.add_axes((0.205, 0.25, 0.78, 0.60))
    matrix = np.array(
        [
            [one(data["task_summary"], model=model, family=family)["correct"] for family in TASKS]
            for model in MODELS
        ]
    )
    assert all(row["n"] == 96 for row in data["task_summary"])
    cmap = LinearSegmentedColormap.from_list("rule-exactness", ["#F3F7FA", "#90B9D8", "#245C99"])
    grid = ax.imshow(matrix, vmin=0, vmax=96, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(TASKS)), TASKS.values(), fontsize=8.2)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=7)
    ax.set_yticks(range(len(MODELS)), [NAMES[m] for m in MODELS])
    ax.tick_params(axis="y", length=0, pad=7)
    for label, model in zip(ax.get_yticklabels(), MODELS, strict=True):
        label.set_color(COLORS[model])
        label.set_fontweight("bold")
    ax.set_xticks(np.arange(-0.5, 8, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 6, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, top=False, left=False, right=False)
    ax.spines[:].set_visible(False)
    for i, row in enumerate(matrix):
        for j, count in enumerate(row):
            ax.text(
                j,
                i,
                str(count),
                ha="center",
                va="center",
                color="white" if count >= 52 else "#263746",
                weight="normal",
                fontsize=9,
            )
    cax = fig.add_axes((0.40, 0.10, 0.34, 0.045))
    bar = fig.colorbar(grid, cax=cax, orientation="horizontal", ticks=[0, 24, 48, 72, 96])
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=2, pad=2)
    fig.text(0.15, 0.12, "Exact answers / 96", ha="left", va="center")
    save(
        fig,
        "revision-rule-matrix",
        "Six models by eight rules, exact counts /96 pooled over T8–64; one shared blue "
        "scale 0–96; uniformly weighted counts and no highlighted rule; all failures retained.",
    )

    # Main diagnostic: all-model extent first, then a matched-instance example.
    fig = plt.figure(figsize=(6.3, 3.30))
    left = fig.add_axes((0.205, 0.27, 0.205, 0.52))
    left.set_xscale("log", base=2)
    left.set_xlim(0.8, 256)
    left.set_ylim(5.65, -0.65)
    left.set_xticks([1, 4, 16, 64, 256], ["1", "4", "16", "64", "256"])
    left.set_yticks(range(6), [NAMES[m] for m in MODELS])
    left.minorticks_off()
    left.tick_params(axis="y", length=0, pad=7)
    left.tick_params(axis="x", length=2.5)
    left.spines[["top", "right", "left"]].set_visible(False)
    left.grid(axis="x", color="#E1E8EE", linewidth=0.55)
    left.set_axisbelow(True)
    left.set_xlabel("Median wrong cells", labelpad=6)
    extent_lengths = (1, 16, 64)
    extent_markers = ("o", "s", "^")
    for y, model in enumerate(MODELS):
        sizes = []
        for offset, length, marker in zip(
            (-0.23, 0, 0.23), extent_lengths, extent_markers, strict=True
        ):
            row = one(data["errors"], model=model, T=length)
            sizes.append(row["valid_wrong_n"])
            assert row["valid_wrong_n"] > 0 and row["median"] > 0
            left.scatter(
                row["median"], y + offset, marker=marker, s=26,
                color=COLORS[model], edgecolor="white", linewidth=0.45, zorder=3,
            )
        left.text(
            1.15, y, " / ".join(str(n) for n in sizes),
            transform=left.get_yaxis_transform(), va="center", fontsize=8.2,
        )
    for label, model in zip(left.get_yticklabels(), MODELS, strict=True):
        label.set_color(COLORS[model])
    fig.text(0.02, 0.96, "(a) Extent across histories", weight="bold", fontsize=9)
    fig.legend(
        [Line2D([], [], color="#445460", marker=m, linestyle="none") for m in extent_markers],
        [f"$T={t}$" for t in extent_lengths],
        loc="upper left", bbox_to_anchor=(0.195, 0.92), ncol=3,
        handletextpad=0.25, columnspacing=0.8, handlelength=1,
    )
    fig.text(0.445, 0.825, "$n$ by history length", fontsize=8.2)
    fig.text(0.205, 0.08, "Valid incorrect predictions", color="#576A78", fontsize=8.2)

    pair = one(
        data["matched_pairs"],
        model_a="moonshotai.kimi-k2.5",
        model_b="minimax.minimax-m2.5",
        T=8,
        subset="joint_valid_incorrect",
    )
    points = pair["paired_points"]
    assert len(points) == pair["n"] == 84
    assert [p["instance_id"] for p in points] == pair["instance_ids"]
    exact_a = one(old["length"], model=pair["model_a"], T=8)
    exact_b = one(old["length"], model=pair["model_b"], T=8)
    assert exact_a["n"] == exact_b["n"] == MAIN_TRAJECTORIES
    right = fig.add_axes((0.68, 0.28, 0.29, 0.55))
    right.set_xscale("log", base=2)
    right.set_yscale("log", base=2)
    right.set_xlim(0.8, 300)
    right.set_ylim(0.8, 300)
    right.set_aspect("equal", adjustable="box")
    ticks = [1, 4, 16, 64, 256]
    right.set_xticks(ticks, [str(t) for t in ticks])
    right.set_yticks(ticks, [str(t) for t in ticks])
    right.minorticks_off()
    right.spines[["top", "right"]].set_visible(False)
    right.grid(color="#E1E8EE", linewidth=0.5)
    right.set_axisbelow(True)
    right.plot([1, 256], [1, 256], color="#91A0AC", linestyle=(0, (3, 3)), linewidth=0.9)
    right.scatter(
        [p["wrong_cells_a"] for p in points],
        [p["wrong_cells_b"] for p in points],
        s=20,
        color="#304B63",
        alpha=0.88,
        edgecolor="white",
        linewidth=0.45,
        zorder=3,
    )
    right.set_xlabel("Kimi K2.5 wrong cells", color=COLORS["moonshotai.kimi-k2.5"], labelpad=5)
    right.set_ylabel("MiniMax M2.5 wrong cells", color=COLORS["minimax.minimax-m2.5"], labelpad=5)
    fig.text(0.655, 0.96, "(b) Shared instances", weight="bold", fontsize=9)
    fig.text(
        0.655,
        0.87,
        f"$T=8$ · {exact_a['correct']}/{exact_a['n']} vs. "
        f"{exact_b['correct']}/{exact_b['n']} EM",
        fontsize=8.2,
    )
    fig.text(
        0.62,
        0.060,
        f"{pair['n']} pairs · medians (x/y): {pair['wrong_cells_median_a']:g} / "
        f"{pair['wrong_cells_median_b']:g}",
        fontsize=8.2,
        color="#576A78",
    )
    fig.text(
        0.62,
        0.015,
        f"Kimi K2.5: {pair['a_fewer_wrong']} fewer · {pair['equal_wrong']} equal · "
        f"{pair['b_fewer_wrong']} more",
        fontsize=8.2,
        color="#576A78",
    )
    save(
        fig,
        "revision-error-contrast",
        "Left: all six models' T1/T16/T64 median wrong-cell counts on valid incorrect states, "
        "with each conditional n. Right: 84 shared T8 Kimi K2.5/MiniMax M2.5 incorrect states; "
        "log-log wrong-cell counts, equal axes, identity line; 68/3/13 pairwise ordering.",
    )

    # Appendix: retain every original task-comparison point.
    fig = plt.figure(figsize=(6.3, 2.2))
    left = fig.add_axes((0.13, 0.29, 0.83, 0.48))
    left.spines[["top", "right", "left"]].set_visible(False)
    left.set_xlim(-7, 256)
    left.set_ylim(-0.50, 1.70)
    left.set_xticks([0, 64, 128, 192, 256])
    left.set_yticks([1, 0], ["MOVE", "LAYER"])
    left.tick_params(axis="y", length=0, pad=7)
    left.tick_params(axis="x", length=2.5)
    left.set_xlabel("Wrong cells", labelpad=6)
    left.grid(axis="x", color="#E1E8EE", linewidth=0.55)
    left.set_axisbelow(True)
    for y, family in ((1, "obstructed_motion"), (0, "boundary_layer")):
        row = one(data["task_errors_T64"], model="gpt-5.6-sol", family=family)
        values = sorted(row["wrong_cells"])
        assert len(values) == row["valid_wrong_n"] == 24
        # A deterministic vertical stack separates neighboring observations;
        # only x encodes a measured quantity.
        placed = []
        for value in values:
            for offset in (
                0,
                0.055,
                -0.055,
                0.11,
                -0.11,
                0.165,
                -0.165,
                0.22,
                -0.22,
                0.275,
                -0.275,
                0.33,
                -0.33,
            ):
                if all(
                    abs(value - other) >= 3.5 or abs(offset - height) >= 0.05
                    for other, height in placed
                ):
                    placed.append((value, offset))
                    break
            else:
                raise AssertionError("Dot strip needs more vertical room")
        left.hlines(y, 0, 256, color="#E6EDF2", linewidth=0.8, zorder=0)
        left.scatter(
            [p[0] for p in placed],
            [y + p[1] for p in placed],
            s=16,
            color=COLORS["gpt-5.6-sol"],
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        median = row["median"]
        left.plot([median, median], [y - 0.37, y + 0.37], color="#304B63", linewidth=1.5, zorder=4)
        left.text(median + 7, y + 0.45, f"median {median:g}", fontsize=8.2, va="center")
    fig.text(0.13, 0.93, "GPT-5.6 Sol · $T=64$ · MOVE and LAYER", weight="bold", fontsize=9)
    fig.text(0.13, 0.06, "Each task: 0/24 exact · 24 valid incorrect predictions", fontsize=8.2)
    save(
        fig, "revision-task-error-contrast",
        "GPT-5.6 Sol T64 MOVE/LAYER: all 24 valid incorrect states per task, "
        "deterministic dot offsets, linear 0–256 axis, medians 9.5/103.",
    )


    fig = plt.figure(figsize=(6.3, 2.42))
    keys = ("changed_only", "unchanged_only", "both")
    labels = ("Update only ($C$)", "Preservation only ($U$)", "Both")
    panels = [fig.add_axes((0.205, 0.21, 0.315, 0.61)), fig.add_axes((0.655, 0.21, 0.315, 0.61))]
    for ax, length in zip(panels, (1, 64), strict=True):
        ax.set_xlim(0, 135)
        ax.set_ylim(-0.65, 5.65)
        ax.invert_yaxis()
        ax.set_yticks(range(6), [NAMES[m] for m in MODELS] if length == 1 else [""] * 6)
        ax.set_xticks([0, 50, 100], ["0", "50", "100"])
        ax.tick_params(axis="y", length=0, pad=7)
        ax.tick_params(axis="x", length=2.5, pad=3)
        ax.spines[["left", "right", "top"]].set_visible(False)
        ax.spines["bottom"].set_bounds(0, 100)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#DDE5EB", linewidth=0.6)
        ax.set_title(f"$T={length}$", loc="left", pad=6)
        for label, model in zip(ax.get_yticklabels(), MODELS, strict=True):
            label.set_color(COLORS[model])
        for y, model in enumerate(MODELS):
            row = one(data["errors"], model=model, T=length)
            n = row["valid_wrong_n"]
            assert sum(row[k] for k in keys) == n and n > 0
            start = 0.0
            for key in keys:
                count = row[key]
                width = 100 * count / n
                ax.barh(
                    y,
                    width,
                    left=start,
                    height=0.65,
                    color=ERROR_COLORS[key],
                    edgecolor="white",
                    linewidth=0.6,
                )
                if width >= 11:
                    label = "1/1" if n == 1 else str(count)
                    ax.text(
                        start + width / 2,
                        y,
                        label,
                        ha="center",
                        va="center",
                        fontsize=8.2,
                        color="white" if key == "both" else "#243C4B",
                    )
                start += width
            ax.text(103, y, f"n={n}", va="center", fontsize=8.2)
    fig.legend(
        [Patch(facecolor=ERROR_COLORS[k]) for k in keys],
        labels,
        loc="upper center",
        bbox_to_anchor=(0.54, 1.02),
        ncol=3,
        columnspacing=1.5,
        handlelength=1.1,
    )
    fig.text(0.55, 0.045, "Share of valid incorrect states (%)", ha="center")
    save(
        fig,
        "revision-error-composition",
        "T1 and T64 six-model error partitions, normalized within valid incorrect states; "
        "C-only amber, U-only teal, both dark ink; every bar labeled n; wide segments "
        "show raw counts. Different lengths have different selected masks.",
    )

    def dimension_panel(ax, model, metric):
        decorate(ax)
        for d, (marker, style) in zip((2, 3, 4), styles, strict=True):
            points = [one(old["dimension_length"], model=model, dimension=d, T=t) for t in LENGTHS]
            ax.plot(
                range(6),
                [100 * r[metric] / r["n"] for r in points],
                color=COLORS[model],
                marker=marker,
                linestyle=style,
                linewidth=1.5,
                markersize=3.5,
                markerfacecolor="white" if d == 3 else COLORS[model],
                markeredgewidth=0.7,
            )

    dimension_handles = [Line2D([], [], color="#445460", marker=m, linestyle=s) for m, s in styles]
    fig, axes = plt.subplots(4, 3, figsize=(6.3, 5.85), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.989, top=0.91, bottom=0.08, hspace=0.78, wspace=0.23)
    for j, model in enumerate(MODELS):
        block, col = divmod(j, 3)
        for metric_index, metric in enumerate(("correct", "valid")):
            ax = axes[block * 2 + metric_index, col]
            dimension_panel(ax, model, metric)
            ax.set_title(NAMES[model], color=COLORS[model], pad=5)
            if col == 0:
                ax.set_ylabel("EM (%)" if metric == "correct" else "Valid (%)")
    fig.legend(
        dimension_handles,
        ["$d=2$", "$d=3$", "$d=4$"],
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.54, 0.995),
    )
    fig.text(0.54, 0.015, "Sequence length $T$", ha="center")
    save(
        fig,
        "revision-dimensions-full",
        "Six model panels, each with EM then validity; n=64 at every dimension and length.",
    )

    fig, axes = plt.subplots(2, 3, figsize=(6.3, 3.12), sharey=True)
    fig.subplots_adjust(left=0.083, right=0.989, top=0.85, bottom=0.15, hspace=0.55, wspace=0.20)
    for index, (ax, model) in enumerate(zip(axes.flat, MODELS, strict=True)):
        dimension_panel(ax, model, "correct")
        ax.set_title(NAMES[model], color=COLORS[model], pad=5)
        if index % 3 == 0:
            ax.set_ylabel("Exact match (%)")
    fig.text(0.54, 0.025, "Sequence length $T$", ha="center")
    fig.legend(
        dimension_handles,
        ["$d=2$", "$d=3$", "$d=4$"],
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.54, 1.0),
        columnspacing=2.2,
    )
    save(
        fig,
        "revision-dimensions",
        "Six equal model panels; d=2/3/4 at all six lengths, n=64 per point; full-set EM.",
    )

    # Main workload view: four task panels, six model rows, all target levels.
    workload = pooled_workload(old["workload"])
    fig, axes = plt.subplots(1, 4, figsize=(6.3, 2.65), sharey=True)
    fig.subplots_adjust(left=0.205, right=0.983, top=0.74, bottom=0.20, wspace=0.27)
    for ax, (family, levels) in zip(axes, WORKLOADS.items(), strict=True):
        ax.set_xlim(-1.5, 33.5)
        ax.set_ylim(5.65, -0.65)
        ax.set_xticks([0, 16, 32])
        ax.set_yticks(range(6), [NAMES[m] for m in MODELS])
        ax.tick_params(axis="y", length=0, pad=7)
        ax.tick_params(axis="x", length=2.5)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#E1E8EE", linewidth=0.55)
        offsets = (-0.17, 0.17) if len(levels) == 2 else (-0.23, 0, 0.23)
        for y, model in enumerate(MODELS):
            for offset, w, (marker, _) in zip(offsets, levels, styles, strict=False):
                row = one(workload, model=model, family=family, w=w)
                ax.scatter(
                    row["correct"], y + offset, s=25, marker=marker,
                    color=COLORS[model], edgecolor="white", linewidth=0.4, zorder=3,
                )
        ax.set_title(TASKS[family], y=1.35, pad=0)
        ax.legend(
            [
                Line2D([], [], color="#445460", marker=m, linestyle="none", markersize=4)
                for m, _ in styles[:len(levels)]
            ],
            [str(w) for w in levels],
            loc="lower center", bbox_to_anchor=(0.5, 1.03), ncol=len(levels),
            handlelength=0.65, handletextpad=0.25, columnspacing=0.6, borderaxespad=0,
        )
    for label, model in zip(axes[0].get_yticklabels(), MODELS, strict=True):
        label.set_color(COLORS[model])
    fig.text(0.595, 0.055, "Exact answers / 32  ·  $T=8,16,32,64$", ha="center")
    save(
        fig, "revision-workload-summary",
        "Four task panels, six model rows, 11 workload levels; 66 exact counts /32 "
        "pooled over T8/16/32/64. Eight trajectories per level; model colors and target shapes.",
    )

    # Detailed workload curves are paginated by rule. Every page includes all
    # six models, so pagination never creates a provider or model subgroup.
    for family_index, (family, levels) in enumerate(WORKLOADS.items()):
        fig, axes = plt.subplots(2, 3, figsize=(6.3, 3.4), sharey=True)
        fig.subplots_adjust(left=0.085, right=0.99, top=0.84, bottom=0.14, hspace=0.6, wspace=0.22)
        for index, (ax, model) in enumerate(zip(axes.flat, MODELS, strict=True)):
            decorate(ax)
            for w, (marker, style) in zip(levels, styles, strict=False):
                points = [
                    one(old["workload"], model=model, family=family, w=w, T=t) for t in LENGTHS
                ]
                ax.plot(
                    range(6),
                    [exact_percent(r) for r in points],
                    color=COLORS[model],
                    marker=marker,
                    linestyle=style,
                    linewidth=1.4,
                    markersize=3.4,
                    markeredgewidth=0.6,
                )
            ax.set_title(NAMES[model], color=COLORS[model], pad=5)
            if index % 3 == 0:
                ax.set_ylabel("Exact match (%)")
        fig.legend(
            [
                Line2D([], [], color="#445460", marker=m, linestyle=s)
                for m, s in styles[: len(levels)]
            ],
            [f"$w={w}$" for w in levels],
            ncol=len(levels),
            loc="upper center",
            bbox_to_anchor=(0.54, 0.995),
        )
        fig.text(0.01, 0.945, TASKS[family], weight="bold")
        fig.text(0.54, 0.025, "Sequence length $T$", ha="center")
        stem = "revision-workloads-full" + (f"-{family_index + 1}" if family_index else "")
        save(
            fig,
            stem,
            f"{TASKS[family]} workload curves; all six models, all levels and lengths; n=8.",
        )

    # All 396 count cells remain legible in equal 2x3 panels at text width.
    fig, axes = plt.subplots(2, 3, figsize=(6.3, 4.22), sharey=True)
    fig.subplots_adjust(left=0.175, right=0.991, top=0.93, bottom=0.10, hspace=0.36, wspace=0.16)
    workload_rows = [(family, w) for family, levels in WORKLOADS.items() for w in levels]
    labels = [f"{TASKS[family]}  {w}" for family, w in workload_rows]
    for ax, model in zip(axes.flat, MODELS, strict=True):
        values = np.array(
            [
                [
                    one(old["workload"], model=model, family=family, w=w, T=t)["correct"]
                    for t in LENGTHS
                ]
                for family, w in workload_rows
            ]
        )
        assert values.shape == (11, 6) and np.all(
            (values >= 0) & (values <= CELL_TRAJECTORIES)
        )
        cmap = LinearSegmentedColormap.from_list(f"workload-{model}", ["#f4f7f9", COLORS[model]])
        ax.imshow(
            values,
            vmin=0,
            vmax=CELL_TRAJECTORIES,
            cmap=cmap,
            aspect="auto",
            interpolation="nearest",
        )
        for i in range(11):
            for j in range(6):
                value = int(values[i, j])
                ax.text(
                    j,
                    i,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if value >= 5 else "#263746",
                    fontsize=8.2,
                )
        ax.set_title(NAMES[model], color=COLORS[model], pad=7)
        ax.set_xticks(range(6), [str(t) for t in LENGTHS])
        ax.set_yticks(range(11), labels)
        ax.tick_params(length=0, pad=4)
        ax.spines[:].set_visible(False)
        for split in (1.5, 4.5, 7.5):
            ax.axhline(split, color="white", linewidth=2)
        for column in np.arange(0.5, 5.5):
            ax.axvline(column, color="white", linewidth=0.6, alpha=0.8)
    fig.text(
        0.59, 0.02,
        f"Sequence length $T$ · each cell: exact matches / {CELL_TRAJECTORIES}",
        ha="center",
    )
    save(
        fig,
        "revision-workloads",
        "All 396 model/task/workload/length counts (0-8), including controls; six equal panels; "
        "rows task and target, columns categorical lengths; model hue, darker means more correct.",
    )

    fig = plt.figure(figsize=(6.3, 3.65))
    partition = fig.add_axes((0.008, 0.28, 0.315, 0.53))
    partition.set_axis_off()
    partition.set_xlim(0, 1)
    partition.set_ylim(0, 7)
    x = (0.02, 0.45, 0.61, 0.79, 0.95)
    partition.add_patch(Rectangle((0, 6), 1, 1, color="#284B63", linewidth=0))
    for position, header in zip(x, ("", "$C$", "$U$", "Both", "$n$"), strict=True):
        partition.text(position, 6.5, header, ha="center", va="center", color="white")
    for i, model in enumerate(MODELS):
        row = one(data["errors"], model=model, T=1)
        y = 5 - i
        partition.add_patch(
            Rectangle((0, y), 1, 1, color="#EEF3F7" if i % 2 == 0 else "white", linewidth=0)
        )
        partition.text(
            x[0], y + 0.5, WRAPPED_NAMES[model], ha="left", va="center", color=COLORS[model]
        )
        for position, key in zip(
            x[1:], ("changed_only", "unchanged_only", "both", "valid_wrong_n"), strict=True
        ):
            partition.text(position, y + 0.5, str(row[key]), ha="center", va="center")
    partition.plot((0, 1), (0, 0), color="#B3C2CE", linewidth=0.7)
    fig.text(0.008, 0.945, "(a) One-step error locations", weight="bold", fontsize=8.7)
    fig.text(0.008, 0.20, "$C$ only / $U$ only / both; $T=1$", fontsize=8.2)
    fig.text(0.008, 0.14, "$n$: valid incorrect states", fontsize=8.2)
    fig.text(0.405, 0.945, "(b) Wrong-cell distributions", weight="bold", fontsize=8.7)
    for i, model in enumerate(MODELS):
        row_index, col = divmod(i, 3)
        ax = fig.add_axes((0.405 + col * 0.20, 0.545 - row_index * 0.43, 0.173, 0.225))
        rows = [one(data["errors"], model=model, T=t) for t in LENGTHS[1:]]
        available = [(k, r) for k, r in enumerate(rows) if r["wrong_cells"]]
        boxstats = [
            {
                "med": r["median"],
                "q1": r["q25"],
                "q3": r["q75"],
                "whislo": min(r["wrong_cells"]),
                "whishi": r["p90"],
                "fliers": [],
            }
            for _, r in available
        ]
        ax.bxp(
            boxstats,
            positions=[k for k, _ in available],
            widths=0.55,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": COLORS[model], "alpha": 0.35, "linewidth": 0.8},
            medianprops={"color": COLORS[model], "linewidth": 1.7},
            whiskerprops={"color": COLORS[model]},
            capprops={"color": COLORS[model]},
        )
        for k, r in available:
            tail = [value for value in r["wrong_cells"] if value > r["p90"]]
            if tail:
                ax.scatter(
                    np.full(len(tail), k),
                    tail,
                    s=6,
                    alpha=0.5,
                    color=COLORS[model],
                    edgecolors="none",
                    zorder=4,
                )
        ax.set_yscale("log")
        ax.set_ylim(0.85, 270)
        ax.set_xlim(-0.5, 4.5)
        ax.set_yticks([1, 4, 16, 64, 256], ["1", "4", "16", "64", "256"])
        ax.minorticks_off()
        ax.set_xticks(range(5), [str(t) for t in LENGTHS[1:]])
        ax.set_title(WRAPPED_NAMES[model], color=COLORS[model], pad=18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#D8E0E7", linewidth=0.5)
        for k, r in enumerate(rows):
            ax.text(k, 305, str(r["valid_wrong_n"]), ha="center", fontsize=8.2, clip_on=False)
        if col == 0:
            ax.set_ylabel("Wrong cells", labelpad=2)
        else:
            ax.set_yticklabels([])
        ax.tick_params(length=2.5, pad=2)
    fig.text(0.70, 0.025, "Sequence length $T$", ha="center")
    save(
        fig,
        "revision-errors",
        "T1 error-location counts on valid incorrect states at left; six equal model-specific "
        "wrong-cell distributions at right, log scale, median/IQR/min-P90 and tail; n annotated.",
    )

    fig, axes = plt.subplots(2, 3, figsize=(6.3, 3.0), sharey=True)
    fig.subplots_adjust(left=0.083, right=0.989, bottom=0.14, top=0.83, hspace=0.56, wspace=0.2)
    statuses = (
        ("correct", "Exact", "#284B63"),
        ("valid_wrong", "Valid, wrong", "#ABC4D5"),
        ("invalid", "Invalid, nonempty", "#75838E"),
        ("empty", "Empty", "#E4B87A"),
    )
    for i, (ax, model) in enumerate(zip(axes.flat, MODELS, strict=True)):
        decorate(ax, ymax=MAIN_TRAJECTORIES)
        bottoms = np.zeros(6)
        rows = [one(data["format_resources"], model=model, T=t) for t in LENGTHS]
        for key, _, color in statuses:
            values = np.array([r[key] for r in rows])
            ax.bar(
                range(6),
                values,
                bottom=bottoms,
                width=0.7,
                color=color,
                edgecolor="white",
                linewidth=0.35,
            )
            bottoms += values
        assert all(bottoms == MAIN_TRAJECTORIES)
        ax.set_title(NAMES[model], color=COLORS[model], pad=5)
        if i % 3 == 0:
            ax.set_ylabel("Outcomes (of 192)")
    fig.legend(
        [Patch(facecolor=c) for _, _, c in statuses],
        [s for _, s, _ in statuses],
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.53, 1.0),
        columnspacing=1.4,
        handlelength=1.3,
        handletextpad=0.5,
    )
    fig.text(0.53, 0.02, "Sequence length $T$", ha="center")
    save(
        fig,
        "revision-outcomes",
        "Six equal panels: disjoint full-denominator outcomes at every main length, n=192 per bar.",
    )

    # A portrait layout keeps the diagnostic inside one appendix column, so it
    # can travel with the analysis instead of forcing a late full-width float
    # page.  The four maps remain large enough to inspect at final page size.
    fig, axes = plt.subplots(3, 4, figsize=(3.25, 4.7))
    fig.subplots_adjust(left=0.19, right=0.985, bottom=0.065, top=0.935, hspace=0.72, wspace=0.18)
    for i, case in enumerate(data["failure_cases"]):
        values = sorted(
            {0}
            | {v for field in ("initial", "gold", "prediction") for _, v in case[field]["cells"]}
        )
        mapping = {v: k for k, v in enumerate(values)}
        for j, field in enumerate(("initial", "gold", "prediction", "errors")):
            ax = axes[i, j]
            grid = np.zeros(case["shape"], dtype=int)
            if field == "errors":
                for error in case["error_coordinates"]:
                    grid[tuple(error["coordinate"])] = 2 if error["reference_changed"] else 1
                from matplotlib.colors import ListedColormap

                cmap = ListedColormap(["#F4F6F8", "#ABC4D5", "#C77C30"])
                ax.imshow(grid, vmin=0, vmax=2, cmap=cmap, interpolation="nearest")
            else:
                for p, v in case[field]["cells"]:
                    grid[tuple(p)] = mapping[v]
                ax.imshow(
                    grid,
                    vmin=0,
                    vmax=max(1, len(values) - 1),
                    cmap="Blues",
                    interpolation="nearest",
                )
            ax.set_xticks([0, 15], ["0", "15"])
            ax.set_yticks([0, 15], ["0", "15"])
            ax.tick_params(length=1.5, labelsize=6.3, pad=1)
            if i == 0:
                ax.set_title(
                    ("Initial", "Reference", "Prediction", "Wrong cells")[j],
                    fontsize=6.8,
                    pad=3,
                )
            if not j:
                ax.set_ylabel(
                    f"{TASKS[case['family']]}\n{NAMES[case['model']]}, T={case['T']}",
                    fontsize=6.8,
                    labelpad=1,
                )
            if j == 3:
                ax.text(
                    0.98,
                    -0.31,
                    f"{case['wrong_cells']} / 256 wrong",
                    transform=ax.transAxes,
                    ha="right",
                    fontsize=6.3,
                )
        axis_box = axes[i, 0].get_position()
        fig.legend(
            [
                Patch(facecolor=plt.get_cmap("Blues")(k / max(1, len(values) - 1)))
                for k in range(len(values))
            ],
            [str(v) for v in values],
            ncol=len(values),
            loc="center",
            bbox_to_anchor=(0.56, axis_box.y0 - 0.027),
            borderaxespad=0,
            handlelength=0.65,
            handletextpad=0.18,
            columnspacing=0.42,
            fontsize=6.3,
        )
    save(
        fig,
        "revision-cases",
        "Portrait three-row diagnostic for one-column placement: deterministic illustrative "
        "2D failure cases with initial/reference/prediction/error maps and per-row shade keys.",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--tables-only", action="store_true",
                        help="Update tables and presentation inventory, preserving figure bytes.")
    args = parser.parse_args()
    assert not (args.check_only and args.tables_only)
    data = json.loads(SUPPLEMENT.read_text())
    assert data["generator_sha256"] == sha(ROOT / "paper/analyze_supplement.py")
    assert all(sha(ROOT / p) == h for p, h in data["source_sha256"].items())
    products = build_tables(data)
    source_hashes = {
        str((DATA / f"{name}.csv").relative_to(ROOT)): sha(DATA / f"{name}.csv")
        for name in ("length", "task_length", "dimension_length", "workload")
    }
    for source in (
        "paper/build_figures.py",
        "paper/model_labels.py",
        "src/spatial_worlds/eval/expanded_analysis_registry.py",
    ):
        source_hashes[source] = sha(ROOT / source)
    analysis_path = DATA / "analysis.json"
    analysis = json.loads(analysis_path.read_text())
    original = analysis["historical_analysis"]
    assert sha(ROOT / original["snapshot"]) == original["snapshot_sha256"]
    source_hashes[str(analysis_path.relative_to(ROOT))] = sha(analysis_path)
    source_hashes[original["snapshot"]] = original["snapshot_sha256"]
    generated = {
        "tables": sorted(
            [f"paper/tables/{name}" for name in products]
            + [
                f"paper/tables/{name}.tex"
                for name in (
                    "all-task-counts",
                    "all-dimension-counts",
                    "all-workload-geometry",
                    "all-workload-connectivity",
                )
            ]
        ),
        "figures": [
            f"paper/figures/{stem}.pdf"
            for stem in (
                "length-overview",
                "revision-rule-matrix",
                "revision-error-contrast",
                "revision-error-composition",
                "revision-workload-summary",
                "revision-task-error-contrast",
                "task-length",
                "revision-dimensions-full",
                "revision-workloads-full",
                "revision-workloads-full-2",
                "revision-workloads-full-3",
                "revision-workloads-full-4",
                "revision-errors",
                "revision-outcomes",
                "revision-cases",
            )
        ],
    }
    coverage = {
        kind: sorted(EXPERIMENT_ASSETS[kind] + APPENDIX_ASSETS[kind])
        for kind in ("tables", "figures")
    }
    data_only = {kind: sorted(set(generated[kind]) - set(coverage[kind])) for kind in coverage}
    assert all(set(coverage[kind]) <= set(generated[kind]) for kind in coverage)
    if args.check_only:
        for name, content in products.items():
            assert (TABLES / name).read_text() == content, f"Stale table: {name}"
        manifest = json.loads(MANIFEST.read_text())
        assert manifest["generator_sha256"] == sha(Path(__file__))
        assert manifest["supplement_sha256"] == sha(SUPPLEMENT)
        assert manifest["source_sha256"] == source_hashes
        assert manifest["coverage"] == coverage
        assert manifest["data_only"] == data_only
        assert manifest["experiment_assets"] == EXPERIMENT_ASSETS
        for name, record in manifest["outputs"].items():
            assert sha(ROOT / name) == record["sha256"], f"Changed asset: {name}"
    else:
        OUT.mkdir(exist_ok=True)
        TABLES.mkdir(exist_ok=True)
        for name, content in products.items():
            (TABLES / name).write_text(content)
        if args.tables_only:
            previous = json.loads(MANIFEST.read_text())
            assets = {p: record for p, record in previous["outputs"].items()
                      if p.startswith("paper/figures/")}
            assert all(sha(ROOT / p) == record["sha256"] for p, record in assets.items())
        else:
            with tempfile.TemporaryDirectory(prefix="spatial-worlds-revision-mpl-") as cache:
                os.environ.setdefault("MPLCONFIGDIR", cache)
                assets = render(data)
        for name in products:
            path = TABLES / name
            assets[str(path.relative_to(ROOT))] = {"sha256": sha(path)}
        MANIFEST.write_text(
            encoded(
                {
                    "generator_sha256": sha(Path(__file__)),
                    "supplement_sha256": sha(SUPPLEMENT),
                    "source_sha256": source_hashes,
                    "outputs": assets,
                    "coverage": coverage,
                    "data_only": data_only,
                    "experiment_assets": EXPERIMENT_ASSETS,
                }
            )
        )
    print(f"Verified {len(products)} expanded tables and all registered figure assets.")


if __name__ == "__main__":
    main()
