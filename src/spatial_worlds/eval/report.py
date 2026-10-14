"""Markdown summaries and optional plots from an evaluation results directory."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .aggregate import BASELINE_PROVIDERS


def _sort_key(values: Sequence[Any]) -> tuple[tuple[int, Any], ...]:
    return tuple((0, v) if isinstance(v, (int, float)) else (1, str(v)) for v in values)


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(value) for value in row) + " |")
    return "\n".join(lines)


def _mean(values: Sequence[Any]) -> float | None:
    numbers = [float(v) for v in values if v is not None]
    return sum(numbers) / len(numbers) if numbers else None


def _suffix_keys(cells: Sequence[Mapping[str, Any]]) -> list[str]:
    keys = {key for cell in cells for key in cell if key.startswith("suffix_em_k")}
    return sorted(keys, key=lambda key: int(key.rsplit("k", 1)[1]))


def _retention_section(model_cells: Sequence[Mapping[str, Any]]) -> list[str]:
    """Per (family, T): erased early events, suffix-baseline EM, and the model's EM."""

    suffix_keys = _suffix_keys(model_cells)
    has_erasure = any(c.get("mean_erasure_fraction") is not None for c in model_cells)
    if not suffix_keys and not has_erasure:
        return []
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for cell in model_cells:
        if int(cell["T"]) >= 1:
            groups[(str(cell["family"]), int(cell["T"]))].append(cell)
    rows = []
    for (family, horizon), group in sorted(groups.items(), key=lambda kv: _sort_key(kv[0])):
        rows.append(
            [
                family,
                horizon,
                len(group),
                _mean([c.get("mean_erasure_fraction") for c in group]),
                *[_mean([c.get(key) for c in group]) for key in suffix_keys],
                _mean([c["em"] for c in group]),
            ]
        )
    labels = [f"suffix-{key.rsplit('k', 1)[1]} EM" for key in suffix_keys]
    return [
        "### History retention: early-event erasure and the suffix baseline",
        "",
        "Erasure is the fraction of effective core events in the first half of the prefix "
        "whose omission leaves the state at T unchanged (simulator diagnostic); suffix-k EM "
        "is the exact-match rate of the solver that executes every SET and only the last k "
        "core events. Means are over the cells (d, w) of the family at that T.",
        "",
        _table(["family", "T", "cells", "erased fraction", *labels, "model EM"], rows),
        "",
    ]


def _workload_section(workload_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    if not workload_rows:
        return []
    groups: dict[tuple[str, Any], list[Mapping[str, Any]]] = defaultdict(list)
    for row in workload_rows:
        groups[(str(row["family"]), row["dimension"])].append(row)
    parts = [
        "### Horizon versus workload: cells ordered by cumulative core scan",
        "",
        "Cumulative core scan is the realized reference work of the prefix (SET events "
        "excluded); conditions with fewer heavier events and more lighter events can be "
        "compared at similar values.",
        "",
    ]
    for (family, dimension), group in sorted(groups.items(), key=lambda kv: _sort_key(kv[0])):
        parts.append(f"{family}, d={dimension}:")
        parts.append("")
        parts.append(
            _table(
                ["w", "T", "n", "cumulative core scan", "mean core scan", "EM", "cell acc"],
                [
                    [
                        r["w"],
                        r["T"],
                        r["n"],
                        _fmt(r.get("mean_cumulative_core_scan"), 0),
                        _fmt(r.get("mean_core_scan"), 1),
                        r["em"],
                        r.get("cell_accuracy"),
                    ]
                    for r in group
                ],
            )
        )
        parts.append("")
    return parts


def _cost_section(cost_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Observed usage and cost by horizon, then the per-family cost basis."""

    if not cost_rows or not any(int(r.get("usage_known") or 0) for r in cost_rows):
        return []
    pooled = sorted((r for r in cost_rows if r["family"] == "all"), key=lambda r: int(r["T"]))
    per_family = [r for r in cost_rows if r["family"] != "all"]
    total = sum(float(r.get("total_cost_usd") or 0.0) for r in pooled)
    known = sum(int(r.get("usage_known") or 0) for r in pooled)
    count = sum(int(r["n"]) for r in pooled)
    parts = [
        "### Cost and usage by horizon",
        "",
        "Means are over the requests that reported usage; a request that returned "
        "nothing reports none and costs nothing here, so the totals are lower bounds "
        "where stalls occurred. Cost is observed usage times the price table given to "
        "evaluate (cached input charged at the input price).",
        "",
        _table(
            [
                "T",
                "n",
                "answered",
                "usage known",
                "no-answer",
                "input tok",
                "output tok",
                "output tok p90",
                "latency s",
                "attempts",
                "$/instance",
                "total $",
            ],
            [
                [
                    r["T"],
                    r["n"],
                    r["answered"],
                    r["usage_known"],
                    r["no_answer_rate"],
                    _fmt(r.get("mean_input_tokens"), 0),
                    _fmt(r.get("mean_output_tokens"), 0),
                    _fmt(r.get("output_tokens_p90"), 0),
                    _fmt(r.get("mean_latency_s"), 0),
                    _fmt(r.get("mean_attempts"), 2),
                    _fmt(r.get("mean_cost_usd"), 3),
                    _fmt(r.get("total_cost_usd"), 2),
                ]
                for r in pooled
            ],
        ),
        "",
        f"Total observed cost: ${total:.2f} over {known} of {count} requests with usage.",
        "",
    ]
    if per_family and any(r.get("mean_cost_usd") is not None for r in per_family):
        horizons = sorted({int(r["T"]) for r in per_family})
        by_family: dict[str, dict[int, Any]] = defaultdict(dict)
        for r in per_family:
            by_family[str(r["family"])][int(r["T"])] = r
        parts.append("### Mean cost per instance by family and T (USD)")
        parts.append("")
        parts.append(
            "The usage basis of the cost projection (`spatial-worlds cost`): mean observed "
            "cost per request, with the number of requests that reported usage."
        )
        parts.append("")
        parts.append(
            _table(
                ["family", *[f"T={t}" for t in horizons]],
                [
                    [
                        family,
                        *[
                            (
                                _fmt(by_family[family][t].get("mean_cost_usd"), 3)
                                + f" ({by_family[family][t]['usage_known']})"
                            )
                            if t in by_family[family]
                            else None
                            for t in horizons
                        ],
                    ]
                    for family in sorted(by_family)
                ],
            )
        )
        parts.append("")
    return parts


HEADLINE_DIMENSIONS = (2, 3, 4)


def _pooled(cells: Sequence[Mapping[str, Any]], key: str) -> float | None:
    """Instance-weighted mean of a per-cell rate over the cells that report it."""

    pairs = [(float(c[key]), int(c["n"])) for c in cells if c.get(key) is not None]
    total = sum(n for _, n in pairs)
    return sum(v * n for v, n in pairs) / total if total else None


def _leaderboard_section(cells: Sequence[Mapping[str, Any]]) -> list[str]:
    """One row per condition: exact match at the headline horizon and its runner-up.

    The headline horizon is T = 64 when the results contain it, otherwise the
    largest horizon present; the second column is T = 32 or the next largest.
    Cells are pooled over the primary dimensions d in {2, 3, 4} (or over every
    dimension when none of those is present) and weighted by instance count.
    """

    horizons = sorted({int(c["T"]) for c in cells})
    if not horizons:
        return []
    primary = 64 if 64 in horizons else horizons[-1]
    lower = [h for h in horizons if h < primary and h >= 8]
    secondary = 32 if 32 in horizons and 32 < primary else (lower[-1] if lower else None)
    dims = sorted({int(c["dimension"]) for c in cells})
    used = [d for d in dims if d in HEADLINE_DIMENSIONS] or dims
    rows: list[list[Any]] = []
    for condition, model in sorted(
        {(str(c.get("condition_id", "legacy")), str(c["model"])) for c in cells}
    ):
        mine = [
            c
            for c in cells
            if str(c["model"]) == model
            and str(c.get("condition_id", "legacy")) == condition
            and int(c["dimension"]) in used
        ]
        at_primary = [c for c in mine if int(c["T"]) == primary]
        if not at_primary:
            continue
        at_secondary = [c for c in mine if secondary is not None and int(c["T"]) == secondary]
        rows.append(
            [
                _pooled(at_primary, "em"),
                f"`{model}`",
                condition[:12],
                _fmt(_pooled(at_primary, "em")),
                sum(int(c["n"]) for c in at_primary),
                _fmt(_pooled(at_secondary, "em")) if at_secondary else "-",
                _fmt(_pooled(at_primary, "no_answer_rate")),
                _fmt(_pooled(at_primary, "mean_cost_usd"), 3),
            ]
        )
    if not rows:
        return []
    rows.sort(key=lambda r: (-(r[0] if r[0] is not None else -1.0), r[1]))
    dims_text = "/".join(str(d) for d in used)
    return [
        "## Leaderboard",
        "",
        f"Exact match at T = {primary} pooled over d = {dims_text} and every family, "
        "weighted by instance count; every attempted failure counts as wrong. "
        + (f"T = {secondary} is the second column. " if secondary is not None else "")
        + "Baseline providers appear on the same footing.",
        "",
        _table(
            [
                "model",
                "condition",
                f"EM T={primary}",
                "n",
                f"EM T={secondary}" if secondary is not None else "EM (none)",
                f"no-answer T={primary}",
                f"$/instance T={primary}",
            ],
            [r[1:] for r in rows],
        ),
        "",
    ]


def _comparison_section(comparison: Sequence[Mapping[str, Any]]) -> list[str]:
    models = sorted({label for item in comparison for label in item["models"]})
    if len(models) < 2:
        return []
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in comparison:
        by_family[str(item["family"])].append(item)
    parts = ["## Model comparison per cell", ""]
    for family in sorted(by_family):
        parts.append(f"### {family}")
        parts.append("")
        parts.append(
            _table(
                ["d", "w", "T", *models],
                [
                    [item["dimension"], item["w"], item["T"]]
                    + [item["models"].get(model) for model in models]
                    for item in sorted(
                        by_family[family],
                        key=lambda item: _sort_key((item["dimension"], item["w"], item["T"])),
                    )
                ],
            )
        )
        parts.append("")
    return parts


def _contrast_section(contrasts: Sequence[Mapping[str, Any]]) -> list[str]:
    if not contrasts:
        return []
    by_model: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for item in contrasts:
        by_model[(str(item["model"]), str(item.get("condition_id", "legacy"))[:12])].append(item)
    parts = [
        "## Dimension contrasts",
        "",
        "EM(d) minus EM at the reference dimension for the same family, w, and T; the "
        "interval is a percentile bootstrap over rollouts of the two independent samples.",
        "",
    ]
    for (model, condition), group in sorted(by_model.items()):
        parts.append(f"### `{model}` — condition `{condition}`")
        parts.append("")
        parts.append(
            _table(
                ["family", "w", "T", "d", "n (ref, d)", "EM ref", "EM d", "difference", "95% CI"],
                [
                    [
                        item["family"],
                        item["w"],
                        item["T"],
                        f"{item['reference_dimension']} vs {item['dimension']}",
                        f"{item['n_reference']}, {item['n']}",
                        item["em_reference"],
                        item["em"],
                        item["difference"],
                        _fmt(item["difference_ci_low"]) + " to " + _fmt(item["difference_ci_high"]),
                    ]
                    for item in sorted(
                        group,
                        key=lambda item: _sort_key(
                            (item["family"], item["w"], item["T"], item["dimension"])
                        ),
                    )
                ],
            )
        )
        parts.append("")
    return parts


def render_markdown(results: Mapping[str, Any]) -> str:
    cells: list[dict[str, Any]] = list(results.get("cells", []))
    curves: list[dict[str, Any]] = list(results.get("curves", []))
    taxonomy: list[dict[str, Any]] = list(results.get("taxonomy", []))
    coverage: list[dict[str, Any]] = list(results.get("coverage", []))
    splits: list[dict[str, Any]] = list(results.get("splits", []))
    workload: list[dict[str, Any]] = list(results.get("workload", []))
    costs: list[dict[str, Any]] = list(results.get("costs", []))
    conditions = sorted(
        {(str(cell.get("condition_id", "legacy")), str(cell["model"])) for cell in cells}
    )
    parts: list[str] = ["# Grid-Bench evaluation summary", ""]
    parts.extend(_leaderboard_section(cells))
    for current_condition, model in conditions:
        parts.append(f"## Model `{model}` — condition `{current_condition[:12]}`")
        parts.append("")
        model_cells = [
            c
            for c in cells
            if str(c["model"]) == model
            and str(c.get("condition_id", "legacy")) == current_condition
        ]
        first_cell = model_cells[0]
        request = first_cell.get("request") or {}
        resource = ", ".join(
            f"{key} = {request[key]}"
            for key in ("max_tokens", "max_completion_tokens", "timeout_s", "max_attempts")
            if key in request
        )
        parts.append(
            "Provider: `"
            + str(first_cell.get("provider") or "-")
            + "`; representation: `"
            + str(first_cell.get("representation") or "cells")
            + "`"
            + (f"; resource policy: {resource}" if resource else "")
            + "."
        )
        parts.append("")
        condition_coverage = next(
            (
                item
                for item in coverage
                if str(item.get("condition_id", "legacy")) == current_condition
                and str(item.get("model")) == model
            ),
            None,
        )
        if condition_coverage is not None:
            parts.append(
                "Coverage: "
                f"{condition_coverage['predicted_instances']}/"
                f"{condition_coverage['dataset_instances']} loaded instances "
                f"({_fmt(condition_coverage['coverage'])})."
            )
            parts.append("")
        families = sorted({str(c["family"]) for c in model_cells})
        for family in families:
            family_cells = [c for c in model_cells if str(c["family"]) == family]
            horizons = sorted({int(c["T"]) for c in family_cells})
            grid: dict[tuple[Any, Any], dict[int, float]] = defaultdict(dict)
            for c in family_cells:
                grid[(c["dimension"], c["w"])][int(c["T"])] = float(c["em"])
            rows = [
                [d, w, *[grid[(d, w)].get(t) for t in horizons]]
                for (d, w) in sorted(grid, key=_sort_key)
            ]
            parts.append(f"### {family}: exact match by (d, w) and T")
            parts.append("")
            parts.append(_table(["d", "w", *[f"T={t}" for t in horizons]], rows))
            parts.append("")
            family_curves = sorted(
                (
                    c
                    for c in curves
                    if str(c["model"]) == model
                    and str(c.get("condition_id", "legacy")) == current_condition
                    and str(c["family"]) == family
                ),
                key=lambda c: _sort_key((c["dimension"], c["w"])),
            )
            if family_curves:
                parts.append(f"### {family}: scaling summaries")
                parts.append("")
                parts.append(
                    "T* (last pass) is the largest grid horizon with EM at or above the "
                    "threshold; T* (first cross) is the log-interpolated horizon at which EM "
                    "first falls below it. The first failing prefix is the first scored "
                    "prefix that was wrong, not the first internal error of one execution. "
                    "T@budget reads the token budget off the mean tokens-versus-events fit "
                    "of answered instances; the p90 variant shifts the fit up by 1.28 "
                    "residual standard deviations."
                )
                parts.append("")
                parts.append(
                    _table(
                        [
                            "d",
                            "w",
                            "rollouts",
                            "T* (last pass)",
                            "T* (first cross)",
                            "censored",
                            "eps",
                            "eps CI",
                            "eps | copy ok (n)",
                            "non-monotone",
                            "tok/event",
                            "T@budget",
                            "T@budget p90",
                            "first no-answer T",
                        ],
                        [
                            [
                                c["dimension"],
                                c["w"],
                                c["n_rollouts"],
                                c.get("t_star_last_pass"),
                                _fmt(c.get("t_star_first_cross"), 1),
                                c["t_star_censored"],
                                _fmt(c["epsilon"], 4),
                                _fmt(c["epsilon_ci_low"], 4)
                                + " to "
                                + _fmt(c["epsilon_ci_high"], 4),
                                _fmt(c.get("epsilon_conditional"), 4)
                                + f" ({c.get('epsilon_conditional_rollouts', '-')})",
                                c["non_monotone_rollouts"],
                                _fmt(c.get("tokens_per_event"), 0),
                                _fmt(c.get("t_at_budget"), 0),
                                _fmt(c.get("t_at_budget_p90"), 0),
                                c.get("t_first_no_answer"),
                            ]
                            for c in family_curves
                        ],
                    )
                )
                parts.append("")
        validity = [
            [
                c["family"],
                c["dimension"],
                c["w"],
                c["T"],
                c["n"],
                c["strict_em"],
                c["em"],
                c["valid_state_rate"],
                c["canonical_rate"],
                c["changed_cell_accuracy"],
                _fmt(c.get("wrong_cells_median"), 0) + " / " + _fmt(c.get("wrong_cells_p90"), 0),
                c["copy_baseline_em"],
                c["mean_reasoning_tokens"],
                c.get("usage_known_rate"),
                _fmt(c.get("output_tokens_p90"), 0),
                _fmt(c.get("output_tokens_p90_imputed"), 0),
                c.get("no_answer_rate"),
            ]
            for c in sorted(
                model_cells, key=lambda c: _sort_key((c["family"], c["dimension"], c["w"], c["T"]))
            )
        ]
        parts.append("### Validity, extraction, resources, and baselines per cell")
        parts.append("")
        parts.append(
            "Wrong cells are the median and 90th percentile over valid but incorrect "
            "answers. Output-token percentiles use observed usage only; the imputed "
            "variant counts each no-answer failure at the full request budget."
        )
        parts.append("")
        parts.append(
            _table(
                [
                    "family",
                    "d",
                    "w",
                    "T",
                    "n",
                    "strict EM",
                    "EM",
                    "valid",
                    "canonical",
                    "changed-cell acc",
                    "wrong cells (med / p90)",
                    "copy EM",
                    "reasoning tok",
                    "usage known",
                    "output tok p90",
                    "p90 imputed",
                    "no-answer",
                ],
                validity,
            )
        )
        parts.append("")
        parts.extend(
            _cost_section(
                [
                    r
                    for r in costs
                    if str(r["model"]) == model
                    and str(r.get("condition_id", "legacy")) == current_condition
                ]
            )
        )
        parts.extend(_retention_section(model_cells))
        parts.extend(
            _workload_section(
                [
                    w
                    for w in workload
                    if str(w["model"]) == model
                    and str(w.get("condition_id", "legacy")) == current_condition
                ]
            )
        )
        model_splits = [
            s
            for s in splits
            if str(s["model"]) == model
            and str(s.get("condition_id", "legacy")) == current_condition
        ]
        if model_splits:
            split_names = sorted({str(s["split"]) for s in model_splits})
            tiers = sorted({str(s.get("tier")) for s in model_splits})
            by_family: dict[tuple[str, str], dict[str, Any]] = defaultdict(dict)
            for s in model_splits:
                by_family[(str(s.get("tier")), str(s["family"]))][str(s["split"])] = s
            parts.append("### Mean EM by dimension grouping")
            parts.append("")
            dims_of = {
                name: next(s["dimensions"] for s in model_splits if s["split"] == name)
                for name in split_names
            }
            dims_used = {
                name: sorted(
                    {
                        int(d)
                        for s in model_splits
                        if s["split"] == name
                        for d in s.get("dimensions_used", s["dimensions"])
                    }
                )
                for name in split_names
            }
            horizons_used = sorted({int(t) for s in model_splits for t in s.get("horizons", [])})
            parts.append(
                "Descriptive groupings: "
                + "; ".join(
                    f"{name} = d in {dims_of[name]} (present: {dims_used[name]})"
                    for name in split_names
                )
                + f". Means are over the cells (d, w, T) of the grouping at T in {horizons_used}"
                + " (controls at T = 0 and T = 1 excluded); a name says which dimensions it "
                "pools and claims nothing about training data."
            )
            parts.append("")
            for tier in tiers:
                keys = sorted(k for k in by_family if k[0] == tier)
                ordered = [k for k in keys if k[1] != "all"] + [k for k in keys if k[1] == "all"]
                parts.append(f"Tier {tier}:")
                parts.append("")
                parts.append(
                    _table(
                        [
                            "family",
                            *[f"{name} EM" for name in split_names],
                            *[f"{name} cells" for name in split_names],
                        ],
                        [
                            [
                                key[1],
                                *[
                                    by_family[key].get(name, {}).get("mean_em")
                                    for name in split_names
                                ],
                                *[
                                    by_family[key].get(name, {}).get("cells")
                                    for name in split_names
                                ],
                            ]
                            for key in ordered
                        ],
                    )
                )
                parts.append("")
        model_taxonomy = [
            t
            for t in taxonomy
            if str(t["model"]) == model
            and str(t.get("condition_id", "legacy")) == current_condition
        ]
        if model_taxonomy:
            parts.append("### Mean EM by dependency-set source and update type")
            parts.append("")
            parts.append(
                _table(
                    ["support", "update", "T", "cells", "mean EM"],
                    [
                        [t["support_source"], t["update_type"], t["T"], t["cells"], t["mean_em"]]
                        for t in model_taxonomy
                    ],
                )
            )
            parts.append("")
    parts.extend(_comparison_section(list(results.get("comparison", []))))
    parts.extend(_contrast_section(list(results.get("contrasts", []))))
    return "\n".join(parts).rstrip() + "\n"


def write_markdown(results_dir: Path) -> Path:
    results = json.loads((results_dir / "results.json").read_text(encoding="utf-8"))
    path = results_dir / "summary.md"
    path.write_text(render_markdown(results), encoding="utf-8")
    return path


def _em_points(curve: Mapping[str, Any]) -> tuple[list[int], list[float]]:
    horizons = [int(t) for t in curve["horizons"]]
    values = [
        curve["em_by_T"][str(t)] if str(t) in curve["em_by_T"] else curve["em_by_T"][t]
        for t in horizons
    ]
    return horizons, values


def _safe(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)


def write_plots(results_dir: Path) -> list[Path]:
    """Per (model, family): EM versus T with baselines, EM versus cumulative work, and erasure."""

    try:
        import matplotlib  # pyright: ignore[reportMissingImports]

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
    except ImportError as error:  # pragma: no cover - depends on the optional extra
        raise RuntimeError(
            "install the 'plots' extra: pip install 'spatial-worlds[plots]'"
        ) from error
    results = json.loads((results_dir / "results.json").read_text(encoding="utf-8"))
    curves: list[dict[str, Any]] = list(results.get("curves", []))
    cells: list[dict[str, Any]] = list(results.get("cells", []))
    written: list[Path] = []
    plot_dir = results_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    def is_baseline(item: Mapping[str, Any]) -> bool:
        return str(item.get("provider")) in BASELINE_PROVIDERS

    baseline_curves: dict[tuple[str, Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for curve in curves:
        if is_baseline(curve):
            baseline_curves[(str(curve["family"]), curve["w"], curve["dimension"])].append(curve)

    # 1. EM versus T, one panel per w, one line per dimension, baselines dashed
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for curve in curves:
        if is_baseline(curve):
            continue
        groups[
            (
                str(curve.get("condition_id", "legacy")),
                str(curve["model"]),
                str(curve["family"]),
            )
        ].append(curve)
    for (current_condition, model, family), items in sorted(groups.items()):
        ws = sorted({c["w"] for c in items}, key=lambda v: (isinstance(v, str), v))
        fig, axes = plt.subplots(1, len(ws), figsize=(4.2 * len(ws), 3.4), squeeze=False)
        for axis, w in zip(axes[0], ws, strict=True):
            for curve in sorted((c for c in items if c["w"] == w), key=lambda c: c["dimension"]):
                horizons, values = _em_points(curve)
                line = axis.plot(
                    [t + 1 for t in horizons], values, marker="o", label=f"d={curve['dimension']}"
                )[0]
                for baseline in baseline_curves.get((family, w, curve["dimension"]), []):
                    b_h, b_v = _em_points(baseline)
                    axis.plot(
                        [t + 1 for t in b_h],
                        b_v,
                        linestyle="--",
                        linewidth=0.9,
                        color=line.get_color(),
                        alpha=0.6,
                        label=f"{baseline['model']} d={curve['dimension']}",
                    )
            axis.set_xscale("log", base=2)
            axis.set_ylim(-0.02, 1.02)
            axis.set_xlabel("T + 1")
            axis.set_ylabel("exact match")
            axis.set_title(f"{family}, w={w}")
            axis.legend(fontsize=6)
        fig.suptitle(f"{model} ({current_condition[:12]})")
        fig.tight_layout()
        path = plot_dir / f"{_safe(model)}__{current_condition[:12]}__{family}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)

    # 2. EM versus cumulative core scan, one panel per dimension, one line per w
    cell_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        if is_baseline(cell) or int(cell["T"]) < 1 or cell.get("mean_cumulative_core_scan") is None:
            continue
        cell_groups[
            (str(cell.get("condition_id", "legacy")), str(cell["model"]), str(cell["family"]))
        ].append(cell)
    for (current_condition, model, family), items in sorted(cell_groups.items()):
        dims = sorted({int(c["dimension"]) for c in items})
        ws = sorted({c["w"] for c in items}, key=lambda v: (isinstance(v, str), v))
        if len(ws) < 2:
            continue
        fig, axes = plt.subplots(1, len(dims), figsize=(4.2 * len(dims), 3.4), squeeze=False)
        for axis, dimension in zip(axes[0], dims, strict=True):
            for w in ws:
                points = sorted(
                    (
                        (float(c["mean_cumulative_core_scan"]), float(c["em"]))
                        for c in items
                        if int(c["dimension"]) == dimension and c["w"] == w
                    ),
                )
                if points:
                    axis.plot(
                        [x for x, _ in points], [y for _, y in points], marker="o", label=f"w={w}"
                    )
            axis.set_xscale("log", base=2)
            axis.set_ylim(-0.02, 1.02)
            axis.set_xlabel("cumulative core scan")
            axis.set_ylabel("exact match")
            axis.set_title(f"{family}, d={dimension}")
            axis.legend(fontsize=7)
        fig.suptitle(f"{model} ({current_condition[:12]}): horizon versus workload")
        fig.tight_layout()
        path = plot_dir / f"{_safe(model)}__{current_condition[:12]}__{family}__workload.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)

    # 3. erasure of early events per family at the largest horizon (a dataset property)
    erasure: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for cell in cells:
        value = cell.get("mean_erasure_fraction")
        if value is not None:
            erasure[str(cell["family"])][int(cell["T"])].append(float(value))
    if erasure:
        families = sorted(erasure)
        values = []
        for family in families:
            top = max(erasure[family])
            values.append(sum(erasure[family][top]) / len(erasure[family][top]))
        fig, axis = plt.subplots(figsize=(1.1 * len(families) + 2, 3.4))
        axis.bar(range(len(families)), values)
        axis.set_xticks(range(len(families)))
        axis.set_xticklabels(families, rotation=45, ha="right", fontsize=8)
        axis.set_ylim(0, 1.02)
        axis.set_ylabel("erased early events")
        axis.set_title("Omission census at the largest horizon")
        fig.tight_layout()
        path = plot_dir / "erasure_by_family.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)
    return written


def report_from_args(args: Any) -> int:
    path = write_markdown(args.results)
    print(f"wrote {path}")
    if args.plots:
        for plot in write_plots(args.results):
            print(f"wrote {plot}")
    return 0
