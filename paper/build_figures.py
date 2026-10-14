"""Build the paper's empirical figures from frozen, offline analysis tables.

Run ``uv run --frozen --extra plots python paper/build_figures.py`` from the
repository root. No campaigns, scores, or source analysis tables are modified.
The count assertions intentionally fail if a later analysis snapshot is supplied
without reviewing the corresponding manuscript statements and captions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

from model_labels import NAMES

from spatial_worlds.eval.expanded_analysis_registry import (
    ANALYSIS_DIR,
    ANALYSIS_ID,
    CELL_TRAJECTORIES,
    COLORS,
    DIMENSION_TRAJECTORIES,
    LENGTHS,
    MAIN_CORRECT,
    MAIN_TRAJECTORIES,
    MARKERS,
    MODELS,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / ANALYSIS_DIR
OUT = ROOT / "paper/figures"
TASKS = {
    "metric_paint": "PAINT",
    "box_reflection": "REFLECT",
    "local_stencil": "STENCIL",
    "occluded_mix": "MIX",
    "component_relabel": "RELABEL",
    "boundary_layer": "LAYER",
    "obstructed_motion": "MOVE",
    "aggregating_compression": "COMPRESS",
}
TABLES = ("length", "task_length", "dimension_length", "workload")
INTEGER_FIELDS = (
    "T",
    "dimension",
    "w",
    "n",
    "correct",
    "valid",
    "valid_wrong",
    "empty",
    "length_empty",
    "other_empty",
    "length_terminated",
    "invalid_nonempty",
    "refusal",
)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_tables() -> dict[str, list[dict]]:
    tables = {}
    for name in TABLES:
        with (DATA / f"{name}.csv").open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        for row in rows:
            for field in INTEGER_FIELDS:
                if field in row:
                    row[field] = int(row[field])
            for field in ("em", "ci95_low", "ci95_high"):
                if field in row:
                    row[field] = float(row[field]) if row[field] else None
        tables[name] = rows
    return tables


def select(rows: list[dict], **conditions) -> list[dict]:
    return sorted(
        (row for row in rows if all(row[key] == value for key, value in conditions.items())),
        key=lambda row: (row["T"], row.get("w", 0), row.get("dimension", 0)),
    )


def one(rows: list[dict], **conditions) -> dict:
    result = select(rows, **conditions)
    assert len(result) == 1, (conditions, len(result))
    return result[0]


def check_tables(tables: dict[str, list[dict]]) -> None:
    assert [len(tables[name]) for name in TABLES] == [36, 288, 108, 396]
    expected_correct = MAIN_CORRECT
    expected_failures = {
        MODELS[0]: (25, 166, 0, 1),
        MODELS[1]: (14, 44, 0, 134),
        MODELS[2]: (1, 150, 32, 9),
    }
    for model in MODELS:
        points = select(tables["length"], model=model)
        assert [row["T"] for row in points] == list(LENGTHS)
        if model in expected_correct:
            assert tuple(row["correct"] for row in points) == expected_correct[model]
        assert all(row["n"] == MAIN_TRAJECTORIES for row in points)
        if points[0]["correct"] in (0, MAIN_TRAJECTORIES):
            assert points[0]["ci95_low"] is None
            assert points[0]["ci_warning"] == "degenerate_bootstrap_not_certainty"
        for row in points:
            assert row["em"] == row["correct"] / row["n"]
            if row["ci95_low"] is not None:
                assert 0 <= row["ci95_low"] <= row["em"] <= row["ci95_high"] <= 1
            parts = [row[key] for key in ("correct", "valid_wrong", "invalid_nonempty", "empty")]
            assert sum(parts) == row["n"]
            assert row["valid"] == row["correct"] + row["valid_wrong"]
            assert row["empty"] == row["length_empty"] + row["other_empty"]
            for table_name, group_n in (
                ("task_length", 24),
                ("dimension_length", DIMENSION_TRAJECTORIES),
            ):
                group = select(tables[table_name], model=model, T=row["T"])
                assert all(part["n"] == group_n for part in group)
                for metric in ("n", "correct", "valid", "empty", "length_terminated"):
                    assert sum(part[metric] for part in group) == row[metric]
        end = points[-1]
        if model in expected_failures:
            assert (
                tuple(end[key] for key in ("correct", "valid_wrong", "invalid_nonempty", "empty"))
                == expected_failures[model]
            )
        for family in TASKS:
            assert [
                row["T"] for row in select(tables["task_length"], model=model, family=family)
            ] == list(LENGTHS)
    sonnet_dimension = select(tables["dimension_length"], model=MODELS[1], T=64)
    assert [row["length_terminated"] for row in sonnet_dimension] == [32, 41, 60]
    expected_mix = {
        MODELS[0]: [(5, 8), (0, 8), (0, 8)],
        MODELS[1]: [(2, 8), (0, 0), (0, 0)],
        MODELS[2]: [(0, 8), (0, 7), (0, 5)],
    }
    for model in MODELS:
        mix = select(tables["workload"], model=model, family="occluded_mix", T=64)
        assert [row["w"] for row in mix] == [6, 12, 30]
        assert all(row["n"] == CELL_TRAJECTORIES for row in mix)
        if model in expected_mix:
            assert [(row["correct"], row["valid"]) for row in mix] == expected_mix[model]
    for name in TABLES:
        assert {row["model"] for row in tables[name]} == set(MODELS)


def draw_figures(tables: dict[str, list[dict]]) -> dict[str, dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.text import Text

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#77838E",
            "axes.linewidth": 0.55,
            "axes.labelcolor": "#263746",
            "text.color": "#263746",
            "xtick.color": "#445460",
            "ytick.color": "#445460",
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "spatial-worlds-paper-20260917",
        }
    )
    outputs = {}

    def save(fig, name: str, description: str) -> None:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        frame = fig.bbox
        for text in fig.findobj(Text):
            if not text.get_visible() or not text.get_text():
                continue
            bounds = text.get_window_extent(renderer)
            assert (
                bounds.x0 >= frame.x0 - 1
                and bounds.y0 >= frame.y0 - 1
                and bounds.x1 <= frame.x1 + 1
                and bounds.y1 <= frame.y1 + 1
            ), (name, "text outside figure boundary", text.get_text(), bounds.bounds)
        for extension in ("pdf", "svg", "png"):
            metadata = {"Creator": "Spatial Worlds offline paper figures"}
            if extension == "pdf":
                metadata.update({"CreationDate": None, "ModDate": None})
            elif extension == "svg":
                metadata["Date"] = None
            path = OUT / f"{name}.{extension}"
            # Fixed physical dimensions preserve font sizes when included at
            # column/text width. Do not use tight bounding boxes here.
            fig.savefig(path, dpi=240, metadata=metadata)
            if extension == "svg":
                # Matplotlib writes trailing spaces in multi-line path data.
                # Normalize them so regenerated assets pass repository hygiene
                # checks without changing the rendered figure.
                svg = path.read_text(encoding="utf-8")
                path.write_text(
                    "\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
                    encoding="utf-8",
                )
            outputs[str(path.relative_to(ROOT))] = {
                "sha256": file_hash(path),
                "width_inches": float(fig.get_figwidth()),
                "height_inches": float(fig.get_figheight()),
                "description": description,
            }
        plt.close(fig)

    def decorate(ax, *, length=True, ylabel="Exact match (%)") -> None:
        ax.spines[["right", "top"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#D8E0E7", linewidth=0.55)
        ax.set_ylim(-3, 104)
        ax.set_yticks([0, 50, 100])
        if length:
            ax.set_xlim(-0.13, 5.13)
            ax.set_xticks(range(6), LENGTHS)
        if ylabel:
            ax.set_ylabel(ylabel, labelpad=4)
        ax.tick_params(pad=2)

    def model_line(ax, model: str, points: list[dict], **kwargs):
        return ax.plot(
            range(6),
            [100 * row["em"] for row in points],
            label=NAMES[model],
            color=COLORS[model],
            marker=MARKERS[model],
            linewidth=1.45,
            markersize=3.3,
            markeredgewidth=0.5,
            markeredgecolor="white",
            **kwargs,
        )

    model_handles = [
        Line2D(
            [],
            [],
            color=COLORS[model],
            marker=MARKERS[model],
            linewidth=1.5,
            markersize=4,
            label=NAMES[model],
        )
        for model in MODELS
    ]

    fig, ax = plt.subplots(figsize=(3.15, 2.58))
    fig.subplots_adjust(left=0.165, bottom=0.20, right=0.975, top=0.69)
    decorate(ax)
    ax.spines[["left", "bottom"]].set_color("#B1BEC9")
    ax.grid(axis="y", color="#E3EAF0", linewidth=0.6)
    ax.set_yticks([0, 25, 50, 75, 100])
    for model in MODELS:
        points = select(tables["length"], model=model)
        model_line(ax, model, points)
        ax.fill_between(
            range(6),
            [
                100 * row["ci95_low"] if row["ci95_low"] is not None else float("nan")
                for row in points
            ],
            [
                100 * row["ci95_high"] if row["ci95_high"] is not None else float("nan")
                for row in points
            ],
            color=COLORS[model],
            alpha=0.07,
            linewidth=0,
        )
    ax.set_xlabel("Event-sequence length $T$", labelpad=3)
    fig.legend(
        # Matplotlib fills legend columns; transpose to retain row-major order.
        handles=[model_handles[i] for i in (0, 2, 4, 1, 3, 5)],
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.53, 1.0),
        handlelength=1.55,
        columnspacing=0.7,
        handletextpad=0.35,
        fontsize=8.2,
    )
    save(
        fig,
        "length-overview",
        "Main-view EM; 96 outcomes per model and length; marginal stratified "
        "trajectory-bootstrap 95% intervals, absent when degenerate.",
    )

    fig, axes = plt.subplots(2, 4, figsize=(6.3, 2.88), sharey=True)
    fig.subplots_adjust(left=0.077, bottom=0.16, right=0.992, top=0.75, hspace=0.76, wspace=0.24)
    for ax, (family, title) in zip(axes.flat, TASKS.items(), strict=True):
        decorate(ax, ylabel="")
        for model in MODELS:
            model_line(ax, model, select(tables["task_length"], model=model, family=family))
        ax.set_title(title, loc="left", pad=4, fontsize=8.6)
    fig.text(0.009, 0.5, "Exact match (%)", rotation=90, va="center", fontsize=8)
    fig.text(0.54, 0.025, "Event-sequence length $T$", ha="center", fontsize=8)
    fig.legend(
        handles=[model_handles[i] for i in (0, 3, 1, 4, 2, 5)],
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.54, 1.01),
        columnspacing=1.15,
        handlelength=1.55,
    )
    save(
        fig,
        "task-length",
        "Task-specific descriptive EM; 12 outcomes per task, model and length; "
        "categorical lengths including controls.",
    )

    # Retain the legacy supplemental filename with complete six-model coverage.
    fig, axes = plt.subplots(2, 3, figsize=(6.3, 3.15), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.14, top=0.86, hspace=0.55, wspace=0.23)
    dimension_styles = (("o", "-"), ("s", "--"), ("^", ":"))
    for ax, model in zip(axes.flat, MODELS, strict=True):
        decorate(ax, ylabel="")
        for dimension, (marker, style) in zip((2, 3, 4), dimension_styles, strict=True):
            points = select(tables["dimension_length"], model=model, dimension=dimension)
            ax.plot(
                range(6),
                [100 * row["em"] for row in points],
                color=COLORS[model],
                marker=marker,
                linestyle=style,
                linewidth=1.4,
                markersize=3.1,
            )
        ax.set_title(NAMES[model], color=COLORS[model])
    fig.text(0.008, 0.51, "Exact match (%)", rotation=90, va="center")
    fig.text(0.54, 0.02, "Sequence length $T$", ha="center")
    fig.legend(
        [Line2D([], [], color="#445460", marker=m, linestyle=s) for m, s in dimension_styles],
        ["$d=2$", "$d=3$", "$d=4$"],
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.54, 1.0),
    )
    save(fig, "structure-load", "Six equal dimension panels; n=64 per dimension and length.")

    fig, ax = plt.subplots(figsize=(3.15, 3.2))
    fig.subplots_adjust(left=0.39, bottom=0.16, right=0.96, top=0.79)
    status = (
        ("correct", "Exact", "#284B63"),
        ("valid_wrong", "Valid, wrong", "#ABC4D5"),
        ("invalid_nonempty", "Invalid, nonempty", "#75838E"),
        ("empty", "Empty", "#E4B87A"),
    )
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#D8E0E7", linewidth=0.55)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_yticks(range(len(MODELS)), [NAMES[model] for model in MODELS])
    ax.tick_params(axis="y", length=0, pad=5)
    for label, model in zip(ax.get_yticklabels(), MODELS, strict=True):
        label.set_color(COLORS[model])
        label.set_weight("bold")
    ax.set_xlim(0, MAIN_TRAJECTORIES)
    ax.set_ylim(len(MODELS) - 0.5, -0.62)
    ax.set_xticks([0, 48, 96, 144, 192])
    ax.set_xlabel("Outcomes at $T=64$ (of 192)", labelpad=3)
    for row_index, model in enumerate(MODELS):
        row = one(tables["length"], model=model, T=64)
        left = 0
        for key, _label, color in status:
            count = row[key]
            if count:
                ax.barh(
                    row_index,
                    count,
                    left=left,
                    height=0.56,
                    color=color,
                    edgecolor="white",
                    linewidth=0.5,
                    zorder=3,
                )
                text_color = "white" if key in ("correct", "invalid_nonempty") else "#263746"
                if count >= 5:
                    ax.text(
                        left + count / 2,
                        row_index,
                        str(count),
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=8.1,
                        zorder=4,
                    )
                else:
                    ax.annotate(
                        str(count),
                        xy=(left + count / 2, row_index - 0.23),
                        xytext=(left + count / 2, row_index - 0.43),
                        ha="center",
                        va="center",
                        fontsize=8,
                        arrowprops={"arrowstyle": "-", "lw": 0.6, "color": "#667B89"},
                        annotation_clip=False,
                    )
            left += count
    fig.legend(
        handles=[Patch(facecolor=color, label=label) for _, label, color in status],
        loc="upper center",
        bbox_to_anchor=(0.53, 1.0),
        ncol=2,
        handlelength=1.3,
        columnspacing=1.0,
        handletextpad=0.5,
        labelspacing=0.4,
    )
    save(
        fig,
        "failures",
        "Disjoint T=64 answer outcomes, n=192 per model. Empty is not synonymous "
        "with length termination; the full token and termination profiles are separate.",
    )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only", action="store_true", help="Validate source counts without rendering."
    )
    args = parser.parse_args()
    before = {
        str((DATA / f"{name}.csv").relative_to(ROOT)): file_hash(DATA / f"{name}.csv")
        for name in TABLES
    }
    for source in (
        "paper/build_figures.py",
        "paper/model_labels.py",
        "src/spatial_worlds/eval/expanded_analysis_registry.py",
    ):
        before[source] = file_hash(ROOT / source)
    tables = read_tables()
    check_tables(tables)
    if args.check_only:
        print("Empirical-figure input checks passed: all counts and partitions match.")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    # Keep matplotlib's cache in a task-owned temporary directory, never under
    # the user's home or a tracked source path.
    with tempfile.TemporaryDirectory(prefix="spatial-worlds-paper-mpl-") as cache:
        os.environ.setdefault("MPLCONFIGDIR", cache)
        outputs = draw_figures(tables)
    after = {path: file_hash(ROOT / path) for path in before}
    assert after == before, "A source table changed while rendering; inspect concurrent work."
    manifest = {
        "analysis_snapshot": ANALYSIS_ID,
        "generator": "paper/build_figures.py",
        "source_sha256": before,
        "outputs": outputs,
        "uncertainty": {
            "figure": "length-overview",
            "source": "length.csv ci95_low / ci95_high",
            "method": "marginal stratified trajectory-percentile bootstrap",
            "resamples": 2000,
            "seed": 0,
            "strata": "task, dimension, and workload",
            "degenerate_intervals": "not plotted; absence does not imply certainty",
        },
        "note": "Source campaigns and derived input tables are read-only; no model calls.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"Validated source counts and rendered {len(outputs)} empirical assets in paper/figures/."
    )


if __name__ == "__main__":
    main()
