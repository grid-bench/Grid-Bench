"""Build the accepted MOVE Figure 1 from checked 2D and 3D literal examples.

Run from the project root:
    PYTHONPATH=src .venv/bin/python paper/build_overview.py

Literal fixtures are verified against the production executor and independent
dictionary oracle. All four reference states are displayed in each row.
Retained diagnostic fixtures are synthetic one-cell errors used in validation.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from itertools import combinations, product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

from spatial_worlds.schema import Event, GoldRecord, Instance, WorldSpec
from spatial_worlds.scoring import score_prediction
from spatial_worlds.simulate import simulate
from spatial_worlds.world import WorldState

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
OUTPUT = PAPER / "figures"
EXAMPLES = PAPER / "overview-examples.json"
EXAMPLE_IDS = ("figure1-move-2d", "figure1-move-3d")
WIDTH, HEIGHT = 6.3, 3.30
INK = "#263E53"
MUTED = "#657889"
BLUE = "#326796"
BLOCK = "#C5DAEA"
LINE = "#DAE2E9"
LIGHT = "#F1F4F7"
COLORS = {0: LIGHT, 1: BLUE, 2: BLOCK}
MIN_FONT = 8.2


def label(ax, x, y, text, *, size=MIN_FONT, color=INK, weight="normal", ha="left"):
    return ax.text(
        x, y, text, fontsize=size, color=color, fontweight=weight,
        ha=ha, va="center", zorder=30,
    )


def arrow(ax, start, end, *, color=MUTED, width=1.15, scale: float = 8):
    patch = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=scale,
        linewidth=width, color=color, shrinkA=0, shrinkB=0, zorder=20,
    )
    ax.add_patch(patch)


def sparse(items):
    return {tuple(point): value for point, value in items}


def checked_examples():
    spec = importlib.util.spec_from_file_location("figure_oracle", ROOT / "tests/oracles.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Independent oracle unavailable")
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    fixtures = json.loads(EXAMPLES.read_text())["examples"]
    assert len(fixtures) == 2
    assert tuple(item["id"] for item in fixtures) == EXAMPLE_IDS
    assert [item["shape"] for item in fixtures] == [[4, 4], [3, 3, 3]]
    examples = []
    for fixture in fixtures:
        world = WorldSpec.from_dict({"shape": fixture["shape"], "values": [0, 1, 2]})
        points = list(product(*(range(n) for n in world.shape)))
        initial_sparse = sparse(fixture["initial"])
        initial = WorldState(world, tuple(sorted(initial_sparse.items())))
        events = tuple(Event.from_dict(event) for event in fixture["events"])
        instance = Instance(fixture["id"], fixture["family"], world, initial, events)
        result = simulate(instance, keep_all_states=True)
        states = [{p: initial_sparse.get(p, 0) for p in points}]
        current = dict(states[0])
        for t, (event, raw_expected) in enumerate(
            zip(events, fixture["expected_states"], strict=True), 1
        ):
            expected_sparse = sparse(raw_expected)
            expected = {p: expected_sparse.get(p, 0) for p in points}
            current, scan, predicate = oracle.apply_oracle(
                fixture["family"], world, current, event,
            )
            actual = {p: result.state_at(t)[p] for p in points}
            assert current == expected == actual, (fixture["id"], t)
            assert scan == result.steps[t - 1].scan
            assert predicate == result.steps[t - 1].predicate
            states.append(expected)
        for state, expected in zip(states, fixture["expected_blue_positions"], strict=True):
            assert [p for p in points if state[p] == 1] == [tuple(expected)]
        # The final MOVE covers one cell, rather than its two-cell limit.
        assert sum(abs(a - b) for a, b in zip(
            fixture["expected_blue_positions"][2], fixture["expected_blue_positions"][3],
            strict=True,
        )) == 1
        diagnostic = dict(fixture["diagnostic"])
        point = tuple(diagnostic["coordinate"])
        predicted = dict(states[-1])
        diagnostic["reference_value"] = predicted[point]
        assert predicted[point] == diagnostic["expected_reference_value"]
        predicted[point] = diagnostic["predicted_value"]
        answer = WorldState(world, tuple((p, v) for p, v in predicted.items() if v))
        gold = GoldRecord(instance.id, instance.family, world, result.final_state)
        score = score_prediction(json.dumps(answer.to_dict()), gold)
        assert score.valid_state and score.wrong_cells == 1 and not score.exact_match
        changed = {p for p in points if states[0][p] != states[-1][p]}
        kind = "changed-cell" if point in changed else "unchanged-cell"
        assert kind == diagnostic["expected_kind"]
        diagnostic["score"] = score.to_dict()
        diagnostic["prediction"] = answer.to_dict()
        examples.append({
            "fixture": fixture, "world": world, "points": points,
            "states": states, "changed": changed, "unchanged": set(points) - changed,
            "diagnostic": diagnostic,
        })
    return examples


def flat_grid(ax, data, center, bottom, *, t):
    cell = .205
    left = center - 2 * cell
    for (row, col), value in data["states"][t].items():
        x, y = left + col * cell, bottom + (3 - row) * cell
        face = Rectangle((x, y), cell, cell, facecolor=COLORS[value],
                         edgecolor="white", linewidth=1.0, zorder=3)
        face.set_gid(f"2d-cell-w{t}-{row}-{col}-value-{value}")
        ax.add_patch(face)
        label(ax, x + cell / 2, y + cell / 2, str(value), size=9.4,
              color="white" if value == 1 else INK, ha="center").set_gid(
                  f"2d-w{t}-{row}-{col}")
    if t == 0:
        for i in range(4):
            label(ax, left - .075, bottom + (3.5 - i) * cell, str(i),
                  color=MUTED, ha="center")
            label(ax, left + (i + .5) * cell, bottom - .085, str(i),
                  color=MUTED, ha="center")


def layered_grid(ax, data, center, *, t):
    dx, dy, step, base, depth = .150, .071, .430, .260, .025

    def project(x, y, z):
        return center + (x - y) * dx, base - (x + y - 3) * dy + z * step

    for x, y in ((0, 0), (3, 0), (0, 3), (3, 3)):
        p, q = project(x, y, 0), project(x, y, 2)
        ax.plot([p[0], q[0]], [p[1], q[1]], color="#D4DEE7",
                linewidth=.55, linestyle=(0, (2, 2)), zorder=1)
    state = data["states"][t]
    for (x, y, z), value in sorted(state.items(), key=lambda item: (item[0][2],
                                                                 sum(item[0][:2]))):
        top = [project(x, y, z), project(x + 1, y, z),
               project(x + 1, y + 1, z), project(x, y + 1, z)]
        for edge, shade in (((1, 2), "#D2DCE5"), ((2, 3), "#E4EAF0")):
            p, q = [top[i] for i in edge]
            ax.add_patch(Polygon([p, q, (q[0], q[1] - depth), (p[0], p[1] - depth)],
                                 facecolor=shade, edgecolor="white", linewidth=.35,
                                 zorder=3 + z * 3))
        face = Polygon(top, facecolor=COLORS[value], edgecolor="white",
                       linewidth=.6, zorder=4 + z * 3)
        face.set_gid(f"3d-cell-w{t}-{x}-{y}-{z}-value-{value}")
        ax.add_patch(face)
        cx, cy = project(x + .5, y + .5, z)
        label(ax, cx, cy + .005, str(value), size=8.2,
              color="white" if value == 1 else INK, ha="center").set_gid(
                  f"3d-w{t}-{x}-{y}-{z}")
    if t == 0:
        for z in range(3):
            label(ax, .045, base + z * step, f"z={z}", color=MUTED)
        # Axis directions align with the exploded layers' actual projection.
        origin = (center + .78, .31)
        for end, name, offset in (((center + 1.0, .20), "x", (.06, -.06)),
                                  ((center + .56, .20), "y", (-.06, -.06)),
                                  ((center + .78, .51), "z", (.10, .0))):
            arrow(ax, origin, end, color=MUTED, width=.6, scale=4)
            label(ax, end[0] + offset[0], end[1] + offset[1], name,
                  color=MUTED, ha="center")


def figure(two, three):
    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "DejaVu Sans"], "font.size": MIN_FONT,
        "svg.fonttype": "none", "svg.hashsalt": "grid-bench-move-reference-sequence",
        "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(WIDTH, HEIGHT), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, WIDTH), ylim=(0, HEIGHT))
    ax.axis("off")
    centers = (.80, 2.36, 3.92, 5.48)

    # Shared visual language; the two rows expose the same three-event dependency.
    for y, letter, title in ((3.17, "A", "2D · 4×4"), (1.71, "B", "3D · 3×3×3")):
        ax.add_patch(Rectangle((.05, y - .12), 6.20, .24,
                               facecolor="#EDF3F8", edgecolor="none"))
        label(ax, .13, y, letter, size=9.4, color="#245C99", weight="bold")
        label(ax, .36, y, title, size=9.0, weight="bold")
    for x, value, meaning in ((2.65, 0, "empty"), (3.76, 1, "entity"), (4.88, 2, "obstacle")):
        ax.add_patch(Rectangle((x, 3.11), .12, .12, facecolor=COLORS[value],
                               edgecolor="#BDCDD8", linewidth=.5))
        label(ax, x + .06, 3.17, str(value),
              color="white" if value == 1 else INK, ha="center")
        label(ax, x + .17, 3.17, meaning, color=MUTED)
    label(ax, 6.11, 1.71, "Reference execution", color=MUTED, ha="right")

    for t, center in enumerate(centers):
        title = ("Initial $W_0$" if t == 0 else "Target $W_3$" if t == 3 else f"$W_{t}$")
        label(ax, center, 2.95, title, size=9.0, ha="center")
        label(ax, center, 1.48, title, size=9.0, ha="center")
        flat_grid(ax, two, center, 2.00, t=t)
        layered_grid(ax, three, center, t=t)

    for data, y, heading_y, detail_y, is_3d in (
        (two, 2.40, 2.70, 2.56, False), (three, .70, 1.31, 1.17, True),
    ):
        first, assignment, last = data["fixture"]["events"]
        position = "(" + ",".join(map(str, assignment["p"])) + ")"
        details = (f"+x: {first['k']} cell" if is_3d else f"right {first['k']} cell",
                   position,
                   f"+z: ≤{last['k']} cells" if is_3d else f"up ≤{last['k']} cells")
        for i, (op, detail) in enumerate(zip(("MOVE", "SET", "MOVE"), details, strict=True)):
            middle = (centers[i] + centers[i + 1]) / 2
            label(ax, middle, heading_y, op, weight="bold", color=BLUE, ha="center")
            label(ax, middle, detail_y, detail, ha="center")
            arrow(ax, (centers[i] + .52, y), (centers[i + 1] - .52, y),
                  color=INK, width=1.0, scale=7)
    return fig


def validate_render(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = fig.axes[0].texts
    for text in texts:
        box = text.get_window_extent(renderer)
        assert box.x0 >= -0.1 and box.x1 <= fig.bbox.x1 + .1, text.get_text()
        assert box.y0 >= -0.1 and box.y1 <= fig.bbox.y1 + .1, text.get_text()
        assert text.get_fontsize() >= MIN_FONT
    assert sum((patch.get_gid() or "").startswith("2d-cell-")
               for patch in fig.axes[0].patches) == 64
    assert sum((patch.get_gid() or "").startswith("3d-cell-")
               for patch in fig.axes[0].patches) == 108
    overlaps = []
    for first, second in combinations(texts, 2):
        a = first.get_window_extent(renderer)
        b = second.get_window_extent(renderer)
        if a.overlaps(b):
            overlaps.append((first.get_text(), second.get_text()))
    assert not overlaps, overlaps
    return overlaps


def main():
    two, three = checked_examples()
    fig = figure(two, three)
    overlaps = validate_render(fig)
    OUTPUT.mkdir(exist_ok=True)
    for suffix in ("svg", "pdf", "png"):
        metadata = {"Date": None} if suffix == "svg" else (
            {"CreationDate": None, "ModDate": None} if suffix == "pdf" else None
        )
        fig.savefig(OUTPUT / f"overview.{suffix}", dpi=300, metadata=metadata)
    svg_path = OUTPUT / "overview.svg"
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n"
    )
    plt.close(fig)
    source_paths = [
        EXAMPLES, Path(__file__), ROOT / "tests/oracles.py",
        *[ROOT / "src/spatial_worlds" / name for name in (
            "scoring.py", "schema.py", "world.py", "simulate.py", "grid.py",
            "family.py", "registry.py", "families/obstructed_motion.py",
        )],
    ]
    manifest = {
        "figure": "overview", "revision": 7,
        "status": "coauthor-requested reference-sequence revision",
        "kind": "complete reference execution; diagnostic fixtures retained for validation",
        "dimensions_inches": [WIDTH, HEIGHT],
        "literal_states_verified": 8,
        "transitions_verified_by_two_executors": 6,
        "synthetic_one_cell_errors_scored": 2,
        "visible_state_cells": 172,
        "design_revision": 3,
        "minimum_label_points": MIN_FONT,
        "text_bbox_intersections": overlaps,
        "example_ids": list(EXAMPLE_IDS),
        "examples_file": EXAMPLES.name,
        "examples_sha256": hashlib.sha256(EXAMPLES.read_bytes()).hexdigest(),
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_paths
        },
        "examples": [{"id": data["fixture"]["id"], "diagnostic": data["diagnostic"]}
                     for data in (two, three)],
        "assets": {f"overview.{suffix}": hashlib.sha256(
            (OUTPUT / f"overview.{suffix}").read_bytes()).hexdigest()
            for suffix in ("svg", "pdf", "png")},
    }
    (OUTPUT / "overview-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified both examples and diagnostic scores. Text intersections: {overlaps}")
    print(f"Accepted overview SVG/PDF/PNG: {OUTPUT}")


if __name__ == "__main__":
    main()
