"""Build checked technical illustrations, literal prompts, and protocol tables.

Uses hand-authored examples and frozen implementation/configuration sources only.
No model calls or dataset changes. --check-only verifies content and asset hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import textwrap
from pathlib import Path
from typing import Any, SupportsIndex

from spatial_worlds.eval.extract import extract_final_answer
from spatial_worlds.generation.config import load_config
from spatial_worlds.generation.shapes import shape_stats
from spatial_worlds.grid import Grid
from spatial_worlds.prompt import COMPLETION_PROMPT_VERSION, render_prompt
from spatial_worlds.registry import get_family
from spatial_worlds.schema import Event, GoldRecord, Instance, WorldSpec
from spatial_worlds.scoring import score_prediction
from spatial_worlds.simulate import simulate
from spatial_worlds.world import WorldState

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
FIXTURES = PAPER / "technical-examples.json"
MANIFEST = PAPER / "figures/technical-manifest.json"
FAMILIES = (
    "metric_paint",
    "box_reflection",
    "local_stencil",
    "occluded_mix",
    "component_relabel",
    "boundary_layer",
    "obstructed_motion",
    "aggregating_compression",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_from_rows(world: WorldSpec, rows: list[list[int]]) -> WorldState:
    assert len(rows) == world.shape[0]
    assert all(len(row) == world.shape[1] for row in rows)
    return WorldState(
        world,
        tuple(
            ((r, c), value) for r, row in enumerate(rows) for c, value in enumerate(row) if value
        ),
    )


def observed_write_support(example: dict) -> set[tuple[int, ...]]:
    """Record actual production assignments without changing any simulator code.

    Final-state differences alone miss transit cells written and later cleared.
    This probes writes only: machine reads are not the canonical scan accounting.
    """

    class WriteRecordingList(list):
        def __init__(self, values: list[int]) -> None:
            super().__init__(values)
            self.written: set[int] = set()

        def __setitem__(self, index: SupportsIndex | slice, value: object, /) -> None:
            assert isinstance(index, int), "unexpected slice write in technical fixture"
            assert isinstance(value, int), "unexpected noninteger cell value"
            self.written.add(index)
            super().__setitem__(index, value)

    world = WorldSpec.from_dict(example["world"])
    initial = state_from_rows(world, example["initial"])
    grid = Grid.from_state(initial)
    recorded = WriteRecordingList(grid.data)
    grid.data = recorded
    get_family(example["family"]).step_grid(grid, Event.from_dict(example["event"]))
    assert grid.data is recorded, "production replaced the instrumented array"
    assert grid.to_state() == state_from_rows(world, example["expected"])
    return {grid.coord(index) for index in recorded.written}


def checked_examples() -> list[dict]:
    spec = importlib.util.spec_from_file_location("technical_oracle", ROOT / "tests/oracles.py")
    assert spec and spec.loader
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    examples = json.loads(FIXTURES.read_text())["examples"]
    assert len(examples) == 4
    for example in examples:
        world = WorldSpec.from_dict(example["world"])
        initial = state_from_rows(world, example["initial"])
        expected = state_from_rows(world, example["expected"])
        event = Event.from_dict(example["event"])
        instance = Instance(example["id"], example["family"], world, initial, (event,))
        result = simulate(instance)
        independent, scan, predicate = oracle.apply_oracle(
            example["family"], world, oracle.to_dense(initial), event
        )
        assert result.final_state == expected, example["id"]
        assert oracle.from_dense(world, independent) == expected, example["id"]
        assert scan == result.steps[0].scan == example["scan"], example["id"]
        assert result.steps[0].effect == example["effect"], example["id"]
        assert result.steps[0].predicate == predicate
        changed = {x for x in oracle.positions(world.shape) if initial[x] != expected[x]}
        selected = {tuple(x) for x in example["selected"]}
        read_only = {tuple(x) for x in example["read_only"]}
        assert changed <= selected and not selected & read_only
        assert observed_write_support(example) == selected, example["id"]
    return examples


def escape(text: str) -> str:
    special = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(special.get(char, char) for char in text)


def table(columns: str, header: str, rows: list[str], caption: str, label: str) -> str:
    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            r"\begingroup",
            r"\rowcolors{2}{TableStripe}{white}",
            r"\begin{tabularx}{\textwidth}{" + columns + "}",
            r"\tablehead",
            header + r" \\",
            r"\midrule",
            *[row + r" \\" for row in rows],
            r"\bottomrule",
            r"\end{tabularx}",
            r"\endgroup",
            r"\caption{" + caption + "}",
            r"\label{" + label + "}",
            r"\end{table*}",
            "",
        ]
    )


def geometry_table() -> str:
    rows = []
    for shape in ((16, 16), (6, 6, 7), (4, 4, 4, 4)):
        stat = shape_stats(shape)
        expected_edges = sum((length - 1) * (stat["cells"] // length) for length in shape)
        assert stat["edge_count"] == expected_edges
        rows.append(
            f"{len(shape)} & ${list(shape)}$ & {stat['cells']} & {stat['edge_count']}"
            f" & {stat['mean_neighbor_count']:.3f}"
            f" & {100 * stat['boundary_cell_fraction']:.2f}\\% & {stat['stencil_scan']}"
        )
    return table(
        "lXrrrrr",
        r"$d$ & \textbf{Shape} & $N$ & $E$ & $\overline{\deg}$"
        r" & \textbf{Boundary} & $N+2E$",
        rows,
        "Exact structural quantities for the three evaluated shapes. Boundary is the "
        "percentage of cells on the geometric world boundary. The final column is "
        "full-world neighborhood-aggregation scan, not the realized workload of the "
        "selected STENCIL or MIX events.",
        "tab:technical-geometry",
    )


def generator_table() -> str:
    config = load_config(ROOT / "configs/main.json")
    by_family = {family.family: family for family in config.families}
    rows = []
    for family_id in FAMILIES:
        family = by_family[family_id]
        op = get_family(family_id).metadata.core_op
        noop = "none" if family.noop_rate is None else f"{family.noop_rate:.1f}"
        rows.append(
            rf"\task{{{op}}} & {family.candidates} & {family.tolerance:.2f} & {noop}"
            rf" & {escape(family.set_policy)}"
        )
    return table(
        "Xrrrl",
        r"\textbf{Rule} & \textbf{Proposal argument} & $\delta$"
        r" & \textbf{No-op target} & \textbf{SET policy}",
        rows,
        "Expanded selector settings of the frozen main configuration. The proposal "
        "argument is passed to the family-specific proposer, not a guarantee of "
        "exactly that many distinct valid candidates. A no-op target of none requests "
        "effective events only. The same policies apply to additional workload levels.",
        "tab:technical-selector",
    )


def parser_cases() -> list[dict]:
    world = WorldSpec((3,), (0, 1, 2, 3))
    gold = GoldRecord(
        "parser-illustration", "metric_paint", world, WorldState(world, (((0,), 1), ((2,), 2)))
    )
    correct = '{"cells":[[[0],1],[[2],2]]}'
    cases = [
        ("Canonical complete state", correct, "none", True, True, True, True),
        ("Reordered correct cells", '{"cells":[[[2],2],[[0],1]]}', "none", True, True, True, True),
        (
            "Valid state, one wrong value",
            '{"cells":[[[0],1],[[2],3]]}',
            "none",
            True,
            True,
            False,
            False,
        ),
        (
            "One enclosing code fence",
            "```json\n" + correct + "\n```",
            "fence",
            True,
            True,
            True,
            False,
        ),
        (
            "Prose followed by correct object",
            "Answer: " + correct,
            "last_object",
            True,
            True,
            True,
            False,
        ),
        (
            "Earlier wrong object, final correct",
            '{"cells":[]} then ' + correct,
            "last_object",
            True,
            True,
            True,
            False,
        ),
        (
            "Duplicate top-level key",
            '{"cells":[],"cells":[[[0],1]]}',
            "failed",
            False,
            False,
            False,
            False,
        ),
        (
            "Duplicate coordinate",
            '{"cells":[[[0],1],[[0],1],[[2],2]]}',
            "none",
            True,
            False,
            False,
            False,
        ),
        (
            "Explicit zero entry",
            '{"cells":[[[0],1],[[1],0],[[2],2]]}',
            "none",
            True,
            False,
            False,
            False,
        ),
        (
            "Unknown extra field",
            '{"cells":[[[0],1],[[2],2]],"note":0}',
            "none",
            True,
            False,
            False,
            False,
        ),
        (
            "Correct then malformed object",
            correct + ' {"cells":}',
            "failed",
            False,
            False,
            False,
            False,
        ),
        (
            "Correct then unclosed object",
            correct + ' {"cells":',
            "failed",
            False,
            False,
            False,
            False,
        ),
        ("Empty response", "", "failed", False, False, False, False),
    ]
    output = []
    for name, raw, method, valid_json, valid_state, em, strict in cases:
        extraction = extract_final_answer(raw)
        score = score_prediction(extraction.text or "", gold)
        raw_score = score_prediction(raw, gold)
        actual = (
            extraction.method,
            score.valid_json,
            score.valid_state,
            score.exact_match,
            raw_score.exact_match,
        )
        assert actual == (method, valid_json, valid_state, em, strict), name
        output.append(
            {
                "name": name,
                "raw": raw,
                "method": method,
                "json": valid_json,
                "state": valid_state,
                "em": em,
                "strict": strict,
            }
        )
    return output


def parser_table() -> str:
    rows = [
        escape(case["name"])
        + " & "
        + escape(case["method"])
        + " & "
        + " & ".join(str(int(case[key])) for key in ("json", "state", "em", "strict"))
        for case in parser_cases()
    ]
    return table(
        "Xlrrrr",
        r"\textbf{Response form} & \textbf{Extraction} & \textbf{JSON}"
        r" & \textbf{State} & \textbf{EM} & \textbf{Strict}",
        rows,
        "Executable scoring illustrations for a three-cell world whose nonzero "
        "reference values are 1 at coordinate [0] and 2 at [2]. JSON and State indicate "
        "validity after formal extraction; EM is semantic full-world exact match; "
        "Strict applies the state scorer directly to the raw response. These are "
        "synthetic parser fixtures, not model outcomes.",
        "tab:technical-parser",
    )


def prompt_objects(examples: list[dict]) -> dict[str, str]:
    example = examples[0]
    world = WorldSpec.from_dict(example["world"])
    instance = Instance(
        example["id"],
        example["family"],
        world,
        state_from_rows(world, example["initial"]),
        (Event.from_dict(example["event"]),),
    )
    prompt = render_prompt(instance, prompt_version=COMPLETION_PROMPT_VERSION)
    markers = (
        "TASK",
        "SPACE",
        "COMMON EVENT",
        "FAMILY RULE",
        "EXAMPLE",
        "INITIAL",
        "EVENTS",
        "OUTPUT",
    )
    for marker in markers:
        assert "\n" + marker + "\n" in prompt
    template = prompt
    for begin, end, replacement in (
        ("FAMILY RULE", "EXAMPLE", "<family-specific rule card>"),
        ("EXAMPLE", "INITIAL", "<family-specific one-dimensional worked example>"),
        ("INITIAL", "EVENTS", "<complete initial state in sparse JSON>"),
        ("EVENTS", "OUTPUT", "<ordered event array in JSON>"),
    ):
        left, rest = template.split(begin + "\n", 1)
        _, right = rest.split("\n\n" + end + "\n", 1)
        template = left + begin + "\n" + replacement + "\n\n" + end + "\n" + right
    template = template.replace("shape = [3,4]", "shape = <axis lengths>")
    template = template.replace(
        "Allowed values: [0,1,2,3].", "Allowed values: <declared alphabet>."
    )
    config = load_config(ROOT / "configs/main.json")
    config_by_family = {family.family: family for family in config.families}
    cards = []
    for family_id in FAMILIES:
        family = get_family(family_id)
        card = family.rule_card(WorldSpec((3, 4), config_by_family[family_id].values))
        cards.append(r"\textbf{" + family.metadata.core_op + r".} " + escape(card) + "\n")
    answer = json.dumps(
        state_from_rows(world, example["expected"]).to_dict(), separators=(",", ":")
    )
    return {"prompt": prompt, "template": template, "cards": "\n".join(cards), "answer": answer}


def literal_panel(text: str, caption: str, label: str) -> str:
    wrapped = "\n".join(
        "\n".join(
            textwrap.wrap(
                line,
                width=87,
                replace_whitespace=False,
                drop_whitespace=False,
                break_on_hyphens=False,
            )
        )
        if line
        else ""
        for line in text.splitlines()
    )
    return "\n".join(
        [
            r"\begin{figure*}[t]",
            r"\centering",
            r"\begin{minipage}{\textwidth}",
            r"\small",
            r"\begin{verbatim}",
            wrapped,
            r"\end{verbatim}",
            r"\end{minipage}",
            r"\caption{" + caption + "}",
            r"\label{" + label + "}",
            r"\end{figure*}",
            "",
        ]
    )


def text_outputs(examples: list[dict]) -> dict[Path, str]:
    prompts = prompt_objects(examples)
    return {
        PAPER / "tables/technical-geometry.tex": geometry_table(),
        PAPER / "tables/technical-selector.tex": generator_table(),
        PAPER / "tables/technical-parser.tex": parser_table(),
        PAPER / "tables/technical-rule-cards.tex": prompts["cards"],
        PAPER / "tables/technical-worked-prompt.txt": prompts["prompt"] + "\n",
        PAPER / "tables/technical-worked-answer.txt": prompts["answer"] + "\n",
        PAPER / "tables/technical-prompt-template.tex": literal_panel(
            prompts["template"],
            "Shared prompt 0.4 template, derived from the actual "
            "renderer. Angle-bracket fields replace instance-specific content. Line "
            "wrapping is typographical; the evaluated representation remains a "
            "coordinate list, not a grid drawing.",
            "fig:technical-prompt-template",
        ),
        PAPER / "tables/technical-worked-prompt.tex": literal_panel(
            prompts["prompt"],
            "Complete renderer-produced prompt for the checked "
            "RELABEL illustration in Figure~\\ref{fig:technical-cases}. The 3-by-4 "
            "world is a synthetic explanatory fixture, not an evaluated formal "
            "instance. The included one-dimensional example is the renderer's "
            "standard rule example, not the answer to the supplied 2D world. "
            "Only printed line wrapping differs from the saved literal prompt.",
            "fig:technical-worked-prompt",
        ),
    }


def draw_examples(examples: list[dict]) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "svg.hashsalt": "spatial-worlds-technical-cases",
        }
    )
    colors = {0: "#F2F4F6", 1: "#D9E7F4", 2: "#9BBFDD", 3: "#527FAB"}
    figure, axes = plt.subplots(2, 2, figsize=(6.3, 4.55))
    figure.subplots_adjust(left=0.015, right=0.985, top=0.98, bottom=0.08, hspace=0.18, wspace=0.08)
    for index, (ax, example) in enumerate(zip(axes.flat, examples, strict=True)):
        ax.set(xlim=(0, 3), ylim=(0, 1.95))
        ax.axis("off")
        short_title = (
            "RELABEL: connected region",
            "LAYER: empty boundary",
            "STENCIL: synchronous reads",
            "MOVE: ordered, obstructed",
        )[index]
        ax.text(
            0.03,
            1.89,
            f"({chr(97 + index)}) {short_title}",
            fontsize=8.8,
            color="#233747",
            weight="bold",
            va="top",
        )
        ax.text(0.03, 1.64, example["annotation"], fontsize=7.6, color="#5A6A77")
        shape = example["world"]["shape"]
        cell = 0.24 if shape[1] == 5 else 0.28
        bottom = 0.43
        for state_index, values in enumerate((example["initial"], example["expected"])):
            left = 0.10 if state_index == 0 else 1.72
            ax.text(
                left + cell * shape[1] / 2,
                1.40,
                "Before" if state_index == 0 else "After",
                ha="center",
                fontsize=8.3,
            )
            for r, row in enumerate(values):
                for c, value in enumerate(row):
                    x, y = left + c * cell, bottom + (shape[0] - 1 - r) * cell
                    ax.add_patch(
                        Rectangle(
                            (x, y),
                            cell,
                            cell,
                            facecolor=colors[value],
                            edgecolor="white",
                            linewidth=1,
                        )
                    )
                    if state_index == 0 and [r, c] in example["selected"]:
                        ax.add_patch(
                            Rectangle(
                                (x + 0.012, y + 0.012),
                                cell - 0.024,
                                cell - 0.024,
                                fill=False,
                                edgecolor="#245C99",
                                linewidth=1.3,
                            )
                        )
                    if state_index == 0 and [r, c] in example["read_only"]:
                        ax.add_patch(
                            Rectangle(
                                (x + 0.012, y + 0.012),
                                cell - 0.024,
                                cell - 0.024,
                                fill=False,
                                edgecolor="#6A7680",
                                linewidth=1,
                                linestyle=(0, (2, 1.5)),
                            )
                        )
                    if state_index == 1 and value != example["initial"][r][c]:
                        ax.add_patch(
                            Rectangle(
                                (x + 0.018, y + 0.018),
                                cell - 0.036,
                                cell - 0.036,
                                fill=False,
                                edgecolor="#B66B24",
                                linewidth=1.3,
                            )
                        )
                    ax.text(
                        x + cell / 2,
                        y + cell / 2,
                        str(value),
                        ha="center",
                        va="center",
                        fontsize=9.2,
                        color="white" if value == 3 else "#233747",
                    )
            ax.add_patch(
                Rectangle(
                    (left, bottom),
                    cell * shape[1],
                    cell * shape[0],
                    fill=False,
                    edgecolor="#A8B7C3",
                    linewidth=0.6,
                )
            )
        ax.add_patch(
            FancyArrowPatch(
                (1.36, 0.86),
                (1.62, 0.86),
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1.1,
                color="#245C99",
            )
        )
        ax.text(
            0.03,
            0.14,
            f"Reference reads: {example['scan']}     Changed cells: {example['effect']}",
            fontsize=8.1,
            color="#233747",
        )
    figure.text(
        0.5,
        0.02,
        "Solid blue: write support    Dashed gray: other reads    Orange: actual changes",
        ha="center",
        fontsize=7.7,
        color="#233747",
    )
    outputs = []
    for extension in ("pdf", "svg", "png"):
        path = PAPER / f"figures/technical-cases.{extension}"
        kwargs: dict[str, Any] = {"dpi": 220} if extension == "png" else {}
        if extension == "pdf":
            kwargs["metadata"] = {"CreationDate": None, "ModDate": None}
        figure.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(figure)
    return outputs


def source_paths() -> list[Path]:
    return [
        Path(__file__),
        FIXTURES,
        ROOT / "tests/oracles.py",
        ROOT / "configs/main.json",
        ROOT / "configs/rq2.json",
        *[
            ROOT / "src/spatial_worlds" / name
            for name in (
                "prompt.py",
                "scoring.py",
                "schema.py",
                "world.py",
                "simulate.py",
                "grid.py",
                "family.py",
                "eval/extract.py",
                "generation/config.py",
                "generation/shapes.py",
                "generation/selection.py",
            )
        ],
        *[ROOT / f"src/spatial_worlds/families/{name}.py" for name in FAMILIES],
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    examples = checked_examples()
    outputs = text_outputs(examples)
    if args.check_only:
        manifest = json.loads(MANIFEST.read_text())
        for path, content in outputs.items():
            assert path.read_text() == content, f"stale technical asset: {path}"
        for group in ("sources", "outputs"):
            for name, expected in manifest[group].items():
                assert digest(ROOT / name) == expected, f"changed {group}: {name}"
        print(
            "Technical assets verified: 4 dual-executor examples, 13 parser cases, "
            "renderer templates/cards, configuration/geometry tables, all hashes."
        )
        return
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    figure_paths = draw_examples(examples)
    manifest = {
        "description": "Checked illustrative technical assets; no formal instances or model calls.",
        "examples": 4,
        "parser_cases": 13,
        "prompt_version": COMPLETION_PROMPT_VERSION,
        "sources": {str(path.relative_to(ROOT)): digest(path) for path in source_paths()},
        "outputs": {
            str(path.relative_to(ROOT)): digest(path) for path in [*outputs, *figure_paths]
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print("Built technical figures, tables, prompts, and hash manifest.")


if __name__ == "__main__":
    main()
