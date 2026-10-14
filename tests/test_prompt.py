from __future__ import annotations

import json
from typing import cast

import pytest

from spatial_worlds.prompt import (
    PROMPT_VERSION,
    render_answer,
    render_grid,
    render_prompt,
    render_query_answer,
)
from spatial_worlds.registry import family_ids, get_family
from spatial_worlds.schema import Event, Instance, WorldSpec
from spatial_worlds.world import WorldState


def test_prompt_is_complete_concise_and_game_neutral(small_instance: Instance) -> None:
    prompt = render_prompt(small_instance)
    lowered = prompt.lower()

    assert PROMPT_VERSION == "spatial-worlds-prompt/0.3"
    assert "SET" in prompt
    assert "PAINT" in prompt
    assert "\nEXAMPLE\nOn shape [7]" in prompt
    assert "shape = [2,2]" in prompt
    assert "[coordinate,value]" in prompt
    assert "W(x) denotes the value of cell x immediately before the current event." in prompt
    assert "no code fence" in prompt
    assert "test-metric-paint" not in prompt
    assert "2048" not in lowered
    assert "othello" not in lowered
    assert "go-like" not in lowered
    assert "game" not in lowered


def test_prompt_template_is_byte_stable(small_instance: Instance) -> None:
    expected = (
        "TASK\n"
        "Apply all events in order and return the final world.\n"
        "\n"
        "SPACE\n"
        "shape = [2,2]\n"
        "Coordinates and axis indices are zero-based. The world is bounded and does not wrap.\n"
        "Unlisted cells have value 0. Allowed values: [0,1,2,3].\n"
        "W(x) denotes the value of cell x immediately before the current event.\n"
        "\n"
        "COMMON EVENT\n"
        "SET(p,v): set cell p to v.\n"
        "\n"
        "FAMILY RULE\n"
        + get_family("metric_paint").rule_card(small_instance.world)
        + "\n\nEXAMPLE\n"
        + PAINT_EXAMPLE
        + "\n\nINITIAL\n"
        '{"cells":[[[0,0],1],[[1,1],2]]}\n'
        "\n"
        "EVENTS\n"
        '[{"op":"SET","p":[0,1],"v":3},{"op":"PAINT","a":[0,0],"r":1,"u":1,"v":2}]\n'
        "\n"
        "OUTPUT\n"
        'Return only {"cells":[[p,v],...]}, where each entry is [coordinate,value]. '
        "Include every nonzero cell exactly once in lexicographic coordinate order. "
        "Output the JSON object only, with no code fence or other text."
    )
    assert render_prompt(small_instance) == expected


def test_every_family_rule_card_renders_and_avoids_game_names() -> None:
    for family_id in family_ids():
        family = get_family(family_id)
        values = (0, 1, 2, 4, 8) if family_id == "aggregating_compression" else (0, 1, 2, 3)
        world = WorldSpec((3, 3), values)
        card = family.rule_card(world)
        assert card.startswith(family.metadata.core_op)
        assert len(card.split(". ")) <= 6
        example = family.example_card(world)
        assert example is not None
        for banned in ("othello", "2048", "go ", "game", "puzzle", "reversi", "flood fill"):
            assert banned not in card.lower()
            assert banned not in example.lower()


PAINT_EXAMPLE = (
    "On shape [7], listing the values of cells [0] to [6] in order. Before: 1 2 0 1 3 1 0. "
    'Event: {"op":"PAINT","a":[3],"r":2,"u":1,"v":3}. After: 1 2 0 3 3 3 0. The ball is cells '
    "[1] to [5]. Among them, [3] and [5] hold 1 and become 3; the 2 at [1] and the 3 at [4] are "
    "unchanged. Cell [0] also holds 1 but lies outside the ball and is unchanged."
)

EXPECTED_EXAMPLES = {
    "metric_paint": PAINT_EXAMPLE,
    "box_reflection": (
        "On shape [6], listing the values of cells [0] to [5] in order. Before: 1 2 3 0 0 1. "
        'Event: {"op":"REFLECT","axis":0,"hi":[4],"lo":[1]}. After: 1 0 0 3 2 1. The box is '
        "cells [1] to [4]; their old values 2 3 0 0 are written back in reverse order as 0 0 3 2, "
        "so the zeros move as well. Cells [0] and [5] are outside the box and keep their values."
    ),
    "local_stencil": (
        "On shape [7], listing the values of cells [0] to [6] in order. Before: 1 2 0 1 3 0 2. "
        'Event: {"op":"STENCIL","a":[3],"r":1}. After: 1 2 3 0 0 0 2. Only cells [2], [3], and '
        "[4] lie within distance 1 of [3]. Cell [2] becomes (0+2+1) mod 4 = 3 from its own value "
        "and its neighbours [1] and [3]; cell [3] becomes (1+0+3) mod 4 = 0; cell [4] becomes "
        "(3+1+0) mod 4 = 0. Cells [1] and [5] are read as neighbours but are not changed."
    ),
    "occluded_mix": (
        "On shape [7], listing the values of cells [0] to [6] in order. Before: 2 1 1 3 0 1 2. "
        'Event: {"op":"MIX","p":[2],"r":2}. After: 1 2 3 2 0 1 2. From [2], cells [1] and [3] '
        "are reached in one step and [0] in two steps. Cell [4] is 0 and cannot be entered, so "
        "[5] and [6] are never reached although [5] is within distance 3 of [2]. Cell [0] "
        "becomes 1 + ((2+1) mod 3) = 1 from its own value and its neighbour [1]; cell [1] "
        "becomes 1 + ((1+2+1) mod 3) = 2; cell [2] becomes 1 + ((1+1+3) mod 3) = 3; cell [3] "
        "becomes 1 + ((3+1+0) mod 3) = 2, reading the 0 at [4] although [4] is not reached. A "
        "reached cell is never set to 0."
    ),
    "component_relabel": (
        "On shape [7], listing the values of cells [0] to [6] in order. Before: 1 1 0 1 2 2 3. "
        'Event: {"op":"RELABEL","p":[4],"v":3}. After: 1 1 0 1 3 3 3. The cells with old value 2 '
        "connected to [4] are [4] and [5]; both become 3. The 1 at [3] and the 3 at [6] are not "
        "part of that set, although [6] ends up adjacent to the new 3s."
    ),
    "boundary_layer": (
        "On shape [7], listing the values of cells [0] to [6] in order. Before: 0 1 1 1 2 0 1. "
        'Event: {"op":"LAYER","p":[2],"s":-1}. After: 0 0 1 1 2 0 1. The cells with value 1 '
        "connected to [2] are [1], [2], and [3]. Cell [1] has the neighbor [0] with value 0, so "
        "it becomes 0; [2] and [3] have no neighbor with value 0 and keep their value. With "
        's = +1 the event {"op":"LAYER","p":[2],"s":1} would instead set [0] to 1. The 1 at [6] '
        "is a different set and is never touched."
    ),
    "obstructed_motion": (
        "On shape [7], listing the values of cells [0] to [6] in order. Before: 1 1 0 0 2 0 1. "
        'Event: {"op":"MOVE","axis":0,"direction":1,"k":2,"v":1}. After: 0 0 1 1 2 0 1. The cells '
        "holding 1 are handled from the highest coordinate down. [6] cannot move because the next "
        "cell is outside the world; [1] moves to [3]; then [0] moves to [2] through the cell that "
        "[1] vacated. The 2 at [4] is never moved."
    ),
    "aggregating_compression": (
        "On shape [6], listing the values of cells [0] to [5] in order. Before: 1 1 2 0 1 0. "
        'Event: {"op":"COMPRESS","axis":0,"direction":-1}. After: 2 2 1 0 0 0. The nonzero values '
        "read from the destination edge are 1 1 2 1. The first pair merges into 2; that merged 2 "
        "is not merged again with the 2 that follows it; the final 1 is kept. The result 2 2 1 is "
        "written from cell [0] and the remaining cells become 0."
    ),
}


def test_worked_examples_are_computed_by_the_simulator_and_stable() -> None:
    assert set(EXPECTED_EXAMPLES) == set(family_ids())
    for family_id, expected in EXPECTED_EXAMPLES.items():
        family = get_family(family_id)
        values = (0, 1, 2, 4, 8) if family_id == "aggregating_compression" else (0, 1, 2, 3)
        assert family.example_card(WorldSpec((4, 4, 4), values)) == expected
        # the after-state in the card is exactly what the reference simulator produces
        example = family.example
        assert example is not None
        line = WorldSpec((len(example.before),), values)
        before = WorldState(line, tuple(((i,), v) for i, v in enumerate(example.before) if v))
        after = family.transition(before, example.event)
        rendered = " ".join(str(v) for v in after.dense_values())
        assert f"After: {rendered}." in expected


def test_examples_are_omitted_when_the_alphabet_cannot_express_them() -> None:
    binary = WorldSpec((3,), (0, 1))
    assert get_family("metric_paint").example_card(binary) is None
    assert get_family("local_stencil").example_card(binary) is None
    assert get_family("aggregating_compression").example_card(WorldSpec((3,), (0, 2, 4))) is None
    # the stencil example adapts its modulus to the world
    card = get_family("local_stencil").example_card(WorldSpec((3,), (0, 1, 2, 3, 4)))
    assert card is not None and "mod 5" in card and "After: 1 2 3 4 4 0 2." in card
    instance = Instance(
        id="binary",
        family="metric_paint",
        world=binary,
        initial=WorldState(binary, (((1,), 1),)),
        events=(Event("PAINT", {"a": [1], "r": 1, "u": 1, "v": 0}),),
    )
    prompt = render_prompt(instance)
    assert "EXAMPLE" not in prompt
    assert "\nFAMILY RULE\n" in prompt and "\n\nINITIAL\n" in prompt


def test_empty_initial_state_still_defines_the_cell_wire_format(
    small_instance: Instance,
) -> None:
    empty_instance = Instance(
        id="empty-initial",
        family=small_instance.family,
        world=small_instance.world,
        initial=WorldState.empty(small_instance.world),
        events=small_instance.events,
    )

    prompt = render_prompt(empty_instance)

    assert 'INITIAL\n{"cells":[]}' in prompt
    assert "each entry is [coordinate,value]" in prompt


def test_grid_representation_for_two_dimensions() -> None:
    world = WorldSpec((2, 3), (0, 1, 2, 4, 8, 16))
    instance = Instance(
        id="grid-2d",
        family="aggregating_compression",
        world=world,
        initial=WorldState(world, (((0, 1), 16), ((1, 0), 2))),
        events=(Event("COMPRESS", {"axis": 1, "direction": -1}),),
    )
    prompt = render_prompt(instance, representation="grid")
    assert "INITIAL\nGrid form of the initial world. Row index is axis 0" in prompt
    assert " 0 16  0\n 2  0  0\n" in prompt
    assert '{"cells"' not in prompt.split("INITIAL")[1].split("EVENTS")[0]
    assert render_prompt(instance) != prompt
    assert render_grid(instance.initial) == " 0 16  0\n 2  0  0"

    one_d = Instance(
        id="grid-1d",
        family="metric_paint",
        world=WorldSpec((4,), (0, 1, 2)),
        initial=WorldState(WorldSpec((4,), (0, 1, 2)), (((2,), 1),)),
        events=(),
    )
    assert "The value at column j is the value of cell [j].\n0 0 1 0\n" in render_prompt(
        one_d, representation="grid"
    )
    three_d = Instance(
        id="grid-3d",
        family="metric_paint",
        world=WorldSpec((2, 2, 2), (0, 1)),
        initial=WorldState.empty(WorldSpec((2, 2, 2), (0, 1))),
        events=(),
    )
    with pytest.raises(ValueError):
        render_prompt(three_d, representation="grid")
    with pytest.raises(ValueError):
        render_prompt(one_d, representation="ascii")


def test_query_contract_changes_only_task_and_output(small_instance: Instance) -> None:
    queried = Instance(
        id="queried",
        family=small_instance.family,
        world=small_instance.world,
        initial=small_instance.initial,
        events=small_instance.events,
        queries=((1, 1), (0, 0)),
    )
    prompt = render_prompt(queried)
    base = render_prompt(small_instance)
    assert prompt.startswith(
        "TASK\nApply all events in order and return the final value of each query cell."
    )
    assert prompt.endswith("QUERIES\n[[1,1],[0,0]]")
    assert 'Return only {"values":[[q,v],...]}' in prompt
    assert prompt.split("SPACE")[1].split("OUTPUT")[0] == base.split("SPACE")[1].split("OUTPUT")[0]
    assert queried.to_dict()["queries"] == [[1, 1], [0, 0]]
    assert Instance.from_dict(queried.to_dict()) == queried
    assert "queries" not in small_instance.to_dict()
    with pytest.raises(ValueError):
        Instance.from_dict({**queried.to_dict(), "queries": [[1, 1], [1, 1]]})
    with pytest.raises(ValueError):
        Instance.from_dict({**queried.to_dict(), "queries": [[5, 1]]})


def test_render_answer_emits_canonical_json(small_instance: Instance) -> None:
    state = WorldState(small_instance.world, (((1, 0), 3), ((0, 1), 2)))

    rendered = render_answer(state)

    assert json.loads(rendered) == {
        "cells": [
            [[0, 1], 2],
            [[1, 0], 3],
        ]
    }
    assert rendered.strip().startswith("{")
    assert rendered.strip().endswith("}")
    assert render_query_answer(state, ((1, 0), (0, 0))) == '{"values":[[[1,0],3],[[0,0],0]]}'


def test_shuffled_representation_reorders_only_the_initial_listing(
    small_instance: Instance,
) -> None:
    from spatial_worlds.prompt import shuffled_cells

    canonical = render_prompt(small_instance)
    shuffled = render_prompt(small_instance, representation="cells_shuffled")
    assert shuffled == render_prompt(small_instance, representation="cells_shuffled")
    assert "The cells below are listed in an arbitrary order." in shuffled
    assert "lexicographic coordinate order" in shuffled  # the output contract is unchanged
    assert canonical.split("INITIAL")[0] == shuffled.split("INITIAL")[0]
    assert canonical.split("EVENTS")[1] == shuffled.split("EVENTS")[1]
    listed = shuffled_cells(small_instance)
    assert sorted(listed, key=str) == sorted(small_instance.initial.to_dict()["cells"], key=str)

    world = WorldSpec((4, 4), (0, 1, 2, 3))
    cells = tuple(((i, j), 1 + (i + j) % 3) for i in range(4) for j in range(4))
    big = Instance(
        id="shuffle-test",
        family="metric_paint",
        world=world,
        initial=WorldState(world, cells),
        events=(Event("SET", {"p": [0, 0], "v": 2}),),
    )
    order = [tuple(cast(list[int], cell[0])) for cell in shuffled_cells(big)]
    assert sorted(order) == [c[0] for c in cells] and order != [c[0] for c in cells]
