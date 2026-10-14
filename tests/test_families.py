"""Hand-computed cases for every family simulator (1D, 2D, and 3D)."""

from __future__ import annotations

import pytest

from grid_bench.registry import get_family
from grid_bench.schema import Event, WorldSpec
from grid_bench.world import WorldState


def line(values: list[int], alphabet: tuple[int, ...] = (0, 1, 2, 3)) -> WorldState:
    world = WorldSpec((len(values),), alphabet)
    return WorldState(world, tuple(((i,), v) for i, v in enumerate(values) if v))


def grid(
    shape: tuple[int, ...], cells: dict[tuple[int, ...], int], alphabet=(0, 1, 2, 3)
) -> WorldState:
    return WorldState(WorldSpec(shape, alphabet), tuple(cells.items()))


def run(family_id: str, state: WorldState, event: Event):
    transition = get_family(family_id).step(state, event)
    return (
        list(transition.after.dense_values()),
        transition.workload.scan,
        transition.workload.effect,
        transition.predicate,
    )


ONE_D_CASES = [
    (
        "metric_paint",
        [1, 2, 1, 0, 1, 3, 1],
        Event("PAINT", {"a": [3], "r": 2, "u": 1, "v": 3}),
        [1, 2, 3, 0, 3, 3, 1],
        5,
        2,
        True,
    ),
    (
        "metric_paint",
        [1, 2, 1, 0, 1, 3, 1],
        Event("PAINT", {"a": [0], "r": 2, "u": 1, "v": 2}),
        [2, 2, 2, 0, 1, 3, 1],
        3,
        2,
        True,
    ),
    (
        "metric_paint",
        [1, 2, 1],
        Event("PAINT", {"a": [1], "r": 0, "u": 1, "v": 2}),
        [1, 2, 1],
        1,
        0,
        False,
    ),
    (
        "box_reflection",
        [1, 2, 3, 0, 0, 1],
        Event("REFLECT", {"lo": [1], "hi": [4], "axis": 0}),
        [1, 0, 0, 3, 2, 1],
        4,
        4,
        None,
    ),
    (
        "box_reflection",
        [1, 2, 3, 0, 0, 1],
        Event("REFLECT", {"lo": [2], "hi": [2], "axis": 0}),
        [1, 2, 3, 0, 0, 1],
        1,
        0,
        None,
    ),
    (
        # the worked example: [0]..[3] are reached, [4] = 0 blocks, sums mod 3 plus 1
        "occluded_mix",
        [2, 1, 1, 3, 0, 1, 2],
        Event("MIX", {"p": [2], "r": 2}),
        [1, 2, 3, 2, 0, 1, 2],
        11,
        4,
        None,
    ),
    (
        # the region is the anchor alone: 1 + ((1 + 0) mod 3) = 2, one neighbour read
        "occluded_mix",
        [1, 0, 1, 1],
        Event("MIX", {"p": [0], "r": 3}),
        [2, 0, 1, 1],
        2,
        1,
        None,
    ),
    (
        # a full read with no change: (2 + 1 + 1) mod 3 = 1, so [1] stays 2
        "occluded_mix",
        [1, 2, 1],
        Event("MIX", {"p": [1], "r": 0}),
        [1, 2, 1],
        3,
        0,
        None,
    ),
    (
        "occluded_mix",
        [0, 1, 1],
        Event("MIX", {"p": [0], "r": 2}),
        [0, 1, 1],
        1,
        0,
        None,
    ),
    (
        # r = 0 recomputes only p from both neighbours: 1 + ((3 + 2 + 1) mod 3) = 1
        "occluded_mix",
        [1, 2, 3, 1, 1],
        Event("MIX", {"p": [2], "r": 0}),
        [1, 2, 1, 1, 1],
        3,
        1,
        None,
    ),
    (
        # the run [3]..[5] is reached entirely; [3] reads the 0 at [2]
        "occluded_mix",
        [1, 1, 0, 1, 1, 1],
        Event("MIX", {"p": [4], "r": 5}),
        [1, 1, 0, 3, 1, 3],
        8,
        2,
        None,
    ),
    (
        "component_relabel",
        [1, 1, 0, 1, 2, 2],
        Event("RELABEL", {"p": [0], "v": 2}),
        [2, 2, 0, 1, 2, 2],
        3,
        2,
        None,
    ),
    (
        "component_relabel",
        [1, 1, 0, 1, 2, 2],
        Event("RELABEL", {"p": [4], "v": 2}),
        [1, 1, 0, 1, 2, 2],
        3,
        0,
        None,
    ),
    (
        "component_relabel",
        [1, 0, 0, 1],
        Event("RELABEL", {"p": [1], "v": 2}),
        [1, 2, 2, 1],
        4,
        2,
        None,
    ),
    ("boundary_layer", [1, 1, 0, 2], Event("LAYER", {"p": [0], "s": -1}), [1, 0, 0, 2], 3, 1, None),
    ("boundary_layer", [1, 1, 0, 2], Event("LAYER", {"p": [1], "s": 1}), [1, 1, 1, 2], 3, 1, None),
    ("boundary_layer", [0, 1, 0], Event("LAYER", {"p": [0], "s": -1}), [0, 1, 0], 1, 0, None),
    ("boundary_layer", [1, 1, 1], Event("LAYER", {"p": [1], "s": -1}), [1, 1, 1], 3, 0, None),
    ("boundary_layer", [2, 1, 1, 3], Event("LAYER", {"p": [1], "s": 1}), [2, 1, 1, 3], 4, 0, None),
    (
        "boundary_layer",
        [0, 1, 1, 1, 0],
        Event("LAYER", {"p": [2], "s": -1}),
        [0, 0, 1, 0, 0],
        5,
        2,
        None,
    ),
    (
        "boundary_layer",
        [0, 1, 1, 1, 0],
        Event("LAYER", {"p": [2], "s": 1}),
        [1, 1, 1, 1, 1],
        5,
        2,
        None,
    ),
    (
        "obstructed_motion",
        [1, 1, 0, 0, 2, 0],
        Event("MOVE", {"v": 1, "axis": 0, "direction": 1, "k": 2}),
        [0, 0, 1, 1, 2, 0],
        6,
        4,
        None,
    ),
    (
        "obstructed_motion",
        [1, 1, 0, 2, 0, 0],
        Event("MOVE", {"v": 1, "axis": 0, "direction": 1, "k": 3}),
        [0, 1, 1, 2, 0, 0],
        6,
        2,
        None,
    ),
    (
        "obstructed_motion",
        [0, 0, 1],
        Event("MOVE", {"v": 1, "axis": 0, "direction": 1, "k": 3}),
        [0, 0, 1],
        1,
        0,
        None,
    ),
    (
        "obstructed_motion",
        [0, 2, 1, 1],
        Event("MOVE", {"v": 1, "axis": 0, "direction": -1, "k": 5}),
        [0, 2, 1, 1],
        4,
        0,
        None,
    ),
    (
        "obstructed_motion",
        [1, 1, 0, 0],
        Event("MOVE", {"v": 1, "axis": 0, "direction": 1, "k": 1}),
        [0, 1, 1, 0],
        4,
        2,
        None,
    ),
    (
        "obstructed_motion",
        [1, 1, 0, 0],
        Event("MOVE", {"v": 1, "axis": 0, "direction": 1, "k": 0}),
        [1, 1, 0, 0],
        2,
        0,
        None,
    ),
]


@pytest.mark.parametrize("family_id,before,event,after,scan,effect,predicate", ONE_D_CASES)
def test_one_dimensional_hand_cases(family_id, before, event, after, scan, effect, predicate):
    assert run(family_id, line(before), event) == (after, scan, effect, predicate)


STENCIL_CASES = [
    ([1, 2, 0, 1], (0, 1, 2), {"a": [0], "r": 3}, [0, 0, 0, 1], 10, 2),
    ([1, 2, 0, 1], (0, 1, 2), {"a": [1], "r": 1}, [0, 0, 0, 1], 8, 2),
    ([1, 2, 0, 1], (0, 1, 2), {"a": [3], "r": 0}, [1, 2, 0, 1], 2, 0),
    ([1, 2, 0, 1, 3, 0, 2], (0, 1, 2, 3), {"a": [3], "r": 1}, [1, 2, 3, 0, 0, 0, 2], 9, 3),
]


@pytest.mark.parametrize("before,alphabet,params,after,scan,effect", STENCIL_CASES)
def test_stencil_one_dimensional(before, alphabet, params, after, scan, effect):
    assert run("local_stencil", line(before, alphabet), Event("STENCIL", params)) == (
        after,
        scan,
        effect,
        None,
    )


COMPRESS_CASES = [
    ([2, 2, 2, 0], -1, [4, 2, 0, 0]),
    ([2, 2, 2, 0], 1, [0, 0, 2, 4]),
    ([2, 2, 4, 4], -1, [4, 8, 0, 0]),
    ([2, 2, 4, 4], 1, [0, 0, 4, 8]),
    ([4, 2, 2, 0], -1, [4, 4, 0, 0]),
    ([4, 2, 2, 0], 1, [0, 0, 4, 4]),
    ([2, 0, 2, 0], -1, [4, 0, 0, 0]),
    ([2, 4, 2, 0], -1, [2, 4, 2, 0]),
    ([0, 8, 0, 8], 1, [0, 0, 0, 16]),
]


@pytest.mark.parametrize("before,direction,after", COMPRESS_CASES)
def test_compress_one_dimensional(before, direction, after):
    alphabet = (0, 2, 4, 8, 16)
    result = run(
        "aggregating_compression",
        line(before, alphabet),
        Event("COMPRESS", {"axis": 0, "direction": direction}),
    )
    assert result[0] == after
    assert result[1] == 4


def test_compress_overflow_raises_schema_error():
    with pytest.raises(ValueError, match="outside world.values"):
        run(
            "aggregating_compression",
            line([4, 4, 0], (0, 2, 4)),
            Event("COMPRESS", {"axis": 0, "direction": -1}),
        )


def test_two_dimensional_cases():
    # PAINT: full example rollout from examples/metric_paint.json, first event.
    s = grid((3, 3), {(0, 0): 1, (0, 1): 1, (1, 1): 2, (2, 2): 3})
    after, scan, effect, predicate = run(
        "metric_paint", s, Event("PAINT", {"a": [0, 0], "r": 1, "u": 1, "v": 2})
    )
    assert (scan, effect, predicate) == (3, 2, True)
    assert after == list(grid((3, 3), {(0, 0): 2, (0, 1): 2, (1, 1): 2, (2, 2): 3}).dense_values())

    # REFLECT along axis 1 of a [3,4] world.
    s = grid((3, 4), {(0, 0): 1, (0, 3): 2, (1, 1): 3, (2, 1): 1})
    after, scan, effect, _ = run(
        "box_reflection", s, Event("REFLECT", {"lo": [0, 0], "hi": [1, 3], "axis": 1})
    )
    assert (scan, effect) == (8, 4)
    assert after == list(grid((3, 4), {(0, 0): 2, (0, 3): 1, (1, 2): 3, (2, 1): 1}).dense_values())
    after, scan, effect, _ = run(
        "box_reflection", s, Event("REFLECT", {"lo": [0, 1], "hi": [2, 1], "axis": 0})
    )
    assert (scan, effect) == (3, 2)
    assert after == list(grid((3, 4), {(0, 0): 1, (0, 1): 1, (0, 3): 2, (1, 1): 3}).dense_values())

    # STENCIL: [2,2] mod 2 diagonal ones is a fixed point; [3,3] mod 4 spreads a plus,
    # whether the whole world or only the plus-shaped ball around the centre is updated.
    s = grid((2, 2), {(0, 0): 1, (1, 1): 1}, (0, 1))
    whole = Event("STENCIL", {"a": [0, 0], "r": 2})
    assert run("local_stencil", s, whole) == ([1, 0, 0, 1], 12, 0, None)
    s = grid((3, 3), {(1, 1): 1})
    after, scan, effect, _ = run("local_stencil", s, Event("STENCIL", {"a": [0, 0], "r": 4}))
    assert (scan, effect) == (33, 4)
    assert after == [0, 1, 0, 1, 1, 1, 0, 1, 0]
    after, scan, effect, _ = run("local_stencil", s, Event("STENCIL", {"a": [1, 1], "r": 1}))
    assert (scan, effect) == (21, 4)
    assert after == [0, 1, 0, 1, 1, 1, 0, 1, 0]
    after, scan, effect, _ = run("local_stencil", s, Event("STENCIL", {"a": [0, 0], "r": 1}))
    assert (scan, effect) == (11, 2)  # corner ball {(0,0),(0,1),(1,0)}: reads 3 + 4 + 4
    assert after == [0, 1, 0, 1, 1, 0, 0, 0, 0]

    # MIX: the region through nonzero cells is cut by the empty cells, so a cell
    # inside the Manhattan ball can stay unreached; every reached cell reads all
    # its in-bounds neighbours (zeros included) and the sum is taken mod 3, plus 1.
    s = grid((3, 3), {(0, 0): 1, (0, 1): 1, (1, 1): 2, (1, 2): 2, (2, 0): 1, (2, 2): 1})
    after, scan, effect, predicate = run("occluded_mix", s, Event("MIX", {"p": [0, 0], "r": 1}))
    assert (scan, effect, predicate) == (7, 2, None)  # two cells, 2 + 3 neighbour reads
    assert after == list(
        grid(
            (3, 3), {(0, 0): 3, (0, 1): 2, (1, 1): 2, (1, 2): 2, (2, 0): 1, (2, 2): 1}
        ).dense_values()
    )
    after, scan, effect, _ = run("occluded_mix", s, Event("MIX", {"p": [0, 0], "r": 2}))
    assert (scan, effect) == (12, 3)
    assert after == list(
        grid(
            (3, 3), {(0, 0): 3, (0, 1): 2, (1, 1): 3, (1, 2): 2, (2, 0): 1, (2, 2): 1}
        ).dense_values()
    )
    after, scan, effect, _ = run("occluded_mix", s, Event("MIX", {"p": [0, 0], "r": 4}))
    # (2,0) is within distance 2 but walled off by zeros; (2,2) is reached at
    # depth 4 and recomputes to its own value: 1 + ((1 + 2 + 0) mod 3) = 1
    assert (scan, effect) == (19, 4)
    assert after == list(
        grid(
            (3, 3), {(0, 0): 3, (0, 1): 2, (1, 1): 3, (1, 2): 3, (2, 0): 1, (2, 2): 1}
        ).dense_values()
    )

    # RELABEL and LAYER on a [3,3] world.
    s = grid((3, 3), {(0, 0): 1, (0, 1): 1, (1, 1): 1, (1, 2): 2, (2, 0): 1, (2, 2): 1})
    after, scan, effect, _ = run("component_relabel", s, Event("RELABEL", {"p": [0, 0], "v": 3}))
    assert (scan, effect) == (7, 3)
    after, scan, effect, _ = run("component_relabel", s, Event("RELABEL", {"p": [2, 2], "v": 0}))
    assert (scan, effect) == (3, 1)
    center = grid((3, 3), {(1, 1): 1, (0, 1): 2, (1, 0): 2, (1, 2): 2, (2, 1): 2})
    # an enclosed component is a no-op for both signs, but the whole boundary is read
    assert run("boundary_layer", center, Event("LAYER", {"p": [1, 1], "s": -1}))[1:] == (5, 0, None)
    assert run("boundary_layer", center, Event("LAYER", {"p": [1, 1], "s": 1}))[1:] == (5, 0, None)
    # the 2 at (0,1) is its own component: it touches two corners and the centre
    after, scan, effect, _ = run("boundary_layer", center, Event("LAYER", {"p": [0, 1], "s": -1}))
    assert (scan, effect) == (4, 1)
    assert after == list(grid((3, 3), {(1, 1): 1, (1, 0): 2, (1, 2): 2, (2, 1): 2}).dense_values())
    after, scan, effect, _ = run("boundary_layer", center, Event("LAYER", {"p": [0, 1], "s": 1}))
    assert (scan, effect) == (4, 2)
    assert after == list(
        grid(
            (3, 3), {(0, 0): 2, (0, 1): 2, (0, 2): 2, (1, 1): 1, (1, 0): 2, (1, 2): 2, (2, 1): 2}
        ).dense_values()
    )
    edge = grid((3, 3), {(0, 0): 1, (0, 1): 1, (1, 0): 2, (1, 1): 2, (0, 2): 2})
    assert run("boundary_layer", edge, Event("LAYER", {"p": [0, 0], "s": -1}))[1:] == (5, 0, None)
    after, scan, effect, _ = run("boundary_layer", edge, Event("LAYER", {"p": [1, 0], "s": -1}))
    assert (scan, effect) == (7, 2)  # both 2s of the pair touch an empty cell; five boundary cells
    assert after == list(grid((3, 3), {(0, 0): 1, (0, 1): 1, (0, 2): 2}).dense_values())

    # MOVE with two fibers and an obstacle.
    s = grid((3, 3), {(0, 0): 1, (2, 0): 1, (1, 1): 1, (0, 2): 2})
    after, scan, effect, _ = run(
        "obstructed_motion", s, Event("MOVE", {"v": 1, "axis": 0, "direction": 1, "k": 2})
    )
    assert (scan, effect) == (6, 4)
    assert after == list(grid((3, 3), {(1, 0): 1, (2, 0): 1, (2, 1): 1, (0, 2): 2}).dense_values())

    # COMPRESS rows then columns of a [2,3] world.
    s = grid((2, 3), {(0, 0): 2, (0, 1): 2, (0, 2): 4, (1, 1): 4, (1, 2): 4}, (0, 2, 4, 8))
    after, scan, _, _ = run(
        "aggregating_compression", s, Event("COMPRESS", {"axis": 1, "direction": -1})
    )
    assert scan == 6
    assert after == [4, 4, 0, 8, 0, 0]
    after, _, _, _ = run(
        "aggregating_compression", s, Event("COMPRESS", {"axis": 0, "direction": 1})
    )
    assert after == [0, 2, 0, 2, 4, 8]


def test_three_dimensional_cases():
    s = grid((3, 3, 3), {(1, 1, 1): 1, (0, 1, 1): 1, (1, 1, 2): 2, (2, 2, 2): 1}, (0, 1, 2))
    after, scan, effect, predicate = run(
        "metric_paint", s, Event("PAINT", {"a": [1, 1, 1], "r": 1, "u": 1, "v": 2})
    )
    assert (scan, effect, predicate) == (7, 2, True)
    assert after == list(
        grid(
            (3, 3, 3), {(0, 1, 1): 2, (1, 1, 1): 2, (1, 1, 2): 2, (2, 2, 2): 1}, (0, 1, 2)
        ).dense_values()
    )

    s = grid((2, 2, 2), {(0, 0, 0): 1, (1, 1, 1): 2}, (0, 1, 2))
    event = Event("REFLECT", {"lo": [0, 0, 0], "hi": [1, 1, 1], "axis": 2})
    after, scan, effect, _ = run("box_reflection", s, event)
    assert (scan, effect) == (8, 4)
    assert after == list(grid((2, 2, 2), {(0, 0, 1): 1, (1, 1, 0): 2}, (0, 1, 2)).dense_values())
    twice = get_family("box_reflection").transition(
        get_family("box_reflection").transition(s, event), event
    )
    assert twice == s

    s = grid((2, 2, 2), {(0, 0, 0): 2}, (0, 1, 2))
    after, scan, effect, _ = run("local_stencil", s, Event("STENCIL", {"a": [0, 0, 0], "r": 3}))
    assert (scan, effect) == (32, 3)
    assert after == [2, 2, 2, 0, 2, 0, 0, 0]
    after, scan, effect, _ = run("local_stencil", s, Event("STENCIL", {"a": [0, 0, 0], "r": 1}))
    assert (scan, effect) == (16, 3)
    assert after == [2, 2, 2, 0, 2, 0, 0, 0]
    after, scan, effect, _ = run("local_stencil", s, Event("STENCIL", {"a": [1, 1, 1], "r": 1}))
    assert (scan, effect) == (16, 0)  # the far ball never reads the single nonzero cell
    assert after == [2, 0, 0, 0, 0, 0, 0, 0]

    s = grid((2, 2, 2), {(0, 0, 0): 1, (1, 0, 0): 1, (1, 1, 0): 2})
    after, scan, effect, _ = run("occluded_mix", s, Event("MIX", {"p": [0, 0, 0], "r": 1}))
    assert (scan, effect) == (8, 2)  # two cells reached, three neighbour reads each
    assert after == list(grid((2, 2, 2), {(0, 0, 0): 3, (1, 0, 0): 2, (1, 1, 0): 2}).dense_values())
    after, scan, effect, _ = run("occluded_mix", s, Event("MIX", {"p": [0, 0, 0], "r": 2}))
    assert (scan, effect) == (12, 3)  # (1,1,0) is reached at depth 2: 1 + ((2+1) mod 3) = 1
    assert after == list(grid((2, 2, 2), {(0, 0, 0): 3, (1, 0, 0): 2, (1, 1, 0): 1}).dense_values())

    s = grid((2, 2, 2), {(0, 0, 0): 1, (1, 0, 0): 1, (1, 1, 0): 2}, (0, 1, 2))
    after, scan, effect, _ = run("component_relabel", s, Event("RELABEL", {"p": [1, 0, 0], "v": 2}))
    assert (scan, effect) == (6, 2)

    cells: dict[tuple[int, ...], int] = {(1, 1, 1): 1}
    for axis in range(3):
        for delta in (-1, 1):
            c = [1, 1, 1]
            c[axis] += delta
            cells[tuple(c)] = 2
    s = grid((3, 3, 3), cells, (0, 1, 2))
    assert run("boundary_layer", s, Event("LAYER", {"p": [1, 1, 1], "s": -1}))[1:] == (7, 0, None)
    opened = s.set((0, 1, 1), 0)
    assert run("boundary_layer", opened, Event("LAYER", {"p": [1, 1, 1], "s": -1}))[1:] == (
        7,
        1,
        None,
    )
    after, scan, effect, _ = run("boundary_layer", opened, Event("LAYER", {"p": [1, 1, 1], "s": 1}))
    assert (scan, effect) == (7, 1)
    # the layer grows back into the opened cell with the component's own label
    assert after == list(opened.set((0, 1, 1), 1).dense_values())

    s = grid((2, 2, 3), {(0, 0, 0): 1, (1, 1, 0): 1, (0, 0, 2): 2}, (0, 1, 2))
    after, scan, effect, _ = run(
        "obstructed_motion", s, Event("MOVE", {"v": 1, "axis": 2, "direction": 1, "k": 5})
    )
    assert scan == 6
    assert after == list(
        grid((2, 2, 3), {(0, 0, 1): 1, (0, 0, 2): 2, (1, 1, 2): 1}, (0, 1, 2)).dense_values()
    )

    s = grid((2, 2, 2), {(0, 0, 0): 2, (0, 0, 1): 2, (1, 1, 0): 4}, (0, 2, 4))
    after, scan, _, _ = run(
        "aggregating_compression", s, Event("COMPRESS", {"axis": 2, "direction": -1})
    )
    assert scan == 8
    assert after == list(grid((2, 2, 2), {(0, 0, 0): 4, (1, 1, 0): 4}, (0, 2, 4)).dense_values())


def test_five_dimensional_stencil_single_cell():
    world = WorldSpec((2,) * 5, (0, 1))
    state = WorldState(world, (((0, 0, 0, 0, 0), 1),))
    family = get_family("local_stencil")
    transition = family.step(state, Event("STENCIL", {"a": [0] * 5, "r": 5}))
    assert transition.workload.scan == 192
    assert transition.after.nonzero_count == 6
    windowed = family.step(state, Event("STENCIL", {"a": [0] * 5, "r": 1}))
    assert windowed.workload.scan == 36
    assert windowed.extras["support"] == 6
    assert windowed.after == transition.after


def test_layer_growth_and_erosion_are_not_inverses():
    family = get_family("boundary_layer")
    before = line([0, 1, 0])
    grown = family.step(before, Event("LAYER", {"p": [1], "s": 1})).after
    eroded = family.step(grown, Event("LAYER", {"p": [1], "s": -1})).after
    assert list(grown.dense_values()) == [1, 1, 1]
    assert eroded == grown  # the exterior is not an empty in-bounds neighbour
    assert eroded != before


def test_compression_can_merge_again_in_the_same_direction():
    family = get_family("aggregating_compression")
    event = Event("COMPRESS", {"axis": 0, "direction": -1})
    before = line([2, 2, 4], (0, 2, 4, 8))
    first = family.step(before, event).after
    second = family.step(first, event).after
    assert list(first.dense_values()) == [4, 4, 0]
    assert list(second.dense_values()) == [8, 0, 0]


def test_stencil_and_mix_differ_even_without_occlusion():
    before = line([1, 1, 1])
    stencil = get_family("local_stencil").step(before, Event("STENCIL", {"a": [1], "r": 1}))
    mix = get_family("occluded_mix").step(before, Event("MIX", {"p": [1], "r": 1}))
    assert list(stencil.after.dense_values()) == [2, 3, 2]
    assert list(mix.after.dense_values()) == [3, 1, 3]
