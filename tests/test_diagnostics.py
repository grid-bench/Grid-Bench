"""Reference-history diagnostics have explicit event semantics."""

import pytest

from grid_bench.diagnostics import erasure_census, suffix_state
from grid_bench.schema import Event, Instance, WorldSpec
from grid_bench.simulate import simulate
from grid_bench.world import WorldState


def _line(values: list[int], events: tuple[Event, ...], identifier: str = "line") -> Instance:
    world = WorldSpec((len(values),), (0, 1, 2, 3))
    initial = WorldState(world, tuple(((i,), v) for i, v in enumerate(values) if v))
    return Instance(
        id=identifier, family="metric_paint", world=world, initial=initial, events=events
    )


def test_suffix_state_executes_every_set_and_core_events_in_last_k_positions() -> None:
    instance = _line(
        [1, 1, 0, 2, 2, 0],
        (
            Event("PAINT", {"a": [1], "r": 1, "u": 1, "v": 3}),
            Event("SET", {"p": [5], "v": 1}),
            Event("PAINT", {"a": [4], "r": 1, "u": 2, "v": 1}),
        ),
    )
    gold = simulate(instance).final_state
    assert suffix_state(instance, 3) == gold
    assert suffix_state(instance, 99) == gold
    # k = 0 keeps the SET stream only
    assert list(suffix_state(instance, 0).dense_values()) == [1, 1, 0, 2, 2, 1]
    # k = 1 adds the last core event, which is the last position
    assert list(suffix_state(instance, 1).dense_values()) == [1, 1, 0, 1, 1, 1]
    # k = 2 covers positions 2 and 3: the SET and the last PAINT, still without the first
    assert list(suffix_state(instance, 2).dense_values()) == [1, 1, 0, 1, 1, 1]
    with pytest.raises(ValueError):
        suffix_state(instance, -1)


def test_erasure_census_counts_overwritten_early_events() -> None:
    # the first paint is overwritten by the two that follow: omitting it is invisible
    overwritten = _line(
        [1, 1, 1, 1],
        (
            Event("PAINT", {"a": [0], "r": 1, "u": 1, "v": 2}),
            Event("PAINT", {"a": [0], "r": 3, "u": 2, "v": 3}),
            Event("PAINT", {"a": [0], "r": 3, "u": 1, "v": 3}),
        ),
    )
    assert erasure_census(overwritten, [3]) == {3: (1, 1)}
    # here the first paint survives to the end, and an effect-free event is not eligible
    retained = _line(
        [1, 1, 1, 1],
        (
            Event("PAINT", {"a": [0], "r": 0, "u": 1, "v": 2}),
            Event("PAINT", {"a": [3], "r": 0, "u": 3, "v": 1}),
            Event("SET", {"p": [3], "v": 3}),
            Event("PAINT", {"a": [2], "r": 0, "u": 1, "v": 3}),
        ),
    )
    assert erasure_census(retained, [2, 4, 9]) == {2: (1, 0), 4: (1, 0)}
    assert erasure_census(retained, [1]) == {}
