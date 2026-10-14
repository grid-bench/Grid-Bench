from __future__ import annotations

import pytest

from grid_bench.schema import WorldSpec
from grid_bench.world import WorldState


def test_world_state_is_persistent_and_uses_zero_as_default() -> None:
    spec = WorldSpec(shape=(2, 2), values=(0, 1, 2, 3))
    state = WorldState(spec, (((0, 1), 2),))

    assert state.get((0, 0)) == 0
    assert state.get((0, 1)) == 2

    updated = state.set((1, 1), 3)
    cleared = updated.set((0, 1), 0)

    assert state.get((1, 1)) == 0
    assert updated.get((1, 1)) == 3
    assert cleared.to_dict() == {"cells": [[[1, 1], 3]]}


def test_world_state_serializes_cells_lexicographically() -> None:
    spec = WorldSpec(shape=(3, 3), values=(0, 1, 2))
    state = WorldState(spec, (((2, 0), 1), ((0, 2), 2), ((0, 1), 1)))

    assert state.to_dict() == {
        "cells": [
            [[0, 1], 1],
            [[0, 2], 2],
            [[2, 0], 1],
        ]
    }


@pytest.mark.parametrize(
    "cells",
    [
        [[[-1, 0], 1]],
        [[[2, 0], 1]],
        [[[0], 1]],
        [[[0, 0], 0]],
        [[[0, 0], 9]],
        [[[0, 0], 1], [[0, 0], 2]],
    ],
)
def test_world_state_rejects_invalid_cells(cells: list[object]) -> None:
    spec = WorldSpec(shape=(2, 2), values=(0, 1, 2))

    with pytest.raises((TypeError, ValueError)):
        WorldState.from_dict({"cells": cells}, spec)


def test_get_rejects_wrong_dimension_and_out_of_bounds() -> None:
    spec = WorldSpec(shape=(2, 2), values=(0, 1))
    state = WorldState(spec)

    with pytest.raises((TypeError, ValueError)):
        state.get((0,))
    with pytest.raises((TypeError, ValueError)):
        state.get((2, 0))
