from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from grid_bench.schema import Event, Instance, WorldSpec  # noqa: E402
from grid_bench.world import WorldState  # noqa: E402


@pytest.fixture
def small_instance() -> Instance:
    world = WorldSpec(shape=(2, 2), values=(0, 1, 2, 3))
    initial = WorldState(world, (((0, 0), 1), ((1, 1), 2)))
    events = (
        Event("SET", {"p": [0, 1], "v": 3}),
        Event("PAINT", {"a": [0, 0], "r": 1, "u": 1, "v": 2}),
    )
    return Instance(
        id="test-metric-paint",
        family="metric_paint",
        world=world,
        initial=initial,
        events=events,
    )
