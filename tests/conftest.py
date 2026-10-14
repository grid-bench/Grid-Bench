from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from spatial_worlds.schema import Event, Instance, WorldSpec  # noqa: E402
from spatial_worlds.world import WorldState  # noqa: E402


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--require-frozen-data",
        action="store_true",
        help="Fail instead of skipping archived-data tests when the archive is absent.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    required = [item for item in items if item.get_closest_marker("frozen_data")]
    if not required:
        return
    root = Path(__file__).resolve().parents[1]
    paths = (
        root / "data/study-v1/formal-128k-high-sol-mini-01/binding.json",
        root / "data/main/manifest.json",
        root / "data/rq2/manifest.json",
    )
    if all(path.is_file() for path in paths):
        return
    if config.getoption("--require-frozen-data") or any(path.exists() for path in paths):
        raise pytest.UsageError(
            "Frozen study archive is missing or incomplete; restore the binding and datasets."
        )
    skip = pytest.mark.skip(reason="frozen study archive is not included in the source snapshot")
    for item in required:
        item.add_marker(skip)


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
