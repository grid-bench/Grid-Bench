"""SET-event policies: density thermostat, distinct-label thermostat, tile spawn."""

from __future__ import annotations

from random import Random

from ..family import set_event
from ..grid import Grid
from ..schema import Event


def thermostat(
    rng: Random, grid: Grid, band: tuple[float, float], labels: tuple[int, ...]
) -> Event:
    """Push the density back into ``band``; inside the band, change a random cell."""

    lo, hi = band
    density = grid.density
    data = grid.data
    if density > hi:
        occupied = grid.nonzero_indices()
        if occupied:
            return set_event(grid.coord(rng.choice(occupied)), 0)
    if density < lo:
        empty = grid.zero_indices()
        if empty:
            return set_event(grid.coord(rng.choice(empty)), rng.choice(labels))
    index = rng.randrange(grid.size)
    choices = [value for value in grid.world.values if value != data[index]]
    return set_event(grid.coord(index), rng.choice(choices))


def distinct(rng: Random, grid: Grid, band: tuple[float, float], labels: tuple[int, ...]) -> Event:
    """Thermostat whose nonzero writes avoid the labels of the cell's neighbours."""

    lo, hi = band
    density = grid.density
    data = grid.data
    if density > hi:
        occupied = grid.nonzero_indices()
        if occupied:
            return set_event(grid.coord(rng.choice(occupied)), 0)
    index = rng.choice(grid.zero_indices()) if density < lo else rng.randrange(grid.size)
    touching = {data[other] for other in grid.neighbors(index)} - {0}
    choices = [label for label in labels if label not in touching and label != data[index]]
    if not choices:
        choices = [value for value in grid.world.values if value != data[index]]
    return set_event(grid.coord(index), rng.choice(choices))


def spawn(rng: Random, grid: Grid, band: tuple[float, float], labels: tuple[int, ...]) -> Event:
    """2048-style spawn: a small tile in an empty cell, or a clear when saturated."""

    small = sorted(labels)[:2]
    empty = grid.zero_indices()
    if grid.density > band[1] or not empty:
        occupied = grid.nonzero_indices()
        if occupied:
            return set_event(grid.coord(rng.choice(occupied)), 0)
    if not empty:
        return thermostat(rng, grid, band, labels)
    value = small[0] if rng.random() < 0.9 or len(small) == 1 else small[1]
    return set_event(grid.coord(rng.choice(empty)), value)


POLICIES = {
    "thermostat": thermostat,
    "distinct": distinct,
    "spawn": spawn,
}


def propose_set(
    policy: str, rng: Random, grid: Grid, band: tuple[float, float], labels: tuple[int, ...]
) -> Event:
    try:
        function = POLICIES[policy]
    except KeyError as error:
        raise ValueError(f"unknown set policy {policy!r}") from error
    return function(rng, grid, band, labels)
