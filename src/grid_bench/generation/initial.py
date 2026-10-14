"""Initial-state samplers with exact density control."""

from __future__ import annotations

from random import Random

from ..grid import Grid
from ..schema import WorldSpec


def _target_count(world: WorldSpec, density: float) -> int:
    return max(1, min(world.volume - 1, round(density * world.volume)))


def uniform(rng: Random, world: WorldSpec, density: float, labels: tuple[int, ...]) -> Grid:
    """Exactly ``round(density * N)`` cells, positions and labels uniform."""

    grid = Grid(world)
    for index in rng.sample(range(world.volume), _target_count(world, density)):
        grid.data[index] = rng.choice(labels)
    return grid


def tiles(rng: Random, world: WorldSpec, density: float, labels: tuple[int, ...]) -> Grid:
    """Small tiles only: the two smallest positive values, 9:1."""

    small = sorted(labels)[:2]
    grid = Grid(world)
    for index in rng.sample(range(world.volume), _target_count(world, density)):
        grid.data[index] = small[0] if rng.random() < 0.9 or len(small) == 1 else small[1]
    return grid


def _blob_radii(grid: Grid, target_scan: int) -> tuple[int, ...]:
    """Radii whose central ball is about half the target component-plus-boundary size."""

    wanted = max(2, target_scan // 2)
    center = tuple(size // 2 for size in grid.shape)
    radius = 0
    while radius < sum(grid.shape) and len(grid.l1_ball(center, radius + 1)) <= wanted:
        radius += 1
    return tuple(sorted({max(0, radius - 1), radius}))


def mixed_blob(
    rng: Random,
    world: WorldSpec,
    density: float,
    labels: tuple[int, ...],
    target_scan: int,
    *,
    blob_fraction: float = 0.7,
    overwrite: float = 0.5,
) -> Grid:
    """Random L1 balls of random labels plus an i.i.d. sprinkle, at exact density."""

    grid = Grid(world)
    data = grid.data
    total = _target_count(world, density)
    blob_budget = int(round(total * blob_fraction))
    radii = _blob_radii(grid, target_scan)
    guard = 0
    while grid.nonzero_count < blob_budget and guard < 50 * total:
        guard += 1
        center = grid.coord(rng.randrange(grid.size))
        label = rng.choice(labels)
        for index in grid.l1_ball(center, rng.choice(radii)):
            if data[index] == 0 or rng.random() < overwrite:
                data[index] = label
    empty = grid.zero_indices()
    rng.shuffle(empty)
    while grid.nonzero_count < total and empty:
        data[empty.pop()] = rng.choice(labels)
    nonzero = grid.nonzero_indices()
    rng.shuffle(nonzero)
    while grid.nonzero_count > total and nonzero:
        data[nonzero.pop()] = 0
    return grid


def _grow_region(rng: Random, grid: Grid, seed_index: int, size: int) -> list[int]:
    """Grow a random connected region of empty cells starting at ``seed_index``."""

    data = grid.data
    members = [seed_index]
    member_set = {seed_index}
    frontier: list[int] = []
    frontier_set: set[int] = set()

    def expand(index: int) -> None:
        for other in grid.neighbors(index):
            if data[other] == 0 and other not in member_set and other not in frontier_set:
                frontier.append(other)
                frontier_set.add(other)

    expand(seed_index)
    while len(members) < size and frontier:
        position = rng.randrange(len(frontier))
        chosen = frontier[position]
        frontier[position] = frontier[-1]
        frontier.pop()
        frontier_set.discard(chosen)
        members.append(chosen)
        member_set.add(chosen)
        expand(chosen)
    return members


def _region_label(rng: Random, grid: Grid, members: list[int], labels: tuple[int, ...]) -> int:
    touching = {grid.data[other] for other in grid.boundary_of(members)} - {0}
    free = [label for label in labels if label not in touching]
    return rng.choice(free) if free else rng.choice(labels)


def _calibrate_region_size(rng: Random, world: WorldSpec, target_scan: int) -> int:
    """Region size whose grown regions have |C ∪ ∂C| closest to ``target_scan``."""

    best_size = 1
    best_distance = float("inf")
    size = 1
    while size <= world.volume:
        grid = Grid(world)
        measured: list[int] = []
        for _ in range(8):
            members = _grow_region(rng, grid, rng.randrange(grid.size), size)
            measured.append(len(members) + len(grid.boundary_of(members)))
        mean = sum(measured) / len(measured)
        distance = abs(mean - target_scan)
        if distance < best_distance:
            best_size, best_distance = size, distance
        if mean >= target_scan or size >= world.volume // 2:
            break
        size = size + 1 if size < 8 else int(size * 1.25) + 1
    return best_size


def regions(
    rng: Random,
    world: WorldSpec,
    density: float,
    labels: tuple[int, ...],
    target_scan: int,
) -> Grid:
    """Random labelled partition: connected regions of controlled size, distinct from neighbours."""

    grid = Grid(world)
    data = grid.data
    total = _target_count(world, density)
    typical = _calibrate_region_size(rng, world, target_scan)
    lo = max(1, typical // 2)
    hi = max(lo, typical + typical // 2)
    empty = grid.zero_indices()
    while grid.nonzero_count < total and empty:
        rng.shuffle(empty)
        seed_index = empty.pop()
        if data[seed_index] != 0:
            continue
        wanted = min(rng.randint(lo, hi), total - grid.nonzero_count)
        members = _grow_region(rng, grid, seed_index, max(1, wanted))
        label = _region_label(rng, grid, members, labels)
        for index in members:
            data[index] = label
        empty = [index for index in empty if data[index] == 0]
    return grid


def sample_initial(
    rng: Random,
    world: WorldSpec,
    density: float,
    initializer: str,
    target_scan: int,
) -> Grid:
    labels = tuple(value for value in world.values if value != 0)
    if initializer == "uniform":
        return uniform(rng, world, density, labels)
    if initializer == "tiles":
        return tiles(rng, world, density, labels)
    if initializer == "mixed_blob":
        return mixed_blob(rng, world, density, labels, target_scan)
    if initializer == "regions":
        return regions(rng, world, density, labels, target_scan)
    raise ValueError(f"unknown initializer {initializer!r}")
