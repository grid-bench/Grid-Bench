"""Helpers shared by the family simulators and event proposers."""

from __future__ import annotations

from collections.abc import Sequence
from random import Random

from ..grid import Grid
from ..schema import WorldSpec


def nonzero_labels(world: WorldSpec) -> tuple[int, ...]:
    return tuple(value for value in world.values if value != 0)


def random_direction(rng: Random, dimension: int) -> tuple[int, ...]:
    """A uniformly random nonzero vector in {-1, 0, 1}^d."""

    while True:
        vector = tuple(rng.choice((-1, 0, 1)) for _ in range(dimension))
        if any(vector):
            return vector


def axis_directions(dimension: int) -> tuple[tuple[int, ...], ...]:
    """The 2d unit steps, ordered by axis and then by sign (-1 before +1)."""

    directions: list[tuple[int, ...]] = []
    for axis in range(dimension):
        for sign in (-1, 1):
            step = [0] * dimension
            step[axis] = sign
            directions.append(tuple(step))
    return tuple(directions)


def least_frequent_label(
    grid: Grid, labels: Sequence[int], *, exclude: int | None = None
) -> int | None:
    counts = grid.value_counts()
    best: int | None = None
    best_count = -1
    for label in labels:
        if label == exclude:
            continue
        count = counts.get(label, 0)
        if best is None or count < best_count:
            best, best_count = label, count
    return best


def log_distance(value: int, target: int) -> float:
    """|ln(value / target)| with a floor so zero never divides."""

    import math

    return abs(math.log(max(value, 1) / max(target, 1)))


def closest(candidates: list[tuple[float, int]], k: int, rng: Random) -> list[int]:
    """Indices of the ``k`` candidates with the smallest distance, ties shuffled."""

    order = list(range(len(candidates)))
    rng.shuffle(order)
    order.sort(key=lambda i: candidates[i][0])
    return [candidates[i][1] for i in order[:k]]


def ball_radii(grid: Grid, target: int) -> list[int]:
    """Radii whose central L1 ball size brackets ``target`` (at most three values)."""

    center = tuple(size // 2 for size in grid.shape)
    best_radius = 0
    best_distance = float("inf")
    radius = 0
    limit = sum(grid.shape)
    while radius <= limit:
        size = len(grid.l1_ball(center, radius))
        distance = log_distance(size, target)
        if distance < best_distance:
            best_radius, best_distance = radius, distance
        if size >= grid.size:
            break
        radius += 1
    return sorted({max(0, best_radius - 1), best_radius, best_radius + 1})


def covering_radius(shape: Sequence[int]) -> int:
    """The smallest radius for which the ball centred at the origin covers the world."""

    return sum(size - 1 for size in shape)


def nonzero_layers(
    grid: Grid, start: int, limit: int
) -> tuple[list[list[int]], list[int], list[int]]:
    """Breadth-first layers through nonzero cells from ``start`` up to depth ``limit``.

    Returns ``(layers, new_blockers, blockers)``: ``layers[k]`` lists the cells
    at distance exactly ``k`` (``layers[0] == [start]``); expanding layer ``k``
    discovers ``new_blockers[k]`` empty in-bounds cells not seen before; and
    ``blockers`` lists every empty cell discovered, i.e. the empty cells
    adjacent to a cell at distance below ``limit`` (below the exhaustion depth
    when the spread dies out earlier).  ``start`` must be nonzero.
    """

    data = grid.data
    table = grid.neighbor_table
    layers = [[start]]
    seen = {start}
    blocked: set[int] = set()
    new_blockers: list[int] = []
    depth = 0
    while depth < limit:
        frontier: list[int] = []
        found = 0
        for index in layers[depth]:
            for other in table[index]:
                if other in seen:
                    continue
                if data[other] == 0:
                    if other not in blocked:
                        blocked.add(other)
                        found += 1
                else:
                    seen.add(other)
                    frontier.append(other)
        new_blockers.append(found)
        if not frontier:
            break
        layers.append(frontier)
        depth += 1
    return layers, new_blockers, sorted(blocked)
