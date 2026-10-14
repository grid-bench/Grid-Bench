from __future__ import annotations

from itertools import product
from random import Random

import pytest

from spatial_worlds.grid import Grid, hamming, shape_tables
from spatial_worlds.schema import WorldSpec
from spatial_worlds.world import WorldState

from oracles import component as oracle_component
from oracles import neighbors as oracle_neighbors
from oracles import random_state, to_dense

SHAPES = [(7,), (3, 4), (2, 3, 3), (2, 2, 2, 3), (2, 2, 2, 2, 2), (4, 4, 5)]


@pytest.mark.parametrize("shape", SHAPES)
def test_index_coord_round_trip_and_lexicographic_order(shape: tuple[int, ...]) -> None:
    grid = Grid(WorldSpec(shape, (0, 1)))
    expected = list(product(*(range(size) for size in shape)))
    assert list(grid.coords) == expected
    for index, coord in enumerate(expected):
        assert grid.index(coord) == index
        assert grid.coord(index) == coord


@pytest.mark.parametrize("shape", SHAPES)
def test_neighbors_boundary_and_edge_count(shape: tuple[int, ...]) -> None:
    grid = Grid(WorldSpec(shape, (0, 1)))
    total_neighbors = 0
    for index, coord in enumerate(grid.coords):
        expected = sorted(grid.index(y) for y in oracle_neighbors(shape, coord))
        assert sorted(grid.neighbors(index)) == expected
        total_neighbors += len(expected)
        on_boundary = any(c == 0 or c == size - 1 for c, size in zip(coord, shape, strict=True))
        assert grid.is_boundary_cell(index) == on_boundary
    assert total_neighbors == 2 * grid.edge_count()


@pytest.mark.parametrize("shape", SHAPES)
def test_ball_box_ray_and_fibers_match_brute_force(shape: tuple[int, ...]) -> None:
    rng = Random(3)
    grid = Grid(WorldSpec(shape, (0, 1)))
    coords = grid.coords
    for _ in range(20):
        center = coords[rng.randrange(grid.size)]
        radius = rng.randrange(0, 4)
        brute = sorted(
            i
            for i, x in enumerate(coords)
            if sum(abs(a - b) for a, b in zip(x, center, strict=True)) <= radius
        )
        assert sorted(grid.l1_ball(center, radius)) == brute
        lo = [rng.randrange(size) for size in shape]
        hi = [rng.randrange(low, size) for low, size in zip(lo, shape, strict=True)]
        brute_box = [
            i
            for i, x in enumerate(coords)
            if all(lo[a] <= x[a] <= hi[a] for a in range(len(shape)))
        ]
        assert grid.box(lo, hi) == brute_box
        while True:
            q = [rng.choice((-1, 0, 1)) for _ in shape]
            if any(q):
                break
        expected_ray = []
        x = list(center)
        while True:
            x = [a + b for a, b in zip(x, q, strict=True)]
            if not all(0 <= a < size for a, size in zip(x, shape, strict=True)):
                break
            expected_ray.append(grid.index(x))
        assert grid.ray(center, q) == expected_ray
    for axis in range(len(shape)):
        fibers = grid.fibers(axis)
        assert len(fibers) == grid.size // shape[axis]
        seen = sorted(i for fiber in fibers for i in fiber)
        assert seen == list(range(grid.size))
        for fiber in fibers:
            cs = [coords[i] for i in fiber]
            assert [c[axis] for c in cs] == list(range(shape[axis]))
            others = {tuple(c[a] for a in range(len(shape)) if a != axis) for c in cs}
            assert len(others) == 1


@pytest.mark.parametrize("shape", SHAPES)
def test_components_match_oracle(shape: tuple[int, ...]) -> None:
    rng = Random(11)
    world = WorldSpec(shape, (0, 1, 2))
    for _ in range(10):
        state = random_state(rng, world, 0.55)
        grid = Grid.from_state(state)
        dense = to_dense(state)
        assert grid.to_state() == state
        labels, comps = grid.label_components()
        assert sorted(i for comp in comps for i in comp) == list(range(grid.size))
        for index in range(grid.size):
            members, boundary = grid.component(index)
            o_members, o_boundary = oracle_component(shape, dense, grid.coord(index))
            assert sorted(members) == sorted(grid.index(x) for x in o_members)
            assert boundary == sorted(grid.index(x) for x in o_boundary)
            assert grid.boundary_of(members) == boundary
            assert labels[index] == labels[members[0]]


def test_hamming_and_copy_independence() -> None:
    world = WorldSpec((3,), (0, 1))
    grid = Grid(world, [0, 1, 0])
    other = grid.copy()
    other.data[0] = 1
    assert grid.data == [0, 1, 0]
    assert hamming(grid.data, other.data) == 1
    assert shape_tables((3,)) is shape_tables((3,))
    with pytest.raises(ValueError):
        Grid(world, [0, 1])
    assert Grid.from_state(WorldState(world)).nonzero_count == 0
