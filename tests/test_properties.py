"""Oracle cross-checks and invariants for every family on random worlds in d = 1..5."""

from __future__ import annotations

from random import Random

import pytest

from grid_bench.grid import Grid
from grid_bench.registry import family_ids, get_family
from grid_bench.schema import Event, SchemaError, WorldSpec
from grid_bench.world import WorldState

from oracles import FAMILY_VALUES, apply_oracle, from_dense, random_event, random_state, to_dense

SHAPES = [(7,), (3, 4), (2, 3, 3), (2, 2, 2, 3), (2, 2, 2, 2, 2)]
PAIRS_PER_SHAPE = 120


def _world(family_id: str, shape: tuple[int, ...]) -> WorldSpec:
    return WorldSpec(shape, FAMILY_VALUES[family_id])


def _initial_values(family_id: str) -> list[int] | None:
    return [1, 2, 4] if family_id == "aggregating_compression" else None


@pytest.mark.parametrize("family_id", sorted(family_ids()))
@pytest.mark.parametrize("shape", SHAPES)
def test_simulator_matches_independent_oracle(family_id: str, shape: tuple[int, ...]) -> None:
    rng = Random(f"{family_id}-{shape}")
    family = get_family(family_id)
    world = _world(family_id, shape)
    for _ in range(PAIRS_PER_SHAPE):
        state = random_state(rng, world, rng.uniform(0.3, 0.8), _initial_values(family_id))
        event = random_event(rng, family_id, world)
        if rng.random() < 0.1:
            event = Event(
                "SET", {"p": [rng.randrange(s) for s in shape], "v": rng.choice(world.values)}
            )
        family.validate_event(world, event)
        expected_after, expected_scan, expected_predicate = apply_oracle(
            family_id, world, to_dense(state), event
        )
        transition = family.step(state, event)
        assert transition.after == from_dense(world, expected_after), (event, state)
        assert transition.workload.scan == expected_scan, (event, state)
        assert transition.predicate == expected_predicate, (event, state)
        before = to_dense(state)
        assert transition.workload.effect == sum(
            1 for x in before if before[x] != expected_after[x]
        )
        assert transition.effective == (
            transition.predicate
            if transition.predicate is not None
            else transition.workload.effect > 0
        )


@pytest.mark.parametrize("shape", SHAPES)
def test_reflect_is_an_involution_and_preserves_box_values(shape: tuple[int, ...]) -> None:
    rng = Random(f"reflect-{shape}")
    family = get_family("box_reflection")
    world = _world("box_reflection", shape)
    for _ in range(40):
        state = random_state(rng, world, 0.5)
        event = random_event(rng, "box_reflection", world)
        once = family.transition(state, event)
        twice = family.transition(once, event)
        assert twice == state
        assert sorted(state.dense_values()) == sorted(once.dense_values())


@pytest.mark.parametrize("shape", SHAPES)
def test_move_preserves_multiset_and_fiber_order(shape: tuple[int, ...]) -> None:
    rng = Random(f"move-{shape}")
    family = get_family("obstructed_motion")
    world = _world("obstructed_motion", shape)
    for _ in range(40):
        state = random_state(rng, world, 0.45)
        event = random_event(rng, "obstructed_motion", world)
        after = family.transition(state, event)
        assert sorted(state.dense_values()) == sorted(after.dense_values())
        v = event.params["v"]
        axis = event.params["axis"]
        direction = event.params["direction"]
        k = event.params["k"]
        before_grid = Grid.from_state(state)
        after_grid = Grid.from_state(after)
        for index in range(before_grid.size):
            if before_grid.data[index] != v and before_grid.data[index] != 0:
                assert after_grid.data[index] == before_grid.data[index]
        for fiber in before_grid.fibers(axis):
            old = [pos for pos, index in enumerate(fiber) if before_grid.data[index] == v]
            new = [pos for pos, index in enumerate(fiber) if after_grid.data[index] == v]
            assert len(old) == len(new)
            for a, b in zip(old, new, strict=True):
                assert 0 <= (b - a) * direction <= k


@pytest.mark.parametrize("shape", SHAPES)
def test_compress_conserves_fiber_sums_and_packs(shape: tuple[int, ...]) -> None:
    rng = Random(f"compress-{shape}")
    family = get_family("aggregating_compression")
    world = _world("aggregating_compression", shape)
    for _ in range(40):
        state = random_state(rng, world, 0.5, [1, 2, 4])
        event = random_event(rng, "aggregating_compression", world)
        transition = family.step(state, event)
        assert transition.workload.scan == world.volume
        before = Grid.from_state(state)
        after = Grid.from_state(transition.after)
        assert after.nonzero_count <= before.nonzero_count
        axis = event.params["axis"]
        direction = event.params["direction"]
        for fiber in before.fibers(axis):
            assert sum(before.data[i] for i in fiber) == sum(after.data[i] for i in fiber)
            ordered = fiber if direction == -1 else fiber[::-1]
            values = [after.data[i] for i in ordered]
            first_zero = next((n for n, value in enumerate(values) if value == 0), len(values))
            assert all(value == 0 for value in values[first_zero:])


@pytest.mark.parametrize("shape", SHAPES)
def test_stencil_is_linear_over_the_residues(shape: tuple[int, ...]) -> None:
    rng = Random(f"stencil-{shape}")
    family = get_family("local_stencil")
    world = _world("local_stencil", shape)
    modulus = len(world.values)
    for _ in range(20):
        event = random_event(rng, "local_stencil", world)
        a = random_state(rng, world, 0.6)
        b = random_state(rng, world, 0.6)
        summed = WorldState(
            world,
            tuple(
                (x, (va + vb) % modulus)
                for x, va, vb in zip(a.positions(), a.dense_values(), b.dense_values(), strict=True)
                if (va + vb) % modulus
            ),
        )
        lhs = family.transition(summed, event).dense_values()
        ra = family.transition(a, event).dense_values()
        rb = family.transition(b, event).dense_values()
        assert list(lhs) == [(x + y) % modulus for x, y in zip(ra, rb, strict=True)]
    empty = WorldState.empty(world)
    whole = Event("STENCIL", {"a": [0] * len(shape), "r": sum(shape)})
    assert family.transition(empty, whole) == empty
    assert family.step(empty, whole).workload.scan == world.volume + 2 * Grid(world).edge_count()


@pytest.mark.parametrize("shape", SHAPES)
def test_layer_paint_relabel_mix_invariants(shape: tuple[int, ...]) -> None:
    rng = Random(f"misc-{shape}")
    for _ in range(40):
        world = _world("boundary_layer", shape)
        state = random_state(rng, world, 0.6)
        event = random_event(rng, "boundary_layer", world)
        transition = get_family("boundary_layer").step(state, event)
        grid = Grid.from_state(state)
        after_grid = Grid.from_state(transition.after)
        anchor = grid.index(event.params["p"])
        changed = {i for i in range(grid.size) if grid.data[i] != after_grid.data[i]}
        assert transition.effective == (transition.workload.effect > 0)
        if grid.data[anchor] == 0:
            assert not changed and transition.workload.scan == 1
            continue
        members, boundary = grid.component(anchor)
        if event.params["s"] == -1:
            assert transition.after.nonzero_count <= state.nonzero_count
            assert changed == {
                i for i in members if any(grid.data[o] == 0 for o in grid.neighbors(i))
            }
            assert all(after_grid.data[i] == 0 for i in changed)
        else:
            assert transition.after.nonzero_count >= state.nonzero_count
            assert changed == {i for i in boundary if grid.data[i] == 0}
            assert all(after_grid.data[i] == grid.data[anchor] for i in changed)
        assert transition.workload.scan == len(members) + len(boundary)

        world = _world("metric_paint", shape)
        state = random_state(rng, world, 0.5)
        event = random_event(rng, "metric_paint", world)
        after = get_family("metric_paint").transition(state, event)
        grid = Grid.from_state(state)
        ball = set(grid.l1_ball(event.params["a"], event.params["r"]))
        after_grid = Grid.from_state(after)
        u, v = event.params["u"], event.params["v"]
        for index in range(grid.size):
            if index not in ball:
                assert after_grid.data[index] == grid.data[index]
            elif u != v:
                assert after_grid.data[index] != u

        world = _world("component_relabel", shape)
        state = random_state(rng, world, 0.5)
        event = random_event(rng, "component_relabel", world)
        after = get_family("component_relabel").transition(state, event)
        grid = Grid.from_state(state)
        members, boundary = grid.component(grid.index(event.params["p"]))
        after_grid = Grid.from_state(after)
        assert all(after_grid.data[i] == event.params["v"] for i in members)
        assert all(grid.data[i] != grid.data[members[0]] for i in boundary)

        world = _world("occluded_mix", shape)
        state = random_state(rng, world, 0.6)
        event = random_event(rng, "occluded_mix", world)
        transition = get_family("occluded_mix").step(state, event)
        grid = Grid.from_state(state)
        after_grid = Grid.from_state(transition.after)
        p = event.params["p"]
        anchor = grid.index(p)
        r = event.params["r"]
        changed = {i for i in range(grid.size) if grid.data[i] != after_grid.data[i]}
        assert transition.effective == (transition.workload.effect > 0)
        if grid.data[anchor] == 0:
            assert not changed and transition.workload.scan == 1
            continue
        # the mix never creates or removes nonzero cells, never leaves the
        # Manhattan ball of radius r, and never changes a cell it cannot reach
        # through nonzero cells (the anchor's nonzero component bounds it)
        assert transition.after.nonzero_count == state.nonzero_count
        assert all(grid.data[i] != 0 and after_grid.data[i] != 0 for i in changed)
        assert changed <= set(grid.l1_ball(p, r))
        reachable = set()
        stack = [anchor]
        while stack:
            index = stack.pop()
            if index in reachable:
                continue
            reachable.add(index)
            stack.extend(o for o in grid.neighbors(index) if grid.data[o] != 0)
        assert changed <= reachable
        residues = len(world.values) - 1
        if r >= sum(grid.shape):
            # every reachable cell is recomputed from its old neighbourhood
            for index in reachable:
                total = grid.data[index] + sum(grid.data[o] for o in grid.neighbors(index))
                assert after_grid.data[index] == 1 + total % residues
            assert transition.workload.scan == sum(
                1 + len(grid.neighbors(index)) for index in reachable
            )


def test_compress_alphabet_overflow_is_rejected() -> None:
    world = WorldSpec((3,), (0, 2, 4))
    state = WorldState(world, (((0,), 4), ((1,), 4)))
    with pytest.raises(SchemaError):
        get_family("aggregating_compression").transition(
            state, Event("COMPRESS", {"axis": 0, "direction": -1})
        )
