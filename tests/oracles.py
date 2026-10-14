"""Independent naive implementations of every family rule.

These are deliberately written over coordinate dictionaries with
``itertools.product`` and no shared code with ``grid_bench.grid`` so that
they can cross-check the reference simulators.  Each oracle returns
``(after, scan, predicate)``.
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import product
from random import Random

from grid_bench.schema import Event, WorldSpec
from grid_bench.world import WorldState

Coord = tuple[int, ...]
Dense = dict[Coord, int]


def positions(shape: tuple[int, ...]) -> list[Coord]:
    return list(product(*(range(size) for size in shape)))


def in_bounds(shape: tuple[int, ...], x: Coord) -> bool:
    return all(0 <= xi < size for xi, size in zip(x, shape, strict=True))


def neighbors(shape: tuple[int, ...], x: Coord) -> Iterator[Coord]:
    for axis in range(len(shape)):
        for delta in (-1, 1):
            y = list(x)
            y[axis] += delta
            if 0 <= y[axis] < shape[axis]:
                yield tuple(y)


def to_dense(state: WorldState) -> Dense:
    dense = dict.fromkeys(positions(state.world.shape), 0)
    for position, value in state.cells:
        dense[position] = value
    return dense


def from_dense(world: WorldSpec, dense: Dense) -> WorldState:
    return WorldState(world, tuple((x, v) for x, v in dense.items() if v))


def paint(shape, W: Dense, a, r, u, v):
    ball = [
        x for x in positions(shape) if sum(abs(xi - ai) for xi, ai in zip(x, a, strict=True)) <= r
    ]
    new = dict(W)
    selected = [x for x in ball if W[x] == u]
    if u != v:
        for x in selected:
            new[x] = v
    return new, len(ball), bool(selected)


def reflect(shape, W: Dense, lo, hi, axis):
    box = [x for x in positions(shape) if all(lo[i] <= x[i] <= hi[i] for i in range(len(shape)))]
    new = dict(W)
    for x in box:
        y = list(x)
        y[axis] = lo[axis] + hi[axis] - x[axis]
        new[tuple(y)] = W[x]
    return new, len(box), None


def stencil(shape, modulus: int, W: Dense, a, r):
    new = dict(W)
    scan = 0
    for x in positions(shape):
        if sum(abs(xi - ai) for xi, ai in zip(x, a, strict=True)) > r:
            continue
        nbrs = list(neighbors(shape, x))
        new[x] = (W[x] + sum(W[y] for y in nbrs)) % modulus
        scan += 1 + len(nbrs)
    return new, scan, None


def component(shape, W: Dense, p):
    value = W[p]
    members = [p]
    seen = {p}
    boundary = set()
    head = 0
    while head < len(members):
        x = members[head]
        head += 1
        for y in neighbors(shape, x):
            if y in seen:
                continue
            if W[y] == value:
                seen.add(y)
                members.append(y)
            else:
                boundary.add(y)
    return members, boundary


def relabel(shape, W: Dense, p, v):
    members, boundary = component(shape, W, p)
    new = dict(W)
    for x in members:
        new[x] = v
    return new, len(members) + len(boundary), None


def layer(shape, W: Dense, p, s):
    if W[p] == 0:
        return dict(W), 1, None
    members, boundary = component(shape, W, p)
    new = dict(W)
    if s == -1:
        for x in members:
            if any(W[y] == 0 for y in neighbors(shape, x)):
                new[x] = 0
    else:
        for y in boundary:
            if W[y] == 0:
                new[y] = W[p]
    return new, len(members) + len(boundary), None


def mix(shape, modulus: int, W: Dense, p, r):
    """Depth-limited breadth-first region through nonzero cells, then a modular sum kept nonzero."""

    if W[p] == 0:
        return dict(W), 1, None
    dist = {p: 0}
    frontier = [p]
    for depth in range(r):
        nxt = []
        for x in frontier:
            for y in neighbors(shape, x):
                if y not in dist and W[y] != 0:
                    dist[y] = depth + 1
                    nxt.append(y)
        if not nxt:
            break
        frontier = nxt
    residues = modulus - 1
    new = dict(W)
    scan = 0
    for x in dist:
        nbrs = list(neighbors(shape, x))
        new[x] = 1 + (W[x] + sum(W[y] for y in nbrs)) % residues
        scan += 1 + len(nbrs)
    return new, scan, None


def move(shape, W: Dense, v, axis, direction, k):
    """Global-order implementation: sort every entity by its axis coordinate."""

    entities = [x for x in positions(shape) if W[x] == v]
    entities.sort(key=lambda x: x[axis], reverse=(direction == 1))
    new = dict(W)
    checks = 0
    for start in entities:
        current = start
        for _ in range(k):
            nxt = list(current)
            nxt[axis] += direction
            if not in_bounds(shape, tuple(nxt)):
                break
            checks += 1
            nxt_t = tuple(nxt)
            if new[nxt_t] != 0:
                break
            new[nxt_t] = v
            new[current] = 0
            current = nxt_t
    return new, len(entities) + checks, None


def compress(shape, allowed: frozenset[int], W: Dense, axis, direction):
    new = dict(W)
    others = [range(size) for a, size in enumerate(shape) if a != axis]
    for combo in product(*others):
        line = []
        for x in range(shape[axis]):
            coord = list(combo)
            coord.insert(axis, x)
            line.append(tuple(coord))
        if direction == 1:
            line.reverse()
        values = [W[x] for x in line if W[x] != 0]
        packed = []
        i = 0
        while i < len(values):
            if i + 1 < len(values) and values[i] == values[i + 1]:
                merged = values[i] * 2
                if merged not in allowed:
                    raise ValueError(f"oracle overflow {merged}")
                packed.append(merged)
                i += 2
            else:
                packed.append(values[i])
                i += 1
        for slot, x in enumerate(line):
            new[x] = packed[slot] if slot < len(packed) else 0
    total = 1
    for size in shape:
        total *= size
    return new, total, None


def apply_oracle(family_id: str, world: WorldSpec, W: Dense, event: Event):
    shape = world.shape
    p = event.params
    if event.op == "SET":
        new = dict(W)
        new[tuple(p["p"])] = p["v"]
        return new, 1, None
    if family_id == "metric_paint":
        return paint(shape, W, tuple(p["a"]), p["r"], p["u"], p["v"])
    if family_id == "box_reflection":
        return reflect(shape, W, tuple(p["lo"]), tuple(p["hi"]), p["axis"])
    if family_id == "local_stencil":
        return stencil(shape, len(world.values), W, tuple(p["a"]), p["r"])
    if family_id == "occluded_mix":
        return mix(shape, len(world.values), W, tuple(p["p"]), p["r"])
    if family_id == "component_relabel":
        return relabel(shape, W, tuple(p["p"]), p["v"])
    if family_id == "boundary_layer":
        return layer(shape, W, tuple(p["p"]), p["s"])
    if family_id == "obstructed_motion":
        return move(shape, W, p["v"], p["axis"], p["direction"], p["k"])
    if family_id == "aggregating_compression":
        return compress(shape, world.allowed_values, W, p["axis"], p["direction"])
    raise KeyError(family_id)


# ----------------------------------------------------------------- random data
FAMILY_VALUES = {
    "metric_paint": (0, 1, 2, 3),
    "box_reflection": (0, 1, 2, 3),
    "local_stencil": (0, 1, 2, 3),
    "occluded_mix": (0, 1, 2, 3),
    "component_relabel": (0, 1, 2, 3),
    "boundary_layer": (0, 1, 2, 3),
    "obstructed_motion": (0, 1, 2, 3),
    "aggregating_compression": (0, 1, 2, 4, 8),
}


def random_state(rng: Random, world: WorldSpec, density: float, initial_values=None) -> WorldState:
    choices = initial_values or [v for v in world.values if v != 0]
    cells = []
    for x in positions(world.shape):
        if rng.random() < density:
            cells.append((x, rng.choice(choices)))
    return WorldState(world, tuple(cells))


def random_coord(rng: Random, shape) -> list[int]:
    return [rng.randrange(size) for size in shape]


def random_event(rng: Random, family_id: str, world: WorldSpec) -> Event:
    shape = world.shape
    values = world.values
    nonzero = [v for v in values if v]
    if family_id == "metric_paint":
        return Event(
            "PAINT",
            {
                "a": random_coord(rng, shape),
                "r": rng.randrange(0, 4),
                "u": rng.choice(values),
                "v": rng.choice(values),
            },
        )
    if family_id == "box_reflection":
        lo = random_coord(rng, shape)
        hi = [rng.randrange(low, size) for low, size in zip(lo, shape, strict=True)]
        return Event("REFLECT", {"lo": lo, "hi": hi, "axis": rng.randrange(len(shape))})
    if family_id == "local_stencil":
        radius = sum(shape) if rng.random() < 0.2 else rng.randrange(0, 4)
        return Event("STENCIL", {"a": random_coord(rng, shape), "r": radius})
    if family_id == "occluded_mix":
        radius = sum(shape) if rng.random() < 0.2 else rng.randrange(0, 4)
        return Event("MIX", {"p": random_coord(rng, shape), "r": radius})
    if family_id == "component_relabel":
        return Event("RELABEL", {"p": random_coord(rng, shape), "v": rng.choice(values)})
    if family_id == "boundary_layer":
        return Event("LAYER", {"p": random_coord(rng, shape), "s": rng.choice((-1, 1))})
    if family_id == "obstructed_motion":
        return Event(
            "MOVE",
            {
                "v": rng.choice(nonzero),
                "axis": rng.randrange(len(shape)),
                "direction": rng.choice((-1, 1)),
                "k": rng.randrange(0, 5),
            },
        )
    if family_id == "aggregating_compression":
        return Event(
            "COMPRESS", {"axis": rng.randrange(len(shape)), "direction": rng.choice((-1, 1))}
        )
    raise KeyError(family_id)
