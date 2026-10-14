"""Dense, mutable lattice used by the reference simulators and generators.

``WorldState`` is the immutable, canonical wire representation.  ``Grid`` is
the working representation: a flat row-major list of integers plus cached
per-shape index tables (coordinates, axis neighbours, fibers).  Simulators
mutate a ``Grid`` in place and convert back to ``WorldState`` only when a
state has to be serialized or compared.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from itertools import product
from math import prod

from .schema import SchemaError, WorldSpec
from .world import Cell, Coordinate, WorldState


class _ShapeTables:
    """Index tables shared by every grid with the same shape."""

    __slots__ = (
        "shape",
        "strides",
        "size",
        "coords",
        "neighbors",
        "boundary",
        "edge_count",
        "_fibers",
    )

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        self.size = prod(shape)
        strides: list[int] = []
        acc = 1
        for size in reversed(shape):
            strides.append(acc)
            acc *= size
        self.strides = tuple(reversed(strides))
        self.coords: tuple[Coordinate, ...] = tuple(product(*(range(size) for size in shape)))
        neighbors: list[tuple[int, ...]] = []
        boundary: list[bool] = []
        for index, coord in enumerate(self.coords):
            cell_neighbors: list[int] = []
            on_boundary = False
            for axis, (position, size) in enumerate(zip(coord, shape, strict=True)):
                stride = self.strides[axis]
                if position > 0:
                    cell_neighbors.append(index - stride)
                else:
                    on_boundary = True
                if position < size - 1:
                    cell_neighbors.append(index + stride)
                else:
                    on_boundary = True
            neighbors.append(tuple(cell_neighbors))
            boundary.append(on_boundary)
        self.neighbors = tuple(neighbors)
        self.boundary = tuple(boundary)
        self.edge_count = sum((size - 1) * (self.size // size) for size in shape)
        self._fibers: dict[int, tuple[tuple[int, ...], ...]] = {}

    def fibers(self, axis: int) -> tuple[tuple[int, ...], ...]:
        cached = self._fibers.get(axis)
        if cached is not None:
            return cached
        stride = self.strides[axis]
        length = self.shape[axis]
        other_ranges = [range(size) for a, size in enumerate(self.shape) if a != axis]
        other_strides = [s for a, s in enumerate(self.strides) if a != axis]
        fibers: list[tuple[int, ...]] = []
        for combo in product(*other_ranges):
            base = sum(c * s for c, s in zip(combo, other_strides, strict=True))
            fibers.append(tuple(base + x * stride for x in range(length)))
        result = tuple(fibers)
        self._fibers[axis] = result
        return result


_TABLES: dict[tuple[int, ...], _ShapeTables] = {}


def shape_tables(shape: Sequence[int]) -> _ShapeTables:
    key = tuple(shape)
    tables = _TABLES.get(key)
    if tables is None:
        tables = _ShapeTables(key)
        _TABLES[key] = tables
    return tables


class Grid:
    """Row-major dense array over a ``WorldSpec`` domain."""

    __slots__ = ("world", "data", "_t")

    def __init__(self, world: WorldSpec, data: list[int] | None = None) -> None:
        if not isinstance(world, WorldSpec):
            raise SchemaError("Grid.world must be a WorldSpec")
        self.world = world
        self._t = shape_tables(world.shape)
        if data is None:
            self.data = [0] * self._t.size
        else:
            if len(data) != self._t.size:
                raise SchemaError("Grid data length does not match the world volume")
            self.data = data

    # ----------------------------------------------------------------- basics
    @property
    def shape(self) -> tuple[int, ...]:
        return self._t.shape

    @property
    def strides(self) -> tuple[int, ...]:
        return self._t.strides

    @property
    def size(self) -> int:
        return self._t.size

    @property
    def dimension(self) -> int:
        return len(self._t.shape)

    @property
    def coords(self) -> tuple[Coordinate, ...]:
        return self._t.coords

    @classmethod
    def from_state(cls, state: WorldState) -> Grid:
        grid = cls(state.world)
        strides = grid._t.strides
        data = grid.data
        for position, value in state.cells:
            data[sum(p * s for p, s in zip(position, strides, strict=True))] = value
        return grid

    def to_cells(self) -> tuple[Cell, ...]:
        coords = self._t.coords
        return tuple((coords[i], v) for i, v in enumerate(self.data) if v)

    def to_state(self) -> WorldState:
        return WorldState(self.world, self.to_cells())

    def copy(self) -> Grid:
        return Grid(self.world, self.data[:])

    def index(self, coord: Sequence[int]) -> int:
        return sum(p * s for p, s in zip(coord, self._t.strides, strict=True))

    def coord(self, index: int) -> Coordinate:
        return self._t.coords[index]

    def in_bounds(self, coord: Sequence[int]) -> bool:
        return len(coord) == self.dimension and all(
            0 <= p < size for p, size in zip(coord, self._t.shape, strict=True)
        )

    def neighbors(self, index: int) -> tuple[int, ...]:
        return self._t.neighbors[index]

    @property
    def neighbor_table(self) -> tuple[tuple[int, ...], ...]:
        return self._t.neighbors

    def is_boundary_cell(self, index: int) -> bool:
        return self._t.boundary[index]

    @property
    def boundary_fraction(self) -> float:
        return sum(self._t.boundary) / self._t.size

    @property
    def mean_neighbor_count(self) -> float:
        return 2 * self._t.edge_count / self._t.size

    def edge_count(self) -> int:
        return self._t.edge_count

    @property
    def nonzero_count(self) -> int:
        return sum(1 for v in self.data if v)

    @property
    def density(self) -> float:
        return self.nonzero_count / self._t.size

    def nonzero_indices(self) -> list[int]:
        return [i for i, v in enumerate(self.data) if v]

    def zero_indices(self) -> list[int]:
        return [i for i, v in enumerate(self.data) if not v]

    def value_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for value in self.data:
            counts[value] = counts.get(value, 0) + 1
        return counts

    # ------------------------------------------------------------- structures
    def ray(self, start: Sequence[int], q: Sequence[int]) -> list[int]:
        """Indices of start+q, start+2q, ... while in bounds (start excluded)."""

        shape = self._t.shape
        strides = self._t.strides
        position = list(start)
        result: list[int] = []
        while True:
            for axis in range(len(position)):
                position[axis] += q[axis]
            if any(p < 0 or p >= shape[axis] for axis, p in enumerate(position)):
                return result
            result.append(sum(p * s for p, s in zip(position, strides, strict=True)))

    def l1_ball(self, center: Sequence[int], radius: int) -> list[int]:
        """Indices of in-bounds cells within Manhattan distance ``radius``."""

        shape = self._t.shape
        strides = self._t.strides
        dimension = len(shape)
        result: list[int] = []

        def recurse(axis: int, base: int, budget: int) -> None:
            if axis == dimension:
                result.append(base)
                return
            c = center[axis]
            lo = max(0, c - budget)
            hi = min(shape[axis] - 1, c + budget)
            stride = strides[axis]
            for x in range(lo, hi + 1):
                recurse(axis + 1, base + x * stride, budget - abs(x - c))

        recurse(0, 0, radius)
        return result

    def box(self, lo: Sequence[int], hi: Sequence[int]) -> list[int]:
        """Indices of the inclusive axis-aligned box, in row-major order."""

        strides = self._t.strides
        ranges = [range(a, b + 1) for a, b in zip(lo, hi, strict=True)]
        return [
            sum(p * s for p, s in zip(coord, strides, strict=True)) for coord in product(*ranges)
        ]

    def fibers(self, axis: int) -> tuple[tuple[int, ...], ...]:
        """Every line parallel to ``axis`` as indices in increasing coordinate order."""

        return self._t.fibers(axis)

    def component(self, index: int) -> tuple[list[int], list[int]]:
        """Maximal equal-valued axis-connected component and its external boundary."""

        data = self.data
        value = data[index]
        neighbors = self._t.neighbors
        members = [index]
        seen = {index}
        boundary: set[int] = set()
        head = 0
        while head < len(members):
            current = members[head]
            head += 1
            for other in neighbors[current]:
                if other in seen:
                    continue
                if data[other] == value:
                    seen.add(other)
                    members.append(other)
                else:
                    boundary.add(other)
        return members, sorted(boundary)

    def label_components(self) -> tuple[list[int], list[list[int]]]:
        """Label every cell with its component id and list the components."""

        labels = [-1] * self._t.size
        components: list[list[int]] = []
        data = self.data
        neighbors = self._t.neighbors
        for start in range(self._t.size):
            if labels[start] >= 0:
                continue
            label = len(components)
            value = data[start]
            members = [start]
            labels[start] = label
            head = 0
            while head < len(members):
                current = members[head]
                head += 1
                for other in neighbors[current]:
                    if labels[other] < 0 and data[other] == value:
                        labels[other] = label
                        members.append(other)
            components.append(members)
        return labels, components

    def boundary_of(self, members: Sequence[int]) -> list[int]:
        member_set = set(members)
        neighbors = self._t.neighbors
        boundary: set[int] = set()
        for index in members:
            for other in neighbors[index]:
                if other not in member_set:
                    boundary.add(other)
        return sorted(boundary)

    def iter_indices(self) -> Iterator[int]:
        return iter(range(self._t.size))


def hamming(before: Sequence[int], after: Sequence[int]) -> int:
    """Number of positions whose value differs."""

    return sum(1 for a, b in zip(before, after, strict=True) if a != b)


__all__ = ["Grid", "hamming", "shape_tables"]
