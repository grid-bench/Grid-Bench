"""Immutable finite-lattice world state."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import product

from .schema import SchemaError, WorldSpec

Coordinate = tuple[int, ...]
Cell = tuple[Coordinate, int]


@dataclass(frozen=True, slots=True)
class WorldState:
    """A world stored canonically as sorted nonzero coordinate-value pairs."""

    world: WorldSpec
    cells: tuple[Cell, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.world, WorldSpec):
            raise SchemaError("WorldState.world must be a WorldSpec")

        raw_cells: Iterable[object]
        if isinstance(self.cells, Mapping):
            raw_cells = self.cells.items()
        else:
            raw_cells = self.cells

        normalized: list[Cell] = []
        seen: set[Coordinate] = set()
        for index, raw_cell in enumerate(raw_cells):
            if not isinstance(raw_cell, (list, tuple)) or len(raw_cell) != 2:
                raise SchemaError(f"cells[{index}] must be [coordinate, value]")
            position = self.world.validate_position(raw_cell[0], label=f"cells[{index}][0]")
            value = self.world.validate_value(raw_cell[1], label=f"cells[{index}][1]")
            if value == 0:
                raise SchemaError("explicit zero-valued cells are not allowed")
            if position in seen:
                raise SchemaError(f"duplicate cell coordinate: {position}")
            seen.add(position)
            normalized.append((position, value))

        normalized.sort(key=lambda cell: cell[0])
        object.__setattr__(self, "cells", tuple(normalized))

    @classmethod
    def empty(cls, world: WorldSpec) -> WorldState:
        return cls(world)

    @classmethod
    def from_dict(cls, data: object, world: WorldSpec) -> WorldState:
        state, _ = cls.parse_dict(data, world)
        return state

    @classmethod
    def parse_dict(cls, data: object, world: WorldSpec) -> tuple[WorldState, bool]:
        """Parse a state and report whether its input cell order was canonical."""

        if not isinstance(data, Mapping):
            raise SchemaError("state must be a JSON object")
        if set(data) != {"cells"}:
            missing = {"cells"} - set(data)
            extra = set(data) - {"cells"}
            if missing:
                raise SchemaError("state is missing key: cells")
            raise SchemaError(f"state has unknown keys: {', '.join(sorted(extra))}")
        raw_cells = data["cells"]
        if not isinstance(raw_cells, (list, tuple)):
            raise SchemaError("state.cells must be an array")

        coordinates: list[Coordinate] = []
        for index, raw_cell in enumerate(raw_cells):
            if not isinstance(raw_cell, (list, tuple)) or len(raw_cell) != 2:
                raise SchemaError(f"cells[{index}] must be [coordinate, value]")
            coordinates.append(world.validate_position(raw_cell[0], label=f"cells[{index}][0]"))
        canonical = all(
            left < right for left, right in zip(coordinates, coordinates[1:], strict=False)
        )
        return cls(world, tuple(raw_cells)), canonical

    def to_dict(self) -> dict[str, list[list[object]]]:
        return {"cells": [[list(position), value] for position, value in self.cells]}

    def as_dict(self) -> dict[Coordinate, int]:
        return dict(self.cells)

    def get(self, position: Sequence[int]) -> int:
        coordinate = self.world.validate_position(position)
        return self.as_dict().get(coordinate, 0)

    def __getitem__(self, position: Sequence[int]) -> int:
        return self.get(position)

    def set(self, position: Sequence[int], value: int) -> WorldState:
        coordinate = self.world.validate_position(position)
        checked_value = self.world.validate_value(value)
        cells = self.as_dict()
        if checked_value == 0:
            cells.pop(coordinate, None)
        else:
            cells[coordinate] = checked_value
        return WorldState(self.world, tuple(cells.items()))

    def set_many(self, updates: Iterable[tuple[Sequence[int], int]]) -> WorldState:
        cells = self.as_dict()
        for position, value in updates:
            coordinate = self.world.validate_position(position)
            checked_value = self.world.validate_value(value)
            if checked_value == 0:
                cells.pop(coordinate, None)
            else:
                cells[coordinate] = checked_value
        return WorldState(self.world, tuple(cells.items()))

    @property
    def nonzero_count(self) -> int:
        return len(self.cells)

    @property
    def density(self) -> float:
        return self.nonzero_count / self.world.volume

    def positions(self) -> Iterator[Coordinate]:
        ranges = (range(size) for size in self.world.shape)
        yield from product(*ranges)

    def dense_values(self) -> tuple[int, ...]:
        cells = self.as_dict()
        return tuple(cells.get(position, 0) for position in self.positions())
