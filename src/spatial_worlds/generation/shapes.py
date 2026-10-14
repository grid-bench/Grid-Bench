"""World-shape tiers and per-shape structural statistics."""

from __future__ import annotations

from math import prod

from ..grid import shape_tables

# Shapes lock the total cell count N (approximately) across dimensions with
# sides as balanced as possible.  Keys are the nominal N level; ``S``, ``M``
# and ``L`` are the aliases used by per-event families.
LEVEL_TABLES: dict[str, dict[int, tuple[int, ...]]] = {
    "81": {1: (81,), 2: (9, 9), 3: (4, 4, 5), 4: (3, 3, 3, 3)},
    "150": {1: (150,), 2: (12, 13), 3: (5, 5, 6), 4: (3, 3, 4, 4)},
    "256": {1: (256,), 2: (16, 16), 3: (6, 6, 7), 4: (4, 4, 4, 4), 5: (3, 3, 3, 3, 3)},
    "512": {1: (512,), 2: (22, 23), 3: (8, 8, 8), 4: (5, 5, 5, 4), 5: (3, 3, 4, 4, 4)},
    "1024": {1: (1024,), 2: (32, 32), 3: (10, 10, 10), 4: (6, 6, 6, 5), 5: (4, 4, 4, 4, 4)},
    "2048": {1: (2048,), 2: (45, 45), 3: (13, 13, 12), 4: (7, 7, 7, 6), 5: (5, 5, 5, 4, 4)},
}

TIER_ALIASES: dict[str, str] = {"S": "81", "M": "256", "L": "1024"}

# N levels used as the workload axis of the global families per tier.
GLOBAL_LEVELS: dict[str, tuple[int, ...]] = {
    "S": (81, 150, 256),
    "M": (150, 256, 512),
    "L": (512, 1024, 2048),
}


def resolve_level(level: str | int) -> str:
    key = str(level)
    key = TIER_ALIASES.get(key, key)
    if key not in LEVEL_TABLES:
        raise KeyError(f"unknown shape level {level!r}")
    return key


def shape_for(level: str | int, dimension: int) -> tuple[int, ...]:
    table = LEVEL_TABLES[resolve_level(level)]
    try:
        return table[dimension]
    except KeyError as error:
        raise KeyError(f"level {level!r} has no shape for dimension {dimension}") from error


def available_dimensions(level: str | int) -> tuple[int, ...]:
    return tuple(sorted(LEVEL_TABLES[resolve_level(level)]))


def shape_stats(shape: tuple[int, ...]) -> dict[str, float | int]:
    tables = shape_tables(shape)
    size = prod(shape)
    return {
        "cells": size,
        "edge_count": tables.edge_count,
        "boundary_cell_fraction": sum(tables.boundary) / size,
        "mean_neighbor_count": 2 * tables.edge_count / size,
        "stencil_scan": size + 2 * tables.edge_count,
        "min_side": min(shape),
    }
