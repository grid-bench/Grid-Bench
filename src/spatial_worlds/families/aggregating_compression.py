"""Aggregating Compression: pack and pairwise-merge along axis fibers."""

from __future__ import annotations

from random import Random

from ..family import (
    Example,
    Family,
    FamilyMetadata,
    StepResult,
    axis_param,
    require_params,
    sign_param,
    validate_positive_alphabet,
)
from ..grid import Grid
from ..schema import Event, SchemaError, WorldSpec


def _validate(world: WorldSpec, event: Event) -> None:
    require_params(event, "axis", "direction")
    axis_param(world, event)
    sign_param(event)


def _step(grid: Grid, event: Event) -> StepResult:
    params = event.params
    axis = params["axis"]
    direction = params["direction"]
    data = grid.data
    allowed = grid.world.allowed_values
    merges = 0
    for fiber in grid.fibers(axis):
        ordered = fiber if direction == -1 else fiber[::-1]
        values = [data[index] for index in ordered if data[index] != 0]
        packed: list[int] = []
        position = 0
        while position < len(values):
            if position + 1 < len(values) and values[position] == values[position + 1]:
                merged = values[position] + values[position + 1]
                if merged not in allowed:
                    raise SchemaError(f"COMPRESS produced value {merged} outside world.values")
                packed.append(merged)
                position += 2
                merges += 1
            else:
                packed.append(values[position])
                position += 1
        for slot, index in enumerate(ordered):
            data[index] = packed[slot] if slot < len(packed) else 0
    return StepResult(scan=grid.size, extras={"merges": merges})


def _propose(rng: Random, grid: Grid, target: int, want_effective: bool, k: int) -> list[Event]:
    del target, want_effective, k
    events = [
        Event("COMPRESS", {"axis": axis, "direction": direction})
        for axis in range(grid.dimension)
        for direction in (-1, 1)
    ]
    rng.shuffle(events)
    return events


FAMILY = Family(
    metadata=FamilyMetadata(
        id="aggregating_compression",
        name="Aggregating Compression",
        status="implemented",
        dependency_structure="Ordered axis fibers",
        summary="Globally compact axis-aligned fibers and merge equal adjacent values.",
        event_ops=("SET", "COMPRESS"),
        support_source="geometry",
        update_type="aggregate",
        order_dependent=True,
        predicate=None,
    ),
    rule=(
        "COMPRESS(axis,direction): on every line parallel to axis, independently: list the "
        "nonzero values in order starting from the destination edge (coordinate 0 if "
        "direction=-1, the highest coordinate if direction=1). Walk that list from its start; "
        "whenever two consecutive values are equal, replace them by one value equal to their "
        "sum and continue after the pair, so a merged value cannot merge again in this event. "
        "Write the resulting list back starting at the destination edge and set every "
        "remaining cell of the line to 0."
    ),
    event_validator=_validate,
    step_fn=_step,
    proposer=_propose,
    world_validator=validate_positive_alphabet,
    example=Example(
        before=(1, 1, 2, 0, 1, 0),
        event=Event("COMPRESS", {"axis": 0, "direction": -1}),
        note=(
            "The nonzero values read from the destination edge are 1 1 2 1. The first pair "
            "merges into 2; that merged 2 is not merged again with the 2 that follows it; the "
            "final 1 is kept. The result 2 2 1 is written from cell [0] and the remaining "
            "cells become 0."
        ),
    ),
)
