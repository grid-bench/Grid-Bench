"""Obstructed Motion: ordered entity transport along an axis with obstruction."""

from __future__ import annotations

from random import Random

from ..family import (
    Example,
    Family,
    FamilyMetadata,
    StepResult,
    axis_param,
    integer_param,
    require_params,
    sign_param,
    validate_discrete_alphabet,
    value_param,
)
from ..grid import Grid
from ..schema import Event, SchemaError, WorldSpec
from ._common import nonzero_labels

MAX_PROPOSED_STEPS = 4


def _validate(world: WorldSpec, event: Event) -> None:
    require_params(event, "v", "axis", "direction", "k")
    value = value_param(world, event, "v")
    if value == 0:
        raise SchemaError("MOVE.v must be nonzero")
    axis_param(world, event)
    sign_param(event)
    integer_param(event, "k", minimum=0)


def _step(grid: Grid, event: Event) -> StepResult:
    params = event.params
    v = params["v"]
    axis = params["axis"]
    direction = params["direction"]
    limit = params["k"]
    data = grid.data
    entities = 0
    moved_entities = 0
    distance = 0
    checks = 0
    for fiber in grid.fibers(axis):
        length = len(fiber)
        order = range(length - 1, -1, -1) if direction == 1 else range(length)
        for position in order:
            if data[fiber[position]] != v:
                continue
            entities += 1
            current = position
            steps = 0
            while steps < limit:
                target = current + direction
                if target < 0 or target >= length:
                    break
                checks += 1
                if data[fiber[target]] != 0:
                    break
                data[fiber[target]] = v
                data[fiber[current]] = 0
                current = target
                steps += 1
            if steps:
                moved_entities += 1
                distance += steps
    return StepResult(
        scan=entities + checks,
        extras={
            "entities": entities,
            "moved": moved_entities,
            "distance": distance,
            "checks": checks,
        },
    )


def _propose(rng: Random, grid: Grid, target: int, want_effective: bool, k: int) -> list[Event]:
    del target, want_effective
    counts = grid.value_counts()
    labels = [label for label in nonzero_labels(grid.world) if counts.get(label, 0) > 0]
    if not labels:
        return []
    events: list[Event] = []
    for v in labels:
        for axis, size in enumerate(grid.shape):
            for direction in (-1, 1):
                for steps in range(1, min(MAX_PROPOSED_STEPS, size - 1) + 1):
                    events.append(
                        Event("MOVE", {"v": v, "axis": axis, "direction": direction, "k": steps})
                    )
    rng.shuffle(events)
    return events[: max(k, 1) * 2]


FAMILY = Family(
    metadata=FamilyMetadata(
        id="obstructed_motion",
        name="Obstructed Motion",
        status="implemented",
        dependency_structure="Occupancy-constrained ordered paths",
        summary="Move selected entities along an axis subject to boundaries and obstruction.",
        event_ops=("SET", "MOVE"),
        support_source="state",
        update_type="transport",
        order_dependent=True,
        predicate=None,
    ),
    rule=(
        "MOVE(v,axis,direction,k): take every cell with value v, ordered by decreasing "
        "coordinate on axis if direction=1 and increasing if direction=-1, so that the cells "
        "nearest the leading edge are handled first. In that order, move each one along axis "
        "in the sign of direction, one cell at a time and at most k cells, stopping as soon as "
        "the next cell is outside the world or nonzero (including cells already moved into). "
        "Each move clears the old cell and writes v to the new cell; values never merge, and a "
        "block of adjacent v cells therefore moves as a unit."
    ),
    event_validator=_validate,
    step_fn=_step,
    proposer=_propose,
    world_validator=validate_discrete_alphabet,
    example=Example(
        before=(1, 1, 0, 0, 2, 0, 1),
        event=Event("MOVE", {"v": 1, "axis": 0, "direction": 1, "k": 2}),
        note=(
            "The cells holding 1 are handled from the highest coordinate down. [6] cannot move "
            "because the next cell is outside the world; [1] moves to [3]; then [0] moves to "
            "[2] through the cell that [1] vacated. The 2 at [4] is never moved."
        ),
    ),
)
