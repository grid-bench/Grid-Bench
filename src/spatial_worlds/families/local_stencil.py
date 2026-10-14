"""Local Stencil: synchronous modular sum over axis neighbourhoods inside a ball."""

from __future__ import annotations

from random import Random

from ..family import (
    Example,
    Family,
    FamilyMetadata,
    StepResult,
    coordinate_param,
    integer_param,
    require_params,
    validate_discrete_alphabet,
)
from ..grid import Grid
from ..schema import Event, WorldSpec
from ._common import ball_radii, covering_radius


def _validate(world: WorldSpec, event: Event) -> None:
    require_params(event, "a", "r")
    coordinate_param(world, event, "a")
    integer_param(event, "r", minimum=0)


def _rule(world: WorldSpec) -> str:
    modulus = len(world.values)
    return (
        "STENCIL(a,r): simultaneously set every cell x with sum_i |x_i-a_i| <= r to "
        f"(W(x) + sum W(y)) mod {modulus}, where y ranges over the in-bounds cells at "
        "Manhattan distance 1 from x, whether or not they lie within distance r of a. "
        "Every update reads the same old world; cells farther than r from a remain unchanged."
    )


def _step(grid: Grid, event: Event) -> StepResult:
    params = event.params
    modulus = len(grid.world.values)
    ball = grid.l1_ball(params["a"], params["r"])
    data = grid.data
    table = grid.neighbor_table
    reads = 0
    updated: list[int] = []
    for index in ball:
        neighbours = table[index]
        reads += len(neighbours)
        updated.append((data[index] + sum(data[other] for other in neighbours)) % modulus)
    for index, value in zip(ball, updated, strict=True):
        data[index] = value
    return StepResult(scan=len(ball) + reads, extras={"support": len(ball), "reads": reads})


def _propose(rng: Random, grid: Grid, target: int, want_effective: bool, k: int) -> list[Event]:
    del want_effective
    if target >= grid.size:
        return [Event("STENCIL", {"a": [0] * grid.dimension, "r": covering_radius(grid.shape)})]
    radii = ball_radii(grid, target)
    events: list[Event] = []
    for _ in range(max(k, 1)):
        center = grid.coord(rng.randrange(grid.size))
        events.append(Event("STENCIL", {"a": list(center), "r": rng.choice(radii)}))
    return events


def _note(world: WorldSpec, before: tuple[int, ...], after: tuple[int, ...]) -> str:
    modulus = len(world.values)

    def terms(index: int) -> str:
        parts = [before[index]] + [
            before[other] for other in (index - 1, index + 1) if 0 <= other < len(before)
        ]
        return "+".join(str(part) for part in parts)

    return (
        "Only cells [2], [3], and [4] lie within distance 1 of [3]. "
        f"Cell [2] becomes ({terms(2)}) mod {modulus} = {after[2]} from its own value and its "
        f"neighbours [1] and [3]; cell [3] becomes ({terms(3)}) mod {modulus} = {after[3]}; "
        f"cell [4] becomes ({terms(4)}) mod {modulus} = {after[4]}. "
        "Cells [1] and [5] are read as neighbours but are not changed."
    )


FAMILY = Family(
    metadata=FamilyMetadata(
        id="local_stencil",
        name="Local Stencil",
        status="implemented",
        dependency_structure="Axis neighbourhoods of every cell in a Manhattan ball",
        summary="Synchronously propagate integer state through local neighbourhoods in a ball.",
        event_ops=("SET", "STENCIL"),
        support_source="geometry",
        update_type="aggregate",
        order_dependent=False,
        predicate=None,
    ),
    rule=_rule,
    event_validator=_validate,
    step_fn=_step,
    proposer=_propose,
    world_validator=validate_discrete_alphabet,
    example=Example(
        before=(1, 2, 0, 1, 3, 0, 2),
        event=Event("STENCIL", {"a": [3], "r": 1}),
        note=_note,
    ),
)
