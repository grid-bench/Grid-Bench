"""Metric Paint: conditional relabeling inside a Manhattan ball."""

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
    value_param,
)
from ..grid import Grid
from ..schema import Event, WorldSpec
from ._common import ball_radii, least_frequent_label, nonzero_labels


def _validate(world: WorldSpec, event: Event) -> None:
    require_params(event, "a", "r", "u", "v")
    coordinate_param(world, event, "a")
    integer_param(event, "r", minimum=0)
    value_param(world, event, "u")
    value_param(world, event, "v")


def _step(grid: Grid, event: Event) -> StepResult:
    params = event.params
    ball = grid.l1_ball(params["a"], params["r"])
    u = params["u"]
    v = params["v"]
    data = grid.data
    selected = [index for index in ball if data[index] == u]
    if u != v:
        for index in selected:
            data[index] = v
    return StepResult(
        scan=len(ball),
        predicate=bool(selected),
        extras={"ball": len(ball), "selected": len(selected)},
    )


def _propose(rng: Random, grid: Grid, target: int, want_effective: bool, k: int) -> list[Event]:
    radii = ball_radii(grid, target)
    labels = nonzero_labels(grid.world)
    values = grid.world.values
    data = grid.data
    events: list[Event] = []
    attempts = 0
    while len(events) < k and attempts < k * 8:
        attempts += 1
        center = grid.coord(rng.randrange(grid.size))
        radius = rng.choice(radii)
        present = {data[index] for index in grid.l1_ball(center, radius)}
        if want_effective:
            u = rng.choice(sorted(present))
        else:
            absent = [value for value in values if value not in present]
            if not absent:
                continue
            u = rng.choice(absent)
        others = [value for value in values if value != u]
        v: int | None = None
        if rng.random() < 0.5:
            v = least_frequent_label(grid, labels, exclude=u)
        if v is None:
            v = rng.choice(others)
        events.append(Event("PAINT", {"a": list(center), "r": radius, "u": u, "v": v}))
    return events


FAMILY = Family(
    metadata=FamilyMetadata(
        id="metric_paint",
        name="Metric Paint",
        status="implemented",
        dependency_structure="Manhattan metric ball",
        summary="Conditionally rewrite cells selected by distance from an event center.",
        event_ops=("SET", "PAINT"),
        support_source="geometry",
        update_type="relabel",
        order_dependent=False,
        predicate="the ball contains at least one cell with value u",
    ),
    rule=(
        "PAINT(a,r,u,v): simultaneously replace u by v at every coordinate x "
        "with sum_i |x_i-a_i| <= r. All other cells remain unchanged."
    ),
    event_validator=_validate,
    step_fn=_step,
    proposer=_propose,
    world_validator=validate_discrete_alphabet,
    example=Example(
        before=(1, 2, 0, 1, 3, 1, 0),
        event=Event("PAINT", {"a": [3], "r": 2, "u": 1, "v": 3}),
        note=(
            "The ball is cells [1] to [5]. Among them, [3] and [5] hold 1 and become 3; the 2 "
            "at [1] and the 3 at [4] are unchanged. Cell [0] also holds 1 but lies outside the "
            "ball and is unchanged."
        ),
    ),
)
