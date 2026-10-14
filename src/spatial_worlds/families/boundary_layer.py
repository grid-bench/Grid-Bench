"""Boundary Layer: strip or add one layer of a component along its empty boundary."""

from __future__ import annotations

from random import Random

from ..family import (
    Example,
    Family,
    FamilyMetadata,
    StepResult,
    coordinate_param,
    require_params,
    sign_param,
    validate_discrete_alphabet,
)
from ..grid import Grid
from ..schema import Event, WorldSpec
from ._common import closest, log_distance


def _validate(world: WorldSpec, event: Event) -> None:
    require_params(event, "p", "s")
    coordinate_param(world, event, "p")
    sign_param(event, "s")


def _step(grid: Grid, event: Event) -> StepResult:
    params = event.params
    index = grid.index(params["p"])
    s = params["s"]
    data = grid.data
    u = data[index]
    if u == 0:
        return StepResult(scan=1, extras={"empty_anchor": 1})
    members, boundary = grid.component(index)
    table = grid.neighbor_table
    liberties = [other for other in boundary if data[other] == 0]
    if s == -1:
        exposed = [member for member in members if any(data[other] == 0 for other in table[member])]
        for member in exposed:
            data[member] = 0
        changed = len(exposed)
    else:
        for other in liberties:
            data[other] = u
        changed = len(liberties)
    return StepResult(
        scan=len(members) + len(boundary),
        extras={
            "component_size": len(members),
            "boundary_size": len(boundary),
            "liberties": len(liberties),
            "changed": changed,
        },
    )


def _propose(rng: Random, grid: Grid, target: int, want_effective: bool, k: int) -> list[Event]:
    data = grid.data
    _, components = grid.label_components()
    scored: list[tuple[float, int]] = []
    candidates: list[tuple[list[int], int]] = []
    for members in components:
        if data[members[0]] == 0:
            continue
        boundary = grid.boundary_of(members)
        liberties = sum(1 for other in boundary if data[other] == 0)
        if (liberties > 0) != want_effective:
            continue
        scan = len(members) + len(boundary)
        # An enclosed component is a no-op for both signs; it is offered only
        # when verifying it costs about the target, never as a cheap event.
        if not want_effective and 2 * scan < target:
            continue
        scored.append((log_distance(scan, target), len(candidates)))
        candidates.append((members, liberties))
    events: list[Event] = []
    for position in closest(scored, k, rng):
        members, _ = candidates[position]
        anchor = grid.coord(rng.choice(members))
        # Both signs are offered; the selector's density term decides between
        # them when the band is at stake and the seeded tie-break otherwise.
        for s in (-1, 1):
            events.append(Event("LAYER", {"p": list(anchor), "s": s}))
    return events


FAMILY = Family(
    metadata=FamilyMetadata(
        id="boundary_layer",
        name="Boundary Layer",
        status="implemented",
        dependency_structure="Connected component and its boundary",
        summary="Strip the exposed layer of a component, or grow it into its empty neighbours.",
        event_ops=("SET", "LAYER"),
        support_source="state",
        update_type="relabel",
        order_dependent=False,
        predicate=None,
    ),
    rule=(
        "LAYER(p,s): if W(p) = 0, nothing changes. Otherwise let u = W(p) and find the maximal "
        "Manhattan-connected set of cells with old value u, including p. If s = -1, set to 0 "
        "every cell of that set that has an in-bounds neighbor with old value 0. If s = +1, set "
        "to u every in-bounds cell with old value 0 that is a neighbor of a cell of that set. "
        "Positions outside the world are not neighbors; nothing else changes."
    ),
    event_validator=_validate,
    step_fn=_step,
    proposer=_propose,
    world_validator=validate_discrete_alphabet,
    example=Example(
        before=(0, 1, 1, 1, 2, 0, 1),
        event=Event("LAYER", {"p": [2], "s": -1}),
        note=(
            "The cells with value 1 connected to [2] are [1], [2], and [3]. Cell [1] has the "
            "neighbor [0] with value 0, so it becomes 0; [2] and [3] have no neighbor with "
            'value 0 and keep their value. With s = +1 the event {"op":"LAYER","p":[2],"s":1} '
            "would instead set [0] to 1. The 1 at [6] is a different set and is never touched."
        ),
    ),
)
