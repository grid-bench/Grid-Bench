"""Component Relabel: rewrite a maximal equal-valued axis-connected component."""

from __future__ import annotations

from random import Random

from ..family import (
    Example,
    Family,
    FamilyMetadata,
    StepResult,
    coordinate_param,
    require_params,
    validate_discrete_alphabet,
    value_param,
)
from ..grid import Grid
from ..schema import Event, WorldSpec
from ._common import closest, log_distance, nonzero_labels


def _validate(world: WorldSpec, event: Event) -> None:
    require_params(event, "p", "v")
    coordinate_param(world, event, "p")
    value_param(world, event, "v")


def _step(grid: Grid, event: Event) -> StepResult:
    params = event.params
    index = grid.index(params["p"])
    v = params["v"]
    members, boundary = grid.component(index)
    data = grid.data
    # Boundary cells that already hold v: after the write the component fuses
    # with their components, which is what makes the realized workload drift.
    merges = 0
    if v != 0 and v != data[index]:
        merges = sum(1 for other in boundary if data[other] == v)
    for member in members:
        data[member] = v
    return StepResult(
        scan=len(members) + len(boundary),
        extras={
            "component_size": len(members),
            "boundary_size": len(boundary),
            "merge_contacts": merges,
        },
    )


def _propose(rng: Random, grid: Grid, target: int, want_effective: bool, k: int) -> list[Event]:
    # A relabel to the component's own value would change nothing, but its
    # outcome is certified by the anchor alone; the proposer therefore offers
    # no ineffective events. Such events remain valid under the rule.
    if not want_effective:
        return []
    labels = nonzero_labels(grid.world)
    data = grid.data
    _, components = grid.label_components()
    scored: list[tuple[float, int]] = []
    candidates: list[list[int]] = []
    for members in components:
        if data[members[0]] == 0:
            continue
        scan = len(members) + len(grid.boundary_of(members))
        scored.append((log_distance(scan, target), len(candidates)))
        candidates.append(members)
    nonzero = grid.nonzero_count
    events: list[Event] = []
    for position in closest(scored, k, rng):
        members = candidates[position]
        anchor = grid.coord(rng.choice(members))
        u = data[members[0]]
        others = [label for label in labels if label != u]
        touching = {data[other] for other in grid.boundary_of(members)} - {0, u}
        separate = [label for label in others if label not in touching]
        can_delete = 4 * len(members) <= nonzero
        # Prefer not to relabel to a label present on the boundary: that fuses the
        # component with its neighbours and the realized workload drifts
        # far above the target.  Deleting (v = 0) is the fallback; a
        # merging label is chosen only when nothing else is possible.
        if rng.random() < 0.15 and can_delete:
            v = 0
        elif separate:
            v = rng.choice(separate)
        elif can_delete:
            v = 0
        else:
            v = rng.choice(others or [0])
        events.append(Event("RELABEL", {"p": list(anchor), "v": v}))
    return events


FAMILY = Family(
    metadata=FamilyMetadata(
        id="component_relabel",
        name="Component Relabel",
        status="implemented",
        dependency_structure="State-induced connected component",
        summary="Relabel a maximal equal-valued component under axis adjacency.",
        event_ops=("SET", "RELABEL"),
        support_source="state",
        update_type="relabel",
        order_dependent=False,
        predicate=None,
    ),
    rule=(
        "RELABEL(p,v): let u = W(p) (u may be 0). Find the maximal set of cells connected to "
        "p by Manhattan-distance-1 steps whose old values are all u, including p. Set every "
        "cell of that set to v."
    ),
    event_validator=_validate,
    step_fn=_step,
    proposer=_propose,
    world_validator=validate_discrete_alphabet,
    example=Example(
        before=(1, 1, 0, 1, 2, 2, 3),
        event=Event("RELABEL", {"p": [4], "v": 3}),
        note=(
            "The cells with old value 2 connected to [4] are [4] and [5]; both become 3. The 1 "
            "at [3] and the 3 at [6] are not part of that set, although [6] ends up adjacent "
            "to the new 3s."
        ),
    ),
)
