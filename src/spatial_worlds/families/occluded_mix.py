"""Occluded Mix: modular aggregation over the region reachable through nonzero cells."""

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
)
from ..grid import Grid
from ..schema import Event, SchemaError, WorldSpec
from ._common import closest, log_distance, nonzero_layers


def _validate(world: WorldSpec, event: Event) -> None:
    require_params(event, "p", "r")
    coordinate_param(world, event, "p")
    integer_param(event, "r", minimum=0)


def _validate_world(world: WorldSpec) -> None:
    """Require the alphabet 0, ..., M-1 with M >= 3, so that M-1 residues exist."""

    if len(world.values) < 3 or world.values != tuple(range(len(world.values))):
        raise SchemaError("family requires world.values = [0, 1, ..., M-1] with M >= 3")


def _residues(world: WorldSpec) -> int:
    return len(world.values) - 1


def _rule(world: WorldSpec) -> str:
    residues = _residues(world)
    return (
        "MIX(p,r): if W(p) = 0, nothing changes. Otherwise find every cell that can be reached "
        "from p by a path of at most r Manhattan-distance-1 steps whose cells all have nonzero "
        "old values (p itself counts as reached with 0 steps; a cell with old value 0 can never "
        "be entered or passed through). Then simultaneously set every reached cell x to "
        f"1 + ((W(x) + sum W(y)) mod {residues}), where y ranges over the in-bounds cells at "
        "Manhattan distance 1 from x, whether or not they were reached. Every update reads the "
        "old world; cells that were not reached remain unchanged."
    )


def _step(grid: Grid, event: Event) -> StepResult:
    params = event.params
    index = grid.index(params["p"])
    data = grid.data
    if data[index] == 0:
        return StepResult(scan=1, extras={"empty_anchor": 1, "support": 0})
    residues = _residues(grid.world)
    layers, _, _ = nonzero_layers(grid, index, params["r"])
    region = [cell for layer in layers for cell in layer]
    table = grid.neighbor_table
    reads = 0
    updated: list[int] = []
    for cell in region:
        neighbours = table[cell]
        reads += len(neighbours)
        updated.append(1 + (data[cell] + sum(data[other] for other in neighbours)) % residues)
    changed = 0
    for cell, value in zip(region, updated, strict=True):
        if data[cell] != value:
            data[cell] = value
            changed += 1
    return StepResult(
        scan=len(region) + reads,
        extras={"support": len(region), "reads": reads, "changed": changed},
    )


def _propose(rng: Random, grid: Grid, target: int, want_effective: bool, k: int) -> list[Event]:
    # No ineffective branch is deliberately proposed. The rule itself allows
    # no-ops (including an empty anchor); recomputation need not change values.
    if not want_effective:
        return []
    occupied = grid.nonzero_indices()
    if not occupied:
        return []
    limit = sum(size - 1 for size in grid.shape)
    scored: list[tuple[float, int]] = []
    events: list[Event] = []
    for _ in range(k * 4):
        start = rng.choice(occupied)
        layers, _, _ = nonzero_layers(grid, start, limit)
        # One breadth-first pass gives the support of the region truncated at
        # every depth; take the depth whose support is closest to the target.
        best_depth = 0
        best_distance = float("inf")
        size = 0
        for depth, layer in enumerate(layers):
            size += len(layer)
            distance = log_distance(size, target)
            if distance < best_distance:
                best_depth, best_distance = depth, distance
        scored.append((best_distance, len(events)))
        events.append(Event("MIX", {"p": list(grid.coord(start)), "r": best_depth}))
        if len(events) >= k * 2:
            break
    return [events[i] for i in closest(scored, k, rng)]


def _note(world: WorldSpec, before: tuple[int, ...], after: tuple[int, ...]) -> str:
    residues = _residues(world)

    def terms(index: int) -> str:
        parts = [before[index]] + [
            before[other] for other in (index - 1, index + 1) if 0 <= other < len(before)
        ]
        return "+".join(str(part) for part in parts)

    return (
        "From [2], cells [1] and [3] are reached in one step and [0] in two steps. Cell [4] is 0 "
        "and cannot be entered, so [5] and [6] are never reached although [5] is within "
        f"distance 3 of [2]. Cell [0] becomes 1 + (({terms(0)}) mod {residues}) = {after[0]} "
        f"from its own value and its neighbour [1]; cell [1] becomes 1 + (({terms(1)}) mod "
        f"{residues}) = {after[1]}; cell [2] becomes 1 + (({terms(2)}) mod {residues}) = "
        f"{after[2]}; cell [3] becomes 1 + (({terms(3)}) mod {residues}) = {after[3]}, reading "
        "the 0 at [4] although [4] is not reached. A reached cell is never set to 0."
    )


FAMILY = Family(
    metadata=FamilyMetadata(
        id="occluded_mix",
        name="Occluded Mix",
        status="implemented",
        dependency_structure=(
            "Axis neighbourhoods of every cell within r steps of p through nonzero cells"
        ),
        summary="Modular aggregation over the region reachable from p through nonzero cells.",
        event_ops=("SET", "MIX"),
        support_source="state",
        update_type="aggregate",
        order_dependent=False,
        predicate=None,
    ),
    rule=_rule,
    event_validator=_validate,
    step_fn=_step,
    proposer=_propose,
    world_validator=_validate_world,
    example=Example(
        before=(2, 1, 1, 3, 0, 1, 2),
        event=Event("MIX", {"p": [2], "r": 2}),
        note=_note,
    ),
)
