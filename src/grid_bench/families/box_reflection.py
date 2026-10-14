"""Box Reflection: mirror the contents of an axis-aligned box along one axis."""

from __future__ import annotations

import math
from random import Random

from ..family import (
    Example,
    Family,
    FamilyMetadata,
    StepResult,
    axis_param,
    require_params,
    validate_discrete_alphabet,
)
from ..grid import Grid
from ..schema import Event, SchemaError, WorldSpec
from ._common import log_distance


def _validate(world: WorldSpec, event: Event) -> None:
    require_params(event, "lo", "hi", "axis")
    lo = world.validate_position(event.params["lo"], label="REFLECT.lo")
    hi = world.validate_position(event.params["hi"], label="REFLECT.hi")
    if any(left > right for left, right in zip(lo, hi, strict=True)):
        raise SchemaError("REFLECT requires lo_i <= hi_i on every axis")
    axis_param(world, event)


def _step(grid: Grid, event: Event) -> StepResult:
    params = event.params
    lo = params["lo"]
    hi = params["hi"]
    axis = params["axis"]
    indices = grid.box(lo, hi)
    data = grid.data
    old = [data[index] for index in indices]
    stride = grid.strides[axis]
    total = lo[axis] + hi[axis]
    coords = grid.coords
    nonzero = 0
    for index, value in zip(indices, old, strict=True):
        mirrored = index + (total - 2 * coords[index][axis]) * stride
        data[mirrored] = value
        if value:
            nonzero += 1
    return StepResult(scan=len(indices), extras={"nonzero_in_box": nonzero})


_SIDE_CACHE: dict[tuple[tuple[int, ...], int], tuple[tuple[int, ...], ...]] = {}


def _side_options(shape: tuple[int, ...], target: int) -> tuple[tuple[int, ...], ...]:
    """Box side tuples whose volume is within tolerance of the target scan."""

    key = (shape, target)
    cached = _SIDE_CACHE.get(key)
    if cached is not None:
        return cached
    result: tuple[tuple[int, ...], ...] = ()
    for tolerance in (0.25, 0.5, 1.0, 4.0):
        options = _enumerate_sides(shape, target, tolerance)
        if options:
            result = tuple(options)
            break
    if not result:
        result = (tuple(min(2, size) for size in shape),)
    _SIDE_CACHE[key] = result
    return result


def _enumerate_sides(
    shape: tuple[int, ...], target: int, tolerance: float
) -> list[tuple[int, ...]]:
    limit = math.log(1 + tolerance)
    ceiling = target * (1 + tolerance) * 1.0001
    options: list[tuple[int, ...]] = []

    def recurse(axis: int, sides: list[int], volume: int) -> None:
        if axis == len(shape):
            if log_distance(volume, target) <= limit and max(sides) >= 2:
                options.append(tuple(sides))
            return
        for side in range(1, shape[axis] + 1):
            if volume * side > ceiling:
                break
            sides.append(side)
            recurse(axis + 1, sides, volume * side)
            sides.pop()

    recurse(0, [], 1)
    return options


def _propose(rng: Random, grid: Grid, target: int, want_effective: bool, k: int) -> list[Event]:
    options = _side_options(grid.shape, target)
    shape = grid.shape
    data = grid.data
    events: list[Event] = []
    for _ in range(k):
        sides = rng.choice(options)
        axes = [axis for axis, side in enumerate(sides) if side >= 2]
        axis = rng.choice(axes)
        chosen: tuple[list[int], list[int]] | None = None
        for _attempt in range(12 if not want_effective else 1):
            lo = [rng.randrange(size - side + 1) for size, side in zip(shape, sides, strict=True)]
            hi = [start + side - 1 for start, side in zip(lo, sides, strict=True)]
            chosen = (lo, hi)
            if want_effective:
                break
            if all(data[index] == 0 for index in grid.box(lo, hi)):
                break
        assert chosen is not None
        events.append(Event("REFLECT", {"lo": chosen[0], "hi": chosen[1], "axis": axis}))
    return events


FAMILY = Family(
    metadata=FamilyMetadata(
        id="box_reflection",
        name="Box Reflection",
        status="implemented",
        dependency_structure="Bijective coordinate transformation",
        summary="Reflect the contents of an axis-aligned box along one coordinate axis.",
        event_ops=("SET", "REFLECT"),
        support_source="geometry",
        update_type="transport",
        order_dependent=False,
        predicate=None,
    ),
    rule=(
        "REFLECT(lo,hi,axis): for every cell x in the inclusive box lo <= x <= hi, the new "
        "value at the mirrored position x' (x'_axis = lo_axis + hi_axis - x_axis, other "
        "coordinates unchanged) is the old value at x, including 0. All writes happen "
        "simultaneously. Cells outside the box remain unchanged."
    ),
    event_validator=_validate,
    step_fn=_step,
    proposer=_propose,
    world_validator=validate_discrete_alphabet,
    example=Example(
        before=(1, 2, 3, 0, 0, 1),
        event=Event("REFLECT", {"lo": [1], "hi": [4], "axis": 0}),
        note=(
            "The box is cells [1] to [4]; their old values 2 3 0 0 are written back in reverse "
            "order as 0 0 3 2, so the zeros move as well. Cells [0] and [5] are outside the box "
            "and keep their values."
        ),
    ),
)
