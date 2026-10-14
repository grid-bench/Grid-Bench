"""Compact, self-contained prompt and answer rendering."""

from __future__ import annotations

import json
from random import Random

from .registry import get_family, validate_instance
from .schema import Instance, SchemaError
from .world import WorldState

LEGACY_PROMPT_VERSION = "spatial-worlds-prompt/0.3"
PROMPT_VERSION = "spatial-worlds-prompt/0.4"
COMPLETION_PREFIX = (
    "This is a deterministic simulation of an abstract integer-valued grid.\n"
    "Keep deliberation concise and reserve enough of the response budget to return the "
    "complete final-state JSON.\n\n"
)
# cells: canonical lexicographic listing (the default and the answer format);
# grid: digit grid for d <= 2; cells_shuffled: the same listing in a seeded
# random order, an input-order ablation that leaves the output contract alone.
REPRESENTATIONS = ("cells", "grid", "cells_shuffled")

_NO_FENCE = "Output the JSON object only, with no code fence or other text."


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def render_grid(state: WorldState) -> str:
    """Fixed-width digit grid for a one- or two-dimensional state."""

    world = state.world
    if world.dimension > 2:
        raise ValueError("grid representation requires dimension <= 2")
    width = max(len(str(value)) for value in world.values)
    values = state.dense_values()
    if world.dimension == 1:
        rows = [values]
    else:
        columns = world.shape[1]
        rows = [values[i * columns : (i + 1) * columns] for i in range(world.shape[0])]
    return "\n".join(" ".join(str(v).rjust(width) for v in row) for row in rows)


def shuffled_cells(instance: Instance) -> list[list[object]]:
    """The initial cells in a random order derived from the instance id."""

    cells = [[list(position), value] for position, value in instance.initial.cells]
    Random(f"cells_shuffled:{instance.id}").shuffle(cells)
    return cells


def _shuffled_block(instance: Instance) -> str:
    note = "The cells below are listed in an arbitrary order."
    return note + "\n" + _compact_json({"cells": shuffled_cells(instance)})


def _grid_block(instance: Instance) -> str:
    if instance.dimension == 1:
        note = "Grid form of the initial world. The value at column j is the value of cell [j]."
    else:
        note = (
            "Grid form of the initial world. Row index is axis 0, column index is axis 1; "
            "the value at row i, column j is the value of cell [i,j]."
        )
    return note + "\n" + render_grid(instance.initial)


def render_prompt(
    instance: Instance,
    *,
    representation: str = "cells",
    prompt_version: str = PROMPT_VERSION,
) -> str:
    """Render the versioned prompt template for one benchmark instance."""

    if prompt_version not in {LEGACY_PROMPT_VERSION, PROMPT_VERSION}:
        raise ValueError(f"unknown prompt version {prompt_version!r}")
    if representation not in REPRESENTATIONS:
        raise ValueError(f"unknown representation {representation!r}")
    validate_instance(instance)
    family = get_family(instance.family)
    shape = _compact_json(list(instance.world.shape))
    values = _compact_json(list(instance.world.values))
    events = _compact_json([event.to_dict() for event in instance.events])
    if representation == "grid":
        if instance.dimension > 2:
            raise ValueError("grid representation requires dimension <= 2")
        initial = _grid_block(instance)
    elif representation == "cells_shuffled":
        initial = _shuffled_block(instance)
    else:
        initial = _compact_json(instance.initial.to_dict())

    if instance.queries:
        task = "Apply all events in order and return the final value of each query cell."
        output = (
            "OUTPUT",
            'Return only {"values":[[q,v],...]} with one entry per query coordinate below, '
            "in the listed order, where v is the final value of cell q (0 if empty). " + _NO_FENCE,
            "",
            "QUERIES",
            _compact_json([list(query) for query in instance.queries]),
        )
    else:
        task = "Apply all events in order and return the final world."
        output = (
            "OUTPUT",
            'Return only {"cells":[[p,v],...]}, where each entry is [coordinate,value]. '
            "Include every nonzero cell exactly once in lexicographic coordinate order. "
            + _NO_FENCE,
        )

    example = family.example_card(instance.world)
    example_section = ("EXAMPLE", example, "") if example is not None else ()
    rendered = "\n".join(
        (
            "TASK",
            task,
            "",
            "SPACE",
            f"shape = {shape}",
            "Coordinates and axis indices are zero-based. The world is bounded and does not wrap.",
            f"Unlisted cells have value 0. Allowed values: {values}.",
            "W(x) denotes the value of cell x immediately before the current event.",
            "",
            "COMMON EVENT",
            "SET(p,v): set cell p to v.",
            "",
            "FAMILY RULE",
            family.rule_card(instance.world),
            "",
            *example_section,
            "INITIAL",
            initial,
            "",
            "EVENTS",
            events,
            "",
            *output,
        )
    )
    return (
        (COMPLETION_PREFIX + rendered) if prompt_version == PROMPT_VERSION else rendered
    )


def render_answer(state: WorldState) -> str:
    """Serialize a state in the exact model-answer format."""

    if not isinstance(state, WorldState):
        raise TypeError("render_answer expects a WorldState")
    return _compact_json(state.to_dict())


def render_query_answer(state: WorldState, queries: tuple[tuple[int, ...], ...]) -> str:
    """Serialize the query-contract answer for ``queries`` against ``state``."""

    if not isinstance(state, WorldState):
        raise TypeError("render_query_answer expects a WorldState")
    if not queries:
        raise SchemaError("render_query_answer requires at least one query")
    return _compact_json({"values": [[list(q), state.get(q)] for q in queries]})
