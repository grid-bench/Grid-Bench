# Task and format

An instance gives an initial grid and a complete ordered event sequence. The
reference simulator applies each event to the state left by its predecessor.
The answer is the complete state after the last supplied event. No additional
event needs to be predicted.

## Coordinates and values

A shape such as `[3, 3]` defines a bounded grid. Coordinates and axis indices
start at zero; coordinates never wrap. Each coordinate holds an integer from
the instance's declared `world.values` alphabet. Unlisted cells have value zero.
The native schema is `spatial-worlds/0.2`.

```json
{
  "schema_version": "spatial-worlds/0.2",
  "id": "paint-demo",
  "family": "metric_paint",
  "world": {"shape": [2, 2], "values": [0, 1, 2]},
  "initial": {"cells": [[[0, 0], 1], [[1, 1], 1]]},
  "events": [{"op": "PAINT", "a": [0, 0], "r": 1, "u": 1, "v": 2}]
}
```

Here PAINT replaces value 1 with value 2 within Manhattan distance 1 of `[0,0]`.
The cell `[1,1]` lies outside this region. The final answer is:

```json
{"cells": [[[0, 0], 2], [[1, 1], 1]]}
```

An answer contains exactly the key `cells`, with every nonzero cell represented
once as `[coordinate, value]`. The canonical order is lexicographic by
coordinate. A reordered complete answer can still be semantically exact;
canonical formatting is reported separately. Duplicate coordinates, explicit
zero entries, out-of-bounds coordinates, and values outside the alphabet are
invalid.

## Operations

The prompt supplies the precise rule for its family and a worked example.
Every family also accepts `SET(p,v)`, which assigns value `v` to coordinate `p`.
The following descriptions orient the reader; use the supplied prompt and
reference implementation when executing an event.

| Operation | What changes |
| --- | --- |
| `PAINT(a,r,u,v)` | Cells equal to `u` within a Manhattan ball become `v`. |
| `REFLECT(lo,hi,axis)` | Values, including zeros, move to reflected positions within the inclusive box. Writes are simultaneous. |
| `STENCIL(a,r)` | Cells in a Manhattan ball become the sum of their own and axis-neighbor values modulo the alphabet size. All reads use the old world. |
| `MIX(p,r)` | A bounded-depth region reachable through nonzero cells is updated from its old values and neighbors using the rule's nonzero modular sum. |
| `RELABEL(p,v)` | The same-valued connected component containing a nonzero anchor receives the new label. |
| `LAYER(p,s)` | A nonzero anchor's component grows into adjacent zeros or loses cells adjacent to in-bounds zeros. |
| `MOVE(v,axis,direction,k)` | Selected values move at most `k` cells, stopping at a boundary or occupied cell. Leading entities move first. |
| `COMPRESS(axis,direction)` | Each axis-aligned line packs toward one edge; adjacent equal values merge once per event. |

For state-dependent operations, a valid event can leave the state unchanged.
Space outside the grid is not an extra layer of zero-valued cells. The event
rules determine which reads and writes happen simultaneously and which use an
explicit processing order.

## Reference records and prompts

`spatial-worlds simulate` produces a native gold record containing the instance
identity, world, `final_state`, and per-event reference workloads. The
`reference_json` field in the data package holds this complete record. The
model-visible `instance_json` contains the initial world and events, not the
reference answer.

The formal prompt version is `spatial-worlds-prompt/0.4`. Native source metadata
retains its historical prompt identity, which may be `0.3`. These fields serve
different purposes: `prompt_version` describes the exported formal prompt;
`source_prompt_version` preserves the original instance metadata. Use the
exported prompt directly or request version 0.4 explicitly when rendering it.

History length counts all supplied events, including SET. Reference scan and
effect counts describe execution work and changed cells under the corresponding
rule. They are not units of model computation.

Complete rule definitions and edge cases are in [TASKS.md](TASKS.md). The native
instance, trace, request, result, and scoring contracts are in
[FORMAT_SPEC.md](FORMAT_SPEC.md).
