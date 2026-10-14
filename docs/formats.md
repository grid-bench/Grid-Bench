# Data formats and Python API

## Instance

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

Required keys are `schema_version`, `id`, `family`, `world`, `initial`, and
`events`. Optional `metadata` contains generation information. The schema
identifier above is retained in serialized data; see [provenance](data.md).

`shape` lists positive side lengths. Coordinates are zero-based integer arrays
with one component per axis. `world.values` is a strictly increasing integer
alphabet including zero, subject to the family's alphabet constraints.
A state lists nonzero cells as `[coordinate, value]`; omitted cells are zero.
Coordinates must be unique and in bounds. Events execute in array order.

Every event is an object with an uppercase `op` and exactly the parameters
specified in the [task definitions](tasks.md). For example,
`{"op":"SET","p":[0,1],"v":2}` assigns value 2 to `[0,1]`.

## Complete-state answer

The example above produces:

```json
{"cells": [[[0, 0], 2], [[1, 1], 1]]}
```

The only permitted key is `cells`. Include every nonzero cell exactly once.
Explicit zeros, duplicate coordinates, out-of-bounds coordinates, values outside
the alphabet, and unknown keys are invalid. Canonical serialization orders
coordinates lexicographically. Reordering cells does not change semantic
exact match; canonical validity is a separate property.

## Reference record

```json
{
  "schema_version": "spatial-worlds/0.2",
  "instance_id": "paint-demo",
  "family": "metric_paint",
  "world": {"shape": [2, 2], "values": [0, 1, 2]},
  "final_state": {"cells": [[[0, 0], 2], [[1, 1], 1]]},
  "workloads": [{"scan": 3, "effect": 1}]
}
```

`workloads` has one entry per event. `scan` and `effect` follow the accounting
in the task definitions. The reference contains the answer and must not be
included in the model input.

## Execute and inspect an instance

```python
import json
from pathlib import Path

from grid_bench import Instance, render_prompt, score_prediction, simulate

instance = Instance.from_dict(json.loads(Path("examples/metric_paint.json").read_text()))
prompt = render_prompt(instance)
run = simulate(instance, keep_all_states=True)
gold = run.gold()
print(run.state_at(0).to_dict())
print(run.final_state.to_dict())
print(score_prediction(json.dumps(gold.final_state.to_dict()), gold).to_dict())
```

`simulate` validates event parameters and state-dependent results as it executes
all events. `run.steps` contains per-event measurements; `keep_all_states=True`
retains every intermediate state. `states_at=[...]` retains selected prefixes.
`render_prompt` uses the same default as `grid-bench render-prompt` and the
evaluator. It supplies the explicit rule and a worked example when the instance
alphabet supports that example.

The direct scorer consumes JSON text. To evaluate delivered model text with
answer extraction and grouped statistics, use the [evaluation API](evaluation.md).

## Optional variants

The study uses full-state answers and the default cell-list representation.
The renderer also supports `representation="grid"` in one or two dimensions,
and `representation="cells_shuffled"` for a deterministic shuffle of the initial
cell list. These alter the input representation, not the transition rules.

An instance can optionally include `queries`, a list of distinct in-bounds
coordinates. Its answer then uses `{"values":[[coordinate,value],...]}`,
including every queried coordinate once and allowing zero values. Canonical
order follows the supplied query list. `score_answer` selects the query scorer for this variant, while
`score_query_prediction` provides the direct JSON check. Query evaluation is
separate from full-state benchmark comparisons.
