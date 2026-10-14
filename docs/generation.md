# Generating and extending tasks

Use the released benchmark for comparisons with the study. The generator lets
you create new datasets by choosing operation families, dimensions, history
lengths, workload targets, and seeds.

## Generate a small dataset

```sh
grid-bench generate --config examples/quickstart-config.json \
  --out /tmp/grid-bench-demo
```

The included configuration creates one PAINT trajectory with three prefixes:

```json
{
  "name": "quickstart",
  "dataset_seed": 7,
  "split": "demo",
  "tier": "S",
  "dims": [2],
  "t_grid": [0, 1, 2],
  "t_max": 2,
  "seeds_per_cell": 1,
  "families": {"metric_paint": {"w_targets": [7]}}
}
```

Use a new or empty output directory; generation rejects nonempty directories
so separate runs cannot mix instances or overwrite their manifest.
`t_max` is the underlying trajectory length;
`t_grid` selects its prefixes. `seeds_per_cell` is the number of trajectories
per family, shape, and workload condition. `tier` selects a shape scale and
`dims` its dimensions. Family defaults are expanded into the output manifest.

The generated directory contains:

```text
manifest.json
<split>/<family>/instances.jsonl
<split>/<family>/gold.jsonl
<split>/<family>/trace.jsonl
```

The manifest records the resolved configuration, seeds, software provenance,
and generation reports. Traces contain event-level workload and density
measurements; `--full-trace` also records intermediate states. Unreachable or
rejected conditions are reported explicitly.

Generated directories use the same loading and evaluation interface. For an
offline scorer check, create answers from the simulator references:

```python
import json
from pathlib import Path

from grid_bench.dataset import load_benchmark

benchmark = load_benchmark("/tmp/grid-bench-demo", view="all")
answers = [
    {"public_id": item.public_id, "response": json.dumps(item.gold.final_state.to_dict())}
    for item in benchmark
]
Path("/tmp/grid-bench-oracle.jsonl").write_text(
    "".join(json.dumps(answer) + "\n" for answer in answers), encoding="utf-8"
)
```

```sh
grid-bench evaluate --data /tmp/grid-bench-demo \
  --predictions /tmp/grid-bench-oracle.jsonl \
  --view all --out /tmp/grid-bench-scores
```

These three reference answers should achieve exact match 1.0. For an actual
model evaluation, send `item.prompt` to the model and save its delivered text
in `response`; see the [evaluation guide](evaluation.md).

## Choose generation conditions

A family's overrides can specify `values`, `density`, `set_rate`, `noop_rate`,
`initializer`, `set_policy`, `w_targets`, `candidates`, and `tolerance`.
The full typed configuration and defaults are in
[generation/config.py](../src/grid_bench/generation/config.py).

Workload targets have three meanings:

| Families | Target |
| --- | --- |
| PAINT, REFLECT, RELABEL, LAYER, MOVE | Canonical per-event `scan` |
| STENCIL, MIX | Number of cells in the update support |
| COMPRESS | Grid size |

Realized workloads need not equal targets. Geometry, connectivity, and
occupancy constrain the available events. `tolerance` defines a relative band
within which proposed events tie in selection; whole-trajectory acceptance
is controlled separately by the `rejection` configuration.

Initializers generate uniform occupancy, connected regions, or small tiles.
SET policies can regulate occupancy, encourage distinct neighboring labels,
or add tiles. The rules accept some no-op events; the sampler's `noop_rate`
controls proposed ineffective events only for families supporting that policy.
Inspect realized values in the traces rather than interpreting requested
parameters as exact observations.

The study's generation settings are retained in [configs/main.json](../configs/main.json)
and [configs/rq2.json](../configs/rq2.json). For new questions, copy and edit a
configuration under a new dataset name. For exact study comparisons, load the
released instances and prompts directly.

## Add an operation family

A family combines metadata, world/event validation, a rule card, a transition
function, and optionally a worked example and an event proposer. See
[family.py](../src/grid_bench/family.py) and the existing
[implementations](../src/grid_bench/families).

1. Define the exact state transition, including boundaries, allowed values,
   no-op cases, simultaneous reads, and any processing order.
2. Implement a `Family` with validators and a `step_fn` that mutates a `Grid`
   and returns `StepResult`. `effect` is calculated centrally from the states.
3. Register it in `families/__init__.py`; add generation defaults and a
   proposer if it should support automatic dataset generation.
4. Add independent small examples and tests for transition semantics, prompt
   rendering, reference execution, and scoring.

A custom `Family` can also be passed directly to `simulate(instance, family=...)`
when testing an extension before registry integration. Keep its event names
and serialized identifiers distinct from existing families.
