# Reproducing study summaries

The offline reproduction helper verifies the companion data package, replays
its reference states, and reconstructs study summaries from recorded
measurements. It needs no model credentials or private research files.

```sh
python -m pip install -e '.[plots]'
python scripts/reproduce_release.py \
  --data /path/to/Grid-Bench-data \
  --out /tmp/grid-bench-reproduced
```

Add `--no-plot` to run without Matplotlib. The base package is sufficient for
reference verification and table reconstruction.

## Checks and outputs

The helper validates file inventories and hashes, checks exact prompt rendering,
executes all 1,488 reference instances, and joins all 8,928 measurement records
by their identities. It writes:

| Output | Contents |
| --- | --- |
| `validation.json` | Checks completed and any discrepancies |
| `tables/overview.csv` | Overall measurements by model |
| `tables/length.csv` | Results by history length |
| `tables/task_length.csv` | Results by operation family and history length |
| `tables/dimension_length.csv` | Results by dimension and history length |
| `tables/workload.csv` | Within-family workload comparisons |
| `tables/resources.csv` | Recorded resource and termination summaries |
| `figures/history-length.png` and `.pdf` | Main-view accuracy curves |

The six tables contain 846 rows in total. Descriptive statistics are recomputed
from individual measurement records. Confidence intervals are retained from
the released trajectory bootstrap; this command does not rerun that bootstrap.
The report records which checks and calculations were performed.

Use a data directory downloaded with `snapshot_download(..., local_dir=...)`
as shown in the [dataset guide](data.md). Local Git metadata and Hugging Face
download metadata under `.cache/huggingface/` are ignored; released files still
undergo checksum verification. Save generated outputs outside the data directory.

## What is reproducible from each input

| Input | Check or computation |
| --- | --- |
| Instances and simulator | Complete reference states and per-event workloads |
| Instances and renderer | Exact model-visible prompts |
| Your own delivered answers | New per-instance scores through the [evaluator](evaluation.md) |
| Released measurement records | Descriptive statistics, supported tables, and accuracy curves |

Collected model answers are not distributed, so the helper cannot independently
rescore those answers. The retained measurements support summary reconstruction;
new answers can be evaluated using the same scoring implementation.

Each trajectory contributes six related prefix instances. The study's intervals
use 2,000 trajectory-bootstrap replicates within the reported strata. The
[dataset guide](data.md) documents the main and workload views, which share
some instances.
