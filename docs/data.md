# Benchmark and measurement data

The companion data package has two configurations, each with one `test` split:

| Configuration | Rows | Contents |
| --- | ---: | --- |
| `benchmark` | 1,488 | Instances, exact prompts, and simulator references |
| `results` | 8,928 | Recorded evaluation measurements for six models |

Both are supplied as JSONL and Parquet under `data/`. There are no training or
validation splits. Download the dataset from
[xzx34/Grid-Bench](https://huggingface.co/datasets/xzx34/Grid-Bench).

## Get the data

Download the complete package with Hugging Face Hub:

```sh
python -m pip install huggingface_hub
```

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="xzx34/Grid-Bench",
    repo_type="dataset",
    local_dir="Grid-Bench-data",
)
```

Pass `Grid-Bench-data` to the evaluator or reproduction script. The complete
download includes the manifest and evidence files required for reproduction.
It contains simulator references and recorded measurements, not model answers.

## Load benchmark instances

```python
from grid_bench.dataset import load_benchmark

benchmark = load_benchmark("/path/to/Grid-Bench-data", view="main")
for item in benchmark:
    print(item.public_id, item.prompt)
```

The loader also accepts `data/benchmark.jsonl` directly or an iterable of rows,
including a Hugging Face Dataset. To load both configurations with the optional
`datasets` library, first install it with `python -m pip install datasets`:

```python
from datasets import load_dataset

benchmark = load_dataset("xzx34/Grid-Bench", "benchmark", split="test")
results = load_dataset("xzx34/Grid-Bench", "results", split="test")
```

## Views and sample units

| View | Instances | Trajectories |
| --- | ---: | ---: |
| `main` | 1,152 | 192 |
| `workload` | 528 | 88 |
| `all` | 1,488 | 248 |

The workload view combines 336 additional instances with 192 reused main
instances. The views overlap: do not add their row counts as if they were
separate test sets. Row flags `in_main` and `in_workload` define membership.

Each trajectory supplies six prefixes, with 0, 1, 8, 16, 32, and 64 events.
Prefixes share an underlying history. Statistical analyses should preserve
that grouping; the study's confidence intervals resample trajectories.

`public_id` uniquely identifies an instance across all source groups.
`public_trajectory_id` identifies its underlying trajectory. Join predictions
and measurements by these identifiers, not by row order.

## Inputs, references, and measurements

- `prompt` is the exact model input used for the study.
- `instance_json` stores the initial world, events, and metadata.
- `reference_json` stores the simulator's final state and event workloads.
- Hash fields identify the source instance, reference, and prompt.

Send only `prompt` to a model. References and measurement records are used on
the evaluation side. JSON-string fields preserve nested event/state structures
without flattening them into columns.

The `results` configuration records correctness, validity, cell-error counts,
termination, request settings, and available token usage for all six models.
It contains no model answer text, predicted grids, answer hashes, or free-text
parsing errors. Its machine-readable schema documents every field.

## Provenance and licensing

The public software uses the Grid-Bench name. Frozen records retain the
serialized identifiers `spatial-worlds/0.2` (schema) and
`spatial-worlds-prompt/0.4` (formal prompt); `source_prompt_version` records the
original source metadata independently. These identifiers support byte-level
provenance and do not select a different public package. Default prompt
rendering matches the formal prompt.

The manifests distinguish original source identities from exported file
hashes. The [reproduction workflow](reproduction.md) verifies the package
inventory and replays all reference instances. Original benchmark data and the
released measurement compilation are licensed under CC BY 4.0; see the
package's license and dataset card.
