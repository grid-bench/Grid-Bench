# Released data

The companion Hugging Face candidate has two configurations, each with one
`test` split. It contains no training or validation split.

| Configuration | Rows | Meaning |
| --- | ---: | --- |
| `benchmark` | 1,488 | Unique grid prediction instances |
| `results` | 8,928 | Six recorded evaluation results per benchmark instance |

Each configuration has JSONL and Parquet representations as
`data/benchmark.{jsonl,parquet}` and `data/results.{jsonl,parquet}`. JSON-string fields retain heterogeneous
native objects without flattening event or state structure.

## Load a local candidate

Install Hugging Face Datasets separately if desired. With a local candidate:

```python
from datasets import load_dataset

benchmark = load_dataset(
    "/absolute/path/to/huggingface", "benchmark", split="test"
)
results = load_dataset(
    "/absolute/path/to/huggingface", "results", split="test"
)
print(benchmark[0]["prompt"])
```

Standard-library users can stream the JSONL files directly. Public download
instructions and immutable remote revisions will be added when the destinations
are confirmed.

## Identities and views

`public_id` combines `source_dataset` and `instance_id`; use it to join results
to benchmark rows. `public_trajectory_id` combines the source dataset and native
`rollout_id`. Native IDs and source metadata remain available independently.

The main view has 1,152 instances from 192 trajectories. Seven additional
workload conditions contribute 336 instances from 56 trajectories. Together
they make 1,488 instances from 248 trajectories. The 528-instance workload view
includes those 336 plus 192 reused main instances. Filter `in_main` or
`in_workload`; do not concatenate the two views as disjoint test sets.

Each trajectory contributes six history lengths: 0, 1, 8, 16, 32, and 64. The
original and expansion cohorts use different trajectory indices and are pooled
in the current release.

## Model inputs and references

- `instance_json` is the native input object.
- `reference_json` is the native gold record, including the final state.
- `prompt` is the exact text supplied in formal evaluation.
- `prompt_version` describes that formal text: `spatial-worlds-prompt/0.4`.
- `source_prompt_version` preserves the original instance metadata.
- Hash fields identify the corresponding source instance, gold, and prompt.

Only `prompt` is model input in the formal interface. Reference answers,
result records, and diagnostic metadata are evaluation-side information.

The results configuration preserves measurements for all six models, including
requests with absent and invalid answers: validity, scores, termination, and
available usage fields. It excludes model text, predicted grids, answer hashes,
and free-text parsing errors. See the dataset card and machine-readable schema
for the full fields. The data are licensed under CC BY 4.0.
