# Evaluating model predictions

The same evaluator serves the command line and Python API. It takes benchmark
instances plus one delivered response per instance, and reports complete-state
accuracy together with cell-level diagnostics. No provider SDK or API key is
needed to score saved answers.

## Collect answers

```python
import json
from pathlib import Path

from grid_bench.dataset import load_benchmark

# Replace your_model with your own inference function.
benchmark = load_benchmark("/path/to/Grid-Bench-data", view="main")
with Path("answers.jsonl").open("w", encoding="utf-8") as output:
    for item in benchmark:
        response = your_model(item.prompt)  # Your model client returns text.
        output.write(json.dumps({
            "public_id": item.public_id,
            "response": response,
        }) + "\n")
```

Use one file per model and evaluation condition. Keep the full delivered answer
in `response`; do not pre-extract JSON or choose among answers using scores.
The evaluator applies the fixed extraction rule. Record your model version,
generation settings, and any tools or extra instructions alongside the run.

## Evaluate

```sh
grid-bench evaluate \
  --data /path/to/Grid-Bench-data \
  --predictions answers.jsonl \
  --view main \
  --out evaluation
```

`--data` accepts the companion package directory, a benchmark JSONL file, or
a dataset directory created by `grid-bench generate`. Available views are
`main` (default), `workload`, and `all`; see [dataset views](data.md).

The output directory contains `summary.json` with aggregate and grouped results,
and `scores.jsonl` with one record for every selected instance. Missing answers
receive zero exact match. Duplicate or unknown `public_id` values are errors.
Known instances outside the selected view are ignored and counted, so a full
answer file can be evaluated on each view without rewriting it.

In Python:

```python
from grid_bench.evaluation import evaluate_predictions, load_benchmark

benchmark = load_benchmark("/path/to/Grid-Bench-data", view="main")
report = evaluate_predictions(benchmark, "answers.jsonl")
print(report["summary"]["em"])
print(report["summary"]["valid_state_rate"])
```

`load_benchmark` also accepts an iterable of benchmark rows, and
`evaluate_predictions` accepts an iterable of prediction records. Set
`verify_references=True` when loading to re-execute the supplied reference states.

## Answer contract and extraction

A complete-state answer is a JSON object containing only `cells`:

```json
{"cells": [[[0, 0], 2], [[1, 1], 1]]}
```

Include each nonzero cell exactly once; omitted cells mean zero. Coordinates
must be in bounds and values must belong to the instance's alphabet. Duplicate
coordinates, explicit zero entries, and extra keys invalidate the state.
Lexicographic coordinate order is canonical, but a different order can still
be semantically correct. See [formats](formats.md) for complete contracts.

The extractor strips surrounding whitespace, accepts a single enclosing code
fence, and otherwise selects the last top-level object candidate. If that last
candidate is malformed or incomplete, extraction fails; it does not fall back
to an earlier object or repair the answer. Only delivered response text is
examined.

## Metrics

| Metric | Meaning |
| --- | --- |
| `em` | Fraction of all selected instances with an exact final state after extraction |
| `strict_em` | Exact-match fraction when scoring delivered text directly |
| `valid_state_rate` | Fraction returning a state that satisfies the answer schema |
| Wrong cells | Number of coordinates differing from the reference in a valid state |
| Cell accuracy | Fraction of grid cells equal to the reference in a valid state |

Invalid, empty, and missing answers have zero exact match. Validity does not
require correctness. `strict_em` checks the raw JSON contract, not byte-level
identity or coordinate order; canonical validity is reported separately.

The aggregate `cell_accuracy_valid_only` averages over `n_valid` returned
states and is null if none are valid. Per-instance invalid states retain a zero
cell-accuracy sentinel; they do not enter this conditional mean. Wrong-cell
summaries use valid incorrect states and report their own count. These summaries
also separate errors on cells that should change from those whose final
value should equal the initial value. These two subsets are defined by initial
and final reference states; a preserved cell may change during intermediate
events. Empty diagnostic subsets have null statistics.

For a single delivered response, use `score_answer(response, instance, gold)`
from `grid_bench.evaluation`; it applies the same extraction and diagnostics.
The low-level `score_prediction` function scores JSON directly against a
`GoldRecord`, without text extraction.
