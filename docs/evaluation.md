# Evaluating predictions

Use the frozen instances and exact prompts from the companion data package.
Each instance is requested separately. Keep the public instance ID beside the
delivered answer so that model results can be joined without depending on file
order.

## Score one answer

This example reads a released row, extracts a complete answer from a local text
file, and uses the official full-world scorer. The reference is used only for
scoring, never as model input.

```python
import json
from pathlib import Path

from spatial_worlds.eval.extract import extract_final_answer
from spatial_worlds.schema import GoldRecord
from spatial_worlds.scoring import score_prediction

data_root = Path("/absolute/path/to/huggingface")
with (data_root / "data/benchmark.jsonl").open() as source:
    row = json.loads(next(source))

model_input = row["prompt"]
# Send model_input to your model and save its delivered answer as answer.txt.
answer = Path("answer.txt").read_text(encoding="utf-8")
gold = GoldRecord.from_dict(json.loads(row["reference_json"]))
extraction = extract_final_answer(answer)
score = score_prediction(extraction.text or "", gold)
print(row["public_id"], score.to_dict())
```

The fixed extractor can locate a complete JSON answer in surrounding text.
It does not fill missing coordinates, repair values, or execute events for the
model. Keep both the delivered text and the extracted answer when recording a
run. Scoring the delivered text directly gives the strict text-contract check;
scoring the extracted text gives semantic full-world exact match.

## Metrics

**Exact match** is one only when a valid complete state equals the reference at
every coordinate. Incorrect, invalid, and absent answers score zero, preserving
the full evaluated denominator.

For a valid state, **wrong cells** counts unequal coordinates and **cell
accuracy** is the fraction of all grid cells with the correct value. The
low-level scorer returns `cell_accuracy = 0.0` and `wrong_cells = null` for an
invalid state; analyses of returned states must use the stated valid-answer
subset rather than treating that sentinel as an observed spatial error count.

Changed/preserved-cell diagnostics compare the final reference with the initial
world. A preserved cell has the same initial and final value; it may have changed
during the intervening history. Empty conditional subsets have no estimated
accuracy and remain null in release summaries.

## Small end-to-end CLI example

The included configuration generates one trajectory at three prefixes. These
commands execute the reference provider and make no API calls:

```sh
spatial-worlds generate --config examples/quickstart-config.json \
  --out /tmp/grid-bench-quickstart
spatial-worlds run --data /tmp/grid-bench-quickstart \
  --provider oracle --out /tmp/grid-bench-oracle.jsonl
spatial-worlds evaluate --data /tmp/grid-bench-quickstart \
  --predictions /tmp/grid-bench-oracle.jsonl \
  --out /tmp/grid-bench-scores
```

This is a simulator/scorer demonstration on newly generated example instances.
For formal comparisons, use the frozen instances and their exact released
prompts, together with the formal extraction rule shown above.

For the frozen release, `scripts/reproduce_release.py` can export native CLI
inputs with `--export-cli /tmp/grid-bench-cli`. The generated guide identifies
the dataset-specific instance and gold paths; preserve those source groupings
when passing `--instances` and `--gold` to the CLI.

## Optional provider runs

Install `.[eval]` only when a provider SDK is needed. Inspect
`spatial-worlds run --help`, supply credentials through the provider's environment
variable, and choose a small explicit `--limit` and generation budget for a first
live run. A new run is a new evaluation condition: record the model identifier,
prompt version, generation parameters, and delivered outcomes.

The published study used prompt 0.4, one answer per instance, no tools, a
128,000-token combined reasoning-and-answer ceiling, and a 7,200-second deadline.
Temperature, `top_p`, and seed were omitted. Five model interfaces requested
high reasoning effort; MiniMax M2.5 used native always-on thinking. The released
scientific request fields preserve these native differences. Historical
collection configurations identify the original evidence; they are not a
ready-made configuration for every user's provider account.
