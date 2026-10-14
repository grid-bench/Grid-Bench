# Grid-Bench

**Grid-Bench: Can LLMs Predict How Discrete Worlds Change?**

Grid-Bench evaluates how language models predict the consequences of spatial
operations. Each instance supplies an initial grid, explicit rules, and an
ordered event history as text. The model returns the complete final grid;
reference execution checks every predicted value.

Eight operation families cover geometry, neighborhoods, connected regions, and
movement in two-, three-, and four-dimensional grids. The accompanying study
evaluates six models on the same **1,488 instances from 248 trajectories**,
producing **8,928 outcomes**.

This is a **private prerelease repository** at
[grid-bench/Grid-Bench](https://github.com/grid-bench/Grid-Bench). The
[project website](https://grid-bench.github.io/) and
[Hugging Face dataset](https://huggingface.co/datasets/xzx34/Grid-Bench) are
confirmed destinations pending launch. Code is licensed under [MIT](LICENSE);
original benchmark data use [CC BY 4.0](LICENSE-DATA). The paper link will be
added after arXiv publication. See [release scope](LICENSE_STATUS.md).

Maintainer: Zixiang Xu ([xzx34](https://github.com/xzx34)). The code and dataset
are maintained as an academic project. The `preferred-citation` entry in
[CITATION.cff](CITATION.cff) records authorship of the accompanying paper.

## Try an example

Python 3.11 or newer is required. The simulator and scorer have no runtime
dependencies beyond the Python standard library.

```sh
python -m pip install -e .
python scripts/quickstart.py
```

The quickstart executes a small grid example, renders its prompt, and compares
the reference answer with copying the initial state. It makes no model calls.

To inspect the same example directly:

```sh
spatial-worlds render-prompt examples/metric_paint.json \
  --prompt-version spatial-worlds-prompt/0.4
spatial-worlds simulate examples/metric_paint.json
```

The package, import namespace, and CLI retain the names `spatial-worlds`,
`spatial_worlds`, and `spatial-worlds` for compatibility with the frozen study.
The benchmark's public name is Grid-Bench.

## Evaluate a model

Use the frozen benchmark package rather than regenerating test questions. Its
`benchmark` configuration contains one row per instance, including the exact
formal prompt and a separately stored reference answer. Send only `prompt` to
the model and retain the instance identity with its answer.

[Evaluation instructions](docs/evaluation.md) explain the JSON answer contract,
the official scorer, and native CLI inputs. [Dataset documentation](docs/data.md)
describes local loading, identities, and the main/workload views. Neither the
quickstart nor installation starts a paid evaluation.

## Reproduce the released results

Place the companion Hugging Face candidate on disk, then run:

```sh
python -m pip install -e '.[plots]'
python scripts/reproduce_release.py \
  --data /absolute/path/to/huggingface \
  --out /tmp/grid-bench-reproduced
```

This offline command replays the released reference instances and recreates
the supported result tables and history-length plot from recorded measurements.
Collected model answers are not distributed, so this command does not rescore
them. Use the scorer with your own predictions to compute new scores. It does not
query providers. [Reproduction instructions](docs/reproduction.md) distinguish
answer rescoring, summary reconstruction, and the archived analysis inputs.

## Task and study

| Operation | Spatial transformation |
| --- | --- |
| PAINT | Replace a selected value inside a Manhattan ball |
| REFLECT | Reflect a rectangular region along one axis |
| STENCIL | Aggregate neighboring values in a metric region |
| MIX | Aggregate values in an occupied, connected neighborhood |
| RELABEL | Change the label of a connected component |
| LAYER | Grow or erode a connected region |
| MOVE | Move a selected value along an axis around obstructions |
| COMPRESS | Pack and merge equal values along axis-aligned lines |

Every history uses one operation family, with possible SET assignments.
The released histories have 0, 1, 8, 16, 32, or 64 events. A model receives the
whole supplied history and predicts its final state in one response. See
[task and format guide](docs/task-and-format.md) for coordinates and semantics.

The study covers GPT-5.6 Sol, Claude 5 Sonnet, GPT-5.4 Mini, GLM-5, Kimi K2.5,
and MiniMax M2.5 in this order. Their shared conditions and model-specific
request settings are recorded in the outcome data. Complete-state accuracy,
cell errors, and changed/preserved-cell diagnostics describe different aspects
of the returned answers.

## Repository map

| Path | Contents |
| --- | --- |
| `src/spatial_worlds/` | Rules, generation, prompt rendering, scoring, and providers |
| `examples/` | Small examples and a three-prefix generation configuration |
| `scripts/quickstart.py` | One-command offline example |
| `scripts/reproduce_release.py` | Offline release verification and result reconstruction |
| `tests/` | Portable tests and marked tests requiring the original evidence archive |
| `configs/` | Generation and collection configuration sources |
| `docs/` | Task, data, evaluation, and reproduction instructions |
| `CITATION.cff` | Software metadata and the accompanying paper citation |

For development:

```sh
python -m pip install -e '.[dev,eval,plots]'
pytest -m 'not frozen_data'
ruff check src tests
pyright
```

See [contribution guidance](CONTRIBUTING.md) and [change history](CHANGELOG.md).
The website is being prepared separately; it is not required to use the code
or data.
