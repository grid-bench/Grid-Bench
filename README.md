# Grid-Bench

**Can LLMs Predict How Discrete Worlds Change?**

[Paper](https://arxiv.org/abs/2610.22312) · [Website](https://grid-bench.github.io/) · [Dataset](https://huggingface.co/datasets/xzx34/Grid-Bench)

Grid-Bench evaluates spatial state prediction under explicit rules. Each
instance supplies an initial grid and an ordered sequence of spatial operations
as text. The model returns the complete final grid, and an executable simulator
checks every predicted value.

Eight operation families cover geometry, neighborhoods, connected regions, and
movement in two-, three-, and four-dimensional grids. The accompanying study
contains **1,488 instances from 248 trajectories** and **8,928 evaluation
records across six models**.

## Quickstart

Requires Python 3.11 or newer. Install from a clone of this repository:

```sh
git clone https://github.com/grid-bench/Grid-Bench.git
cd Grid-Bench
python -m pip install -e .
python scripts/quickstart.py
```

The quickstart runs a small example and scores the reference answer and a
copy-initial baseline. It needs no credentials or model calls. The simulator,
generator, prompt renderer, and scorer use only the Python standard library.

```sh
grid-bench render-prompt examples/metric_paint.json
grid-bench simulate examples/metric_paint.json
```

The Python package is `grid_bench`; the command-line tool is `grid-bench`.

## Evaluate your model

Obtain the companion data directory using the [data guide](docs/data.md#get-the-data),
then send each row's `prompt` to your model. Save
one JSONL record per response:

```json
{"public_id":"<the instance's public_id>","response":"<the model's returned text>"}
```

Then score the answers directly:

```sh
grid-bench evaluate \
  --data /path/to/Grid-Bench-data \
  --predictions answers.jsonl \
  --out evaluation
```

The default `main` view includes 1,152 instances. Use `--view workload` for
within-family workload comparisons or `--view all` for all 1,488 instances.
These views overlap; use `main` for comparisons with the main study.
The evaluator writes summary metrics and per-instance scores, counting missing
answers in the selected denominator. See the [evaluation guide](docs/evaluation.md)
for the Python API, answer contract, and metrics.

## Generate and extend tasks

```sh
grid-bench generate --config examples/quickstart-config.json \
  --out /tmp/grid-bench-demo
```

| Operation | Transformation |
| --- | --- |
| PAINT | Replace a selected value inside a Manhattan ball |
| REFLECT | Reflect values within a rectangular region |
| STENCIL | Aggregate neighboring values in a metric region |
| MIX | Aggregate values in an occupied, connected neighborhood |
| RELABEL | Change the label of a connected component |
| LAYER | Grow or erode a connected region |
| MOVE | Move selected values along an axis around obstructions |
| COMPRESS | Pack and merge equal values along axis-aligned lines |

Each history uses one operation family, with optional SET assignments. The
model receives the complete history and predicts its final state in one
response. The [task definitions](docs/tasks.md) specify all transitions;
the [generation guide](docs/generation.md) explains configuration and extensions.

## Reproduce the study summaries

```sh
python -m pip install -e '.[plots]'
python scripts/reproduce_release.py \
  --data /path/to/Grid-Bench-data \
  --out /tmp/grid-bench-reproduced
```

This offline workflow verifies prompts and reference states, reconstructs six
result tables, and plots accuracy by history length. It uses released
measurements; collected model answers are not distributed. See the
[reproduction guide](docs/reproduction.md) for the checks and outputs.

## Data, citation, and license

The [dataset guide](docs/data.md) describes the benchmark and measurement
configurations, sample units, and provenance. Code is licensed under
[MIT](LICENSE); original benchmark data use [CC BY 4.0](LICENSE-DATA).
[CITATION.cff](CITATION.cff) contains the paper citation and software metadata.
Maintainer: Zixiang Xu ([xzx34](https://github.com/xzx34)).

Explore the [project website](https://grid-bench.github.io/) and download the
[Hugging Face dataset](https://huggingface.co/datasets/xzx34/Grid-Bench).
Read the [paper on arXiv](https://arxiv.org/abs/2610.22312). To cite the benchmark:

```bibtex
@misc{xu2026gridbench,
  title = {Grid-Bench: Can {LLMs} Predict How Discrete Worlds Change?},
  author = {Xu, Zixiang and Song, Zirui and Zheng, Wanrong and Zhang, Chiyu and Huang, Yue and Peng, Yuyang and Zhang, Kexin and Chen, Jinkun and Chen, Xiuying and Zhang, Xiangliang and Liu, Yan and Itti, Laurent and Shyu, Frank and Luo, Xiaoqiang and Pandey, Sandeep and Liu, Xi and Simon, Luke and Hu, Xiyang and Zhao, Yue},
  year = {2026},
  eprint = {2610.22312},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  doi = {10.48550/arXiv.2610.22312},
  url = {https://arxiv.org/abs/2610.22312}
}
```

## Development

```sh
python -m pip install -e '.[dev,plots]'
pytest
ruff check src tests scripts examples
pyright
```

The repository maintains the implementation directly in `src/grid_bench/`,
with self-contained tests in `tests/`. See [CONTRIBUTING.md](CONTRIBUTING.md)
for changes to rules, scoring, and data, and [CHANGELOG.md](CHANGELOG.md) for
release notes.
