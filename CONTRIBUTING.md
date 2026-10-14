# Contributing

Grid-Bench is maintained directly in this repository. Start with the offline
example in the [README](README.md), then run the development checks:

```sh
python -m pip install -e '.[dev,plots]'
pytest
ruff check src tests scripts examples
pyright
```

Tests run without credentials or a private research archive. Keep code,
comments, and documentation in English. Report a bug with a minimal input,
expected behavior, and the package version; include the seed and configuration
for generation problems.

## Rules and scoring

Changes to transitions, parsing, or scoring should include a compact fixture
and an independent expected result. Cover boundaries, no-op cases, simultaneous
updates, and processing order when they apply. See the
[task definitions](docs/tasks.md), [formats](docs/formats.md), and
[generation guide](docs/generation.md).

Use `grid_bench` for Python imports and `grid-bench` for commands. New public
interfaces should use the same prompt and scoring defaults as the evaluator.
Keep experimental variants explicit so an API call and a CLI invocation do not
silently implement different evaluation conditions.

## Released data

Preserve published instances, prompts, reference states, and their identifiers.
A correction requires a new data revision and a changelog entry identifying
the affected records. Reference execution and scoring equivalence matter when
refactoring; internal file layouts and Python names may evolve.

The public data contain original synthetic instances and measurements, not
collected model answer text or predicted states. Use synthetic fixtures in
pull requests and tests. Do not include credentials or provider account details.

Code contributions use the [MIT license](LICENSE). Original data and examples
use [CC BY 4.0](LICENSE-DATA). Paper authorship is recorded separately in
[CITATION.cff](CITATION.cff).
