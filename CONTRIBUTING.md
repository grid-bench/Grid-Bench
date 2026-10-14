# Contributing

Start with the small offline example in the [README](README.md). Changes to
reference transitions, parsing, or scoring should include a compact fixture
showing the behavior and a test that checks it independently of the changed
implementation.

## Development checks

```sh
python -m pip install -e '.[dev,eval,plots]'
pytest -m 'not frozen_data'
ruff check src tests
pyright
```

Keep code, comments, documentation, and reports in English. Describe a proposed
change with the affected operation or interface, expected behavior, and a minimal
reproducer. Include a random seed and input instance when reporting generation
problems. Omit API credentials and provider account details from reports.

## Compatibility and evidence

The public benchmark is Grid-Bench; the Python package and protocol identifiers
retain their existing names. Avoid renaming serialized fields or replacing
frozen data in place. A correction to released instances, references, prompts,
or scores requires a new data revision and a changelog entry explaining the
affected records.

Keep formatting-only and semantic changes separate when practical. New provider
support should preserve the requested settings, delivered answer, and outcome
identity. It must not retry a completed answer based on its score.

Public issue and pull-request destinations will be added when the repository is
published. The current repository is private; see
[licensing and release scope](LICENSE_STATUS.md) for the MIT code and CC BY 4.0 data.
