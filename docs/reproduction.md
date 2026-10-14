# Reproducing the release

The code candidate and data candidate form one release. Keep their manifests
together so a result can be traced to the same instances, prompts, outcomes, and
analysis snapshot.

## Offline verification and result reconstruction

From the code directory, with the companion Hugging Face directory available:

```sh
python -m pip install -e '.[plots]'
python scripts/reproduce_release.py \
  --data /absolute/path/to/huggingface \
  --out /tmp/grid-bench-reproduced
```

The helper reads the frozen inputs and delivered answers, checks reference
execution and scoring, and writes supported summaries and a history-length
figure to the output directory. It does not generate new model results.
Its output report records the exact checks completed and files produced.

Outputs include `validation.json`, six CSV tables under `tables/` (`overview`,
`length`, `task_length`, `dimension_length`, `workload`, and `resources`), and
`figures/history-length.png` plus its PDF version. Descriptive statistics are
recomputed; trajectory-bootstrap intervals are read from the frozen analysis.

The released study has 1,488 reference states and 8,928 model-instance outcomes.
The main history-length comparison uses 1,152 instances per model. The separate
workload view reuses 192 of these and includes 336 additional instances; it
contains 528 unique instances, not an extra 528 questions in the benchmark.

## Three reproduction levels

| Operation | Inputs | Result |
| --- | --- | --- |
| Execute references and score a new model | Instances, simulator, scorer, and new answers | Independently computed target states and scores |
| Re-extract and rescore the collected answers | References, delivered text, and frozen scoring/extraction settings | A check of all released outcome scores |
| Rebuild the selected result summaries and plots | Outcome scores and frozen analysis/diagnostic evidence | The supported aggregate tables and assets |

Archived analysis and diagnostic snapshots preserve the complete reported
statistics. Reusing those files does not rerun their original bootstrap or
reconstruct provider requests. The portable helper states which values it
recomputes and which intervals or diagnostics it reads from frozen evidence.
The original campaign-ledger verification tools require the original collection
archive and are separate from this portable workflow.

## Sample units and versions

Each trajectory contributes six prefix instances at T=0,1,8,16,32,64. Prefixes
share an underlying history. The released study's intervals resample complete
trajectories within the reported strata, with 2,000 replicates. Rows and prefixes
should not be substituted for independent trajectories in those intervals.

The public release identity is separate from package version 0.5.0, native
schema `spatial-worlds/0.2`, and formal prompt `spatial-worlds-prompt/0.4`.
Historical metadata retains its original values. Manifests distinguish source
hashes from the bytes of sanitized public exports.

## Native CLI inputs

```sh
python scripts/reproduce_release.py \
  --data /absolute/path/to/huggingface \
  --out /tmp/grid-bench-reproduced \
  --export-cli /tmp/grid-bench-cli
```

Use the guide emitted in that directory for the corresponding native
`instances.jsonl` and `gold.jsonl` files. This conversion retains source IDs and
groups so that identical-looking IDs from different source datasets cannot be
silently joined.

The native files are grouped under `main/` and `rq2/`. The latter contains the
additional workload instances; the full workload view also uses selected main
instances identified by `in_workload` in the release rows.

## Validation

```sh
python -m pip install -e '.[dev,eval,plots]'
pytest -m 'not frozen_data'
ruff check src tests
pyright
```

Portable tests use synthetic fixtures and the included examples. Tests marked
`frozen_data` inspect the original research archive and are excluded by the
command above. The candidate validation report records the tests actually run;
historical workspace test counts are not substitutes for a fresh candidate run.
