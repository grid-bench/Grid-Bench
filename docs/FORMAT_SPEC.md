# Format Specification

This document defines the data, prompt, and evaluation contracts of version
0.5. The schema version is `spatial-worlds/0.2`; incompatible schema changes
require a new schema version. A stored instance contains enough information
to reproduce and audit its gold state. Study selections and resource policies
are described in [evaluation conditions](evaluation.md), separately from the
generic schema. Historical sections below retain their protocol identities;
the current release contains both formal cohorts described in [released data](data.md).

## 1. Coordinate and state conventions

- A world has dimension `d = len(shape)` and positive side lengths.
- Coordinates are zero-based integer arrays of length `d`; axis indices are
  zero-based integers in `[0, d)`.
- Coordinate `p` is in bounds exactly when `0 <= p[i] < shape[i]` on every axis.
- The domain is bounded and does not wrap.
- Cell values are integers explicitly listed in `world.values`, which is
  strictly increasing and includes the default value `0`.
- Serialized state entries contain only nonzero values; omitted coordinates
  are `0`. A semantic state has at most one value per coordinate.

The canonical order for coordinates is ascending lexicographic tuple order.
Canonical state serialization lists every nonzero coordinate exactly once in
that order. The compact table is representation-neutral: generated worlds have
moderate to high occupancy.

## 2. Instance document

```json
{
  "schema_version": "spatial-worlds/0.2",
  "id": "metric_paint/S/d2/w7/dev/0003/t002",
  "family": "metric_paint",
  "world": {"shape": [9, 9], "values": [0, 1, 2, 3]},
  "initial": {"cells": [[[0, 2], 1], [[5, 6], 3]]},
  "events": [
    {"op": "PAINT", "a": [4, 4], "r": 1, "u": 2, "v": 1},
    {"op": "SET", "p": [3, 4], "v": 2}
  ],
  "metadata": {"seed": 1234, "tier": "S", "dimension": 2, "horizon": 2},
  "queries": [[1, 2], [0, 0]]
}
```

Required keys: `schema_version`, `id`, `family`, `world`, `initial`, `events`.
Optional keys: `metadata` (object, default `{}`) and `queries` (array of
distinct in-bounds coordinates; absent or empty for the default contract).

Instances produced by the generator carry the following metadata:

| Key | Meaning |
| --- | --- |
| `seed`, `dataset_seed`, `split` | rollout seed, dataset seed, split token |
| `tier`, `level`, `shape`, `cells`, `dimension` | shape tier alias, nominal N level, shape, N, d |
| `horizon`, `t_max`, `prefix_of` | this prefix length, the rollout length, the rollout id |
| `nominal_scan_workload`, `w_kind` | the cell's w and its kind: `scan` (per-event scan target), `support` (update-support target: the ball size of `STENCIL`, where a value equal to the tier's nominal N means the whole world, or the region size of `MIX`), or `cells` (an N level) |
| `set_rate`, `noop_rate`, `core_events`, `set_events` | SET proportion, predicate-false target or provisional no-effect proposal (`null` for the families that have none), and prefix counts |
| `initial_density`, `boundary_cell_fraction`, `mean_neighbor_count` | state and shape statistics |
| `initializer`, `set_policy`, `generator_version`, `prompt_version` | provenance |
| `query_policy` | present only when `queries` was requested: `uniform:k` (cells drawn without looking at the final state, the default) or `half-changed:k` (a changed-cell-first diagnostic with an answer-related sampling prior) |

Instance ids follow `{family}/{tier}/d{d}/w{w}/{split}/{index:04d}/t{T:03d}`;
the part before `/t` is the rollout id shared by every prefix of one rollout.

## 3. Events

Every event is a JSON object with an uppercase `op`; event order is array
order. All families accept `{"op": "SET", "p": [..], "v": v}`.

| Family | Event |
| --- | --- |
| Metric Paint | `{"op":"PAINT","a":p,"r":r,"u":u,"v":v}` |
| Box Reflection | `{"op":"REFLECT","lo":p,"hi":p,"axis":a}` |
| Local Stencil | `{"op":"STENCIL","a":p,"r":r}` |
| Occluded Mix | `{"op":"MIX","p":p,"r":r}` |
| Component Relabel | `{"op":"RELABEL","p":p,"v":v}` |
| Boundary Layer | `{"op":"LAYER","p":p,"s":s}` |
| Obstructed Motion | `{"op":"MOVE","v":v,"axis":a,"direction":s,"k":k}` |
| Aggregating Compression | `{"op":"COMPRESS","axis":a,"direction":s}` |

Validation checks coordinate dimensions and bounds, axis range, nonnegative
radii and limits, allowed values, signed `direction` and `LAYER.s` in
`{-1,+1}`, nonzero `MOVE.v`, and `lo_i <= hi_i`; `MIX` additionally requires
an alphabet `0, ..., M-1` with `M >= 3`, so that `M - 1` residues exist. A
valid event has deterministic total semantics, including a specified no-op
when its state predicate is not met.

`spatial-worlds validate-instance` also replays the full sequence, so it rejects
a syntactically valid event whose state-dependent result leaves the declared
alphabet (for example, an out-of-alphabet `COMPRESS` merge).

## 4. Gold record

```json
{
  "schema_version": "spatial-worlds/0.2",
  "instance_id": "metric_paint/S/d2/w7/dev/0003/t002",
  "family": "metric_paint",
  "world": {"shape": [9, 9], "values": [0, 1, 2, 3]},
  "final_state": {"cells": [[[0, 2], 1], [[3, 4], 2], [[5, 6], 3]]},
  "workloads": [{"scan": 5, "effect": 0}, {"scan": 1, "effect": 1}]
}
```

`workloads[t]` describes `events[t]`: `scan` is the length of the family's
canonical dependency/read trace, counting repeated cell incidences when the
rule reads the same coordinate for distinct dependencies, and `effect` is the
number of cells whose value changed. A generated gold record has exactly one
entry per event.

## 5. Trace record

The generator writes one trace per rollout (`trace.jsonl`), separate from the
public instances:

```json
{
  "rollout_id": "metric_paint/S/d2/w7/dev/0003",
  "family": "metric_paint", "tier": "S", "dimension": 2, "shape": [9, 9], "w": 7,
  "seed": 1234, "attempt": 0, "t_grid": [0, 1, 2, 4, 8],
  "steps": [
    {"t": 1, "op": "PAINT", "scan": 7, "effect": 3, "predicate": true, "effective": true,
     "kind_wanted": true, "kind_miss": false, "extras": {"ball": 7, "selected": 3}, "density": 0.55}
  ],
  "stats": {"core_scan_mean": 7.1, "core_w_mean": 7.1, "w_target": 7, "effective_fraction": 0.92, "...": "..."},
  "states": {"0": [...], "1": [...]}
}
```

`predicate` is the outcome of the family's explicit rule condition (`null`
for families without one); `effective` is `predicate` when defined and
`effect > 0` otherwise; `kind_wanted` is the generator's plan for that core
event (`null` for SET) and `kind_miss` records that no candidate of the wanted
kind existed. `core_w_mean` is the mean realized workload in the cell's
`w_kind` units (equal to `core_scan_mean` for scan-targeted families, the
mean update-support size for `STENCIL` and `MIX`) and `w_target` the value
the selector aimed for. `states` is present only with `--full-trace`.

## 6. Model answer

Default contract:

```json
{"cells": [[[0, 2], 1], [[3, 4], 2], [[5, 6], 3]]}
```

Query contract (instances with `queries`):

```json
{"values": [[[1, 2], 0], [[0, 0], 3]]}
```

No prose, markdown fence, reasoning trace, copied rules, shape, or zero-valued
entries (default contract) are part of the canonical answer. Evaluation
distinguishes parse validity, semantic validity, and canonical validity. For
the default contract, a semantically valid prediction with entries in a
different order can be exactly correct while failing canonical validity;
duplicates, out-of-bounds coordinates, explicit zeros, unknown keys, or invalid
values make it invalid. For the query contract every query coordinate must
appear exactly once with an allowed value; canonical validity additionally
requires the listed order.

## 7. Prompt rendering

The historical renderer retains `PROMPT_VERSION = "spatial-worlds-prompt/0.3"`
for byte-identical replay. Current formal collection, generic runs and the
`render-prompt` CLI explicitly select `spatial-worlds-prompt/0.4`.
The unchanged 0.3 base template is:

```text
TASK
Apply all events in order and return the final world.

SPACE
shape = [L1,...,Ld]
Coordinates and axis indices are zero-based. The world is bounded and does not wrap.
Unlisted cells have value 0. Allowed values: [...].
W(x) denotes the value of cell x immediately before the current event.

COMMON EVENT
SET(p,v): set cell p to v.

FAMILY RULE
[the family's rule card]

EXAMPLE
[the family's one-dimensional worked example]

INITIAL
{"cells":[...]}

EVENTS
[...]

OUTPUT
Return only {"cells":[[p,v],...]}, where each entry is [coordinate,value]. Include every nonzero cell exactly once in lexicographic coordinate order. Output the JSON object only, with no code fence or other text.
```

The current formal version `spatial-worlds-prompt/0.4`
prepends the following two lines and one blank line to the unchanged 0.3 bytes:

```text
This is a deterministic simulation of an abstract integer-valued grid.
Keep deliberation concise and reserve enough of the response budget to return the complete final-state JSON.
```

This amendment does not alter the task, examples, answer schema, or scoring. The
formal selection records `prompt_sha256`; historical requests retain their
original prompt identity. See [release provenance](reproduction.md).

Supported input representations and output-contract variants are:

- **Grid representation** (`--representation grid`, `d <= 2` only): the
  INITIAL section becomes a note plus a fixed-width, right-aligned digit grid
  whose rows are axis 0 and columns axis 1 (one row for `d = 1`).
- **Shuffled cells** (`--representation cells_shuffled`, any `d`): the INITIAL
  section becomes the note "The cells below are listed in an arbitrary order."
  followed by the same `{"cells":[...]}` object with its entries in a random
  order seeded by the instance id. The OUTPUT contract is unchanged, so this is
  a pure input-order ablation; `canonical_rate` then measures whether the model
  re-sorts on its own.
- **Query contract** (instances with `queries`): the TASK line reads "Apply
  all events in order and return the final value of each query cell." and the
  OUTPUT section becomes the `values` contract followed by a `QUERIES` section
  listing the coordinates.

The `EXAMPLE` section is the only tutorial material in a prompt: one
one-dimensional worked example per family, written as "On shape [n], listing
the values of cells [0] to [n-1] in order. Before: ... Event: {...}. After:
... " followed by a fixed explanatory note. Its after-state is produced by the
reference simulator when the prompt is rendered, and the section is omitted
when the instance alphabet cannot express the example's values. The prompt
never contains a game name, benchmark taxonomy, or family analogy. JSON is
rendered compactly so prompt length is reproducible.

## 8. Dataset layout and manifest

```text
data/<name>/
  manifest.json
  diagnostics.jsonl          (written by spatial-worlds diagnose)
  sha256.txt                 (historical frozen inventory; preserved)
  content_hashes.json        (new safe generation, if requested)
  <split>/<family>/instances.jsonl
  <split>/<family>/gold.jsonl
  <split>/<family>/trace.jsonl
```

`manifest.json` records `name`, `generator_version`, `schema_version`,
`prompt_version`, `python_version`, `created_utc`, `git_commit`, `git_dirty`, the
invoking `command`, the fully expanded `config` and its `config_sha256`, and one
report per benchmark cell with rollout, instance, rejection, and infeasibility
counts plus a `summary` of realized quantities averaged over the cell's
rollouts: `core_scan_mean`, `core_w_mean`, `core_effect_mean` (changed cells
per core event), `core_scan_effective_mean` and `core_scan_ineffective_mean`
(the scan of the events that did and did not take effect, `null` when a kind
did not occur), `effective_fraction`, `kind_miss_fraction`,
`predicate_false_fraction`, `returns_blocked` (candidates discarded per rollout
by the recent-state return check in
[`generation/rollout.py`](../src/spatial_worlds/generation/rollout.py)), and the density range. Commit
and dirty state are both `null` outside a Git source checkout. A cell is
`infeasible` when its target is structurally unreachable on its shape or when
every attempt was rejected; nothing is silently relaxed.

The only built-in preset is the `smoke` development fixture; the smaller
[quickstart configuration](../examples/quickstart-config.json) is intended for a first run. The four
retained study/calibration JSON configurations correspond to the existing
frozen datasets; their full horizon grids and eight test rollouts are retained
while the completed model study queries prefixes through T=64. A family's configured `tolerance` is the relative band around the workload
target inside which candidate events tie during selection (they are then
drawn uniformly); it is not a per-event acceptance threshold. Whole-rollout
rejection is a separate mechanism implemented in
[`generation/rollout.py`](../src/spatial_worlds/generation/rollout.py).

Seeds are derived as the first eight bytes of
`sha256("{dataset_seed}|{family}|{tier}|d={d}|w={w}|split={split}|i={index}|a={attempt}")`,
so an instance is reproducible from the manifest and its id, and splits are
disjoint generator streams. Byte-identical common rollouts require the same
seed, split, effective cell configuration, generation policy and code—not just
the same id. These conditions hold for the middle-level cells shared between
`configs/rq2.json` and `configs/main.json`.

`scripts/regen.sh` now refuses existing generation targets and accepts only four
explicit study dataset names. `--check` regenerates in owned random temporary
directories and compares all instance/gold/trace files, root-level diagnostics,
normalized configuration and nonvolatile manifest fields; mismatches exit nonzero.
Old `sha256.txt` inventories are not rewritten or blindly trusted. Fresh generation
can write `content_hashes.json`. Use the CLI's `run` and `diagnose` commands with
explicit output paths for local baselines or diagnostics; preserve frozen
outputs.

`diagnostics.jsonl` holds one record per rollout with the history-retention
diagnostics implemented in
[`diagnostics.py`](../src/spatial_worlds/diagnostics.py): `rollout_id`, `family`, `tier`,
`dimension`, `w`, `t_max`, `suffix_lengths`, `erasure_by_T` (for every grid
horizon `T >= 2`: `eligible` effective core events in positions `1..T/2`,
how many of them are `erased`, and the `fraction`), and `suffix_by_T` (for
every grid horizon `T >= 1` and every suffix length `k`: `wrong_cells` and
`em` of the solver that executes every `SET` and the core events in the
last `k` event positions). Evaluation joins the record of a row's rollout to the row.

## 9. Prediction and result records

### 9.1 Historical / generic workflow

This subsection documents `run/evaluate` for historical reproduction and custom
experiments. It is NOT the formal study protocol. Study v1 uses Section 9.2;
the legacy evaluator rejects v1 records rather than silently applying its old policy.

`spatial-worlds run` appends one record per instance to a JSONL file:

| Key | Meaning |
| --- | --- |
| `instance_id`, `model`, `provider`, `run_id` | instance, requested model identifier, provider, and invocation identity; `model` is not proof of the resolved upstream version |
| `condition_id` | SHA-256 identity of provider, model, representation, prompt version, and request parameters |
| `representation`, `output_contract`, `prompt_version`, `request` | how the prompt was rendered and requested; `request` carries the provider's parameters plus the resource policy (`timeout_s`, the wall-clock deadline per attempt, and `max_attempts`, the retry limit), so the condition identity changes whenever the regime does |
| `prompt_sha256`, `prompt_chars` | prompt fingerprint |
| `raw_text`, `extracted_text`, `extraction_method`, `extraction_applied` | the response and the extractor's verdict |
| `usage` (`input_tokens`, `output_tokens`, `reasoning_tokens`, `cached_input_tokens`) | provider-reported usage |
| `finish_reason`, `latency_s`, `attempts`, `error`, `timestamp` | outcome |
| `failure_kind` | legacy error classification: `budget`, `deadline`, `empty`, `refusal`, `status`, or `transport`; this is not itself a validity judgment |

Resume lookup uses `(condition_id, instance_id, prompt_sha256)`. By default the
latest matching record is skipped regardless of outcome; with
`--retry-errors`, a latest failed record is queried again and the appended
replacement becomes authoritative. Thus retries have explicit
last-write-wins semantics without mixing model or prompt conditions. `budget`
and `deadline` failures are excluded from `--retry-errors` unless
`--retry-kinds budget,deadline` (or `all`) names them. This is a retry policy,
not a claim that stochastic generation or gateway deadlines are deterministic.
Records written before `failure_kind` existed are classified from their
`error` text and `finish_reason` when evaluated.

Evaluation validates explicit condition ids and prompt hashes. If input files
contain duplicate records for the same condition, instance, and prompt
identity, the last record in input order is scored. `spatial-worlds evaluate`
writes `results.jsonl` (one active scored row per such key), `summary.csv` (per
condition and benchmark cell), `curves.csv` and `results.json` (per
condition-specific curve, with `em_by_T`, `t_star_last_pass` (the largest
grid horizon with EM at or above the threshold), `t_star_first_cross` (the
log-interpolated horizon at which EM first falls below it), `t_star_censored`,
`epsilon` with a bootstrap interval over rollouts, `epsilon_conditional` with
`epsilon_conditional_rollouts` (rollouts whose copy control was observed and
correct; `null` when there is none), `first_failing_prefix`,
`non_monotone_rollouts`, `no_answer_by_T`, and a linear fit of output tokens
against core events on answered instances only: `tokens_fit_n`,
`tokens_intercept`, `tokens_per_event`, `tokens_r2`, `tokens_resid_sd`, the
horizon `t_at_budget` at which the mean line reaches `--token-budget` (default
128000) and `t_at_budget_p90` for the line shifted up by 1.28 residual standard
deviations, next to `t_first_no_answer`), `workload.csv` (cells of one model,
family, and dimension ordered by cumulative core scan), `contrasts.csv`
(`EM(d) - EM(2)` per family, w, and T with a two-sample rollout bootstrap
interval), `costs.csv` (observed usage, latency, and cost per condition,
model, and horizon, pooled over families and per family), per-condition
dataset coverage, mean EM per dimension grouping (`d1-3` and `d4-5`, per
tier, per family and over all families, over the horizons `T >= 8` or those
given by `--split-horizons` and the dimensions given by `--split-dimensions`,
with the horizons, the dimensions actually present (`dimensions_used`), and
cell counts used), a per-cell model comparison, and `summary.md`, which opens
with a `## Leaderboard` table (one row per model and condition: exact match
at T = 64, or the largest horizon present, pooled over d ∈ {2, 3, 4} and every
family and weighted by instance count; the T = 32 or next-largest column; the
no-answer rate and mean cost per instance at that horizon). Per cell,
`summary.csv` also carries `no_answer_rate` (classified budget and deadline
failures), `other_error_rate`, `usage_known_rate`,
`output_tokens_p90` (observed usage only), `output_tokens_p90_imputed` (each
no-answer instance counted as the whole request budget, labelled as an
assumption), `wrong_cells_median` and `wrong_cells_p90` over valid but
incorrect answers, and, when the dataset carries diagnostics,
`mean_erasure_fraction` and `suffix_em_k<k>` with `mean_suffix_wrong_cells_k<k>`.
Rows and cells built from a dataset with traces also carry `mean_core_scan`,
`mean_core_effect`, and the cumulative `cumulative_scan`,
`cumulative_core_scan`, and `cumulative_effect` of the prefix. With
`--prices FILE` (a table `{"usd_per_million": {model: {"input": x,
"output": y}}}`, see `configs/prices.json`), every row carries `cost_usd`,
the observed usage times the model's price (`null` without usage or a
price; cached input is charged at the input price), and cells carry
`mean_cost_usd`, `total_cost_usd`, and `cost_known_rate`. `spatial-worlds
cost --data DIR --filter F... --basis results.jsonl --prices FILE --model M
[--basis-model B] [--multiplier k]` projects the cost of a selection from a
scored run: each selected instance takes the mean observed usage of the
basis rows with the same family and T (falling back to the same T, then to
the overall mean; the counts per level are reported), the output tokens are
multiplied by `k`, and the model's price applies.
Unknown instance ids are rejected unless `--ignore-unknown` skips and counts
them (used when predictions of a dataset that shares cells are scored against
another). Partial coverage is allowed for filtered or limited runs, but is
reported in `results.json` and warned about by the CLI.

These legacy outputs preserve older behavior: a budget/deadline label can coexist
with a valid answer, and the legacy leaderboard pools available rows. It must not
be used as the formal fixed-mask report. An error string alone does not force EM=0.

### 9.2 Study 1.1 records and state

Evaluation version: `spatial-worlds-evaluation/1.1` (1.0 records remain historical).
Scoring version: `spatial-worlds-scoring/1.0`.
Generator/schema/prompt versions and frozen files do not change.

The shared selection manifest names every model/instance request, its stage and
dataset, prompt/instance/gold hashes, and the original dataset-manifest hashes.
The condition hash retains the existing provider/model/representation/prompt/request
formula. Its `request` now includes normalized actual parameters, expected response
model, route, deadline, attempt policy and the evaluation/scoring versions.
Stage and selection hashes stay outside request identity to preserve exact reuse.
For a smaller technically qualified formal panel, an analysis manifest records
its own hash, the candidate-selection parent hash, and the included model list.

`data/study-v1/attempts.sqlite` is the authoritative append-only attempt/event
history plus immutable bindings, answer locks, pauses and spending reservations.
The ledger path does not change with `--out`. A single writer serializes dispatch.
Portable `attempts.jsonl`, `attempt_events.jsonl` and `provenance.json` exports
retain raw answers, partial text, resource observations and source/evidence metadata.
They do not include authentication headers or arbitrary SDK exception messages.

| Field | Meaning |
| --- | --- |
| `answer_eligible` | Explicit finish marker or the study's own terminal deadline; not inferred from JSON contents. |
| `finish_reason` | Provider end marker, when received. |
| `termination` | `stop`, `length`, `deadline`, `transport`, `status`, or `protocol`. |
| `answer_status` | `valid`, `invalid`, `empty`, `refusal`, `protocol_violation`, `unresolved`, or `unqueried`. |
| `em` | Main-analysis semantic EM; false for eligible invalid/empty/refused output; null for protocol violations, unfinalized/unresolved/unqueried rows. |
| `diagnostic_em`, `diagnostic_wrong_cells` | Raw-content diagnostics retained even for protocol-disqualified rows; never substituted for main EM. |
| `analysis_eligible` | Finalized eligible answer and no protocol violation; independent of correctness. |
| `usage_complete` | Whether final usage was observed; partial usage is retained but cannot release a reservation as if complete. |
| `response` | Observed response ID, model and available backend fingerprint; not proof of fixed weights. |
| `protocol_violation` | An independently recorded resource/identity violation requiring investigation. |
| `known_usage_cost_upper_usd` | Sum of conservatively priced known usage across attempts, not an actual bill. |
| `unknown_or_inflight_reserved_usd` | Unreleased upper-bound reservations for unknown/in-flight attempts. |

Upon receiving a finish marker, persist the answer lock before collecting late
usage. Losing that usage never permits retry. Before a marker, explicit temporary
infrastructure failure may recover independently of text contents. The 7,200-second
whole-attempt deadline is terminal and retains partial text. Three attempts maximum
apply across resumed runs; no best-of, overwrite or retry of terminal model outcomes.
An unclosed attempt without a lock requires reconciliation, not blind resubmission.

`study evaluate` writes `results.jsonl`, `summary.json`, `data_profile.json`,
`report.md` and optional SVG task curves. It enumerates the expected mask even
for zero-record models. Incomplete selections retain null EM and cannot produce
a complete formal ranking. Reports provide T=32/64 views, cell Wilson intervals,
paired comparisons, independent Newcombe–Wilson contrasts and direct length
interactions with whole-trajectory uncertainty. Wrong-cell distributions include
their analysis-eligible valid-incorrect non-refusal denominator, separated by
normal/length/deadline termination; dataset diagnostics are deduplicated by instance.
Epsilon and answered-only token fits do not appear in the formal default report.

For OpenAI's official API, `max_completion_tokens` includes reasoning and visible
output; interrupted streams may lack the final usage chunk. These facts informed
the OpenAI adapter but do not verify a gateway's translation or another provider's
semantics. [Token parameter](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)
and [stream usage](https://developers.openai.com/api/reference/resources/chat).

### 9.3 Historical 128K calibration protocols

Protocols 1.2–1.5 identify closed calibration conditions that established 128K
feasibility, prompt 0.4 behaviour, and endpoint compatibility. They retain their
own request identities and raw records and are never pooled with formal outcomes.
These historical conditions are outside the current release selection. See
[release provenance](reproduction.md) for the included formal evidence.

### 9.4 Completed first-round formal 128K protocol

Evaluation protocol 1.6 fixes prompt 0.4, high effort, 128,000 combined tokens,
trace v2, scoring 1.0, and omitted temperature/`top_p`/seed. Collection protocol
1.8 retained 223 Sol and 196 Mini outcomes from three immutable source scopes,
then collected the remaining 521 and 548 jobs with continuous refill. These
counts describe the historical handoff, not remaining work. The campaign
completed at 2026-09-16T09:13:18Z with 744 outcomes per model. Dispatch
starts at four calls per model/eight total and may promote to eight per model/
sixteen total after the declared infrastructure-only check. Every provider call
records its campaign, collection protocol, dispatch stage, and actual concurrency
limits; protocol-1.7 calls retain their original provenance.

An accepted terminal response is a model outcome even when empty, malformed,
incorrect, refusal-labelled, or length-limited. Only failures before a scorable
outcome can receive bounded infrastructure retries. There are no correctness or
empty-rate dispatch gates. Main and workload completeness remain distinct,
including 96 reused middle rows, and incomplete selections are never ranked. See
the [evaluation guide](evaluation.md) for current budgets and the
[reproduction guide](reproduction.md) for release inputs.

## 10. Answer extraction

### Historical extractor

The historical extractor is deterministic and never repairs content:

1. Strip leading and trailing whitespace.
2. If the entire text is one fenced code block, take its content (`fence`).
3. If the text parses as a JSON object, use it as is (`none` or `fence`).
4. Otherwise scan for balanced top-level `{...}` spans outside string
   literals and use the last one that parses as an object (`last_object`).
5. Otherwise the extraction fails and the raw text is scored (`failed`).

`strict_em` scores the raw text; `em` scores the extracted text. Both use the
same semantic state comparison, so `strict_em` is not byte-for-byte equality
or a test of canonical coordinate order. Duplicates, explicit zeros and
unknown keys remain invalid under either metric. Canonical validity is a
separate field. Choosing the raw-text metric does not determine termination
classification or answer eligibility for an interrupted stream.

### Formal scoring/1.0 extractor

Steps 1–3 remain the same. Otherwise select only the last top-level object
candidate, including an unclosed candidate, and parse it once. If it is malformed,
incomplete or absent, extraction fails: there is no fallback to an earlier object,
repair, gold-dependent choice, or reasoning-channel extraction.
For example, `{"cells":[]} ... {"cells":[` fails under scoring/1.0, whereas the
historical extractor can return the first object. Old records are never silently
rescored into the new condition. Request termination and answer eligibility are
established before semantic scoring and remain independent fields.
