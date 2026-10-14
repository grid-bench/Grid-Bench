# Changelog

## 0.6.0 — 2026-10-14

- Establish Grid-Bench as an independently maintained package with the
  `grid-bench` distribution and CLI, and the `grid_bench` Python namespace.
- Provide one evaluation path for benchmark rows, local data packages, and
  generated instances, with consistent prompt and answer-scoring defaults.
- Keep all selected instances in score denominators; reject duplicate and
  unknown prediction identities.
- Retain the eight reference simulators and configurable trajectory generator.
- Add self-contained tests and offline reconstruction of six study summaries.
- Reject reused generation directories and settings that produce empty or
  duplicate instance selections.
- Support verification of local Hugging Face downloads with cache metadata.
- Publish original instances and measurement records separately from code;
  exclude collected model text, predicted states, and answer-bearing cases.
- License code under MIT and original data under CC BY 4.0.

The formal instances, prompts, reference states, and recorded measurements
retain their original identities. The software rename does not change them.
