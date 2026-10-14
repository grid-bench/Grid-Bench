"""Score prediction records and aggregate them per cell and per scaling curve."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..prompt import COMPLETION_PROMPT_VERSION, PROMPT_VERSION, render_prompt
from ..registry import get_family
from ..schema import GoldRecord, Instance, SchemaError, parse_json_document
from ..scoring import ScoreResult, score_prediction, score_query_prediction
from ..world import WorldState
from .cost import Price, load_prices, record_cost_usd
from .providers import NO_ANSWER_FAILURES
from .run import (
    CONDITION_FIELDS,
    Dataset,
    condition_id,
    failure_kind_of,
    iter_jsonl,
    load_dataset,
)
from .stats import (
    bootstrap_difference,
    bootstrap_epsilon,
    epsilon_mle,
    linear_fit,
    mean,
    percentile,
    residual_sd,
    wilson_interval,
)

CELL_KEYS = ("condition_id", "model", "family", "tier", "dimension", "w", "T")
CURVE_KEYS = ("condition_id", "model", "family", "tier", "dimension", "w")
# Descriptive dimension groupings reported next to the per-cell numbers.  The
# names say which dimensions they pool and carry no claim about what a model
# has or has not seen in training.  Split means use the headline horizons only
# (``HEADLINE_MIN_T`` and above), so the T = 0 copy control and the T = 1
# single-event anchor never enter them.
DIMENSION_SPLITS = {"d1-3": (1, 2, 3), "d4-5": (4, 5)}
HEADLINE_MIN_T = 8
# Baseline providers whose curves are drawn next to the model curves.
BASELINE_PROVIDERS = ("copy-initial", "all-zero", "suffix", "oracle")


def _score(text: str | None, instance: Instance, gold: GoldRecord) -> ScoreResult:
    if instance.queries:
        return score_query_prediction(text or "", gold, instance.queries)
    return score_prediction(text or "", gold)


def _predicted_values(
    text: str | None, instance: Instance, gold: GoldRecord, result: ScoreResult
) -> dict[tuple[int, ...], int] | None:
    """Parsed predicted values by coordinate when the answer is semantically valid."""

    if text is None or not result.valid_state:
        return None
    try:
        payload = parse_json_document(text)
        if instance.queries:
            return {tuple(entry[0]): int(entry[1]) for entry in payload["values"]}
        state, _ = WorldState.parse_dict(payload, gold.world)
        return state.as_dict()
    except (SchemaError, ValueError, TypeError, KeyError):
        return None


def _diagnostic_fields(diagnostics: Mapping[str, Any] | None, horizon: int) -> dict[str, Any]:
    """Row fields taken from the rollout's history-retention diagnostics, if any."""

    fields: dict[str, Any] = {"erasure_fraction": None, "erasure_eligible": None}
    if diagnostics is None:
        return fields
    erasure = (diagnostics.get("erasure_by_T") or {}).get(str(horizon))
    if erasure:
        fields["erasure_fraction"] = erasure.get("fraction")
        fields["erasure_eligible"] = erasure.get("eligible")
    suffix = (diagnostics.get("suffix_by_T") or {}).get(str(horizon)) or {}
    for k in diagnostics.get("suffix_lengths") or []:
        item = suffix.get(str(k))
        fields[f"suffix_em_k{k}"] = None if item is None else bool(item["em"])
        fields[f"suffix_wrong_cells_k{k}"] = None if item is None else int(item["wrong_cells"])
    return fields


def score_record(
    record: Mapping[str, Any],
    instance: Instance,
    gold: GoldRecord,
    trace: Mapping[str, Any] | None,
    diagnostics: Mapping[str, Any] | None = None,
    prices: Mapping[str, Price] | None = None,
) -> dict[str, Any]:
    raw = record.get("raw_text")
    extracted = record.get("extracted_text")
    strict = _score(raw, instance, gold)
    loose = _score(extracted if extracted is not None else raw, instance, gold)
    metadata = instance.metadata
    horizon = instance.horizon

    initial = instance.initial.as_dict()
    final = gold.final_state.as_dict()
    if instance.queries:
        domain = list(instance.queries)
    else:
        domain = list(instance.initial.positions())
    changed = [x for x in domain if initial.get(x, 0) != final.get(x, 0)]
    unchanged_count = len(domain) - len(changed)
    predicted = _predicted_values(
        extracted if extracted is not None else raw, instance, gold, loose
    )
    changed_accuracy: float | None = None
    unchanged_accuracy: float | None = None
    if predicted is not None:
        if changed:
            changed_accuracy = sum(
                1 for x in changed if predicted.get(x, 0) == final.get(x, 0)
            ) / len(changed)
        if unchanged_count:
            changed_set = set(changed)
            unchanged_accuracy = (
                sum(
                    1
                    for x in domain
                    if x not in changed_set and predicted.get(x, 0) == final.get(x, 0)
                )
                / unchanged_count
            )

    row: dict[str, Any] = {
        "instance_id": instance.id,
        "condition_id": record.get("condition_id"),
        "model": record.get("model"),
        "provider": record.get("provider"),
        "run_id": record.get("run_id"),
        "request": record.get("request") or {},
        "prompt_version": record.get("prompt_version"),
        "prompt_sha256": record.get("prompt_sha256"),
        "family": instance.family,
        "tier": metadata.get("tier"),
        "level": metadata.get("level"),
        "dimension": instance.dimension,
        "shape": list(instance.world.shape),
        "cells": instance.world.volume,
        "initial_density": instance.initial.density,
        "final_density": gold.final_state.density,
        "boundary_fraction": 1
        - math.prod(max(side - 2, 0) for side in instance.world.shape) / instance.world.volume,
        "T": horizon,
        "w": metadata.get("nominal_scan_workload"),
        "w_kind": metadata.get("w_kind"),
        "seed": metadata.get("seed"),
        "rollout_id": metadata.get("prefix_of", instance.id),
        "split": metadata.get("split"),
        "set_rate": metadata.get("set_rate"),
        "n_core": metadata.get("core_events", horizon),
        "n_set": metadata.get("set_events", 0),
        "output_contract": "query" if instance.queries else "cells",
        "representation": record.get("representation", "cells"),
        "em": bool(loose.exact_match),
        "strict_em": bool(strict.exact_match),
        "valid_json": bool(loose.valid_json),
        "valid_state": bool(loose.valid_state),
        "canonical_format": bool(loose.canonical_format),
        "wrong_cells": loose.wrong_cells,
        "total_cells": loose.total_cells,
        "cell_accuracy": loose.cell_accuracy,
        "changed_cells": len(changed),
        "changed_cell_accuracy": changed_accuracy,
        "unchanged_cell_accuracy": unchanged_accuracy,
        "copy_baseline_em": not changed,
        "copy_baseline_cell_accuracy": unchanged_count / len(domain) if domain else 1.0,
        "zero_baseline_em": all(final.get(x, 0) == 0 for x in domain),
        "zero_baseline_cell_accuracy": (
            sum(final.get(x, 0) == 0 for x in domain) / len(domain) if domain else 1.0
        ),
        "score_error": loose.error,
        "extraction_method": record.get("extraction_method"),
        "extraction_applied": record.get("extraction_applied"),
        "error": record.get("error"),
        "failure_kind": failure_kind_of(record),
        "finish_reason": record.get("finish_reason"),
        "usage": record.get("usage") or {},
        "latency_s": record.get("latency_s"),
        "attempts": record.get("attempts"),
        "prompt_chars": record.get("prompt_chars"),
        # observed usage times the model's price (None without usage or a price)
        "cost_usd": record_cost_usd(
            record.get("usage"), prices.get(str(record.get("model"))) if prices else None
        ),
    }
    if trace is not None:
        steps = list(trace.get("steps", []))[:horizon]
        core = [s for s in steps if s.get("kind_wanted") is not None]
        row["mean_core_scan"] = mean([float(s["scan"]) for s in core])
        row["mean_core_effect"] = mean([float(s["effect"]) for s in core])
        row["ineffective_core_events"] = sum(1 for s in core if not s.get("effective"))
        row["predicate_false_events"] = sum(1 for s in core if s.get("predicate") is False)
        row["first_predicate"] = steps[0].get("predicate") if steps else None
        row["first_effective"] = steps[0].get("effective") if steps else None
        # cumulative reference work of the prefix: every event, and core events only
        row["cumulative_scan"] = sum(int(s["scan"]) for s in steps)
        row["cumulative_core_scan"] = sum(int(s["scan"]) for s in core)
        row["cumulative_effect"] = sum(int(s["effect"]) for s in steps)
    else:
        row["mean_core_scan"] = None
        row["mean_core_effect"] = None
        row["ineffective_core_events"] = None
        row["predicate_false_events"] = None
        row["first_predicate"] = None
        row["first_effective"] = None
        row["cumulative_scan"] = None
        row["cumulative_core_scan"] = None
        row["cumulative_effect"] = None
    row.update(_diagnostic_fields(diagnostics, horizon))
    return row


def _legacy_condition_id(record: Mapping[str, Any]) -> str:
    return condition_id(
        provider=record.get("provider", "legacy"),
        model=record.get("model"),
        representation=record.get("representation", "cells"),
        prompt_version=record.get("prompt_version", "legacy"),
        request=record.get("request", {}),
    )


def _normalize_record(record: Mapping[str, Any], instance: Instance) -> tuple[dict[str, Any], str]:
    """Validate runner identity fields and add a condition id to legacy records."""

    normalized = dict(record)
    explicit_condition = record.get("condition_id")
    if explicit_condition is not None:
        if not isinstance(explicit_condition, str) or not explicit_condition:
            raise ValueError(f"prediction for {instance.id!r} has an invalid condition_id")
        missing = [field for field in CONDITION_FIELDS if field not in record]
        if missing:
            raise ValueError(
                f"prediction for {instance.id!r} has condition_id but is missing "
                f"{', '.join(missing)}"
            )
        expected_condition = condition_id(
            provider=record["provider"],
            model=record["model"],
            representation=record["representation"],
            prompt_version=record["prompt_version"],
            request=record["request"],
        )
        if explicit_condition != expected_condition:
            raise ValueError(f"prediction for {instance.id!r} has a mismatched condition_id")
        current_condition = explicit_condition
    elif all(field in record for field in CONDITION_FIELDS):
        current_condition = condition_id(
            provider=record["provider"],
            model=record["model"],
            representation=record["representation"],
            prompt_version=record["prompt_version"],
            request=record["request"],
        )
    else:
        current_condition = _legacy_condition_id(record)
    normalized["condition_id"] = current_condition

    representation = record.get("representation", "cells")
    if not isinstance(representation, str):
        raise ValueError(f"prediction for {instance.id!r} has an invalid representation")
    if "prompt_version" in record and record["prompt_version"] not in {
        PROMPT_VERSION,
        COMPLETION_PROMPT_VERSION,
    }:
        raise ValueError(
            f"prediction for {instance.id!r} uses prompt_version "
            f"{record['prompt_version']!r}; expected {PROMPT_VERSION!r}"
        )
    expected_contract = "query" if instance.queries else "cells"
    if "output_contract" in record and record["output_contract"] != expected_contract:
        raise ValueError(f"prediction for {instance.id!r} has a mismatched output_contract")

    recorded_hash = record.get("prompt_sha256")
    if recorded_hash is not None:
        if not isinstance(recorded_hash, str):
            raise ValueError(f"prediction for {instance.id!r} has an invalid prompt_sha256")
        prompt = render_prompt(
            instance,
            representation=representation,
            prompt_version=record.get("prompt_version", PROMPT_VERSION),
        )
        expected_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if recorded_hash != expected_hash:
            raise ValueError(f"prediction for {instance.id!r} has a mismatched prompt_sha256")
        prompt_identity = recorded_hash
    else:
        if explicit_condition is not None:
            raise ValueError(
                f"prediction for {instance.id!r} has condition_id but no prompt_sha256"
            )
        prompt_identity = "legacy"
    return normalized, prompt_identity


def score_records(
    dataset: Dataset,
    predictions: Iterable[Mapping[str, Any]],
    prices: Mapping[str, Price] | None = None,
) -> list[dict[str, Any]]:
    by_id = {instance.id: instance for instance in dataset.instances}
    active: dict[
        tuple[str, str, str],
        tuple[int, dict[str, Any], Instance, GoldRecord, Mapping[str, Any] | None],
    ] = {}
    for index, record in enumerate(predictions):
        if record.get("evaluation_protocol"):
            raise ValueError("versioned study records require 'study evaluate'; no legacy mixing")
        instance = by_id.get(str(record.get("instance_id")))
        if instance is None:
            raise ValueError(
                f"prediction references unknown instance_id {record.get('instance_id')!r}"
            )
        gold = dataset.gold.get(instance.id)
        if gold is None:
            raise ValueError(f"no gold record for {instance.id}")
        normalized, prompt_identity = _normalize_record(record, instance)
        trace = dataset.traces.get(str(instance.metadata.get("prefix_of", "")))
        key = (str(normalized["condition_id"]), instance.id, prompt_identity)
        active[key] = (index, normalized, instance, gold, trace)

    rows: list[dict[str, Any]] = []
    for _, record, instance, gold, trace in sorted(active.values(), key=lambda item: item[0]):
        rollout_id = str(instance.metadata.get("prefix_of", instance.id))
        rows.append(
            score_record(record, instance, gold, trace, dataset.diagnostics.get(rollout_id), prices)
        )
    return rows


def score_formal_record(
    record: Mapping[str, Any],
    instance: Instance,
    gold: GoldRecord,
    trace: Mapping[str, Any] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Versioned scoring of a locked answer; incomplete infrastructure is EM=null."""
    from .extract import SCORING_VERSION, extract_final_answer

    if record.get("scoring_version") != SCORING_VERSION:
        raise ValueError("formal scoring requires the declared scoring version")
    raw = record.get("raw_text") or ""
    extraction = extract_final_answer(raw)
    normalized = {
        **record,
        "extracted_text": extraction.text or "",
        "extraction_method": extraction.method,
        "extraction_applied": extraction.applied,
    }
    row = score_record(normalized, instance, gold, trace, diagnostics)
    row.update(
        {
            key: record.get(key)
            for key in (
                "evaluation_protocol",
                "scoring_version",
                "answer_eligible",
                "termination",
                "response",
                "usage_complete",
                "protocol_violation",
                "stage",
                "dataset",
            )
        }
    )
    # Preserve content diagnostics even when collection conditions disqualify the row.
    row["diagnostic_em"] = row["em"]
    row["diagnostic_strict_em"] = row["strict_em"]
    row["diagnostic_valid_state"] = row["valid_state"]
    row["diagnostic_wrong_cells"] = row["wrong_cells"]
    if not record.get("answer_eligible") or record.get("collection_finalized") is False:
        row.update(
            {
                key: None
                for key in (
                    "em",
                    "strict_em",
                    "valid_state",
                    "valid_json",
                    "wrong_cells",
                    "cell_accuracy",
                    "changed_cell_accuracy",
                    "unchanged_cell_accuracy",
                )
            }
        )
        row["answer_status"] = "unqueried" if not record.get("attempts") else "unresolved"
    elif record.get("refusal"):
        row.update(em=False, strict_em=False, answer_status="refusal")
    else:
        row["answer_status"] = (
            "empty" if not raw.strip() else "valid" if row["valid_state"] else "invalid"
        )
    row["analysis_eligible"] = (
        bool(record.get("answer_eligible"))
        and record.get("collection_finalized") is not False
        and not bool(record.get("protocol_violation"))
    )
    if record.get("protocol_violation"):
        row["diagnostic_answer_status"] = row["answer_status"]
        row["answer_status"] = "protocol_violation"
        row.update(
            {
                key: None
                for key in (
                    "em",
                    "strict_em",
                    "valid_state",
                    "valid_json",
                    "wrong_cells",
                    "cell_accuracy",
                    "changed_cell_accuracy",
                    "unchanged_cell_accuracy",
                )
            }
        )
    return row


def _key(row: Mapping[str, Any], keys: Sequence[str]) -> tuple[Any, ...]:
    return tuple(row.get(key) for key in keys)


def _sort_key(values: Sequence[Any]) -> tuple[tuple[int, Any], ...]:
    return tuple((0, v) if isinstance(v, (int, float)) else (1, str(v)) for v in values)


def _rate(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return sum(1 for r in rows if r.get(field)) / len(rows)


def _mean_field(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    values = [float(r[field]) for r in rows if r.get(field) is not None]
    return mean(values)


def _mean_usage(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    values = [
        float(r["usage"][field])
        for r in rows
        if r.get("usage") and r["usage"].get(field) is not None
    ]
    return mean(values)


def _request_budget(row: Mapping[str, Any]) -> int | None:
    request = row.get("request") or {}
    for key in ("max_tokens", "max_completion_tokens"):
        value = request.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _output_tokens(row: Mapping[str, Any]) -> float | None:
    """Observed output tokens of a record, or None when the provider reported none."""

    usage = row.get("usage") or {}
    value = usage.get("output_tokens")
    return None if value is None else float(value)


def _imputed_output_tokens(row: Mapping[str, Any]) -> float | None:
    """Observed output tokens, with a no-answer failure imputed at the whole request budget.

    The imputed quantity is reported under its own name so that an assumption
    never masquerades as a measurement.
    """

    observed = _output_tokens(row)
    if observed is not None:
        return observed
    if row.get("failure_kind") in NO_ANSWER_FAILURES:
        budget = _request_budget(row)
        return float(budget) if budget is not None else None
    return None


def _is_no_answer(row: Mapping[str, Any]) -> bool:
    return row.get("failure_kind") in NO_ANSWER_FAILURES


def summarize_cells(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_key(row, CELL_KEYS)].append(row)
    summaries: list[dict[str, Any]] = []
    for key in sorted(groups, key=_sort_key):
        group = groups[key]
        n = len(group)
        em_hits = sum(1 for r in group if r["em"])
        strict_hits = sum(1 for r in group if r["strict_em"])
        em_lo, em_hi = wilson_interval(em_hits, n)
        pred_false = [r for r in group if r.get("predicate_false_events")]
        pred_true = [r for r in group if r.get("predicate_false_events") == 0]
        wrong = sorted(
            float(r["wrong_cells"])
            for r in group
            if r.get("valid_state") and not r.get("em") and r.get("wrong_cells") is not None
        )
        suffix_keys = sorted(
            {key for r in group for key in r if key.startswith("suffix_em_k")},
            key=lambda key: int(key.rsplit("k", 1)[1]),
        )
        retention: dict[str, Any] = {}
        for suffix_key in suffix_keys:
            k = suffix_key.rsplit("k", 1)[1]
            observed = [r for r in group if r.get(suffix_key) is not None]
            retention[suffix_key] = _rate(observed, suffix_key) if observed else None
            retention[f"mean_suffix_wrong_cells_k{k}"] = _mean_field(
                group, f"suffix_wrong_cells_k{k}"
            )
        summaries.append(
            {
                **dict(zip(CELL_KEYS, key, strict=True)),
                "provider": group[0].get("provider"),
                "representation": group[0].get("representation"),
                "prompt_version": group[0].get("prompt_version"),
                "request": group[0].get("request") or {},
                "n": n,
                "em": em_hits / n,
                "em_ci_low": em_lo,
                "em_ci_high": em_hi,
                "strict_em": strict_hits / n,
                "valid_json_rate": _rate(group, "valid_json"),
                "valid_state_rate": _rate(group, "valid_state"),
                "canonical_rate": _rate(group, "canonical_format"),
                "error_rate": sum(1 for r in group if r.get("error")) / n,
                "no_answer_rate": sum(1 for r in group if _is_no_answer(r)) / n,
                "other_error_rate": sum(1 for r in group if r.get("error") and not _is_no_answer(r))
                / n,
                "truncated_rate": sum(
                    1 for r in group if r.get("finish_reason") in ("length", "max_tokens")
                )
                / n,
                "cell_accuracy": _mean_field(group, "cell_accuracy"),
                "wrong_cells_median": percentile(wrong, 50) if wrong else None,
                "wrong_cells_n": len(wrong),
                "wrong_cells_population": "valid_incorrect",
                "wrong_cells_p90": percentile(wrong, 90) if wrong else None,
                "changed_cell_accuracy": _mean_field(group, "changed_cell_accuracy"),
                "unchanged_cell_accuracy": _mean_field(group, "unchanged_cell_accuracy"),
                "copy_baseline_em": _rate(group, "copy_baseline_em"),
                "copy_baseline_cell_accuracy": _mean_field(group, "copy_baseline_cell_accuracy"),
                "mean_changed_cells": _mean_field(group, "changed_cells"),
                "mean_core_scan": _mean_field(group, "mean_core_scan"),
                "mean_core_effect": _mean_field(group, "mean_core_effect"),
                "mean_cumulative_scan": _mean_field(group, "cumulative_scan"),
                "mean_cumulative_core_scan": _mean_field(group, "cumulative_core_scan"),
                "mean_cumulative_effect": _mean_field(group, "cumulative_effect"),
                "mean_erasure_fraction": _mean_field(group, "erasure_fraction"),
                **retention,
                "mean_predicate_false_events": _mean_field(group, "predicate_false_events"),
                "em_with_predicate_false": _rate(pred_false, "em") if pred_false else None,
                "em_without_predicate_false": _rate(pred_true, "em") if pred_true else None,
                "mean_output_tokens": _mean_usage(group, "output_tokens"),
                "usage_known_rate": sum(1 for r in group if _output_tokens(r) is not None) / n,
                "output_tokens_p90": _tokens_percentile(group, 90),
                "output_tokens_p90_imputed": _tokens_percentile(group, 90, imputed=True),
                "mean_reasoning_tokens": _mean_usage(group, "reasoning_tokens"),
                "mean_input_tokens": _mean_usage(group, "input_tokens"),
                "mean_latency_s": _mean_field(group, "latency_s"),
                "mean_prompt_chars": _mean_field(group, "prompt_chars"),
                "mean_cost_usd": _mean_field(group, "cost_usd"),
                "total_cost_usd": sum(
                    float(r["cost_usd"]) for r in group if r.get("cost_usd") is not None
                ),
                "cost_known_rate": sum(1 for r in group if r.get("cost_usd") is not None) / n,
            }
        )
    return summaries


def _tokens_percentile(
    rows: Sequence[Mapping[str, Any]], q: float, *, imputed: bool = False
) -> float | None:
    source = _imputed_output_tokens if imputed else _output_tokens
    values = sorted(v for v in (source(r) for r in rows) if v is not None)
    return percentile(values, q) if values else None


# z-score of the 90th percentile of a normal distribution, used to place a
# p90 line on top of the mean token fit
_Z90 = 1.2816


def _token_fit(group: Sequence[Mapping[str, Any]], token_budget: int) -> dict[str, Any]:
    """Fit output tokens against core events and locate the token budget on the fit.

    Only answered instances (no error of any kind) with observed usage enter
    the ordinary-least-squares regression; a failed request that still
    reported usage would otherwise drag the slope.  ``t_at_budget`` reads the
    budget off the mean line and ``t_at_budget_p90`` off the mean line shifted
    up by 1.28 residual standard deviations, an approximate 90th percentile.
    No-answer instances are reported separately through ``t_first_no_answer``
    so that a censored heavy cell is visible rather than averaged away.
    """

    points = [
        (float(r["n_core"]), float(r["usage"]["output_tokens"]))
        for r in group
        if int(r.get("T") or 0) >= 1
        and r.get("error") is None
        and r.get("usage")
        and r["usage"].get("output_tokens") is not None
        and r.get("n_core") is not None
    ]
    fit = linear_fit(points)
    result: dict[str, Any] = {
        "tokens_fit_n": len(points),
        "tokens_intercept": None,
        "tokens_per_event": None,
        "tokens_r2": None,
        "tokens_resid_sd": None,
        "n_core_at_budget": None,
        "t_at_budget": None,
        "t_at_budget_p90": None,
    }
    if fit is not None:
        intercept, slope, r_squared = fit
        spread = residual_sd(points, intercept, slope)
        result.update(
            {
                "tokens_intercept": intercept,
                "tokens_per_event": slope,
                "tokens_r2": r_squared,
                "tokens_resid_sd": spread,
            }
        )
        if slope > 0:
            set_rate = float(group[0].get("set_rate") or 0.0)
            n_core = (token_budget - intercept) / slope
            n_core_p90 = (token_budget - intercept - _Z90 * spread) / slope
            result["n_core_at_budget"] = n_core
            if set_rate < 1:
                result["t_at_budget"] = n_core / (1.0 - set_rate)
                result["t_at_budget_p90"] = n_core_p90 / (1.0 - set_rate)
    no_answer_ts = sorted({int(r["T"]) for r in group if _is_no_answer(r)})
    result["t_first_no_answer"] = no_answer_ts[0] if no_answer_ts else None
    return result


def _first_crossing(points: Sequence[tuple[int, float]], threshold: float) -> float | None:
    """The horizon at which the curve first falls below ``threshold``, interpolated on log2(T+1).

    ``None`` when the curve never reaches the threshold; the last horizon when
    it never falls below it (the curve is then censored).
    """

    last_above: tuple[int, float] | None = None
    for t, value in points:
        if value >= threshold:
            last_above = (t, value)
        elif last_above is not None:
            ta, va = last_above
            xa, xb = math.log2(ta + 1), math.log2(t + 1)
            if va == value:
                return float(ta)
            x = xa + (va - threshold) / (va - value) * (xb - xa)
            return 2**x - 1
    if last_above is None:
        return None
    return float(last_above[0])


def summarize_curves(
    rows: Sequence[Mapping[str, Any]],
    *,
    threshold: float = 0.5,
    bootstrap: int = 200,
    token_budget: int = 128000,
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_key(row, CURVE_KEYS)].append(row)
    curves: list[dict[str, Any]] = []
    for key in sorted(groups, key=_sort_key):
        group = groups[key]
        by_t: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        by_rollout: dict[str, dict[int, Mapping[str, Any]]] = defaultdict(dict)
        for row in group:
            by_t[int(row["T"])].append(row)
            by_rollout[str(row["rollout_id"])][int(row["T"])] = row
        horizons = sorted(by_t)
        em_by_t = {t: _rate(by_t[t], "em") for t in horizons}
        n_by_t = {t: len(by_t[t]) for t in horizons}
        points = [(t, em_by_t[t]) for t in horizons]
        above = [t for t, value in points if value >= threshold]
        t_star_last_pass = max(above) if above else None
        censored = bool(points) and points[-1][1] >= threshold

        all_obs: list[list[tuple[int, bool]]] = []
        conditional_obs: list[list[tuple[int, bool]]] = []
        first_failing: dict[str, int] = defaultdict(int)
        non_monotone = 0
        for rollout_rows in by_rollout.values():
            observations = [
                (int(r["n_core"]), bool(r["em"])) for t, r in sorted(rollout_rows.items()) if t >= 1
            ]
            all_obs.append(observations)
            # the conditional estimate needs an observed, correct copy control
            base = rollout_rows.get(0)
            if base is not None and base["em"]:
                conditional_obs.append(observations)
            failures = [t for t, r in sorted(rollout_rows.items()) if not r["em"]]
            if failures:
                first_failing[str(failures[0])] += 1
                if any(t > failures[0] and r["em"] for t, r in rollout_rows.items()):
                    non_monotone += 1
            else:
                first_failing["never"] += 1
        flat_all = [obs for rollout in all_obs for obs in rollout]
        flat_conditional = [obs for rollout in conditional_obs for obs in rollout]
        epsilon_all = epsilon_mle(flat_all)
        epsilon_conditional = epsilon_mle(flat_conditional) if conditional_obs else None
        ci = bootstrap_epsilon(all_obs, resamples=bootstrap) if epsilon_all is not None else None
        moment = {
            t: (1 - em_by_t[t] ** (1 / by_t[t][0]["n_core"])) if em_by_t[t] > 0 else 1.0
            for t in horizons
            if t >= 1 and by_t[t][0].get("n_core")
        }
        curves.append(
            {
                **dict(zip(CURVE_KEYS, key, strict=True)),
                "provider": group[0].get("provider"),
                "representation": group[0].get("representation"),
                "prompt_version": group[0].get("prompt_version"),
                "request": group[0].get("request") or {},
                "n_rollouts": len(by_rollout),
                "horizons": horizons,
                "em_by_T": em_by_t,
                "n_by_T": n_by_t,
                "threshold": threshold,
                # largest grid horizon whose EM is still at or above the threshold
                "t_star_last_pass": t_star_last_pass,
                # first horizon at which EM falls below the threshold (log-interpolated)
                "t_star_first_cross": _first_crossing(points, threshold),
                "t_star_censored": censored,
                "epsilon": epsilon_all,
                "epsilon_ci_low": ci[0] if ci else None,
                "epsilon_ci_high": ci[1] if ci else None,
                "epsilon_conditional": epsilon_conditional,
                "epsilon_conditional_rollouts": len(conditional_obs),
                "epsilon_moment_by_T": moment,
                # first prefix of a rollout scored wrong: separate requests, not
                # the first internal error of one execution
                "first_failing_prefix": dict(first_failing),
                "non_monotone_rollouts": non_monotone,
                "no_answer_by_T": {
                    t: sum(1 for r in by_t[t] if _is_no_answer(r)) / len(by_t[t]) for t in horizons
                },
                "token_budget": token_budget,
                **_token_fit(group, token_budget),
                "em_at_T1_predicate_true": _rate(
                    [r for r in by_t.get(1, []) if r.get("first_predicate") is True], "em"
                )
                if any(r.get("first_predicate") is True for r in by_t.get(1, []))
                else None,
                "em_at_T1_predicate_false": _rate(
                    [r for r in by_t.get(1, []) if r.get("first_predicate") is False], "em"
                )
                if any(r.get("first_predicate") is False for r in by_t.get(1, []))
                else None,
            }
        )
    return curves


def split_summary(
    cells: Sequence[Mapping[str, Any]],
    *,
    horizons: Iterable[int] | None = None,
    dimensions: Iterable[int] | None = None,
) -> list[dict[str, Any]]:
    """Mean EM per (model, tier, family, dimension split) over the headline horizons.

    ``horizons`` is the explicit mask of grid horizons that enter the mean
    (default: every horizon at or above ``HEADLINE_MIN_T``) and ``dimensions``
    the explicit mask of dimensions (default: every dimension of the
    grouping); each retained cell (d, w, T) weighs the same, and the
    horizons, dimensions, and cell counts actually used are reported next to
    the mean so that a split never silently mixes controls, tiers,
    supplementary dimensions, or coverage.
    """

    mask = None if horizons is None else {int(t) for t in horizons}
    allowed = None if dimensions is None else {int(d) for d in dimensions}
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for cell in cells:
        horizon = int(cell["T"])
        if mask is None:
            if horizon < HEADLINE_MIN_T:
                continue
        elif horizon not in mask:
            continue
        dimension = int(cell["dimension"])
        if allowed is not None and dimension not in allowed:
            continue
        for split, dims in DIMENSION_SPLITS.items():
            if dimension not in dims:
                continue
            for family in (str(cell["family"]), "all"):
                key = (cell.get("condition_id"), cell["model"], cell.get("tier"), family, split)
                groups[key].append(cell)
    return [
        {
            "condition_id": key[0],
            "model": key[1],
            "tier": key[2],
            "family": key[3],
            "split": key[4],
            "dimensions": list(DIMENSION_SPLITS[key[4]]),
            "dimensions_used": sorted({int(c["dimension"]) for c in group}),
            "horizons": sorted({int(c["T"]) for c in group}),
            "cells": len(group),
            "instances": sum(int(c["n"]) for c in group),
            "mean_em": sum(float(c["em"]) for c in group) / len(group),
        }
        for key, group in sorted(groups.items(), key=lambda kv: tuple(str(v) for v in kv[0]))
    ]


def workload_summary(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Cells of one (model, family, tier, dimension) ordered by cumulative core scan.

    This is the view for the horizon-versus-workload question: conditions with
    fewer heavier events and more lighter events can be read off at similar
    cumulative reference work.
    """

    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for cell in cells:
        if int(cell["T"]) < 1:
            continue
        key = (
            cell.get("condition_id"),
            cell["model"],
            cell["family"],
            cell.get("tier"),
            cell["dimension"],
        )
        groups[key].append(cell)
    rows: list[dict[str, Any]] = []
    for key in sorted(groups, key=_sort_key):
        group = groups[key]
        ordered = sorted(
            group,
            key=lambda c: (
                float(c.get("mean_cumulative_core_scan") or 0.0),
                _sort_key((c["w"], c["T"])),
            ),
        )
        for cell in ordered:
            rows.append(
                {
                    "condition_id": key[0],
                    "model": key[1],
                    "family": key[2],
                    "tier": key[3],
                    "dimension": key[4],
                    "w": cell["w"],
                    "T": cell["T"],
                    "n": cell["n"],
                    "mean_cumulative_core_scan": cell.get("mean_cumulative_core_scan"),
                    "mean_cumulative_scan": cell.get("mean_cumulative_scan"),
                    "mean_core_scan": cell.get("mean_core_scan"),
                    "em": cell["em"],
                    "cell_accuracy": cell.get("cell_accuracy"),
                }
            )
    return rows


def model_comparison(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """EM of every model side by side per benchmark cell (family, tier, d, w, T)."""

    groups: dict[tuple[Any, ...], dict[str, float]] = defaultdict(dict)
    for cell in cells:
        key = (cell["family"], cell.get("tier"), cell["dimension"], cell["w"], cell["T"])
        label = str(cell["model"])
        if label in groups[key]:
            label = f"{label}@{str(cell.get('condition_id'))[:8]}"
        groups[key][label] = float(cell["em"])
    return [
        {
            "family": key[0],
            "tier": key[1],
            "dimension": key[2],
            "w": key[3],
            "T": key[4],
            "models": dict(sorted(models.items())),
            "n_models": len(models),
        }
        for key, models in sorted(groups.items(), key=lambda kv: _sort_key(kv[0]))
    ]


def dimension_contrasts(
    rows: Sequence[Mapping[str, Any]],
    *,
    reference_dimension: int = 2,
    bootstrap: int = 200,
) -> list[dict[str, Any]]:
    """EM(d) - EM(reference) per (model, family, tier, w, T) with a two-sample rollout bootstrap.

    Rows at different dimensions come from different rollouts, so the two
    samples are independent; each row is one rollout's prefix at that horizon.
    """

    groups: dict[tuple[Any, ...], dict[int, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (
            row.get("condition_id"),
            row["model"],
            row["family"],
            row.get("tier"),
            row["w"],
            row["T"],
        )
        groups[key][int(row["dimension"])].append(bool(row["em"]))
    contrasts: list[dict[str, Any]] = []
    for key in sorted(groups, key=_sort_key):
        by_dimension = groups[key]
        reference = by_dimension.get(reference_dimension)
        if not reference:
            continue
        for dimension in sorted(by_dimension):
            if dimension == reference_dimension:
                continue
            other = by_dimension[dimension]
            interval = bootstrap_difference(reference, other, resamples=bootstrap)
            contrasts.append(
                {
                    "condition_id": key[0],
                    "model": key[1],
                    "family": key[2],
                    "tier": key[3],
                    "w": key[4],
                    "T": key[5],
                    "reference_dimension": reference_dimension,
                    "dimension": dimension,
                    "n_reference": len(reference),
                    "n": len(other),
                    "em_reference": sum(reference) / len(reference),
                    "em": sum(other) / len(other),
                    "difference": sum(other) / len(other) - sum(reference) / len(reference),
                    "difference_ci_low": interval[0] if interval else None,
                    "difference_ci_high": interval[1] if interval else None,
                }
            )
    return contrasts


def cost_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Observed usage, latency, and cost per (condition, model, T) and per family.

    Rows with ``family = "all"`` pool the families of a horizon; the per-family
    rows are the usage basis that the ``cost`` command projects from.  Means
    use the rows that reported usage; ``cost_known_rate`` says how many did.
    """

    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        for family in ("all", str(row["family"])):
            groups[(row.get("condition_id"), row.get("model"), family, int(row["T"]))].append(row)
    summaries: list[dict[str, Any]] = []
    for key in sorted(groups, key=_sort_key):
        group = groups[key]
        costs = [float(r["cost_usd"]) for r in group if r.get("cost_usd") is not None]
        summaries.append(
            {
                "condition_id": key[0],
                "model": key[1],
                "family": key[2],
                "T": key[3],
                "n": len(group),
                "answered": sum(1 for r in group if r.get("error") is None),
                "usage_known": sum(1 for r in group if _output_tokens(r) is not None),
                "no_answer_rate": sum(1 for r in group if _is_no_answer(r)) / len(group),
                "mean_input_tokens": _mean_usage(group, "input_tokens"),
                "mean_output_tokens": _mean_usage(group, "output_tokens"),
                "output_tokens_p90": _tokens_percentile(group, 90),
                "mean_latency_s": _mean_field(group, "latency_s"),
                "mean_attempts": _mean_field(group, "attempts"),
                "mean_cost_usd": mean(costs),
                "total_cost_usd": sum(costs),
                "cost_known_rate": len(costs) / len(group),
            }
        )
    return summaries


def taxonomy_summary(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Mean EM per (model, support_source, update_type, T) across families."""

    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for cell in cells:
        metadata = get_family(str(cell["family"])).metadata
        key = (
            cell.get("condition_id"),
            cell["model"],
            metadata.support_source,
            metadata.update_type,
            cell["T"],
        )
        groups[key].append(float(cell["em"]))
    return [
        {
            "condition_id": key[0],
            "model": key[1],
            "support_source": key[2],
            "update_type": key[3],
            "T": key[4],
            "cells": len(values),
            "mean_em": sum(values) / len(values),
        }
        for key, values in sorted(groups.items(), key=lambda kv: tuple(str(v) for v in kv[0]))
    ]


def coverage_summary(dataset: Dataset, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Coverage of the loaded dataset, reported separately for every condition."""

    groups: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    identity: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in rows:
        key = (row.get("condition_id"), row.get("model"))
        groups[key].add(str(row["instance_id"]))
        identity[key] = row
    total = len(dataset.instances)
    summaries: list[dict[str, Any]] = []
    for key in sorted(groups, key=_sort_key):
        row = identity[key]
        count = len(groups[key])
        summaries.append(
            {
                "condition_id": key[0],
                "model": key[1],
                "provider": row.get("provider"),
                "representation": row.get("representation"),
                "predicted_instances": count,
                "dataset_instances": total,
                "missing_instances": total - count,
                "coverage": count / total if total else 1.0,
            }
        )
    return summaries


def write_results(
    out: Path,
    rows: Sequence[Mapping[str, Any]],
    cells: Sequence[Mapping[str, Any]],
    curves: Sequence[Mapping[str, Any]],
    taxonomy: Sequence[Mapping[str, Any]],
    coverage: Sequence[Mapping[str, Any]] = (),
    splits: Sequence[Mapping[str, Any]] = (),
    workload: Sequence[Mapping[str, Any]] = (),
    comparison: Sequence[Mapping[str, Any]] = (),
    contrasts: Sequence[Mapping[str, Any]] = (),
    costs: Sequence[Mapping[str, Any]] = (),
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "results.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    _write_csv(out / "summary.csv", cells)
    curve_rows = [
        {
            **{k: v for k, v in curve.items() if not isinstance(v, (dict, list))},
            "em_by_T": json.dumps(curve["em_by_T"]),
            "first_failing_prefix": json.dumps(curve["first_failing_prefix"]),
            "no_answer_by_T": json.dumps(curve.get("no_answer_by_T", {})),
        }
        for curve in curves
    ]
    _write_csv(out / "curves.csv", curve_rows)
    _write_csv(out / "workload.csv", workload)
    _write_csv(out / "contrasts.csv", contrasts)
    _write_csv(out / "costs.csv", costs)
    (out / "results.json").write_text(
        json.dumps(
            {
                "cells": list(cells),
                "curves": list(curves),
                "taxonomy": list(taxonomy),
                "coverage": list(coverage),
                "splits": list(splits),
                "workload": list(workload),
                "comparison": list(comparison),
                "contrasts": list(contrasts),
                "costs": list(costs),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})


def evaluate(
    dataset: Dataset,
    prediction_files: Sequence[Path],
    out: Path,
    *,
    threshold: float = 0.5,
    bootstrap: int = 200,
    token_budget: int = 128000,
    ignore_unknown: bool = False,
    split_horizons: Iterable[int] | None = None,
    split_dimensions: Iterable[int] | None = None,
    prices: Mapping[str, Price] | None = None,
) -> dict[str, Any]:
    predictions: list[dict[str, Any]] = []
    for path in prediction_files:
        predictions.extend(iter_jsonl(path))
    ignored = 0
    if ignore_unknown:
        known = {instance.id for instance in dataset.instances}
        kept = [record for record in predictions if str(record.get("instance_id")) in known]
        ignored = len(predictions) - len(kept)
        predictions = kept
    rows = score_records(dataset, predictions, prices)
    cells = summarize_cells(rows)
    curves = summarize_curves(
        rows, threshold=threshold, bootstrap=bootstrap, token_budget=token_budget
    )
    taxonomy = taxonomy_summary(cells)
    coverage = coverage_summary(dataset, rows)
    splits = split_summary(cells, horizons=split_horizons, dimensions=split_dimensions)
    workload = workload_summary(cells)
    comparison = model_comparison(cells)
    contrasts = dimension_contrasts(rows, bootstrap=bootstrap)
    costs = cost_summary(rows)
    write_results(
        out,
        rows,
        cells,
        curves,
        taxonomy,
        coverage,
        splits,
        workload,
        comparison,
        contrasts,
        costs,
    )
    from .report import write_markdown

    write_markdown(out)
    return {
        "instances": len(rows),
        "ignored_unknown": ignored,
        "cells": len(cells),
        "curves": len(curves),
        "conditions": len(coverage),
        "incomplete_conditions": sum(1 for item in coverage if item["missing_instances"]),
        "out": str(out),
    }


def evaluate_from_args(args: Any) -> int:
    dataset = load_dataset(args.data, instances=args.instances, gold=args.gold, traces=args.traces)
    if not dataset.gold:
        raise ValueError("evaluation needs gold records (--data or --gold)")
    prices_path = getattr(args, "prices", None)
    summary = evaluate(
        dataset,
        list(args.predictions),
        args.out,
        threshold=args.threshold,
        bootstrap=args.bootstrap,
        token_budget=args.token_budget,
        ignore_unknown=bool(getattr(args, "ignore_unknown", False)),
        split_horizons=getattr(args, "split_horizons", None),
        split_dimensions=getattr(args, "split_dimensions", None),
        prices=load_prices(prices_path) if prices_path is not None else None,
    )
    print(json.dumps(summary))
    if summary["instances"] == 0:
        print("warning: no prediction matched a dataset instance", file=sys.stderr)
    elif summary["incomplete_conditions"]:
        print(
            f"warning: {summary['incomplete_conditions']} condition(s) do not cover the "
            "entire loaded dataset; see results.json coverage",
            file=sys.stderr,
        )
    return 0
