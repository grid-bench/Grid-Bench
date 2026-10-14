"""Reporting for a campaign's declared formal mask."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .protocol import FORMAL128_PROTOCOL, AttemptResult
from .study_final_config import formal_scopes, model_order
from .study_final_store import FinalStore
from .study_report import build_report, materialize_rows, write_report


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    temporary.replace(path)


def _campaign(scope: str, manifest: dict[str, Any]) -> str:
    for source in manifest["lineage"]:
        if scope == source["scope"]:
            return str(source["campaign_id"])
    return manifest["campaign_id"]


def _collection(attempt: dict[str, Any]) -> dict[str, Any]:
    return json.loads(attempt["job"]).get("collection") or {}


def evaluate(
    out: Path,
    manifest: dict[str, Any],
    datasets: dict[str, Any],
    store: FinalStore,
) -> dict[str, Any]:
    # Collection imports reporting only after its own module is initialized, so
    # this local import avoids a module-level cycle while sharing one definition.
    from .study_final import is_model_outcome

    scopes = list(formal_scopes(manifest))
    rows = materialize_rows(
        manifest,
        datasets,
        store,
        protocol=FORMAL128_PROTOCOL,
        prompt_version=manifest["prompt_version"],
        attempt_scopes=scopes,
        terminal_predicate=lambda result, job: is_model_outcome(
            AttemptResult(**dict(result)), dict(job)
        ),
        finalized_only=True,
    )
    attempts = [attempt for scope in scopes for attempt in store.all_attempts(scope)]
    histories: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for attempt in attempts:
        job = json.loads(attempt["job"])
        histories[(job["model"], job["instance_id"])].append(attempt)
    for row in rows:
        history = histories[(row["model"], row["instance_id"])]
        outcomes = []
        for attempt in history:
            if not attempt["result"]:
                continue
            result = AttemptResult(**json.loads(attempt["result"]))
            if is_model_outcome(result, json.loads(attempt["job"])):
                outcomes.append(result)
        if len(outcomes) > 1:
            raise ValueError("a formal job has more than one model outcome")
        row.update(
            provider_call_ids=[attempt["id"] for attempt in history],
            provider_calls=len(history),
            technical_calls=sum(bool(attempt["result"]) for attempt in history) - len(outcomes),
            inflight_calls=sum(not attempt["result"] for attempt in history),
            collection_campaigns=[_campaign(attempt["scope"], manifest) for attempt in history],
            collection_concurrencies=[
                int(_collection(attempt).get("per_model_concurrency", 1))
                for attempt in history
            ],
            collection_total_concurrencies=[
                int(_collection(attempt).get("total_concurrency", 1)) for attempt in history
            ],
            collection_dispatch_stages=[
                str(_collection(attempt).get("dispatch_stage", "serial"))
                for attempt in history
            ],
            collection_execution_stages=[
                str(_collection(attempt).get("execution_stage", "full"))
                for attempt in history
            ],
            collection_protocols=[
                str(_collection(attempt).get("protocol", "spatial-worlds-evaluation/1.6"))
                for attempt in history
            ],
            visible_text_empty=(not outcomes[0].text.strip()) if outcomes else None,
        )
    report = build_report(manifest, rows)
    models = model_order(manifest)
    planned = Counter(job["model"] for job in manifest["jobs"])
    completeness = {}
    for model in models:
        model_rows = [row for row in rows if row["model"] == model]
        completeness[model] = {
            "planned": planned[model],
            "scored": sum(row["em"] is not None for row in model_rows),
            "provider_calls": sum(row["provider_calls"] for row in model_rows),
            "technical_provider_calls": sum(row["technical_calls"] for row in model_rows),
            "inflight_calls": sum(row["inflight_calls"] for row in model_rows),
            "empty": sum(row["visible_text_empty"] is True for row in model_rows),
        }
        completeness[model]["complete"] = (
            planned[model] > 0 and completeness[model]["scored"] == planned[model]
        )
    complete = bool(completeness) and all(value["complete"] for value in completeness.values())
    declared_counts = ", ".join(f"{model}: {planned[model]}" for model in models)
    pilot_jobs = manifest.get("pilot", {}).get("jobs", [])
    pilot_keys = {
        (job.get("model"), job["dataset"], job["instance_id"]) for job in pilot_jobs
    }
    pilot_completed = sum(
        row["em"] is not None
        for row in rows
        if (row["model"], row["dataset"], row["instance_id"]) in pilot_keys
        or (None, row["dataset"], row["instance_id"]) in pilot_keys
    )
    pilot_planned = sum(
        1 if job.get("model") else len(models) for job in pilot_jobs
    )
    pilot = {
        "planned": pilot_planned,
        "completed": pilot_completed,
        "complete": pilot_planned > 0 and pilot_completed == pilot_planned,
    }
    collection_notes = (
        [
            f"Collection protocol 1.9 uses a serial {len(pilot_jobs)}-job pilot "
            "before full collection, "
            "under the same 128K/high/prompt-0.4 model-visible condition.",
            "Pilot completion is not campaign completion. Pilot outcomes remain in the "
            "declared formal mask and are not sampled again during full collection.",
            "Full collection starts at four calls for the model and may promote to eight "
            "after the declared infrastructure-only check; pilot outcomes do not count "
            "toward promotion.",
        ]
        if manifest["collection_protocol"] == "spatial-worlds-evaluation/1.9"
        else [
            f"Collection protocol 2.1 uses a serial {len(pilot_jobs)}-job native-Bedrock "
            "pilot before full collection under the same 128K/prompt-0.4 condition.",
            "GLM and Kimi receive reasoning_effort=high through Bedrock's native additional "
            "model fields; acceptance does not establish identical internal effort scales.",
            "Pilot outcomes remain in the declared formal mask and are not sampled again.",
        ]
        if manifest["collection_protocol"] == "spatial-worlds-evaluation/2.1"
        else [
            f"Collection protocol 2.2 uses a serial {len(pilot_jobs)}-job native-Bedrock "
            "pilot before full collection under the same 128K/prompt-0.4 condition.",
            "MiniMax uses provider-native always-on thinking; no reasoning-effort field "
            "is sent, so this condition is not labeled high effort.",
            "Pilot outcomes remain in the declared formal mask and are not sampled again. "
            "Full collection starts at four calls and may promote to eight after the "
            "declared infrastructure-only check.",
        ]
        if manifest["collection_protocol"] == "spatial-worlds-evaluation/2.2"
        else [
            "Collection protocols 1.6--1.8 retain one model-visible "
            "128K/high/prompt-0.4 condition while recording their operational regimes.",
            "The protocol-1.8 continuation starts at four calls per model and may promote "
            "to eight per model after the declared infrastructure-only check.",
        ]
    )
    report.update(
        campaign_id=manifest["campaign_id"],
        complete=complete,
        formal_ranking_available=complete,
        collection_protocol=manifest["collection_protocol"],
        collection_protocols=sorted(
            {
                protocol
                for row in rows
                for protocol in row.get("collection_protocols", [])
            }
        ),
        model_completeness=completeness,
        formal_accounting=store.accounting(scopes),
        continuation_accounting=store.accounting((manifest["phase"],)),
        notes=[
            *report["notes"],
            f"Declared formal rows: {declared_counts}. Empty, length-limited, malformed "
            "and wrong outputs remain scored outcomes; length termination alone does "
            "not make a complete answer incorrect.",
            "Every provider call is retained, and no delivered model outcome is retried.",
            *collection_notes,
        ],
    )
    if pilot_jobs:
        report["pilot"] = pilot
    out.mkdir(parents=True, exist_ok=True)
    write_report(out, manifest, report, rows)
    _write_json(out / "summary.json", report)
    _write_jsonl(out / "results.jsonl", rows)
    _write_jsonl(out / "attempts.jsonl", store.all_attempts(manifest["phase"]))
    display_names = {
        "gpt-5.6-sol": "GPT-5.6 Sol",
        "gpt-5.4-mini": "GPT-5.4 Mini",
        "claude-sonnet": "Claude 5 Sonnet",
        "zai.glm-5": "GLM-5",
        "moonshotai.kimi-k2.5": "Kimi K2.5",
        "minimax.minimax-m2.5": "MiniMax M2.5",
    }
    title = " / ".join(display_names.get(model, model) for model in models)
    collection_lines = [
        f"# {title} formal evaluation",
        "",
        f"Campaign: `{manifest['campaign_id']}`.",
        "",
        "The declared evaluation mask is complete."
        if complete else "Evaluation is ongoing; a complete ranking is not yet available.",
        "",
        "| Model | Scored / planned | Empty | Total calls | Technical calls | In flight |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in models:
        value = completeness[model]
        collection_lines.append(
            f"| {model} | {value['scored']} / {value['planned']} | "
            f"{value['empty']} | {value['provider_calls']} | "
            f"{value['technical_provider_calls']} | {value['inflight_calls']} |"
        )
    collection_lines += [
        "",
        "Empty, length-limited, malformed, and incorrect answers are scored outcomes, "
        "not reasons to retry.",
        "Length termination alone does not make an answer incorrect; delivered "
        "complete answers are scored on their content.",
        "Only infrastructure failures without a scored outcome can be retried; "
        "every call is retained.",
        "Collection stage, per-model concurrency, and total concurrency are recorded "
        "per call without retroactively changing earlier records.",
        "",
    ]
    if manifest["collection_protocol"] in {
        "spatial-worlds-evaluation/1.9",
        "spatial-worlds-evaluation/2.1",
    }:
        collection_lines += [
            f"Pilot outcomes scored: {pilot_completed} / {pilot_planned}. "
            "Pilot completion is not campaign completion.",
            f"The collection protocol first completes {len(pilot_jobs)} serial pilot jobs.",
            "Pilot outcomes remain in the formal mask and are not sampled again; "
            "there is no empty-answer or accuracy gate.",
            "Full collection starts at the frozen initial concurrency. Only outcomes "
            "newly collected in that stage inform concurrency promotion.",
            "",
        ]
    (out / "collection.md").write_text("\n".join(collection_lines))
    return report
