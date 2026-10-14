"""Offline one-model reporting and budget regressions; no provider calls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from spatial_worlds.eval.protocol import FORMAL128_PROTOCOL, AttemptResult
from spatial_worlds.eval.run import Dataset, condition_id
from spatial_worlds.eval.study_final_report import evaluate
from spatial_worlds.eval.study_final_store import FinalStore, FormalRunError, digest
from spatial_worlds.prompt import COMPLETION_PROMPT_VERSION, render_prompt
from spatial_worlds.schema import Instance, WorldSpec
from spatial_worlds.simulate import simulate
from spatial_worlds.world import WorldState


def _selection() -> tuple[dict, Dataset]:
    world = WorldSpec((2, 2), (0, 1, 2, 3))
    instances = [
        Instance(
            id=f"sonnet-report-{index}/t000",
            family="metric_paint",
            world=world,
            initial=WorldState(world, (((0, 0), 1),)),
            events=(),
            metadata={
                "tier": "M",
                "nominal_scan_workload": 1,
                "prefix_of": f"sonnet-report-{index}",
            },
        )
        for index in range(3)
    ]
    dataset = Dataset(instances, {item.id: simulate(item).gold() for item in instances})
    request = {
        "parameters": {"max_completion_tokens": 128000, "reasoning_effort": "high"},
        "expected_model": ["claude-sonnet", "vertex_ai/claude-sonnet-5"],
        "evaluation_protocol": FORMAL128_PROTOCOL.version,
    }
    jobs = [
        {
            "model": "claude-sonnet",
            "provider": "openai",
            "dataset": "main",
            "stage": "main",
            "instance_id": item.id,
            "T": 0,
            "request": request,
            "condition_id": condition_id(
                provider="openai",
                model="claude-sonnet",
                representation="cells",
                prompt_version=COMPLETION_PROMPT_VERSION,
                request=request,
            ),
            "prompt_sha256": hashlib.sha256(
                render_prompt(item, prompt_version=COMPLETION_PROMPT_VERSION).encode()
            ).hexdigest(),
            "instance_sha256": digest(item.to_dict()),
            "gold_sha256": digest(dataset.gold[item.id].to_dict()),
        }
        for item in instances
    ]
    manifest = {
        "campaign_id": "sonnet-report-test",
        "phase": "formal128:sonnet-report-test",
        "collection_protocol": "spatial-worlds-evaluation/1.9",
        "prompt_version": COMPLETION_PROMPT_VERSION,
        "models": [{"id": "claude-sonnet"}],
        "lineage": [],
        "jobs": jobs,
        "pilot": {
            "concurrency": 1,
            "jobs": [
                {"dataset": job["dataset"], "instance_id": job["instance_id"]}
                for job in jobs[:2]
            ],
        },
        "workload_reuse_ids": [],
        "selection_sha256": digest(jobs),
    }
    return manifest, dataset


def _saved_job(job: dict) -> dict:
    return {
        **job,
        "collection": {
            "protocol": "spatial-worlds-evaluation/1.9",
            "dispatch_stage": "pilot",
            "execution_stage": "pilot",
            "per_model_concurrency": 1,
            "total_concurrency": 1,
            "pricing": {"input_per_million": 2.0, "output_per_million": 10.0},
        },
    }


@pytest.mark.parametrize("completed", [2, 3])
def test_single_model_report_counts_declared_jobs_and_scores_delivered_outcomes(
    tmp_path: Path, completed: int
) -> None:
    manifest, dataset = _selection()
    scope = manifest["phase"]
    with FinalStore(tmp_path / "ledger.sqlite", manifest["campaign_id"]) as store:
        store.ensure_run(digest(manifest), ["claude-sonnet"])
        store.set_running(["claude-sonnet"])
        for index, job in enumerate(manifest["jobs"][:completed]):
            attempt = store.begin_batch(
                scope,
                [_saved_job(job)],
                [2_000_000],
                formal_scopes=(scope,),
                formal_cap=800_000_000,
                per_model_concurrency=1,
                total_concurrency=1,
            )[0]
            store.finish(
                attempt,
                AttemptResult(
                    text="" if index == 0 else json.dumps(dataset.instances[index].initial.to_dict()),
                    usage={"input_tokens": 10, "output_tokens": 128000},
                    response={"id": f"resp_{index}", "model": "vertex_ai/claude-sonnet-5"},
                    finish_reason="length",
                    termination="length",
                    answer_eligible=True,
                    usage_complete=True,
                ).to_dict(),
            )
        report = evaluate(tmp_path / "report", manifest, {"main": dataset}, store)
        assert report["complete"] is (completed == 3)
        assert report["formal_ranking_available"] is (completed == 3)
        assert report["model_completeness"] == {
            "claude-sonnet": {
                "planned": 3,
                "scored": completed,
                "provider_calls": completed,
                "technical_provider_calls": 0,
                "inflight_calls": 0,
                "empty": 1,
                "complete": completed == 3,
            }
        }
        assert report["formal_accounting"]["attempts"] == completed
        assert report["pilot"] == {"planned": 2, "completed": 2, "complete": True}
        assert all(value == [] for value in report["paired_model_comparisons"].values())
    rows = [json.loads(line) for line in (tmp_path / "report/results.jsonl").read_text().splitlines()]
    assert [row["em"] for row in rows] == [False, True, True if completed == 3 else None]
    assert rows[0]["answer_status"] == "empty"
    assert rows[1]["termination"] == "length" and rows[1]["answer_status"] == "valid"
    assert rows[0]["collection_dispatch_stages"] == ["pilot"]
    assert rows[0]["collection_execution_stages"] == ["pilot"]
    assert "# Claude 5 Sonnet formal evaluation" in (tmp_path / "report/collection.md").read_text()
    notes = "\n".join(report["notes"])
    assert "Pilot completion is not campaign completion" in notes
    assert "1.8" not in notes


def test_budget_error_reports_campaign_cap(tmp_path: Path) -> None:
    manifest, _ = _selection()
    with FinalStore(tmp_path / "ledger.sqlite", manifest["campaign_id"]) as store:
        store.ensure_run(digest(manifest), ["claude-sonnet"])
        store.set_running(["claude-sonnet"])
        with pytest.raises(FormalRunError, match="formal 800 USD ceiling"):
            store.begin_batch(
                manifest["phase"],
                [_saved_job(manifest["jobs"][0])],
                [800_000_001],
                formal_scopes=(manifest["phase"],),
                formal_cap=800_000_000,
                per_model_concurrency=1,
                total_concurrency=1,
            )
        assert store.all_attempts(manifest["phase"]) == []
