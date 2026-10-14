"""Exercise the pilot/full boundary through the real ledger without API calls."""

from __future__ import annotations

import copy
import json
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from spatial_worlds.eval import study_final
from spatial_worlds.eval.protocol import AttemptResult
from spatial_worlds.eval.study_final_config import build_manifest, load_config
from spatial_worlds.eval.study_final_store import FinalStore, FormalRunError, digest, job_key

ROOT = Path(__file__).resolve().parents[1]
MODEL = "claude-sonnet"
PRICE = {"max_input_tokens": 1000000, "input_per_million": 2, "output_per_million": 10}


@pytest.fixture(scope="module")
def source():
    return build_manifest(ROOT, load_config(ROOT / "configs/formal_sonnet.json"))


@pytest.fixture
def manifest(source):
    value = copy.deepcopy(source[0])
    pilot = study_final._pilot_jobs(value)
    keys = {job_key(job) for job in pilot}
    rest = [job for job in value["jobs"] if job_key(job) not in keys][:12]
    value["jobs"] = [*pilot, *rest]
    value["remaining_counts"] = {MODEL: len(value["jobs"])}
    value["manifest_sha256"] = digest(value)
    return value


def outcome(job, text="", finish="stop"):
    return AttemptResult(
        text=text,
        usage={"input_tokens": 20, "output_tokens": 100, "reasoning_tokens": 0},
        response={"id": "msg_test123", "model": job["model"]},
        finish_reason=finish,
        termination="length" if finish == "length" else "stop",
        answer_eligible=True,
        usage_complete=True,
    )


class Credit:
    def authorize(self, items, _pricing):
        return [3280000] * len(items)


def run_stage(store, manifest, datasets, out, stage, stop=None):
    store.set_running((MODEL,))
    with ThreadPoolExecutor(max_workers=8) as pool:
        study_final._run_continuous(
            store,
            cast(Any, SimpleNamespace()),
            manifest,
            datasets,
            {MODEL: PRICE},
            {MODEL: object()},
            pool,
            out,
            stop or threading.Event(),
            cast(Any, Credit()),
            execution_stage=stage,
        )


def fake_execute(monkeypatch, calls, transform=None):
    def execute(store, attempt_id, job, *_args):
        calls.append(job_key(job))
        result = transform(job) if transform else outcome(job)
        store.finish(attempt_id, result.to_dict())
        return result

    monkeypatch.setattr(study_final, "_execute_attempt", execute)


@pytest.mark.frozen_data
def test_pilot_boundary_resume_and_full_never_resample(tmp_path, monkeypatch, source, manifest):
    calls = []
    fake_execute(monkeypatch, calls)
    with FinalStore(tmp_path / "attempts.sqlite", manifest["campaign_id"]) as store:
        store.ensure_run(manifest["manifest_sha256"], (MODEL,))
        run_stage(store, manifest, source[1], tmp_path, "pilot")
        expected = [job_key(job) for job in study_final._pilot_jobs(manifest)]
        assert calls == expected
        assert (store.run_state() or {})["pause_reason"] == "pilot_complete"
        assert store.model_states()[MODEL]["status"] == "paused"
        status = study_final._status(store, manifest)
        assert status["pilot"] == {"planned": 10, "completed": 10, "complete": True}
        assert status["models"][MODEL]["remaining"] == 12
        assert status["dispatch"]["promotion_progress"] == {MODEL: 0}
        run_stage(store, manifest, source[1], tmp_path, "pilot")
        assert calls == expected
        run_stage(store, manifest, source[1], tmp_path, "full")
        assert len(calls) == len(set(calls)) == 22
        assert (store.run_state() or {})["status"] == "complete"
        rows = store.all_attempts(manifest["phase"])
        pilot = [json.loads(row["job"])["collection"] for row in rows[:10]]
        assert all(row["total_concurrency"] == 1 for row in pilot)
        full = [json.loads(row["job"])["collection"] for row in rows[10:]]
        assert all(row["execution_stage"] == "full" for row in full)
        assert all(row["total_concurrency"] == 4 for row in full[:8])
        assert any(row["total_concurrency"] == 8 for row in full[8:])


@pytest.mark.frozen_data
def test_full_rejects_incomplete_pilot_without_dispatch(tmp_path, source, manifest):
    with FinalStore(tmp_path / "attempts.sqlite", manifest["campaign_id"]) as store:
        store.ensure_run(manifest["manifest_sha256"], (MODEL,))
        with pytest.raises(FormalRunError, match="pilot outcomes"):
            run_stage(store, manifest, source[1], tmp_path, "full")
        assert store.all_attempts(manifest["phase"]) == []


@pytest.mark.frozen_data
@pytest.mark.parametrize(
    "finish,text", [("length", ""), ("content_filter", ""), ("stop", "bad JSON")]
)
def test_delivered_failures_do_not_block_pilot_or_retry(
    tmp_path,
    monkeypatch,
    source,
    manifest,
    finish,
    text,
):
    calls = []
    fake_execute(monkeypatch, calls, lambda job: outcome(job, text, finish))
    with FinalStore(tmp_path / "attempts.sqlite", manifest["campaign_id"]) as store:
        store.ensure_run(manifest["manifest_sha256"], (MODEL,))
        run_stage(store, manifest, source[1], tmp_path, "pilot")
        assert len(calls) == len(set(calls)) == 10
        assert (store.run_state() or {})["pause_reason"] == "pilot_complete"
        assert study_final._status(store, manifest)["pilot"]["complete"]


@pytest.mark.frozen_data
def test_technical_failure_pauses_then_resumes_only_unfinished_pilot(
    tmp_path,
    monkeypatch,
    source,
    manifest,
):
    calls = []
    failed_key = job_key(study_final._pilot_jobs(manifest)[2])

    def transform(job):
        if job_key(job) == failed_key:
            return AttemptResult(termination="transport", error="temporary", retryable=True)
        return outcome(job)

    fake_execute(monkeypatch, calls, transform)
    monkeypatch.setattr(study_final, "_retry_delay", lambda *_: 0.0)
    with FinalStore(tmp_path / "attempts.sqlite", manifest["campaign_id"]) as store:
        store.ensure_run(manifest["manifest_sha256"], (MODEL,))
        run_stage(store, manifest, source[1], tmp_path, "pilot")
        assert Counter(calls)[failed_key] == 3
        assert len(calls) == 5
        assert (store.run_state() or {})["pause_reason"] == "one_or_more_models_paused"
        fake_execute(monkeypatch, calls)
        run_stage(store, manifest, source[1], tmp_path, "pilot")
        assert len(calls) == 13 and len(set(calls)) == 10
        assert (store.run_state() or {})["pause_reason"] == "pilot_complete"


@pytest.mark.frozen_data
def test_pilot_stop_is_resumable_and_does_not_finish_campaign(
    tmp_path,
    monkeypatch,
    source,
    manifest,
):
    stop = threading.Event()
    calls = []

    def transform(job):
        stop.set()
        return outcome(job)

    fake_execute(monkeypatch, calls, transform)
    with FinalStore(tmp_path / "attempts.sqlite", manifest["campaign_id"]) as store:
        store.ensure_run(manifest["manifest_sha256"], (MODEL,))
        run_stage(store, manifest, source[1], tmp_path, "pilot", stop)
        assert len(calls) == 1
        assert (store.run_state() or {})["pause_reason"] == "user_stop"
        assert not store.inflight(manifest["phase"])
        fake_execute(monkeypatch, calls)
        run_stage(store, manifest, source[1], tmp_path, "pilot")
        assert len(calls) == len(set(calls)) == 10


@pytest.mark.frozen_data
def test_pilot_does_not_promote_full_and_full_capacity_signal_falls_back(manifest):
    def row(job, stage, result):
        saved = copy.deepcopy(job)
        saved["collection"] = {"execution_stage": stage, "dispatch_stage": "initial"}
        return {
            "job": json.dumps(saved),
            "job_key": job_key(job),
            "model": MODEL,
            "result": json.dumps(result.to_dict()),
        }

    rows = [row(job, "pilot", outcome(job)) for job in manifest["jobs"][:10]]
    assert study_final._dispatch_state(rows, manifest, "full")["stage"] == "initial"
    rows.extend(row(job, "full", outcome(job)) for job in manifest["jobs"][10:18])
    assert study_final._dispatch_state(rows, manifest, "full")["total_concurrency"] == 8
    failed = AttemptResult(
        termination="status",
        error="provider_error: status=429",
        retryable=True,
        trace={"error": {"status": 429}},
    )
    rows.append(row(manifest["jobs"][18], "full", failed))
    assert study_final._dispatch_state(rows, manifest, "full")["total_concurrency"] == 4
