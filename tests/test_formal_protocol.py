"""Offline regressions for the one active formal workflow; no API calls."""

from __future__ import annotations

import copy
import json
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any, cast

import pytest

from spatial_worlds.cli import _build_parser
from spatial_worlds.eval.aggregate import score_formal_record, score_records
from spatial_worlds.eval.extract import SCORING_VERSION, extract_answer, extract_final_answer
from spatial_worlds.eval.protocol import (
    AttemptResult,
    consume_attempt,
    error_event,
    normalize_usage,
    openai_events,
)
from spatial_worlds.eval.run import Dataset
from spatial_worlds.eval.stats import newcombe_difference, stratified_cluster_interval
from spatial_worlds.eval.study_final import (
    FINISH_REASONS,
    _dispatch_state,
    _operational_concurrency_override,
    _recover_inflight,
    _run_continuous,
    is_model_outcome,
    is_retryable_technical,
)
from spatial_worlds.eval.study_final_config import (
    MODEL_ORDER,
    build_manifest,
    load_config,
)
from spatial_worlds.eval.study_final_gateway import Gateway, _endpoint, _environment, reservation
from spatial_worlds.eval.study_final_store import FinalStore, FormalRunError, digest, job_key
from spatial_worlds.eval.study_report import paired_comparisons, summarize_group
from spatial_worlds.schema import Instance, WorldSpec
from spatial_worlds.simulate import simulate
from spatial_worlds.world import WorldState

ROOT = Path(__file__).resolve().parents[1]
PRICE = {"max_input_tokens": 1_000, "input_per_million": 1.0, "output_per_million": 2.0}


@pytest.fixture
def instance() -> Instance:
    world = WorldSpec((2, 2), (0, 1, 2, 3))
    return Instance(
        id="test/t000",
        family="metric_paint",
        world=world,
        initial=WorldState(world, (((0, 0), 1),)),
        events=(),
        metadata={"tier": "M", "nominal_scan_workload": 1, "prefix_of": "test"},
    )


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"cells":[]}', '{"cells":[]}'),
        ('```json\n{"cells":[]}\n```', '{"cells":[]}'),
        ('old {"cells":[]} final {"cells":[', None),
        ('old {"cells":[]} final {"cells": BAD}', None),
        ('reason {"s":"}"} answer {"cells":[]}', '{"cells":[]}'),
        ("Nothing here", None),
        ("[1,2,3]", None),
    ],
)
def test_final_candidate_only(raw: str, expected: str | None) -> None:
    assert extract_final_answer(raw).text == expected


def test_legacy_extraction_stays_reproducible() -> None:
    raw = '{"cells":[]} new {"cells":['
    assert extract_answer(raw).text == '{"cells":[]}'
    assert extract_final_answer(raw).text is None


@pytest.mark.parametrize("kind", ["stop", "length", "deadline"])
def test_eligible_answer_scores_its_contents(instance: Instance, kind: str) -> None:
    record = {
        "scoring_version": SCORING_VERSION,
        "answer_eligible": True,
        "raw_text": json.dumps(instance.initial.to_dict()),
        "termination": kind,
        "error": "deadline" if kind == "deadline" else None,
    }
    row = score_formal_record(record, instance, simulate(instance).gold())
    assert row["em"] is True
    assert row["answer_status"] == "valid"


@pytest.mark.parametrize(
    "raw,status", [("", "empty"), ("not JSON", "invalid"), ('{"cells":[]}', "valid")]
)
def test_delivered_wrong_answers_are_scored(instance: Instance, raw: str, status: str) -> None:
    row = score_formal_record(
        {
            "scoring_version": SCORING_VERSION,
            "answer_eligible": True,
            "raw_text": raw,
            "termination": "stop",
        },
        instance,
        simulate(instance).gold(),
    )
    assert row["em"] is False
    assert row["answer_status"] == status


def test_partial_text_before_network_end_is_unresolved(instance: Instance) -> None:
    row = score_formal_record(
        {
            "scoring_version": SCORING_VERSION,
            "answer_eligible": False,
            "attempts": 1,
            "raw_text": json.dumps(instance.initial.to_dict()),
            "termination": "transport",
        },
        instance,
        simulate(instance).gold(),
    )
    assert row["em"] is None and row["answer_status"] == "unresolved"


def test_legacy_aggregate_rejects_formal_records(instance: Instance) -> None:
    with pytest.raises(ValueError, match="study evaluate"):
        score_records(
            Dataset([instance], {instance.id: simulate(instance).gold()}),
            [{"evaluation_protocol": "new", "instance_id": instance.id}],
        )


def test_finish_marker_locks_answer_before_lost_usage() -> None:
    logged: list[dict] = []

    def events():
        yield {"kind": "text", "value": '{"cells":[]}'}
        yield {"kind": "end", "value": "stop"}
        yield {"kind": "text", "value": "LATE"}
        raise ConnectionError("secret-key-must-not-appear")

    result = consume_attempt(events, on_event=logged.append)
    assert result.answer_eligible and not result.retryable
    assert result.text == '{"cells":[]}' and result.termination == "stop"
    assert sum(event["kind"] == "answer_locked" for event in logged) == 1
    assert "secret-key" not in json.dumps(logged)


def test_pre_marker_error_is_retryable_even_after_partial_json() -> None:
    def events():
        yield {"kind": "text", "value": '{"cells":[]}'}
        raise ConnectionError("broken")

    result = consume_attempt(events)
    assert not result.answer_eligible and result.retryable
    assert result.text == '{"cells":[]}'


def test_eof_without_marker_is_infrastructure() -> None:
    result = consume_attempt(lambda: iter([{"kind": "text", "value": "answer"}]))
    assert result.termination == "transport" and result.retryable
    assert not result.answer_eligible


def test_deadline_bounds_blocking_connection() -> None:
    created = threading.Event()

    def events():
        created.set()
        time.sleep(0.2)
        yield {"kind": "handle", "handle": NS(close=lambda: None)}

    started = time.monotonic()
    result = consume_attempt(events, deadline_s=0.025)
    assert time.monotonic() - started < 0.15
    assert created.is_set() and result.termination == "deadline"


@pytest.mark.parametrize(
    "status,retryable",
    [(400, False), (401, False), (403, False), (408, True), (409, True), (429, True), (503, True)],
)
def test_status_classification_is_sanitized(status: int, retryable: bool) -> None:
    class FakeStatusError(RuntimeError):
        status_code: int
        response: object

    error = FakeStatusError("Authorization: secret")
    error.status_code = status
    error.response = NS(headers={"retry-after": "12"})
    event = error_event(error)
    assert event["retryable"] is retryable and event["retry_after"] == 12
    assert "secret" not in json.dumps(event)


def test_usage_does_not_double_count_reasoning() -> None:
    usage = normalize_usage(
        NS(
            prompt_tokens=100,
            completion_tokens=500,
            completion_tokens_details=NS(reasoning_tokens=450),
        ),
        "openai",
    )
    assert usage["output_tokens"] == 500


def test_formal_openai_stream_sends_only_the_frozen_parameters() -> None:
    sent: dict = {}
    model = "gpt-5.6-sol"

    def create(**kwargs):
        sent.update(kwargs)
        return iter(
            [
                NS(id="", model=model, choices=[], usage=None),
                NS(
                    id="resp_123",
                    model=model,
                    system_fingerprint=None,
                    choices=[
                        NS(
                            index=0,
                            delta=NS(
                                content='{"cells":[]}',
                                refusal=None,
                                tool_calls=None,
                                function_call=None,
                            ),
                            finish_reason="stop",
                        )
                    ],
                    usage=NS(
                        prompt_tokens=10,
                        completion_tokens=20,
                        prompt_tokens_details=None,
                        completion_tokens_details=NS(reasoning_tokens=5),
                    ),
                ),
            ]
        )

    client = NS(chat=NS(completions=NS(create=create)))
    params = {"max_completion_tokens": 128000, "reasoning_effort": "high"}
    result = consume_attempt(
        lambda: openai_events(client, model, "prompt", params, trace=True, trace_version=2),
        trace_enabled=True,
    )
    assert sent == {
        "model": model,
        "messages": [{"role": "user", "content": "prompt"}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "n": 1,
        **params,
    }
    assert result.text == '{"cells":[]}' and result.finish_reason == "stop"
    assert result.response == {"id": "resp_123", "model": model}
    assert result.usage_complete and result.trace_error is None


def _job(index: int = 0, model: str = "gpt-5.6-sol") -> dict:
    expected = [model, f"azure/{model}"]
    return {
        "model": model,
        "provider": "openai",
        "dataset": "main",
        "stage": "main",
        "instance_id": f"instance-{index}",
        "T": index,
        "condition_id": f"condition-{model}",
        "prompt_sha256": f"prompt-{index}",
        "instance_sha256": f"instance-sha-{index}",
        "gold_sha256": f"gold-{index}",
        "request": {
            "parameters": {"max_completion_tokens": 128000, "reasoning_effort": "high"},
            "expected_model": expected,
            "evaluation_protocol": "spatial-worlds-evaluation/1.6",
        },
    }


def _outcome(job: dict, *, text: str = "", finish: str = "stop") -> AttemptResult:
    return AttemptResult(
        text=text,
        usage={"input_tokens": 10, "output_tokens": 20},
        response={"id": "resp_123", "model": job["model"]},
        finish_reason=finish,
        termination="length" if finish == "length" else "stop",
        answer_eligible=True,
        usage_complete=True,
    )


@pytest.mark.parametrize("finish", sorted(FINISH_REASONS))
def test_empty_delivered_outputs_are_terminal_model_outcomes(finish: str) -> None:
    job = _job()
    result = _outcome(job, finish=finish)
    assert is_model_outcome(result, job)
    assert not is_retryable_technical(result)


def test_wrong_or_malformed_text_is_never_retried() -> None:
    job = _job()
    for text in ("wrong", "{", '{"cells":[]}'):
        assert is_model_outcome(_outcome(job, text=text), job)


@pytest.mark.parametrize(
    "result",
    [
        AttemptResult(error="stream_ended_without_finish_marker", retryable=True),
        AttemptResult(termination="deadline", error="whole_attempt_deadline", answer_eligible=True),
    ],
)
def test_only_pre_score_technical_failures_retry(result: AttemptResult) -> None:
    assert not is_model_outcome(result, _job())
    assert is_retryable_technical(result)


def test_identity_or_cap_violation_is_not_a_model_outcome() -> None:
    job = _job()
    missing_id = _outcome(job)
    missing_id.response.pop("id")
    assert not is_model_outcome(missing_id, job)
    excessive = _outcome(job)
    excessive.usage["output_tokens"] = 128001
    assert not is_model_outcome(excessive, job)


def _persisted(job: dict) -> dict:
    value = copy.deepcopy(job)
    value["collection"] = {"pricing": PRICE, "protocol": "test"}
    return value


def test_store_is_single_writer_per_campaign_and_budget_is_atomic(tmp_path: Path) -> None:
    path = tmp_path / "ledger.sqlite"
    with FinalStore(path, "campaign") as store:
        store.ensure_run("manifest", MODEL_ORDER)
        store.set_running(MODEL_ORDER)
        with pytest.raises(FormalRunError, match="writer"):
            FinalStore(path, "campaign")
        with FinalStore(path, "other") as other:
            other.ensure_run("other-manifest", MODEL_ORDER)
            other.set_running(MODEL_ORDER)
            other_ids = other.begin_batch(
                "other-scope",
                [_persisted(_job(99))],
                [100_000],
                formal_scopes=("other-scope",),
                formal_cap=100_000,
                per_model_concurrency=1,
                total_concurrency=1,
            )
            # The first campaign's one-call limit does not include another
            # campaign's in-flight call, even when they share the SQLite file.
            ids = store.begin_batch(
                "scope",
                [_persisted(_job())],
                [100_000],
                formal_scopes=("scope",),
                formal_cap=100_001,
                per_model_concurrency=1,
                total_concurrency=1,
            )
            with pytest.raises(FormalRunError, match="already in flight"):
                store.begin_batch(
                    "scope",
                    [_persisted(_job(99))],
                    [1],
                    formal_scopes=("scope",),
                    formal_cap=100_002,
                    per_model_concurrency=2,
                    total_concurrency=2,
                )
            other.finish(other_ids[0], _outcome(_job(99)).to_dict())
        store.finish(ids[0], _outcome(_job()).to_dict())
        second = store.begin_batch(
            "scope",
            [_persisted(_job(1))],
            [1],
            formal_scopes=("scope",),
            formal_cap=100_001,
            per_model_concurrency=2,
            total_concurrency=2,
        )
        with pytest.raises(FormalRunError, match="already in flight"):
            store.begin_batch(
                "scope",
                [_persisted(_job(1))],
                [1],
                formal_scopes=("scope",),
                formal_cap=100_002,
                per_model_concurrency=3,
                total_concurrency=3,
            )
        store.finish(second[0], _outcome(_job(1)).to_dict())
        with pytest.raises(FormalRunError, match="ceiling"):
            store.begin_batch(
                "scope",
                [_persisted(_job(2))],
                [100_000],
                formal_scopes=("scope",),
                formal_cap=100_020,
                per_model_concurrency=2,
                total_concurrency=2,
            )


def test_concurrent_campaigns_share_an_atomic_account_ceiling(tmp_path: Path) -> None:
    path = tmp_path / "ledger.sqlite"
    account = {
        "spend_usd": 9.0,
        "max_budget_usd": 10.0,
        "budget_reset_at": "2026-10-01T00:00:00Z",
    }
    shared = {"group": "same-key", "buffer_usd": 0.1}
    with FinalStore(path, "first") as first, FinalStore(path, "second") as second:
        for store in (first, second):
            store.ensure_run(store.campaign_id, MODEL_ORDER)
            store.ensure_account_group(shared["group"], account, shared["buffer_usd"])
            store.set_running(MODEL_ORDER)
        first.begin_batch(
            "first-scope",
            [_persisted(_job())],
            [600_000],
            formal_scopes=("first-scope",),
            formal_cap=1_000_000,
            per_model_concurrency=1,
            total_concurrency=1,
            shared_account=shared,
            live_account=account,
        )
        with pytest.raises(FormalRunError, match="shared account ceiling"):
            second.begin_batch(
                "second-scope",
                [_persisted(_job(1))],
                [400_000],
                formal_scopes=("second-scope",),
                formal_cap=1_000_000,
                per_model_concurrency=1,
                total_concurrency=1,
                shared_account=shared,
                live_account=account,
            )
        assert second.all_attempts("second-scope") == []


def test_fallback_does_not_block_another_model_while_old_calls_drain(tmp_path: Path) -> None:
    path = tmp_path / "ledger.sqlite"
    first_model, second_model = MODEL_ORDER
    with FinalStore(path, "campaign") as store:
        store.ensure_run("manifest", MODEL_ORDER)
        store.set_running(MODEL_ORDER)
        old = store.begin_batch(
            "scope",
            [_persisted(_job(index, first_model)) for index in range(3)],
            [1, 1, 1],
            formal_scopes=("scope",),
            formal_cap=100,
            per_model_concurrency=8,
            total_concurrency=16,
        )
        other = store.begin_batch(
            "scope",
            [_persisted(_job(10, second_model))],
            [1],
            formal_scopes=("scope",),
            formal_cap=100,
            per_model_concurrency=2,
            total_concurrency=4,
        )
        assert len(other) == 1
        store.finish(other[0], _outcome(_job(10, second_model)).to_dict())
        with pytest.raises(FormalRunError, match="per-model concurrency"):
            store.begin_batch(
                "scope",
                [_persisted(_job(11, first_model))],
                [1],
                formal_scopes=("scope",),
                formal_cap=100,
                per_model_concurrency=2,
                total_concurrency=4,
            )
        for attempt_id, index in zip(old, range(3), strict=True):
            store.finish(attempt_id, _outcome(_job(index, first_model)).to_dict())


def test_store_preserves_every_retry_and_releases_known_cost(tmp_path: Path) -> None:
    path = tmp_path / "ledger.sqlite"
    job = _job()
    with FinalStore(path, "campaign") as store:
        store.ensure_run("manifest", MODEL_ORDER)
        store.set_running(MODEL_ORDER)
        first = store.begin_batch(
            "scope",
            [_persisted(job)],
            [500_000],
            formal_scopes=("scope",),
            formal_cap=1_000_000,
            per_model_concurrency=1,
            total_concurrency=1,
        )[0]
        store.finish(first, AttemptResult(error="network", retryable=True).to_dict())
        second = store.begin_batch(
            "scope",
            [_persisted(job)],
            [500_000],
            formal_scopes=("scope",),
            formal_cap=1_000_000,
            per_model_concurrency=1,
            total_concurrency=1,
        )[0]
        store.finish(second, _outcome(job).to_dict())
        rows = store.history(job)
        assert [row["number"] for row in rows] == [1, 2]
        assert rows[0]["known_upper"] is None and rows[1]["known_upper"] == 50
        assert store.accounting(("scope",))["accounted_upper_usd"] == 0.50005


def test_incomplete_usage_never_releases_the_reservation(tmp_path: Path) -> None:
    path = tmp_path / "ledger.sqlite"
    job = _job()
    with FinalStore(path, "campaign") as store:
        store.ensure_run("manifest", MODEL_ORDER)
        store.set_running(MODEL_ORDER)
        attempt = store.begin_batch(
            "scope",
            [_persisted(job)],
            [500_000],
            formal_scopes=("scope",),
            formal_cap=1_000_000,
            per_model_concurrency=1,
            total_concurrency=1,
        )[0]
        result = _outcome(job)
        result.usage_complete = False
        store.finish(attempt, result.to_dict())
        row = store.history(job)[0]
        assert row["known_upper"] is None and row["accounted"] == 500_000


def test_restart_recovers_a_locked_answer_without_resampling(tmp_path: Path) -> None:
    path = tmp_path / "ledger.sqlite"
    job = _job()
    with FinalStore(path, "campaign") as store:
        store.ensure_run("manifest", MODEL_ORDER)
        store.set_running(MODEL_ORDER)
        attempt = store.begin_batch(
            "scope",
            [_persisted(job)],
            [500_000],
            formal_scopes=("scope",),
            formal_cap=1_000_000,
            per_model_concurrency=1,
            total_concurrency=1,
        )[0]
        store.event(attempt, {"kind": "answer_locked", "value": _outcome(job).to_dict()})
        store.event(
            attempt,
            {
                "kind": "usage",
                "value": {"input_tokens": 10, "output_tokens": 20},
                "final": True,
            },
        )
    with FinalStore(path, "campaign") as store:
        _recover_inflight(store, "scope")
        rows = store.history(job)
        assert len(rows) == 1 and rows[0]["result"] is not None
        assert is_model_outcome(AttemptResult(**json.loads(rows[0]["result"])), job)


class _SchedulerStore:
    def __init__(self) -> None:
        self.batches: list[tuple[float, list[dict]]] = []
        self.paused: list[tuple[str, str, str | None]] = []
        self.states: dict[str, dict[str, Any]] = {
            model: {"status": "active"} for model in MODEL_ORDER
        }
        self.run: dict[str, Any] = {"status": "running"}
        self.next_id = 1

    def run_state(self) -> dict:
        return self.run

    def model_states(self) -> dict:
        return copy.deepcopy(self.states)

    def all_attempts(self, _scope: str) -> list[dict]:
        return []

    def begin_batch(
        self, _scope: str, jobs: list[dict], _reservations: list[int], **_: object
    ) -> list[int]:
        self.batches.append((time.monotonic(), jobs))
        ids = list(range(self.next_id, self.next_id + len(jobs)))
        self.next_id += len(jobs)
        return ids

    def record_dispatch(self, _document: dict) -> None:
        pass

    def set_model_state(self, model: str, state: str, reason: str | None = None) -> None:
        self.paused.append((model, state, reason))
        self.states[model] = {"status": state, "pause_reason": reason}

    def set_run_state(self, state: str, reason: str | None = None) -> None:
        self.run = {"status": state, "pause_reason": reason}


class _Credit:
    def authorize(self, items: list, _pricing: dict) -> list[int]:
        return [1] * len(items)


def _scheduler_manifest(jobs: list[dict]) -> dict:
    counts = Counter(job["model"] for job in jobs)
    return {
        "campaign_id": "campaign",
        "phase": "scope",
        "prompt_version": "p",
        "collection_protocol": "test",
        "formal_budget_usd": 10,
        "credit_buffer_usd": 1,
        "metadata_refresh_s": 10_000,
        "lineage": [],
        "lineage_outcomes": {model: 0 for model in MODEL_ORDER},
        "remaining_counts": {model: counts[model] for model in MODEL_ORDER},
        "jobs": jobs,
        "resources": {
            "max_tokens": 128000,
            "deadline_s": 7200,
            "usage_grace_s": 5,
            "technical_attempts_per_run": 3,
        },
        "dispatch": {
            "initial_per_model_concurrency": 1,
            "initial_total_concurrency": 2,
            "promotion_outcomes_per_model": 99,
            "promoted_per_model_concurrency": 2,
            "promoted_total_concurrency": 4,
            "account_refresh_s": 300,
        },
    }


def test_continuous_scheduler_refills_around_a_slow_call(tmp_path: Path, monkeypatch) -> None:
    from spatial_worlds.eval import study_final

    jobs = [
        *[_job(index, "gpt-5.6-sol") for index in range(3)],
        *[_job(index + 10, "gpt-5.4-mini") for index in range(3)],
    ]
    slow_finished = 0.0

    def execute(_store, _attempt_id, job, _prompt, _client, _manifest):
        nonlocal slow_finished
        if job["model"] == "gpt-5.6-sol" and job["instance_id"] == "instance-0":
            time.sleep(0.2)
            slow_finished = time.monotonic()
        else:
            time.sleep(0.01)
        return _outcome(job)

    monkeypatch.setattr(study_final, "_execute_attempt", execute)
    monkeypatch.setattr(study_final, "_publish", lambda *a, **k: {})
    monkeypatch.setattr(study_final, "render_prompt", lambda *a, **k: "prompt")
    dataset = NS(instances=[NS(id=job["instance_id"]) for job in jobs])
    manifest = _scheduler_manifest(jobs)
    store = _SchedulerStore()
    with ThreadPoolExecutor(max_workers=2) as pool:
        _run_continuous(
            cast(FinalStore, store),
            cast(Gateway, NS()),
            manifest,
            {"main": dataset},
            {model: PRICE for model in MODEL_ORDER},
            {model: NS() for model in MODEL_ORDER},
            pool,
            tmp_path,
            threading.Event(),
            cast(Any, _Credit()),
        )
    assert len(store.batches) > 2 and slow_finished > 0
    assert any(
        timestamp_value < slow_finished
        and any(job["model"] == "gpt-5.4-mini" for job in batch)
        for timestamp_value, batch in store.batches[1:]
    )
    assert store.run["status"] == "complete"


def test_retry_is_nonblocking_and_uses_only_the_failed_job(tmp_path: Path, monkeypatch) -> None:
    from spatial_worlds.eval import study_final

    left, right = _job(1), _job(2, "gpt-5.4-mini")
    jobs = [left, right]
    calls = Counter()

    def execute(_store, _attempt_id, job, _prompt, _client, _manifest):
        calls[job_key(job)] += 1
        if job is left and calls[job_key(job)] == 1:
            return AttemptResult(error="temporary", retryable=True)
        return _outcome(job)

    monkeypatch.setattr(study_final, "_execute_attempt", execute)
    monkeypatch.setattr(study_final, "_publish", lambda *a, **k: {})
    monkeypatch.setattr(study_final, "_retry_delay", lambda *_: 0.0)
    monkeypatch.setattr(study_final, "render_prompt", lambda *a, **k: "prompt")
    manifest = _scheduler_manifest(jobs)
    store = _SchedulerStore()
    dataset = NS(instances=[NS(id=job["instance_id"]) for job in jobs])
    with ThreadPoolExecutor(max_workers=2) as pool:
        _run_continuous(
            cast(FinalStore, store),
            cast(Gateway, NS()),
            manifest,
            {"main": dataset},
            {model: PRICE for model in MODEL_ORDER},
            {model: NS() for model in MODEL_ORDER},
            pool,
            tmp_path,
            threading.Event(),
            cast(Any, _Credit()),
        )
    assert calls == Counter({job_key(left): 2, job_key(right): 1})
    assert not any(state == "paused" for _, state, _ in store.paused)


@pytest.mark.frozen_data
def test_manifest_is_the_frozen_two_model_mask() -> None:
    config = load_config(ROOT / "configs/formal.json")
    manifest, _ = build_manifest(ROOT, config)
    assert manifest["manifest_sha256"] == digest(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    assert manifest["resources"] == {
        "max_tokens": 128000,
        "deadline_s": 7200,
        "usage_grace_s": 5,
        "technical_attempts_per_run": 3,
    }
    assert manifest["dispatch"] == {
        "initial_per_model_concurrency": 4,
        "initial_total_concurrency": 8,
        "promotion_outcomes_per_model": 8,
        "promoted_per_model_concurrency": 8,
        "promoted_total_concurrency": 16,
        "account_refresh_s": 300,
    }
    assert manifest["lineage_outcomes"] == {
        "gpt-5.6-sol": 223,
        "gpt-5.4-mini": 196,
    }
    assert manifest["remaining_counts"] == {
        "gpt-5.6-sol": 521,
        "gpt-5.4-mini": 548,
    }


@pytest.mark.frozen_data
def test_dispatch_promotes_only_on_outcomes_and_falls_back_on_capacity() -> None:
    manifest, _ = build_manifest(ROOT, load_config(ROOT / "configs/formal.json"))
    selected = {
        model: [job for job in manifest["jobs"] if job["model"] == model][:9]
        for model in MODEL_ORDER
    }
    rows = []
    for model in MODEL_ORDER:
        for job in selected[model][:8]:
            saved = copy.deepcopy(job)
            saved["collection"] = {"dispatch_stage": "initial"}
            rows.append(
                {
                    "job": json.dumps(saved),
                    "job_key": job_key(job),
                    "model": model,
                    "result": json.dumps(_outcome(job).to_dict()),
                }
            )
    promoted = _dispatch_state(rows, manifest)
    assert promoted["stage"] == "promoted"
    assert promoted["total_concurrency"] == 16

    failed = selected["gpt-5.6-sol"][8]
    saved = copy.deepcopy(failed)
    saved["collection"] = {"dispatch_stage": "promoted"}
    rows.append(
        {
            "job": json.dumps(saved),
            "job_key": job_key(failed),
            "model": "gpt-5.6-sol",
            "result": json.dumps(
                AttemptResult(
                    termination="status",
                    error="provider_error: status=429",
                    retryable=True,
                    trace={"error": {"status": 429, "code": "rate_limit_exceeded"}},
                ).to_dict()
            ),
        }
    )
    fallback = _dispatch_state(rows, manifest, "full")
    assert fallback["stage"] == "fallback"
    assert fallback["total_concurrency"] == 8
    assert fallback["capacity_signals"] == ["status_429"]


@pytest.mark.frozen_data
def test_aws_fallback_persists_audited_operational_concurrency() -> None:
    manifest, _ = build_manifest(ROOT, load_config(ROOT / "configs/formal_aws.json"))
    job = manifest["jobs"][0]
    override = _operational_concurrency_override(manifest, "full", 4, 8)
    assert override is not None
    saved = copy.deepcopy(job)
    saved["collection"] = {
        "dispatch_stage": "promoted",
        "operational_concurrency_override": override,
    }
    rows = [
        {
            "job": json.dumps(saved),
            "job_key": job_key(job),
            "model": job["model"],
            "result": json.dumps(
                AttemptResult(
                    termination="transport",
                    error="provider_error: status=None",
                    retryable=True,
                ).to_dict()
            ),
        }
    ]
    fallback = _dispatch_state(rows, manifest, "full")
    assert fallback["stage"] == "fallback"
    assert fallback["per_model_concurrency"] == 4
    assert fallback["total_concurrency"] == 8
    assert fallback["operational_concurrency_override"] == override


@pytest.mark.frozen_data
def test_operational_concurrency_override_cannot_exceed_frozen_maximum() -> None:
    manifest, _ = build_manifest(ROOT, load_config(ROOT / "configs/formal_aws.json"))
    with pytest.raises(FormalRunError, match="reviewed maximum"):
        _operational_concurrency_override(manifest, "full", 9, 16)


@pytest.mark.frozen_data
def test_operational_concurrency_override_can_start_full_stage_at_reviewed_maximum() -> None:
    manifest, _ = build_manifest(ROOT, load_config(ROOT / "configs/formal_minimax.json"))
    override = _operational_concurrency_override(manifest, "full", 8, 8)
    assert override == {
        "per_model_concurrency": 8,
        "total_concurrency": 8,
        "reason": "user_authorized_runtime_adjustment",
    }


def test_environment_parser_never_executes_shell(tmp_path: Path, monkeypatch) -> None:
    config = load_config(ROOT / "configs/formal.json")
    (tmp_path / ".env").write_text(
        "NAIRR_GATEWAY_BASE_URL='https://unit.invalid/v1'\n"
        "NAIRR_GATEWAY_KEY='$(touch should-not-exist)'\n"
        "UNRELATED=ignored\n"
    )
    monkeypatch.delenv("NAIRR_GATEWAY_BASE_URL", raising=False)
    monkeypatch.setenv("NAIRR_GATEWAY_KEY", "process-wins")
    values = _environment(tmp_path, config)
    assert values["NAIRR_GATEWAY_KEY"] == "process-wins"
    assert "UNRELATED" not in values and not (tmp_path / "should-not-exist").exists()
    assert _endpoint(tmp_path, config) == ("https://unit.invalid/v1", "process-wins")


@pytest.mark.parametrize(
    "base",
    [
        "ftp://unit.invalid/v1",
        "https://user:secret@unit.invalid/v1",
        "https://unit.invalid/v1?key=secret",
        "https://unit.invalid/v1#fragment",
    ],
)
def test_unsafe_gateway_endpoint_is_rejected(tmp_path: Path, monkeypatch, base: str) -> None:
    config = load_config(ROOT / "configs/formal.json")
    monkeypatch.setenv("NAIRR_GATEWAY_BASE_URL", base)
    monkeypatch.setenv("NAIRR_GATEWAY_KEY", "test")
    with pytest.raises(FormalRunError, match="unsafe"):
        _endpoint(tmp_path, config)


def test_gateway_inspection_binds_mapping_limits_prices_and_account() -> None:
    config = load_config(ROOT / "configs/formal.json")
    gateway = object.__new__(Gateway)
    gateway.config = config
    documents = {
        "/models": ({"data": [{"id": model} for model in MODEL_ORDER]}, "1.2.3"),
        "/model/info": (
            {
                "data": [
                    {
                        "model_name": model["id"],
                        "litellm_params": {"model": model["expected_upstream_model"]},
                        "model_info": {
                            "max_input_tokens": 200000,
                            "max_output_tokens": 128000,
                            "supports_reasoning": True,
                            "input_cost_per_token": 1e-6,
                            "output_cost_per_token": 2e-6,
                        },
                    }
                    for model in config["models"]
                ]
            },
            "1.2.3",
        ),
        "/key/info": ({"info": {"spend": 10.0, "max_budget": 100.0}}, "1.2.3"),
    }
    gateway._fetch = lambda path: documents[path]  # type: ignore[method-assign]
    result = gateway.inspect()
    assert result["account"]["remaining_usd"] == 90
    assert result["gateway_versions"] == ["1.2.3"]
    assert result["pricing"]["gpt-5.6-sol"] == {
        "max_input_tokens": 200000,
        "input_per_million": 1.0,
        "output_per_million": 2.0,
    }
    assert reservation(result["pricing"]["gpt-5.6-sol"], 128000) == 456000


def test_cli_exposes_only_one_formal_workflow() -> None:
    parser = _build_parser()
    parsed = parser.parse_args(["study", "formal", "status"])
    assert parsed.study_command == "formal" and parsed.formal_command == "status"
    with pytest.raises(SystemExit):
        parser.parse_args(["study", "formal-gated", "status"])


def test_wilson_difference_and_cluster_warning() -> None:
    interval = newcombe_difference([False] * 4, [True] * 4)
    assert interval is not None and 0 < interval[0] < 1 and interval[1] == 1
    warning = stratified_cluster_interval({"stratum": {"r0": [0, 0], "r1": [0, 0]}})
    assert warning["ci95"] is None and "degenerate" in str(warning["warning"])


def test_paired_comparison_never_uses_observed_intersection() -> None:
    rows = [
        {"model": "a", "instance_id": "1", "em": True},
        {"model": "b", "instance_id": "1", "em": None},
    ]
    assert paired_comparisons(rows)[0]["reason"] == "incomplete_selection"
    rows[1]["instance_id"] = "2"
    assert paired_comparisons(rows)[0]["reason"] == "different_declared_masks"


def test_wrong_cell_statistics_disclose_conditioning(instance: Instance) -> None:
    row = score_formal_record(
        {"scoring_version": SCORING_VERSION, "answer_eligible": True, "raw_text": '{"cells":[]}'},
        instance,
        simulate(instance).gold(),
    )
    invalid = copy.deepcopy(row)
    invalid.update(valid_state=False, wrong_cells=None, answer_status="invalid")
    group = summarize_group([row, invalid], cell=True)
    assert group["wrong_cells_n"] == 1
    assert group["wrong_cells_population"] == "analysis_eligible_valid_incorrect_nonrefusal"
