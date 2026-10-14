"""Native Bedrock formal configuration and stream translation, without API calls."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from spatial_worlds.eval.protocol import consume_attempt
from spatial_worlds.eval.run import condition_id
from spatial_worlds.eval.study_final import _pilot_jobs, is_model_outcome
from spatial_worlds.eval.study_final_bedrock import bedrock_events
from spatial_worlds.eval.study_final_config import (
    AWS_MODELS,
    AWS_PILOT,
    MINIMAX_MODEL,
    MINIMAX_PILOT,
    build_manifest,
    load_config,
    model_order,
)
from spatial_worlds.eval.study_final_store import digest, job_key

ROOT = Path(__file__).resolve().parents[1]


class FakeStream:
    def __init__(self, events):
        self.events = events
        self.closed = False

    def __iter__(self):
        return iter(self.events)

    def close(self):
        self.closed = True


class FakeClient:
    def __init__(self, events):
        self.stream = FakeStream(events)
        self.request = None

    def converse_stream(self, **kwargs):
        self.request = kwargs
        return {
            "ResponseMetadata": {"RequestId": "12345678-abcd-4321-a000-123456789abc"},
            "stream": self.stream,
        }


def test_bedrock_stream_separates_reasoning_and_answer() -> None:
    client = FakeClient(
        [
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"reasoningContent": {"text": "private chain"}},
                }
            },
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 1,
                    "delta": {"text": '{"cells":[]}'},
                }
            },
            {"messageStop": {"stopReason": "end_turn"}},
            {
                "metadata": {
                    "usage": {"inputTokens": 101, "outputTokens": 202, "totalTokens": 303}
                }
            },
        ]
    )
    params = {"max_tokens": 128000, "reasoning_effort": "high"}
    result = consume_attempt(
        lambda: bedrock_events(client, "zai.glm-5", "prompt", params, trace=True),
        trace_enabled=True,
    )
    assert client.request == {
        "modelId": "zai.glm-5",
        "messages": [{"role": "user", "content": [{"text": "prompt"}]}],
        "inferenceConfig": {"maxTokens": 128000},
        "additionalModelRequestFields": {"reasoning_effort": "high"},
    }
    assert result.text == '{"cells":[]}'
    assert "private chain" not in json.dumps(result.to_dict())
    assert result.finish_reason == "end_turn" and result.termination == "stop"
    assert result.response == {
        "id": "req_12345678-abcd-4321-a000-123456789abc",
        "model": "zai.glm-5",
    }
    assert result.usage["input_tokens"] == 101
    assert result.usage["output_tokens"] == 202
    assert result.usage_complete
    assert result.trace and result.trace["reasoning_present"]
    assert result.trace["reasoning_characters"] == len("private chain")
    job = {
        "request": {"expected_model": ["zai.glm-5"]},
    }
    assert is_model_outcome(result, job)


@pytest.mark.frozen_data
def test_aws_manifest_reuses_the_frozen_mask_with_new_conditions() -> None:
    config = load_config(ROOT / "configs/formal_aws.json")
    manifest, _ = build_manifest(ROOT, config)
    models = tuple(model["id"] for model in AWS_MODELS)
    assert model_order(manifest) == models
    assert Counter(job["model"] for job in manifest["jobs"]) == {
        "zai.glm-5": 744,
        "moonshotai.kimi-k2.5": 744,
    }
    assert len({job_key(job) for job in manifest["jobs"]}) == 1488
    assert manifest["lineage_outcomes"] == {model: 0 for model in models}
    assert manifest["remaining_counts"] == {model: 744 for model in models}
    for job in manifest["jobs"]:
        assert job["provider"] == "bedrock"
        assert job["request"]["parameters"] == {
            "max_tokens": 128000,
            "reasoning_effort": "high",
        }
        assert job["request_sha256"] == digest(job["request"])
        assert job["condition_id"] == condition_id(
            provider="bedrock",
            model=job["model"],
            representation="cells",
            prompt_version=manifest["prompt_version"],
            request=job["request"],
        )


@pytest.mark.frozen_data
def test_aws_pilot_selects_six_distinct_model_jobs() -> None:
    manifest, _ = build_manifest(ROOT, load_config(ROOT / "configs/formal_aws.json"))
    selected = _pilot_jobs(manifest)
    assert manifest["pilot"] == AWS_PILOT
    assert len(selected) == len({job_key(job) for job in selected}) == 6
    assert [job["T"] for job in selected] == [0, 8, 64, 0, 8, 64]
    assert [job["model"] for job in selected] == [
        "zai.glm-5",
        "zai.glm-5",
        "zai.glm-5",
        "moonshotai.kimi-k2.5",
        "moonshotai.kimi-k2.5",
        "moonshotai.kimi-k2.5",
    ]


@pytest.mark.frozen_data
def test_minimax_manifest_uses_native_thinking_without_effort_field() -> None:
    config = load_config(ROOT / "configs/formal_minimax.json")
    manifest, _ = build_manifest(ROOT, config)
    assert model_order(manifest) == (MINIMAX_MODEL["id"],)
    assert Counter(job["model"] for job in manifest["jobs"]) == {
        MINIMAX_MODEL["id"]: 744
    }
    assert len({job_key(job) for job in manifest["jobs"]}) == 744
    assert manifest["lineage_outcomes"] == {MINIMAX_MODEL["id"]: 0}
    assert manifest["remaining_counts"] == {MINIMAX_MODEL["id"]: 744}
    for job in manifest["jobs"]:
        assert job["provider"] == "bedrock"
        assert job["request"]["parameters"] == {"max_tokens": 128000}
        assert "reasoning_effort" not in job["request"]["parameters"]


@pytest.mark.frozen_data
def test_minimax_pilot_selects_four_unprobed_formal_jobs() -> None:
    manifest, _ = build_manifest(ROOT, load_config(ROOT / "configs/formal_minimax.json"))
    selected = _pilot_jobs(manifest)
    assert manifest["pilot"] == MINIMAX_PILOT
    assert len(selected) == len({job_key(job) for job in selected}) == 4
    assert [job["T"] for job in selected] == [0, 1, 8, 64]
    assert {job["model"] for job in selected} == {MINIMAX_MODEL["id"]}


def test_minimax_bedrock_request_omits_additional_effort_fields() -> None:
    client = FakeClient(
        [
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "{}"}}},
            {"messageStop": {"stopReason": "end_turn"}},
            {
                "metadata": {
                    "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}
                }
            },
        ]
    )
    list(
        bedrock_events(
            client,
            "minimax.minimax-m2.5",
            "prompt",
            {"max_tokens": 128000},
        )
    )
    assert client.request == {
        "modelId": "minimax.minimax-m2.5",
        "messages": [{"role": "user", "content": [{"text": "prompt"}]}],
        "inferenceConfig": {"maxTokens": 128000},
    }
