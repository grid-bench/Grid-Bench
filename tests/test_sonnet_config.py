"""Frozen Sonnet sample selection and compatibility with the completed campaign."""

from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import pytest

from spatial_worlds.eval.run import condition_id
from spatial_worlds.eval.study_final_config import (
    MODEL_ORDER,
    PARENT_BINDING,
    SONNET_MODEL,
    SONNET_PILOT,
    build_manifest,
    load_config,
    model_order,
    pending_jobs,
)
from spatial_worlds.eval.study_final_store import digest, job_key

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def sonnet_manifest() -> dict:
    manifest, _ = build_manifest(ROOT, load_config(ROOT / "configs/formal_sonnet.json"))
    return manifest


@pytest.mark.frozen_data
def test_legacy_manifest_keeps_frozen_identity() -> None:
    manifest, _ = build_manifest(ROOT, load_config(ROOT / "configs/formal.json"))
    assert manifest["manifest_sha256"] == (
        "537f59bed3acbce81f81e9c230f7265fc38ec77d46f42a894c47ca15fe30aa2a"
    )
    assert manifest["selection_sha256"] == (
        "b50789debdae01c8e3ed5d0d32759c4e13fa1e7bcb8b0922d1eeb214d0796ca4"
    )
    assert model_order(manifest) == MODEL_ORDER
    assert model_order({}) == MODEL_ORDER
    assert "pilot" not in manifest


@pytest.mark.frozen_data
def test_sonnet_reuses_exact_instances_but_new_request_identity(sonnet_manifest: dict) -> None:
    parent = json.loads((ROOT / PARENT_BINDING).read_text())
    original = [job for job in parent["jobs"] if job["model"] == MODEL_ORDER[0]]
    jobs = sonnet_manifest["jobs"]
    assert len(jobs) == 744
    assert Counter(job["dataset"] for job in jobs) == {"main": 576, "rq2": 168}
    assert model_order(sonnet_manifest) == ("claude-sonnet",)
    assert len({job_key(job) for job in jobs}) == 744
    for job, source in zip(jobs, original, strict=True):
        for field in (
            "dataset",
            "stage",
            "instance_id",
            "T",
            "prompt_sha256",
            "instance_sha256",
            "gold_sha256",
        ):
            assert job[field] == source[field]
        assert job["request"]["expected_model"] == SONNET_MODEL["accepted_response_models"]
        assert job["request"]["parameters"] == {
            "max_completion_tokens": 128000,
            "reasoning_effort": "high",
        }
        assert job["request_sha256"] == digest(job["request"])
        assert job["condition_id"] == condition_id(
            provider="openai",
            model="claude-sonnet",
            representation="cells",
            prompt_version=sonnet_manifest["prompt_version"],
            request=job["request"],
        )
        assert job["condition_id"] != source["condition_id"]
        assert job_key(job) != job_key(source)
    assert sonnet_manifest["dataset_manifests"] == parent["dataset_manifests"]
    assert sonnet_manifest["workload_reuse_ids"] == parent["workload_reuse_ids"]
    assert sonnet_manifest["selection_sha256"] != parent["selection_sha256"]
    assert sonnet_manifest["manifest_sha256"] == digest(
        {key: value for key, value in sonnet_manifest.items() if key != "manifest_sha256"}
    )


@pytest.mark.frozen_data
def test_pilot_is_ten_ordered_unique_formal_jobs(sonnet_manifest: dict) -> None:
    assert sonnet_manifest["pilot"] == SONNET_PILOT
    by_id = {(job["dataset"], job["instance_id"]): job for job in sonnet_manifest["jobs"]}
    selected = [by_id[(job["dataset"], job["instance_id"])] for job in SONNET_PILOT["jobs"]]
    assert len({job_key(job) for job in selected}) == 10
    assert [job["T"] for job in selected] == [0, 1, 1, 8, 16, 16, 32, 64, 64, 64]
    assert len({job["instance_id"].split("/")[0] for job in selected}) == 8
    assert {job["instance_id"].split("/")[2] for job in selected} == {"d2", "d3", "d4"}
    assert all("/test/0000/" in job["instance_id"] for job in selected)


@pytest.mark.frozen_data
def test_sonnet_has_no_imported_development_outcomes(sonnet_manifest: dict) -> None:
    assert sonnet_manifest["lineage"] == []
    assert sonnet_manifest["lineage_outcomes"] == {"claude-sonnet": 0}
    assert sonnet_manifest["remaining_counts"] == {"claude-sonnet": 744}
    assert len(pending_jobs(sonnet_manifest, set())["claude-sonnet"]) == 744
    assert sonnet_manifest["formal_budget_usd"] == 800
    assert sonnet_manifest["credit_buffer_usd"] == 25
    assert sonnet_manifest["dispatch"]["initial_total_concurrency"] == 4
    assert sonnet_manifest["dispatch"]["promoted_total_concurrency"] == 8


@pytest.mark.parametrize("mutation", ["effort", "alias", "budget", "pilot_order", "lineage"])
def test_sonnet_config_rejects_unreviewed_changes(tmp_path: Path, mutation: str) -> None:
    config = copy.deepcopy(load_config(ROOT / "configs/formal_sonnet.json"))
    if mutation == "effort":
        config["models"][0]["effort"] = "max"
    elif mutation == "alias":
        config["models"][0]["accepted_response_models"].append("other-model")
    elif mutation == "budget":
        config["formal_budget_usd"] = 801
    elif mutation == "pilot_order":
        config["pilot"]["jobs"].reverse()
    else:
        config["lineage"] = [{"campaign_id": "historical-calibration"}]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_config(path)
