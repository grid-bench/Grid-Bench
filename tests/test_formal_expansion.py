"""The second four-trajectory cohort is fixed before any paid collection."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from spatial_worlds.eval.study_final_config import build_manifest, load_config
from spatial_worlds.eval.study_final_store import job_key

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.frozen_data
@pytest.mark.parametrize(
    "source_file,expansion_file",
    [
        ("formal.json", "formal_expand08.json"),
        ("formal_sonnet.json", "formal_sonnet_expand08.json"),
        ("formal_aws.json", "formal_aws_expand08.json"),
        ("formal_minimax.json", "formal_minimax_expand08.json"),
    ],
)
def test_expansion_is_disjoint_and_preserves_model_conditions(
    source_file: str, expansion_file: str
) -> None:
    source, _ = build_manifest(ROOT, load_config(ROOT / "configs" / source_file))
    expansion, datasets = build_manifest(ROOT, load_config(ROOT / "configs" / expansion_file))
    models = tuple(model["id"] for model in source["models"])
    assert expansion["source_manifest_sha256"] == source["manifest_sha256"]
    assert Counter(job["model"] for job in expansion["jobs"]) == {
        model: 744 for model in models
    }
    assert len({job_key(job) for job in expansion["jobs"]}) == 744 * len(models)
    assert not {job_key(job) for job in expansion["jobs"]} & {
        job_key(job) for job in source["jobs"]
    }
    assert expansion["lineage"] == []
    assert expansion["lineage_outcomes"] == {model: 0 for model in models}
    assert expansion["remaining_counts"] == {model: 744 for model in models}
    assert len(expansion["workload_reuse_ids"]) == 96
    assert all(
        any(f"/test/{index:04d}/" in job["instance_id"] for index in range(4, 8))
        for job in expansion["jobs"]
    )
    assert all(
        job["request"] == old["request"] and job["condition_id"] == old["condition_id"]
        for job, old in zip(expansion["jobs"], source["jobs"], strict=True)
    )
    reuse = set(expansion["workload_reuse_ids"])
    assert reuse <= {job["instance_id"] for job in expansion["jobs"] if job["dataset"] == "main"}
    assert not reuse & {job["instance_id"] for job in expansion["jobs"] if job["dataset"] == "rq2"}
    assert set(datasets) == {"main", "rq2"}


@pytest.mark.frozen_data
def test_all_models_share_the_same_new_instances() -> None:
    manifests = [
        build_manifest(ROOT, load_config(ROOT / "configs" / name))[0]
        for name in (
            "formal_expand08.json",
            "formal_sonnet_expand08.json",
            "formal_aws_expand08.json",
            "formal_minimax_expand08.json",
        )
    ]
    selections = [
        {(job["dataset"], job["instance_id"]) for job in manifest["jobs"]}
        for manifest in manifests
    ]
    assert len(selections[0]) == 744
    assert all(selection == selections[0] for selection in selections[1:])
    assert len({instance_id.rsplit("/t", 1)[0] for _, instance_id in selections[0]}) == 124
