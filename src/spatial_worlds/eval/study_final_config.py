"""Configuration and immutable selection for the reviewed formal campaigns."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from ..prompt import COMPLETION_PROMPT_VERSION, render_prompt
from .protocol import FORMAL128_PROTOCOL
from .run import condition_id, load_dataset
from .study_final_store import FormalRunError, digest, job_key

CONFIG_PATH = Path("configs/formal.json")
PARENT_BINDING = Path("data/study-v1/formal-128k-high-sol-mini-01/binding.json")
LEDGER = Path("data/study-v1/attempts.sqlite")
FULL_PER_MODEL = 744
MODEL_ORDER = ("gpt-5.6-sol", "gpt-5.4-mini")
SONNET_CAMPAIGN = "formal-128k-high-sonnet-01"
SONNET_MODEL = {
    "id": "claude-sonnet",
    "provider": "openai",
    "route": "gateway",
    "effort": "high",
    "expected_upstream_model": "vertex_ai/claude-sonnet-5",
    "accepted_response_models": [
        "claude-sonnet",
        "claude-sonnet-5",
        "vertex_ai/claude-sonnet-5",
    ],
}
SONNET_PILOT = {
    "concurrency": 1,
    "jobs": [
        {"dataset": "main", "instance_id": "aggregating_compression/M/d2/w256/test/0000/t000"},
        {"dataset": "main", "instance_id": "box_reflection/M/d2/w64/test/0000/t001"},
        {"dataset": "main", "instance_id": "component_relabel/M/d2/w100/test/0000/t001"},
        {"dataset": "main", "instance_id": "local_stencil/M/d4/w10/test/0000/t008"},
        {"dataset": "main", "instance_id": "metric_paint/M/d3/w100/test/0000/t016"},
        {"dataset": "rq2", "instance_id": "box_reflection/M/d2/w16/test/0000/t016"},
        {"dataset": "main", "instance_id": "boundary_layer/M/d4/w60/test/0000/t032"},
        {"dataset": "main", "instance_id": "obstructed_motion/M/d4/w30/test/0000/t064"},
        {"dataset": "main", "instance_id": "occluded_mix/M/d2/w12/test/0000/t064"},
        {"dataset": "rq2", "instance_id": "occluded_mix/M/d2/w30/test/0000/t064"},
    ],
}
AWS_CAMPAIGN = "formal-128k-high-glm-kimi-01"
AWS_MODELS = [
    {
        "id": "zai.glm-5",
        "provider": "bedrock",
        "route": "bedrock",
        "effort": "high",
        "accepted_response_models": ["zai.glm-5"],
        "max_input_tokens": 202752,
        "input_per_million": 1.0,
        "output_per_million": 3.2,
    },
    {
        "id": "moonshotai.kimi-k2.5",
        "provider": "bedrock",
        "route": "bedrock",
        "effort": "high",
        "accepted_response_models": ["moonshotai.kimi-k2.5"],
        "max_input_tokens": 262144,
        "input_per_million": 0.6,
        "output_per_million": 3.0,
    },
]
AWS_PILOT = {
    "concurrency": 1,
    "jobs": [
        {
            "model": model,
            "dataset": dataset,
            "instance_id": instance_id,
        }
        for model in ("zai.glm-5", "moonshotai.kimi-k2.5")
        for dataset, instance_id in (
            ("main", "aggregating_compression/M/d2/w256/test/0000/t000"),
            ("main", "local_stencil/M/d4/w10/test/0000/t008"),
            ("main", "occluded_mix/M/d2/w12/test/0000/t064"),
        )
    ],
}
MINIMAX_CAMPAIGN = "formal-128k-native-minimax-01"
MINIMAX_MODEL = {
    "id": "minimax.minimax-m2.5",
    "provider": "bedrock",
    "route": "bedrock",
    "effort": "native_always_on",
    "accepted_response_models": ["minimax.minimax-m2.5"],
    "max_input_tokens": 200704,
    "input_per_million": 0.3,
    "output_per_million": 1.2,
}
MINIMAX_PILOT = {
    "concurrency": 1,
    "jobs": [
        {
            "model": "minimax.minimax-m2.5",
            "dataset": "main",
            "instance_id": "component_relabel/M/d3/w100/test/0003/t000",
        },
        {
            "model": "minimax.minimax-m2.5",
            "dataset": "main",
            "instance_id": "boundary_layer/M/d4/w60/test/0003/t001",
        },
        {
            "model": "minimax.minimax-m2.5",
            "dataset": "main",
            "instance_id": "metric_paint/M/d3/w100/test/0003/t008",
        },
        {
            "model": "minimax.minimax-m2.5",
            "dataset": "main",
            "instance_id": "box_reflection/M/d2/w64/test/0003/t064",
        },
    ],
}
FINAL_SCOPE_PREFIX = "formal128:"
LINEAGE_FIELDS = (
    "id",
    "scope",
    "job_key",
    "model",
    "number",
    "started",
    "reservation",
    "accounted",
    "known_upper",
    "job",
    "result",
)

EXPECTED_LINEAGE = [
    {
        "campaign_id": "formal-128k-high-sol-mini-01",
        "scope": "formal128:formal-128k-high-sol-mini-01",
        "attempts": 48,
        "outcomes": {"gpt-5.6-sol": 48, "gpt-5.4-mini": 0},
        "sha256": "cede8af0e1f61ba716cd83151b08065381cd1557e6fc8472b31cc82362063ba6",
    },
    {
        "campaign_id": "formal-128k-high-mini-canary-03",
        "scope": "formal128:formal-128k-high-mini-canary-03",
        "attempts": 20,
        "outcomes": {"gpt-5.6-sol": 0, "gpt-5.4-mini": 20},
        "sha256": "33129df0d05cb2dc12072c46ea3a1616029530283a9c7e5cf5a75a51bb1e3e3e",
    },
    {
        "campaign_id": "formal-128k-high-sol-mini-final-05",
        "scope": "formal128:formal-128k-high-sol-mini-final-05",
        "attempts": 352,
        "outcomes": {"gpt-5.6-sol": 175, "gpt-5.4-mini": 176},
        "sha256": "a20d3d0694714914ba0be990bc153c61760874dfbeb242e11336a5188c94a8fb",
    },
]

EXPANSION_SOURCES = {
    "formal-128k-high-sol-mini-expand08-01": ("formal.json", "formal-128k-high-sol-mini-final-06"),
    "formal-128k-high-sonnet-expand08-01": ("formal_sonnet.json", SONNET_CAMPAIGN),
    "formal-128k-high-glm-kimi-expand08-01": ("formal_aws.json", AWS_CAMPAIGN),
    "formal-128k-native-minimax-expand08-01": ("formal_minimax.json", MINIMAX_CAMPAIGN),
}
EXPANSION_INDEX = re.compile(r"/test/(000[0-3])(?=/t(?:000|001|008|016|032|064)$)")


def load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if "expansion_of" in value:
        source = EXPANSION_SOURCES.get(value.get("campaign_id"))
        if source is None or value.get("expansion_of") != source[1]:
            raise ValueError("unknown formal eight-trajectory expansion")
        group = (
            "aws-bedrock-expand08"
            if source[1] in {AWS_CAMPAIGN, MINIMAX_CAMPAIGN}
            else "nairr-expand08"
        )
        buffer = 95 if group.startswith("aws") else 0
        if set(value) != {"campaign_id", "expansion_of", "shared_account"} or value[
            "shared_account"
        ] != {"group": group, "buffer_usd": buffer}:
            raise ValueError("eight-trajectory expansion changes the reviewed account policy")
        original = load_config(path.parent / source[0])
        original["campaign_id"] = value["campaign_id"]
        original["expansion_of"] = source[1]
        original["shared_account"] = value["shared_account"]
        original["lineage"] = []
        original.pop("pilot", None)
        original["credit_buffer_usd"] = buffer
        if group.startswith("aws"):
            # All completed AWS exploratory and formal usage predates this
            # expansion; both concurrent campaigns must start at one baseline.
            original["routes"]["bedrock"]["known_prior_spend_usd"] = 171.795913
        return original
    if value.get("campaign_id") == MINIMAX_CAMPAIGN:
        expected = {
            "campaign_id": MINIMAX_CAMPAIGN,
            "collection_protocol": "spatial-worlds-evaluation/2.2",
            "evaluation_protocol": FORMAL128_PROTOCOL.version,
            "scoring_version": "spatial-worlds-scoring/1.0",
            "prompt_version": COMPLETION_PROMPT_VERSION,
            "credit_buffer_usd": 95,
            "formal_budget_usd": 150,
            "metadata_refresh_s": 7200,
        }
        if any(value.get(key) != item for key, item in expected.items()):
            raise ValueError("MiniMax formal config changes the reviewed protocol")
        if value.get("resources") != {
            "max_tokens": 128000,
            "deadline_s": 7200,
            "usage_grace_s": 5,
            "technical_attempts_per_run": 3,
        }:
            raise ValueError("MiniMax formal resources differ from the reviewed protocol")
        if value.get("dispatch") != {
            "initial_per_model_concurrency": 4,
            "initial_total_concurrency": 4,
            "promotion_outcomes_per_model": 8,
            "promoted_per_model_concurrency": 8,
            "promoted_total_concurrency": 8,
            "account_refresh_s": 300,
        }:
            raise ValueError("MiniMax formal dispatch differs from the reviewed policy")
        if value.get("lineage") != [] or value.get("models") != [MINIMAX_MODEL]:
            raise ValueError("MiniMax formal model identity or lineage changed")
        if value.get("pilot") != MINIMAX_PILOT:
            raise ValueError("MiniMax formal pilot differs from the frozen sequence")
        if value.get("routes") != {
            "bedrock": {
                "region": "us-east-1",
                "bearer_token_env": "AWS_BEARER_TOKEN_BEDROCK",
                "monthly_spend_stop_usd": 500,
                "known_prior_spend_usd": 141.0,
                "budget_reset_at": "2026-10-01T00:00:00Z",
            }
        }:
            raise ValueError("MiniMax formal route or local spend controls changed")
        return value
    if value.get("campaign_id") == AWS_CAMPAIGN:
        expected = {
            "campaign_id": AWS_CAMPAIGN,
            "collection_protocol": "spatial-worlds-evaluation/2.1",
            "evaluation_protocol": FORMAL128_PROTOCOL.version,
            "scoring_version": "spatial-worlds-scoring/1.0",
            "prompt_version": COMPLETION_PROMPT_VERSION,
            "credit_buffer_usd": 95,
            "formal_budget_usd": 400,
            "metadata_refresh_s": 7200,
        }
        if any(value.get(key) != item for key, item in expected.items()):
            raise ValueError("AWS formal config changes the reviewed protocol")
        if value.get("resources") != {
            "max_tokens": 128000,
            "deadline_s": 7200,
            "usage_grace_s": 5,
            "technical_attempts_per_run": 3,
        }:
            raise ValueError("AWS formal resources differ from the reviewed protocol")
        if value.get("dispatch") != {
            "initial_per_model_concurrency": 2,
            "initial_total_concurrency": 4,
            "promotion_outcomes_per_model": 8,
            "promoted_per_model_concurrency": 8,
            "promoted_total_concurrency": 16,
            "account_refresh_s": 300,
        }:
            raise ValueError("AWS formal dispatch differs from the reviewed policy")
        if value.get("lineage") != [] or value.get("models") != AWS_MODELS:
            raise ValueError("AWS formal model identities or lineage changed")
        if value.get("pilot") != AWS_PILOT:
            raise ValueError("AWS formal pilot differs from the frozen sequence")
        if value.get("routes") != {
            "bedrock": {
                "region": "us-east-1",
                "bearer_token_env": "AWS_BEARER_TOKEN_BEDROCK",
                "monthly_spend_stop_usd": 500,
                "known_prior_spend_usd": 0.384,
                "budget_reset_at": "2026-10-01T00:00:00Z",
            }
        }:
            raise ValueError("AWS formal route or local spend controls changed")
        return value
    sonnet = value.get("campaign_id") == SONNET_CAMPAIGN
    expected = {
        "campaign_id": SONNET_CAMPAIGN if sonnet else "formal-128k-high-sol-mini-final-06",
        "collection_protocol": (
            "spatial-worlds-evaluation/1.9" if sonnet else "spatial-worlds-evaluation/1.8"
        ),
        "evaluation_protocol": FORMAL128_PROTOCOL.version,
        "scoring_version": "spatial-worlds-scoring/1.0",
        "prompt_version": COMPLETION_PROMPT_VERSION,
        "credit_buffer_usd": 25,
        "formal_budget_usd": 800 if sonnet else 1300,
        "metadata_refresh_s": 7200,
        "allow_http_endpoint": True,
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise ValueError("formal config changes the reviewed final protocol")
    if value.get("resources") != {
        "max_tokens": 128000,
        "deadline_s": 7200,
        "usage_grace_s": 5,
        "technical_attempts_per_run": 3,
    }:
        raise ValueError("formal resources differ from the reviewed final protocol")
    if value.get("dispatch") != {
        "initial_per_model_concurrency": 4,
        "initial_total_concurrency": 4 if sonnet else 8,
        "promotion_outcomes_per_model": 8,
        "promoted_per_model_concurrency": 8,
        "promoted_total_concurrency": 8 if sonnet else 16,
        "account_refresh_s": 300,
    }:
        raise ValueError("formal dispatch differs from the reviewed progressive policy")
    if value.get("lineage") != ([] if sonnet else EXPECTED_LINEAGE):
        raise ValueError("formal lineage differs from the frozen source snapshots")
    if sonnet:
        if value.get("models") != [SONNET_MODEL]:
            raise ValueError("Sonnet model identity differs from the reviewed gateway route")
        if value.get("pilot") != SONNET_PILOT:
            raise ValueError("Sonnet pilot differs from the frozen ten-row sequence")
    elif [model.get("id") for model in value.get("models", [])] != list(MODEL_ORDER):
        raise ValueError("formal model order changed")
    if set(value.get("routes", {})) != {"gateway"}:
        raise ValueError("formal runner requires the single gateway route")
    return value


def model_order(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    """Read campaign models; retain legacy defaults for older small test manifests."""
    models = manifest.get("models")
    return tuple(str(model["id"]) for model in models) if models is not None else MODEL_ORDER


def paths(root: Path, config: dict[str, Any]) -> tuple[Path, Path]:
    out = root / "data/study-v1" / config["campaign_id"]
    ledger = root / LEDGER
    if any(path.is_symlink() for path in (root / "data", out.parent, out, ledger)):
        raise FormalRunError("refusing symlinked formal data paths")
    return out, ledger


def request_parameters(model: Mapping[str, Any]) -> dict[str, Any]:
    """Return the frozen model-visible generation parameters for one route."""
    if model["provider"] != "bedrock":
        return {"max_completion_tokens": 128000, "reasoning_effort": "high"}
    parameters: dict[str, Any] = {"max_tokens": 128000}
    effort = model.get("effort")
    if effort == "high":
        parameters["reasoning_effort"] = "high"
    elif effort != "native_always_on":
        raise FormalRunError("unsupported Bedrock reasoning-effort condition")
    return parameters


def _validate_job(job: dict[str, Any], model: dict[str, Any]) -> None:
    request = job.get("request") or {}
    provider = model["provider"]
    parameters = request_parameters(model)
    if (
        job.get("model") != model["id"]
        or job.get("provider") != provider
        or job.get("dataset") not in {"main", "rq2"}
        or job.get("stage") != job.get("dataset")
        or job.get("T") not in {0, 1, 8, 16, 32, 64}
        or request.get("parameters") != parameters
        or request.get("expected_model") != model["accepted_response_models"]
        or request.get("timeout_s") != 7200
        or request.get("usage_grace_s") != 5.0
        or request.get("n") != 1
        or request.get("evaluation_protocol") != FORMAL128_PROTOCOL.version
        or request.get("scoring_version") != "spatial-worlds-scoring/1.0"
    ):
        raise FormalRunError("a formal request differs from its frozen 128K condition")


def build_manifest(root: Path, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if config.get("expansion_of"):
        return _build_expansion_manifest(root, config)
    parent_path = root / PARENT_BINDING
    parent = json.loads(parent_path.read_text())
    if digest(parent) != config["parent_binding_sha256"]:
        raise FormalRunError("the original fixed formal binding changed")
    datasets = {name: load_dataset(root / "data" / name) for name in ("main", "rq2")}
    instances = {
        name: {instance.id: instance for instance in dataset.instances}
        for name, dataset in datasets.items()
    }
    jobs = parent.get("jobs", [])
    if len(jobs) != FULL_PER_MODEL * len(MODEL_ORDER):
        raise FormalRunError("the original formal mask is not 744 rows per model")
    sonnet = config["campaign_id"] == SONNET_CAMPAIGN
    bedrock = config["campaign_id"] in {AWS_CAMPAIGN, MINIMAX_CAMPAIGN}
    if sonnet:
        model = config["models"][0]
        jobs = deepcopy([job for job in jobs if job.get("model") == MODEL_ORDER[0]])
        for job in jobs:
            job["model"] = model["id"]
            job["provider"] = model["provider"]
            job["request"]["expected_model"] = deepcopy(model["accepted_response_models"])
            job["request_sha256"] = digest(job["request"])
            job["condition_id"] = condition_id(
                provider=job["provider"],
                model=job["model"],
                representation="cells",
                prompt_version=config["prompt_version"],
                request=job["request"],
            )
    elif bedrock:
        source = deepcopy([job for job in jobs if job.get("model") == MODEL_ORDER[0]])
        jobs = []
        base_url = f"https://bedrock-runtime.{config['routes']['bedrock']['region']}.amazonaws.com"
        for model in config["models"]:
            selected = deepcopy(source)
            for job in selected:
                job["model"] = model["id"]
                job["provider"] = model["provider"]
                job["request"] = {
                    "parameters": request_parameters(model),
                    "base_url": base_url,
                    "expected_model": deepcopy(model["accepted_response_models"]),
                    "timeout_s": 7200,
                    "max_attempts": 1,
                    "usage_grace_s": 5.0,
                    "n": 1,
                    "evaluation_protocol": FORMAL128_PROTOCOL.version,
                    "scoring_version": "spatial-worlds-scoring/1.0",
                }
                job["request_sha256"] = digest(job["request"])
                job["condition_id"] = condition_id(
                    provider=job["provider"],
                    model=job["model"],
                    representation="cells",
                    prompt_version=config["prompt_version"],
                    request=job["request"],
                )
            jobs.extend(selected)
    models = model_order(config)
    for model_name in models:
        model = next(item for item in config["models"] if item["id"] == model_name)
        selected = [job for job in jobs if job.get("model") == model_name]
        if len(selected) != FULL_PER_MODEL:
            raise FormalRunError(f"{model_name} does not have 744 frozen jobs")
        for job in selected:
            _validate_job(job, model)
            try:
                instance = instances[job["dataset"]][job["instance_id"]]
                gold = datasets[job["dataset"]].gold[job["instance_id"]]
            except KeyError as error:
                raise FormalRunError("formal data no longer contains a frozen job") from error
            expected = {
                "prompt_sha256": hashlib.sha256(
                    render_prompt(instance, prompt_version=COMPLETION_PROMPT_VERSION).encode()
                ).hexdigest(),
                "instance_sha256": digest(instance.to_dict()),
                "gold_sha256": digest(gold.to_dict()),
            }
            if any(job.get(key) != item for key, item in expected.items()):
                raise FormalRunError("prompt, instance or gold changed for the formal mask")
    lineage_outcomes = {
        model: sum(int(source["outcomes"][model]) for source in config["lineage"])
        for model in models
    }
    expected_outcomes = (
        {model: 0 for model in models}
        if bedrock
        else {"claude-sonnet": 0}
        if sonnet
        else {"gpt-5.6-sol": 223, "gpt-5.4-mini": 196}
    )
    if lineage_outcomes != expected_outcomes:
        raise FormalRunError("formal lineage outcome counts changed")
    manifest = {
        "campaign_id": config["campaign_id"],
        "phase": f"{FINAL_SCOPE_PREFIX}{config['campaign_id']}",
        "collection_protocol": config["collection_protocol"],
        "evaluation_protocol": config["evaluation_protocol"],
        "scoring_version": config["scoring_version"],
        "prompt_version": config["prompt_version"],
        "resources": config["resources"],
        "dispatch": config["dispatch"],
        "formal_budget_usd": config["formal_budget_usd"],
        "credit_buffer_usd": config["credit_buffer_usd"],
        "metadata_refresh_s": config["metadata_refresh_s"],
        "models": config["models"],
        "lineage": config["lineage"],
        "lineage_outcomes": lineage_outcomes,
        "dataset_manifests": parent["dataset_manifests"],
        "workload_reuse_ids": parent["workload_reuse_ids"],
        "jobs": jobs,
        "remaining_counts": {model: FULL_PER_MODEL - lineage_outcomes[model] for model in models},
        "config_sha256": digest(config),
        "parent_binding_sha256": config["parent_binding_sha256"],
    }
    if sonnet or bedrock:
        manifest["pilot"] = deepcopy(config["pilot"])
        selected = {(job["model"], job["dataset"], job["instance_id"]) for job in jobs}
        pilot = {
            (
                item.get("model", models[0]),
                item["dataset"],
                item["instance_id"],
            )
            for item in manifest["pilot"]["jobs"]
        }
        if len(selected) != FULL_PER_MODEL * len(models) or not pilot <= selected:
            raise FormalRunError("formal pilot is not a subset of the unique model mask")
    manifest["selection_sha256"] = digest(
        [
            [
                job["model"],
                job["instance_id"],
                job["prompt_sha256"],
                job["instance_sha256"],
                job["gold_sha256"],
            ]
            for job in jobs
        ]
    )
    manifest["manifest_sha256"] = digest(manifest)
    return manifest, datasets


def _expansion_id(instance_id: str) -> str:
    match = EXPANSION_INDEX.search(instance_id)
    if match is None:
        raise FormalRunError("source formal job is outside the four-trajectory cohort")
    next_index = int(match.group(1)) + 4
    return instance_id[: match.start(1)] + f"{next_index:04d}" + instance_id[match.end(1) :]


def _build_expansion_manifest(
    root: Path, config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_file, source_id = EXPANSION_SOURCES[config["campaign_id"]]
    source_config = load_config(root / "configs" / source_file)
    source, datasets = build_manifest(root, source_config)
    if source["campaign_id"] != source_id:
        raise FormalRunError("the four-trajectory source campaign changed")
    instances = {
        name: {instance.id: instance for instance in dataset.instances}
        for name, dataset in datasets.items()
    }
    jobs = []
    for old in source["jobs"]:
        job = deepcopy(old)
        job["instance_id"] = _expansion_id(old["instance_id"])
        try:
            instance = instances[job["dataset"]][job["instance_id"]]
            gold = datasets[job["dataset"]].gold[job["instance_id"]]
        except KeyError as error:
            raise FormalRunError("the selected expansion instance is missing") from error
        if instance.horizon != job["T"]:
            raise FormalRunError("expansion changed a selected length")
        job["prompt_sha256"] = hashlib.sha256(
            render_prompt(instance, prompt_version=config["prompt_version"]).encode()
        ).hexdigest()
        job["instance_sha256"] = digest(instance.to_dict())
        job["gold_sha256"] = digest(gold.to_dict())
        model = next(item for item in config["models"] if item["id"] == job["model"])
        _validate_job(job, model)
        jobs.append(job)
    models = model_order(config)
    if (
        len(jobs) != FULL_PER_MODEL * len(models)
        or len({job_key(job) for job in jobs}) != len(jobs)
        or {job["instance_id"] for job in jobs}
        & {job["instance_id"] for job in source["jobs"]}
    ):
        raise FormalRunError("eight-trajectory expansion is not a disjoint 744-row mask")
    manifest = deepcopy(source)
    manifest.pop("pilot", None)
    manifest.update(
        {
            "campaign_id": config["campaign_id"],
            "phase": f"{FINAL_SCOPE_PREFIX}{config['campaign_id']}",
            "expansion_of": source_id,
            "source_manifest_sha256": source["manifest_sha256"],
            "lineage": [],
            "lineage_outcomes": {model: 0 for model in models},
            "remaining_counts": {model: FULL_PER_MODEL for model in models},
            "jobs": jobs,
            "workload_reuse_ids": [_expansion_id(value) for value in source["workload_reuse_ids"]],
            "formal_budget_usd": config["formal_budget_usd"],
            "credit_buffer_usd": config["credit_buffer_usd"],
            "shared_account": config["shared_account"],
            "config_sha256": digest(config),
        }
    )
    if len(manifest["workload_reuse_ids"]) != 96:
        raise FormalRunError("the expansion changed the workload reuse mask")
    manifest["selection_sha256"] = digest(
        [
            [
                job["model"],
                job["instance_id"],
                job["prompt_sha256"],
                job["instance_sha256"],
                job["gold_sha256"],
            ]
            for job in jobs
        ]
    )
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = digest(manifest)
    return manifest, datasets


def lineage_scopes(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(source["scope"]) for source in manifest["lineage"])


def formal_scopes(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    return (*lineage_scopes(manifest), str(manifest["phase"]))


def lineage_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    snapshot = [{key: row.get(key) for key in LINEAGE_FIELDS} for row in rows]
    return digest(snapshot)


def pending_jobs(
    manifest: Mapping[str, Any], completed_keys: set[str]
) -> dict[str, list[dict[str, Any]]]:
    expected = {job_key(job) for job in manifest["jobs"]}
    if not completed_keys <= expected:
        raise FormalRunError("lineage contains an outcome outside the frozen mask")
    result = {
        model: [
            job
            for job in manifest["jobs"]
            if job["model"] == model and job_key(job) not in completed_keys
        ]
        for model in model_order(manifest)
    }
    expected_counts = manifest["remaining_counts"]
    if any(len(result[model]) != int(expected_counts[model]) for model in model_order(manifest)):
        raise FormalRunError("lineage does not select the frozen remaining job set")
    return result
