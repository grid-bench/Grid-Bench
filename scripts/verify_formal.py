"""Read-only integrity check for the single formal workflow; never queries an API."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from spatial_worlds.eval.study_final import _status, _validate_lineage
from spatial_worlds.eval.study_final_config import (
    build_manifest,
    load_config,
    model_order,
    paths,
    request_parameters,
)
from spatial_worlds.eval.study_final_store import FinalStore

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"(?<!!)\[[^]]*\]\(([^)]+)\)")
ACTIVE_STUDY_FILES = {
    "study.py",
    "study_final.py",
    "study_final_bedrock.py",
    "study_final_config.py",
    "study_final_gateway.py",
    "study_final_report.py",
    "study_final_store.py",
    "study_report.py",
}


def documentation_check() -> dict[str, int]:
    documents = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
    links = 0
    for document in documents:
        for raw in LINK.findall(document.read_text()):
            target = raw.strip().strip("<>").split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            links += 1
            resolved = (document.parent / target).resolve()
            if not resolved.exists():
                raise AssertionError(f"broken local link in {document.name}: {raw}")
    return {"documents": len(documents), "local_links": links}


def audit(config_path: Path | None = None) -> dict:
    config = load_config(config_path or ROOT / "configs/formal.json")
    manifest, _ = build_manifest(ROOT, config)
    models = model_order(manifest)
    out, ledger = paths(ROOT, config)
    source_files = {path.name for path in (ROOT / "src/spatial_worlds/eval").glob("study*.py")}
    if source_files != ACTIVE_STUDY_FILES:
        raise AssertionError(f"unexpected study workflow files: {sorted(source_files)}")
    with FinalStore(ledger, manifest["campaign_id"], readonly=True) as store:
        _validate_lineage(store, manifest)
        status = _status(store, manifest)
    if set(status["models"]) != set(models):
        raise AssertionError("formal status does not contain the configured models")
    if any(value["planned"] != 744 for value in status["models"].values()):
        raise AssertionError("formal mask is not 744 rows per model")
    return {
        "campaign_id": manifest["campaign_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "selection_sha256": manifest["selection_sha256"],
        "lineage_outcomes": manifest["lineage_outcomes"],
        "remaining_counts": manifest["remaining_counts"],
        "model_visible_settings": {
            "combined_generation_tokens": manifest["resources"]["max_tokens"],
            "reasoning_effort": {
                model: request_parameters(value).get(
                    "reasoning_effort", value.get("effort")
                )
                for model, value in (
                    (model, next(item for item in manifest["models"] if item["id"] == model))
                    for model in models
                )
            },
            "prompt_version": manifest["prompt_version"],
            "temperature_top_p_seed": "omitted",
        },
        "progress": {
            model: {
                key: status["models"][model][key]
                for key in ("completed", "remaining", "empty", "technical_calls")
            }
            for model in models
        },
        "inflight": status["inflight"],
        "dispatch": status["dispatch"],
        "formal_accounting": status["formal_accounting"],
        "documents": documentation_check(),
        "paid_calls": 0,
        "output_directory": str(out.relative_to(ROOT)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/formal.json")
    args = parser.parse_args()
    print(json.dumps(audit(args.config.resolve()), indent=2))
