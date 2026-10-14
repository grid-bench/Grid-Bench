"""Check deterministic regeneration in owned scratch space; refuse replacement."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from spatial_worlds.diagnostics import diagnose_dataset, write_diagnostics
from spatial_worlds.eval.run import load_dataset
from spatial_worlds.generation import build_dataset
from spatial_worlds.generation.config import load_config

NAMES = ("calib3", "calib3w", "main", "rq2")
VOLATILE = {"created_utc", "git_commit", "git_dirty", "command", "python_version"}


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(path: Path) -> dict:
    manifest = json.loads((path / "manifest.json").read_text())
    files = sorted(path.glob("*/*/*.jsonl")) + [path / "diagnostics.jsonl"]
    return {
        "manifest": {key: value for key, value in manifest.items() if key not in VOLATILE},
        "files": {str(file.relative_to(path)): file_digest(file) for file in files},
    }


def compare(reference: Path, generated: Path) -> bool:
    return inventory(reference) == inventory(generated)


def validate_target(root: Path, name: str, check: bool) -> Path:
    if name not in NAMES:
        raise ValueError(f"dataset must be one of {NAMES}")
    data = root / "data"
    target = data / name
    if data.is_symlink() or target.is_symlink():
        raise ValueError("refusing a symlinked data directory")
    if check and not target.is_dir():
        raise ValueError(f"missing reference dataset: {name}")
    if not check and target.exists():
        raise ValueError(f"{name} already exists; use --check (replacement is never automatic)")
    return target


def regenerate(root: Path, name: str, target: Path, workers: int) -> None:
    build_dataset(load_config(root / "configs" / f"{name}.json"), target, workers=workers)
    dataset = load_dataset(target)
    write_diagnostics(
        diagnose_dataset(dataset.instances, dataset.gold), target / "diagnostics.jsonl"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", "8")))
    parser.add_argument("names", nargs="*")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    names = args.names or list(NAMES)
    try:
        targets = {name: validate_target(root, name, args.check) for name in names}
        if args.workers < 1:
            raise ValueError("workers must be positive")
        for name, target in targets.items():
            if args.check:
                with tempfile.TemporaryDirectory(prefix="spatial-worlds-check-") as scratch:
                    generated = Path(scratch) / name
                    regenerate(root, name, generated, args.workers)
                    if not compare(target, generated):
                        print(f"{name}: HASHES OR NONVOLATILE MANIFEST DIFFER")
                        return 1
                    print(f"{name}: content, diagnostics, normalized config and manifest match")
            else:
                regenerate(root, name, target, args.workers)
                (target / "content_hashes.json").write_text(
                    json.dumps(inventory(target), indent=2) + "\n"
                )
                print(f"{name}: generated; no previous dataset or baseline was removed")
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
