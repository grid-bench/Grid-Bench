"""Deterministic, Python-version-independent seed derivation."""

from __future__ import annotations

import hashlib


def derive_seed(
    dataset_seed: int,
    family: str,
    tier: str,
    dimension: int,
    w: int,
    split: str,
    index: int,
    attempt: int = 0,
) -> int:
    key = f"{dataset_seed}|{family}|{tier}|d={dimension}|w={w}|split={split}|i={index}|a={attempt}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")
