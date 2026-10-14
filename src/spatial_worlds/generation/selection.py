"""Candidate scoring for the workload-targeted event selector."""

from __future__ import annotations

import math
from random import Random


def score_candidate(
    scan: int,
    target: int,
    density_after: float,
    band: tuple[float, float],
    *,
    density_weight: float = 2.0,
    tolerance: float = 0.0,
) -> float:
    """Log-distance to the scan target plus a penalty for leaving the density band.

    Candidates within ``tolerance`` (a relative band around the target) score
    zero on workload, so that :func:`choose` picks uniformly among them instead
    of always returning the single closest one; without this a family with few
    components would replay the same component event after event.
    """

    workload = abs(math.log(max(scan, 1) / max(target, 1)))
    workload = max(0.0, workload - math.log1p(max(tolerance, 0.0)))
    lo, hi = band
    width = max(hi - lo, 1e-9)
    mid = (lo + hi) / 2
    excess = max(0.0, abs(density_after - mid) - width / 2)
    return workload + density_weight * excess / width


def choose(rng: Random, scores: list[float]) -> int:
    """Index of the minimal score; ties are broken by the seeded generator."""

    if not scores:
        raise ValueError("choose() requires at least one candidate")
    best = min(scores)
    ties = [index for index, score in enumerate(scores) if score == best]
    return rng.choice(ties)
