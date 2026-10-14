"""Event-type schedules: the SET interleave and the effective/no-op plan."""

from __future__ import annotations

import math
from fractions import Fraction
from random import Random


def _rate(rate: float) -> Fraction:
    if isinstance(rate, Fraction):
        return rate
    return Fraction(str(rate))


def is_set(t: int, set_rate: float) -> bool:
    """Whether the 1-indexed event ``t`` is a SET under the deterministic interleave."""

    rate = _rate(set_rate)
    return math.floor(t * rate) > math.floor((t - 1) * rate)


def set_schedule(t_max: int, set_rate: float) -> list[bool]:
    """``schedule[t-1]`` is True when event ``t`` is a SET.

    Every prefix of length ``T`` contains exactly ``floor(T * set_rate)`` SET
    events, and the first event is always a core event for ``set_rate < 1``.
    """

    if not 0 <= set_rate < 1:
        raise ValueError("set_rate must lie in [0, 1)")
    return [is_set(t, set_rate) for t in range(1, t_max + 1)]


def set_count(horizon: int, set_rate: float) -> int:
    return math.floor(horizon * _rate(set_rate))


def core_count(horizon: int, set_rate: float) -> int:
    return horizon - set_count(horizon, set_rate)


def effective_schedule(
    n_core: int, noop_rate: float | None, rng: Random, *, block: int = 8
) -> list[bool]:
    """Block-randomized plan: ``True`` means the core event should be effective.

    Each full block of ``block`` core events receives the exact no-op quota
    implied by ``noop_rate`` (cumulative rounding over whole blocks), shuffled
    inside the block, and the plan is truncated to ``n_core``.  Quotas never
    depend on ``n_core``, so ``effective_schedule(n, ...)`` is a prefix of
    ``effective_schedule(N, ...)`` for ``n <= N`` when both start from the same
    generator state.  The very first core event is always effective.
    """

    if noop_rate is None:
        return [True] * n_core
    if not 0 <= noop_rate < 1:
        raise ValueError("noop_rate must lie in [0, 1)")
    rate = _rate(noop_rate)
    plan: list[bool] = []
    start = 0
    while len(plan) < n_core:
        end = start + block
        quota = math.floor(end * rate) - math.floor(start * rate)
        chunk = [False] * quota + [True] * (block - quota)
        rng.shuffle(chunk)
        if start == 0 and not chunk[0]:
            swap = chunk.index(True)
            chunk[0], chunk[swap] = chunk[swap], chunk[0]
        plan.extend(chunk)
        start = end
    return plan[:n_core]
