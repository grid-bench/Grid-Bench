"""Small statistical helpers: Wilson intervals, the per-step error MLE, bootstrap."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from random import Random


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""

    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, center - half), min(1.0, center + half))


def _log_likelihood(epsilon: float, observations: Sequence[tuple[int, bool]]) -> float:
    total = 0.0
    log_survival = math.log1p(-epsilon)
    for n_core, correct in observations:
        if n_core <= 0:
            continue
        if correct:
            total += n_core * log_survival
        else:
            failure = -math.expm1(n_core * log_survival)
            total += math.log(max(failure, 1e-300))
    return total


def epsilon_mle(observations: Sequence[tuple[int, bool]]) -> float | None:
    """MLE of the per-core-event error rate under independent errors.

    Model: P(correct after n core events) = (1 - epsilon) ** n.  Observations
    with ``n_core == 0`` carry no information and are ignored.
    """

    informative = [(n, y) for n, y in observations if n > 0]
    if not informative:
        return None
    if all(y for _, y in informative):
        return 0.0
    if not any(y for _, y in informative):
        return 1.0
    lo_exp, hi_exp = math.log10(1e-5), math.log10(0.999)
    points = 400
    grid = [10 ** (lo_exp + (hi_exp - lo_exp) * i / (points - 1)) for i in range(points)]
    values = [_log_likelihood(eps, informative) for eps in grid]
    best = max(range(points), key=values.__getitem__)
    a = grid[max(best - 1, 0)]
    b = grid[min(best + 1, points - 1)]
    golden = (math.sqrt(5) - 1) / 2
    c = b - golden * (b - a)
    d = a + golden * (b - a)
    for _ in range(60):
        if _log_likelihood(c, informative) > _log_likelihood(d, informative):
            b = d
        else:
            a = c
        c = b - golden * (b - a)
        d = a + golden * (b - a)
    return (a + b) / 2


def bootstrap_epsilon(
    rollouts: Sequence[Sequence[tuple[int, bool]]],
    *,
    resamples: int = 200,
    seed: int = 0,
) -> tuple[float, float] | None:
    """Percentile bootstrap over rollouts (clusters) of :func:`epsilon_mle`."""

    if not rollouts or resamples <= 0:
        return None
    rng = Random(seed)
    estimates: list[float] = []
    for _ in range(resamples):
        sample = [rollouts[rng.randrange(len(rollouts))] for _ in rollouts]
        flat = [obs for rollout in sample for obs in rollout]
        value = epsilon_mle(flat)
        if value is not None:
            estimates.append(value)
    if not estimates:
        return None
    estimates.sort()
    return (percentile(estimates, 2.5), percentile(estimates, 97.5))


def percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("percentile of empty sequence")
    position = (len(sorted_values) - 1) * q / 100
    lower = math.floor(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def linear_fit(points: Sequence[tuple[float, float]]) -> tuple[float, float, float] | None:
    """Ordinary least squares ``y = a + b x``; returns ``(a, b, r_squared)``.

    ``None`` when fewer than two distinct x values are available.  With two
    points the fit is exact and ``r_squared`` is 1.
    """

    if len(points) < 2:
        return None
    mean_x = sum(x for x, _ in points) / len(points)
    mean_y = sum(y for _, y in points) / len(points)
    sxx = sum((x - mean_x) ** 2 for x, _ in points)
    if sxx <= 0:
        return None
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in points)
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    total = sum((y - mean_y) ** 2 for _, y in points)
    residual = sum((y - intercept - slope * x) ** 2 for x, y in points)
    r_squared = 1.0 if total <= 0 else max(0.0, 1.0 - residual / total)
    return (intercept, slope, r_squared)


def residual_sd(points: Sequence[tuple[float, float]], intercept: float, slope: float) -> float:
    """Sample standard deviation of the residuals ``y - (a + b x)`` (0 for fewer than 3 points)."""

    if len(points) < 3:
        return 0.0
    residuals = [y - intercept - slope * x for x, y in points]
    mean_residual = sum(residuals) / len(residuals)
    variance = sum((r - mean_residual) ** 2 for r in residuals) / (len(residuals) - 2)
    return math.sqrt(max(variance, 0.0))


def bootstrap_difference(
    reference: Sequence[bool],
    other: Sequence[bool],
    *,
    resamples: int = 200,
    seed: int = 0,
) -> tuple[float, float] | None:
    """Percentile bootstrap interval of ``mean(other) - mean(reference)`` (independent samples)."""

    if not reference or not other or resamples <= 0:
        return None
    rng = Random(seed)
    estimates: list[float] = []
    for _ in range(resamples):
        a = [reference[rng.randrange(len(reference))] for _ in reference]
        b = [other[rng.randrange(len(other))] for _ in other]
        estimates.append(sum(b) / len(b) - sum(a) / len(a))
    estimates.sort()
    return (percentile(estimates, 2.5), percentile(estimates, 97.5))


def newcombe_difference(
    reference: Sequence[bool],
    other: Sequence[bool],
) -> tuple[float, float] | None:
    """95% Newcombe method 10 (no CC), other minus independent reference.

    Not appropriate for two models evaluated on the SAME instances.
    Unlike an empirical percentile bootstrap, this remains informative at
    the 0/n and n/n boundaries. It is an approximate, marginal interval.
    """
    if not reference or not other:
        return None
    a, b = sum(other) / len(other), sum(reference) / len(reference)
    la, ua = wilson_interval(sum(other), len(other))
    lb, ub = wilson_interval(sum(reference), len(reference))
    return (
        max(-1.0, a - b - math.hypot(a - la, ub - b)),
        min(1.0, a - b + math.hypot(ua - a, b - lb)),
    )


def stratified_cluster_interval(
    strata: Mapping[str, Mapping[str, Sequence[float]]],
    *,
    resamples: int = 2000,
    seed: int = 0,
) -> dict[str, object]:
    """Mean CI, resampling WHOLE trajectories within fixed design strata.

    Paired comparisons supply per-instance differences in the same clusters.
    Marginal intervals, not simultaneous bands or a remedy for small samples.
    """
    groups = [list(strata[key].values()) for key in sorted(strata)]
    warning = None
    if not groups or resamples < 1 or any(len(group) < 2 for group in groups):
        return {"ci95": None, "warning": "insufficient_trajectory_clusters"}
    rng = Random(seed)
    estimates = []
    for _ in range(resamples):
        values = [
            value for group in groups for _ in group for value in group[rng.randrange(len(group))]
        ]
        if not values:
            return {"ci95": None, "warning": "empty_clusters"}
        estimates.append(sum(values) / len(values))
    estimates.sort()
    low, high = percentile(estimates, 2.5), percentile(estimates, 97.5)
    if low == high:
        warning = "degenerate_bootstrap_not_certainty"
    return {
        "ci95": None if warning else [low, high],
        "warning": warning,
        "method": "stratified_trajectory_percentile_bootstrap",
        "resamples": resamples,
        "seed": seed,
        "n_trajectories": sum(len(group) for group in groups),
    }
