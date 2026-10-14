"""Sequential rollout generation with workload targeting and degeneracy checks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from random import Random
from statistics import fmean, median
from typing import Any

from ..family import Family, effective_outcome
from ..grid import Grid, hamming
from ..registry import get_family
from ..schema import Event, GoldRecord, Instance, SchemaError, WorldSpec
from ..workload import TransitionWorkload
from ..world import WorldState
from .config import CellSpec, RejectionConfig
from .initial import sample_initial
from .schedule import effective_schedule, set_schedule
from .selection import choose, score_candidate
from .setpolicy import propose_set


class GenerationError(RuntimeError):
    """A candidate rollout violated a rejection criterion."""


# An effective core event may not return the world to one of this many
# preceding states (exactly, or within one cell when it changes at least four),
# so that a family whose events have inverses cannot alternate an event and
# its undo on the same support.
RETURN_WINDOW = 8
RETURN_MIN_EFFECT = 4


def _returns(candidate: _Candidate, recent: list[list[int]]) -> bool:
    if candidate.effect == 0:
        return False
    for state in recent:
        distance = hamming(candidate.data, state)
        if distance == 0 or (distance <= 1 and candidate.effect >= RETURN_MIN_EFFECT):
            return True
    return False


@dataclass(frozen=True, slots=True)
class RolloutStep:
    t: int
    event: Event
    scan: int
    effect: int
    predicate: bool | None
    effective: bool
    kind_wanted: bool | None
    kind_miss: bool
    extras: Mapping[str, int]
    density: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "t": self.t,
            "op": self.event.op,
            "scan": self.scan,
            "effect": self.effect,
            "predicate": self.predicate,
            "effective": self.effective,
            "kind_wanted": self.kind_wanted,
            "kind_miss": self.kind_miss,
            "extras": dict(self.extras),
            "density": round(self.density, 6),
        }


@dataclass(frozen=True, slots=True)
class Rollout:
    cell: CellSpec
    seed: int
    attempt: int
    world: WorldSpec
    initial: WorldState
    events: tuple[Event, ...]
    steps: tuple[RolloutStep, ...]
    states_at: Mapping[int, WorldState]
    stats: Mapping[str, Any]

    @property
    def horizon(self) -> int:
        return len(self.events)

    def workloads(self, horizon: int) -> tuple[TransitionWorkload, ...]:
        return tuple(TransitionWorkload(s.scan, s.effect) for s in self.steps[:horizon])

    def gold(self, horizon: int, instance_id: str) -> GoldRecord:
        return GoldRecord(
            instance_id=instance_id,
            family=self.cell.family,
            world=self.world,
            final_state=self.states_at[horizon],
            workloads=self.workloads(horizon),
        )

    def instance(
        self,
        horizon: int,
        instance_id: str,
        metadata: Mapping[str, Any],
        queries: tuple[tuple[int, ...], ...] = (),
    ) -> Instance:
        return Instance(
            id=instance_id,
            family=self.cell.family,
            world=self.world,
            initial=self.initial,
            events=self.events[:horizon],
            metadata=metadata,
            queries=queries,
        )


@dataclass(slots=True)
class _Candidate:
    event: Event
    data: list[int]
    scan: int
    effect: int
    predicate: bool | None
    effective: bool
    density: float
    extras: Mapping[str, int]


def _evaluate(
    family: Family, grid: Grid, event: Event, allowed: frozenset[int]
) -> _Candidate | None:
    scratch = grid.copy()
    try:
        result = family.step_grid(scratch, event)
    except SchemaError:
        return None
    if any(value not in allowed for value in scratch.data):
        return None
    effect = hamming(grid.data, scratch.data)
    return _Candidate(
        event=event,
        data=scratch.data,
        scan=result.scan,
        effect=effect,
        predicate=result.predicate,
        effective=effective_outcome(result.predicate, effect),
        density=scratch.density,
        extras=result.extras,
    )


def _workload_target(cell: CellSpec, size: int) -> int:
    """The workload the candidate selector aims for, in the cell's ``w_kind`` units."""

    if cell.w_kind == "scan":
        return cell.w
    if cell.w_kind == "support":
        return size if cell.whole_world else cell.w
    return size


def _measure(cell: CellSpec, scan: int, extras: Mapping[str, int]) -> int:
    """The realized workload of one core event in the cell's ``w_kind`` units."""

    if cell.w_kind == "support":
        return int(extras.get("support", scan))
    return scan


def generate_rollout(
    cell: CellSpec,
    seed: int,
    *,
    attempt: int = 0,
    rejection: RejectionConfig | None = None,
    retain: Iterable[int] | None = None,
) -> Rollout:
    """Generate one rollout of ``cell.t_max`` events or raise :class:`GenerationError`.

    ``retain`` lists the horizons whose states are kept (default: the cell's T
    grid); horizon 0 and ``t_max`` are always kept.
    """

    rejection = rejection or RejectionConfig.from_dict(None)
    keep = set(cell.t_grid if retain is None else retain) | {0, cell.t_max}
    rng = Random(seed)
    world = WorldSpec(cell.shape, cell.values)
    family = get_family(cell.family)
    family.validate_world(world)
    if family.proposer is None:
        raise GenerationError(f"family {cell.family} has no event proposer")
    labels = tuple(value for value in world.values if value != 0)
    allowed = world.allowed_values
    lo, hi = cell.density
    band = (lo, hi)
    initial_density = rng.uniform(lo, hi)
    grid = sample_initial(rng, world, initial_density, cell.initializer, cell.w)
    initial_state = grid.to_state()

    schedule = set_schedule(cell.t_max, cell.set_rate)
    n_core = sum(1 for is_set in schedule if not is_set)
    # The plan comes from its own stream so that a rollout with a shorter t_max
    # and the same seed is exactly a prefix of the longer one.
    plan = effective_schedule(n_core, cell.noop_rate, Random(f"{seed}:plan"))
    target = _workload_target(cell, grid.size)
    slack = rejection.density_slack
    events: list[Event] = []
    steps: list[RolloutStep] = []
    states_at: dict[int, WorldState] = {0: initial_state}
    core_index = 0
    scratch_events = 0
    recent: list[list[int]] = []
    returns_blocked = 0

    for t in range(1, cell.t_max + 1):
        previous = grid.data[:]
        if schedule[t - 1]:
            event = propose_set(cell.set_policy, rng, grid, band, labels)
            family.validate_event(world, event)
            candidate = _evaluate(family, grid, event, allowed)
            if candidate is None:
                raise GenerationError(f"SET at t={t} produced an invalid state")
            kind_wanted: bool | None = None
            kind_miss = False
        else:
            kind_wanted = plan[core_index]
            core_index += 1
            evaluated: list[_Candidate] = []
            for wanted in (kind_wanted, not kind_wanted):
                proposals = family.proposer(rng, grid, target, wanted, cell.candidates)
                for event in proposals:
                    family.validate_event(world, event)
                    scratch_events += 1
                    item = _evaluate(family, grid, event, allowed)
                    if item is not None:
                        evaluated.append(item)
                if any(c.effective == kind_wanted for c in evaluated):
                    break
            if not evaluated:
                raise GenerationError(f"no valid candidate at t={t}")
            fresh = [c for c in evaluated if not _returns(c, recent)]
            if fresh:
                returns_blocked += len(evaluated) - len(fresh)
                evaluated = fresh
            matching = [c for c in evaluated if c.effective == kind_wanted]
            kind_miss = not matching
            pool = matching or evaluated
            scores = [
                score_candidate(
                    _measure(cell, c.scan, c.extras),
                    target,
                    c.density,
                    band,
                    tolerance=cell.tolerance,
                )
                for c in pool
            ]
            candidate = pool[choose(rng, scores)]
        grid.data = candidate.data
        recent.append(previous)
        del recent[:-RETURN_WINDOW]
        events.append(candidate.event)
        steps.append(
            RolloutStep(
                t=t,
                event=candidate.event,
                scan=candidate.scan,
                effect=candidate.effect,
                predicate=candidate.predicate,
                effective=candidate.effective,
                kind_wanted=kind_wanted,
                kind_miss=kind_miss,
                extras=candidate.extras,
                density=candidate.density,
            )
        )
        density = candidate.density
        if density < max(0.02, lo - slack) or density > min(0.98, hi + slack):
            raise GenerationError(f"density {density:.3f} left the band at t={t}")
        if grid.nonzero_count == 0:
            raise GenerationError(f"world became empty at t={t}")
        if cell.w_kind == "support" and not schedule[t - 1]:
            support = int(candidate.extras.get("support", grid.size))
            if candidate.effect < rejection.stencil_min_effect_fraction * support:
                raise GenerationError(f"{cell.family} near a fixed point at t={t}")
        if t in keep:
            states_at[t] = grid.to_state()

    core_steps = [s for s in steps if s.kind_wanted is not None]
    core_scans = [s.scan for s in core_steps]
    mean_scan = fmean(core_scans) if core_scans else 0.0
    core_measures = [_measure(cell, s.scan, s.extras) for s in core_steps]
    mean_measure = fmean(core_measures) if core_measures else 0.0
    ratio_lo, ratio_hi = rejection.scan_ratio_band
    if (
        cell.w_kind in ("scan", "support")
        and core_measures
        and not (ratio_lo * target <= mean_measure <= ratio_hi * target)
    ):
        raise GenerationError(
            f"mean core workload {mean_measure:.1f} is outside [{ratio_lo}, {ratio_hi}] x {target}"
        )
    effective_fraction = (
        sum(1 for s in core_steps if s.effective) / len(core_steps) if core_steps else 0.0
    )
    if core_steps and effective_fraction < rejection.floor_for(cell.family):
        raise GenerationError(f"effective fraction {effective_fraction:.2f} below floor")
    kind_miss_fraction = (
        sum(1 for s in core_steps if s.kind_miss) / len(core_steps) if core_steps else 0.0
    )
    if core_steps and kind_miss_fraction > rejection.max_kind_miss:
        raise GenerationError(f"kind miss fraction {kind_miss_fraction:.2f} too high")

    densities = [initial_density] + [s.density for s in steps]
    predicate_steps = [s for s in core_steps if s.predicate is not None]
    stats: dict[str, Any] = {
        "initial_density": round(initial_state.density, 6),
        "density_min": round(min(densities), 6),
        "density_max": round(max(densities), 6),
        "final_density": round(steps[-1].density, 6),
        "core_events": len(core_steps),
        "set_events": len(steps) - len(core_steps),
        "core_scan_mean": round(mean_scan, 3),
        "core_scan_median": float(median(core_scans)) if core_scans else 0.0,
        "core_scan_min": min(core_scans) if core_scans else 0,
        "core_scan_max": max(core_scans) if core_scans else 0,
        "core_w_mean": round(mean_measure, 3),
        "core_effect_mean": round(fmean(s.effect for s in core_steps), 3) if core_steps else 0.0,
        "core_scan_effective_mean": (
            round(fmean(s.scan for s in core_steps if s.effective), 3)
            if any(s.effective for s in core_steps)
            else None
        ),
        "core_scan_ineffective_mean": (
            round(fmean(s.scan for s in core_steps if not s.effective), 3)
            if any(not s.effective for s in core_steps)
            else None
        ),
        "w_target": target,
        "effective_fraction": round(effective_fraction, 4),
        "predicate_false_fraction": (
            round(sum(1 for s in predicate_steps if s.predicate is False) / len(predicate_steps), 4)
            if predicate_steps
            else None
        ),
        "kind_miss_fraction": round(kind_miss_fraction, 4),
        "candidates_evaluated": scratch_events,
        "returns_blocked": returns_blocked,
        "attempt": attempt,
    }
    return Rollout(
        cell=cell,
        seed=seed,
        attempt=attempt,
        world=world,
        initial=initial_state,
        events=tuple(events),
        steps=tuple(steps),
        states_at=states_at,
        stats=stats,
    )
