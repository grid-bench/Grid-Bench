"""Reference simulation of a complete instance with a per-step trace."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .family import Family, effective_outcome, validate_family_instance
from .grid import Grid, hamming
from .registry import get_family
from .schema import GoldRecord, Instance, SchemaError
from .workload import TransitionWorkload
from .world import WorldState


@dataclass(frozen=True, slots=True)
class StepTrace:
    """Measurements recorded for one executed event."""

    index: int
    op: str
    scan: int
    effect: int
    predicate: bool | None
    effective: bool
    extras: Mapping[str, int]
    density: float

    def to_dict(self) -> dict[str, object]:
        return {
            "t": self.index + 1,
            "op": self.op,
            "scan": self.scan,
            "effect": self.effect,
            "predicate": self.predicate,
            "effective": self.effective,
            "extras": dict(self.extras),
            "density": self.density,
        }


@dataclass(frozen=True, slots=True)
class SimulationResult:
    instance_id: str
    family: str
    initial: WorldState
    final_state: WorldState
    steps: tuple[StepTrace, ...]
    states: Mapping[int, WorldState]

    @property
    def horizon(self) -> int:
        return len(self.steps)

    @property
    def workloads(self) -> tuple[TransitionWorkload, ...]:
        return tuple(TransitionWorkload(step.scan, step.effect) for step in self.steps)

    def state_at(self, horizon: int) -> WorldState:
        try:
            return self.states[horizon]
        except KeyError as error:
            raise KeyError(f"state at horizon {horizon} was not retained") from error

    def gold(self, horizon: int | None = None, *, instance_id: str | None = None) -> GoldRecord:
        """Gold record for the full rollout or for a retained prefix."""

        if horizon is None:
            horizon = self.horizon
        final = self.final_state if horizon == self.horizon else self.state_at(horizon)
        return GoldRecord(
            instance_id=self.instance_id if instance_id is None else instance_id,
            family=self.family,
            world=self.initial.world,
            final_state=final,
            workloads=self.workloads[:horizon],
        )


def simulate(
    instance: Instance,
    *,
    states_at: Iterable[int] | None = None,
    keep_all_states: bool = False,
    family: Family | None = None,
) -> SimulationResult:
    """Run every event of ``instance`` and return the final state with a trace.

    ``states_at`` lists horizons whose intermediate states must be retained
    (0 and the final horizon are always retained).  Any transition that
    produces a value outside ``world.values`` raises :class:`SchemaError`.
    """

    if family is None:
        family = get_family(instance.family)
    validate_family_instance(family, instance)
    horizon = instance.horizon
    wanted = {0, horizon}
    if states_at is not None:
        for t in states_at:
            if not isinstance(t, int) or t < 0 or t > horizon:
                raise SchemaError(f"states_at entry {t!r} is outside [0, {horizon}]")
            wanted.add(t)
    if keep_all_states:
        wanted.update(range(horizon + 1))

    grid = Grid.from_state(instance.initial)
    states: dict[int, WorldState] = {0: instance.initial}
    steps: list[StepTrace] = []
    allowed = instance.world.allowed_values
    for index, event in enumerate(instance.events):
        before = grid.data[:]
        result = family.step_grid(grid, event)
        if any(value not in allowed for value in grid.data):
            raise SchemaError(f"events[{index}] {event.op} produced a value outside world.values")
        effect = hamming(before, grid.data)
        steps.append(
            StepTrace(
                index=index,
                op=event.op,
                scan=result.scan,
                effect=effect,
                predicate=result.predicate,
                effective=effective_outcome(result.predicate, effect),
                extras=result.extras,
                density=grid.density,
            )
        )
        if index + 1 in wanted:
            states[index + 1] = grid.to_state()
    final = states[horizon]
    return SimulationResult(
        instance_id=instance.id,
        family=instance.family,
        initial=instance.initial,
        final_state=final,
        steps=tuple(steps),
        states=MappingProxyType(states),
    )


def gold_record(instance: Instance) -> GoldRecord:
    """Convenience wrapper: the gold record of a full simulation."""

    return simulate(instance).gold()


__all__ = ["SimulationResult", "StepTrace", "gold_record", "simulate"]
