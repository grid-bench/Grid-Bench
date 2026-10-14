"""Family metadata, validation, the shared SET transition, and the step contract.

A family supplies a ``step`` function that mutates a dense :class:`Grid` in
place and reports the canonical scan count, the outcome of the rule's explicit
predicate (when the rule has one), and diagnostic extras.  ``effect`` is never
reported by a family: it is computed generically as the Hamming distance
between the before and after grids, so no family can account for it
differently.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from random import Random
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from .grid import Grid, hamming
from .schema import Event, Instance, SchemaError, WorldSpec
from .workload import TransitionWorkload
from .world import WorldState

EventValidator = Callable[[WorldSpec, Event], None]
WorldValidator = Callable[[WorldSpec], None]
RuleCard = str | Callable[[WorldSpec], str]

STATUS_VALUES = ("draft", "implemented", "released")
SUPPORT_SOURCES = ("geometry", "state")
UPDATE_TYPES = ("relabel", "transport", "aggregate")


def _accept_world(world: WorldSpec) -> None:
    del world


def validate_discrete_alphabet(world: WorldSpec) -> None:
    """Require the finite label alphabet 0, 1, ..., M-1 with M >= 2."""

    if len(world.values) < 2 or world.values != tuple(range(len(world.values))):
        raise SchemaError("family requires world.values = [0, 1, ..., M-1] with M >= 2")


def validate_positive_alphabet(world: WorldSpec) -> None:
    """Require zero plus at least one strictly positive stored value."""

    if len(world.values) < 2 or any(value <= 0 for value in world.values if value != 0):
        raise SchemaError("family requires 0 and at least one positive world value")


@dataclass(frozen=True, slots=True)
class FamilyMetadata:
    id: str
    name: str
    status: str
    dependency_structure: str
    summary: str
    event_ops: tuple[str, ...]
    support_source: str = "geometry"
    update_type: str = "relabel"
    order_dependent: bool = False
    predicate: str | None = None

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", self.id):
            raise SchemaError("family id must be lower snake_case")
        for field_name in ("name", "dependency_structure", "summary"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise SchemaError(f"family metadata {field_name} must be a non-empty string")
        if self.status not in STATUS_VALUES:
            raise SchemaError(f"family status must be one of {', '.join(STATUS_VALUES)}")
        if self.support_source not in SUPPORT_SOURCES:
            raise SchemaError(f"family support_source must be one of {', '.join(SUPPORT_SOURCES)}")
        if self.update_type not in UPDATE_TYPES:
            raise SchemaError(f"family update_type must be one of {', '.join(UPDATE_TYPES)}")
        if not isinstance(self.order_dependent, bool):
            raise SchemaError("family order_dependent must be a boolean")
        if self.predicate is not None and (
            not isinstance(self.predicate, str) or not self.predicate
        ):
            raise SchemaError("family predicate must be None or a non-empty string")
        event_ops = tuple(self.event_ops)
        if not event_ops or event_ops[0] != "SET":
            raise SchemaError("family event_ops must begin with SET")
        if len(set(event_ops)) != len(event_ops):
            raise SchemaError("family event_ops must be unique")
        if any(re.fullmatch(r"[A-Z][A-Z0-9_]*", op) is None for op in event_ops):
            raise SchemaError("family event_ops must use uppercase operation names")
        object.__setattr__(self, "event_ops", event_ops)

    @property
    def core_op(self) -> str:
        return self.event_ops[1]

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "dependency_structure": self.dependency_structure,
            "summary": self.summary,
            "event_ops": list(self.event_ops),
            "support_source": self.support_source,
            "update_type": self.update_type,
            "order_dependent": self.order_dependent,
            "predicate": self.predicate,
        }


@dataclass(frozen=True, slots=True)
class StepResult:
    """What a family reports after mutating a grid in place."""

    scan: int
    predicate: bool | None = None
    extras: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.scan, int) or isinstance(self.scan, bool) or self.scan < 0:
            raise SchemaError("StepResult.scan must be a nonnegative integer")
        if self.predicate is not None and not isinstance(self.predicate, bool):
            raise SchemaError("StepResult.predicate must be None or a boolean")
        object.__setattr__(self, "extras", MappingProxyType(dict(self.extras)))


@dataclass(frozen=True, slots=True)
class Transition:
    """One executed event: the after-state plus every recorded measurement."""

    after: WorldState
    workload: TransitionWorkload
    predicate: bool | None
    effective: bool
    extras: Mapping[str, int]


StepFunction = Callable[[Grid, Event], StepResult]
ProposeFunction = Callable[[Random, Grid, int, bool, int], list[Event]]
NoteFunction = Callable[[WorldSpec, tuple[int, ...], tuple[int, ...]], str]


@dataclass(frozen=True, slots=True)
class Example:
    """A one-dimensional worked example rendered after the rule card.

    ``before`` lists the values of cells [0], [1], ... of a one-dimensional
    world that shares the instance's alphabet.  The after-state is computed by
    the family's own step function when the card is rendered, so the example
    can never disagree with the simulator; ``note`` explains the outcome and
    may be a function of ``(world, before, after)``.
    """

    before: tuple[int, ...]
    event: Event
    note: str | NoteFunction

    def __post_init__(self) -> None:
        values = tuple(self.before)
        if not values or any(not isinstance(v, int) or isinstance(v, bool) for v in values):
            raise SchemaError("Example.before must be a non-empty tuple of integers")
        object.__setattr__(self, "before", values)
        if not isinstance(self.event, Event):
            raise SchemaError("Example.event must be an Event")
        if not callable(self.note) and (not isinstance(self.note, str) or not self.note):
            raise SchemaError("Example.note must be a non-empty string or a callable")


@runtime_checkable
class FamilyProtocol(Protocol):
    @property
    def metadata(self) -> FamilyMetadata: ...

    def rule_card(self, world: WorldSpec) -> str: ...

    def example_card(self, world: WorldSpec) -> str | None: ...

    def validate_world(self, world: WorldSpec) -> None: ...

    def validate_event(self, world: WorldSpec, event: Event) -> None: ...

    def step_grid(self, grid: Grid, event: Event) -> StepResult: ...

    def step(self, state: WorldState, event: Event) -> Transition: ...

    def transition(self, state: WorldState, event: Event) -> WorldState: ...

    def workload(
        self,
        before: WorldState,
        event: Event,
        after: WorldState | None = None,
    ) -> TransitionWorkload: ...


def require_params(event: Event, *names: str) -> None:
    expected = set(names)
    actual = set(event.params)
    if actual != expected:
        missing = expected - actual
        extra = actual - expected
        parts: list[str] = []
        if missing:
            parts.append(f"missing {', '.join(sorted(missing))}")
        if extra:
            parts.append(f"unknown {', '.join(sorted(extra))}")
        raise SchemaError(f"{event.op} parameters: {'; '.join(parts)}")


def coordinate_param(world: WorldSpec, event: Event, name: str) -> tuple[int, ...]:
    return world.validate_position(event.params[name], label=f"{event.op}.{name}")


def value_param(world: WorldSpec, event: Event, name: str) -> int:
    return world.validate_value(event.params[name], label=f"{event.op}.{name}")


def integer_param(
    event: Event,
    name: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    value = event.params[name]
    if not isinstance(value, int) or isinstance(value, bool):
        raise SchemaError(f"{event.op}.{name} must be an integer")
    if minimum is not None and value < minimum:
        raise SchemaError(f"{event.op}.{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise SchemaError(f"{event.op}.{name} must be at most {maximum}")
    return value


def axis_param(world: WorldSpec, event: Event, name: str = "axis") -> int:
    return integer_param(event, name, minimum=0, maximum=world.dimension - 1)


def sign_param(event: Event, name: str = "direction") -> int:
    value = integer_param(event, name)
    if value not in (-1, 1):
        raise SchemaError(f"{event.op}.{name} must be -1 or 1")
    return value


def vector_param(world: WorldSpec, event: Event, name: str) -> tuple[int, ...]:
    value = event.params[name]
    if not isinstance(value, (list, tuple)) or len(value) != world.dimension:
        raise SchemaError(f"{event.op}.{name} must have dimension {world.dimension}")
    vector = tuple(value)
    if any(
        not isinstance(component, int) or isinstance(component, bool) or component not in (-1, 0, 1)
        for component in vector
    ):
        raise SchemaError(f"{event.op}.{name} components must be -1, 0, or 1")
    if not any(vector):
        raise SchemaError(f"{event.op}.{name} must be nonzero")
    return vector


def validate_set(world: WorldSpec, event: Event) -> None:
    require_params(event, "p", "v")
    coordinate_param(world, event, "p")
    value_param(world, event, "v")


def set_event(position: Any, value: int) -> Event:
    return Event("SET", {"p": list(position), "v": value})


@dataclass(frozen=True, slots=True)
class Family:
    """Declarative family: metadata, rule card, validators, simulator step, proposer."""

    metadata: FamilyMetadata
    rule: RuleCard
    event_validator: EventValidator
    step_fn: StepFunction | None = None
    proposer: ProposeFunction | None = None
    world_validator: WorldValidator = _accept_world
    example: Example | None = None

    def rule_card(self, world: WorldSpec) -> str:
        return self.rule(world) if callable(self.rule) else self.rule

    def example_card(self, world: WorldSpec) -> str | None:
        """Render the worked example for ``world``'s alphabet, or None if it does not fit.

        The example lives on a one-dimensional world with the same alphabet as
        ``world``; its after-state is produced by :meth:`step_grid`.
        """

        example = self.example
        if example is None or self.step_fn is None:
            return None
        if any(value not in world.allowed_values for value in example.before):
            return None
        line = WorldSpec((len(example.before),), world.values)
        grid = Grid(line, list(example.before))
        try:
            self.validate_event(line, example.event)
            self.step_grid(grid, example.event)
        except SchemaError:
            return None
        if any(value not in world.allowed_values for value in grid.data):
            return None
        before = example.before
        after = tuple(grid.data)
        note = example.note(world, before, after) if callable(example.note) else example.note
        size = len(before)
        event = json.dumps(example.event.to_dict(), ensure_ascii=False, separators=(",", ":"))
        return (
            f"On shape [{size}], listing the values of cells [0] to [{size - 1}] in order. "
            f"Before: {' '.join(str(v) for v in before)}. "
            f"Event: {event}. "
            f"After: {' '.join(str(v) for v in after)}. "
            f"{note}"
        )

    def validate_world(self, world: WorldSpec) -> None:
        self.world_validator(world)

    def validate_event(self, world: WorldSpec, event: Event) -> None:
        self.validate_world(world)
        if event.op == "SET":
            validate_set(world, event)
            return
        if event.op not in self.metadata.event_ops:
            allowed = ", ".join(self.metadata.event_ops)
            raise SchemaError(
                f"event {event.op!r} is not valid for {self.metadata.id}; expected {allowed}"
            )
        self.event_validator(world, event)

    @property
    def implemented(self) -> bool:
        return self.step_fn is not None

    def step_grid(self, grid: Grid, event: Event) -> StepResult:
        """Apply an already-validated event to ``grid`` in place."""

        if event.op == "SET":
            index = grid.index(event.params["p"])
            grid.data[index] = event.params["v"]
            return StepResult(scan=1)
        if self.step_fn is None:
            raise NotImplementedError(
                f"{self.metadata.id}.{event.op} transition is a draft and is not implemented"
            )
        return self.step_fn(grid, event)

    def step(self, state: WorldState, event: Event) -> Transition:
        self.validate_event(state.world, event)
        before = Grid.from_state(state)
        after = before.copy()
        result = self.step_grid(after, event)
        effect = hamming(before.data, after.data)
        effective = result.predicate if result.predicate is not None else effect > 0
        return Transition(
            after=after.to_state(),
            workload=TransitionWorkload(scan=result.scan, effect=effect),
            predicate=result.predicate,
            effective=effective,
            extras=result.extras,
        )

    def transition(self, state: WorldState, event: Event) -> WorldState:
        return self.step(state, event).after

    def workload(
        self,
        before: WorldState,
        event: Event,
        after: WorldState | None = None,
    ) -> TransitionWorkload:
        transition = self.step(before, event)
        if after is not None:
            if after.world != before.world:
                raise SchemaError("before and after states use different world specifications")
            if after != transition.after:
                raise SchemaError(f"after state is inconsistent with the {event.op} transition")
        return transition.workload

    def apply_event(self, state: WorldState, event: Event) -> tuple[WorldState, TransitionWorkload]:
        transition = self.step(state, event)
        return transition.after, transition.workload


DraftFamily = Family


def effective_outcome(predicate: bool | None, effect: int) -> bool:
    """The generator's notion of an event that 'did something'."""

    return predicate if predicate is not None else effect > 0


def validate_family_instance(family: FamilyProtocol, instance: Instance) -> None:
    if instance.family != family.metadata.id:
        raise SchemaError(
            f"instance family {instance.family!r} does not match {family.metadata.id!r}"
        )
    family.validate_world(instance.world)
    for index, event in enumerate(instance.events):
        try:
            family.validate_event(instance.world, event)
        except (SchemaError, ValueError) as error:
            raise SchemaError(f"invalid events[{index}]: {error}") from error
