"""Immutable, JSON-serializable schemas for benchmark artifacts."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .workload import TransitionWorkload
    from .world import WorldState


SCHEMA_VERSION = "spatial-worlds/0.2"


class SchemaError(ValueError):
    """Raised when a benchmark artifact does not satisfy the schema."""


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SchemaError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise SchemaError(f"non-finite JSON number: {value}")


def parse_json_document(text: str) -> Any:
    """Parse strict JSON, rejecting duplicate object keys and non-finite numbers."""

    if not isinstance(text, str):
        raise TypeError("JSON document must be a string")
    return json.loads(
        text,
        object_pairs_hook=_unique_json_object,
        parse_constant=_reject_json_constant,
    )


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def freeze_json(value: Any) -> Any:
    """Recursively copy JSON-like data into immutable containers."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if _is_int(value):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SchemaError("JSON numbers must be finite")
        return value
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item) for item in value)
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise SchemaError("JSON object keys must be strings")
            frozen[key] = freeze_json(item)
        return MappingProxyType(frozen)
    raise SchemaError(f"unsupported JSON value: {type(value).__name__}")


def thaw_json(value: Any) -> Any:
    """Convert immutable JSON-like data back to ordinary dicts and lists."""

    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


def _require_object(data: object, label: str) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise SchemaError(f"{label} must be a JSON object")
    if any(not isinstance(key, str) for key in data):
        raise SchemaError(f"{label} keys must be strings")
    return data


def _check_keys(
    data: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
    label: str,
) -> None:
    optional = optional or set()
    missing = required - data.keys()
    extra = data.keys() - required - optional
    if missing:
        raise SchemaError(f"{label} is missing keys: {', '.join(sorted(missing))}")
    if extra:
        raise SchemaError(f"{label} has unknown keys: {', '.join(sorted(extra))}")


@dataclass(frozen=True, slots=True)
class WorldSpec:
    """Finite rectangular lattice and its allowed integer cell values."""

    shape: tuple[int, ...]
    values: tuple[int, ...]

    def __post_init__(self) -> None:
        shape = tuple(self.shape)
        values = tuple(self.values)
        if not shape:
            raise SchemaError("world.shape must contain at least one axis")
        if any(not _is_int(size) or size <= 0 for size in shape):
            raise SchemaError("every world.shape entry must be a positive integer")
        if not values:
            raise SchemaError("world.values must not be empty")
        if any(not _is_int(value) for value in values):
            raise SchemaError("every world.values entry must be an integer")
        if len(set(values)) != len(values):
            raise SchemaError("world.values must not contain duplicates")
        if tuple(sorted(values)) != values:
            raise SchemaError("world.values must be in strictly increasing order")
        if 0 not in values:
            raise SchemaError("world.values must include the default value 0")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "values", values)

    @property
    def dimension(self) -> int:
        return len(self.shape)

    @property
    def volume(self) -> int:
        return math.prod(self.shape)

    @property
    def allowed_values(self) -> frozenset[int]:
        return frozenset(self.values)

    def validate_position(self, position: object, *, label: str = "position") -> tuple[int, ...]:
        if not isinstance(position, (list, tuple)):
            raise SchemaError(f"{label} must be a coordinate array")
        result = tuple(position)
        if len(result) != self.dimension:
            raise SchemaError(f"{label} has dimension {len(result)}; expected {self.dimension}")
        for axis, (coordinate, size) in enumerate(zip(result, self.shape, strict=True)):
            if not _is_int(coordinate):
                raise SchemaError(f"{label}[{axis}] must be an integer")
            if coordinate < 0 or coordinate >= size:
                raise SchemaError(f"{label}[{axis}]={coordinate} is outside [0, {size})")
        return result

    def validate_value(self, value: object, *, label: str = "value") -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise SchemaError(f"{label} must be an integer")
        if value not in self.allowed_values:
            raise SchemaError(f"{label}={value} is not in world.values")
        return value

    def to_dict(self) -> dict[str, object]:
        return {"shape": list(self.shape), "values": list(self.values)}

    @classmethod
    def from_dict(cls, data: object) -> WorldSpec:
        obj = _require_object(data, "world")
        _check_keys(obj, required={"shape", "values"}, label="world")
        shape = obj["shape"]
        values = obj["values"]
        if not isinstance(shape, (list, tuple)):
            raise SchemaError("world.shape must be an array")
        if not isinstance(values, (list, tuple)):
            raise SchemaError("world.values must be an array")
        return cls(tuple(shape), tuple(values))


@dataclass(frozen=True, slots=True)
class Event:
    """An operation name with immutable JSON-like parameters."""

    op: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.op, str) or not self.op:
            raise SchemaError("event.op must be a non-empty string")
        if re.fullmatch(r"[A-Z][A-Z0-9_]*", self.op) is None:
            raise SchemaError("event.op must use uppercase letters, digits, and underscores")
        if not isinstance(self.params, Mapping):
            raise SchemaError("event.params must be a mapping")
        if "op" in self.params:
            raise SchemaError("event.params must not contain the reserved key 'op'")
        object.__setattr__(self, "params", freeze_json(self.params))

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            **{key: thaw_json(self.params[key]) for key in sorted(self.params)},
        }

    @classmethod
    def from_dict(cls, data: object) -> Event:
        obj = _require_object(data, "event")
        if "op" not in obj:
            raise SchemaError("event is missing key: op")
        op = obj["op"]
        if not isinstance(op, str):
            raise SchemaError("event.op must be a string")
        return cls(op=op, params={key: value for key, value in obj.items() if key != "op"})


@dataclass(frozen=True, slots=True)
class Instance:
    """A complete initial world and the event sequence to simulate."""

    id: str
    family: str
    world: WorldSpec
    initial: WorldState
    events: tuple[Event, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    queries: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        from .world import WorldState

        if self.schema_version != SCHEMA_VERSION:
            raise SchemaError(
                f"unsupported schema_version {self.schema_version!r}; expected {SCHEMA_VERSION!r}"
            )
        if not isinstance(self.id, str) or not self.id:
            raise SchemaError("instance.id must be a non-empty string")
        if not isinstance(self.family, str) or not self.family:
            raise SchemaError("instance.family must be a non-empty string")
        if not isinstance(self.world, WorldSpec):
            raise SchemaError("instance.world must be a WorldSpec")
        if not isinstance(self.initial, WorldState):
            raise SchemaError("instance.initial must be a WorldState")
        if self.initial.world != self.world:
            raise SchemaError("instance.initial uses a different world specification")
        events = tuple(self.events)
        if any(not isinstance(event, Event) for event in events):
            raise SchemaError("instance.events must contain Event values")
        object.__setattr__(self, "events", events)
        if not isinstance(self.metadata, Mapping):
            raise SchemaError("instance.metadata must be an object")
        object.__setattr__(self, "metadata", freeze_json(self.metadata))
        if not isinstance(self.queries, (list, tuple)):
            raise SchemaError("instance.queries must be an array of coordinates")
        queries: list[tuple[int, ...]] = []
        seen: set[tuple[int, ...]] = set()
        for index, query in enumerate(self.queries):
            coordinate = self.world.validate_position(query, label=f"queries[{index}]")
            if coordinate in seen:
                raise SchemaError(f"duplicate query coordinate: {list(coordinate)}")
            seen.add(coordinate)
            queries.append(coordinate)
        object.__setattr__(self, "queries", tuple(queries))

    @property
    def horizon(self) -> int:
        return len(self.events)

    @property
    def dimension(self) -> int:
        return self.world.dimension

    def prefix(
        self,
        horizon: int,
        *,
        id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Instance:
        """The instance with only the first ``horizon`` events."""

        if not isinstance(horizon, int) or horizon < 0 or horizon > len(self.events):
            raise SchemaError("prefix horizon must lie in [0, len(events)]")
        return Instance(
            id=self.id if id is None else id,
            family=self.family,
            world=self.world,
            initial=self.initial,
            events=self.events[:horizon],
            metadata=self.metadata if metadata is None else metadata,
            schema_version=self.schema_version,
            queries=(),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "id": self.id,
            "family": self.family,
            "world": self.world.to_dict(),
            "initial": self.initial.to_dict(),
            "events": [event.to_dict() for event in self.events],
            "metadata": thaw_json(self.metadata),
        }
        if self.queries:
            payload["queries"] = [list(query) for query in self.queries]
        return payload

    @classmethod
    def from_dict(cls, data: object) -> Instance:
        from .world import WorldState

        obj = _require_object(data, "instance")
        _check_keys(
            obj,
            required={"schema_version", "id", "family", "world", "initial", "events"},
            optional={"metadata", "queries"},
            label="instance",
        )
        if not isinstance(obj["events"], (list, tuple)):
            raise SchemaError("instance.events must be an array")
        queries = obj.get("queries", ())
        if not isinstance(queries, (list, tuple)):
            raise SchemaError("instance.queries must be an array")
        world = WorldSpec.from_dict(obj["world"])
        return cls(
            id=obj["id"],
            family=obj["family"],
            world=world,
            initial=WorldState.from_dict(obj["initial"], world),
            events=tuple(Event.from_dict(event) for event in obj["events"]),
            metadata=obj.get("metadata", {}),
            schema_version=obj["schema_version"],
            queries=tuple(tuple(q) if isinstance(q, (list, tuple)) else q for q in queries),
        )


@dataclass(frozen=True, slots=True)
class GoldRecord:
    """Gold final state plus optional per-transition workload measurements."""

    instance_id: str
    family: str
    world: WorldSpec
    final_state: WorldState
    workloads: tuple[TransitionWorkload, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        from .workload import TransitionWorkload
        from .world import WorldState

        if self.schema_version != SCHEMA_VERSION:
            raise SchemaError(
                f"unsupported schema_version {self.schema_version!r}; expected {SCHEMA_VERSION!r}"
            )
        if not isinstance(self.instance_id, str) or not self.instance_id:
            raise SchemaError("gold.instance_id must be a non-empty string")
        if not isinstance(self.family, str) or not self.family:
            raise SchemaError("gold.family must be a non-empty string")
        if not isinstance(self.world, WorldSpec):
            raise SchemaError("gold.world must be a WorldSpec")
        if not isinstance(self.final_state, WorldState):
            raise SchemaError("gold.final_state must be a WorldState")
        if self.final_state.world != self.world:
            raise SchemaError("gold.final_state uses a different world specification")
        workloads = tuple(self.workloads)
        if any(not isinstance(item, TransitionWorkload) for item in workloads):
            raise SchemaError("gold.workloads must contain TransitionWorkload values")
        object.__setattr__(self, "workloads", workloads)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "instance_id": self.instance_id,
            "family": self.family,
            "world": self.world.to_dict(),
            "final_state": self.final_state.to_dict(),
            "workloads": [item.to_dict() for item in self.workloads],
        }

    @classmethod
    def from_dict(cls, data: object) -> GoldRecord:
        from .workload import TransitionWorkload
        from .world import WorldState

        obj = _require_object(data, "gold record")
        _check_keys(
            obj,
            required={
                "schema_version",
                "instance_id",
                "family",
                "world",
                "final_state",
                "workloads",
            },
            label="gold record",
        )
        if not isinstance(obj["workloads"], (list, tuple)):
            raise SchemaError("gold.workloads must be an array")
        world = WorldSpec.from_dict(obj["world"])
        return cls(
            instance_id=obj["instance_id"],
            family=obj["family"],
            world=world,
            final_state=WorldState.from_dict(obj["final_state"], world),
            workloads=tuple(TransitionWorkload.from_dict(item) for item in obj["workloads"]),
            schema_version=obj["schema_version"],
        )
