"""Workload measurements attached to individual transitions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .schema import SchemaError


def _nonnegative_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SchemaError(f"{label} must be a nonnegative integer")
    return value


@dataclass(frozen=True, slots=True)
class TransitionWorkload:
    """Structural work inspected by a transition and its realized state changes."""

    scan: int
    effect: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "scan", _nonnegative_int(self.scan, "workload.scan"))
        object.__setattr__(self, "effect", _nonnegative_int(self.effect, "workload.effect"))

    def __add__(self, other: TransitionWorkload) -> TransitionWorkload:
        if not isinstance(other, TransitionWorkload):
            return NotImplemented
        return TransitionWorkload(self.scan + other.scan, self.effect + other.effect)

    def to_dict(self) -> dict[str, int]:
        return {"scan": self.scan, "effect": self.effect}

    @classmethod
    def from_dict(cls, data: object) -> TransitionWorkload:
        if not isinstance(data, Mapping):
            raise SchemaError("workload must be a JSON object")
        if set(data) != {"scan", "effect"}:
            raise SchemaError("workload must contain exactly scan and effect")
        return cls(scan=data["scan"], effect=data["effect"])
