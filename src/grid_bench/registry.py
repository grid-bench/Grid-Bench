"""Stable registry for the eight transition families."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from .families import ALL_FAMILIES
from .family import Family, validate_family_instance
from .schema import Instance, SchemaError

_families = {family.metadata.id: family for family in ALL_FAMILIES}
if len(_families) != len(ALL_FAMILIES):
    raise RuntimeError("duplicate family id in registry")

FAMILIES: Mapping[str, Family] = MappingProxyType(_families)


def family_ids() -> tuple[str, ...]:
    return tuple(sorted(FAMILIES))


def list_families() -> tuple[Family, ...]:
    return tuple(FAMILIES[family_id] for family_id in family_ids())


def get_family(family_id: str) -> Family:
    try:
        return FAMILIES[family_id]
    except KeyError as error:
        raise KeyError(f"unknown family id: {family_id!r}") from error


def validate_instance(instance: Instance) -> None:
    if not isinstance(instance, Instance):
        raise SchemaError("validate_instance expects an Instance")
    try:
        family = get_family(instance.family)
    except KeyError as error:
        raise SchemaError(str(error)) from error
    validate_family_instance(family, instance)
