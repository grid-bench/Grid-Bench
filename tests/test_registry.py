from __future__ import annotations

import pytest

from spatial_worlds.family import Family
from spatial_worlds.registry import family_ids, get_family, list_families, validate_instance
from spatial_worlds.schema import Event, Instance, WorldSpec
from spatial_worlds.world import WorldState

EXPECTED_FAMILY_IDS = {
    "metric_paint",
    "box_reflection",
    "local_stencil",
    "occluded_mix",
    "component_relabel",
    "boundary_layer",
    "obstructed_motion",
    "aggregating_compression",
}

EXPECTED_TAXONOMY = {
    "metric_paint": ("geometry", "relabel", False, True),
    "box_reflection": ("geometry", "transport", False, False),
    "local_stencil": ("geometry", "aggregate", False, False),
    "occluded_mix": ("state", "aggregate", False, False),
    "component_relabel": ("state", "relabel", False, False),
    "boundary_layer": ("state", "relabel", False, False),
    "obstructed_motion": ("state", "transport", True, False),
    "aggregating_compression": ("geometry", "aggregate", True, False),
}


def test_registry_contains_exactly_the_eight_implemented_families() -> None:
    assert set(family_ids()) == EXPECTED_FAMILY_IDS
    assert len(list_families()) == 8

    for family_id in EXPECTED_FAMILY_IDS:
        family = get_family(family_id)
        assert isinstance(family, Family)
        assert family.metadata.status == "implemented"
        assert family.implemented
        assert family.proposer is not None
        support, update, ordered, has_predicate = EXPECTED_TAXONOMY[family_id]
        assert family.metadata.support_source == support
        assert family.metadata.update_type == update
        assert family.metadata.order_dependent is ordered
        assert (family.metadata.predicate is not None) is has_predicate
        assert family.metadata.to_dict()["core_op" if False else "id"] == family_id


def test_registry_rejects_unknown_family() -> None:
    with pytest.raises((KeyError, ValueError)):
        get_family("unknown_family")


def test_validate_instance_accepts_registered_family(small_instance: Instance) -> None:
    validate_instance(small_instance)


def test_core_transition_and_workload_are_executable(small_instance: Instance) -> None:
    family = get_family(small_instance.family)

    after = family.transition(small_instance.initial, small_instance.events[1])
    workload = family.workload(small_instance.initial, small_instance.events[1], after)

    assert after.get((0, 0)) == 2
    assert workload.scan == 3
    assert workload.effect == 1


def test_common_set_transition_and_workload_are_executable(
    small_instance: Instance,
) -> None:
    family = get_family(small_instance.family)

    after, workload = family.apply_event(small_instance.initial, small_instance.events[0])

    assert after.get((0, 1)) == 3
    assert workload.scan == 1
    assert workload.effect == 1
    assert family.workload(after, small_instance.events[0]).effect == 0


def test_workload_rejects_inconsistent_after_state(small_instance: Instance) -> None:
    family = get_family(small_instance.family)
    with pytest.raises(ValueError, match="inconsistent"):
        family.workload(small_instance.initial, small_instance.events[0], small_instance.initial)


def test_family_world_validation_cannot_be_bypassed_by_set_only_events() -> None:
    world = WorldSpec(shape=(2,), values=(0,))
    instance = Instance(
        id="invalid-local-alphabet",
        family="local_stencil",
        world=world,
        initial=WorldState.empty(world),
        events=(Event("SET", {"p": [0], "v": 0}),),
    )

    with pytest.raises(ValueError, match="M >= 2"):
        validate_instance(instance)


def test_occluded_mix_validates_its_alphabet_and_parameters() -> None:
    # two values leave a single residue, so the modular sum would be constant
    binary = WorldSpec(shape=(3,), values=(0, 1))
    instance = Instance(
        id="invalid-mix-alphabet",
        family="occluded_mix",
        world=binary,
        initial=WorldState.empty(binary),
        events=(Event("MIX", {"p": [0], "r": 1}),),
    )
    with pytest.raises(ValueError, match="M >= 3"):
        validate_instance(instance)
    world = WorldSpec(shape=(3,), values=(0, 1, 2))
    negative = Instance(
        id="invalid-mix-radius",
        family="occluded_mix",
        world=world,
        initial=WorldState.empty(world),
        events=(Event("MIX", {"p": [0], "r": -1}),),
    )
    with pytest.raises(ValueError, match="at least 0"):
        validate_instance(negative)
    with_extra = Instance(
        id="invalid-mix-value",
        family="occluded_mix",
        world=world,
        initial=WorldState.empty(world),
        events=(Event("MIX", {"p": [0], "r": 1, "v": 1}),),
    )
    with pytest.raises(ValueError, match="unknown v"):
        validate_instance(with_extra)


def test_boundary_layer_requires_a_signed_direction() -> None:
    world = WorldSpec(shape=(3,), values=(0, 1))
    instance = Instance(
        id="invalid-layer-sign",
        family="boundary_layer",
        world=world,
        initial=WorldState.empty(world),
        events=(Event("LAYER", {"p": [0], "s": 0}),),
    )
    with pytest.raises(ValueError, match="-1 or 1"):
        validate_instance(instance)


def test_draft_family_without_step_raises_not_implemented() -> None:
    base = get_family("metric_paint")
    draft = Family(
        metadata=base.metadata,
        rule=base.rule,
        event_validator=base.event_validator,
        world_validator=base.world_validator,
    )
    world = WorldSpec(shape=(2,), values=(0, 1))
    assert not draft.implemented
    with pytest.raises(NotImplementedError):
        draft.transition(
            WorldState.empty(world), Event("PAINT", {"a": [0], "r": 0, "u": 0, "v": 1})
        )
