from __future__ import annotations

import json
from pathlib import Path
from random import Random

import pytest

from spatial_worlds.registry import family_ids, get_family
from spatial_worlds.schema import Event, GoldRecord, Instance, SchemaError, WorldSpec
from spatial_worlds.simulate import gold_record, simulate

from oracles import FAMILY_VALUES, random_event, random_state

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "metric_paint.json"
EXAMPLE_GOLD = Path(__file__).resolve().parents[1] / "examples" / "metric_paint.gold.json"


def _random_instance(rng: Random, family_id: str, shape: tuple[int, ...], horizon: int) -> Instance:
    world = WorldSpec(shape, FAMILY_VALUES[family_id])
    initial_values = [1, 2] if family_id == "aggregating_compression" else None
    events = []
    for t in range(horizon):
        if t % 3 == 2:
            events.append(
                Event(
                    "SET", {"p": [rng.randrange(s) for s in shape], "v": rng.choice(world.values)}
                )
            )
        else:
            events.append(random_event(rng, family_id, world))
    return Instance(
        id=f"random/{family_id}",
        family=family_id,
        world=world,
        initial=random_state(rng, world, 0.5, initial_values),
        events=tuple(events),
    )


def test_example_gold_matches_the_shipped_record() -> None:
    instance = Instance.from_dict(json.loads(EXAMPLE.read_text()))
    gold = GoldRecord.from_dict(json.loads(EXAMPLE_GOLD.read_text()))
    assert gold_record(instance) == gold
    assert [w.to_dict() for w in gold.workloads] == [
        {"scan": 3, "effect": 2},
        {"scan": 1, "effect": 1},
        {"scan": 4, "effect": 2},
    ]


@pytest.mark.parametrize("family_id", sorted(family_ids()))
def test_simulation_is_deterministic_and_prefix_consistent(family_id: str) -> None:
    rng = Random(family_id)
    shape = (3, 4) if family_id != "aggregating_compression" else (3, 3)
    instance = _random_instance(rng, family_id, shape, 9)
    try:
        result = simulate(instance, states_at=[3, 6], keep_all_states=True)
    except SchemaError as error:
        assert family_id == "aggregating_compression", error
        return
    assert simulate(instance).final_state == result.final_state
    assert len(result.workloads) == instance.horizon
    assert result.state_at(0) == instance.initial
    for t in range(instance.horizon + 1):
        prefix = instance.prefix(t)
        assert simulate(prefix).final_state == result.state_at(t)
        assert result.gold(t).workloads == result.workloads[:t]
    assert result.gold().final_state == result.final_state
    assert [step.effect for step in result.steps] == [w.effect for w in result.workloads]
    for step in result.steps:
        assert step.predicate is None or isinstance(step.predicate, bool)
        assert 0.0 <= step.density <= 1.0


def test_simulate_rejects_states_at_outside_range(small_instance: Instance) -> None:
    with pytest.raises(SchemaError):
        simulate(small_instance, states_at=[7])
    with pytest.raises(KeyError):
        simulate(small_instance).state_at(1)


def test_simulate_validates_the_selected_family(small_instance: Instance) -> None:
    with pytest.raises(SchemaError, match="does not match"):
        simulate(small_instance, family=get_family("local_stencil"))
