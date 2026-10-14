from __future__ import annotations

import pytest

from grid_bench.schema import SCHEMA_VERSION, Event, Instance, WorldSpec


def test_world_spec_round_trip() -> None:
    spec = WorldSpec.from_dict({"shape": [2, 3, 4], "values": [0, 1, 2]})

    assert spec.shape == (2, 3, 4)
    assert spec.values == (0, 1, 2)
    assert spec.to_dict() == {"shape": [2, 3, 4], "values": [0, 1, 2]}


@pytest.mark.parametrize(
    "payload",
    [
        {"shape": [], "values": [0, 1]},
        {"shape": [2, 0], "values": [0, 1]},
        {"shape": [2, 2], "values": [1, 2]},
        {"shape": [2, 2], "values": [0, 1, 1]},
    ],
)
def test_world_spec_rejects_invalid_domains(payload: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        WorldSpec.from_dict(payload)


def test_event_preserves_operation_parameters() -> None:
    event = Event.from_dict({"op": "SET", "p": [1, 0], "v": 2})

    assert event.op == "SET"
    assert event.params["p"] == (1, 0)
    assert event.params["v"] == 2
    assert event.to_dict() == {"op": "SET", "p": [1, 0], "v": 2}


def test_event_rejects_reserved_op_parameter() -> None:
    with pytest.raises(ValueError, match="reserved"):
        Event("SET", {"op": "PAINT", "p": [1, 0], "v": 2})


def test_instance_json_round_trip(small_instance: Instance) -> None:
    payload = small_instance.to_dict()

    assert payload["schema_version"] == SCHEMA_VERSION
    assert Instance.from_dict(payload) == small_instance


def test_instance_requires_current_schema_version(small_instance: Instance) -> None:
    payload = small_instance.to_dict()
    payload["schema_version"] = "spatial-worlds/99"

    with pytest.raises((TypeError, ValueError)):
        Instance.from_dict(payload)
