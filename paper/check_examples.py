"""Check literal manuscript examples against two executors, without changing files.

Run from the repository root with ``uv run --frozen python paper/check_examples.py``.
The expected arrays in examples.json were calculated by hand. This script never
regenerates them from either executor and never contacts a model provider.
"""

from __future__ import annotations

import importlib.util
import json
from itertools import product
from pathlib import Path
from typing import Any

from spatial_worlds.schema import Event, GoldRecord, Instance, WorldSpec, parse_json_document
from spatial_worlds.scoring import score_prediction
from spatial_worlds.simulate import simulate
from spatial_worlds.world import WorldState

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = Path(__file__).with_name("examples.json")


def flatten(values: Any, shape: tuple[int, ...]) -> list[int]:
    """Validate a nested display array and flatten in lexicographic order."""
    if not shape:
        if type(values) is not int:
            raise ValueError("display cells must be integers")
        return [values]
    if not isinstance(values, list) or len(values) != shape[0]:
        raise ValueError(f"display array does not match shape {shape}")
    return [item for part in values for item in flatten(part, shape[1:])]


def dense_display(values: Any, world: WorldSpec) -> dict[tuple[int, ...], int]:
    coordinates = product(*(range(size) for size in world.shape))
    dense = dict(zip(coordinates, flatten(values, world.shape), strict=True))
    for value in dense.values():
        world.validate_value(value)
    return dense


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, received {actual!r}")


def main() -> None:
    specification = importlib.util.spec_from_file_location(
        "paper_example_oracles", ROOT / "tests" / "oracles.py"
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load independent test oracles")
    oracle = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(oracle)
    fixture = parse_json_document(EXAMPLES.read_text(encoding="utf-8"))
    examples = fixture["examples"]
    require_equal(len(examples), 10, "example count")
    require_equal(len({example["id"] for example in examples}), 10, "unique example IDs")
    require_equal(len({example["family"] for example in examples}), 8, "family coverage")
    transitions = 0
    for example in examples:
        world = WorldSpec.from_dict(example["world"])
        initial_dense = dense_display(example["initial"], world)
        initial = WorldState(
            world, tuple((position, value) for position, value in initial_dense.items() if value)
        )
        events = tuple(Event.from_dict(event) for event in example["events"])
        for field in ("expected_states", "expected_scan", "expected_effect"):
            require_equal(len(example[field]), len(events), f"{example['id']}: {field} count")
        instance = Instance(example["id"], example["family"], world, initial, events)
        result = simulate(instance, keep_all_states=True)
        oracle_dense = dict(initial_dense)
        previous = dict(initial_dense)
        changed_per_step = []
        for index, event in enumerate(events):
            label = f"{example['id']} event {index + 1} ({event.op})"
            expected = dense_display(example["expected_states"][index], world)
            production = result.state_at(index + 1)
            production_dense = {position: production[position] for position in expected}
            oracle_dense, scan, predicate = oracle.apply_oracle(
                instance.family, world, oracle_dense, event
            )
            require_equal(production_dense, expected, f"{label}: production state")
            require_equal(oracle_dense, expected, f"{label}: independent oracle state")
            require_equal(scan, example["expected_scan"][index], f"{label}: oracle scan")
            require_equal(result.steps[index].scan, scan, f"{label}: production scan")
            require_equal(result.steps[index].predicate, predicate, f"{label}: predicate")
            require_equal(
                result.steps[index].effect, example["expected_effect"][index], f"{label}: effect"
            )
            changed_per_step.append([list(p) for p in expected if previous[p] != expected[p]])
            previous = dict(expected)
            transitions += 1
        final_dense = {p: result.final_state[p] for p in initial_dense}
        changed = [list(p) for p in initial_dense if initial_dense[p] != final_dense[p]]
        derived = {
            "changed_per_step": changed_per_step,
            "changed_final_coordinates": changed,
            "changed_final_count": len(changed),
            "unchanged_final_count": len(initial_dense) - len(changed),
            "total_cells": len(initial_dense),
            "initial_nonzero_count": sum(value != 0 for value in initial_dense.values()),
            "final_nonzero_count": sum(value != 0 for value in final_dense.values()),
        }
        for field, actual in derived.items():
            if field in example:
                require_equal(actual, example[field], f"{example['id']}: {field}")
        if "expected_final_state" in example:
            encoded = json.dumps(example["expected_final_state"], separators=(",", ":"))
            parsed = parse_json_document(encoded)
            state, canonical = WorldState.parse_dict(parsed, world)
            require_equal(canonical, True, f"{example['id']}: canonical sparse JSON")
            require_equal(state, result.final_state, f"{example['id']}: sparse final state")
            print(f"{example['id']} final JSON: {encoded}")
        for prediction in example.get("illustrative_predictions", []):
            point = tuple(prediction["coordinate"])
            require_equal(final_dense[point], prediction["reference_value"],
                          f"{example['id']}: toy-error reference")
            dense = dict(final_dense)
            dense[point] = prediction["predicted_value"]
            answer = WorldState(world, tuple((p, value) for p, value in dense.items() if value))
            gold = GoldRecord(example["id"], example["family"], world, result.final_state)
            score = score_prediction(json.dumps(answer.to_dict()), gold).to_dict()
            for field, expected in prediction["score"].items():
                require_equal(score[field], expected, f"{example['id']}: toy-error {field}")
            require_equal([p for p in dense if dense[p] != final_dense[p]], [point],
                          f"{example['id']}: exactly one toy error")
        print(f"PASS {example['id']}: {len(events)} transition(s)")
    require_equal(transitions, 14, "transition count")
    print(f"Verified {len(examples)} examples, {transitions} transitions, two executors each.")


if __name__ == "__main__":
    main()
