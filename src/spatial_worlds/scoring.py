"""Strict parsing and exact-state scoring for model predictions."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from .schema import GoldRecord, SchemaError, parse_json_document
from .world import WorldState


@dataclass(frozen=True, slots=True)
class ScoreResult:
    """Exact-match score plus diagnostics that never alter the primary metric."""

    exact_match: bool
    valid_json: bool
    valid_state: bool
    canonical_format: bool
    wrong_cells: int | None
    total_cells: int
    cell_accuracy: float
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _invalid(total: int, *, valid_json: bool, error: str) -> ScoreResult:
    return ScoreResult(
        exact_match=False,
        valid_json=valid_json,
        valid_state=False,
        canonical_format=False,
        wrong_cells=None,
        total_cells=total,
        cell_accuracy=0.0,
        error=error,
    )


def score_prediction(prediction: str, gold: GoldRecord) -> ScoreResult:
    """Score one complete prediction against a gold final state.

    The primary metric is semantic full-world exact match. Canonical formatting
    is reported separately, so a reordered but otherwise correct coordinate
    table remains semantically exact.
    """

    if not isinstance(prediction, str):
        raise TypeError("prediction must be a string")
    if not isinstance(gold, GoldRecord):
        raise TypeError("gold must be a GoldRecord")

    total = gold.world.volume
    try:
        payload = parse_json_document(prediction)
    except (json.JSONDecodeError, SchemaError, ValueError) as error:
        return _invalid(total, valid_json=False, error=str(error))

    try:
        predicted, canonical = WorldState.parse_dict(payload, gold.world)
    except (SchemaError, TypeError, ValueError) as error:
        return _invalid(total, valid_json=True, error=str(error))

    predicted_cells = predicted.as_dict()
    gold_cells = gold.final_state.as_dict()
    occupied = predicted_cells.keys() | gold_cells.keys()
    wrong = sum(
        predicted_cells.get(position, 0) != gold_cells.get(position, 0) for position in occupied
    )
    return ScoreResult(
        exact_match=wrong == 0,
        valid_json=True,
        valid_state=True,
        canonical_format=canonical,
        wrong_cells=wrong,
        total_cells=total,
        cell_accuracy=(total - wrong) / total,
        error=None,
    )


def score_query_prediction(
    prediction: str, gold: GoldRecord, queries: Sequence[Sequence[int]]
) -> ScoreResult:
    """Score a query-contract answer ``{"values": [[q, v], ...]}``.

    Semantic validity requires every query coordinate exactly once with an
    allowed value; canonical validity additionally requires the listed order.
    ``total_cells`` is the number of queries and ``wrong_cells`` counts wrong
    query values.
    """

    if not isinstance(prediction, str):
        raise TypeError("prediction must be a string")
    if not isinstance(gold, GoldRecord):
        raise TypeError("gold must be a GoldRecord")
    expected = [gold.world.validate_position(q, label="query") for q in queries]
    if not expected:
        raise SchemaError("score_query_prediction requires at least one query")
    if len(set(expected)) != len(expected):
        raise SchemaError("queries must be unique")
    total = len(expected)

    try:
        payload = parse_json_document(prediction)
    except (json.JSONDecodeError, SchemaError, ValueError) as error:
        return _invalid(total, valid_json=False, error=str(error))

    try:
        if not isinstance(payload, Mapping):
            raise SchemaError("answer must be a JSON object")
        if set(payload) != {"values"}:
            raise SchemaError("answer must contain exactly the key 'values'")
        entries = payload["values"]
        if not isinstance(entries, list):
            raise SchemaError("answer.values must be an array")
        if len(entries) != total:
            raise SchemaError(f"answer.values has {len(entries)} entries; expected {total}")
        answered: dict[tuple[int, ...], int] = {}
        order: list[tuple[int, ...]] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, list) or len(entry) != 2:
                raise SchemaError(f"values[{index}] must be [coordinate, value]")
            coordinate = gold.world.validate_position(entry[0], label=f"values[{index}][0]")
            value = gold.world.validate_value(entry[1], label=f"values[{index}][1]")
            if coordinate in answered:
                raise SchemaError(f"duplicate query coordinate: {list(coordinate)}")
            answered[coordinate] = value
            order.append(coordinate)
        if set(answered) != set(expected):
            raise SchemaError("answer.values does not cover exactly the query coordinates")
    except (SchemaError, TypeError, ValueError) as error:
        return _invalid(total, valid_json=True, error=str(error))

    final = gold.final_state.as_dict()
    wrong = sum(answered[q] != final.get(q, 0) for q in expected)
    return ScoreResult(
        exact_match=wrong == 0,
        valid_json=True,
        valid_state=True,
        canonical_format=order == expected,
        wrong_cells=wrong,
        total_cells=total,
        cell_accuracy=(total - wrong) / total,
        error=None,
    )
