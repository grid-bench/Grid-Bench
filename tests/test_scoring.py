from __future__ import annotations

import json

import pytest

from grid_bench.schema import GoldRecord, WorldSpec
from grid_bench.scoring import score_prediction
from grid_bench.world import WorldState


def make_gold(cells: tuple[tuple[tuple[int, ...], int], ...]) -> GoldRecord:
    world = WorldSpec(shape=(2, 2), values=(0, 1, 2))
    return GoldRecord(
        instance_id="score-test",
        family="metric_paint",
        world=world,
        final_state=WorldState(world, cells),
    )


def test_exact_prediction_scores_perfectly() -> None:
    gold = make_gold((((0, 0), 1), ((1, 1), 2)))
    prediction = json.dumps(gold.final_state.to_dict(), separators=(",", ":"))

    score = score_prediction(prediction, gold)

    assert score.exact_match
    assert score.valid_json
    assert score.valid_state
    assert score.canonical_format
    assert score.wrong_cells == 0
    assert score.total_cells == 4
    assert score.cell_accuracy == pytest.approx(1.0)
    assert score.error is None


def test_unordered_prediction_is_semantically_exact_but_not_canonical() -> None:
    gold = make_gold((((0, 0), 1), ((1, 1), 2)))
    prediction = '{"cells":[[[1,1],2],[[0,0],1]]}'

    score = score_prediction(prediction, gold)

    assert score.exact_match
    assert score.valid_state
    assert not score.canonical_format


def test_one_wrong_cell_reports_cell_accuracy() -> None:
    gold = make_gold((((0, 0), 1),))

    score = score_prediction('{"cells":[]}', gold)

    assert not score.exact_match
    assert score.valid_state
    assert score.wrong_cells == 1
    assert score.total_cells == 4
    assert score.cell_accuracy == pytest.approx(0.75)


def test_malformed_json_is_reported_without_raising() -> None:
    score = score_prediction("not json", make_gold(()))

    assert not score.exact_match
    assert not score.valid_json
    assert not score.valid_state
    assert score.error


def test_explicit_zero_is_an_invalid_prediction_state() -> None:
    score = score_prediction('{"cells":[[[0,0],0]]}', make_gold(()))

    assert score.valid_json
    assert not score.valid_state
    assert not score.exact_match


def test_duplicate_json_keys_are_rejected() -> None:
    score = score_prediction('{"cells":[],"cells":[]}', make_gold(()))

    assert not score.valid_json
    assert not score.valid_state
    assert "duplicate" in (score.error or "")


def test_query_scoring_is_strict_and_reports_query_accuracy() -> None:
    from grid_bench.scoring import score_query_prediction

    gold = make_gold((((0, 0), 1), ((1, 1), 2)))
    queries = [(0, 0), (1, 0), (1, 1)]

    exact = score_query_prediction('{"values":[[[0,0],1],[[1,0],0],[[1,1],2]]}', gold, queries)
    assert exact.exact_match and exact.canonical_format and exact.total_cells == 3

    reordered = score_query_prediction('{"values":[[[1,1],2],[[0,0],1],[[1,0],0]]}', gold, queries)
    assert reordered.exact_match and reordered.valid_state and not reordered.canonical_format

    wrong = score_query_prediction('{"values":[[[0,0],1],[[1,0],1],[[1,1],2]]}', gold, queries)
    assert not wrong.exact_match and wrong.wrong_cells == 1
    assert wrong.cell_accuracy == pytest.approx(2 / 3)

    for bad in (
        '{"values":[[[0,0],1],[[1,0],0]]}',
        '{"values":[[[0,0],1],[[0,0],0],[[1,1],2]]}',
        '{"values":[[[0,0],1],[[1,0],0],[[1,1],9]]}',
        '{"cells":[]}',
        "nope",
    ):
        result = score_query_prediction(bad, gold, queries)
        assert not result.exact_match and not result.valid_state
