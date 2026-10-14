"""The official extractor never falls back to an earlier answer."""

import pytest

from grid_bench.eval.extract import extract_final_answer


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"cells":[]}', '{"cells":[]}'),
        ('```json\n{"cells":[]}\n```', '{"cells":[]}'),
        ('old {"cells":[]} final {"cells":[', None),
        ('old {"cells":[]} final {"cells": BAD}', None),
        ('reason {"s":"}"} answer {"cells":[]}', '{"cells":[]}'),
        ("Nothing here", None),
        ("[1,2,3]", None),
    ],
)
def test_final_candidate_only(raw: str, expected: str | None) -> None:
    assert extract_final_answer(raw).text == expected
