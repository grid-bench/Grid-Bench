"""Deterministic minimal extraction of a JSON answer from a model response.

The extractor never repairs anything.  It only removes one enclosing code
fence or locates the last balanced top-level JSON object in the text; the
extracted text is then scored by the same fail-closed scorer as the raw text.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass

from ..schema import SchemaError, parse_json_document

_FENCE = re.compile(r"^```[A-Za-z0-9_-]*[ \t]*\r?\n(.*?)\r?\n?```$", re.DOTALL)

METHODS = ("none", "fence", "last_object", "failed")


@dataclass(frozen=True, slots=True)
class Extraction:
    text: str | None
    method: str

    @property
    def applied(self) -> bool:
        return self.method != "none"


def _parses_to_object(text: str) -> bool:
    try:
        return isinstance(parse_json_document(text), Mapping)
    except (json.JSONDecodeError, SchemaError, ValueError):
        return False






SCORING_VERSION = "spatial-worlds-scoring/1.0"


def extract_final_answer(raw: str) -> Extraction:
    """Formal protocol: parse only the LAST candidate, never repair or fall back.

    An unclosed or malformed trailing object invalidates an earlier complete answer.
    Braces inside JSON strings do not start candidates.
    This selection is entirely independent of gold.
    """
    if not isinstance(raw, str):
        raise TypeError("raw response must be a string")
    text = raw.strip()
    match = _FENCE.fullmatch(text)
    method = "fence" if match else "none"
    if match:
        text = match.group(1).strip()
    if _parses_to_object(text):
        return Extraction(text, method)
    depth, start = 0, None
    quoted = escaped = False
    candidate = None
    for index, char in enumerate(text):
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"' and depth:
            quoted = True
        elif char == "{":
            if depth == 0:
                start = index
                candidate = None
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                candidate = text[start : index + 1]
    if depth or candidate is None or not _parses_to_object(candidate):
        return Extraction(None, "failed")
    return Extraction(candidate, "last_object")
