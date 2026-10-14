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


def top_level_object_spans(text: str) -> list[tuple[int, int]]:
    """Spans of every balanced ``{...}`` opened at brace depth zero outside strings."""

    spans: list[tuple[int, int]] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            if depth > 0:
                in_string = True
            continue
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0:
                spans.append((start, index + 1))
    return spans


def extract_answer(raw: str) -> Extraction:
    """Locate the answer object in ``raw`` without modifying its content."""

    if not isinstance(raw, str):
        raise TypeError("raw response must be a string")
    stripped = raw.strip()
    method = "none"
    match = _FENCE.match(stripped)
    if match is not None:
        stripped = match.group(1).strip()
        method = "fence"
    if stripped and _parses_to_object(stripped):
        return Extraction(stripped, method)
    for begin, end in reversed(top_level_object_spans(stripped)):
        candidate = stripped[begin:end]
        if _parses_to_object(candidate):
            return Extraction(candidate, "last_object")
    return Extraction(None, "failed")


SCORING_VERSION = "spatial-worlds-scoring/1.0"


def extract_final_answer(raw: str) -> Extraction:
    """Formal protocol: parse only the LAST candidate, never repair or fall back.

    Unlike the historical extractor, an unclosed or malformed trailing object
    invalidates an earlier complete answer. Braces inside JSON strings do not
    start candidates. This selection is entirely independent of gold.
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
