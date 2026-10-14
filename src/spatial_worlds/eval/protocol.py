"""Versioned one-answer streaming protocol, independent of the legacy runner.

Only the main consumer mutates state or writes events. A daemon owns both
connection creation and reading; deadline cancellation never waits for socket
cleanup, and late events cannot change a locked answer. No gold is accepted.
"""

from __future__ import annotations

import json
import math
import queue
import re
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

PROTOCOL_VERSION = "spatial-worlds-evaluation/1.6"
MAX_TOKENS = 128000
DEADLINE_S = 7200
MAX_ATTEMPTS = 1
USAGE_GRACE_S = 5.0
RETRY_STATUSES = frozenset({408, 409, 429})


@dataclass(frozen=True)
class ProtocolContext:
    """Explicit, immutable resources; legacy calls retain their original identity."""

    version: str = PROTOCOL_VERSION
    max_tokens: int = MAX_TOKENS
    deadline_s: int = DEADLINE_S
    max_attempts: int = MAX_ATTEMPTS
    usage_grace_s: float = USAGE_GRACE_S

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_tokens, int)
            or isinstance(self.max_tokens, bool)
            or self.version != PROTOCOL_VERSION
            or self.max_tokens != MAX_TOKENS
            or self.deadline_s != DEADLINE_S
            or isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or self.usage_grace_s != USAGE_GRACE_S
        ):
            raise ValueError("unapproved evaluation protocol resources")


FORMAL128_PROTOCOL = ProtocolContext()


@dataclass
class AttemptResult:
    text: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    response: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None
    termination: str = "transport"
    answer_eligible: bool = False
    retryable: bool = False
    retry_after: float | None = None
    error: str | None = None
    latency_s: float = 0.0
    refusal: bool = False
    usage_complete: bool = False
    protocol_violation: str | None = None
    trace: dict[str, Any] | None = None
    trace_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if self.trace is None and self.trace_error is None:
            value.pop("trace")
            value.pop("trace_error")
        return value


def _get(obj: Any, key: str, default: Any = None) -> Any:
    return obj.get(key, default) if isinstance(obj, Mapping) else getattr(obj, key, default)


def _tokens(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def normalize_usage(raw: Any, provider: str) -> dict[str, Any]:
    """Combined generation tokens; cache fields are never added twice.

    Anthropic's input_tokens excludes cache reads/writes. Missing cache
    components leave TOTAL input unknown, rather than pretending they are 0.
    """
    if raw is None:
        return {}
    if provider == "anthropic":
        uncached = _tokens(_get(raw, "input_tokens"))
        read = _tokens(_get(raw, "cache_read_input_tokens"))
        write = _tokens(_get(raw, "cache_creation_input_tokens"))
        return {
            "input_tokens": (uncached + read + write)
            if uncached is not None and read is not None and write is not None
            else None,
            "uncached_input_tokens": uncached,
            "cached_input_tokens": read,
            "cache_write_input_tokens": write,
            "output_tokens": _tokens(_get(raw, "output_tokens")),
            "reasoning_tokens": _tokens(
                _get(_get(raw, "output_tokens_details"), "thinking_tokens")
            ),
        }
    cached = _get(_get(raw, "prompt_tokens_details"), "cached_tokens")
    if cached is None:
        cached = _get(raw, "prompt_cache_hit_tokens")
    return {
        "input_tokens": _tokens(_get(raw, "prompt_tokens")),
        "output_tokens": _tokens(_get(raw, "completion_tokens")),
        "reasoning_tokens": _tokens(
            _get(_get(raw, "completion_tokens_details"), "reasoning_tokens")
        ),
        "cached_input_tokens": _tokens(cached),
    }


def _plain(obj: Any) -> dict[str, Any]:
    if isinstance(obj, Mapping):
        return dict(obj)
    dump = getattr(obj, "model_dump", None)
    return dump(exclude_none=True) if dump else dict(vars(obj))


TRACE_LIMIT_BYTES = 16384
TRACE_ERRORS = frozenset({"trace_collection_failed", "trace_missing", "trace_oversized"})
_ERROR_CODES = frozenset(
    {
        "invalid_request_error",
        "unsupported_parameter",
        "unsupported_value",
        "invalid_value",
        "authentication_error",
        "rate_limit_exceeded",
        "insufficient_quota",
        "context_length_exceeded",
        "model_not_found",
        "permission_denied",
        "timeout",
        "server_error",
        "internal_error",
    }
)
_ERROR_PARAMS = frozenset(
    {
        "model",
        "reasoning_effort",
        "max_tokens",
        "max_completion_tokens",
        "stream",
        "stream_options",
        "messages",
        "output_config.effort",
        "thinking",
        "thinking.type",
    }
)


def _safe_response_identifier(value: Any) -> bool:
    return isinstance(value, str) and bool(
        re.fullmatch(
            r"(?:(?:chatcmpl-|msg_|resp_|req_|cmpl-)[A-Za-z0-9_.:-]{1,120}|[0-9]{1,30})", value
        )
    )


def _response_models(model: str) -> set[str]:
    values = {model, f"azure/{model}", f"vertex_ai/{model}"}
    if model == "claude-sonnet":
        values |= {"claude-sonnet-5", "vertex_ai/claude-sonnet-5"}
    return values


def _safe_response_fields(chunk: Any, model: str) -> tuple[dict, bool]:
    """Version 1.3 metadata only; unexpected fields never enter the durable event log."""
    result, invalid = {}, False
    for key in ("id", "model", "system_fingerprint"):
        value = _get(chunk, key)
        # The NAIRR Azure route can emit an empty bootstrap ID before its real ID.
        # Treat that placeholder as absent; never store it or erase a later ID.
        if value is None or (key == "id" and value == ""):
            continue
        allowed = (
            _safe_response_identifier(value)
            if key == "id"
            else isinstance(value, str) and value in _response_models(model)
            if key == "model"
            else isinstance(value, str) and bool(re.fullmatch(r"fp_[A-Za-z0-9_-]{1,120}", value))
        )
        if allowed:
            result[key] = value
        else:
            invalid = True
    return result, invalid


def safe_error_metadata(error: Exception) -> dict:
    """Never stringify an exception, message, arbitrary body, header or unknown token."""
    status = getattr(error, "status_code", None)
    body = getattr(error, "body", None)
    item = body.get("error", body) if isinstance(body, Mapping) else {}
    code = _get(item, "code", getattr(error, "code", None))
    kind = _get(item, "type", getattr(error, "type", None))
    param = _get(item, "param", getattr(error, "param", None))

    def safe(value: Any, allowed: frozenset[str]) -> str | None:
        return (
            None
            if value is None
            else value
            if isinstance(value, str) and value in allowed
            else "unrecognized"
        )

    return {
        "status": status
        if isinstance(status, int) and not isinstance(status, bool) and 100 <= status <= 599
        else None,
        "code": safe(code, _ERROR_CODES),
        "type": safe(kind, _ERROR_CODES),
        "param": safe(param, _ERROR_PARAMS),
    }


class _SafeTrace:
    def __init__(self, model: str, version: int = 1):
        if version not in (1, 2):
            raise ValueError("unknown trace version")
        self.model = model
        self.failed = False
        self.value: dict[str, Any] = {
            "version": version,
            "chunks": 0,
            "response_ids": [],
            "response_models": [],
            "finish_reasons": [],
            "content_types": {},
            "content_characters": 0,
            "refusal_present": False,
            "refusal_characters": 0,
            "reasoning_present": False,
            "alternative_content_present": False,
            "usage": {},
            "error": None,
        }
        if version == 2:
            self.value.update(
                upstream_stop_reasons=[],
                refusal_categories=[],
                stop_details_present=False,
                refusal_explanation_present=False,
            )

    def observe(self, chunk: Any) -> None:
        value = self.value
        value["chunks"] += 1
        ident = _get(chunk, "id")
        if _safe_response_identifier(ident):
            self._unique("response_ids", ident)
        model = _get(chunk, "model")
        allowed_models = _response_models(self.model)
        if isinstance(model, str) and model in allowed_models:
            self._unique("response_models", model)
        for key in ("message", "output", "output_text", "audio", "content"):
            value["alternative_content_present"] |= _get(chunk, key) is not None
        for choice in _get(chunk, "choices", []) or []:
            delta = _get(choice, "delta")
            content = _get(delta, "content")
            kind = (
                "none"
                if content is None
                else "str"
                if isinstance(content, str)
                else "list"
                if isinstance(content, list)
                else "dict"
                if isinstance(content, Mapping)
                else "other"
            )
            value["content_types"][kind] = value["content_types"].get(kind, 0) + 1
            if isinstance(content, str):
                value["content_characters"] += len(content)
            elif content is not None:
                value["alternative_content_present"] = True
            refusal = _get(delta, "refusal")
            value["refusal_present"] |= refusal is not None
            if isinstance(refusal, str):
                value["refusal_characters"] += len(refusal)
            for key in ("reasoning", "reasoning_content", "thinking", "reasoning_details"):
                value["reasoning_present"] |= _get(delta, key) is not None
            for key in ("message", "text", "output", "audio"):
                value["alternative_content_present"] |= _get(choice, key) is not None
            value["alternative_content_present"] |= _get(delta, "audio") is not None
            finish = _get(choice, "finish_reason")
            if finish is not None:
                self._unique(
                    "finish_reasons",
                    finish
                    if isinstance(finish, str)
                    and finish
                    in {
                        "stop",
                        "length",
                        "content_filter",
                        "refusal",
                        "tool_calls",
                        "function_call",
                    }
                    else "unrecognized",
                )
        usage = _get(chunk, "usage")
        if usage is not None:
            value["usage"] = normalize_usage(usage, "openai")
        if value["version"] == 2:
            containers = [chunk]
            for choice in (_get(chunk, "choices", []) or [])[:1]:
                containers.extend([choice, _get(choice, "delta")])
            for container in containers:
                self._observe_stop_details(container)
                self._observe_stop_details(_get(container, "provider_specific_fields"), native=True)

    def _observe_stop_details(self, container: Any, *, native: bool = False) -> None:
        # Never dump a provider object: it may carry thinking blocks, keys or arbitrary text.
        stops = {
            "end_turn",
            "max_tokens",
            "stop_sequence",
            "refusal",
            "content_filter",
            "model_context_window_exceeded",
            "tool_use",
            "pause_turn",
            "stop",
            "length",
        }
        for key in ("stop_reason", "finish_reason") if native else ("stop_reason",):
            stop = _get(container, key)
            if stop is not None:
                self._unique(
                    "upstream_stop_reasons",
                    stop if isinstance(stop, str) and stop in stops else "unrecognized",
                )
        details = _get(container, "stop_details")
        if details is not None:
            self.value["stop_details_present"] = True
            category = _get(details, "category")
            categories = {"cyber", "bio", "frontier_llm", "reasoning_extraction", "general_harms"}
            if category is not None:
                self._unique(
                    "refusal_categories",
                    category
                    if isinstance(category, str) and category in categories
                    else "unrecognized",
                )
            self.value["refusal_explanation_present"] |= _get(details, "explanation") is not None

    def _unique(self, key: str, item: str) -> None:
        if item not in self.value[key] and len(self.value[key]) < 4:
            self.value[key].append(item)

    def snapshot(self) -> dict:
        encoded = json.dumps(self.value, separators=(",", ":"), allow_nan=False)
        if len(encoded.encode()) > TRACE_LIMIT_BYTES:
            raise ValueError("trace_oversized")
        return json.loads(encoded)


def openai_events(
    client: Any,
    model: str,
    prompt: str,
    params: Mapping[str, Any],
    *,
    trace: bool = False,
    trace_version: int = 1,
) -> Iterator[dict]:
    observer = _SafeTrace(model, trace_version) if trace else None
    if observer is not None:
        yield {"kind": "trace_update", "value": observer.snapshot()}
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        stream_options={"include_usage": True},
        n=1,
        **params,
    )
    yield {"kind": "handle", "handle": stream}
    for chunk in stream:
        if observer is not None and not observer.failed:
            try:
                observer.observe(chunk)
                yield {"kind": "trace_update", "value": observer.snapshot()}
            except Exception:
                observer.failed = True
                yield {"kind": "trace_error", "value": "trace_collection_failed"}
        if trace:
            response, invalid_metadata = _safe_response_fields(chunk, model)
            if invalid_metadata:
                yield {"kind": "trace_error", "value": "trace_collection_failed"}
        else:
            response = {key: _get(chunk, key) for key in ("id", "model", "system_fingerprint")}
        yield {"kind": "response", "value": {k: v for k, v in response.items() if v is not None}}
        choices = _get(chunk, "choices", []) or []
        if len(choices) > 1 or any(_get(choice, "index", 0) != 0 for choice in choices):
            yield {"kind": "protocol_error", "value": "multiple_choices"}
            return
        for choice in choices:
            delta = _get(choice, "delta")
            content = _get(delta, "content")
            if content and (not trace or isinstance(content, str)):
                yield {"kind": "text", "value": content}
            if _get(delta, "refusal"):
                yield {"kind": "refusal"}
            if _get(delta, "tool_calls") or _get(delta, "function_call"):
                yield {"kind": "protocol_error", "value": "unexpected_tool_call"}
                return
            finish = _get(choice, "finish_reason")
            if finish:
                if trace and (
                    not isinstance(finish, str)
                    or finish
                    not in {
                        "stop",
                        "length",
                        "content_filter",
                        "refusal",
                        "tool_calls",
                        "function_call",
                    }
                ):
                    yield {"kind": "trace_error", "value": "trace_collection_failed"}
                else:
                    yield {"kind": "end", "value": finish}
        if _get(chunk, "usage") is not None:
            yield {
                "kind": "usage",
                "value": normalize_usage(_get(chunk, "usage"), "openai"),
                "final": True,
            }


def error_event(error: Exception) -> dict[str, Any]:
    """Do not persist arbitrary SDK exception strings (may include credentials)."""
    status = getattr(error, "status_code", None)
    retryable = status in RETRY_STATUSES or (isinstance(status, int) and 500 <= status < 600)
    if status is None:
        retryable = isinstance(error, (ConnectionError, TimeoutError, OSError)) or any(
            token in type(error).__name__ for token in ("Connection", "Timeout", "RateLimit")
        )
        retryable = retryable or any(
            cls.__module__.split(".")[0] in {"httpx", "httpx2", "httpcore"}
            and cls.__name__ in {"TransportError", "NetworkError", "ProtocolError"}
            for cls in type(error).__mro__
        )
    retry_after = None
    headers = getattr(getattr(error, "response", None), "headers", {}) or {}
    value = headers.get("retry-after")
    if value is not None:
        try:
            retry_after = float(value)
        except (ValueError, TypeError):
            try:
                retry_after = (
                    parsedate_to_datetime(str(value)) - datetime.now(UTC)
                ).total_seconds()
            except (ValueError, TypeError, OverflowError):
                pass
    if retry_after is not None and (not math.isfinite(retry_after) or retry_after < 0):
        retry_after = None
    return {
        "kind": "error",
        "value": f"{type(error).__name__}: status={status}",
        "retryable": retryable,
        "retry_after": retry_after,
        "termination": "transport" if status is None else "status",
    }


def consume_attempt(
    factory: Callable[[], Iterator[dict]],
    *,
    deadline_s: float = DEADLINE_S,
    usage_grace_s: float = USAGE_GRACE_S,
    on_event: Callable[[dict[str, Any]], None] = lambda _: None,
    trace_enabled: bool = False,
    should_cancel: Callable[[], bool] | None = None,
) -> AttemptResult:
    """One complete attempt: bounded connect/read/cleanup, no hidden retry."""
    if should_cancel is not None and should_cancel():
        result = AttemptResult(
            termination="cancelled",
            error="user_stop",
            protocol_violation="user_stop",
            trace_error="trace_missing" if trace_enabled else None,
        )
        on_event({"kind": "cancelled", "value": "user_stop"})
        return result
    started = time.monotonic()
    deadline = started + deadline_s
    events: queue.Queue[dict] = queue.Queue(maxsize=256)
    cancelled = threading.Event()
    handles: list[Any] = []

    def emit(event: dict) -> bool:
        while not cancelled.is_set():
            try:
                events.put(event, timeout=0.1)
                return True
            except queue.Full:
                pass
        return False

    def worker() -> None:
        try:
            for event in factory():
                if event["kind"] == "handle":
                    handle = event["handle"]
                    handles.append(handle)
                    if cancelled.is_set():
                        handle.close()
                        break
                elif not emit(event):
                    break
        except Exception as error:
            failure = error_event(error)
            if trace_enabled:
                try:
                    metadata = safe_error_metadata(error)
                    emit({"kind": "trace_error_metadata", "value": metadata})
                    failure["value"] = f"provider_error: status={metadata['status']}"
                except Exception:
                    emit({"kind": "trace_error", "value": "trace_collection_failed"})
                    failure["value"] = "provider_error"
            emit(failure)
        finally:
            emit({"kind": "eof"})

    def cleanup() -> None:
        for handle in handles:
            try:
                handle.close()
            except Exception:
                pass

    thread = threading.Thread(target=worker, daemon=True, name="study-stream")
    thread.start()
    result = AttemptResult()
    locked_at: float | None = None
    usage_after_end = False
    final_usage = False
    try:
        while True:
            if should_cancel is not None and should_cancel():
                result.termination = "cancelled"
                result.error = "user_stop"
                result.protocol_violation = "user_stop"
                on_event({"kind": "cancelled", "value": "user_stop"})
                break
            now = time.monotonic()
            cutoff = min(deadline, locked_at + usage_grace_s) if locked_at else deadline
            if now >= cutoff:
                if locked_at is None:
                    result.termination = "deadline"
                    result.answer_eligible = True
                    result.error = "whole_attempt_deadline"
                    on_event({"kind": "answer_locked", "value": result.to_dict()})
                break
            try:
                event = events.get(timeout=min(0.1, cutoff - now))
            except queue.Empty:
                continue
            kind = event["kind"]
            if kind == "trace_update":
                if trace_enabled:
                    result.trace = event["value"]
                continue  # Aggregated once below; never persist per-chunk trace copies.
            if kind == "trace_error_metadata":
                if trace_enabled:
                    if result.trace is None:
                        result.trace = _SafeTrace("").value
                    result.trace["error"] = event["value"]
                continue
            if kind == "trace_error":
                if trace_enabled:
                    result.trace_error = (
                        event["value"]
                        if event["value"] in TRACE_ERRORS
                        else "trace_collection_failed"
                    )
                continue
            if kind == "text" and locked_at is None:
                result.text += event["value"]
            elif kind == "refusal" and locked_at is None:
                result.refusal = True
            elif kind == "response":
                for key, value in event["value"].items():
                    old = result.response.get(key)
                    if key in {"id", "model"} and old and value and old != value:
                        result.protocol_violation = "response_identity_changed_within_attempt"
                        if locked_at is None:
                            result.error = result.protocol_violation
                            result.termination = "protocol"
                        on_event({"kind": "protocol_error", "value": result.protocol_violation})
                        return result
                    if value is not None:
                        if key not in {"id", "model"} or value:
                            result.response[key] = value
            elif kind == "usage":
                result.usage.update(event["value"])
                final_usage = bool(event.get("final"))
                usage_after_end = locked_at is not None
                result.usage_complete = final_usage and locked_at is not None
            elif kind == "end" and locked_at is None:
                if event["value"] not in {
                    "stop",
                    "length",
                    "max_tokens",
                    "end_turn",
                    "stop_sequence",
                    "refusal",
                    "content_filter",
                }:
                    result.termination = "protocol"
                    result.error = "unrecognized_finish_marker"
                    on_event({"kind": "protocol_error", "value": result.error})
                    break
                result.finish_reason = event["value"]
                result.refusal = result.refusal or result.finish_reason in {
                    "refusal",
                    "content_filter",
                }
                result.termination = (
                    "length" if result.finish_reason in {"length", "max_tokens"} else "stop"
                )
                result.answer_eligible = True
                locked_at = time.monotonic()
                result.usage_complete = final_usage
                on_event({"kind": "answer_locked", "value": result.to_dict()})
            elif kind in {"error", "transport_error", "protocol_error", "eof"}:
                if kind == "protocol_error":
                    result.protocol_violation = event.get("value", "invalid_stream")
                if locked_at is None:
                    result.termination = (
                        "protocol"
                        if kind == "protocol_error"
                        else event.get("termination", "transport")
                    )
                    result.error = event.get("value", "stream_ended_without_finish_marker")
                    result.retryable = event.get("retryable", kind in {"eof", "transport_error"})
                    result.retry_after = event.get("retry_after")
                on_event(event)
                break
            on_event(event)
            if locked_at is not None and usage_after_end:
                break
    finally:
        result.latency_s = time.monotonic() - started
        cancelled.set()
        threading.Thread(target=cleanup, daemon=True, name="study-stream-close").start()
        if trace_enabled:
            result.usage_complete = result.usage_complete and all(
                _tokens(result.usage.get(key)) is not None
                for key in ("input_tokens", "output_tokens")
            )
            if result.trace is None and result.trace_error is None:
                result.trace_error = "trace_missing"
            if result.trace is not None:
                result.trace["status"] = (
                    "complete"
                    if result.trace_error is None
                    and (result.usage_complete or result.trace.get("error") is not None)
                    else "partial"
                )
                try:
                    if len(json.dumps(result.trace, allow_nan=False).encode()) > TRACE_LIMIT_BYTES:
                        result.trace = None
                        result.trace_error = "trace_oversized"
                    else:
                        on_event({"kind": "trace_summary", "value": result.trace})
                except Exception:
                    result.trace_error = "trace_collection_failed"
            if result.trace_error:
                on_event({"kind": "trace_error", "value": result.trace_error})
    return result
