"""Model providers: Anthropic, OpenAI-compatible, and deterministic baselines.

SDKs are imported lazily inside the provider constructors so the core package
stays dependency-free.  Providers return a :class:`Completion`; transient
failures raise :class:`RetryableError` so the runner can back off and retry.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlparse

from ..diagnostics import DEFAULT_SUFFIX_LENGTHS, suffix_state
from ..prompt import render_answer, render_query_answer
from ..schema import GoldRecord, Instance
from ..world import WorldState

# Failure classes recorded in ``failure_kind``. Budget and deadline records are
# excluded from ``--retry-errors`` by default as a policy, not because repeating
# a request is guaranteed to reproduce the failure.
FAILURE_KINDS = ("budget", "deadline", "empty", "refusal", "status", "transport")
# The historical constant name is kept for import compatibility; it denotes
# default retry exclusions, not a causal claim of deterministic failure.
DETERMINISTIC_FAILURES = frozenset({"budget", "deadline"})
NO_ANSWER_FAILURES = DETERMINISTIC_FAILURES


class RetryableError(RuntimeError):
    """A transient provider failure (rate limit, overload, network)."""

    def __init__(
        self, message: str, retry_after: float | None = None, *, kind: str = "transport"
    ) -> None:
        super().__init__(message)
        self.retry_after = retry_after
        self.kind = kind


@dataclass(slots=True)
class Completion:
    text: str | None
    usage: dict[str, int | None] = field(default_factory=dict)
    finish_reason: str | None = None
    latency_s: float = 0.0
    error: str | None = None
    failure_kind: str | None = None
    raw: Any = None


def _no_text_completion(
    usage: dict[str, int | None], finish_reason: str | None, latency: float
) -> Completion:
    """Completion for a response that carried no answer text."""

    if finish_reason in ("length", "max_tokens"):
        return Completion(
            text=None,
            usage=usage,
            finish_reason=finish_reason,
            latency_s=latency,
            error="budget exhausted before an answer",
            failure_kind="budget",
        )
    return Completion(
        text=None,
        usage=usage,
        finish_reason=finish_reason,
        latency_s=latency,
        error="empty response",
        failure_kind="empty",
    )


class Provider(Protocol):
    name: str
    model: str

    def complete(
        self, prompt: str, *, instance: Instance, gold: GoldRecord | None
    ) -> Completion: ...

    def request_params(self) -> dict[str, Any]: ...


def _answer_for(instance: Instance, state: Any) -> str:
    if instance.queries:
        return render_query_answer(state, instance.queries)
    return render_answer(state)


class CopyInitialProvider:
    """Baseline that returns the initial world unchanged."""

    name = "copy-initial"

    def __init__(self, model: str | None = None) -> None:
        self.model = model or "copy-initial"

    def complete(self, prompt: str, *, instance: Instance, gold: GoldRecord | None) -> Completion:
        del prompt, gold
        return Completion(text=_answer_for(instance, instance.initial), finish_reason="baseline")

    def request_params(self) -> dict[str, Any]:
        return {}


class AllZeroProvider(CopyInitialProvider):
    """Sparse-state control: return zero at every coordinate, without reading gold."""

    name = "all-zero"

    def __init__(self, model: str | None = None) -> None:
        self.model = model or self.name

    def complete(self, prompt: str, *, instance: Instance, gold: GoldRecord | None) -> Completion:
        del prompt, gold
        return Completion(
            text=_answer_for(instance, WorldState.empty(instance.world)), finish_reason="baseline"
        )


class OracleProvider:
    """Baseline that returns the gold answer; validates the whole pipeline."""

    name = "oracle"

    def __init__(self, model: str | None = None) -> None:
        self.model = model or "oracle"

    def complete(self, prompt: str, *, instance: Instance, gold: GoldRecord | None) -> Completion:
        del prompt
        if gold is None:
            return Completion(text=None, error="oracle provider requires gold records")
        return Completion(text=_answer_for(instance, gold.final_state), finish_reason="oracle")

    def request_params(self) -> dict[str, Any]:
        return {}


def parse_suffix_length(model: str | None, default: int = DEFAULT_SUFFIX_LENGTHS[2]) -> int:
    """The ``k`` of a ``suffix-k`` model name (``None`` or ``suffix`` means the default)."""

    if model is None or model == "suffix":
        return default
    prefix, _, tail = model.partition("-")
    if prefix != "suffix" or not tail.isdigit():
        raise ValueError(f"suffix provider models are named suffix-K, not {model!r}")
    return int(tail)


class SuffixProvider:
    """Baseline that executes every SET but only the core events among the last k positions.

    It is the approximate solver of the history-retention diagnostics
    (``spatial_worlds.diagnostics``): its exact-match rate on a cell shows how
    much of the final state the recent events alone determine.
    """

    name = "suffix"

    def __init__(self, model: str | None = None) -> None:
        self.k = parse_suffix_length(model)
        self.model = f"suffix-{self.k}"

    def complete(self, prompt: str, *, instance: Instance, gold: GoldRecord | None) -> Completion:
        del prompt, gold
        return Completion(
            text=_answer_for(instance, suffix_state(instance, self.k)), finish_reason="baseline"
        )

    def request_params(self) -> dict[str, Any]:
        return {"suffix_length": self.k}


def _parse_extra(params: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(params) if params else {}


class _StreamOutcome:
    """What the stream-consuming thread collected, plus its termination state."""

    __slots__ = ("text_parts", "finish_reason", "usage", "error", "cancelled", "done")

    def __init__(self) -> None:
        self.text_parts: list[str] = []
        self.finish_reason: str | None = None
        self.usage: Any = None
        self.error: BaseException | None = None
        self.cancelled = False
        self.done = threading.Event()


def _consume_stream(stream: Any, outcome: _StreamOutcome) -> None:
    try:
        for chunk in stream:
            if outcome.cancelled:
                break
            if getattr(chunk, "usage", None) is not None:
                outcome.usage = chunk.usage
            for choice in chunk.choices or ():
                content = getattr(choice.delta, "content", None) if choice.delta else None
                if content:
                    outcome.text_parts.append(content)
                if choice.finish_reason:
                    outcome.finish_reason = choice.finish_reason
    except BaseException as error:  # re-raised in the requesting thread
        outcome.error = error
    finally:
        outcome.done.set()


def _close_quietly(stream: Any) -> None:
    try:
        stream.close()
    except Exception:  # pragma: no cover - closing a broken stream must not mask the deadline
        pass


def consume_with_deadline(stream: Any, started: float, deadline_s: float) -> _StreamOutcome:
    """Read a streamed completion, enforcing a wall-clock deadline measured from ``started``.

    The stream is consumed in a helper thread so that the deadline holds even
    while the iterator blocks on a socket that a proxy keeps alive with
    heartbeats; on expiry the stream is closed and a ``deadline`` failure is
    raised regardless of whether any chunk ever arrived.
    """

    outcome = _StreamOutcome()
    worker = threading.Thread(
        target=_consume_stream, args=(stream, outcome), name="spatial-worlds-stream", daemon=True
    )
    worker.start()
    remaining = deadline_s - (time.monotonic() - started)
    if not outcome.done.wait(timeout=max(0.0, remaining)):
        outcome.cancelled = True
        _close_quietly(stream)
        raise RetryableError(f"transient: no completion within {deadline_s:.0f}s", kind="deadline")
    if outcome.error is not None:
        raise outcome.error
    return outcome


class OpenAICompatibleProvider:
    """Chat Completions against OpenAI or any OpenAI-compatible server.

    Responses are streamed so that long reasoning generations keep the
    connection alive; the text deltas are concatenated and usage is read from
    the final chunk (``stream_options.include_usage``).  ``timeout`` doubles as
    a wall-clock deadline for the whole completion because heartbeat frames from
    a proxy would otherwise defeat the SDK's read timeout.
    """

    name = "openai"

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key_env: str | None = None,
        max_tokens: int | None = 128000,
        effort: str | None = "high",
        temperature: float | None = None,
        timeout: float = 7200.0,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            import openai  # pyright: ignore[reportMissingImports]
        except ImportError as error:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                "install the 'openai' extra: pip install 'spatial-worlds[eval]'"
            ) from error
        self._openai = openai
        key_env = api_key_env or "OPENAI_API_KEY"
        api_key = os.environ.get(key_env)
        if api_key is None and base_url is None:
            raise RuntimeError(f"environment variable {key_env} is not set")
        self.model = model
        self.base_url = base_url
        self.client = openai.OpenAI(
            base_url=base_url, api_key=api_key or "EMPTY", timeout=timeout, max_retries=0
        )
        self.deadline_s = timeout
        self.params: dict[str, Any] = {}
        official = base_url is None or urlparse(base_url).hostname == "api.openai.com"
        if max_tokens is not None:
            self.params["max_completion_tokens" if official else "max_tokens"] = max_tokens
        if effort is not None:
            self.params["reasoning_effort"] = effort
        if temperature is not None:
            self.params["temperature"] = temperature
        self.params.update(_parse_extra(extra))

    def request_params(self) -> dict[str, Any]:
        # The wall-clock deadline is part of the resource policy and therefore
        # of the condition identity: two runs that differ only in how long they
        # wait can differ in their failure rates.
        return {"base_url": self.base_url, **self.params, "timeout_s": self.deadline_s}

    def complete(self, prompt: str, *, instance: Instance, gold: GoldRecord | None) -> Completion:
        del instance, gold
        openai = self._openai
        started = time.monotonic()
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                stream_options={"include_usage": True},
                **self.params,
            )
            # Proxies keep idle streams alive with heartbeats, so the SDK read
            # timeout never fires on a stalled generation; the deadline is
            # enforced from outside the iterator instead.
            outcome = consume_with_deadline(stream, started, self.deadline_s)
            text_parts = outcome.text_parts
            finish_reason = outcome.finish_reason
            usage_obj: Any = outcome.usage
        except openai.RateLimitError as error:
            raise RetryableError(f"rate limited: {error}", _retry_after(error)) from error
        except openai.APITimeoutError as error:
            # The SDK timeout fires only when no bytes at all arrive within the
            # deadline, the same stall the streaming deadline above catches.
            raise RetryableError(f"transient: {error}", kind="deadline") from error
        except (openai.APIConnectionError, openai.InternalServerError) as error:
            raise RetryableError(f"transient: {error}") from error
        except openai.APIStatusError as error:
            if error.status_code in (408, 409, 429) or error.status_code >= 500:
                raise RetryableError(f"status {error.status_code}: {error}") from error
            return Completion(
                text=None,
                error=f"status {error.status_code}: {error}",
                latency_s=time.monotonic() - started,
                failure_kind="status",
            )
        latency = time.monotonic() - started
        text = "".join(text_parts) if text_parts else None
        usage: dict[str, int | None] = {}
        if usage_obj is not None:
            details = getattr(usage_obj, "completion_tokens_details", None)
            cached = getattr(
                getattr(usage_obj, "prompt_tokens_details", None), "cached_tokens", None
            )
            if cached is None:  # DeepSeek reports cache hits as a top-level field
                cached = getattr(usage_obj, "prompt_cache_hit_tokens", None)
            usage = {
                "input_tokens": usage_obj.prompt_tokens,
                "output_tokens": usage_obj.completion_tokens,
                "reasoning_tokens": getattr(details, "reasoning_tokens", None) if details else None,
                "cached_input_tokens": cached,
            }
        if text is None:
            return _no_text_completion(usage, finish_reason, latency)
        return Completion(text=text, usage=usage, finish_reason=finish_reason, latency_s=latency)


class AnthropicProvider:
    """Claude via the Anthropic Messages API (streaming, adaptive thinking)."""

    name = "anthropic"

    def __init__(
        self,
        model: str,
        *,
        api_key_env: str | None = None,
        max_tokens: int | None = 128000,
        effort: str | None = "high",
        thinking_budget: int | None = None,
        temperature: float | None = None,
        timeout: float = 7200.0,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            import anthropic  # pyright: ignore[reportMissingImports]
        except ImportError as error:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                "install the 'anthropic' extra: pip install 'spatial-worlds[eval]'"
            ) from error
        self._anthropic = anthropic
        api_key = os.environ.get(api_key_env) if api_key_env else None
        if api_key_env is not None and api_key is None:
            raise RuntimeError(f"environment variable {api_key_env} is not set")
        self.client = anthropic.Anthropic(api_key=api_key, timeout=timeout, max_retries=0)
        self.model = model
        self.timeout_s = timeout
        self.max_tokens = max_tokens or 128000
        self.params: dict[str, Any] = {}
        if thinking_budget is not None:
            self.params["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
        else:
            self.params["thinking"] = {"type": "adaptive"}
        if effort is not None:
            self.params["output_config"] = {"effort": effort}
        if temperature is not None:
            self.params["temperature"] = temperature
        self.params.update(_parse_extra(extra))

    def request_params(self) -> dict[str, Any]:
        return {"max_tokens": self.max_tokens, **self.params, "timeout_s": self.timeout_s}

    def complete(self, prompt: str, *, instance: Instance, gold: GoldRecord | None) -> Completion:
        del instance, gold
        anthropic = self._anthropic
        started = time.monotonic()
        try:
            with self.client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
                **self.params,
            ) as stream:
                message = stream.get_final_message()
        except anthropic.RateLimitError as error:
            raise RetryableError(f"rate limited: {error}", _retry_after(error)) from error
        except anthropic.APITimeoutError as error:
            raise RetryableError(f"transient: {error}", kind="deadline") from error
        except anthropic.APIConnectionError as error:
            raise RetryableError(f"transient: {error}") from error
        except anthropic.APIStatusError as error:
            if error.status_code in (408, 409, 429, 529) or error.status_code >= 500:
                raise RetryableError(f"status {error.status_code}: {error}") from error
            return Completion(
                text=None,
                error=f"status {error.status_code}: {error}",
                latency_s=time.monotonic() - started,
                failure_kind="status",
            )
        latency = time.monotonic() - started
        text = "".join(
            str(getattr(block, "text", ""))
            for block in message.content
            if getattr(block, "type", None) == "text"
        )
        details_tokens = getattr(message.usage, "output_tokens_details", None)
        usage = {
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
            "reasoning_tokens": getattr(details_tokens, "thinking_tokens", None),
            "cached_input_tokens": getattr(message.usage, "cache_read_input_tokens", None),
        }
        if message.stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            return Completion(
                text=text if text else None,
                usage=usage,
                finish_reason=message.stop_reason,
                latency_s=latency,
                error=f"refusal: {getattr(details, 'category', None)}",
                failure_kind="refusal",
            )
        if message.stop_reason == "max_tokens":
            # Termination and answer validity are separate. The scorer can
            # accept a complete answer even when the message hit max_tokens.
            return Completion(
                text=text if text else None,
                usage=usage,
                finish_reason=message.stop_reason,
                latency_s=latency,
                error="budget exhausted before an answer",
                failure_kind="budget",
            )
        if not text:
            return _no_text_completion(usage, message.stop_reason, latency)
        return Completion(
            text=text, usage=usage, finish_reason=message.stop_reason, latency_s=latency
        )


def _retry_after(error: Any) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get("retry-after")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def make_provider(
    kind: str,
    model: str | None,
    *,
    base_url: str | None = None,
    api_key_env: str | None = None,
    max_tokens: int | None = 128000,
    effort: str | None = "high",
    thinking_budget: int | None = None,
    temperature: float | None = None,
    timeout: float = 7200.0,
    extra: Mapping[str, Any] | None = None,
) -> Provider:
    if kind == "copy-initial":
        return CopyInitialProvider(model)
    if kind == "all-zero":
        return AllZeroProvider(model)
    if kind == "oracle":
        return OracleProvider(model)
    if kind == "suffix":
        return SuffixProvider(model)
    if model is None:
        raise ValueError(f"--model is required for provider {kind!r}")
    if kind == "openai":
        return OpenAICompatibleProvider(
            model,
            base_url=base_url,
            api_key_env=api_key_env,
            max_tokens=max_tokens,
            effort=effort,
            temperature=temperature,
            timeout=timeout,
            extra=extra,
        )
    if kind == "anthropic":
        return AnthropicProvider(
            model,
            api_key_env=api_key_env,
            max_tokens=max_tokens,
            effort=effort,
            thinking_budget=thinking_budget,
            temperature=temperature,
            timeout=timeout,
            extra=extra,
        )
    raise ValueError(f"unknown provider {kind!r}")
