"""Native AWS Bedrock streaming access for formal and diagnostic collection."""

from __future__ import annotations

import os
import shlex
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from .study_final_store import FormalRunError, timestamp


def _environment(root: Path, config: Mapping[str, Any]) -> dict[str, str]:
    route = config["routes"]["bedrock"]
    wanted = {route["bearer_token_env"], "AWS_REGION", "AWS_DEFAULT_REGION"}
    values: dict[str, str] = {}
    path = root / ".env"
    if path.exists():
        for number, line in enumerate(path.read_text().splitlines(), 1):
            line = line.strip()
            if line.startswith("export "):
                line = line[7:].lstrip()
            key, separator, raw = line.partition("=")
            key = key.strip()
            if separator and key in wanted:
                tokens = shlex.split(raw, comments=True, posix=True)
                if len(tokens) > 1:
                    raise FormalRunError(f".env line {number}: required value must be quoted")
                values[key] = tokens[0] if tokens else ""
    values.update({key: os.environ[key] for key in wanted if key in os.environ})
    return values


class BedrockGateway:
    """Small provider facade matching the formal scheduler's gateway contract."""

    def __init__(self, root: Path, config: Mapping[str, Any]) -> None:
        self.config = config
        route = config["routes"]["bedrock"]
        values = _environment(root, config)
        token_name = str(route["bearer_token_env"])
        token = values.get(token_name, "")
        if not token:
            raise FormalRunError("AWS Bedrock bearer credential is missing")
        self.region = str(
            values.get("AWS_REGION")
            or values.get("AWS_DEFAULT_REGION")
            or route["region"]
        )
        if self.region != route["region"]:
            raise FormalRunError("AWS region differs from the frozen formal route")
        # Botocore's Bedrock bearer-token provider reads this service-specific
        # variable. Loading it from .env keeps credentials out of configs/artifacts.
        os.environ[token_name] = token
        self.base_url = f"https://bedrock-runtime.{self.region}.amazonaws.com"

    def account(self) -> dict[str, Any]:
        route = self.config["routes"]["bedrock"]
        prior = float(route["known_prior_spend_usd"])
        maximum = float(route["monthly_spend_stop_usd"])
        return {
            "checked_at": timestamp(),
            "spend_usd": prior,
            "max_budget_usd": maximum,
            "remaining_usd": maximum - prior,
            "budget_reset_at": route["budget_reset_at"],
            "source": "configured_spend_stop_with_known_project_usage_only",
            "billing_lag_warning": True,
        }

    def inspect(self) -> dict[str, Any]:
        pricing: dict[str, dict[str, Any]] = {}
        models = []
        for model in self.config["models"]:
            if model.get("provider") != "bedrock" or model.get("route") != "bedrock":
                raise FormalRunError("AWS formal model has a different provider route")
            item = {
                "max_input_tokens": int(model["max_input_tokens"]),
                "input_per_million": float(model["input_per_million"]),
                "output_per_million": float(model["output_per_million"]),
            }
            pricing[str(model["id"])] = item
            models.append(
                {
                    "requested_model": model["id"],
                    "upstream_model": model["id"],
                    "max_output_tokens_requested": self.config["resources"]["max_tokens"],
                    **item,
                }
            )
        return {
            "checked_at": timestamp(),
            "provider": "aws-bedrock-converse-stream",
            "region": self.region,
            "models": models,
            "pricing": pricing,
            "account": self.account(),
        }

    def clients(self) -> dict[str, Any]:
        try:
            import boto3
            from botocore.config import Config
        except ImportError as error:  # pragma: no cover - exercised by deployment checks
            raise FormalRunError("install the eval extra to use AWS Bedrock") from error
        deadline = int(self.config["resources"]["deadline_s"])
        total = int(self.config["dispatch"]["promoted_total_concurrency"])
        sdk_config = Config(
            connect_timeout=20,
            read_timeout=deadline,
            max_pool_connections=max(4, total),
            retries={"total_max_attempts": 1, "mode": "standard"},
        )
        return {
            model["id"]: boto3.client(
                "bedrock-runtime", region_name=self.region, config=sdk_config
            )
            for model in self.config["models"]
        }


def _request_id(response: Mapping[str, Any]) -> str:
    metadata = response.get("ResponseMetadata") or {}
    value = metadata.get("RequestId")
    if not isinstance(value, str) or not value or len(value) > 120:
        raise FormalRunError("Bedrock response did not contain a safe request id")
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-"
    if any(character not in allowed for character in value):
        raise FormalRunError("Bedrock response request id has an unexpected form")
    return f"req_{value}"


def _finish_reason(value: Any) -> tuple[str | None, bool]:
    if value in {"end_turn", "stop_sequence", "max_tokens"}:
        return str(value), False
    if value in {"content_filtered", "guardrail_intervened"}:
        return "content_filter", True
    if value == "model_context_window_exceeded":
        return "max_tokens", False
    return None, False


def bedrock_events(
    client: Any,
    model: str,
    prompt: str,
    params: Mapping[str, Any],
    *,
    trace: bool = False,
) -> Iterator[dict[str, Any]]:
    """Translate ConverseStream events into the provider-neutral formal protocol.

    Reasoning blocks are counted but never copied into the answer or durable trace.
    """

    max_tokens = params.get("max_tokens")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool):
        raise FormalRunError("Bedrock max_tokens is invalid")
    request: dict[str, Any] = {
        "modelId": model,
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"maxTokens": max_tokens},
    }
    additional = {key: value for key, value in params.items() if key != "max_tokens"}
    if additional:
        request["additionalModelRequestFields"] = additional

    trace_value: dict[str, Any] = {
        "version": 2,
        "provider": "aws-bedrock-converse-stream",
        "events": 0,
        "content_characters": 0,
        "reasoning_characters": 0,
        "reasoning_present": False,
        "first_event_ms": None,
        "first_reasoning_ms": None,
        "first_text_ms": None,
        "upstream_stop_reasons": [],
        "usage": {},
        "error": None,
    }
    if trace:
        yield {"kind": "trace_update", "value": dict(trace_value)}
    started = time.monotonic()
    response = client.converse_stream(**request)
    yield {"kind": "response", "value": {"id": _request_id(response), "model": model}}
    stream = response.get("stream")
    if stream is None:
        raise FormalRunError("Bedrock response did not contain an event stream")
    yield {"kind": "handle", "handle": stream}

    for event in stream:
        elapsed_ms = round((time.monotonic() - started) * 1000)
        trace_value["events"] += 1
        if trace_value["first_event_ms"] is None:
            trace_value["first_event_ms"] = elapsed_ms
        if "contentBlockDelta" in event:
            delta = event["contentBlockDelta"].get("delta") or {}
            text = delta.get("text")
            if isinstance(text, str) and text:
                trace_value["content_characters"] += len(text)
                if trace_value["first_text_ms"] is None:
                    trace_value["first_text_ms"] = elapsed_ms
                yield {"kind": "text", "value": text}
            reasoning = delta.get("reasoningContent")
            if isinstance(reasoning, Mapping):
                reasoning_text = reasoning.get("text")
                if isinstance(reasoning_text, str):
                    trace_value["reasoning_characters"] += len(reasoning_text)
                    trace_value["reasoning_present"] = True
                    if trace_value["first_reasoning_ms"] is None:
                        trace_value["first_reasoning_ms"] = elapsed_ms
        elif "messageStop" in event:
            upstream = event["messageStop"].get("stopReason")
            if isinstance(upstream, str):
                trace_value["upstream_stop_reasons"] = [upstream]
            finish, refusal = _finish_reason(upstream)
            if finish is None:
                yield {"kind": "protocol_error", "value": "unrecognized_bedrock_stop_reason"}
                return
            if refusal:
                yield {"kind": "refusal"}
            if trace:
                yield {"kind": "trace_update", "value": dict(trace_value)}
            yield {"kind": "end", "value": finish}
        elif "metadata" in event:
            usage = event["metadata"].get("usage") or {}
            normalized = {
                "input_tokens": usage.get("inputTokens"),
                "output_tokens": usage.get("outputTokens"),
                "reasoning_tokens": None,
                "cached_input_tokens": usage.get("cacheReadInputTokens"),
            }
            trace_value["usage"] = dict(normalized)
            if trace:
                yield {"kind": "trace_update", "value": dict(trace_value)}
            yield {"kind": "usage", "value": normalized, "final": True}
        elif any(str(key).endswith("Exception") for key in event):
            yield {
                "kind": "error",
                "value": "bedrock_stream_exception",
                "retryable": True,
                "termination": "transport",
            }
            return

