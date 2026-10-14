"""Minimal NAIRR gateway access used by the active formal runner."""

from __future__ import annotations

import json
import math
import os
import shlex
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .study_final_store import FormalRunError, timestamp


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _environment(root: Path, config: Mapping[str, Any]) -> dict[str, str]:
    route = config["routes"]["gateway"]
    wanted = {route["base_url_env"], route["api_key_env"]}
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


def _endpoint(root: Path, config: Mapping[str, Any]) -> tuple[str, str]:
    route = config["routes"]["gateway"]
    env = _environment(root, config)
    base = env.get(route["base_url_env"], "").rstrip("/")
    key = env.get(route["api_key_env"], "")
    parsed = urlparse(base)
    allowed_scheme = parsed.scheme == "https" or (
        parsed.scheme == "http" and config.get("allow_http_endpoint") is True
    )
    if (
        not allowed_scheme
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise FormalRunError("configured gateway endpoint is missing or unsafe")
    if not key:
        raise FormalRunError("gateway credential is missing")
    return base, key


class Gateway:
    def __init__(self, root: Path, config: Mapping[str, Any]) -> None:
        self.config = config
        self.base_url, self._key = _endpoint(root, config)
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())

    def _url(self, path: str) -> str:
        if path in {"/model/info", "/key/info"} and self.base_url.endswith("/v1"):
            return self.base_url[:-3] + path
        return self.base_url + path

    def _fetch(self, path: str) -> tuple[dict[str, Any], str | None]:
        last_status = None
        for attempt in range(3):
            request = urllib.request.Request(
                self._url(path), headers={"Authorization": f"Bearer {self._key}"}
            )
            try:
                with self._opener.open(request, timeout=20) as response:
                    payload = response.read(2_000_001)
                    version = response.headers.get("x-litellm-version")
                if len(payload) > 2_000_000:
                    raise FormalRunError("gateway metadata response is too large")
                value = json.loads(payload)
                if not isinstance(value, dict):
                    raise FormalRunError("gateway metadata is not a JSON object")
                return value, version
            except (urllib.error.URLError, OSError, json.JSONDecodeError) as error:
                last_status = getattr(error, "code", None)
                if attempt < 2:
                    time.sleep((2, 5)[attempt])
        raise FormalRunError(f"gateway metadata unavailable (status={last_status})")

    @staticmethod
    def _account(document: Mapping[str, Any]) -> dict[str, Any]:
        info = document.get("info") or {}
        spend, maximum = info.get("spend"), info.get("max_budget")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in (spend, maximum)
        ):
            raise FormalRunError("gateway account balance is unavailable")
        assert isinstance(spend, (int, float))
        assert isinstance(maximum, (int, float))
        if info.get("blocked") is True or maximum < spend:
            raise FormalRunError("gateway account is blocked or over budget")
        return {
            "checked_at": timestamp(),
            "spend_usd": float(spend),
            "max_budget_usd": float(maximum),
            "remaining_usd": float(maximum - spend),
            "budget_reset_at": info.get("budget_reset_at"),
        }

    def account(self) -> dict[str, Any]:
        document, _ = self._fetch("/key/info")
        return self._account(document)

    def inspect(self) -> dict[str, Any]:
        listed_document, listed_version = self._fetch("/models")
        info_document, info_version = self._fetch("/model/info")
        account_document, account_version = self._fetch("/key/info")
        listed = {
            item.get("id") for item in listed_document.get("data", []) if isinstance(item, Mapping)
        }
        entries = {}
        for entry in info_document.get("data", []):
            if not isinstance(entry, Mapping):
                continue
            name = entry.get("model_name") or entry.get("id")
            if name in {model["id"] for model in self.config["models"]}:
                entries[name] = entry
        pricing = {}
        models = []
        for model in self.config["models"]:
            name = model["id"]
            if name not in listed or name not in entries:
                raise FormalRunError(f"{name} is not listed by the gateway")
            entry = entries[name]
            metadata = entry.get("model_info") or {}
            params = entry.get("litellm_params") or {}
            output_limit = metadata.get("max_output_tokens") or metadata.get("max_tokens")
            if (
                params.get("model") != model["expected_upstream_model"]
                or not isinstance(output_limit, int)
                or output_limit < 128000
                or metadata.get("supports_reasoning") is not True
            ):
                raise FormalRunError(f"{name} mapping or 128K reasoning support changed")
            input_limit = metadata.get("max_input_tokens")
            input_cost = metadata.get("input_cost_per_token")
            output_cost = metadata.get("output_cost_per_token")
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
                for value in (input_limit, input_cost, output_cost)
            ):
                raise FormalRunError(f"{name} pricing or input limit is unavailable")
            assert isinstance(input_limit, (int, float))
            assert isinstance(input_cost, (int, float))
            assert isinstance(output_cost, (int, float))
            pricing[name] = {
                "max_input_tokens": int(input_limit),
                "input_per_million": float(input_cost) * 1_000_000,
                "output_per_million": float(output_cost) * 1_000_000,
            }
            models.append(
                {
                    "requested_model": name,
                    "upstream_model": params.get("model"),
                    "max_input_tokens": input_limit,
                    "max_output_tokens": output_limit,
                    **pricing[name],
                }
            )
        versions = {value for value in (listed_version, info_version, account_version) if value}
        return {
            "checked_at": timestamp(),
            "gateway_versions": sorted(versions),
            "models": models,
            "pricing": pricing,
            "account": self._account(account_document),
        }

    def clients(self) -> dict[str, Any]:
        import openai

        return {
            model["id"]: openai.OpenAI(
                api_key=self._key,
                base_url=self.base_url,
                timeout=float(self.config["resources"]["deadline_s"]),
                max_retries=0,
                http_client=openai.DefaultHttpxClient(follow_redirects=False, trust_env=False),
            )
            for model in self.config["models"]
        }


def reservation(pricing: Mapping[str, Any], max_tokens: int) -> int:
    return math.ceil(
        float(pricing["max_input_tokens"]) * float(pricing["input_per_million"])
        + max_tokens * float(pricing["output_per_million"])
    )
