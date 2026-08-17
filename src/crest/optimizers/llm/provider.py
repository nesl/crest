# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Provider-independent request/response records and deterministic fake."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class LLMRequest:
    """Provider-neutral structured JSON completion request."""

    system_prompt: str
    user_prompt: str
    prompt_version: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def messages(self) -> list[dict[str, str]]:
        """Return the OpenAI-compatible chat message shape."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt},
        ]


@dataclass(frozen=True)
class LLMResponse:
    """Normalized provider response with raw provenance."""

    content: str
    provider: str
    model: str
    raw_response: dict[str, Any]
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMProvider(Protocol):
    """Small synchronous boundary implemented by all MVP providers."""

    def complete_json(self, request: LLMRequest) -> LLMResponse:
        """Return one JSON completion for ``request``."""


class FakeProvider:
    """Return configured JSON responses without network access."""

    def __init__(
        self,
        responses: list[str | dict[str, Any]],
        *,
        model: str = "fake-model",
    ) -> None:
        """Initialize a finite response queue and request capture list."""
        self._responses = deque(responses)
        self.model = model
        self.requests: list[LLMRequest] = []

    def complete_json(self, request: LLMRequest) -> LLMResponse:
        """Return the next configured response and record the request."""
        self.requests.append(request)
        if not self._responses:
            raise RuntimeError("FakeProvider has no configured responses remaining.")
        configured = self._responses.popleft()
        content = configured if isinstance(configured, str) else json.dumps(configured)
        return LLMResponse(
            content=content,
            provider="fake",
            model=self.model,
            raw_response={"content": content},
            finish_reason="stop",
            metadata={"response_source": "configured_fixture"},
        )


class OpenAICompatibleProvider:
    """Synchronous JSON client for OpenRouter and OpenAI-compatible APIs."""

    def __init__(
        self,
        *,
        provider_name: str,
        base_url: str,
        api_key_env: str,
        model: str,
        temperature: float = 0.4,
        timeout_s: float = 60.0,
        extra_headers: dict[str, str] | None = None,
        urlopen: Any = urllib.request.urlopen,
    ) -> None:
        """Store provider settings without reading or persisting the API key."""
        self.provider_name = provider_name
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.model = model
        self.temperature = float(temperature)
        self.timeout_s = float(timeout_s)
        self.extra_headers = dict(extra_headers or {})
        self._urlopen = urlopen

    def build_http_request(self, request: LLMRequest) -> urllib.request.Request:
        """Construct the authenticated chat-completions request."""
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Required API key environment variable '{self.api_key_env}' is not set.")
        payload = {
            "model": self.model,
            "messages": request.messages(),
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }
        return urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers=headers,
            method="POST",
        )

    def complete_json(self, request: LLMRequest) -> LLMResponse:
        """Execute and normalize one OpenAI-compatible JSON completion."""
        http_request = self.build_http_request(request)
        started = time.monotonic()
        try:
            with self._urlopen(http_request, timeout=self.timeout_s) as response:
                raw_bytes = response.read()
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{self.provider_name} HTTP {exc.code}: {error_body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"{self.provider_name} request failed: {exc.reason}") from exc
        latency_ms = (time.monotonic() - started) * 1000.0
        try:
            raw_response = json.loads(raw_bytes.decode("utf-8"))
            choice = raw_response["choices"][0]
            content = choice["message"]["content"]
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"{self.provider_name} returned an invalid chat-completions response.") from exc
        if not isinstance(content, str):
            raise RuntimeError(f"{self.provider_name} returned non-string message content.")
        usage = raw_response.get("usage")
        return LLMResponse(
            content=content,
            provider=self.provider_name,
            model=str(raw_response.get("model", self.model)),
            raw_response=raw_response,
            usage=dict(usage) if isinstance(usage, dict) else None,
            finish_reason=choice.get("finish_reason"),
            latency_ms=latency_ms,
            metadata={
                "base_url": self.base_url,
                "temperature": self.temperature,
                "timeout_s": self.timeout_s,
            },
        )


def build_provider(config: Any) -> LLMProvider:
    """Build the configured provider while keeping choices out of the NAS loop."""
    getter = getattr(config, "get", None)

    def cfg_get(key: str, default: Any = None) -> Any:
        if callable(getter):
            return getter(key, default)
        return getattr(config, key, default)

    provider_name = str(cfg_get("provider", "")).strip().lower()
    if provider_name == "fake":
        return FakeProvider(list(cfg_get("responses", [])), model=str(cfg_get("model", "fake-model")))
    if provider_name in {"openrouter", "openai_compatible"}:
        return OpenAICompatibleProvider(
            provider_name=provider_name,
            base_url=str(cfg_get("base_url")),
            api_key_env=str(cfg_get("api_key_env")),
            model=str(cfg_get("model")),
            temperature=float(cfg_get("temperature", 0.4)),
            timeout_s=float(cfg_get("timeout_s", 60.0)),
            extra_headers=dict(cfg_get("extra_headers", {})),
        )
    raise ValueError(f"Unsupported LLM provider: {provider_name!r}.")
