# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Provider-independent request/response records and deterministic fake."""

from __future__ import annotations

import http.client
import json
import math
import os
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from datetime import timezone
from email.utils import parsedate_to_datetime
from typing import Any, Protocol

from ...component_selection import cfg_get


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


class ProviderError(RuntimeError):
    """Base provider failure; only response-content errors allow repair."""


class ProviderAuthenticationError(ProviderError):
    """Provider credentials or credits require operator intervention."""


class ProviderTransportError(ProviderError):
    """A provider request failed after its bounded transport retry policy."""


class ProviderResponseError(ProviderError):
    """The provider returned a malformed completion envelope; repair is allowed."""


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
        json_response_mode: bool = True,
        extra_headers: dict[str, str] | None = None,
        urlopen: Any = urllib.request.urlopen,
        max_transport_retries: int = 5,
        retry_backoff_s: float = 5.0,
        retry_backoff_cap_s: float = 60.0,
        sleep: Any = time.sleep,
        wall_clock: Any = time.time,
    ) -> None:
        """Store provider settings without reading or persisting the API key."""
        self.provider_name = provider_name
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.model = model
        self.temperature = float(temperature)
        self.timeout_s = float(timeout_s)
        self.json_response_mode = bool(json_response_mode)
        self.extra_headers = dict(extra_headers or {})
        if type(max_transport_retries) is not int or max_transport_retries < 0:
            raise ValueError("max_transport_retries must be a nonnegative integer")
        for name, value in (("retry_backoff_s", retry_backoff_s),
                            ("retry_backoff_cap_s", retry_backoff_cap_s)):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be a finite nonnegative number")
        self.max_transport_retries = max_transport_retries
        self.retry_backoff_s = float(retry_backoff_s)
        self.retry_backoff_cap_s = float(retry_backoff_cap_s)
        self._urlopen = urlopen
        self._sleep = sleep
        self._wall_clock = wall_clock

    def _retry_after_seconds(self, value: str | None) -> float | None:
        """Parse a server minimum delay without shortening its requested wait.

        Parameters
        ----------
        value : str or None
            Retry-After delta-seconds or HTTP-date header value.

        Returns
        -------
        float or None
            Nonnegative seconds from the injected wall clock, or ``None`` for
            an absent or malformed header. Past dates return zero.
        """
        if not value:
            return None
        value = value.strip()
        if value.isascii() and value.isdecimal():
            return float(value)
        try:
            deadline = parsedate_to_datetime(value)
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=timezone.utc)
            return max(0.0, deadline.timestamp() - self._wall_clock())
        except (TypeError, ValueError, OverflowError, OSError):
            return None

    def build_http_request(self, request: LLMRequest) -> urllib.request.Request:
        """Construct the authenticated chat-completions request."""
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise ProviderAuthenticationError(f"Required API key environment variable '{self.api_key_env}' is not set.")
        payload = {
            "model": self.model,
            "messages": request.messages(),
            "temperature": self.temperature,
        }
        if self.json_response_mode:
            payload["response_format"] = {"type": "json_object"}
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
        """Execute one completion with bounded transient transport retries.

        Parameters
        ----------
        request : LLMRequest
            Provider-neutral prompt and provenance metadata.

        Returns
        -------
        LLMResponse
            Normalized completion, including the number of transport attempts.

        Raises
        ------
        ProviderAuthenticationError
            Credentials, permissions, or credits require operator intervention.
        ProviderTransportError
            A request is nonretryable or its transient retry budget is exhausted.
        ProviderResponseError
            The received envelope is malformed and content repair is permitted.
        """
        http_request = self.build_http_request(request)
        started = time.monotonic()
        backoff_s = min(self.retry_backoff_s, self.retry_backoff_cap_s)
        for attempt in range(self.max_transport_retries + 1):
            retry_after_s = None
            try:
                with self._urlopen(http_request, timeout=self.timeout_s) as response:
                    raw_bytes = response.read()
                break
            except urllib.error.HTTPError as exc:
                try:
                    error_body = exc.read().decode("utf-8", errors="replace")
                except (OSError, http.client.HTTPException):
                    # The known status still determines retry/authentication policy.
                    error_body = "error response body could not be read"
                message = f"{self.provider_name} HTTP {exc.code}: {error_body}"
                if exc.code in {401, 402, 403}:
                    raise ProviderAuthenticationError(
                        f"{message}. Check {self.api_key_env}, account access, and available credits."
                    ) from exc
                if not (exc.code == 429 or 500 <= exc.code <= 599):
                    raise ProviderTransportError(
                        f"{message}. Check provider/model configuration before resuming."
                    ) from exc
                failure = exc
                reason = message
                retry_after_s = self._retry_after_seconds((exc.headers or {}).get("Retry-After"))
            except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
                failure = exc
                reason = f"{self.provider_name} request failed: {exc}"
            if attempt == self.max_transport_retries:
                raise ProviderTransportError(
                    f"{reason}. Exhausted {attempt + 1} transport attempts; "
                    "check provider availability before resuming."
                ) from failure
            if retry_after_s is not None and retry_after_s > self.retry_backoff_cap_s:
                raise ProviderTransportError(
                    f"{reason}. Retry-After requires {retry_after_s:g}s, exceeding "
                    f"retry_backoff_cap_s={self.retry_backoff_cap_s:g}s; stopped "
                    "without retrying before the server deadline. Increase the cap "
                    "or resume after the requested delay."
                ) from failure
            self._sleep(max(backoff_s, retry_after_s or 0.0))
            backoff_s = min(self.retry_backoff_cap_s, backoff_s * 2.0)
        latency_ms = (time.monotonic() - started) * 1000.0
        try:
            raw_response = json.loads(raw_bytes.decode("utf-8"))
            choice = raw_response["choices"][0]
            content = choice["message"]["content"]
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise ProviderResponseError(f"{self.provider_name} returned an invalid chat-completions response.") from exc
        if not isinstance(content, str):
            raise ProviderResponseError(f"{self.provider_name} returned non-string message content.")
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
                "transport_attempts": attempt + 1,
                "base_url": self.base_url,
                "temperature": self.temperature,
                "timeout_s": self.timeout_s,
                "json_response_mode": self.json_response_mode,
            },
        )


def build_provider(config: Any) -> LLMProvider:
    """Build the configured provider while keeping choices out of the NAS loop."""
    provider_name = str(cfg_get(config, "provider", "")).strip().lower()
    if provider_name == "fake":
        return FakeProvider(list(cfg_get(config, "responses", [])), model=str(cfg_get(config, "model", "fake-model")))
    if provider_name in {"openrouter", "openai_compatible"}:
        return OpenAICompatibleProvider(
            provider_name=provider_name,
            base_url=str(cfg_get(config, "base_url")),
            api_key_env=str(cfg_get(config, "api_key_env")),
            model=str(cfg_get(config, "model")),
            temperature=float(cfg_get(config, "temperature", 0.4)),
            timeout_s=float(cfg_get(config, "timeout_s", 60.0)),
            json_response_mode=bool(
                cfg_get(config, "json_response_mode", provider_name == "openrouter")
            ),
            extra_headers=dict(cfg_get(config, "extra_headers", {})),
            max_transport_retries=cfg_get(config, "max_transport_retries", 5),
            retry_backoff_s=cfg_get(config, "retry_backoff_s", 5.0),
            retry_backoff_cap_s=cfg_get(config, "retry_backoff_cap_s", 60.0),
        )
    raise ValueError(f"Unsupported LLM provider: {provider_name!r}.")
