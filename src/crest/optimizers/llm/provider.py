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
from ...interfaces import OptimizerBudgetExhausted


@dataclass(frozen=True)
class LLMRequest:
    """Provider-neutral structured JSON completion request."""

    system_prompt: str
    user_prompt: str
    prompt_version: str
    metadata: dict[str, Any] = field(default_factory=dict)
    cache_fixed_prefix: bool = False
    prompt_cache_key: str | None = None

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


class SpendLimitExceeded(ProviderError, OptimizerBudgetExhausted):
    """The next billable request would exceed the configured spend limit."""


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
        temperature: float | None = 0.4,
        reasoning_effort: str | None = None,
        service_tier: str | None = None,
        timeout_s: float = 60.0,
        json_response_mode: bool = True,
        extra_headers: dict[str, str] | None = None,
        max_output_tokens: int | None = None,
        output_token_parameter: str = "max_completion_tokens",
        spend_tracker: Any = None,
        urlopen: Any = urllib.request.urlopen,
        max_transport_retries: int = 5,
        retry_backoff_s: float = 5.0,
        retry_backoff_cap_s: float = 60.0,
        sleep: Any = time.sleep,
        wall_clock: Any = time.time,
    ) -> None:
        """Store provider settings without reading or persisting the API key.

        Parameters
        ----------
        temperature : float or None, optional
            Sampling temperature; ``None`` omits the API field.
        reasoning_effort : str or None, optional
            Explicit reasoning effort; ``None`` uses the provider default.
        service_tier : str or None, optional
            Explicit service tier; ``None`` uses the provider default.
        max_output_tokens : int or None, optional
            Positive completion-token ceiling; ``None`` omits the API limit.
        output_token_parameter : str, optional
            API limit field, either ``max_completion_tokens`` or ``max_tokens``.
        spend_tracker : Any, optional
            Shared persistent ledger reserving and settling every HTTP attempt.
        """
        self.provider_name = provider_name
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.model = model
        if temperature is not None and (
            isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature) or temperature < 0
        ):
            raise ValueError("temperature must be a finite nonnegative number or None")
        self.temperature = None if temperature is None else float(temperature)
        for name, value, allowed in (
            ("reasoning_effort", reasoning_effort, {"none", "minimal", "low", "medium", "high", "xhigh", "max"}),
            ("service_tier", service_tier, {"auto", "default", "flex", "scale", "priority", "fast"}),
        ):
            if value is not None and (not isinstance(value, str) or value not in allowed):
                raise ValueError(f"{name} must be one of {sorted(allowed)} or None")
        self.reasoning_effort = reasoning_effort
        self.service_tier = service_tier
        self.timeout_s = float(timeout_s)
        self.json_response_mode = bool(json_response_mode)
        self.extra_headers = dict(extra_headers or {})
        if max_output_tokens is not None and (
            type(max_output_tokens) is not int or max_output_tokens <= 0
        ):
            raise ValueError("max_output_tokens must be a positive integer or None")
        if not isinstance(output_token_parameter, str) or output_token_parameter not in {
            "max_tokens", "max_completion_tokens"
        }:
            raise ValueError("output_token_parameter must be max_tokens or max_completion_tokens")
        self.max_output_tokens = max_output_tokens
        self.output_token_parameter = output_token_parameter
        self.spend_tracker = spend_tracker
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
        }
        # Explicit null temperature supports models that reject sampling controls.
        for name in ("temperature", "reasoning_effort", "service_tier"):
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        if self.json_response_mode:
            payload["response_format"] = {"type": "json_object"}
        if self.max_output_tokens is not None:
            payload[self.output_token_parameter] = self.max_output_tokens
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
        SpendLimitExceeded
            Reserving another HTTP attempt would exceed the configured spend cap.
        """
        http_request = self.build_http_request(request)
        payload = json.loads(http_request.data)
        started = time.monotonic()
        backoff_s = min(self.retry_backoff_s, self.retry_backoff_cap_s)
        for attempt in range(self.max_transport_retries + 1):
            retry_after_s = None
            # Reserve each HTTP attempt independently. Failed or ambiguous
            # attempts remain charged because the server may have processed them.
            reservation_id = (
                self.spend_tracker.reserve(payload)
                if self.spend_tracker is not None else None
            )
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
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProviderResponseError(f"{self.provider_name} returned an invalid chat-completions response.") from exc
        usage = raw_response.get("usage") if isinstance(raw_response, dict) else None
        if isinstance(self, OpenAIResponsesProvider):
            usage = self.normalize_usage(usage)
        try:
            spending = (
                self.spend_tracker.settle(reservation_id, usage)
                if self.spend_tracker is not None else None
            )
        except ProviderError as exc:
            if isinstance(self, OpenAIResponsesProvider):
                # Preserve the paid response even when accounting halts on overrun.
                exc.raw_response = raw_response
                exc.wire_payload = payload
                exc.usage = usage
            raise
        if isinstance(self, OpenAIResponsesProvider):
            return self.normalize_response(raw_response, usage, payload, latency_ms, attempt + 1, spending)
        try:
            choice = raw_response["choices"][0]
            content = choice["message"]["content"]
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise ProviderResponseError(f"{self.provider_name} returned an invalid chat-completions response.") from exc
        if not isinstance(content, str):
            raise ProviderResponseError(f"{self.provider_name} returned non-string message content.")
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
                **({"reasoning_effort": self.reasoning_effort} if self.reasoning_effort is not None else {}),
                **({"service_tier": self.service_tier} if self.service_tier is not None else {}),
                "timeout_s": self.timeout_s,
                "json_response_mode": self.json_response_mode,
                **({"spending": spending} if spending is not None else {}),
            },
        )


class OpenAIResponsesProvider(OpenAICompatibleProvider):
    """Responses wire format with the existing transport and durable accounting."""

    def build_http_request(self, request: LLMRequest) -> urllib.request.Request:
        """Build explicit caching wire, with no breakpoints unless opted in.

        Parameters
        ----------
        request : LLMRequest
            Fixed system context and chronological user context.

        Returns
        -------
        urllib.request.Request
            Authenticated POST whose JSON body contains no secrets.
        """
        # Reuse credential/header handling without changing the legacy chat wire.
        legacy = super().build_http_request(request)
        payload = {
            "model": self.model,
            "instructions": request.system_prompt,
            "input": request.user_prompt,
            "prompt_cache_options": {"mode": "explicit"},
            "store": False,
        }
        if request.cache_fixed_prefix:
            # Reuse ranker replay's explicit developer block, stable key, and TTL.
            payload.pop("instructions")
            payload["input"] = [
                {"role": "developer", "content": [{"type": "input_text",
                    "text": request.system_prompt,
                    "prompt_cache_breakpoint": {"mode": "explicit"}}]},
                {"role": "user", "content": [{"type": "input_text", "text": request.user_prompt}]},
            ]
            payload["prompt_cache_options"] = {"mode": "explicit", "ttl": "30m"}
            if request.prompt_cache_key:
                payload["prompt_cache_key"] = request.prompt_cache_key
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.reasoning_effort is not None:
            payload["reasoning"] = {"effort": self.reasoning_effort}
        if self.service_tier is not None:
            payload["service_tier"] = self.service_tier
        if self.max_output_tokens is not None:
            payload["max_output_tokens"] = self.max_output_tokens
        if self.json_response_mode:
            payload["text"] = {"format": {"type": "json_object"}}
        return urllib.request.Request(
            f"{self.base_url}/responses",
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers=dict(legacy.header_items()), method="POST",
        )

    @staticmethod
    def normalize_usage(usage: Any) -> dict[str, Any] | None:
        """Preserve Responses counters and add the common accounting names.

        Parameters
        ----------
        usage : object
            Raw Responses usage object or missing value.

        Returns
        -------
        dict or None
            Common token names alongside original usage and cache counters.
        """
        if not isinstance(usage, dict):
            return None
        details = usage.get("input_tokens_details")
        details = details if isinstance(details, dict) else {}
        return {
            **usage,
            "prompt_tokens": usage.get("input_tokens"),
            "completion_tokens": usage.get("output_tokens"),
            "cached_tokens": details.get("cached_tokens", 0),
            "cache_write_tokens": details.get("cache_write_tokens", 0),
        }

    def normalize_response(self, raw: Any, usage: Any, payload: dict[str, Any],
                           latency_ms: float, attempts: int, spending: Any) -> LLMResponse:
        """Extract only completed assistant text and preserve complete provenance.

        Parameters
        ----------
        raw : object
            Decoded server envelope.
        usage : object
            Normalized token usage.
        payload : dict
            Secret-free exact wire JSON.
        latency_ms : float
            Total transport time.
        attempts : int
            Number of reserved HTTP attempts.
        spending : object
            Durable settlement result.

        Returns
        -------
        LLMResponse
            Completed text, raw response, usage, status, and returned tier.

        Raises
        ------
        ProviderResponseError
            Refused, incomplete, failed, empty, or malformed response.
        """
        try:
            if raw["status"] != "completed":
                raise ValueError("response did not complete")
            texts = []
            for item in raw["output"]:
                if item.get("type") != "message":
                    continue
                if item.get("role") != "assistant" or item.get("status") != "completed":
                    raise ValueError("assistant message did not complete")
                for block in item["content"]:
                    if block.get("type") == "refusal":
                        raise ValueError("response refused")
                    if block.get("type") == "output_text":
                        if not isinstance(block.get("text"), str):
                            raise ValueError("non-string output text")
                        texts.append(block["text"])
            content = "".join(texts)
            if not content:
                raise ValueError("missing assistant text")
        except (KeyError, TypeError, AttributeError, ValueError) as exc:
            error = ProviderResponseError(f"{self.provider_name} returned an invalid Responses envelope: {exc}.")
            error.raw_response = raw
            error.wire_payload = payload
            error.usage = usage
            error.spending = spending
            raise error from exc
        return LLMResponse(
            content=content, provider=self.provider_name,
            model=str(raw.get("model", self.model)), raw_response=raw, usage=usage,
            finish_reason=raw["status"], latency_ms=latency_ms,
            metadata={"wire_payload": payload, "transport_attempts": attempts,
                      "base_url": self.base_url, "status": raw["status"],
                      "service_tier": raw.get("service_tier"),
                      "requested_service_tier": self.service_tier,
                      **({"spending": spending} if spending is not None else {})},
        )


def build_provider(config: Any, *, spend_tracker: Any = None) -> LLMProvider:
    """Build a provider with optional per-attempt spend accounting.

    Parameters
    ----------
    config : Any
        Normalized LLM configuration including provider and output-token settings.
    spend_tracker : Any, optional
        Persistent tracker exposing ``reserve(payload)`` and ``settle(id, usage)``.

    Returns
    -------
    LLMProvider
        Configured provider using the shared tracker for every HTTP attempt.
    """
    provider_name = str(cfg_get(config, "provider", "")).strip().lower()
    if provider_name == "fake":
        return FakeProvider(list(cfg_get(config, "responses", [])), model=str(cfg_get(config, "model", "fake-model")))
    if provider_name in {"openrouter", "openai_compatible", "openai_responses"}:
        provider_type = OpenAIResponsesProvider if provider_name == "openai_responses" else OpenAICompatibleProvider
        return provider_type(
            provider_name=provider_name,
            base_url=str(cfg_get(config, "base_url")),
            api_key_env=str(cfg_get(config, "api_key_env")),
            model=str(cfg_get(config, "model")),
            temperature=cfg_get(config, "temperature", 0.4),
            reasoning_effort=cfg_get(config, "reasoning_effort", None),
            service_tier=cfg_get(config, "service_tier", None),
            timeout_s=float(cfg_get(config, "timeout_s", 60.0)),
            json_response_mode=bool(
                cfg_get(config, "json_response_mode", provider_name in {"openrouter", "openai_responses"})
            ),
            extra_headers=dict(cfg_get(config, "extra_headers", {})),
            max_output_tokens=cfg_get(config, "max_output_tokens", None),
            output_token_parameter=cfg_get(config, "output_token_parameter", "max_completion_tokens"),
            spend_tracker=spend_tracker,
            max_transport_retries=cfg_get(config, "max_transport_retries", 5),
            retry_backoff_s=cfg_get(config, "retry_backoff_s", 5.0),
            retry_backoff_cap_s=cfg_get(config, "retry_backoff_cap_s", 60.0),
        )
    raise ValueError(f"Unsupported LLM provider: {provider_name!r}.")
