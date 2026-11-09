# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Shared synchronous HTTP retries and per-attempt spend accounting."""
from __future__ import annotations

import http.client
import json
import math
import os
import time
import urllib.error
import urllib.request
from datetime import timezone
from email.utils import parsedate_to_datetime
from typing import Any

from .records import (LLMRequest, LLMResponse, ProviderError, ProviderAuthenticationError,
                      ProviderTransportError, ProviderResponseError)


class HTTPProvider:
    """Provide synchronous JSON completions with retries and optional spending limits.

    Parameters
    ----------
    provider_name : str
        Identifier attached to normalized completions.
    base_url : str
        API base URL without the chat-completions endpoint suffix.
    api_key_env : str
        Environment-variable name resolved only when constructing a request.
    model : str
        Provider model identifier.
    temperature : float or None, optional
        Finite nonnegative sampling temperature; ``None`` omits the API field.
    reasoning_effort : str or None, optional
        Explicit supported reasoning level; ``None`` uses provider defaults.
    service_tier : str or None, optional
        Explicit supported service tier; ``None`` uses provider defaults.
    timeout_s : float, optional
        Timeout passed to each HTTP attempt.
    json_response_mode : bool, optional
        Request an API JSON-object response format.
    extra_headers : dict of str to str or None, optional
        Additional headers, including provider-specific routing or authorization.
    max_output_tokens : int or None, optional
        Positive output-token ceiling; ``None`` omits the API limit.
    output_token_parameter : str, optional
        API field carrying the output ceiling; Responses also accepts max_output_tokens.
    spend_tracker : Any, optional
        Persistent accounting owner reserving and settling each billable attempt.
    urlopen : callable, optional
        Injectable HTTP opener accepting the request and timeout.
    max_transport_retries : int, optional
        Additional attempts after retryable transport or server failures.
    retry_backoff_s : float, optional
        Initial exponential retry delay in seconds.
    retry_backoff_cap_s : float, optional
        Cap on local backoff; server Retry-After can require a longer wait.
    sleep : callable, optional
        Injectable delay function accepting seconds.
    wall_clock : callable, optional
        Injectable Unix-time clock for Retry-After HTTP dates.

    Raises
    ------
    ValueError
        If generation controls, output limits, or retry settings are invalid."""

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
        """Store settings; see the class parameter reference."""
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
        allowed_parameters = {"max_tokens", "max_completion_tokens"}
        if self.provider_name == "openai_responses":
            allowed_parameters.add("max_output_tokens")
        if not isinstance(output_token_parameter, str) or output_token_parameter not in allowed_parameters:
            raise ValueError(f"output_token_parameter must be one of {sorted(allowed_parameters)}")
        self.max_output_tokens = max_output_tokens
        self.output_token_parameter = "max_output_tokens" if self.provider_name == "openai_responses" else output_token_parameter
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

    def _http_request(self, endpoint: str, payload: dict[str, Any]) -> urllib.request.Request:
        """Encode a payload and resolve credentials only at request time."""
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise ProviderAuthenticationError(f"Required API key environment variable '{self.api_key_env}' is not set.")
        return urllib.request.Request(
            f"{self.base_url}/{endpoint}",
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", **self.extra_headers},
            method="POST",
        )

    @staticmethod
    def normalize_usage(usage: Any) -> dict[str, Any] | None:
        """Return common chat token counters when available."""
        return dict(usage) if isinstance(usage, dict) else None

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
        def transport_error(error_type, message):
            """Attach the exact secret-free request to a transport failure.

            Parameters
            ----------
            error_type : type
                Provider exception class.
            message : str
                Transport diagnostic.

            Returns
            -------
            ProviderError
                Exception carrying the request body for ledger persistence.
            """
            error = error_type(message)
            error.wire_payload = payload
            return error

        started = time.monotonic()
        backoff_s = min(self.retry_backoff_s, self.retry_backoff_cap_s)
        for attempt in range(self.max_transport_retries + 1):
            retry_after_s = None
            # Reserve each HTTP attempt independently. Failed or ambiguous
            # attempts remain charged because the server may have processed them.
            try:
                reservation_id = (
                    self.spend_tracker.reserve(payload)
                    if self.spend_tracker is not None else None
                )
            except ProviderError as exc:
                exc.wire_payload = payload
                raise
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
                    raise transport_error(ProviderAuthenticationError,
                        f"{message}. Check {self.api_key_env}, account access, and available credits."
                    ) from exc
                if not (exc.code == 429 or 500 <= exc.code <= 599):
                    raise transport_error(ProviderTransportError,
                        f"{message}. Check provider/model configuration before resuming."
                    ) from exc
                failure = exc
                reason = message
                retry_after_s = self._retry_after_seconds((exc.headers or {}).get("Retry-After"))
            except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
                failure = exc
                reason = f"{self.provider_name} request failed: {exc}"
            if attempt == self.max_transport_retries:
                raise transport_error(ProviderTransportError,
                    f"{reason}. Exhausted {attempt + 1} transport attempts; "
                    "check provider availability before resuming."
                ) from failure
            if retry_after_s is not None and retry_after_s > self.retry_backoff_cap_s:
                raise transport_error(ProviderTransportError,
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
            raise transport_error(ProviderResponseError, f"{self.provider_name} returned an invalid JSON response.") from exc
        usage = self.normalize_usage(raw_response.get("usage") if isinstance(raw_response, dict) else None)
        spending = None
        try:
            spending = (
                self.spend_tracker.settle(reservation_id, usage)
                if self.spend_tracker is not None else None
            )
            return self.normalize_response(raw_response, usage, payload, latency_ms, attempt + 1, spending)
        except ProviderError as exc:
            exc.raw_response = raw_response
            exc.wire_payload = payload
            exc.usage = usage
            if spending is not None or not hasattr(exc, "spending"):
                exc.spending = spending
            raise
