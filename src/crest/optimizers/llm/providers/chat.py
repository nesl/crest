# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Chat Completions adapter."""
from __future__ import annotations

import urllib.request
from typing import Any

from .records import LLMRequest, LLMResponse, ProviderResponseError
from .transport import HTTPProvider

class OpenAICompatibleProvider(HTTPProvider):
    """Chat Completions provider; settings are documented on HTTPProvider."""

    def build_http_request(self, request: LLMRequest) -> urllib.request.Request:
        """Construct an authenticated chat-completions HTTP request.

        Parameters
        ----------
        request : LLMRequest
            Provider-neutral prompt messages and metadata.

        Returns
        -------
        urllib.request.Request
            POST request with configured model, generation controls, and auth headers.

        Raises
        ------
        ProviderAuthenticationError
            If the configured API-key environment variable is unset or empty.
        """
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
        return self._http_request("chat/completions", payload)

    def normalize_response(self, raw_response: Any, usage: Any, payload: dict[str, Any],
                           latency_ms: float, attempts: int, spending: Any) -> LLMResponse:
        """Extract chat text and preserve provider provenance."""
        try:
            choice = raw_response["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
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
                "transport_attempts": attempts,
                "base_url": self.base_url,
                "temperature": self.temperature,
                **({"reasoning_effort": self.reasoning_effort} if self.reasoning_effort is not None else {}),
                **({"service_tier": self.service_tier} if self.service_tier is not None else {}),
                "timeout_s": self.timeout_s,
                "json_response_mode": self.json_response_mode,
                **({"spending": spending} if spending is not None else {}),
            },
        )
