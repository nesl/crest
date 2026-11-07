# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Responses adapter with explicit prompt caching."""
from __future__ import annotations

import urllib.request
from typing import Any

from .records import LLMRequest, LLMResponse, ProviderResponseError
from .chat import OpenAICompatibleProvider


class OpenAIResponsesProvider(OpenAICompatibleProvider):
    """Responses provider; settings are documented on HTTPProvider."""

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
        payload = {
            "model": self.model,
            "instructions": request.system_prompt,
            "input": request.user_prompt,
            "prompt_cache_options": {"mode": "explicit"},
            "store": False,
        }
        if request.cache_fixed_prefix:
            # Cache the stable developer prefix with an explicit breakpoint and TTL.
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
        return self._http_request("responses", payload)

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
            "cached_tokens": details.get("cached_tokens"),
            "cache_write_tokens": details.get("cache_write_tokens"),
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
