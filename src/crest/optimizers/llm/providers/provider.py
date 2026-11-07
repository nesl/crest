# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Public provider imports and configuration factory."""
from __future__ import annotations

# Keep module objects available for existing transport monkeypatch targets.
import os
import time
import urllib.request
from typing import Any
from crest.component_selection import cfg_get
from .records import (LLMRequest, LLMResponse, LLMProvider, FakeProvider, ProviderError,
                      ProviderAuthenticationError, ProviderTransportError, ProviderResponseError,
                      SpendLimitExceeded)
from .chat import OpenAICompatibleProvider
from .responses import OpenAIResponsesProvider


def default_json_response_mode(provider_name: str) -> bool:
    """Enable structured JSON responses for providers with a known JSON API."""
    return provider_name in {"openrouter", "openai_responses"}


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

    Raises
    ------
    ValueError
        If the provider name or constructor settings are unsupported or invalid.
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
            temperature=cfg_get(config, "temperature", None if provider_name == "openai_responses" else 0.4),
            reasoning_effort=cfg_get(config, "reasoning_effort", None),
            service_tier=cfg_get(config, "service_tier", None),
            timeout_s=float(cfg_get(config, "timeout_s", 60.0)),
            json_response_mode=bool(
                cfg_get(config, "json_response_mode", default_json_response_mode(provider_name))
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
