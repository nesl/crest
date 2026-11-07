# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Provider-neutral records, failures, and offline fixtures."""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Protocol
from crest.interfaces import OptimizerBudgetExhausted

@dataclass(frozen=True)
class LLMRequest:
    """Represent a provider-neutral structured JSON completion request.

    Parameters
    ----------
    system_prompt : str
        System instruction text.
    user_prompt : str
        Serialized experiment evidence and current request.
    prompt_version : str
        Prompt-policy version retained in provenance.
    metadata : dict, optional
        Request purpose, study identity, and repair metadata.
    cache_fixed_prefix : bool, optional
        Request provider caching of the fixed prompt prefix when supported.
    prompt_cache_key : str or None, optional
        Optional provider cache routing key.
    """

    system_prompt: str
    user_prompt: str
    prompt_version: str
    metadata: dict[str, Any] = field(default_factory=dict)
    cache_fixed_prefix: bool = False
    prompt_cache_key: str | None = None

    def messages(self) -> list[dict[str, str]]:
        """Render the system and user chat messages.

        Returns
        -------
        list of dict
            Two role/content mappings in system, then user order.
        """
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt},
        ]


@dataclass(frozen=True)
class LLMResponse:
    """Represent normalized completion content and provider provenance.

    Parameters
    ----------
    content : str
        Completion text to parse as JSON.
    provider : str
        Provider identifier.
    model : str
        Model identifier returned by the provider or configured default.
    raw_response : dict
        Original decoded provider response.
    usage : dict or None, optional
        Normalized token usage when reported.
    finish_reason : str or None, optional
        Normalized completion stop or truncation reason.
    latency_ms : float, optional
        Elapsed provider request time in milliseconds.
    metadata : dict, optional
        Additional provider and request provenance.
    """

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
        """Execute one provider-neutral structured completion.

        Parameters
        ----------
        request : LLMRequest
            Prompt messages and provenance metadata.

        Returns
        -------
        LLMResponse
            Normalized completion and raw provider provenance.

        Raises
        ------
        ProviderError
            If provider authentication, transport, response handling, or spending fails.
        """


class FakeProvider:
    """Return a finite configured response sequence without network access.

    Parameters
    ----------
    responses : list of str or dict
        Completion text or JSON objects returned in order.
    model : str, optional
        Model identifier attached to fixture responses.
    """

    def __init__(
        self,
        responses: list[str | dict[str, Any]],
        *,
        model: str = "fake-model",
    ) -> None:
        """Initialize a finite response queue and request capture list.

        Parameters
        ----------
        responses : list of str or dict
            Completion fixtures consumed in order.
        model : str, optional
            Model identifier attached to fixture responses.
        """
        self._responses = deque(responses)
        self.model = model
        self.requests: list[LLMRequest] = []

    def complete_json(self, request: LLMRequest) -> LLMResponse:
        """Record a request and consume the next configured response.

        Parameters
        ----------
        request : LLMRequest
            Request appended to the capture list even when fixtures are exhausted.

        Returns
        -------
        LLMResponse
            Fixture text with fake-provider provenance and stop finish reason.

        Raises
        ------
        RuntimeError
            If no configured response remains.
        """
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


