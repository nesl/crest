# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Provider-independent request/response records and deterministic fake."""

from __future__ import annotations

import json
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
    raise ValueError(f"Unsupported LLM provider: {provider_name!r}.")
