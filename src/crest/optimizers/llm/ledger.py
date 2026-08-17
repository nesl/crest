# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Append-only filesystem ledger for LLM optimizer provenance."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .provider import LLMRequest, LLMResponse


def _json_safe(value: Any) -> Any:
    """Normalize dataclasses, paths, tuples, and mappings for JSON output."""
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class LLMLedger:
    """Write reproducible requests, responses, decisions, and events."""

    def __init__(self, root: Path) -> None:
        """Create the ledger root and numbered request directory."""
        self.root = Path(root)
        self.requests_dir = self.root / "requests"
        self.requests_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        """Write one stable, human-readable JSON document."""
        path.write_text(
            json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    def _append_jsonl(self, filename: str, payload: Any) -> None:
        """Append one stable JSON object to a ledger stream."""
        with (self.root / filename).open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)
                + "\n"
            )

    def write_exchange(self, request_id: int, request: LLMRequest, response: LLMResponse) -> None:
        """Persist one numbered request/response pair."""
        if request_id < 1:
            raise ValueError("request_id must be greater than or equal to 1.")
        stem = f"{request_id:06d}"
        request_payload = {
            "messages": request.messages(),
            "prompt_version": request.prompt_version,
            "metadata": request.metadata,
        }
        self._write_json(self.requests_dir / f"{stem}.request.json", request_payload)
        self._write_json(self.requests_dir / f"{stem}.response.json", response)

    def record_prompt_context(self, context: dict[str, Any]) -> None:
        """Append the bounded context supplied for one request."""
        self._append_jsonl("prompt_contexts.jsonl", context)

    def record_accepted(self, payload: Any) -> None:
        """Append one accepted-candidate record."""
        self._append_jsonl("accepted_candidates.jsonl", payload)

    def record_rejected(self, payload: Any) -> None:
        """Append one rejected-candidate record."""
        self._append_jsonl("rejected_candidates.jsonl", payload)

    def record_event(self, payload: Any) -> None:
        """Append one optimizer lifecycle/fallback event."""
        self._append_jsonl("optimizer_events.jsonl", payload)
