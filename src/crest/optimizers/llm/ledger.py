# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Append-only filesystem ledger for LLM optimizer provenance."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any
from functools import partial

from ..history import to_plain

from .provider import LLMRequest, LLMResponse

_json_safe = partial(to_plain, nonfinite_to_none=True)


class LLMLedger:
    """Write reproducible requests, responses, decisions, and events.

    Parameters
    ----------
    root : pathlib.Path
        Artifact directory owned by one ledger writer.
    """

    def __init__(self, root: Path) -> None:
        """Create artifact directories and recover the next request counter.

        Parameters
        ----------
        root : pathlib.Path
            Ledger directory; created with missing parents.

        Raises
        ------
        OSError
            If the ledger directory cannot be created or inspected.
        """
        self.root = Path(root)
        self.requests_dir = self.root / "requests"
        self.requests_dir.mkdir(parents=True, exist_ok=True)
        # Recover the counter once; this ledger has one writer per directory.
        self._next_request_id = max((
            int(path.name.split(".", 1)[0])
            for path in self.requests_dir.glob("*.request.json")
            if path.name.split(".", 1)[0].isdigit()
        ), default=0) + 1

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        """Write one stable, human-readable JSON document.

        Parameters
        ----------
        path : pathlib.Path
            Destination file in an existing directory.
        payload : Any
            Evidence normalized to JSON-safe plain data before writing.

        Raises
        ------
        OSError
            If the destination cannot be written.
        """
        path.write_text(
            json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    def _append_jsonl(self, filename: str, payload: Any) -> None:
        """Append one normalized JSON record to a ledger stream.

        Parameters
        ----------
        filename : str
            Stream filename relative to the ledger root.
        payload : Any
            Record to normalize and append.

        Raises
        ------
        OSError
            If the stream cannot be opened or written.
        """
        with (self.root / filename).open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)
                + "\n"
            )

    def write_exchange(self, request_id: int, request: LLMRequest, response: LLMResponse) -> None:
        """Persist one numbered request and normalized response.

        Parameters
        ----------
        request_id : int
            Positive request sequence number.
        request : LLMRequest
            Prompt and provenance to persist before its response.
        response : LLMResponse
            Normalized completion and provider provenance.

        Raises
        ------
        ValueError
            If the request number is less than one.
        OSError
            If an artifact cannot be written.
        """
        self.write_request(request_id, request)
        self.write_response(request_id, response)

    def next_request_id(self) -> int:
        """Allocate the next monotonically increasing request number.

        Returns
        -------
        int
            Previously unused number; advances the in-memory counter.
        """
        request_id = self._next_request_id
        self._next_request_id += 1
        return request_id

    def write_request(self, request_id: int, request: LLMRequest) -> None:
        """Persist one numbered request before the provider call.

        Parameters
        ----------
        request_id : int
            Positive request sequence number.
        request : LLMRequest
            Prompt, version, and metadata to persist with a content hash.

        Raises
        ------
        ValueError
            If the request number is less than one.
        OSError
            If the artifact cannot be written.
        """
        if request_id < 1:
            raise ValueError("request_id must be greater than or equal to 1.")
        self._next_request_id = max(self._next_request_id, request_id + 1)
        stem = f"{request_id:06d}"
        request_payload = {
            "messages": request.messages(),
            "prompt_version": request.prompt_version,
            "metadata": request.metadata,
        }
        request_payload["request_hash"] = hashlib.sha256(
            json.dumps(request_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self._write_json(self.requests_dir / f"{stem}.request.json", request_payload)

    def write_response(self, request_id: int, response: Any) -> None:
        """Persist one numbered response or provider failure.

        Parameters
        ----------
        request_id : int
            Positive request sequence number.
        response : LLMResponse or Mapping
            Normalized response or provider-error mapping to persist with a hash.

        Raises
        ------
        ValueError
            If the request number is less than one.
        OSError
            If the artifact cannot be written.
        """
        if request_id < 1:
            raise ValueError("request_id must be greater than or equal to 1.")
        stem = f"{request_id:06d}"
        response_payload = _json_safe(response)
        response_hash = hashlib.sha256(
            json.dumps(response_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self._write_json(
            self.requests_dir / f"{stem}.response.json",
            {**response_payload, "response_hash": response_hash},
        )

    def record_prompt_context(self, context: dict[str, Any]) -> None:
        """Append complete prompt context supplied for a request.

        Parameters
        ----------
        context : Mapping
            Complete prompt context supplied for a request.

        Raises
        ------
        OSError
            If the ledger stream cannot be written.
        """
        self._append_jsonl("prompt_contexts.jsonl", context)

    def record_returned(self, payload: Any) -> None:
        """Append accepted generation result returned to the runner; enqueue remains study evidence.

        Parameters
        ----------
        payload : Mapping
            Accepted generation result returned to the runner; enqueue remains study evidence.

        Raises
        ------
        OSError
            If the ledger stream cannot be written.
        """
        self._append_jsonl("returned_candidates.jsonl", {
            "schema_version": 2, "status": "returned_to_runner", **payload,
        })

    def record_accepted(self, payload: Any) -> None:
        """Append historical v1 accepted-candidate record.

        Parameters
        ----------
        payload : Mapping
            Historical v1 accepted-candidate record.

        Raises
        ------
        OSError
            If the ledger stream cannot be written.
        """
        self._append_jsonl("accepted_candidates.jsonl", payload)

    def record_rejected(self, payload: Any) -> None:
        """Append candidate rejection category, explanation, and provenance.

        Parameters
        ----------
        payload : Mapping
            Candidate rejection category, explanation, and provenance.

        Raises
        ------
        OSError
            If the ledger stream cannot be written.
        """
        self._append_jsonl("rejected_candidates.jsonl", payload)

    def record_event(self, payload: Any) -> None:
        """Append optimizer lifecycle, fallback, or memory event.

        Parameters
        ----------
        payload : Mapping
            Optimizer lifecycle, fallback, or memory event.

        Raises
        ------
        OSError
            If the ledger stream cannot be written.
        """
        self._append_jsonl("optimizer_events.jsonl", payload)
