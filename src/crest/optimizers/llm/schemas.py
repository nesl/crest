# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Structured LLM envelopes and local candidate validation."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ...search_space import SearchSpaceDescriptor, candidate_error


class CandidateBatch(BaseModel):
    """Provider-independent envelope for a non-empty candidate batch."""

    model_config = ConfigDict(extra="forbid", strict=True)

    candidates: list[dict[str, Any]] = Field(min_length=1)


@dataclass(frozen=True)
class CandidateRejection:
    """Machine-readable reason one candidate was rejected locally."""

    index: int
    code: str
    message: str
    candidate: dict[str, Any]


@dataclass(frozen=True)
class CandidateValidationResult:
    """Accepted candidates and structured local rejection records."""

    accepted: tuple[dict[str, Any], ...]
    rejected: tuple[CandidateRejection, ...]


def candidate_fingerprint(candidate: Mapping[str, Any]) -> str:
    """Return a deterministic, type-preserving candidate identity."""
    typed_items = [
        (key, type(value).__name__, value)
        for key, value in sorted(candidate.items())
    ]
    return json.dumps(typed_items, sort_keys=True, separators=(",", ":"), allow_nan=False)


def validate_candidate_batch(
    batch: CandidateBatch,
    descriptor: SearchSpaceDescriptor,
    *,
    batch_size: int,
    completed_candidates: Iterable[Mapping[str, Any]] = (),
    queued_candidates: Iterable[Mapping[str, Any]] = (),
) -> CandidateValidationResult:
    """Validate exact keys, values, batch bounds, and duplicate identities."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be an integer greater than or equal to 1.")

    completed_fingerprints = {candidate_fingerprint(item) for item in completed_candidates}
    queued_fingerprints = {candidate_fingerprint(item) for item in queued_candidates}
    accepted_fingerprints: set[str] = set()
    accepted: list[dict[str, Any]] = []
    rejected: list[CandidateRejection] = []

    def reject(index: int, code: str, message: str, candidate: Mapping[str, Any]) -> None:
        rejected.append(
            CandidateRejection(
                index=index,
                code=code,
                message=message,
                candidate=dict(candidate),
            )
        )

    for index, candidate in enumerate(batch.candidates):
        if index >= batch_size:
            reject(index, "batch_size_exceeded", f"Only {batch_size} candidate(s) were requested.", candidate)
            continue

        invalid_value = candidate_error(candidate, descriptor)
        if invalid_value is not None:
            reject(index, invalid_value[0], invalid_value[1], candidate)
            continue

        fingerprint = candidate_fingerprint(candidate)
        if fingerprint in completed_fingerprints:
            reject(index, "duplicate_completed", "Candidate duplicates a completed trial.", candidate)
            continue
        if fingerprint in queued_fingerprints:
            reject(index, "duplicate_queued", "Candidate duplicates a queued trial.", candidate)
            continue
        if fingerprint in accepted_fingerprints:
            reject(index, "duplicate_batch", "Candidate duplicates an earlier candidate in this batch.", candidate)
            continue

        normalized = dict(candidate)
        accepted.append(normalized)
        accepted_fingerprints.add(fingerprint)

    return CandidateValidationResult(tuple(accepted), tuple(rejected))
