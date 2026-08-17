# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Structured LLM envelopes and local candidate validation."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .search_space import SearchParam, SearchSpaceDescriptor


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


def _categorical_match(value: Any, choices: tuple[Any, ...]) -> bool:
    """Match categorical values without Python's bool/int equality ambiguity."""
    return any(type(value) is type(choice) and value == choice for choice in choices)


def _value_error(param: SearchParam, value: Any) -> tuple[str, str] | None:
    """Return a structured code/message when ``value`` violates ``param``."""
    if param.kind == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            return "type_error", f"'{param.name}' must be an integer."
        if value < param.low or value > param.high:
            return "out_of_range", f"'{param.name}' must be between {param.low} and {param.high}."
        return None

    if param.kind == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return "type_error", f"'{param.name}' must be a finite number."
        if not math.isfinite(float(value)):
            return "non_finite_float", f"'{param.name}' must be finite."
        if value < param.low or value > param.high:
            return "out_of_range", f"'{param.name}' must be between {param.low} and {param.high}."
        return None

    choices = param.choices or ()
    if isinstance(value, float) and not math.isfinite(value):
        return "non_finite_float", f"'{param.name}' must be finite."
    if not _categorical_match(value, choices):
        return "invalid_categorical", f"'{param.name}' must be one of {choices!r}."
    return None


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

    expected_keys = set(descriptor)
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

        actual_keys = set(candidate)
        missing = sorted(expected_keys - actual_keys)
        if missing:
            reject(index, "missing_keys", f"Missing required keys: {', '.join(missing)}.", candidate)
            continue
        unknown = sorted(actual_keys - expected_keys)
        if unknown:
            reject(index, "unknown_keys", f"Unknown keys: {', '.join(unknown)}.", candidate)
            continue

        invalid_value = None
        for name in descriptor:
            invalid_value = _value_error(descriptor[name], candidate[name])
            if invalid_value is not None:
                break
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
