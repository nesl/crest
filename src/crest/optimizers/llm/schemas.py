# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Structured LLM envelopes and local candidate validation."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..search_space import SearchSpaceDescriptor, candidate_error


class CandidateBatch(BaseModel):
    """Validate a provider-independent nonempty candidate envelope.

    Parameters
    ----------
    candidates : list of dict
        Raw candidate mappings or policy envelopes for local validation.

    Raises
    ------
    pydantic.ValidationError
        If candidates are absent, empty, incorrectly typed, or extra fields are supplied.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    candidates: list[dict[str, Any]] = Field(min_length=1)


@dataclass(frozen=True)
class CandidateRejection:
    """Record a machine-readable local candidate rejection.

    Parameters
    ----------
    index : int
        Zero-based index in the provider response.
    code : str
        Stable rejection category.
    message : str
        Candidate-specific validation explanation.
    candidate : dict
        Rejected candidate or original envelope.
    """

    index: int
    code: str
    message: str
    candidate: dict[str, Any]


@dataclass(frozen=True)
class CandidateValidationResult:
    """Collect locally accepted candidates and structured rejections.

    Parameters
    ----------
    accepted : tuple of dict
        Canonical legal candidates in response order.
    rejected : tuple of CandidateRejection
        Validation failures with original response indices.
    rationales : tuple of str or None, optional
        Rationale aligned with each accepted candidate.
    """

    accepted: tuple[dict[str, Any], ...]
    rejected: tuple[CandidateRejection, ...]
    rationales: tuple[str | None, ...] = ()


def candidate_fingerprint(candidate: Mapping[str, Any]) -> str:
    """Encode a deterministic candidate identity that preserves scalar types.

    Parameters
    ----------
    candidate : Mapping of str to Any
        Raw parameter mapping with finite JSON-serializable values.

    Returns
    -------
    str
        Sorted compact JSON containing each name, value type, and value.

    Raises
    ------
    ValueError
        If a candidate value is nonfinite.
    TypeError
        If a value cannot be JSON encoded.
    """
    typed_items = [
        (key, type(value).__name__, value)
        for key, value in sorted(candidate.items())
    ]
    return json.dumps(typed_items, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _canonical_candidate(
    candidate: Mapping[str, Any], descriptor: SearchSpaceDescriptor
) -> dict[str, Any]:
    """Normalize validated numeric values while preserving categorical types.

    Parameters
    ----------
    candidate : Mapping[str, Any]
        Candidate already validated against the descriptor.
    descriptor : SearchSpaceDescriptor
        Active parameter declarations.

    Returns
    -------
    dict[str, Any]
        Parameters with the same numeric types Optuna records.
    """
    return {
        name: (
            float(candidate[name]) if param.kind == "float"
            else int(candidate[name]) if param.kind == "int" else candidate[name]
        )
        for name, param in descriptor.items()
    }


def _history_fingerprints(
    candidates: Iterable[Mapping[str, Any]], descriptor: SearchSpaceDescriptor
) -> set[str]:
    """Build canonical identities for valid historical candidates.

    Parameters
    ----------
    candidates : Iterable[Mapping[str, Any]]
        Historical parameter dictionaries, possibly incomplete.
    descriptor : SearchSpaceDescriptor
        Active parameter declarations.

    Returns
    -------
    set[str]
        Identities for candidates that belong to the active search space.
    """
    return {
        candidate_fingerprint(_canonical_candidate(candidate, descriptor))
        for candidate in candidates
        if candidate_error(candidate, descriptor) is None
    }


def validate_candidate_batch(
    batch: CandidateBatch,
    descriptor: SearchSpaceDescriptor,
    *,
    batch_size: int,
    completed_candidates: Iterable[Mapping[str, Any]] = (),
    queued_candidates: Iterable[Mapping[str, Any]] = (),
    require_rationale: bool = False,
) -> CandidateValidationResult:
    """Validate raw keys, values, batch size, rationale, and duplicate identities.

    Parameters
    ----------
    batch : CandidateBatch
        Parsed provider response envelope.
    descriptor : SearchSpaceDescriptor
        Active raw parameter distributions.
    batch_size : int
        Positive maximum number of candidates accepted from response positions.
    completed_candidates : Iterable of Mapping, optional
        Terminal parameters excluded as duplicates.
    queued_candidates : Iterable of Mapping, optional
        Waiting parameters excluded as duplicates.
    require_rationale : bool, optional
        Require exactly params and a nonempty rationale of at most 2000 characters.

    Returns
    -------
    CandidateValidationResult
        Canonical accepted candidates, aligned rationales, and per-entry rejections.

    Raises
    ------
    ValueError
        If batch_size is not a positive integer.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be an integer greater than or equal to 1.")

    completed_fingerprints = _history_fingerprints(completed_candidates, descriptor)
    queued_fingerprints = _history_fingerprints(queued_candidates, descriptor)
    accepted_fingerprints: set[str] = set()
    accepted: list[dict[str, Any]] = []
    rationales: list[str | None] = []
    rejected: list[CandidateRejection] = []

    def reject(index: int, code: str, message: str, candidate: Mapping[str, Any]) -> None:
        """Append a structured rejection for one response entry.

        Parameters
        ----------
        index : int
            Zero-based response position.
        code : str
            Stable rejection category.
        message : str
            Human-readable validation explanation.
        candidate : Mapping
            Rejected entry copied into the rejection record.
        """
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

        rationale = None
        if require_rationale:
            if (set(candidate) != {"params", "rationale"}
                    or not isinstance(candidate.get("params"), dict)
                    or not isinstance(candidate.get("rationale"), str)
                    or not candidate["rationale"].strip()
                    or len(candidate["rationale"]) > 2000):
                reject(index, "invalid_rationale_envelope",
                       "Each candidate requires exactly params and a nonempty rationale of at most 2000 characters.", candidate)
                continue
            rationale = candidate["rationale"].strip()
            candidate = candidate["params"]

        invalid_value = candidate_error(candidate, descriptor)
        if invalid_value is not None:
            reject(index, invalid_value[0], invalid_value[1], candidate)
            continue

        normalized = _canonical_candidate(candidate, descriptor)
        fingerprint = candidate_fingerprint(normalized)
        if fingerprint in completed_fingerprints:
            reject(index, "duplicate_completed", "Candidate duplicates a terminal trial.", candidate)
            continue
        if fingerprint in queued_fingerprints:
            reject(index, "duplicate_queued", "Candidate duplicates a queued trial.", candidate)
            continue
        if fingerprint in accepted_fingerprints:
            reject(index, "duplicate_batch", "Candidate duplicates an earlier candidate in this batch.", candidate)
            continue

        accepted.append(normalized)
        rationales.append(rationale)
        accepted_fingerprints.add(fingerprint)

    return CandidateValidationResult(tuple(accepted), tuple(rejected), tuple(rationales))


