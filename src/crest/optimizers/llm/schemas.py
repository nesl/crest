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



def validate_policy_batch(batch, descriptor, *, quotas, targets=(), parents=(),
                          completed_candidates=(), queued_candidates=(), retained=(),
                          used_parents=(), distinct_parents=True, history_matches=()):
    """Validate next-study entries independently, retaining valid quota subsets.

    Parameters
    ----------
    batch : CandidateBatch
        Parsed provider envelope.
    descriptor : SearchSpaceDescriptor
        Active search parameters.
    quotas : dict
        Remaining local and exploration slots.
    targets, parents, retained, used_parents, history_matches : iterable
        Fixed target cells, eligible parents, accepted entries, used decoded
        parent identities, and rich terminal duplicate records.
    completed_candidates, queued_candidates : iterable
        Architectures which cannot be proposed again.
    distinct_parents : bool
        Require a different decoded parent while enough parents exist.

    Returns
    -------
    tuple
        Accepted metadata entries and candidate rejection records.
    """
    from .full_history import candidate_cell, decoded_fingerprint
    remaining = dict(quotas)
    targeted = bool(targets)
    accepted, rejected = [], []
    blocked = list(completed_candidates) + list(queued_candidates) + [x["params"] for x in retained]
    raw_seen = {candidate_fingerprint(_canonical_candidate(p, descriptor)) for p in blocked
                if candidate_error(p, descriptor) is None}
    decoded_seen = {decoded_fingerprint(p) for p in blocked} - {None}
    parent_map = {p["trial_number"]: p for p in parents}
    used = set(used_parents)
    available_parent_ids = {decoded_fingerprint(p["params"]) for p in parents} - {None}
    for index, entry in enumerate(batch.candidates):
        def reject(code, message):
            """Append a rejection for the current response entry.

            Parameters
            ----------
            code : str
                Stable validation failure category.
            message : str
                Candidate-specific explanation returned in refill feedback.

            Returns
            -------
            None
                Appends to this validation call's rejection list.
            """
            rejected.append(CandidateRejection(index, code, message, dict(entry)))
        intent = entry.get("intent")
        allowed = {"params", "rationale", "intent", "parent_trial", "intended_change", "target_cell"}
        if (set(entry) - allowed or not isinstance(entry.get("params"), dict)
                or not isinstance(entry.get("rationale"), str)
                or not entry["rationale"].strip() or len(entry["rationale"]) > 2000
                or intent not in ("local", "exploration")):
            reject("invalid_policy_envelope", "Require params, rationale, and local/exploration intent.")
            continue
        params = entry["params"]
        error = candidate_error(params, descriptor)
        if error:
            reject(*error); continue
        params = _canonical_candidate(params, descriptor)
        raw, decoded = candidate_fingerprint(params), decoded_fingerprint(params)
        if raw in raw_seen or (decoded is not None and decoded in decoded_seen):
            matches = [p for p in history_matches if
                       candidate_fingerprint(p["params"]) == raw or
                       (decoded is not None and decoded_fingerprint(p["params"]) == decoded)]
            # Keep every matched trial ID while bounding repeated architecture details.
            feedback = {"matching_trials": [p["trial"] for p in matches],
                        "match_details": matches[:3],
                        "additional_details_in_full_history": max(0, len(matches) - 3)}
            reject("duplicate_raw" if raw in raw_seen else "duplicate_decoded",
                   "Matching terminal/pending/retained architecture: " + json.dumps(feedback, default=str, separators=(",", ":")))
            continue
        if remaining.get(intent, 0) <= 0:
            reject("quota_overflow", f"No remaining {intent} slots."); continue
        metadata = dict(entry, params=params)
        if intent == "local":
            parent_number = entry.get("parent_trial")
            parent = parent_map.get(parent_number) if isinstance(parent_number, int) and not isinstance(parent_number, bool) else None
            if parent is None or isinstance(entry.get("parent_trial"), bool):
                reject("invalid_parent", "Local candidates require an eligible feasible historical trial."); continue
            parent_id = decoded_fingerprint(parent["params"])
            if distinct_parents and parent_id in used and available_parent_ids - used:
                reject("parent_reused", "Choose a distinct decoded feasible parent while one remains available."); continue
            metadata["parent_diversity_exception"] = parent_id in used
            used.add(parent_id)
        elif entry.get("parent_trial") is not None:
            reject("invalid_parent", "Exploration candidates must use a null parent_trial."); continue
        elif targeted:
            target = next((t for t in targets if t["cell_id"] == candidate_cell(params)), None)
            if target is None:
                reject("wrong_exploration_target", "Candidate must belong to a remaining Python-assigned cell."); continue
            metadata["assigned_target"] = target
            targets = list(targets); targets.remove(target)
        remaining[intent] -= 1
        accepted.append(metadata); raw_seen.add(raw)
        if decoded is not None: decoded_seen.add(decoded)
    return tuple(accepted), tuple(rejected)
