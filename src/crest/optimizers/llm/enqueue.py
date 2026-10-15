# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Generate and validate proposals; queue mutation belongs to the runner."""

from __future__ import annotations

import json
import random
from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from ...pipeline_types import CandidateProposal, TrialRecord
from .ledger import LLMLedger
from .prompt_builder import PromptContext, build_candidate_request
from .provider import LLMProvider, ProviderResponseError
from .schemas import CandidateBatch, validate_candidate_batch
from .search_space import SearchParam, SearchSpaceDescriptor


def _history_candidates(
    history: tuple[TrialRecord, ...],
    descriptor: SearchSpaceDescriptor,
    states: set[str],
) -> list[dict[str, Any]]:
    """Extract full candidates while leaving infrastructure failures retryable.

    Parameters
    ----------
    history : tuple[TrialRecord, ...]
        Detached study evidence, including queued and terminal trials.
    descriptor : SearchSpaceDescriptor
        Active parameter declarations.
    states : set[str]
        Uppercase Optuna state names to include.

    Returns
    -------
    list[dict[str, Any]]
        Full descriptor-shaped candidates reconstructed from proposal metadata,
        queued fixed parameters, and actual sampled parameters, in that order.
    """
    candidates = []
    for trial in history:
        state = str(trial.state).strip().upper()
        if state not in states:
            continue
        failure_class = trial.user_attrs.get("measurement_failure_class")
        if state != "WAITING" and failure_class == "infrastructure":
            continue
        # A failed process or recovered orphan says nothing about this design.
        # Only explicit deterministic rejection evidence blocks a FAIL retry.
        if state == "FAIL" and not (
            failure_class == "deterministic"
            or trial.user_attrs.get("candidate_failure_class") == "deterministic"
            or trial.user_attrs.get("pruned") is True
            or trial.user_attrs.get("feasibility_status") == "infeasible"
        ):
            continue
        params: dict[str, Any] = {}
        provenance = trial.user_attrs.get("crest_proposal", {})
        proposed = provenance.get("proposal_params", {}) if isinstance(provenance, Mapping) else {}
        fixed = trial.system_attrs.get("fixed_params", {})
        for source in (proposed, fixed, trial.params):
            if isinstance(source, Mapping):
                params.update(source)
        if all(name in params for name in descriptor):
            candidates.append({name: params[name] for name in descriptor})
    return candidates


def _sample_param(param: SearchParam, rng: random.Random) -> Any:
    """Sample one locally valid fallback value from a declaration."""
    if param.kind == "int":
        return rng.randint(param.low, param.high)
    if param.kind == "float":
        return rng.uniform(float(param.low), float(param.high))
    return rng.choice(param.choices)


def sample_random_candidate(
    descriptor: SearchSpaceDescriptor,
    *,
    rng: random.Random,
) -> dict[str, Any]:
    """Sample one raw candidate from exactly the active descriptor."""
    return {name: _sample_param(descriptor[name], rng) for name in descriptor}


def generate_llm_batch(
    history: tuple[TrialRecord, ...],
    provider: LLMProvider,
    context: PromptContext,
    ledger: LLMLedger,
    *,
    prompt_version: str,
    max_repair_attempts: int,
    rng: random.Random,
) -> tuple[CandidateProposal, ...]:
    """Return validated candidates with source provenance and one fallback."""
    completed = _history_candidates(history, context.descriptor, {"COMPLETE", "PRUNED", "FAIL"})
    queued = _history_candidates(history, context.descriptor, {"WAITING"})
    repair_feedback = None

    for attempt in range(max_repair_attempts + 1):
        request_id = ledger.next_request_id()
        request = build_candidate_request(
            context,
            prompt_version=prompt_version,
            repair_feedback=repair_feedback,
        )
        prompt_payload = context.as_prompt_payload()
        prompt_payload.update({"request_id": request_id, "repair_attempt": attempt,
                               "repair_feedback": repair_feedback})
        ledger.record_prompt_context(prompt_payload)
        ledger.write_request(request_id, request)
        try:
            response = provider.complete_json(request)
        except Exception as exc:
            ledger.write_response(
                request_id,
                {"error_type": type(exc).__name__, "error": str(exc)},
            )
            ledger.record_event(
                {"event": "provider_error", "request_id": request_id, "repair_attempt": attempt}
            )
            if isinstance(exc, ProviderResponseError):
                repair_feedback = f"The prior provider response was invalid: {exc}. Return valid JSON."
                continue
            raise
        ledger.write_response(request_id, response)

        try:
            raw_envelope = json.loads(response.content)
        except (TypeError, json.JSONDecodeError) as exc:
            ledger.record_rejected(
                {
                    "request_id": request_id,
                    "code": "invalid_json",
                    "message": str(exc),
                    "raw_content": response.content,
                }
            )
            repair_feedback = "The prior response was not valid JSON. Return only the required JSON envelope."
            continue
        try:
            batch = CandidateBatch.model_validate(raw_envelope)
        except ValidationError as exc:
            ledger.record_rejected(
                {
                    "request_id": request_id,
                    "code": "schema_validation",
                    "message": str(exc),
                    "raw_content": response.content,
                }
            )
            repair_feedback = "The prior JSON did not match the required CandidateBatch envelope."
            continue

        validation = validate_candidate_batch(
            batch,
            context.descriptor,
            batch_size=context.batch_size,
            completed_candidates=completed,
            queued_candidates=queued,
            require_rationale=context.require_rationale,
        )
        for rejection in validation.rejected:
            ledger.record_rejected({"request_id": request_id, **rejection.__dict__})
        if not validation.accepted:
            repair_feedback = "The prior candidates were rejected: " + "; ".join(
                f"{item.code}: {item.message}" for item in validation.rejected
            )
            continue

        proposals = tuple(CandidateProposal(candidate, {
            "proposal_source": "llm", "optimizer": "llm_generator",
            "request_id": request_id, "batch_id": f"llm-{request_id:06d}",
            "batch_index": index,
            **({"rationale": validation.rationales[index]}
               if validation.rationales[index] is not None else {}),
        }) for index, candidate in enumerate(validation.accepted))
        for proposal in proposals:
            ledger.record_returned({**proposal.provenance, "candidate": proposal.params})
        ledger.record_event({
            "event": "batch_returned_to_runner", "schema_version": 2,
            "request_id": request_id, "accepted": len(proposals),
            "rejected": len(validation.rejected),
        })
        return proposals

    for _ in range(100):
        candidate = sample_random_candidate(context.descriptor, rng=rng)
        validation = validate_candidate_batch(
            CandidateBatch(candidates=[candidate]),
            context.descriptor,
            batch_size=1,
            completed_candidates=completed,
            queued_candidates=queued,
        )
        if validation.accepted:
            candidate = validation.accepted[0]
            proposal = CandidateProposal(candidate, {
                "proposal_source": "random_fallback", "optimizer": "llm_generator",
                "request_id": request_id, "batch_id": f"llm-{request_id:06d}",
                "batch_index": 0,
            })
            ledger.record_returned({**proposal.provenance, "candidate": candidate})
            ledger.record_event({"event": "random_fallback", "schema_version": 2,
                                 "status": "returned_to_runner", "request_id": request_id,
                                 "candidate": candidate})
            return (proposal,)
    raise RuntimeError("Unable to sample a unique random fallback candidate after 100 attempts.")
