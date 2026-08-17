# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Generate, validate, log, and enqueue one LLM candidate batch."""

from __future__ import annotations

import json
import random
from typing import Any

from pydantic import ValidationError

from .ledger import LLMLedger
from .prompt_builder import PromptContext, build_candidate_request
from .provider import LLMProvider
from .schemas import CandidateBatch, validate_candidate_batch
from .search_space import SearchParam, SearchSpaceDescriptor


def _study_candidates(study: Any, descriptor: SearchSpaceDescriptor, state_name: str) -> list[dict[str, Any]]:
    """Extract exact descriptor-shaped candidates in one Optuna state."""
    candidates = []
    for trial in study.trials:
        if str(getattr(getattr(trial, "state", None), "name", "")) != state_name:
            continue
        params = getattr(trial, "params", {})
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


def enqueue_llm_batch(
    study: Any,
    provider: LLMProvider,
    context: PromptContext,
    ledger: LLMLedger,
    *,
    prompt_version: str,
    max_repair_attempts: int,
    rng: random.Random,
) -> tuple[dict[str, Any], ...]:
    """Request candidates, reject invalid output, and guarantee one fallback."""
    completed = _study_candidates(study, context.descriptor, "COMPLETE")
    queued = _study_candidates(study, context.descriptor, "WAITING")

    for attempt in range(max_repair_attempts + 1):
        request_id = ledger.next_request_id()
        request = build_candidate_request(context, prompt_version=prompt_version)
        prompt_payload = context.as_prompt_payload()
        prompt_payload.update({"request_id": request_id, "repair_attempt": attempt})
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
            continue
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
            continue

        validation = validate_candidate_batch(
            batch,
            context.descriptor,
            batch_size=context.batch_size,
            completed_candidates=completed,
            queued_candidates=queued,
        )
        for rejection in validation.rejected:
            ledger.record_rejected({"request_id": request_id, **rejection.__dict__})
        if not validation.accepted:
            continue

        for candidate in validation.accepted:
            study.enqueue_trial(candidate)
            ledger.record_accepted(
                {"request_id": request_id, "source": "llm", "candidate": candidate}
            )
        ledger.record_event(
            {
                "event": "batch_enqueued",
                "request_id": request_id,
                "accepted": len(validation.accepted),
                "rejected": len(validation.rejected),
            }
        )
        return validation.accepted

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
            study.enqueue_trial(candidate)
            ledger.record_accepted({"source": "random_fallback", "candidate": candidate})
            ledger.record_event({"event": "random_fallback", "candidate": candidate})
            return validation.accepted
    raise RuntimeError("Unable to sample a unique random fallback candidate after 100 attempts.")
