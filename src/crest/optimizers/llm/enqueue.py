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
    max_refill_attempts: int = 2,
) -> tuple[CandidateProposal, ...]:
    """Return validated candidates with source provenance and one fallback."""
    if getattr(context, "full_history_mode", False):
        return _generate_policy_batch(history, provider, context, ledger, prompt_version=prompt_version,
                                      max_repair_attempts=max_repair_attempts,
                                      max_refill_attempts=max_refill_attempts, rng=rng)
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



def _policy_feedback(retained, rejections, *, malformed=False):
    """Build structurally bounded feedback without repeated history or rationales.

    Parameters
    ----------
    retained, rejections : object
        Accepted entries and validation failures. Missing slots and targets
        already appear in the current request.
    malformed : bool
        Add a constant envelope repair instruction, retaining prior rejections.

    Returns
    -------
    str
        Compact JSON feedback bounded independently of provider response text.
    """
    compact = []
    for rejection in rejections[:5]:
        item = {"index": rejection.index, "code": rejection.code}
        if rejection.code in {"duplicate_raw", "duplicate_decoded"}:
            detail = json.loads(rejection.message.split(": ", 1)[1])
            ids = detail["matching_trials"]
            item.update(matching_trials=ids[:8], additional_matches=max(0, len(ids) - 8))
        compact.append(item)
    return json.dumps({"retained_candidates": [{k: entry[k] for k in
                           ("params", "intent", "parent_trial", "assigned_target") if k in entry}
                           for entry in retained],
                       "rejections": compact, "additional_rejections": max(0, len(rejections) - 5),
                       **({"repair": "Return only the required valid CandidateBatch JSON envelope."}
                          if malformed else {})}, separators=(",", ":"), default=str)


def _generate_policy_batch(history, provider, context, ledger, *, prompt_version,
                           max_repair_attempts, max_refill_attempts, rng):
    """Generate a retained subset under shared repair and refill budgets.

    Parameters
    ----------
    history, provider, context, ledger : object
        Study snapshot and existing generation services.
    prompt_version : str
        Version recorded with each request.
    max_repair_attempts, max_refill_attempts : int
        Shared malformed-response and missing-slot budgets.
    rng : random.Random
        Existing explicit fallback random source.

    Returns
    -------
    tuple[CandidateProposal, ...]
        Valid retained proposals, or an explicitly attributed random fallback.
    """
    from dataclasses import replace
    from .schemas import validate_policy_batch
    from .full_history import (reconstruct_params, decoded_architecture, decoded_fingerprint,
                               parameter_differences, nearest_prior_distances, outcome_class, outcome_reason, observed_measurement)
    from .spending import SpendInputLimitExceeded
    if not (isinstance(max_repair_attempts, int) and not isinstance(max_repair_attempts, bool)
            and 0 <= max_repair_attempts <= 1
            and isinstance(max_refill_attempts, int) and not isinstance(max_refill_attempts, bool)
            and 0 <= max_refill_attempts <= 2):
        raise ValueError("Full history mode requires repair attempts <= 1 and refill attempts <= 2.")
    completed = []
    for trial in history:
        if outcome_class(trial) in {"successful", "deterministic", "infeasible"}:
            params = reconstruct_params(trial, context.descriptor)
            if all(name in params for name in context.descriptor) and params not in completed:
                completed.append(params)
    queued = []
    for trial in history:
        if str(trial.state).strip().upper() in {"WAITING", "RUNNING"}:
            params = reconstruct_params(trial, context.descriptor)
            if all(name in params for name in context.descriptor):
                queued.append(params)
    matches = []
    for trial in history:
        params = reconstruct_params(trial, context.descriptor)
        if params in completed or params in queued:
            attrs = trial.user_attrs
            metrics = attrs.get("task_metrics", {})
            decoded = decoded_architecture(params)
            reason = outcome_reason(trial)
            if reason is not None:
                reason = str(reason)
                if len(reason) > 512:
                    reason = reason[:512] + " [truncated failure detail]"
            matches.append({"trial": trial.number, "params": params,
                            "decoded_widths": decoded["widths"] if decoded else None,
                            "state": str(trial.state), "outcome": {
                                "score": trial.values[0] if trial.values and len(trial.values) == 1 else None,
                                "macro_f1": metrics.get("macro_f1") if isinstance(metrics, Mapping) else None,
                                "latency_ms": observed_measurement(trial, "latency_ms"),
                                "energy_mj_per_inference": observed_measurement(trial, "energy_mj_per_inference"),
                                "feasibility": attrs.get("feasibility_status", attrs.get("feasible")),
                                "failure_class": None if outcome_class(trial) in {"successful", "pending"} else outcome_class(trial), "failure_reason": reason}})
    retained, proposals, targets = [], [], list(context.exploration_targets)
    quotas = dict(context.allocation)
    repairs = refills = 0
    feedback = None
    used_parents = set()
    stop_reason = None
    last_rejections = []
    while True:
        current = replace(context, batch_size=sum(quotas.values()), allocation=dict(quotas),
                          exploration_targets=tuple(targets))
        request_id = ledger.next_request_id()
        request = build_candidate_request(current, prompt_version=prompt_version, repair_feedback=feedback)
        ledger.record_prompt_context({**current.as_prompt_payload(), "request_id": request_id,
                                     "repair_attempt": repairs, "refill_attempt": refills,
                                     "retained_candidates": retained, "repair_feedback": feedback})
        ledger.write_request(request_id, request)
        malformed = None
        try:
            response = provider.complete_json(request)
        except Exception as exc:
            ledger.write_response(request_id, {"error_type": type(exc).__name__, "error": str(exc),
                                              **{name: getattr(exc, name) for name in
                                                 ("raw_response", "wire_payload", "usage", "spending")
                                                 if getattr(exc, name, None) is not None}})
            ledger.record_event({"event": "provider_error", "request_id": request_id})
            if isinstance(exc, SpendInputLimitExceeded) and proposals:
                stop_reason = "input_guard"
                break
            if not isinstance(exc, ProviderResponseError): raise
            malformed = str(exc)
        else:
            ledger.write_response(request_id, response)
            try:
                batch = CandidateBatch.model_validate(json.loads(response.content))
            except (ValueError, TypeError, ValidationError) as exc:
                malformed = str(exc)
        if malformed is not None:
            ledger.record_rejected({"request_id": request_id, "code": "malformed_response", "message": malformed})
            if repairs >= max_repair_attempts:
                stop_reason = "repair_budget_exhausted"
                break
            repairs += 1
            feedback = _policy_feedback(retained, last_rejections, malformed=True)
            continue
        accepted, rejected = validate_policy_batch(
            batch, context.descriptor, quotas=quotas, targets=targets, parents=context.eligible_parents,
            completed_candidates=completed, queued_candidates=queued, retained=retained,
            used_parents=used_parents, history_matches=matches)
        for rejection in rejected:
            ledger.record_rejected({"request_id": request_id, **rejection.__dict__})
        for index, entry in enumerate(accepted):
            intent = entry["intent"]
            quotas[intent] -= 1
            if "assigned_target" in entry: targets.remove(entry["assigned_target"])
            provenance = {"proposal_source": "llm", "optimizer": "llm_generator",
                          "request_id": request_id, "batch_id": f"llm-{request_id:06d}",
                          "batch_index": index, **{k: v for k, v in entry.items() if k != "params"},
                          "nearest_prior_distances": nearest_prior_distances(entry["params"], history)}
            if intent == "exploration":
                provenance["exploration_policy"] = ("compulsory_coverage" if "assigned_target" in entry
                    else "post_coverage" if context.coverage.get("goal_complete") else "untargeted")
            if intent == "local":
                parent = next(p for p in context.eligible_parents if p["trial_number"] == entry["parent_trial"])
                used_parents.add(decoded_fingerprint(parent["params"]))
                provenance["parent_differences"] = parameter_differences(entry["params"], parent["params"])
                if entry.get("parent_diversity_exception"):
                    ledger.record_event({"event": "parent_diversity_exception", "request_id": request_id,
                                         "parent_trial": entry["parent_trial"]})
            retained.append(entry)
            proposals.append(CandidateProposal(entry["params"], provenance))
        if sum(quotas.values()) == 0:
            break
        if refills >= max_refill_attempts:
            stop_reason = "refill_budget_exhausted"
            break
        refills += 1
        last_rejections = rejected
        feedback = _policy_feedback(retained, rejected)
    ledger.record_event({"event": "batch_policy_exhausted" if sum(quotas.values()) else "batch_policy_filled",
                         "accepted": len(proposals), "unfilled": quotas, "unfilled_targets": targets,
                         "unfilled_reason": stop_reason if sum(quotas.values()) else None,
                         "repair_calls": repairs, "refill_calls": refills})
    if proposals:
        for proposal in proposals:
            ledger.record_returned({**proposal.provenance, "candidate": proposal.params})
        return tuple(proposals)
    # Reuse the legacy fallback path without making another content request.
    for _ in range(100):
        candidate = sample_random_candidate(context.descriptor, rng=rng)
        valid, _ = validate_policy_batch(
            CandidateBatch(candidates=[{"params": candidate, "rationale": "Explicit random fallback", "intent": "exploration"}]),
            context.descriptor, quotas={"exploration": 1}, completed_candidates=completed,
            queued_candidates=queued)
        if valid:
            proposal = CandidateProposal(candidate, {"proposal_source": "random_fallback", "optimizer": "llm_generator",
                         "request_id": request_id, "batch_id": f"llm-{request_id:06d}", "batch_index": 0,
                         "fallback_reason": "no_valid_llm_candidates_after_bounded_requests"})
            ledger.record_returned({**proposal.provenance, "candidate": candidate})
            ledger.record_event({"event": "random_fallback", **proposal.provenance})
            return (proposal,)
    raise RuntimeError("Unable to sample a unique random fallback candidate after 100 attempts.")
