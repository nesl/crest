# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Serialize bounded search evidence into provider-neutral completion requests.

PromptContext carries search-space declarations, objective text, budgets, recent
trials, anchors, and optional semantic context. build_candidate_request combines
that evidence with phase guidance, the candidate envelope, and repair feedback.
It performs no provider calls and does not validate returned candidates; those
responsibilities belong to provider and schemas respectively."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any

from .phase_policy import PhaseState, resolve_phase
from .provider import LLMRequest
from .objective_renderer import render_objective
from ..search_space import SearchParam, SearchSpaceDescriptor

SYSTEM_PROMPT = (
    "You are the CREST candidate generator. Return exact raw Optuna trial parameters "
    "inside the supplied search space. Do not invent decoded or build-time fields. "
    "The knowledge_base summary carries older evidence; pending_trials are unsummarized "
    "results outside the recent window. Anchors retain the scalar best candidates or "
    "Pareto knee-region representatives across the full study. Trial numbers shared by "
    "anchors, recent_trials, pending_trials, "
    "or summarized evidence refer to the same observation; do not count this as independent "
    "confirmation. Treat memory as fallible observations, "
    "not hard constraints; retain its conditions and uncertainty."
)





def _param_schema(
    param: SearchParam,
    semantics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert a raw declaration and optional meaning to a prompt schema.

    Parameters
    ----------
    param : SearchParam
        Active raw parameter declaration.
    semantics : dict or None, optional
        Whitelisted semantic fields added to the distribution schema.

    Returns
    -------
    dict
        JSON-safe distribution kind, bounds or choices, and supplied semantics.
    """
    schema: dict[str, Any] = {"kind": param.kind}
    if param.kind == "categorical":
        schema["choices"] = list(param.choices or ())
    else:
        schema["low"] = param.low
        schema["high"] = param.high
    if semantics:
        schema.update(semantics)
    return schema


@dataclass(frozen=True)
class PromptContext:
    """Carry immutable prompt inputs for bounded or full-history generation.

    Parameters
    ----------
    study_name : str
        Study identifier retained in prompt and ledger metadata.
    model_family : str
        Registered model-family name.
    descriptor : SearchSpaceDescriptor
        Exact legal raw parameter names and distributions.
    objective_summary : str
        Human-readable score or objective contract.
    attempted_trials : int
        Total attempted trials including failures and warm-up.
    feasible_completed_trials : int
        Successful completions counted toward the runner target.
    target_feasible_trials : int
        Positive target for feasible completions.
    max_total_attempts : int
        Positive total attempt cap.
    batch_size : int
        Maximum candidates requested in this round.
    task_context, device_context, runtime_context : dict, optional
        Legacy explicit semantic context sections.
    recent_trials : tuple of dict, optional
        Newest bounded study evidence records.
    pending_trials : tuple of dict, optional
        Unsummarized evidence exposed alongside recent trials.
    anchors : tuple of dict, optional
        Representative successful historical candidates.
    knowledge_base : dict, optional
        Accepted persistent memory summary and coverage status.
    semantic_context : dict, optional
        Versioned canonical semantic record.
    require_rationale : bool, optional
        Require rationale envelopes for generated candidates.
    full_history_mode : bool, optional
        Use exact columnar history and quota-constrained proposals.
    full_history : dict, optional
        Full-history records and pending work.
    coverage : dict, optional
        Decoded coverage-cell attempt and feasibility counts.
    allocation : dict of str to int, optional
        Remaining local and exploration candidate quotas.
    exploration_targets : tuple of dict, optional
        Python-assigned coverage cells for exploration slots.
    eligible_parents : tuple of dict, optional
        Feasible historical architectures available for local proposals.
    """

    study_name: str
    model_family: str
    descriptor: SearchSpaceDescriptor
    objective_summary: str
    attempted_trials: int
    feasible_completed_trials: int
    target_feasible_trials: int
    max_total_attempts: int
    batch_size: int
    task_context: dict[str, Any] = field(default_factory=dict)
    device_context: dict[str, Any] = field(default_factory=dict)
    runtime_context: dict[str, Any] = field(default_factory=dict)
    recent_trials: tuple[dict[str, Any], ...] = ()
    pending_trials: tuple[dict[str, Any], ...] = ()
    anchors: tuple[dict[str, Any], ...] = ()
    knowledge_base: dict[str, Any] = field(default_factory=dict)
    semantic_context: dict[str, Any] = field(default_factory=dict)

    require_rationale: bool = False
    full_history_mode: bool = False
    full_history: dict[str, Any] = field(default_factory=dict)
    coverage: dict[str, Any] = field(default_factory=dict)
    allocation: dict[str, int] = field(default_factory=dict)
    exploration_targets: tuple[dict[str, Any], ...] = ()
    eligible_parents: tuple[dict[str, Any], ...] = ()

    def phase(self) -> PhaseState:
        """Resolve the phase from attempt and feasible-completion progress.

        Returns
        -------
        PhaseState
            Deterministic phase and actual remaining attempt counters.

        Raises
        ------
        ValueError
            If budget counters or limits are invalid.
        """
        return resolve_phase(
            self.attempted_trials, self.max_total_attempts,
            feasible_completed=self.feasible_completed_trials,
            target_feasible=self.target_feasible_trials,
        )

    def as_prompt_payload(self) -> dict[str, Any]:
        """Render semantic context, budget, and evidence as JSON-safe prompt data.

        Returns
        -------
        dict
            Bounded prompt payload or full-history fixed/history/request sections.

        Raises
        ------
        ValueError
            If budget counters or phase limits are invalid.
        """
        phase = self.phase()
        semantic_record = self.semantic_context
        semantic_payload = semantic_record.get("payload", {}) if semantic_record else {}
        parameter_semantics = semantic_payload.get("parameter_semantics", {})
        payload = {
            "study_name": self.study_name,
            "model_family": self.model_family,
            "search_space": {
                name: _param_schema(
                    self.descriptor[name],
                    parameter_semantics.get(name),
                )
                for name in self.descriptor
            },
            "objective_summary": self.objective_summary,
            "trial_budget": {
                "attempted": self.attempted_trials,
                "feasible_completed": self.feasible_completed_trials,
                "target_feasible": self.target_feasible_trials,
                "max_total_attempts": self.max_total_attempts,
                "remaining_attempts": phase.remaining,
                "batch_size": self.batch_size,
                "phase": phase.name,
                "phase_instruction": phase.instruction,
            },
            "recent_trials": list(self.recent_trials),
            "pending_trials": list(self.pending_trials),
            "anchors": list(self.anchors),
            "knowledge_base": self.knowledge_base,
            "output_contract": {
                "format": {"candidates": ["exact raw parameter dictionary"]},
                "candidate_count": self.batch_size,
                "exact_keys": list(self.descriptor),
                "no_extra_keys": True,
                "json_only": True,
            },
        }
        if self.require_rationale:
            payload["output_contract"]["format"] = {"candidates": [{
                "params": "exact raw parameter dictionary",
                "rationale": "nonempty explanation, at most 2000 characters",
            }]}
            payload["output_contract"]["rationale_instruction"] = (
                "For each architecture explain the design hypothesis, expected objective "
                "tradeoffs, and relevant evidence or uncertainty. Keep this concise. "
                "Place exactly the search-space keys inside params."
            )
        if semantic_record:
            payload["semantic_context"] = {
                "enabled": bool(semantic_record["enabled"]),
                "version": str(semantic_record["version"]),
                "hash": str(semantic_record["hash"]),
            }
            if semantic_record["enabled"]:
                payload["semantic_context"]["guidance"] = list(
                    semantic_payload.get("guidance", [])
                )
                for field_name in (
                    "dataset_context",
                    "task_context",
                    "model_context",
                    "device_context",
                    "runtime_context",
                ):
                    context_value = semantic_payload.get(field_name)
                    if context_value:
                        payload[field_name] = context_value
        else:
            # Preserve the direct PromptContext API for existing callers.
            payload["task_context"] = self.task_context
            payload["device_context"] = self.device_context
            payload["runtime_context"] = self.runtime_context
        return payload


def build_candidate_request(
    context: PromptContext,
    *,
    prompt_version: str,
    repair_feedback: str | None = None,
) -> LLMRequest:
    """Build a deterministic JSON-only candidate-generation request.

    Parameters
    ----------
    context : PromptContext
        Search space, semantic facts, study evidence, and budget.
    prompt_version : str
        Nonempty version identifying the prompt policy.
    repair_feedback : str or None, optional
        Validation feedback for a repaired or refilled request.

    Returns
    -------
    LLMRequest
        Provider-neutral messages and reproducibility metadata.

    Raises
    ------
    ValueError
        If the prompt version is empty or context budget counters are invalid.
    """
    if not prompt_version.strip():
        raise ValueError("prompt_version must be a non-empty string.")
    payload = context.as_prompt_payload()
    if repair_feedback:
        payload["repair_feedback"] = repair_feedback
    metadata = {
        "purpose": "candidate_generation",
        "study_name": context.study_name,
        "phase": context.phase().name,
        "batch_size": context.batch_size,
        "repair": bool(repair_feedback),
    }
    if context.semantic_context:
        metadata.update(
            {
                "semantic_context_enabled": bool(context.semantic_context["enabled"]),
                "semantic_context_version": str(context.semantic_context["version"]),
                "semantic_context_hash": str(context.semantic_context["hash"]),
            }
        )
    return LLMRequest(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=json.dumps(payload, sort_keys=True, separators=(",", ":")),
        prompt_version=prompt_version,
        metadata=metadata,
    )

