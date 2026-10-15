# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Bounded, provider-neutral prompt construction for candidate generation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any

from .phase_policy import PhaseState, resolve_phase
from .provider import LLMRequest
from .objective_renderer import render_objective
from .search_space import SearchParam, SearchSpaceDescriptor

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

FULL_HISTORY_SYSTEM = (
    "You are the optimizer for a hardware-in-the-loop neural architecture search. "
    "Return only the requested JSON envelope. "
    "Use exact raw Optuna parameters inside the declared search space. The columnar "
    "history includes every finished trial (complete, failed, or pruned). Row values "
    "follow history_columns; pending uses the same columns. Missing values are unknown, "
    "not zero. If failure_reason_encoding is dictionary_index, failure_reason integers "
    "index history.failure_reasons. Infrastructure and unclassified failures are not "
    "architecture evidence. "
    "Failure or infeasibility penalty scores are not measured task quality. "
    "Raw parameter aliases can decode to the same network; avoid both exact and decoded "
    "duplicates. widths lists actual pointwise channels per block: base_channels is "
    "scaled by pointwise_scale, and raising a nonbinding max_channels changes nothing. "
    "Training observations are noisy: a single best score is not proof of "
    "superiority. Local candidates must name a feasible parent and explain an actual "
    "change; select from eligible_parents and use distinct decoded parents when possible. "
    "Exploration candidates must match assigned num_blocks and per-block (time, frequency) "
    "strides, using a listed schedule alias, while compulsory coverage "
    "is active; assigned targets take precedence over generic phase advice. "
    "After coverage is complete, use outcomes and uncertainty to choose "
    "exploration rather than rotating through poorly performing cells indefinitely. "
    "Coverage counts are evidence counts, not predictions of quality. Rationales may "
    "express uncertainty; improvement is not guaranteed. During refill the caller "
    "retains accepted candidates: return only replacements for missing slots."
)


def _columnar_coverage(coverage: dict[str, Any]) -> dict[str, Any]:
    """Remove repeated coverage field names without dropping observations.

    Parameters
    ----------
    coverage : dict
        Python-computed coverage used internally by allocation and validation.

    Returns
    -------
    dict
        Equivalent tables with one column header and ordered row arrays.
    """
    result = dict(coverage)
    for name in ("cells", "width_signatures", "depth_first_width"):
        records = result.get(name, [])
        columns = list(records[0]) if records else []
        result[name] = {"columns": columns,
                        "rows": [[record.get(key) for key in columns] for record in records]}
    return result


def _param_schema(
    param: SearchParam,
    semantics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert one declaration to a JSON-safe prompt schema."""
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
    """Small bounded context passed to the MVP prompt builder."""

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
        """Resolve the deterministic current budget phase."""
        return resolve_phase(
            self.attempted_trials, self.max_total_attempts,
            feasible_completed=self.feasible_completed_trials,
            target_feasible=self.target_feasible_trials,
        )

    def as_prompt_payload(self) -> dict[str, Any]:
        """Return the complete JSON-safe prompt and ledger context."""
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
        if self.full_history_mode:
            if phase.name == "finalization":
                payload["trial_budget"]["phase_instruction"] = (
                    "Refine promising regions with new decoded networks; do not repeat "
                    "an existing network to confirm its score."
                )
            fixed = {key: value for key, value in payload.items() if key not in {
                "trial_budget", "recent_trials", "pending_trials", "anchors", "knowledge_base",
            }}
            fixed["output_contract"] = {
                "format": {"candidates": [{"params": "exact raw parameter dictionary",
                    "rationale": "nonempty hypothesis, at most 2000 characters",
                    "intent": "local or exploration", "parent_trial": "feasible trial number for local; null for exploration"}]},
                "exact_keys": list(self.descriptor), "json_only": True,
            }
            fixed["history_columns"] = self.full_history.get("columns", [])
            # Keep the objective readable without changing legacy prompt bytes.
            try:
                fixed["objective_summary"] = json.loads(self.objective_summary)
            except (ValueError, TypeError):
                pass
            objective_definition = fixed["objective_summary"]
            explanation = render_objective(objective_definition, fixed)
            if explanation is not None:
                fixed["objective_summary"] = explanation
            return {
                "objective_definition": objective_definition,
                "fixed_context": fixed,
                "history": {key: value for key, value in self.full_history.items()
                            if key not in {"columns", "pending"}},
                "current_request": {
                    "trial_budget": payload["trial_budget"],
                    "candidate_count": self.batch_size,
                    "allocation": self.allocation,
                    "coverage": _columnar_coverage(self.coverage),
                    "exploration_targets": list(self.exploration_targets),
                    "eligible_parents": [parent["trial_number"] for parent in self.eligible_parents],
                    "pending": self.full_history.get("pending", []),
                },
            }
        return payload


def build_candidate_request(
    context: PromptContext,
    *,
    prompt_version: str,
    repair_feedback: str | None = None,
) -> LLMRequest:
    """Build a deterministic JSON-only provider request."""
    if not prompt_version.strip():
        raise ValueError("prompt_version must be a non-empty string.")
    payload = context.as_prompt_payload()
    if context.full_history_mode:
        if repair_feedback:
            payload["current_request"]["repair_feedback"] = repair_feedback
        return LLMRequest(
            system_prompt=FULL_HISTORY_SYSTEM + "\n\n" + json.dumps(
                payload["fixed_context"], sort_keys=True, separators=(",", ":")),
            user_prompt=json.dumps({"history": payload["history"],
                                    "current_request": payload["current_request"]},
                                   separators=(",", ":")),
            prompt_version=prompt_version,
            metadata={"purpose": "candidate_generation", "study_name": context.study_name,
                      "phase": context.phase().name, "batch_size": context.batch_size,
                      "repair": bool(repair_feedback), "history_mode": "full"},
        )
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


def enrich_full_history(context: PromptContext, history: Any) -> PromptContext:
    """Attach exact evidence and the next-study allocation to a shared context.

    Parameters
    ----------
    context : PromptContext
        Semantic and budget context used by both campaign and bounded smoke.
    history : sequence
        Detached trial records; never reads or writes a live study.

    Returns
    -------
    PromptContext
        Full-history context with Python-assigned targets and feasible parents.

    Raises
    ------
    ValueError
        If the model family is unsupported or no target slots are available.
    """
    from .full_history import (build_full_history, build_coverage,
                               assign_coverage_targets, build_eligible_parents)
    from .phase_policy import resolve_allocation

    if context.model_family != "audio_dscnn":
        raise ValueError("Full-history decoded policy currently supports audio_dscnn only.")
    parents = build_eligible_parents(history)
    coverage = build_coverage(history, context.descriptor)
    allocation = resolve_allocation(context.phase(), context.batch_size,
                                    has_feasible_parent=bool(parents))
    targets = (() if coverage["goal_complete"] else
               tuple(assign_coverage_targets(coverage, allocation["exploration"])))
    if not coverage["goal_complete"]:
        coverage["unassigned_exploration_slots"] = allocation["exploration"] - len(targets)
        coverage["unassigned_reason"] = ("coverage_goal_slots_already_reserved"
            if coverage["unassigned_exploration_slots"] else None)
        allocation["exploration"] = len(targets)
    size = sum(allocation.values())
    if not size:
        raise ValueError("Coverage slots are already pending; resolve pending work before requesting more.")
    return replace(context, full_history_mode=True, require_rationale=True,
                   full_history=build_full_history(history), coverage=coverage,
                   allocation=allocation, exploration_targets=targets,
                   eligible_parents=parents, batch_size=size,
                   recent_trials=(), pending_trials=(), anchors=(), knowledge_base={})
