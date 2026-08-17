# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Bounded, provider-neutral prompt construction for candidate generation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .phase_policy import PhaseState, resolve_phase
from .provider import LLMRequest
from .search_space import SearchParam, SearchSpaceDescriptor

SYSTEM_PROMPT = (
    "You are the CREST candidate generator. Return exact raw Optuna trial parameters "
    "inside the supplied search space. Do not invent decoded or build-time fields."
)


def _param_schema(param: SearchParam) -> dict[str, Any]:
    """Convert one declaration to a JSON-safe prompt schema."""
    schema: dict[str, Any] = {"kind": param.kind}
    if param.kind == "categorical":
        schema["choices"] = list(param.choices or ())
    else:
        schema["low"] = param.low
        schema["high"] = param.high
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
    anchors: tuple[dict[str, Any], ...] = ()
    knowledge_base: dict[str, Any] = field(default_factory=dict)

    def phase(self) -> PhaseState:
        """Resolve the deterministic current budget phase."""
        return resolve_phase(self.attempted_trials, self.max_total_attempts)

    def as_prompt_payload(self) -> dict[str, Any]:
        """Return the complete JSON-safe prompt and ledger context."""
        phase = self.phase()
        return {
            "study_name": self.study_name,
            "model_family": self.model_family,
            "search_space": {
                name: _param_schema(self.descriptor[name])
                for name in self.descriptor
            },
            "objective_summary": self.objective_summary,
            "task_context": self.task_context,
            "device_context": self.device_context,
            "runtime_context": self.runtime_context,
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
    if repair_feedback:
        payload["repair_feedback"] = repair_feedback
    return LLMRequest(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=json.dumps(payload, sort_keys=True, separators=(",", ":")),
        prompt_version=prompt_version,
        metadata={
            "study_name": context.study_name,
            "phase": context.phase().name,
            "batch_size": context.batch_size,
            "repair": bool(repair_feedback),
        },
    )
