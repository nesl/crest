# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Deterministic trial-budget phase policy for LLM prompts."""

from __future__ import annotations

from dataclasses import dataclass


PHASE_INSTRUCTIONS = {
    "exploration": "Prioritize diverse valid coverage while avoiding clearly infeasible regions.",
    "specification": "Compare promising regions and test focused open questions.",
    "finalization": "Refine or confirm strong candidates and avoid unsupported risky jumps.",
}


@dataclass(frozen=True)
class PhaseState:
    """Resolved phase and budget counters included in every prompt."""

    name: str
    instruction: str
    attempted: int
    max_attempts: int
    remaining: int


def resolve_phase(attempted: int, max_attempts: int) -> PhaseState:
    """Resolve exploration/specification/finalization from attempted budget."""
    if isinstance(attempted, bool) or not isinstance(attempted, int) or attempted < 0:
        raise ValueError("attempted must be a non-negative integer.")
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError("max_attempts must be an integer greater than or equal to 1.")
    bounded_attempted = min(attempted, max_attempts)
    fraction = bounded_attempted / max_attempts
    if fraction < 0.35:
        name = "exploration"
    elif fraction < 0.85:
        name = "specification"
    else:
        name = "finalization"
    return PhaseState(
        name=name,
        instruction=PHASE_INSTRUCTIONS[name],
        attempted=attempted,
        max_attempts=max_attempts,
        remaining=max(0, max_attempts - attempted),
    )
