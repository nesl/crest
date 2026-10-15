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


def resolve_phase(
    attempted: int, max_attempts: int, *,
    feasible_completed: int = 0, target_feasible: int | None = None,
) -> PhaseState:
    """Resolve phase using progress toward either stopping limit.

    Parameters
    ----------
    attempted : int
        Attempts consumed, including warm-up and failures.
    max_attempts : int
        Total attempt limit.
    feasible_completed : int, optional
        Feasible completions counted by the runner.
    target_feasible : int or None, optional
        Feasible completion target. None retains attempt-only scheduling.

    Returns
    -------
    PhaseState
        Phase from the greater budget fraction, with actual attempt counters.
    """
    if isinstance(attempted, bool) or not isinstance(attempted, int) or attempted < 0:
        raise ValueError("attempted must be a non-negative integer.")
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError("max_attempts must be an integer greater than or equal to 1.")
    bounded_attempted = min(attempted, max_attempts)
    fraction = bounded_attempted / max_attempts
    if isinstance(feasible_completed, bool) or not isinstance(feasible_completed, int) or feasible_completed < 0:
        raise ValueError("feasible_completed must be a non-negative integer.")
    if target_feasible is not None:
        if isinstance(target_feasible, bool) or not isinstance(target_feasible, int) or target_feasible < 1:
            raise ValueError("target_feasible must be a positive integer or None.")
        fraction = max(fraction, min(feasible_completed / target_feasible, 1.0))
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


# Explicit local/exploration counts keep reduced batches scientifically defined.
PHASE_ALLOCATIONS = {
    "exploration": ((0, 1), (1, 1), (1, 2), (2, 2), (2, 3)),
    "specification": ((1, 0), (1, 1), (2, 1), (2, 2), (3, 2)),
    "finalization": ((1, 0), (1, 1), (2, 1), (3, 1), (4, 1)),
}


def resolve_allocation(phase: str | PhaseState, batch_size: int, *, has_feasible_parent: bool = True) -> dict[str, int]:
    """Resolve the explicit next-study intent quotas.

    Parameters
    ----------
    phase : str or PhaseState
        Existing budget phase.
    batch_size : int
        Requested candidates, from one through five.
    has_feasible_parent : bool, optional
        Whether a measured feasible historical parent is available.

    Returns
    -------
    dict[str, int]
        Local and exploration counts summing to the requested batch size.
    """
    name = phase.name if isinstance(phase, PhaseState) else phase
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 5:
        raise ValueError("next-study batch_size must be an integer from 1 through 5.")
    if name not in PHASE_ALLOCATIONS:
        raise ValueError(f"Unknown phase: {name}")
    local, exploration = PHASE_ALLOCATIONS[name][batch_size - 1]
    return {"local": local if has_feasible_parent else 0,
            "exploration": exploration if has_feasible_parent else batch_size}
