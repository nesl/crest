# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Choose deterministic prompt guidance from runner budget progress.

resolve_phase maps the greater of attempt progress and feasible-completion
progress to exploration, specification, or finalization. PhaseState retains the
actual remaining attempt budget for the prompt. These instructions influence
candidate suggestions only; the runner remains responsible for stopping limits
and no phase changes the legal search-space descriptor."""

from __future__ import annotations

from dataclasses import dataclass


PHASE_INSTRUCTIONS = {
    "exploration": "Prioritize diverse valid coverage while avoiding clearly infeasible regions.",
    "specification": "Compare promising regions and test focused open questions.",
    "finalization": "Refine or confirm strong candidates and avoid unsupported risky jumps.",
}


@dataclass(frozen=True)
class PhaseState:
    """Represent a resolved prompt phase and actual attempt counters.

    Parameters
    ----------
    name : str
        Exploration, specification, or finalization phase.
    instruction : str
        Phase-specific candidate-generation instruction.
    attempted : int
        Actual attempts already consumed.
    max_attempts : int
        Total attempt cap.
    remaining : int
        Nonnegative remaining attempt count.
    """

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

    Raises
    ------
    ValueError
        If counters are negative, limits are nonpositive, or values are not integers.
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


