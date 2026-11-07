# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Validate proposal batches and run them through the shared Optuna objective."""
from collections.abc import Mapping
import json
from uuid import uuid4

from crest.pipeline_types import CandidateProposal, ExplicitRound, NativeRound
from .history import build_trial_snapshot
from .search_space import validate_candidate
from .study_state import PROPOSAL_ATTR


def prepare_explicit_round(proposal, descriptor, budget, name):
    """Validate every candidate and its provenance before reserving any trials.

    Returns detached parameter/metadata pairs. A malformed later candidate must
    not leave an earlier candidate queued in the study.
    """
    if not isinstance(proposal.candidates, tuple) or not proposal.candidates:
        raise ValueError("ExplicitRound requires a nonempty ordered tuple of candidates.")
    size = len(proposal.candidates)
    if size > budget.permitted_round_size:
        raise ValueError("ExplicitRound exceeds the permitted round size.")
    if descriptor is None:
        raise ValueError("Explicit rounds require a declared raw search-space descriptor.")
    round_id = uuid4().hex
    validated = []
    for index, candidate in enumerate(proposal.candidates):
        if not isinstance(candidate, CandidateProposal) or not isinstance(candidate.params, Mapping):
            raise ValueError("ExplicitRound members must be CandidateProposal parameter mappings.")
        validate_candidate(candidate.params, descriptor)
        params = dict(candidate.params)
        if not isinstance(candidate.provenance, Mapping):
            raise ValueError("Candidate provenance must be a JSON-safe mapping.")
        metadata = dict(candidate.provenance)
        metadata.update(optimizer=name, round_id=round_id, batch_index=index, proposal_params=dict(params))
        metadata.setdefault("proposal_source", name)
        metadata = json.loads(json.dumps(metadata, sort_keys=True, allow_nan=False))
        validated.append((dict(params), metadata))
    return validated


def execute_search_round(study, optimizer, name, descriptor, budget, *, objective,
                         callbacks, catch):
    """Ask for a round, validate it, reserve explicit candidates, then evaluate.

    Native rounds let Optuna sample inside the objective. Explicit rounds reserve
    fixed parameters first; both use the same evaluation and callback lifecycle.
    """
    proposal = optimizer.propose_round(build_trial_snapshot(study), budget)
    if isinstance(proposal, NativeRound):
        size = proposal.n_trials
        if isinstance(size, bool) or not isinstance(size, int) or not 0 < size <= budget.permitted_round_size:
            raise ValueError("NativeRound.n_trials must be positive and within the permitted round size.")
    elif isinstance(proposal, ExplicitRound):
        validated = prepare_explicit_round(proposal, descriptor, budget, name)
        size = len(validated)
        # Validation is all-or-nothing, but these database writes are individual
        # reservations, not one transaction. Resume drains WAITING trials first.
        for params, metadata in validated:
            study.enqueue_trial(params, user_attrs={PROPOSAL_ATTR: metadata})
    else:
        raise ValueError("Optimizer must return NativeRound or ExplicitRound.")
    study.optimize(objective, n_trials=size, callbacks=callbacks, catch=catch)
    return size
