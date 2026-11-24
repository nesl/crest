# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Persisted optimizer identity and explicit recovery of interrupted studies.

Attribute names are a storage contract: changing them would hide existing study
identity or provenance. Validate identity before initialization; write it only
once the optimizer and feasibility policy have initialized successfully.
"""
from __future__ import annotations

import json
import logging
import time

import optuna
from optuna.trial import TrialState
from typing import Any

from crest.search.training_randomness import training_randomness_policy

logger = logging.getLogger(__name__)

# Study-level identity and recovery audit trail.
OPTIMIZER_SIGNATURE_ATTR = "crest_optimizer_signature"
ORPHAN_TRIAL_RECOVERY_ATTR = "crest_orphan_trial_recovery"
# Trial-level source, round membership, and reserved parameter values.
PROPOSAL_ATTR = "crest_proposal"


def build_optimizer_signature(
    name,
    optimizer,
    config,
    sampler,
    sampler_options,
    training_config,
    evaluation_policy=None,
):
    """Describe effective settings as detached JSON data for strict resume checks."""
    options = dict(sampler_options)
    if "constraints_func" in options:
        options["constraints_func"] = "crest.persisted_feasibility_constraints.v1"
    public_sampler_names = {
        optuna.samplers.TPESampler: "optuna.samplers.TPESampler",
        optuna.samplers.NSGAIISampler: "optuna.samplers.NSGAIISampler",
    }
    sampler_name = public_sampler_names.get(
        type(sampler), type(sampler).__module__ + "." + type(sampler).__qualname__
    )
    signature = {
        "version": 1, "optimizer": name,
        "config": optimizer.identity_config(config),
        "sampler": {"class": sampler_name, "options": options},
    }
    # Round-trip rejects non-JSON settings and ensures detached ordinary data.
    training_policy = training_randomness_policy(training_config)
    if training_policy is not None:
        signature["training_randomness"] = training_policy
    if evaluation_policy is not None:
        signature["evaluation_policy"] = dict(evaluation_policy)
    signature = json.loads(json.dumps(signature, sort_keys=True, allow_nan=False))
    return signature


def validate_optimizer_identity(study, signature, optimizer):
    """Check stored identity without mutating the study or initializing a proposer."""
    stored = study.user_attrs.get(OPTIMIZER_SIGNATURE_ATTR)
    if stored is None:
        if study.trials:
            raise RuntimeError(
                "Existing nonempty study has no CREST optimizer signature and cannot be resumed. "
                "Select a new study name to continue searching; existing results remain available "
                "for analysis and plotting."
            )
        return
    # Only known version-1 private paths from the reviewed implementation alias
    # public names. Keep optimizer/config/options and all other fields strict.
    legacy_sampler_names = {
        "optuna.samplers._tpe.sampler.TPESampler": "optuna.samplers.TPESampler",
        "optuna.samplers.nsgaii._sampler.NSGAIISampler": "optuna.samplers.NSGAIISampler",
    }
    comparable = stored
    if isinstance(stored, dict) and isinstance(stored.get("sampler"), dict):
        stored_sampler = stored["sampler"]
        old_name = stored_sampler.get("class")
        if isinstance(old_name, str) and old_name in legacy_sampler_names:
            comparable = {**stored, "sampler": {
                **stored_sampler, "class": legacy_sampler_names[old_name],
            }}
    if (
        isinstance(comparable, dict)
        and isinstance(signature.get("evaluation_policy"), dict)
        and "evaluation_policy" not in comparable
    ):
        comparable = {
            **comparable,
            "evaluation_policy": {
                "hil_first": True,
                "training_batch_size": 256,
            },
        }
    comparable = optimizer.normalize_stored_identity(comparable, signature)
    if comparable != signature:
        raise RuntimeError(
            "Active optimizer or sampler settings do not match the existing study optimizer signature. "
            "Restore the original configuration or select a new study name."
        )


def store_optimizer_signature(study: optuna.Study, signature: dict[str, Any]) -> None:
    """Persist campaign identity after successful component initialization.

    Parameters
    ----------
    study : optuna.Study
        Validated study. An existing signature is left unchanged.
    signature : dict[str, Any]
        Validated JSON-safe identity to store for future resume checks.

    Notes
    -----
    The caller must validate the study before initialization. Nonempty unsigned
    studies are rejected rather than assigned an unverified historical identity.
    """
    if study.user_attrs.get(OPTIMIZER_SIGNATURE_ATTR) is not None:
        return
    study.set_user_attr(OPTIMIZER_SIGNATURE_ATTR, signature)


def recover_orphaned_trials(study: optuna.Study, config: Any) -> None:
    """Fail explicitly listed RUNNING trials after workers have stopped.

    Parameters
    ----------
    study : optuna.Study
        Validated, initialized single-writer study being resumed.
    config : Any
        Normalized optimizer config. ``recover_orphaned_trials: [12]``
        attests that the workers for those exact trial numbers have stopped.

    Returns
    -------
    None
        Existing attempts remain in the budget; recovery evidence is stored
        in study attributes before the public Optuna state transition.
        Reusing a target list cannot recover newly allocated trial numbers.

    Raises
    ------
    ValueError
        If a requested number is unknown or still queued for execution.
    """
    expected_numbers = list(config.recover_orphaned_trials)
    if not expected_numbers:
        return
    trials = study.get_trials(deepcopy=False)
    unknown = sorted(set(expected_numbers) - {trial.number for trial in trials})
    if unknown:
        raise ValueError(
            f"optimizer.recover_orphaned_trials contains trial numbers absent from this study: {unknown}. "
            "List only existing orphan trials after confirming their workers have stopped."
        )
    queued = [trial.number for trial in trials
              if trial.state == TrialState.WAITING and trial.number in expected_numbers]
    if queued:
        raise ValueError(
            f"optimizer.recover_orphaned_trials contains WAITING trial numbers: {queued}. "
            "Queued trials have not started; list only existing RUNNING or terminal trials."
        )
    running = [trial for trial in trials
               if trial.state == TrialState.RUNNING and trial.number in expected_numbers]
    if not running:
        logger.warning("[NAS] Orphan recovery targets %s contain no RUNNING trials; no state changed.",
                       expected_numbers)
        return
    events = list(study.user_attrs.get(ORPHAN_TRIAL_RECOVERY_ATTR, []))
    for trial in running:
        # Persist intent first so an interrupted recovery remains auditable.
        events.append({"version": 2, "trial_number": trial.number,
                       "expected_trial_numbers": expected_numbers,
                       "attestation": "listed_trial_workers_stopped",
                       "recorded_at_unix": time.time(), "target_state": "FAIL"})
        study.set_user_attr(ORPHAN_TRIAL_RECOVERY_ATTR, events)
        study.tell(trial.number, state=TrialState.FAIL)
        logger.warning("[NAS] Explicit orphan recovery marked trial %s FAIL.", trial.number)

