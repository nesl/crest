# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""SQLite campaign extension preserves evidence and requires explicit legacy consent."""
import copy
from pathlib import Path
from unittest.mock import MagicMock, patch

from addict import Dict
import optuna
from optuna.trial import TrialState, create_trial
import pytest

from crest.model import _normalize_optimizer_config
from crest.optimizers.search_space import SearchParam
from nas_model_client import (
    FEASIBILITY_POLICY_SIGNATURE_ATTR, LEGACY_OPTIMIZER_ADOPTION_ATTR,
    NASModelClient, OPTIMIZER_SIGNATURE_ATTR, PROPOSAL_ATTR,
)
from test.test_nas_model_client import _build_test_client


@pytest.fixture
def campaign(tmp_path):
    client = _build_test_client(tmp_path)
    client.config.optimizer = Dict(type="optuna")
    client.config.device.hil = False
    client.config.device.compile_when_hil_disabled = "false"
    client.config.training.train = False
    client.config.training.nas_trials = 1
    client.config.training.max_total_trials = 4
    client.config.nas.score = Dict(type="scoring-function", metrics={}, params={
        "terms": [{"type": "weighted", "metric": "flops", "weight": -1.0}],
    })
    client.model_family.trial_search_space = MagicMock(return_value=[SearchParam("width", "int", low=2, high=8)])
    client.objective = lambda trial: float(trial.suggest_int("width", 2, 8))
    return client, f"sqlite:///{tmp_path / 'resume.db'}"


def legacy_study(storage):
    study = optuna.create_study(study_name="campaign", storage=storage, direction="maximize")
    for width, state in ((3, TrialState.COMPLETE), (4, TrialState.PRUNED),
                         (5, TrialState.FAIL), (6, TrialState.RUNNING)):
        study.add_trial(create_trial(
            state=state, value=float(width) if state in (TrialState.COMPLETE, TrialState.PRUNED) else None,
            params={"width": width}, distributions={"width": optuna.distributions.IntDistribution(2, 8)},
            user_attrs={"historic_note": f"keep-{width}"}, system_attrs={"historic_system": width},
        ))
    return study


def assert_unchanged(storage, trials, attrs):
    study = optuna.load_study(study_name="campaign", storage=storage)
    assert study.trials == trials
    assert study.user_attrs == attrs


def test_native_legacy_sqlite_adoption_and_repeated_extension_preserve_old_evidence(campaign):
    client, storage = campaign
    study = legacy_study(storage)
    before = study.trials
    client.config.training.nas_trials = 2
    client.config.training.max_total_trials = 5
    client.config.optimizer.adopt_legacy_study = True
    result = client.run_nas("campaign", storage)
    assert result.trials[:4] == before
    assert len(result.trials) == 5
    assert result.trials[3].state == TrialState.RUNNING
    assert PROPOSAL_ATTR not in result.trials[4].user_attrs
    signature = result.user_attrs[OPTIMIZER_SIGNATURE_ATTR]
    assert signature["sampler"]["class"] == "optuna.samplers.TPESampler"
    assert "adopt_legacy_study" not in signature["config"]
    marker = result.user_attrs[LEGACY_OPTIMIZER_ADOPTION_ATTR]
    assert marker == {"version": 1, "attestation": "matching_original_config", "trial_count": 4}
    adopted_trials = result.trials
    del client.config.optimizer["adopt_legacy_study"]
    client.config.training.nas_trials = 3
    client.config.training.max_total_trials = 6
    resumed = client.run_nas("campaign", storage)
    assert len(resumed.trials) == 6
    assert resumed.trials[:5] == adopted_trials
    assert resumed.user_attrs[OPTIMIZER_SIGNATURE_ATTR] == signature
    assert resumed.user_attrs[LEGACY_OPTIMIZER_ADOPTION_ATTR] == marker


def test_adopted_running_trial_keeps_consuming_attempt_cap(campaign):
    client, storage = campaign
    study = legacy_study(storage)
    before = study.trials
    client.config.optimizer.adopt_legacy_study = True
    client.config.training.nas_trials = 2
    client.config.training.max_total_trials = 4
    client.objective = MagicMock()
    result = client.run_nas("campaign", storage)
    client.objective.assert_not_called()
    assert result.trials == before





@pytest.mark.parametrize("state", [TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL])
@pytest.mark.parametrize("params", [{"width": 3}, {}])
def test_unsigned_terminal_enqueue_rejects_native_adoption_without_user_provenance(campaign, state, params):
    client, storage = campaign
    study = optuna.create_study(study_name="campaign", storage=storage, direction="maximize")
    study.enqueue_trial(params)

    def finish_trial(trial):
        width = trial.suggest_int("width", 2, 8)
        if state == TrialState.PRUNED:
            raise optuna.TrialPruned()
        if state == TrialState.FAIL:
            raise RuntimeError("synthetic failure")
        return float(width)

    study.optimize(finish_trial, n_trials=1, catch=(RuntimeError,))
    trial = study.trials[0]
    assert trial.state == state
    assert trial.system_attrs["fixed_params"] == params
    assert trial.user_attrs == {}
    client.config.optimizer.adopt_legacy_study = True
    client.config.training.nas_trials = 2
    client.objective = MagicMock()
    before, attrs = study.trials, study.user_attrs
    with patch.object(client, "_validate_or_store_feasibility_signature",
                      side_effect=AssertionError("feasibility validation reached")):
        with pytest.raises(RuntimeError, match="enqueued fixed_params"):
            client.run_nas("campaign", storage)
    client.objective.assert_not_called()
    assert_unchanged(storage, before, attrs)








@pytest.mark.parametrize("policy", ["missing", "mismatch", "disabled", "missing_trial_evidence"])
def test_legacy_adoption_obeys_existing_feasibility_checks_before_stamping(campaign, policy):
    client, storage = campaign
    study = legacy_study(storage)
    client.config.optimizer.adopt_legacy_study = True
    client.config.nas.feasibility = Dict(train_if_infeasible=False, rules=[{
        "rule": "latency_budget", "metric": "latency_ms", "condition": "<=",
        "reference": {"type": "literal", "value": 10},
    }])
    if policy != "missing":
        signature = client._feasibility_policy_signature()
        if policy == "mismatch":
            signature["train_if_infeasible"] = True
        study.set_user_attr(FEASIBILITY_POLICY_SIGNATURE_ATTR, signature)
    if policy == "disabled":
        client.config.nas.feasibility = Dict(train_if_infeasible=False, rules=[])
    before, attrs = study.trials, study.user_attrs
    client.objective = MagicMock()
    with pytest.raises(RuntimeError, match="feasibility"):
        client.run_nas("campaign", storage)
    client.objective.assert_not_called()
    assert_unchanged(storage, before, attrs)





@pytest.mark.parametrize("multi,old_name,public_name", [
    (False, "optuna.samplers._tpe.sampler.TPESampler", "optuna.samplers.TPESampler"),
    (True, "optuna.samplers.nsgaii._sampler.NSGAIISampler", "optuna.samplers.NSGAIISampler"),
])
def test_exact_old_private_sampler_identity_resumes_but_options_remain_strict(campaign, multi, old_name, public_name):
    client, storage = campaign
    if multi:
        client.config.nas.score = Dict(type="multi-objective", metrics={}, params={"objectives": [
            {"name": "flops", "metric": "flops", "direction": "minimize"},
            {"name": "ram", "metric": "ram_bytes", "direction": "minimize"},
        ]})
    study = optuna.create_study(study_name="campaign", storage=storage, directions=client._study_directions())
    *_, signature = client._select_optimizer(study, client._build_sampler())
    study.set_user_attr(OPTIMIZER_SIGNATURE_ATTR, signature)
    signature = study.user_attrs[OPTIMIZER_SIGNATURE_ATTR]
    assert signature["sampler"]["class"] == public_name
    signature["sampler"]["class"] = old_name
    study.set_user_attr(OPTIMIZER_SIGNATURE_ATTR, signature)
    client._select_optimizer(study, client._build_sampler())
    assert study.user_attrs[OPTIMIZER_SIGNATURE_ATTR] == signature
    for field in ("options", "optimizer", "config", "class"):
        changed = copy.deepcopy(signature)
        if field == "options":
            changed["sampler"]["options"]["seed"] = 99
        elif field == "optimizer":
            changed["optimizer"] = "different_optimizer"
        elif field == "config":
            changed["config"] = {"unrecognized": True}
        else:
            changed["sampler"]["class"] = "some.private.module." + public_name.rsplit(".", 1)[-1]
        study.set_user_attr(OPTIMIZER_SIGNATURE_ATTR, changed)
        client.config.optimizer.adopt_legacy_study = True
        client.objective = MagicMock()
        with pytest.raises(RuntimeError, match="optimizer signature"):
            client.run_nas("campaign", storage)
        client.objective.assert_not_called()
        assert study.user_attrs[OPTIMIZER_SIGNATURE_ATTR] == changed
        assert LEGACY_OPTIMIZER_ADOPTION_ATTR not in study.user_attrs


@pytest.mark.parametrize("sampler_cls", [optuna.samplers.TPESampler, optuna.samplers.NSGAIISampler])
def test_new_sampler_identity_survives_internal_module_relocation(campaign, sampler_cls):
    client, storage = campaign
    study = optuna.create_study(study_name="campaign", storage=storage, direction="maximize")
    sampler = sampler_cls()
    with patch.object(sampler_cls, "__module__", "optuna.internal.relocated"):
        *_, signature = client._select_optimizer(study, sampler)
        study.set_user_attr(OPTIMIZER_SIGNATURE_ATTR, signature)
    assert study.user_attrs[OPTIMIZER_SIGNATURE_ATTR]["sampler"]["class"] == "optuna.samplers." + sampler_cls.__name__


@pytest.mark.parametrize("value", [None, 0, 1, "true", "false", {}, []])
def test_legacy_adoption_is_a_strict_boolean(value):
    with pytest.raises(ValueError, match="adopt_legacy_study must be a boolean"):
        _normalize_optimizer_config(Dict(optimizer={"type": "optuna", "adopt_legacy_study": value}))





def test_matching_feasibility_legacy_adoption_preserves_persisted_policy(campaign):
    client, storage = campaign
    client.config.optimizer.adopt_legacy_study = True
    client.config.nas.feasibility = Dict(train_if_infeasible=False, rules=[{
        "rule": "latency_budget", "metric": "latency_ms", "condition": "<=",
        "reference": {"type": "literal", "value": 10},
    }])
    study = optuna.create_study(study_name="campaign", storage=storage, direction="maximize")
    study.add_trial(create_trial(value=3, user_attrs={
        "feasibility_status": "feasible", "feasibility_constraints": [-1.0],
    }))
    policy = client._feasibility_policy_signature()
    study.set_user_attr(FEASIBILITY_POLICY_SIGNATURE_ATTR, policy)
    before = study.trials
    result = client.run_nas("campaign", storage)
    assert result.trials == before
    assert result.user_attrs[FEASIBILITY_POLICY_SIGNATURE_ATTR] == policy
    assert result.user_attrs[OPTIMIZER_SIGNATURE_ATTR]["sampler"]["options"]["constraints_func"] == "crest.persisted_feasibility_constraints.v1"
    assert LEGACY_OPTIMIZER_ADOPTION_ATTR in result.user_attrs


def test_adoption_flag_does_not_bypass_signed_waiting_identity_mismatch(campaign):
    client, storage = campaign
    study = optuna.create_study(study_name="campaign", storage=storage, direction="maximize")
    *_, signature = client._select_optimizer(study, client._build_sampler())
    study.set_user_attr(OPTIMIZER_SIGNATURE_ATTR, signature)
    signature = study.user_attrs[OPTIMIZER_SIGNATURE_ATTR]
    signature["sampler"]["options"]["n_startup_trials"] = 16
    study.set_user_attr(OPTIMIZER_SIGNATURE_ATTR, signature)
    study.enqueue_trial({"width": 7})
    client.config.optimizer.adopt_legacy_study = True
    before, attrs = study.trials, study.user_attrs
    client.objective = MagicMock()
    with pytest.raises(RuntimeError, match="optimizer signature"):
        client.run_nas("campaign", storage)
    client.objective.assert_not_called()
    assert_unchanged(storage, before, attrs)


@pytest.mark.parametrize("stored_directions", [["minimize"], ["maximize", "minimize"], ["minimize", "maximize"]])
def test_legacy_adoption_rejects_stored_direction_mismatch(campaign, stored_directions):
    client, storage = campaign
    client.config.optimizer.adopt_legacy_study = True
    if len(stored_directions) == 2:
        client.config.nas.score = Dict(type="multi-objective", metrics={}, params={"objectives": [
            {"name": "flops", "metric": "flops", "direction": "minimize"},
            {"name": "ram", "metric": "ram_bytes", "direction": "minimize"},
        ]})
    study = optuna.create_study(study_name="campaign", storage=storage, directions=stored_directions)
    study.add_trial(create_trial(values=[3.0] * len(stored_directions)))
    before, attrs = study.trials, study.user_attrs
    client.objective = MagicMock()
    with pytest.raises(RuntimeError, match="study directions"):
        client.run_nas("campaign", storage)
    client.objective.assert_not_called()
    assert_unchanged(storage, before, attrs)





def configure_feasibility(client):
    client.config.nas.feasibility = Dict(train_if_infeasible=False, rules=[{
        "rule": "flops_budget", "metric": "flops", "condition": "<=",
        "reference": {"type": "literal", "value": 10},
    }])






    client.objective = MagicMock()
    with patch("crest.optimizers.optuna.OptunaOptimizer.initialize", side_effect=RuntimeError("setup failed")):
        with pytest.raises(RuntimeError, match="setup failed"):
            client.run_nas("campaign", storage)
    client.objective.assert_not_called()
    assert_unchanged(storage, before, attrs)



