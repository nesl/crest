# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""SQLite campaign extension preserves evidence and rejects unsigned history."""
import copy
from pathlib import Path
from unittest.mock import MagicMock, patch

from addict import Dict
import optuna
from optuna.trial import TrialState, create_trial
import pytest

from crest.model import _normalize_optimizer_config
from crest.optimizers.llm.providers.provider import FakeProvider
from crest.optimizers.search_space import SearchParam
from nas_model_client import (
    FEASIBILITY_POLICY_SIGNATURE_ATTR,
    OPTIMIZER_SIGNATURE_ATTR, PROPOSAL_ATTR,
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


def sign_study(client, study):
    *_, signature = client._select_optimizer(study, client._build_sampler())
    study.set_user_attr(OPTIMIZER_SIGNATURE_ATTR, signature)


def signed_study(client, storage):
    study = optuna.create_study(study_name="campaign", storage=storage, direction="maximize")
    sign_study(client, study)
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


def test_signed_sqlite_repeated_extension_preserves_old_evidence(campaign):
    client, storage = campaign
    study = signed_study(client, storage)
    before = study.trials
    client.config.training.nas_trials = 2
    client.config.training.max_total_trials = 5
    result = client.run_nas("campaign", storage)
    assert result.trials[:4] == before
    assert len(result.trials) == 5
    assert result.trials[3].state == TrialState.RUNNING
    assert PROPOSAL_ATTR not in result.trials[4].user_attrs
    signature = result.user_attrs[OPTIMIZER_SIGNATURE_ATTR]
    assert signature["sampler"]["class"] == "optuna.samplers.TPESampler"
    assert "adopt_legacy_study" not in signature["config"]
    previous_trials = result.trials
    client.config.training.nas_trials = 3
    client.config.training.max_total_trials = 6
    resumed = client.run_nas("campaign", storage)
    assert len(resumed.trials) == 6
    assert resumed.trials[:5] == previous_trials
    assert resumed.user_attrs[OPTIMIZER_SIGNATURE_ATTR] == signature


def test_signed_running_trial_keeps_consuming_attempt_cap(campaign):
    client, storage = campaign
    study = signed_study(client, storage)
    before = study.trials
    client.config.training.nas_trials = 2
    client.config.training.max_total_trials = 4
    client.objective = MagicMock()
    result = client.run_nas("campaign", storage)
    client.objective.assert_not_called()
    assert result.trials == before


@pytest.mark.parametrize("state", [TrialState.COMPLETE, TrialState.FAIL, TrialState.PRUNED,
                                   TrialState.RUNNING, TrialState.WAITING])
def test_unsigned_history_rejected_before_initialization_or_recovery(campaign, state):
    client, storage = campaign
    study = optuna.create_study(study_name="campaign", storage=storage, direction="maximize")
    study.add_trial(create_trial(
        state=state, value=3.0 if state == TrialState.COMPLETE else None,
        params={"width": 3}, distributions={"width": optuna.distributions.IntDistribution(2, 8)},
        user_attrs={"historic_note": "keep"},
    ))
    before, attrs = study.trials, study.user_attrs
    client.objective = MagicMock()
    with patch.object(client, "_initialize_optimizer") as initialize, \
         patch.object(client, "_recover_orphaned_trials") as recover:
        with pytest.raises(RuntimeError, match="no CREST optimizer signature.*new study"):
            client.run_nas("campaign", storage)
    initialize.assert_not_called()
    recover.assert_not_called()
    client.objective.assert_not_called()
    assert_unchanged(storage, before, attrs)


@pytest.mark.parametrize("policy", ["missing", "mismatch", "disabled", "missing_trial_evidence"])
def test_signed_resume_obeys_existing_feasibility_checks(campaign, policy):
    client, storage = campaign
    client.config.nas.feasibility = Dict(train_if_infeasible=False, rules=[{
        "rule": "latency_budget", "metric": "latency_ms", "condition": "<=",
        "reference": {"type": "literal", "value": 10},
    }])
    active_feasibility = client.config.nas.feasibility
    if policy == "disabled":
        client.config.nas.feasibility = Dict(train_if_infeasible=False, rules=[])
    study = signed_study(client, storage)
    client.config.nas.feasibility = active_feasibility
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
        client.objective = MagicMock()
        with pytest.raises(RuntimeError, match="optimizer signature"):
            client.run_nas("campaign", storage)
        client.objective.assert_not_called()
        assert study.user_attrs[OPTIMIZER_SIGNATURE_ATTR] == changed


@pytest.mark.parametrize("sampler_cls", [optuna.samplers.TPESampler, optuna.samplers.NSGAIISampler])
def test_new_sampler_identity_survives_internal_module_relocation(campaign, sampler_cls):
    client, storage = campaign
    study = optuna.create_study(study_name="campaign", storage=storage, direction="maximize")
    sampler = sampler_cls()
    with patch.object(sampler_cls, "__module__", "optuna.internal.relocated"):
        *_, signature = client._select_optimizer(study, sampler)
        study.set_user_attr(OPTIMIZER_SIGNATURE_ATTR, signature)
    assert study.user_attrs[OPTIMIZER_SIGNATURE_ATTR]["sampler"]["class"] == "optuna.samplers." + sampler_cls.__name__


@pytest.mark.parametrize("value", [None, False, True, 0, 1, "true", "false", {}, []])
def test_obsolete_adoption_option_is_rejected(value):
    with pytest.raises(ValueError, match="no longer supported"):
        _normalize_optimizer_config(Dict(optimizer={"type": "optuna", "adopt_legacy_study": value}))


def test_signed_resume_preserves_matching_feasibility_policy(campaign):
    client, storage = campaign
    client.config.nas.feasibility = Dict(train_if_infeasible=False, rules=[{
        "rule": "latency_budget", "metric": "latency_ms", "condition": "<=",
        "reference": {"type": "literal", "value": 10},
    }])
    study = optuna.create_study(study_name="campaign", storage=storage, direction="maximize")
    sign_study(client, study)
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


@pytest.mark.parametrize("feasibility_enabled", [False, True])
def test_historical_evaluation_policy_defaults_resume_without_rewriting(campaign, feasibility_enabled):
    """Missing historical order/batch identity normalizes to true/256 only."""
    client, storage = campaign
    if feasibility_enabled:
        client.config.nas.feasibility = Dict(train_if_infeasible=False, rules=[{
            "rule": "latency_budget", "metric": "latency_ms", "condition": "<=",
            "reference": {"type": "literal", "value": 10},
        }])
    study = optuna.create_study(study_name="campaign", storage=storage, direction="maximize")
    *_, signature = client._select_optimizer(study, client._build_sampler())
    historical = copy.deepcopy(signature)
    historical.pop("evaluation_policy")
    study.set_user_attr(OPTIMIZER_SIGNATURE_ATTR, historical)

    client._select_optimizer(study, client._build_sampler())

    assert study.user_attrs[OPTIMIZER_SIGNATURE_ATTR] == historical


@pytest.mark.parametrize("field,value", [("hil_first", False), ("training_batch_size", 64)])
def test_historical_evaluation_policy_rejects_nonlegacy_active_values(campaign, field, value):
    """An unsigned historical policy cannot be adopted as a changed policy."""
    client, storage = campaign
    study = optuna.create_study(study_name="campaign", storage=storage, direction="maximize")
    *_, signature = client._select_optimizer(study, client._build_sampler())
    historical = copy.deepcopy(signature)
    historical.pop("evaluation_policy")
    study.set_user_attr(OPTIMIZER_SIGNATURE_ATTR, historical)
    if field == "hil_first":
        client.config.nas.hil_first = value
    else:
        client.config.training.batch_size = value

    with pytest.raises(RuntimeError, match="optimizer signature"):
        client._select_optimizer(study, client._build_sampler())
    assert study.user_attrs[OPTIMIZER_SIGNATURE_ATTR] == historical


@pytest.mark.parametrize("hil_first", ["false", 0, None])
def test_optimizer_identity_rejects_nonboolean_hil_first_when_loader_is_bypassed(campaign, hil_first):
    """Programmatic configs cannot coerce ambiguous order values in identity."""
    client, storage = campaign
    client.config.nas.hil_first = hil_first
    study = optuna.create_study(study_name="campaign", storage=storage, direction="maximize")

    with pytest.raises(ValueError, match="nas.hil_first"):
        client._select_optimizer(study, client._build_sampler())


def test_nonempty_historical_default_policy_resumes_without_rewriting(campaign):
    """A signed legacy study resumes under true/256 without metadata adoption."""
    client, storage = campaign
    study = signed_study(client, storage)
    historical = copy.deepcopy(study.user_attrs[OPTIMIZER_SIGNATURE_ATTR])
    historical.pop("evaluation_policy")
    study.set_user_attr(OPTIMIZER_SIGNATURE_ATTR, historical)
    before = study.trials
    client.objective = MagicMock()

    resumed = client.run_nas("campaign", storage)

    client.objective.assert_not_called()
    assert resumed.trials == before
    assert resumed.user_attrs[OPTIMIZER_SIGNATURE_ATTR] == historical


@pytest.mark.parametrize("feasibility_enabled", [False, True])
@pytest.mark.parametrize("field,value", [("hil_first", False), ("training_batch_size", 64)])
@pytest.mark.parametrize("stored_policy", ["missing", "explicit_changed"])
def test_run_nas_rejects_changed_evaluation_policy_before_objective(
    campaign, feasibility_enabled, field, value, stored_policy
):
    """Order and batch identity stay unconditional and strict on real resume."""
    client, storage = campaign
    if feasibility_enabled:
        client.config.nas.feasibility = Dict(train_if_infeasible=False, rules=[{
            "rule": "latency_budget", "metric": "latency_ms", "condition": "<=",
            "reference": {"type": "literal", "value": 10},
        }])
    study = signed_study(client, storage)
    stored = copy.deepcopy(study.user_attrs[OPTIMIZER_SIGNATURE_ATTR])
    if stored_policy == "missing":
        stored.pop("evaluation_policy")
        if field == "hil_first":
            client.config.nas.hil_first = value
        else:
            client.config.training.batch_size = value
    else:
        stored["evaluation_policy"][field] = value
    study.set_user_attr(OPTIMIZER_SIGNATURE_ATTR, stored)
    before, attrs = study.trials, study.user_attrs
    client.objective = MagicMock()

    with patch.object(client, "_initialize_optimizer") as initialize:
        with pytest.raises(RuntimeError, match="optimizer signature"):
            client.run_nas("campaign", storage)

    initialize.assert_not_called()
    client.objective.assert_not_called()
    assert_unchanged(storage, before, attrs)


def test_signed_waiting_identity_mismatch_rejected(campaign):
    client, storage = campaign
    study = optuna.create_study(study_name="campaign", storage=storage, direction="maximize")
    *_, signature = client._select_optimizer(study, client._build_sampler())
    study.set_user_attr(OPTIMIZER_SIGNATURE_ATTR, signature)
    signature = study.user_attrs[OPTIMIZER_SIGNATURE_ATTR]
    signature["sampler"]["options"]["n_startup_trials"] = 16
    study.set_user_attr(OPTIMIZER_SIGNATURE_ATTR, signature)
    study.enqueue_trial({"width": 7})
    before, attrs = study.trials, study.user_attrs
    client.objective = MagicMock()
    with pytest.raises(RuntimeError, match="optimizer signature"):
        client.run_nas("campaign", storage)
    client.objective.assert_not_called()
    assert_unchanged(storage, before, attrs)


def test_fresh_study_initialization_failure_does_not_write_identity(campaign):
    client, storage = campaign
    study = optuna.create_study(study_name="campaign", storage=storage, direction="maximize")
    before, attrs = study.trials, study.user_attrs
    client.objective = MagicMock()
    with patch("crest.optimizers.optuna.OptunaOptimizer.initialize", side_effect=RuntimeError("setup failed")):
        with pytest.raises(RuntimeError, match="setup failed"):
            client.run_nas("campaign", storage)
    client.objective.assert_not_called()
    assert_unchanged(storage, before, attrs)


@pytest.mark.parametrize("artifact_kind", ["directory", "file"])
def test_unsigned_unqueued_terminal_history_rejects_llm_artifact_path(campaign, artifact_kind):
    client, storage = campaign
    study = optuna.create_study(study_name="campaign", storage=storage, direction="maximize")
    study.optimize(client.objective, n_trials=1)
    assert study.trials[0].state == TrialState.COMPLETE
    assert study.trials[0].system_attrs == {}
    assert study.trials[0].user_attrs == {}
    artifact_path = Path(client.config.outputs.models_dir) / study.study_name / "llm_optimizer"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    if artifact_kind == "directory":
        artifact_path.mkdir()
    else:
        artifact_path.touch()
    client.config.training.nas_trials = 2
    client.objective = MagicMock()
    before, attrs = study.trials, study.user_attrs
    with patch.object(client, "_validate_or_store_feasibility_signature",
                      side_effect=AssertionError("feasibility validation reached")):
        with pytest.raises(RuntimeError, match="no CREST optimizer signature.*new study"):
            client.run_nas("campaign", storage)
    client.objective.assert_not_called()
    assert_unchanged(storage, before, attrs)


@pytest.mark.parametrize("optimizer", ["optuna", "llm_generator"])
def test_signed_resume_accepts_fixed_params_llm_artifacts_and_waiting_work(campaign, optimizer):
    client, storage = campaign
    client.config.optimizer = Dict(type=optimizer)
    if optimizer == "llm_generator":
        client.config.optimizer.llm = Dict(provider="fake", responses=[{"candidates": [{"width": 5}]}],
            memory={"enabled": False})
    study = optuna.create_study(study_name="campaign", storage=storage, direction="maximize")
    *_, signature = client._select_optimizer(study, client._build_sampler())
    study.set_user_attr(OPTIMIZER_SIGNATURE_ATTR, signature)
    study.enqueue_trial({"width": 3})
    study.optimize(client.objective, n_trials=1)
    assert study.trials[0].system_attrs["fixed_params"] == {"width": 3}
    assert study.trials[0].user_attrs == {}
    study.enqueue_trial({"width": 7})
    artifact_path = Path(client.config.outputs.models_dir) / study.study_name / "llm_optimizer"
    artifact_path.mkdir(parents=True)
    before, signature = study.trials, study.user_attrs[OPTIMIZER_SIGNATURE_ATTR]
    client.config.training.nas_trials = 2
    client.config.training.max_total_trials = 2
    provider = FakeProvider([])
    with patch("crest.optimizers.llm.component.build_provider", return_value=provider):
        result = client.run_nas("campaign", storage)
    assert provider.requests == []
    assert result.trials[:1] == before[:1]
    assert len(result.trials) == 2
    assert result.trials[1].state == TrialState.COMPLETE
    assert result.trials[1].params == {"width": 7}
    assert result.user_attrs[OPTIMIZER_SIGNATURE_ATTR] == signature


@pytest.mark.parametrize("optimizer", ["optuna", "llm_generator"])
def test_signed_native_and_llm_budget_only_extension(campaign, optimizer):
    client, storage = campaign
    client.config.optimizer = Dict(type=optimizer)
    if optimizer == "llm_generator":
        client.config.optimizer.llm = Dict(provider="fake", responses=[{"candidates": [{"width": 3}]}],
            batch_size=1, memory={"enabled": False})
    with patch("crest.optimizers.llm.component.build_provider", return_value=FakeProvider([{"candidates": [{"width": 3}]}])):
        first = client.run_nas("campaign", storage)
    first_trials = first.trials
    signature = first.user_attrs[OPTIMIZER_SIGNATURE_ATTR]
    client.config.training.nas_trials = 2
    client.config.training.max_total_trials = 5
    with patch("crest.optimizers.llm.component.build_provider", return_value=FakeProvider([{"candidates": [{"width": 7}]}])):
        second = client.run_nas("campaign", storage)
    assert len(second.trials) == 2
    assert second.trials[:1] == first_trials
    assert second.user_attrs[OPTIMIZER_SIGNATURE_ATTR] == signature
    if optimizer == "llm_generator":
        assert second.trials[1].params == {"width": 7}
        assert second.trials[1].user_attrs[PROPOSAL_ATTR]["optimizer"] == optimizer


@pytest.mark.parametrize("provenance", [PROPOSAL_ATTR, "proposal_source"])
@pytest.mark.parametrize("optimizer", ["optuna", "llm_generator"])
def test_unsigned_provenance_rejected_before_provider_setup(campaign, provenance, optimizer):
    client, storage = campaign
    client.config.optimizer = Dict(type=optimizer)
    if optimizer == "llm_generator":
        client.config.optimizer.llm = Dict(provider="fake", responses=[{"candidates": [{"width": 7}]}],
            memory={"enabled": False})
    study = optuna.create_study(study_name="campaign", storage=storage, direction="maximize")
    study.add_trial(create_trial(value=7, user_attrs={provenance: {"optimizer": "llm_generator"}}))
    before, attrs = study.trials, study.user_attrs
    client.objective = MagicMock()
    with patch("crest.optimizers.llm.component.build_provider", side_effect=AssertionError("provider called")):
        with pytest.raises(RuntimeError, match="no CREST optimizer signature.*new study"):
            client.run_nas("campaign", storage)
    client.objective.assert_not_called()
    assert_unchanged(storage, before, attrs)








def configure_live_llm(client):
    client.config.optimizer = Dict(type="llm_generator", llm={
        "provider": "openai_compatible", "base_url": "https://example.test/v1",
        "model": "test-model", "api_key_env": "CREST_TEST_ONLY_API_KEY",
        "batch_size": 1, "memory": {"enabled": False},
    })


def configure_feasibility(client):
    client.config.nas.feasibility = Dict(train_if_infeasible=False, rules=[{
        "rule": "flops_budget", "metric": "flops", "condition": "<=",
        "reference": {"type": "literal", "value": 10},
    }])


@pytest.mark.parametrize("entrypoint", ["run", "smoke"])
@pytest.mark.parametrize("failure", ["credentials", "descriptor", "context"])
def test_failed_fresh_llm_setup_leaves_both_signatures_unset_and_native_can_retry(
        campaign, monkeypatch, entrypoint, failure):
    client, storage = campaign
    configure_live_llm(client)
    configure_feasibility(client)
    monkeypatch.delenv("CREST_TEST_ONLY_API_KEY", raising=False)
    if failure != "credentials":
        monkeypatch.setenv("CREST_TEST_ONLY_API_KEY", "test-key")
    if failure == "descriptor":
        client.model_family.trial_search_space = None
    context_error = ValueError("synthetic context failure")
    client.objective = MagicMock(side_effect=AssertionError("evaluation before setup"))
    previous = (client.config.device.hil, client.config.training.train,
                client.config.training.nas_epochs, client.config.training.nas_trials,
                client.config.training.max_total_trials)
    def invoke():
        if entrypoint == "run":
            return client.run_nas("campaign", storage)
        client.smoke_test(train=False, hil=False, trials=1, epochs=1, study_name="campaign")
        return optuna.load_study(study_name="campaign", storage=smoke_storage)
    smoke_storage = f"sqlite:///{Path(client.config.outputs.models_dir) / 'campaign' / 'optuna_smoke_test.db'}"
    failure_storage = storage if entrypoint == "run" else smoke_storage
    with patch("crest.optimizers.llm.component.build_provider", side_effect=AssertionError("provider before validation")), \
         patch("nas_model_client.build_semantic_context", side_effect=context_error if failure == "context" else None) as context:
        if failure != "context":
            context.side_effect = None
            context.return_value = {}
        with pytest.raises((RuntimeError, ValueError), match={
                "credentials": "Required API key", "descriptor": "descriptor", "context": "synthetic context",
        }[failure]):
            invoke()
    study = optuna.load_study(study_name="campaign", storage=failure_storage)
    assert study.trials == []
    assert study.user_attrs == {}
    client.objective.assert_not_called()
    assert previous == (client.config.device.hil, client.config.training.train,
                        client.config.training.nas_epochs, client.config.training.nas_trials,
                        client.config.training.max_total_trials)
    client.config.optimizer = Dict(type="optuna")
    def objective(trial):
        trial.set_user_attr("feasibility_status", "feasible")
        trial.set_user_attr("feasibility_constraints", [-1.0])
        return float(trial.suggest_int("width", 2, 8))
    client.objective = objective
    result = invoke()
    assert len(result.trials) == 1
    assert result.user_attrs[OPTIMIZER_SIGNATURE_ATTR]["optimizer"] == "optuna"
    assert result.user_attrs[FEASIBILITY_POLICY_SIGNATURE_ATTR] == client._feasibility_policy_signature()


@pytest.mark.parametrize("entrypoint", ["run", "smoke"])
def test_successful_llm_setup_signs_after_initialize_before_provider_request_or_evaluation(
        campaign, monkeypatch, entrypoint):
    client, storage = campaign
    configure_live_llm(client)
    configure_feasibility(client)
    monkeypatch.setenv("CREST_TEST_ONLY_API_KEY", "test-key")
    if entrypoint == "smoke":
        storage = f"sqlite:///{Path(client.config.outputs.models_dir) / 'campaign' / 'optuna_smoke_test.db'}"
    events = []
    provider = FakeProvider([{"candidates": [{"width": 3}]}])
    complete = provider.complete_json
    def stored():
        return optuna.load_study(study_name="campaign", storage=storage).user_attrs
    def build(config):
        assert stored() == {}
        events.append("initialize")
        return provider
    def request(prompt):
        attrs = stored()
        assert attrs[OPTIMIZER_SIGNATURE_ATTR]["optimizer"] == "llm_generator"
        assert attrs[FEASIBILITY_POLICY_SIGNATURE_ATTR] == client._feasibility_policy_signature()
        assert "timeout_s" not in attrs[OPTIMIZER_SIGNATURE_ATTR]["config"]
        assert "json_response_mode" in attrs[OPTIMIZER_SIGNATURE_ATTR]["config"]
        events.append("request")
        return complete(prompt)
    provider.complete_json = request
    def objective(trial):
        assert OPTIMIZER_SIGNATURE_ATTR in stored()
        events.append("evaluate")
        trial.set_user_attr("feasibility_status", "feasible")
        trial.set_user_attr("feasibility_constraints", [-1.0])
        return float(trial.suggest_int("width", 2, 8))
    client.objective = objective
    with patch("crest.optimizers.llm.component.build_provider", side_effect=build):
        if entrypoint == "run":
            client.run_nas("campaign", storage)
        else:
            client.smoke_test(train=False, hil=False, trials=1, epochs=1, study_name="campaign")
    assert events == ["initialize", "request", "evaluate"]


@pytest.mark.parametrize("change", [None, "model", "json_response_mode", "extra_identity", "version",
                                     "bad_timeout", "bool_timeout", "infinite_timeout", "nan_timeout",
                                     "small_timeout", "zero_timeout", "negative_timeout"])
def test_old_builtin_llm_timeout_signature_resumes_without_rewriting_but_other_identity_stays_strict(
        campaign, monkeypatch, change):
    client, storage = campaign
    configure_live_llm(client)
    monkeypatch.setenv("CREST_TEST_ONLY_API_KEY", "test-key")
    study = optuna.create_study(study_name="campaign", storage=storage, direction="maximize")
    *_, signature = client._select_optimizer(study, client._build_sampler())
    signature["config"]["timeout_s"] = 60.0
    client.config.optimizer.llm.timeout_s = 120.0
    if change in ("model", "json_response_mode"):
        client.config.optimizer.llm[change] = "other-model" if change == "model" else True
    elif change == "extra_identity":
        signature["config"]["unknown_setting"] = "keep strict"
    elif change == "version":
        signature["version"] = 2
    elif change in ("bad_timeout", "bool_timeout", "infinite_timeout", "nan_timeout",
                    "small_timeout", "zero_timeout", "negative_timeout"):
        signature["config"]["timeout_s"] = {"bad_timeout": "60", "bool_timeout": True,
            "infinite_timeout": float("inf"), "nan_timeout": float("nan"),
            "small_timeout": 0.0001, "zero_timeout": 0, "negative_timeout": -1}[change]
    study.set_user_attr(OPTIMIZER_SIGNATURE_ATTR, signature)
    study.enqueue_trial({"width": 3})
    client.objective = MagicMock(side_effect=lambda trial: float(trial.suggest_int("width", 2, 8)))
    if change is not None:
        with patch("crest.optimizers.llm.component.build_provider", side_effect=AssertionError("initialized mismatch")):
            with pytest.raises(RuntimeError, match="optimizer signature"):
                client.run_nas("campaign", storage)
        client.objective.assert_not_called()
        assert study.trials[0].state == TrialState.WAITING
    else:
        provider = FakeProvider([])
        with patch("crest.optimizers.llm.component.build_provider", return_value=provider):
            client.run_nas("campaign", storage)
        assert provider.requests == []
        assert client.objective.call_count == 1
        assert study.trials[0].state == TrialState.COMPLETE
        assert study.user_attrs[OPTIMIZER_SIGNATURE_ATTR] == signature
