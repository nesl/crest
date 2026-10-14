"""Fixed warm-up uses the real runner, Optuna history, and existing resume guards."""
import json
from unittest.mock import patch

import optuna
import pytest
from addict import Dict
from optuna.trial import TrialState

from crest.optimizers.llm.provider import FakeProvider
from nas_model_client import MeasurementInfrastructureError, PROPOSAL_ATTR
from test.test_optimizer_resume import campaign


@pytest.fixture
def warmup(campaign, tmp_path):
    """Reuse the lightweight runner fixture with fifteen ordered fixed attempts."""
    client, storage = campaign
    params = [{'width': 2 + i % 6} for i in range(15)]
    path = tmp_path / 'warmup.json'
    path.write_text(json.dumps(params))
    response = {'candidates': [{'width': 8}]}
    client.config.optimizer = Dict(type='llm_generator', llm={
        'provider': 'fake', 'responses': [response], 'batch_size': 1,
        'warmup_params_file': str(path), 'recent_trial_window': 15,
        'memory': {'enabled': False}})
    client.config.training.nas_trials = 20
    client.config.training.max_total_trials = 16
    provider = FakeProvider([response])
    return client, storage, path, params, provider


@pytest.mark.parametrize('smoke', [False, True])
def test_fifteen_attempts_precede_first_provider_call(warmup, smoke):
    """Failures count once and the first LLM prompt contains all warm-up outcomes."""
    client, storage, path, params, provider = warmup
    def objective(trial):
        """Record attempted parameters and simulate two distinct terminal failures."""
        width = trial.suggest_int('width', 2, 8)
        if trial.number < 15:
            assert provider.requests == []
            assert width == params[trial.number]['width']
        if trial.number == 6:
            trial.set_user_attr('measurement_failure_class', 'infrastructure')
            raise MeasurementInfrastructureError('fixture failure')
        if trial.number == 9:
            raise optuna.TrialPruned('fixture rejection')
        return float(width)
    client.objective = objective
    with patch('crest.optimizers.llm.component.build_provider', return_value=provider):
        if smoke:
            client.smoke_test(trials=16, study_name='campaign')
            storage = f"sqlite:///{client._artifacts_dir() / 'optuna_smoke_test.db'}"
            study = optuna.load_study(study_name='campaign', storage=storage)
        else:
            study = client.run_nas('campaign', storage=storage)
    assert len(study.trials) == 16
    assert [t.params for t in study.trials[:15]] == params
    assert [t.user_attrs[PROPOSAL_ATTR]['warmup_index'] for t in study.trials[:15]] == list(range(15))
    assert study.trials[6].state == TrialState.FAIL
    assert study.trials[9].state == TrialState.PRUNED
    assert len(provider.requests) == 1
    history = json.loads(provider.requests[0].user_prompt)['recent_trials']
    assert [r['number'] for r in history] == list(range(15))
    assert history[6]['state'] == 'fail' and history[9]['state'] == 'pruned'


def test_resume_at_budget_and_relocated_file_preserves_order(warmup):
    """Budget extension and an identical relocated file must not replay warm-up."""
    client, storage, path, params, provider = warmup
    with patch('crest.optimizers.llm.component.build_provider', return_value=provider):
        client.config.training.max_total_trials = 7
        study = client.run_nas('campaign', storage=storage)
        assert len(study.trials) == 7 and not provider.requests
        moved = path.with_name('relocated.json')
        moved.write_text(json.dumps(params, indent=4))
        client.config.optimizer.llm.warmup_params_file = str(moved)
        client.config.training.max_total_trials = 16
        study = client.run_nas('campaign', storage=storage)
    assert len(study.trials) == 16
    assert [t.params for t in study.trials[:15]] == params
    assert len(provider.requests) == 1


def test_waiting_resume_and_changed_file_guard(warmup):
    """A queued warm-up entry drains once, but a changed sequence fails before execution."""
    client, storage, path, params, provider = warmup
    with patch('crest.optimizers.llm.component.build_provider', return_value=provider):
        with patch.object(optuna.Study, 'optimize', side_effect=RuntimeError('before execution')):
            with pytest.raises(RuntimeError, match='before execution'):
                client.run_nas('campaign', storage=storage)
        study = optuna.load_study(study_name='campaign', storage=storage)
        assert [t.state for t in study.trials] == [TrialState.WAITING]
        path.write_text(json.dumps([{'width': 7}] + params[1:]))
        with pytest.raises(RuntimeError, match='signature'):
            client.run_nas('campaign', storage=storage)
        assert len(study.get_trials()) == 1 and not provider.requests
        path.write_text(json.dumps(params))
        client.config.training.max_total_trials = 2
        study = client.run_nas('campaign', storage=storage)
    assert [t.params for t in study.trials] == params[:2]
    assert not provider.requests


@pytest.mark.parametrize('smoke', [False, True])
def test_orphan_running_warmup_blocks_provider(warmup, smoke):
    """Unfinished warm-up work requires explicit recovery in both runner paths."""
    client, storage, path, params, provider = warmup
    client.study_name = 'campaign'
    if smoke:
        storage = f"sqlite:///{client._artifacts_dir() / 'optuna_smoke_test.db'}"
    def invoke():
        if smoke:
            client.smoke_test(trials=1, study_name='campaign')
        else:
            client.run_nas('campaign', storage=storage)
    with patch('crest.optimizers.llm.component.build_provider', return_value=provider):
        with patch.object(optuna.Study, 'optimize', side_effect=RuntimeError('before execution')):
            with pytest.raises(RuntimeError):
                invoke()
        study = optuna.load_study(study_name='campaign', storage=storage)
        study.ask()  # Simulate interruption after reservation became RUNNING.
        with pytest.raises(RuntimeError, match='unfinished'):
            invoke()
    assert study.trials[0].state == TrialState.RUNNING
    assert len(study.get_trials()) == 1 and not provider.requests


@pytest.mark.parametrize('payload', [[], {}, [{}], [{'width': 99}], [{'width': 2}, {}]])
def test_invalid_file_fails_before_any_evaluation(warmup, payload):
    """Validate every warm-up entry up front, including entries after a valid first one."""
    client, storage, path, params, provider = warmup
    path.write_text(json.dumps(payload))
    with patch.object(client, 'objective') as objective, patch('crest.optimizers.llm.component.build_provider') as build:
        with pytest.raises(ValueError):
            client.run_nas('campaign', storage=storage)
    objective.assert_not_called()
    build.assert_not_called()


@pytest.mark.parametrize('smoke', [False, True])
def test_explicit_recovery_advances_orphaned_warmup_once(warmup, smoke):
    """A stopped-worker attestation fails the orphan and continues the fixed sequence."""
    client, storage, path, params, provider = warmup
    client.study_name = 'campaign'
    def invoke():
        if smoke:
            client.smoke_test(trials=1, study_name='campaign')
        else:
            client.run_nas('campaign', storage=storage)
    if smoke:
        storage = f"sqlite:///{client._artifacts_dir() / 'optuna_smoke_test.db'}"
    with patch('crest.optimizers.llm.component.build_provider', return_value=provider):
        with patch.object(optuna.Study, 'optimize', side_effect=RuntimeError('interrupted')):
            with pytest.raises(RuntimeError, match='interrupted'):
                invoke()
        study = optuna.load_study(study_name='campaign', storage=storage)
        orphan = study.ask()
        client.config.optimizer.recover_orphaned_trials = [orphan.number]
        client.config.training.max_total_trials = 2
        invoke()
    assert [t.state for t in study.trials] == [TrialState.FAIL, TrialState.COMPLETE]
    assert study.trials[1].params == params[1]
    assert [t.user_attrs[PROPOSAL_ATTR]['warmup_index'] for t in study.trials] == [0, 1]
    assert provider.requests == []
