# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Paid request limits remain shared and persistent across real runner rounds."""

import json
import os
from pathlib import Path
from unittest.mock import Mock, patch

import optuna
import pytest
from addict import Dict
from optuna.trial import TrialState

from crest.optimizers.llm.provider import build_provider
from nas_model_client import MeasurementInfrastructureError, PROPOSAL_ATTR
from test.test_llm_openai_provider import FakeHTTPResponse
from test.test_optimizer_resume import campaign


def paid_campaign(client, tmp_path, *, memory=False, max_usd=4.0):
    """Configure fifteen fixed attempts and conservative one-dollar reservations.

    Parameters
    ----------
    client : NASModelClient
        Lightweight runner supplied by the campaign fixture.
    tmp_path : pathlib.Path
        Directory for the ordered warm-up parameter file.
    memory : bool, optional
        Enable one summary after the complete warm-up sequence.
    max_usd : float, optional
        Persistent campaign spending cap.

    Returns
    -------
    list of dict
        Fixed warm-up candidates in execution order.
    """
    params = [{'width': 2 + i % 6} for i in range(15)]
    path = tmp_path / 'warmup.json'
    path.write_text(json.dumps(params))
    client.config.optimizer = Dict(type='llm_generator', llm={
        'provider': 'openai_compatible', 'base_url': 'https://example.test/v1',
        'model': 'test-model', 'api_key_env': 'CREST_TEST_ONLY_API_KEY',
        'batch_size': 1, 'warmup_params_file': str(path), 'recent_trial_window': 15,
        'max_usd': max_usd, 'max_input_tokens': 100000, 'max_output_tokens': 1000,
        'input_usd_per_million_tokens': 9, 'output_usd_per_million_tokens': 100,
        'pricing_source': 'test fixture fixed pricing',
        'memory': {'enabled': memory, 'start_after_trials': 15, 'interval_trials': 15},
    })
    client.config.training.nas_trials = 20
    client.config.training.max_total_trials = 16
    return params


def completion(content):
    """Return a valid completion envelope costing nine tenths of a reservation.

    Parameters
    ----------
    content : dict
        Candidate or memory response returned by the mocked service.

    Returns
    -------
    FakeHTTPResponse
        OpenAI-compatible completion including explicit billed token counts.
    """
    return FakeHTTPResponse({
        'choices': [{'message': {'content': json.dumps(content)}, 'finish_reason': 'stop'}],
        'usage': {'prompt_tokens': 90000, 'completion_tokens': 900, 'total_tokens': 90900},
    })


def provider_factory(transport):
    """Inject HTTP transport while preserving provider construction and accounting.

    Parameters
    ----------
    transport : callable
        Mock HTTP opener used by every reconstructed provider.

    Returns
    -------
    callable
        Factory accepting the component's provider and spending arguments.
    """
    def build(config, **kwargs):
        """Construct the production provider and replace only its HTTP opener."""
        provider = build_provider(config, **kwargs)
        provider._urlopen = transport
        return provider
    return build


def ledger_root(client):
    """Return the campaign's persistent LLM artifact directory."""
    return Path(client.config.outputs.models_dir) / 'campaign' / 'llm_optimizer'


def test_warmup_resume_is_free_and_first_paid_prompt_keeps_all_fifteen_outcomes(campaign, tmp_path):
    """Fixed attempts consume no reservations and retain failures on paid resume."""
    client, storage = campaign
    params = paid_campaign(client, tmp_path, max_usd=1)
    transport = Mock(return_value=completion({'candidates': [{'width': 8}]}))

    def objective(trial):
        """Preserve representative terminal failures in the fixed history."""
        width = trial.suggest_int('width', 2, 8)
        if trial.number < 15:
            transport.assert_not_called()
            assert width == params[trial.number]['width']
        if trial.number == 6:
            trial.set_user_attr('measurement_failure_class', 'infrastructure')
            raise MeasurementInfrastructureError('fixture failure')
        if trial.number == 9:
            raise optuna.TrialPruned('fixture rejection')
        return float(width)

    client.objective = objective
    with patch.dict(os.environ, {'CREST_TEST_ONLY_API_KEY': 'test-only'}), patch(
            'crest.optimizers.llm.component.build_provider', side_effect=provider_factory(transport)):
        for cap in (7, 15):
            client.config.training.max_total_trials = cap
            study = client.run_nas('campaign', storage=storage)
            assert len(study.trials) == cap
            transport.assert_not_called()
            # Constructing or resuming a tracker cannot reserve a paid request.
            state_path = ledger_root(client) / 'spend_state.json'
            if state_path.exists():
                state = json.loads(state_path.read_text())
                assert state['charged_usd'] == '0'
                assert state['reservations'] == {}
        client.config.training.max_total_trials = 16
        study = client.run_nas('campaign', storage=storage)

    assert len(study.trials) == 16
    assert [trial.params for trial in study.trials[:15]] == params
    assert study.trials[6].state == TrialState.FAIL
    assert study.trials[9].state == TrialState.PRUNED
    assert study.trials[15].user_attrs[PROPOSAL_ATTR]['proposal_source'] == 'llm'
    transport.assert_called_once()
    request = json.loads(transport.call_args.args[0].data)
    assert request['max_completion_tokens'] == 1000
    history = json.loads(request['messages'][1]['content'])['recent_trials']
    assert [trial['number'] for trial in history] == list(range(15))
    assert history[6]['state'] == 'fail' and history[9]['state'] == 'pruned'
    state = json.loads((ledger_root(client) / 'spend_state.json').read_text())
    assert state['charged_usd'] == '0.9'
    assert len(state['reservations']) == 1


def test_memory_and_candidate_repairs_share_cap_and_resume_cannot_reset_it(campaign, tmp_path):
    """All repair calls are charged before a resumed campaign stops at its cap."""
    client, storage = campaign
    paid_campaign(client, tmp_path, memory=True)
    transport = Mock(side_effect=[
        completion({}), completion({'updates': [], 'retire': []}),
        completion({}), completion({'candidates': [{'width': 8}]}),
    ])
    with patch.dict(os.environ, {'CREST_TEST_ONLY_API_KEY': 'test-only'}), patch(
            'crest.optimizers.llm.component.build_provider', side_effect=provider_factory(transport)):
        study = client.run_nas('campaign', storage=storage)
        assert len(study.trials) == 16
        assert transport.call_count == 4
        state_path = ledger_root(client) / 'spend_state.json'
        before = json.loads(state_path.read_text())
        assert float(before['charged_usd']) == pytest.approx(3.6)
        assert len(before['reservations']) == 4
        assert all(record['status'] == 'settled' for record in before['reservations'].values())
        prompts = [json.loads(call.args[0].data)['messages'][1]['content']
                   for call in transport.call_args_list]
        memory_prompt, memory_repair, proposal, proposal_repair = map(json.loads, prompts)
        assert memory_prompt['new_trials'] == memory_repair['new_trials']
        assert memory_repair['repair_feedback']
        assert proposal_repair['repair_feedback']
        assert [trial['number'] for trial in proposal['recent_trials']] == list(range(15))
        assert len(list((ledger_root(client) / 'memory' / 'requests').glob('*.response.json'))) == 2
        assert len(list((ledger_root(client) / 'requests').glob('*.response.json'))) == 2

        client.config.training.max_total_trials = 17
        for _ in range(2):
            resumed = client.run_nas('campaign', storage=storage)
            assert resumed.trials == study.trials
            assert transport.call_count == 4
            assert json.loads(state_path.read_text()) == before
        del client.config.optimizer.llm['max_usd']
        try:
            with pytest.raises(ValueError, match='retain max_usd'):
                client.run_nas('campaign', storage=storage)
        finally:
            client.config.optimizer.llm.max_usd = 4.0
        assert optuna.load_study(study_name='campaign', storage=storage).trials == study.trials
        assert transport.call_count == 4
        assert json.loads(state_path.read_text()) == before

    assert all(trial.user_attrs[PROPOSAL_ATTR]['proposal_source'] != 'random_fallback'
               for trial in resumed.trials)


def test_smoke_cap_exhaustion_stops_before_http_and_restores_temporary_config(campaign, tmp_path):
    """An unaffordable smoke request stops cleanly, including on repeated resume."""
    client, _ = campaign
    paid_campaign(client, tmp_path, max_usd=0)
    del client.config.optimizer.llm['warmup_params_file']
    transport = Mock(side_effect=AssertionError('unreserved HTTP request'))
    client.objective = Mock(side_effect=AssertionError('unexpected trial evaluation'))
    original = (client.config.device.hil, client.config.training.train,
                client.config.training.nas_epochs)
    with patch.dict(os.environ, {'CREST_TEST_ONLY_API_KEY': 'test-only'}), patch(
            'crest.optimizers.llm.component.build_provider', side_effect=provider_factory(transport)):
        for _ in range(2):
            client.smoke_test(train=True, hil=True, epochs=1, trials=1, study_name='campaign')
            assert (client.config.device.hil, client.config.training.train,
                    client.config.training.nas_epochs) == original
            storage = f"sqlite:///{client._artifacts_dir() / 'optuna_smoke_test.db'}"
            assert optuna.load_study(study_name='campaign', storage=storage).trials == []
            state = json.loads((ledger_root(client) / 'spend_state.json').read_text())
            assert state['charged_usd'] == '0' and state['reservations'] == {}
    transport.assert_not_called()
    client.objective.assert_not_called()


@pytest.mark.parametrize('multiobjective', [False, True])
def test_scoring_zero_cap_saves_evidence_skips_final_training_and_resumes(campaign, tmp_path, multiobjective):
    """Empty budget stops retain scoring artifacts and clear after a paid extension."""
    client, _ = campaign
    paid_campaign(client, tmp_path, max_usd=0)
    del client.config.optimizer.llm['warmup_params_file']
    client.config.training.nas_trials = 1
    client.config.training.max_total_trials = 1
    if multiobjective:
        client.config.nas.score = Dict(type='multi-objective', metrics={}, params={'objectives': [
            {'name': 'flops', 'metric': 'flops', 'direction': 'minimize'},
            {'name': 'flops_max', 'metric': 'flops', 'direction': 'maximize'},
        ]})
    transport = Mock(return_value=completion({'candidates': [{'width': 8}]}))
    original_objective = client.objective
    client.objective = Mock(side_effect=AssertionError('unexpected zero-cap evaluation'))
    with patch.dict(os.environ, {'CREST_TEST_ONLY_API_KEY': 'test-only'}), patch(
            'crest.optimizers.llm.component.build_provider', side_effect=provider_factory(transport)), patch.object(
            client, 'train_best_trial') as train_best:
        client.run_scoring_nas('campaign')
        artifacts = client._artifacts_dir()
        storage = f"sqlite:///{artifacts / 'optuna.db'}"
        stopped = optuna.load_study(study_name='campaign', storage=storage)
        assert stopped.trials == []
        assert stopped.user_attrs['crest_optimizer_budget_stop']
        assert (artifacts / 'trials.csv').exists()
        state_path = ledger_root(client) / 'spend_state.json'
        assert json.loads(state_path.read_text())['reservations'] == {}
        transport.assert_not_called()
        client.objective.assert_not_called()
        train_best.assert_not_called()

        client.config.optimizer.llm.max_usd = 1

        def objective(trial):
            """Return correctly shaped objective values for the single resumed trial."""
            value = original_objective(trial)
            return (value, value) if multiobjective else value

        client.objective = objective
        resumed = client.run_nas('campaign', storage=storage)
        assert len(resumed.trials) == 1
        assert resumed.trials[0].state == TrialState.COMPLETE
        assert resumed.user_attrs['crest_optimizer_budget_stop'] is None
        transport.assert_called_once()
        train_best.assert_not_called()
