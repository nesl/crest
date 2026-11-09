"""Check that portable campaign configs retain the paper's experiment axes."""

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from crest.component_selection import resolve_component_selection
from crest.model import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / 'src/config/case_study_configs'
CAMPAIGNS = sorted(CONFIGS.glob('cs*/*.yaml'))


def test_campaign_inventory():
    """Keep all six campaigns for each case study discoverable."""
    assert len(CAMPAIGNS) == 18
    assert not list(CONFIGS.glob('*.yaml'))
    for case in ('cs1', 'cs2', 'cs3'):
        assert len(list((CONFIGS / case).glob('*.yaml'))) == 6


@pytest.mark.parametrize('path', CAMPAIGNS, ids=lambda p: f'{p.parent.name}/{p.stem}')
def test_campaign_loads_without_data_credentials_or_hardware(path, monkeypatch):
    """Validate shipped schema, component selection, and portable resource paths.

    Parameters
    ----------
    path : pathlib.Path
        Campaign YAML being checked.
    monkeypatch : pytest.MonkeyPatch
        Environment isolation for credentials.
    """
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    config = load_config(path)
    selection = resolve_component_selection(config)
    assert selection
    assert config.training.nas_epochs == 55
    if config.device.name == 'STM32_NUCLEO_N657X0_Q':
        assert (ROOT / config.device.stm32.project_root / 'Appli/Src/crest_dut_runner.c').is_file()
        assert (ROOT / config.device.stm32.weights_memory_pool).is_file()
    if config.optimizer.type == 'llm_generator':
        assert Path(config.optimizer.llm.warmup_params_file) == CONFIGS / 'cs3/warmup_params.json'
        assert config.optimizer.llm.api_key_env == 'OPENAI_API_KEY'


@pytest.mark.parametrize('period', (100, 200, 500))
def test_cs1_objectives_keep_historical_schedule_and_precision(period):
    """Distinguish energy objective from two-pass collection and older budgets.

    Parameters
    ----------
    period : int
        Release period in milliseconds.
    """
    for objective in ('continuous', 'periodic'):
        config = yaml.safe_load((CONFIGS / f'cs1/stm32_{objective}_{period}ms.yaml').read_text())
        device, training = config['device'], config['training']
        data = config['dataset']['params']
        budget = device.get('latency_budget_ms', 1000 * data['stride'] / data['sampling_rate_hz'])
        assert budget == period
        metrics = [item['metric'] for item in config['nas']['score']['params']['objectives']]
        assert metrics == ['rmse_total', 'energy_mj_per_inference' if objective == 'continuous' else 'cadenced_energy_mj_per_window']
        assert device['runtime_mode'] == ('back_to_back' if period == 200 and objective == 'continuous' else 'cadenced')
        assert training['max_total_trials'] == (600 if period == 200 else 256)
        assert training['quantization']['search'] is (period == 200)


def test_cs2_paper_proxies_search_both_precisions_without_hardware():
    """Guard the paper's mixed-precision proxies against generic-template drift."""
    for name in ('flops', 'memory'):
        config = load_config(CONFIGS / f'cs2/oxiod_{name}_proxy.yaml')
        assert config.device.hil is False
        assert config.training.quantization.search is True
        assert list(config.training.quantization.choices) == ['float', 'int8_ptq']
        assert config.training.random_seed == 42
        assert config.training.nas_trials == config.training.max_total_trials == 150


def test_cs3_policies_share_exact_warmups_and_keep_measurement_differences():
    """Preserve the recorded warm-up list and target-specific measurement work."""
    warmup = CONFIGS / 'cs3/warmup_params.json'
    assert hashlib.sha256(warmup.read_bytes()).hexdigest() == 'e7d7c363e605da016637bf2dc4deadf634f9b5fd18e918d8d0da8a29a34e8d9e'
    assert len(json.loads(warmup.read_text())) == 15
    for board in ('portenta', 'stm32'):
        for policy in ('tpe', 'memory', 'history'):
            config = load_config(CONFIGS / f'cs3/audio_{board}_{policy}.yaml')
            assert config.training.nas_trials == 200
            assert config.training.max_total_trials == 300
            assert config.device.runtime_mode == ('back_to_back' if (board, policy) == ('portenta', 'tpe') else 'cadenced')
            if policy == 'tpe':
                assert config.optimizer.type == 'optuna'
            else:
                llm = config.optimizer.llm
                assert llm.model == 'gpt-6-luna'
                assert llm.reasoning_effort == 'medium'
                assert llm.require_rationale is True
                assert llm.memory.enabled is (policy == 'memory')
                assert llm.get('history_mode', 'bounded') == ('full' if policy == 'history' else 'bounded')
