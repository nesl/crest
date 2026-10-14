"""Regression checks for full, transport-free CS3 prompt review."""
import importlib.util
import json
import os
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('cs3_api_smoke', ROOT / 'experiments/cs3/api_smoke.py')
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


def test_render_only_full_campaign_context_without_provider(tmp_path):
    """Render both production purposes while forbidding real transport setup.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated pytest output directory.
    """
    output = tmp_path / 'render'
    with patch.object(smoke, 'build_provider', side_effect=AssertionError('No provider initialization')), \
            patch('socket.socket.connect', side_effect=AssertionError('No network')), \
            patch.dict(os.environ, {}, clear=True):
        smoke.run(output, False, render_only=True)
    report = json.loads((output / 'prompts.json').read_text())
    assert report['http_requests'] == 0
    assert 'Fake empty' in report['memory_result']
    assert [r['purpose'] for r in report['requests']] == ['memory', 'candidate']
    for index, request in enumerate(report['requests'], 1):
        ledger = json.loads((output / 'requests' / f'{index:06d}.request.json').read_text())
        assert ledger['messages'] == request['messages']
        body = json.loads(request['messages'][1]['content'])
        context = body['context'] if index == 1 else body
        assert context['semantic_context']['enabled'] is True
        device = context['device_context']
        assert device['name'] == 'STM32_NUCLEO_N657X0_Q'
        assert device['ram_capacity_bytes'] == 4194304
        assert device['flash_capacity_bytes'] == 16777216
        assert len(context['task_context']['feasibility']['rules']) == 4
        assert context['task_context']['objective']['score_type'] == 'scoring-function'
        assert 'synthetic' in context['objective_summary']
        assert 'reconstructed offline proxy' in context['objective_summary']
        params = context['search_space']
        assert len(params) == 14
        assert all(p['description'] and p['typical_effects'] for p in params.values())
        assert params['kernel_time']['units'] == 'frames'
        assert params['base_channels']['units'] == 'channels'


def test_offline_transport_smoke_preserves_accounting(tmp_path):
    """Exercise production HTTP serialization with offline responses and no network.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated pytest output directory.
    """
    output = tmp_path / 'transport'
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'offline-not-a-key'}), \
            patch('socket.socket.connect', side_effect=AssertionError('No network')):
        smoke.run(output, False)
    report = json.loads((output / 'report.json').read_text())
    assert report['passed']
    assert report['http_requests'] == 2
    assert len(report['candidate_validation']['accepted']) == 5
    assert all(r['status'] == 'settled' for r in report['spending']['reservations'].values())
