"""Tariff selection, custom catalogs, and study-bound pricing evidence."""
import json

import pytest

from crest.optimizers.llm.config import normalize_llm_config
from crest.optimizers.llm.providers.pricing import resolve_pricing
from crest.optimizers.llm.spending import SpendTracker, SpendAccountingError


def config():
    return {'type': 'llm_generator', 'llm': {
        'provider': 'openai_compatible', 'base_url': 'https://api.openai.com/v1',
        'model': 'gpt-4.1-mini', 'api_key_env': 'TEST_KEY',
        'max_usd': 2, 'max_input_tokens': 1000, 'max_output_tokens': 100,
        'memory': {'enabled': False}}}


def custom(tmp_path, entry=None):
    entry = entry or resolve_pricing(config()['llm'])
    path = tmp_path / 'prices.json'
    path.write_text(json.dumps({'schema_version': 1, 'rates': [entry]}))
    supplied = config()
    supplied['llm']['pricing_file'] = str(path)
    return supplied, path


def test_bundled_rates_and_repeat_normalization():
    normalized = normalize_llm_config(config())
    assert normalized.llm.input_usd_per_million_tokens == 0.4
    assert normalized.llm.output_usd_per_million_tokens == 1.6
    assert normalize_llm_config(normalized) == normalized


@pytest.mark.parametrize('field,value', [('provider', 'openrouter'), ('model', 'unknown'),
    ('base_url', 'https://vendor.example/v1'), ('service_tier', 'priority')])
def test_no_price_guessing(field, value):
    supplied = config()
    supplied['llm'][field] = value
    with pytest.raises(ValueError, match='pricing entry'):
        normalize_llm_config(supplied)


@pytest.mark.parametrize('field,value', [('currency', 'EUR'), ('verified_on', 'yesterday'),
    ('source', ''), ('input_usd_per_million_tokens', 0),
    ('output_usd_per_million_tokens', float('nan'))])
def test_invalid_tariff(tmp_path, field, value):
    entry = resolve_pricing(config()['llm'])
    entry[field] = value
    supplied, _ = custom(tmp_path, entry)
    with pytest.raises(ValueError):
        normalize_llm_config(supplied)


def test_missing_malformed_and_ambiguous_catalog(tmp_path):
    supplied, path = custom(tmp_path)
    for content in ['not json', '{}', json.dumps({'schema_version': 1, 'rates':
            [resolve_pricing(config()['llm'])] * 2})]:
        path.write_text(content)
        with pytest.raises(ValueError):
            normalize_llm_config(supplied)
    path.unlink()
    with pytest.raises(ValueError, match='Cannot read'):
        normalize_llm_config(supplied)


def test_snapshot_resume_relocation_and_changed_rate(tmp_path):
    supplied, path = custom(tmp_path)
    normalized = normalize_llm_config(supplied)
    root = tmp_path / 'study'
    SpendTracker(root, normalized.llm)
    state = json.loads((root / 'spend_state.json').read_text())
    assert state['binding']['pricing_snapshot'] == normalized.llm.pricing_snapshot
    assert str(path) not in json.dumps(state)
    # Same tariff at a different path remains the same experiment.
    relocated = tmp_path / 'relocated.json'
    relocated.write_text(path.read_text())
    supplied['llm']['pricing_file'] = str(relocated)
    SpendTracker(root, normalize_llm_config(supplied).llm)
    changed = json.loads(path.read_text())
    changed['rates'][0]['output_usd_per_million_tokens'] = 9
    path.write_text(json.dumps(changed))
    # Re-normalizing an already normalized config must re-read its catalog.
    updated = normalize_llm_config(normalized)
    assert updated.llm.output_usd_per_million_tokens == 9
    with pytest.raises(SpendAccountingError, match='Spend settings changed'):
        SpendTracker(root, updated.llm)
    assert json.loads((root / 'spend_state.json').read_text()) == state


def test_custom_catalog_for_private_endpoint(tmp_path):
    entry = resolve_pricing(config()['llm'])
    entry.update(model='private-model', base_url='https://vendor.example/v1',
                 service_tier='priority', input_usd_per_million_tokens=2)
    supplied, _ = custom(tmp_path, entry)
    supplied['llm'].update({k: entry[k] for k in ('model', 'base_url', 'service_tier')})
    assert normalize_llm_config(supplied).llm.input_usd_per_million_tokens == 2


def test_catalog_and_inline_prices_are_not_ambiguous(tmp_path):
    supplied, _ = custom(tmp_path)
    supplied['llm']['input_usd_per_million_tokens'] = 999
    with pytest.raises(ValueError, match='not both'):
        normalize_llm_config(supplied)


def test_yaml_relative_pricing_path(tmp_path, monkeypatch):
    import yaml
    from crest.model import load_config
    supplied, path = custom(tmp_path)
    supplied['llm']['pricing_file'] = path.name
    document = {
        'optimizer': supplied, 'device': {'name': 'TEST_DEVICE'},
        'training': {'nas_trials': 5, 'quantization': {'mode': 'int8_ptq', 'search': False, 'choices': ['int8_ptq']}},
        'score': {'type': 'scoring-function', 'params': {'terms': [
            {'type': 'weighted', 'metric': 'flops', 'weight': -1.0}]}},
        'outputs': {'models_dir': str(tmp_path / 'models'),
                    'candidate_dir': str(tmp_path / 'candidate'), 'artifact_stem': 'test'},
    }
    document['nas'] = {'score': document.pop('score')}
    config_path = tmp_path / 'experiment.yaml'
    config_path.write_text(yaml.safe_dump(document))
    monkeypatch.chdir(tmp_path.parent)
    loaded = load_config(config_path)
    assert loaded.optimizer.llm.pricing_file == str(path.resolve())
    assert loaded.optimizer.llm.input_usd_per_million_tokens == 0.4
