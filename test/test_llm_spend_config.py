"""Configuration contracts for optional output and monetary limits."""
from copy import deepcopy

import pytest

from crest.optimizers.llm.config import normalize_llm_config
from crest.optimizers.llm.component import LLMGeneratorOptimizer


def config():
    """Provide unpriced transport settings without credentials or network access."""
    return {'type': 'llm_generator', 'llm': {
        'provider': 'openai_compatible', 'base_url': 'https://provider.example/v1',
        'api_key_env': 'TEST_KEY', 'model': 'fixture-model', 'memory': {'enabled': False}}}


def priced_config():
    """Use explicit synthetic prices rather than assuming real provider rates."""
    result = config()
    result['llm'].update(max_usd=1, max_input_tokens=100000, max_output_tokens=1000,
                         input_usd_per_million_tokens=1, output_usd_per_million_tokens=4,
                         pricing_source='synthetic test tariff')
    return result


@pytest.mark.parametrize('field,value', [
    ('max_output_tokens', 0), ('max_output_tokens', True),
    ('output_token_parameter', 'unsupported'), ('output_token_parameter', []),
    ('max_input_tokens', False), ('max_usd', -1), ('max_usd', float('nan')),
    ('input_usd_per_million_tokens', 0), ('output_usd_per_million_tokens', float('inf')),
    ('pricing_source', ''),
])
def test_invalid_spend_settings_fail_before_provider(field, value):
    """Reject unsafe limits and prices before setup can make any request."""
    supplied = priced_config()
    supplied['llm'][field] = value
    with pytest.raises(ValueError):
        normalize_llm_config(supplied)


@pytest.mark.parametrize('field', ['max_output_tokens', 'max_input_tokens', 'pricing_source',
                                   'input_usd_per_million_tokens', 'output_usd_per_million_tokens'])
def test_cap_requires_complete_accounting_settings(field):
    """A dollar cap cannot be enabled with an implicit tariff or token bound."""
    supplied = priced_config()
    del supplied['llm'][field]
    with pytest.raises(ValueError):
        normalize_llm_config(supplied)


def test_output_limit_is_identity_but_cap_can_extend():
    """Keep old identity unchanged until generation limits are explicitly enabled."""
    optimizer = LLMGeneratorOptimizer()
    baseline = optimizer.identity_config(config())
    capped = priced_config()
    identity = optimizer.identity_config(capped)
    assert set(identity) - set(baseline) == {'max_output_tokens', 'output_token_parameter'}
    assert identity['output_token_parameter'] == 'max_completion_tokens'
    capped['llm']['max_usd'] = 10
    assert optimizer.identity_config(capped) == identity
    capped['llm']['max_output_tokens'] = 2000
    assert optimizer.identity_config(capped) != identity
    capped['llm']['max_usd'] = 0
    assert normalize_llm_config(capped).llm.max_usd == 0
