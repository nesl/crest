# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Compatibility imports and shared adapter failure provenance."""
import io
import json
from unittest.mock import Mock

import pytest

from crest.optimizers.llm.providers import provider
from crest.optimizers.llm.providers.chat import OpenAICompatibleProvider
from crest.optimizers.llm.providers.responses import OpenAIResponsesProvider
from crest.optimizers.llm.providers.records import LLMRequest, ProviderResponseError, SpendLimitExceeded


def test_legacy_imports_reexport_adapter_classes():
    assert provider.OpenAICompatibleProvider is OpenAICompatibleProvider
    assert provider.OpenAIResponsesProvider is OpenAIResponsesProvider
    assert provider.LLMRequest is LLMRequest
    assert provider.ProviderResponseError is ProviderResponseError


@pytest.mark.parametrize('adapter', [OpenAICompatibleProvider, OpenAIResponsesProvider])
@pytest.mark.parametrize('settlement_fails', [False, True])
def test_paid_error_preserves_evidence(adapter, settlement_fails, monkeypatch):
    monkeypatch.setenv('TEST_ADAPTER_KEY', 'secret')
    raw = {'usage': {'input_tokens': 3, 'output_tokens': 2, 'prompt_tokens': 3}}
    spending = {'actual_cost_usd': 0.1}
    tracker = Mock()
    if settlement_fails:
        tracker.settle.side_effect = SpendLimitExceeded('overrun')
    else:
        tracker.settle.return_value = spending
    opener = Mock(return_value=io.BytesIO(json.dumps(raw).encode()))
    client = adapter(provider_name='test', base_url='https://example.invalid/v1',
                     api_key_env='TEST_ADAPTER_KEY', model='test-model',
                     urlopen=opener, spend_tracker=tracker)
    expected = SpendLimitExceeded if settlement_fails else ProviderResponseError
    with pytest.raises(expected) as caught:
        client.complete_json(LLMRequest('system', 'user', 'v1'))
    error = caught.value
    assert error.raw_response == raw
    assert error.wire_payload == json.loads(opener.call_args.args[0].data)
    assert error.usage['prompt_tokens'] == 3
    assert error.spending == (None if settlement_fails else spending)
