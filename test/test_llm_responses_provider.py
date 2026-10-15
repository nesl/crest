# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Offline Responses wire, content, and durable accounting regressions."""

import io
import json
import urllib.error
from unittest.mock import Mock

import pytest

from crest.optimizers.llm.provider import (
    LLMRequest, OpenAIResponsesProvider, ProviderResponseError,
    ProviderTransportError, SpendLimitExceeded, build_provider,
)
from crest.optimizers.llm.spending import SpendTracker, SpendInputLimitExceeded


class HTTPResponse(io.BytesIO):
    """Inject a JSON HTTP response without accessing a network."""

    def __init__(self, payload):
        super().__init__(json.dumps(payload).encode())


def config(**updates):
    """Build conservative test prices.

    Returns
    -------
    dict
        Offline provider configuration.
    """
    return dict(provider="openai_responses", model="gpt-6-luna", base_url="https://example.invalid/v1",
                api_key_env="CREST_TEST_RESPONSES_KEY", temperature=None, reasoning_effort="medium",
                max_output_tokens=100, max_input_tokens=2048, max_usd="0.01",
                input_usd_per_million_tokens="1", output_usd_per_million_tokens="2",
                pricing_source="offline configured bound", output_token_parameter="max_completion_tokens",
                **updates)


def envelope(**updates):
    """Build a complete Responses fixture.

    Returns
    -------
    dict
        Completed assistant output and actual usage.
    """
    return {"model": "gpt-6-luna", "status": "completed", "service_tier": "default",
            "output": [{"type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": '{"candidates":[]}'}]}],
            "usage": {"input_tokens": 30, "output_tokens": 5,
                      "input_tokens_details": {"cached_tokens": 10, "cache_write_tokens": 2}},
            **updates}


@pytest.fixture
def llm_request():
    """Keep fixed prefix followed by chronological context.

    Returns
    -------
    LLMRequest
        Minimal ordered provider llm_request.
    """
    return LLMRequest("fixed JSON instructions", "history rows\ndynamic suffix", "next-v1")


@pytest.fixture(autouse=True)
def credentials(monkeypatch):
    """Use a fake credential so tests cannot require account access.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Temporary environment mutation fixture.
    """
    monkeypatch.setenv("CREST_TEST_RESPONSES_KEY", "offline-secret")


def test_responses_wire_is_no_cache_and_ordered(llm_request):
    # Purpose: verify opt-in Responses uses supported fields without any writes.
    provider = build_provider(config())
    assert isinstance(provider, OpenAIResponsesProvider)
    wire = provider.build_http_request(llm_request)
    payload = json.loads(wire.data)
    assert wire.full_url == "https://example.invalid/v1/responses"
    assert payload["instructions"] == llm_request.system_prompt
    assert payload["input"] == llm_request.user_prompt
    assert payload["prompt_cache_options"] == {"mode": "explicit"}
    assert "prompt_cache_breakpoint" not in wire.data.decode()
    assert payload["max_output_tokens"] == 100
    assert payload["reasoning"] == {"effort": "medium"}
    assert payload["text"] == {"format": {"type": "json_object"}}
    assert not payload["store"]
    assert "messages" not in payload and "temperature" not in payload
    assert "offline-secret" not in wire.data.decode()


def test_normalized_response_and_conservative_cache_accounting(tmp_path, llm_request):
    # Purpose: actual usage settles at configured full rates and survives resume.
    tracker = SpendTracker(tmp_path, config())
    provider = build_provider(config(), spend_tracker=tracker)
    provider._urlopen = Mock(return_value=HTTPResponse(envelope()))
    result = provider.complete_json(llm_request)
    assert result.content == '{"candidates":[]}'
    assert result.finish_reason == "completed"
    assert result.usage["prompt_tokens"] == 30
    assert result.usage["cached_tokens"] == 10
    assert result.usage["cache_write_tokens"] == 2
    assert result.metadata["service_tier"] == "default"
    assert result.metadata["spending"]["charged_usd"] == "0.00004"
    saved = json.loads((tmp_path / "spend_state.json").read_text())
    assert saved["reservations"]["1"]["wire_payload"] == result.metadata["wire_payload"]
    assert saved["reservations"]["1"]["cached_tokens"] == 10
    SpendTracker(tmp_path, config())


@pytest.mark.parametrize("changed", [
    {"status": "incomplete"}, {"status": "failed"}, {"output": []},
    {"output": [{"type": "message", "role": "assistant", "status": "completed",
                 "content": [{"type": "refusal", "refusal": "no"}]}]},
    {"output": None}, {"output": [{"type": "message", "content": []}]},
])
def test_invalid_envelopes_keep_raw_provenance_and_actual_spend(tmp_path, llm_request, changed):
    # Purpose: malformed/refused/incomplete output allows repair after settlement.
    tracker = SpendTracker(tmp_path, config())
    provider = build_provider(config(), spend_tracker=tracker)
    raw = envelope(**changed)
    provider._urlopen = Mock(return_value=HTTPResponse(raw))
    with pytest.raises(ProviderResponseError) as error:
        provider.complete_json(llm_request)
    assert error.value.raw_response == raw
    assert error.value.spending["charged_usd"] == "0.00004"


def test_retries_reserve_each_attempt_and_preserve_unknown_cost(tmp_path, llm_request):
    # Purpose: ambiguous HTTP attempts remain charged after a successful retry.
    tracker = SpendTracker(tmp_path, config())
    provider = build_provider(config(), spend_tracker=tracker)
    provider._sleep = Mock()
    provider._urlopen = Mock(side_effect=[urllib.error.URLError("offline"), HTTPResponse(envelope())])
    result = provider.complete_json(llm_request)
    assert result.metadata["transport_attempts"] == 2
    saved = json.loads((tmp_path / "spend_state.json").read_text())
    assert saved["reservations"]["1"]["status"] == "reserved"
    assert saved["charged_usd"] == "0.002288"
    SpendTracker(tmp_path, config())


def test_spend_cap_stops_before_network_and_input_guard_is_unchanged(tmp_path, llm_request):
    # Purpose: Responses cannot bypass cap/input protections via its new shape.
    settings = config()
    settings["max_usd"] = "0.002248"
    tracker = SpendTracker(tmp_path, settings)
    provider = build_provider(settings, spend_tracker=tracker)
    provider._sleep = Mock()
    provider._urlopen = Mock(side_effect=urllib.error.URLError("offline"))
    with pytest.raises(SpendLimitExceeded):
        provider.complete_json(llm_request)
    assert provider._urlopen.call_count == 1
    with pytest.raises(SpendInputLimitExceeded):
        tracker.reserve({"instructions": "x" * 2048, "input": "history"})


def test_cache_miss_counters_are_explicit(llm_request):
    # Purpose: no-cache accounting retains zero counters when details are absent.
    provider = build_provider(config())
    provider._urlopen = Mock(return_value=HTTPResponse(envelope(usage={"input_tokens": 4, "output_tokens": 2})))
    result = provider.complete_json(llm_request)
    assert result.usage["cached_tokens"] == 0
    assert result.usage["cache_write_tokens"] == 0


def test_malformed_raw_usage_keeps_reservation(tmp_path, llm_request):
    # Purpose: missing token counts never release a durable reservation.
    tracker = SpendTracker(tmp_path, config())
    provider = build_provider(config(), spend_tracker=tracker)
    provider._urlopen = Mock(return_value=HTTPResponse(envelope(usage={"input_tokens": 4})))
    result = provider.complete_json(llm_request)
    assert result.metadata["spending"]["status"] == "unknown"
    assert result.metadata["spending"]["charged_usd"] == "0.002248"
