# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for the OpenRouter/OpenAI-compatible provider."""

import http.client
import io
import json
import os
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import Mock, call, patch

from addict import Dict

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crest.optimizers.llm.provider import (  # noqa: E402
    LLMRequest,
    OpenAICompatibleProvider,
    ProviderAuthenticationError,
    ProviderResponseError,
    ProviderTransportError,
    build_provider,
)


class FakeHTTPResponse:
    """Context-managed byte response used by the injected transport."""

    def __init__(self, payload) -> None:
        """Encode one JSON response payload."""
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        """Return this response from a context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Leave the fake response context without suppressing errors."""
        return False

    def read(self) -> bytes:
        """Return the configured encoded response body."""
        return self.payload


def request() -> LLMRequest:
    """Build one provider-neutral completion request."""
    return LLMRequest(
        system_prompt="system",
        user_prompt='{"search_space":{}}',
        prompt_version="v1",
        metadata={"phase": "exploration"},
    )


class OpenAICompatibleProviderTests(unittest.TestCase):
    """Verify request construction and response normalization without network."""

    def test_request_construction_is_openai_compatible_and_authenticated(self) -> None:
        """OpenRouter uses chat completions, JSON mode, and configured headers."""
        provider = OpenAICompatibleProvider(
            provider_name="openrouter",
            base_url="https://openrouter.ai/api/v1/",
            api_key_env="TEST_LLM_KEY",
            model="openai/test-model",
            temperature=0.25,
            timeout_s=12,
            json_response_mode=True,
            extra_headers={"HTTP-Referer": "https://example.test", "X-Title": "CREST"},
        )
        with patch.dict(os.environ, {"TEST_LLM_KEY": "secret-value"}):
            http_request = provider.build_http_request(request())

        payload = json.loads(http_request.data)
        self.assertEqual(http_request.full_url, "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(http_request.method, "POST")
        self.assertEqual(http_request.get_header("Authorization"), "Bearer secret-value")
        self.assertEqual(http_request.get_header("Content-type"), "application/json")
        self.assertEqual(http_request.get_header("Http-referer"), "https://example.test")
        self.assertEqual(payload["model"], "openai/test-model")
        self.assertEqual(payload["messages"], request().messages())
        self.assertEqual(payload["temperature"], 0.25)
        self.assertEqual(payload["response_format"], {"type": "json_object"})

    def test_generic_provider_can_omit_json_response_format(self) -> None:
        """Compatibility mode avoids an option unsupported by some models."""
        provider = OpenAICompatibleProvider(
            provider_name="openai_compatible",
            base_url="https://llm.example/v1",
            api_key_env="TEST_LLM_KEY",
            model="example-model",
            json_response_mode=False,
        )
        with patch.dict(os.environ, {"TEST_LLM_KEY": "secret-value"}):
            http_request = provider.build_http_request(request())

        payload = json.loads(http_request.data)
        self.assertNotIn("response_format", payload)

    def test_response_content_usage_and_provider_metadata_are_normalized(self) -> None:
        """Provider-specific response shape stays isolated behind LLMResponse."""
        captured = {}

        def urlopen(http_request, timeout):
            captured.update({"request": http_request, "timeout": timeout})
            return FakeHTTPResponse(
                {
                    "model": "openai/resolved-model",
                    "choices": [
                        {
                            "message": {"content": '{"candidates":[{"width":4}]}'},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
                }
            )

        provider = OpenAICompatibleProvider(
            provider_name="openai_compatible",
            base_url="https://llm.example/v1",
            api_key_env="TEST_LLM_KEY",
            model="requested-model",
            timeout_s=7,
            urlopen=urlopen,
        )
        with patch.dict(os.environ, {"TEST_LLM_KEY": "secret-value"}):
            response = provider.complete_json(request())

        self.assertEqual(response.content, '{"candidates":[{"width":4}]}')
        self.assertEqual(response.model, "openai/resolved-model")
        self.assertEqual(response.usage["total_tokens"], 18)
        self.assertEqual(response.finish_reason, "stop")
        self.assertEqual(response.metadata["base_url"], "https://llm.example/v1")
        self.assertEqual(captured["timeout"], 7.0)
        self.assertNotIn("secret-value", repr(response))

    def test_missing_api_key_fails_before_transport(self) -> None:
        """The configured environment-variable name is required at call time."""
        provider = OpenAICompatibleProvider(
            provider_name="openrouter",
            base_url="https://openrouter.ai/api/v1",
            api_key_env="DEFINITELY_MISSING_LLM_KEY",
            model="openai/test-model",
        )
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ProviderAuthenticationError, "DEFINITELY_MISSING_LLM_KEY"):
                provider.complete_json(request())

    def test_authentication_and_credit_errors_fail_immediately(self) -> None:
        """Invalid credentials/credits cannot enter repair or transport retries."""
        for status in (401, 402, 403):
            with self.subTest(status=status):
                transport = Mock(side_effect=urllib.error.HTTPError(
                    "https://llm.example/v1", status, "denied", {}, io.BytesIO(b"account blocked")))
                sleeper = Mock()
                provider = OpenAICompatibleProvider(
                    provider_name="test", base_url="https://llm.example/v1", api_key_env="TEST_LLM_KEY",
                    model="test", urlopen=transport, sleep=sleeper)
                with patch.dict(os.environ, {"TEST_LLM_KEY": "secret"}):
                    with self.assertRaisesRegex(ProviderAuthenticationError, f"HTTP {status}.*credits"):
                        provider.complete_json(request())
                self.assertEqual(transport.call_count, 1)
                sleeper.assert_not_called()

    def test_transient_http_failures_back_off_before_success(self) -> None:
        """429 and server failures retry internally and report actual attempts."""
        transport = Mock(side_effect=[
            urllib.error.HTTPError("https://llm.example/v1", 429, "rate limited", {}, io.BytesIO(b"busy")),
            urllib.error.HTTPError("https://llm.example/v1", 503, "unavailable", {}, io.BytesIO(b"busy")),
            FakeHTTPResponse({"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]}),
        ])
        sleeper = Mock()
        provider = OpenAICompatibleProvider(
            provider_name="test", base_url="https://llm.example/v1", api_key_env="TEST_LLM_KEY",
            model="test", urlopen=transport, sleep=sleeper)
        with patch.dict(os.environ, {"TEST_LLM_KEY": "secret"}):
            response = provider.complete_json(request())
        self.assertEqual(transport.call_count, 3)
        self.assertEqual(sleeper.call_args_list, [call(5.0), call(10.0)])
        self.assertEqual(response.metadata["transport_attempts"], 3)

    def test_timeouts_and_network_errors_exhaust_a_bounded_retry_budget(self) -> None:
        """The default budget permits six attempts with capped exponential waits."""
        for error in (TimeoutError("timed out"), urllib.error.URLError("connection lost")):
            with self.subTest(error=error):
                transport = Mock(side_effect=error)
                sleeper = Mock()
                provider = OpenAICompatibleProvider(
                    provider_name="test", base_url="https://llm.example/v1", api_key_env="TEST_LLM_KEY",
                    model="test", urlopen=transport, sleep=sleeper)
                with patch.dict(os.environ, {"TEST_LLM_KEY": "secret"}):
                    with self.assertRaisesRegex(ProviderTransportError, "Exhausted 6 transport attempts"):
                        provider.complete_json(request())
                self.assertEqual(transport.call_count, 6)
                self.assertEqual(sleeper.call_args_list,
                                 [call(5.0), call(10.0), call(20.0), call(40.0), call(60.0)])

    def test_retry_after_is_a_minimum_for_delta_seconds_and_http_dates(self) -> None:
        """Server guidance overrides shorter backoff but never shortens it."""
        # Freeze Unix time at the HTTP-date epoch so no wall-clock waits occur.
        cases = [("20", 20.0), ("Thu, 01 Jan 1970 00:00:30 GMT", 30.0),
                 ("1", 5.0), ("Wed, 31 Dec 1969 23:59:59 GMT", 5.0),
                 ("malformed", 5.0)]
        for status in (429, 503):
            for header, delay in cases:
                with self.subTest(status=status, header=header):
                    transport = Mock(side_effect=[
                        urllib.error.HTTPError("https://llm.example/v1", status, "busy",
                                               {"Retry-After": header}, io.BytesIO(b"busy")),
                        FakeHTTPResponse({"choices": [{"message": {"content": "{}"}}]}),
                    ])
                    sleeper = Mock()
                    provider = OpenAICompatibleProvider(
                        provider_name="test", base_url="https://llm.example/v1",
                        api_key_env="TEST_LLM_KEY", model="test", urlopen=transport,
                        sleep=sleeper, wall_clock=lambda: 0.0)
                    with patch.dict(os.environ, {"TEST_LLM_KEY": "secret"}):
                        response = provider.complete_json(request())
                    sleeper.assert_called_once_with(delay)
                    self.assertEqual(response.metadata["transport_attempts"], 2)

    def test_retry_after_above_cap_stops_without_retrying_early(self) -> None:
        """A bounded client must stop if it cannot honor the server deadline."""
        for header in ("61", "Thu, 01 Jan 1970 00:01:01 GMT"):
            with self.subTest(header=header):
                transport = Mock(side_effect=urllib.error.HTTPError(
                    "https://llm.example/v1", 429, "busy", {"Retry-After": header},
                    io.BytesIO(b"busy")))
                sleeper = Mock()
                provider = OpenAICompatibleProvider(
                    provider_name="test", base_url="https://llm.example/v1",
                    api_key_env="TEST_LLM_KEY", model="test", urlopen=transport,
                    sleep=sleeper, wall_clock=lambda: 0.0)
                with patch.dict(os.environ, {"TEST_LLM_KEY": "secret"}):
                    with self.assertRaisesRegex(ProviderTransportError,
                                                "Retry-After requires 61s.*stopped"):
                        provider.complete_json(request())
                self.assertEqual(transport.call_count, 1)
                sleeper.assert_not_called()

    def test_http_protocol_errors_retry_including_incomplete_response_reads(self) -> None:
        """HTTPException subclasses from connection setup or reading are transient."""
        incomplete = FakeHTTPResponse({})
        incomplete.read = Mock(side_effect=http.client.IncompleteRead(b"partial"))
        for result in (http.client.BadStatusLine("broken"), incomplete):
            with self.subTest(result=result):
                transport = Mock(side_effect=[result, FakeHTTPResponse(
                    {"choices": [{"message": {"content": "{}"}}]})])
                sleeper = Mock()
                provider = OpenAICompatibleProvider(
                    provider_name="test", base_url="https://llm.example/v1",
                    api_key_env="TEST_LLM_KEY", model="test", urlopen=transport,
                    sleep=sleeper)
                with patch.dict(os.environ, {"TEST_LLM_KEY": "secret"}):
                    response = provider.complete_json(request())
                self.assertEqual(response.metadata["transport_attempts"], 2)
                sleeper.assert_called_once_with(5.0)

    def test_incomplete_http_error_body_preserves_status_and_retry_after(self) -> None:
        """A broken error body cannot bypass known authentication or retry rules."""
        for status, error_type in ((401, ProviderAuthenticationError), (429, None)):
            with self.subTest(status=status):
                error = urllib.error.HTTPError("https://llm.example/v1", status, "busy",
                                               {"Retry-After": "8"}, io.BytesIO())
                error.read = Mock(side_effect=http.client.IncompleteRead(b"partial"))
                transport = Mock(side_effect=[error, FakeHTTPResponse(
                    {"choices": [{"message": {"content": "{}"}}]})])
                sleeper = Mock()
                provider = OpenAICompatibleProvider(
                    provider_name="test", base_url="https://llm.example/v1",
                    api_key_env="TEST_LLM_KEY", model="test", urlopen=transport,
                    sleep=sleeper)
                with patch.dict(os.environ, {"TEST_LLM_KEY": "secret"}):
                    if error_type is not None:
                        with self.assertRaises(error_type):
                            provider.complete_json(request())
                        sleeper.assert_not_called()
                        self.assertEqual(transport.call_count, 1)
                    else:
                        response = provider.complete_json(request())
                        self.assertEqual(response.metadata["transport_attempts"], 2)
                        sleeper.assert_called_once_with(8.0)

    def test_custom_budget_and_zero_retry_setting_bound_network_attempts(self) -> None:
        """Configured retries and caps constrain requests without real sleeping."""
        for retries, delays in ((0, []), (3, [3.0, 4.0, 4.0])):
            with self.subTest(retries=retries):
                transport = Mock(side_effect=TimeoutError("outage"))
                sleeper = Mock()
                provider = OpenAICompatibleProvider(
                    provider_name="test", base_url="https://llm.example/v1",
                    api_key_env="TEST_LLM_KEY", model="test", urlopen=transport,
                    sleep=sleeper, max_transport_retries=retries,
                    retry_backoff_s=3.0, retry_backoff_cap_s=4.0)
                with patch.dict(os.environ, {"TEST_LLM_KEY": "secret"}):
                    with self.assertRaises(ProviderTransportError):
                        provider.complete_json(request())
                self.assertEqual(transport.call_count, retries + 1)
                self.assertEqual(sleeper.call_args_list, [call(delay) for delay in delays])

    def test_nonretryable_http_error_stops_and_malformed_envelope_is_repairable(self) -> None:
        """Bad configuration is fatal; a received malformed response is content failure."""
        cases = [
            (urllib.error.HTTPError("https://llm.example/v1", 400, "bad request", {}, io.BytesIO(b"invalid model")), ProviderTransportError),
            (FakeHTTPResponse({"choices": []}), ProviderResponseError),
        ]
        for result, error_type in cases:
            with self.subTest(error_type=error_type):
                transport = Mock(side_effect=result) if isinstance(result, Exception) else Mock(return_value=result)
                sleeper = Mock()
                provider = OpenAICompatibleProvider(
                    provider_name="test", base_url="https://llm.example/v1", api_key_env="TEST_LLM_KEY",
                    model="test", urlopen=transport, sleep=sleeper)
                with patch.dict(os.environ, {"TEST_LLM_KEY": "secret"}), self.assertRaises(error_type):
                    provider.complete_json(request())
                self.assertEqual(transport.call_count, 1)
                sleeper.assert_not_called()

    def test_provider_factory_passes_configured_openrouter_settings(self) -> None:
        """The NAS loop can remain unaware of provider-specific constructors."""
        provider = build_provider(
            Dict(
                provider="openrouter",
                base_url="https://openrouter.ai/api/v1",
                api_key_env="OPENROUTER_API_KEY",
                model="openai/test-model",
                temperature=0.4,
                timeout_s=60.0,
                json_response_mode=True,
                extra_headers=Dict(),
                max_transport_retries=7,
                retry_backoff_s=8.0,
                retry_backoff_cap_s=90.0,
            )
        )

        self.assertIsInstance(provider, OpenAICompatibleProvider)
        self.assertEqual(provider.provider_name, "openrouter")
        self.assertEqual(provider.model, "openai/test-model")
        self.assertTrue(provider.json_response_mode)
        self.assertEqual(provider.max_transport_retries, 7)
        self.assertEqual(provider.retry_backoff_s, 8.0)
        self.assertEqual(provider.retry_backoff_cap_s, 90.0)


if __name__ == "__main__":
    unittest.main()
