"""Offline checks of count-request fidelity and budget accounting."""
import json
import unittest
from unittest.mock import patch

import estimate


class CountingTests(unittest.TestCase):
    def test_ledger_conversion_preserves_exact_messages(self):
        messages = [{"role": "system", "content": "Return JSON."},
                    {"role": "user", "content": '{"recent_trials":[]}'}]
        payload, source = estimate.prepare_request({"messages": messages, "request_hash": "x"}, "test-model")
        self.assertEqual(payload, {"model": "test-model", "input": messages})
        self.assertIn("chat", source)

    def test_native_request_preserves_options(self):
        original = {"input": "Hello", "text": {"format": {"type": "json_object"}}}
        result, source = estimate.prepare_request(original, "test-model")
        self.assertEqual(result["text"], original["text"])
        self.assertEqual(source, "native_responses_request")

    def test_no_silent_dropping_of_chat_options(self):
        with self.assertRaises(ValueError):
            estimate.prepare_request({"messages": [{"role": "user", "content": "Hi"}], "tools": []}, "test")

    def test_count_endpoint_only_and_response_validation(self):
        captured = {}
        class Reply:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def read(self):
                return b'{"input_tokens": 123}'
        def transport(request, timeout):
            captured["url"] = request.full_url
            captured["body"] = json.loads(request.data)
            return Reply()
        result = estimate.count_input_tokens({"model": "test", "input": "Hi"}, "test-secret", opener=transport)
        self.assertEqual(result, 123)
        self.assertEqual(captured["url"], "https://api.openai.com/v1/responses/input_tokens")
        self.assertEqual(captured["body"], {"model": "test", "input": "Hi"})

    def test_missing_key_never_sends_request(self):
        with patch("estimate.urllib.request.urlopen") as transport:
            with self.assertRaises(ValueError):
                estimate.count_input_tokens({}, None)
            transport.assert_not_called()

    def test_cost_includes_retries_and_all_billed_output(self):
        result = estimate.project([1000, 3000], trials=150, batch_size=5,
                                  extra_calls_per_round=.1, repair_extra_input_tokens=100,
                                  output_tokens_per_call=500, input_price=2, output_price=8)
        self.assertEqual(result["generation_rounds"], 30)
        self.assertEqual(result["expected_api_calls"], 33)
        middle = result["scenarios"][1]
        self.assertEqual(middle["projected_input_tokens"], 66300)
        self.assertEqual(middle["projected_output_tokens"], 16500)
        self.assertAlmostEqual(middle["total_cost_usd"], .2646)

    def test_partial_batches_acceptance_and_unknown_output(self):
        result = estimate.project([1000], trials=11, batch_size=5, acceptance_rate=.5)
        self.assertEqual(result["generation_rounds"], 5)
        self.assertIsNone(result["scenarios"][0]["projected_output_tokens"])
        self.assertIsNone(result["scenarios"][0]["total_cost_usd"])

    def test_invalid_scenarios_fail_before_network(self):
        for bad in [0, float("nan"), float("inf"), -1, 1.1]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                estimate.project([1000], trials=150, batch_size=5, acceptance_rate=bad)


if __name__ == "__main__":
    unittest.main()
