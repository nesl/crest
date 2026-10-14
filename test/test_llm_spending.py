# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Persistent conservative provider spend-accounting regression tests."""

import json
import tempfile
from decimal import Decimal, localcontext
import unittest
from pathlib import Path
from unittest.mock import patch

from crest.optimizers.llm.provider import ProviderError, SpendLimitExceeded
from crest.optimizers.llm.spending import (
    SpendAccountingError,
    SpendInputLimitExceeded,
    SpendTracker,
)


def configuration(**updates):
    """Return an inexpensive provider configuration with exact decimal prices."""
    return {
        "max_usd": "0.01",
        "input_usd_per_million_tokens": "1",
        "output_usd_per_million_tokens": "2",
        "pricing_source": "provider price sheet, retrieved 2026-10-13",
        "max_input_tokens": 2048,
        "max_output_tokens": 100,
        "model": "test-model",
        "provider": "openai_compatible",
        "base_url": "https://example.invalid/v1",
        "output_token_parameter": "max_completion_tokens",
        **updates,
    }


PAYLOAD = {"messages": [{"role": "user", "content": "hello"}]}


class SpendTrackerTests(unittest.TestCase):
    """Check paid attempts, conservative uncertainty, and safe restart behavior."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def read_state(self):
        return json.loads((self.root / "spend_state.json").read_text())

    def test_service_tier_binds_accounting_without_changing_legacy_binding(self):
        """Resuming spend state requires the same explicit billing tier."""
        legacy_root = self.root / "legacy"
        SpendTracker(legacy_root, configuration())
        legacy_state = json.loads((legacy_root / "spend_state.json").read_text())
        self.assertNotIn("service_tier", legacy_state["binding"])
        # Omitting the option preserves older accounting files and their binding.
        SpendTracker(legacy_root, configuration())
        SpendTracker(self.root, configuration(service_tier="default"))
        self.assertEqual(self.read_state()["binding"]["service_tier"], "default")
        SpendTracker(self.root, configuration(service_tier="default"))
        for changed in (configuration(service_tier="priority"), configuration()):
            with self.subTest(config=changed), self.assertRaises(SpendAccountingError):
                SpendTracker(self.root, changed)

    def test_reservation_is_full_cost_and_persisted_before_return(self):
        tracker = SpendTracker(self.root, configuration())
        reservation = tracker.reserve(PAYLOAD)
        state = self.read_state()
        self.assertEqual(state["charged_usd"], "0.002248")
        self.assertEqual(state["reservations"][reservation]["status"], "reserved")
        self.assertFalse(list(self.root.glob(".spend-*.tmp")))

    def test_known_usage_releases_unused_cost_and_settlement_is_idempotent(self):
        tracker = SpendTracker(self.root, configuration())
        reservation = tracker.reserve(PAYLOAD)
        accounting = tracker.settle(reservation, {"prompt_tokens": 10, "completion_tokens": 5})
        self.assertEqual(accounting["charged_usd"], "0.00002")
        self.assertEqual(accounting["total_charged_usd"], "0.00002")
        self.assertTrue(accounting["usage_known"])
        self.assertEqual(tracker.settle(reservation, None), accounting)
        self.assertEqual(self.read_state()["charged_usd"], "0.00002")

    def test_unknown_and_malformed_usage_keeps_full_charge(self):
        cases = [None, {}, {"prompt_tokens": 1}, {"prompt_tokens": True, "completion_tokens": 0},
                 {"prompt_tokens": 1, "completion_tokens": -1},
                 {"prompt_tokens": 1.0, "completion_tokens": 1},
                 {"prompt_tokens": 1, "completion_tokens": "1"}, []]
        for index, usage in enumerate(cases):
            with self.subTest(usage=usage):
                tracker = SpendTracker(self.root / str(index), configuration())
                accounting = tracker.settle(tracker.reserve(PAYLOAD), usage)
                self.assertEqual(accounting["status"], "unknown")
                self.assertEqual(accounting["charged_usd"], "0.002248")
                self.assertFalse(accounting["usage_known"])

    def test_crash_and_separate_transport_attempts_retain_each_reservation(self):
        config = configuration(max_usd="0.004496")
        first = SpendTracker(self.root, config).reserve(PAYLOAD)
        resumed = SpendTracker(self.root, config)
        second = resumed.reserve(PAYLOAD)
        self.assertNotEqual(first, second)
        self.assertEqual(self.read_state()["charged_usd"], "0.004496")
        with self.assertRaises(SpendLimitExceeded):
            resumed.reserve(PAYLOAD)
        self.assertEqual(len(self.read_state()["reservations"]), 2)

    def test_exact_cap_and_zero_cap_use_decimal_arithmetic(self):
        tracker = SpendTracker(self.root, configuration(max_usd="0.002248"))
        tracker.reserve(PAYLOAD)
        with self.assertRaises(SpendLimitExceeded):
            tracker.reserve(PAYLOAD)
        zero = SpendTracker(self.root / "zero", configuration(max_usd=0))
        with self.assertRaises(SpendLimitExceeded):
            zero.reserve(PAYLOAD)

    def test_high_precision_prices_do_not_round_accumulated_charges(self):
        config = configuration(max_usd="1", input_usd_per_million_tokens="0.123456789012345678901234567890123456789")
        tracker = SpendTracker(self.root, config)
        first, second = tracker.reserve(PAYLOAD), tracker.reserve(PAYLOAD)
        with localcontext() as context:
            context.prec = 100
            reserved = (Decimal(config["input_usd_per_million_tokens"]) * 2048 + 200) / 1_000_000
            actual = (Decimal(config["input_usd_per_million_tokens"]) * 3 + 4) / 1_000_000
            self.assertEqual(Decimal(self.read_state()["charged_usd"]), reserved * 2)
            tracker.settle(first, {"prompt_tokens": 3, "completion_tokens": 2})
            self.assertEqual(Decimal(self.read_state()["charged_usd"]), reserved + actual)
        resumed = SpendTracker(self.root, config)
        resumed.settle(second, {"prompt_tokens": 0, "completion_tokens": 0})
        self.assertEqual(Decimal(self.read_state()["charged_usd"]), actual)

    def test_input_guard_counts_unicode_and_framing_before_reserve(self):
        messages = [{"role": "user", "content": "café 😀"}]
        bound = len(json.dumps(messages, separators=(",", ":")).encode("utf-8")) + 1024
        tracker = SpendTracker(self.root, configuration(max_input_tokens=bound))
        tracker.reserve({"messages": messages})
        too_small = SpendTracker(self.root / "small", configuration(max_input_tokens=bound - 1))
        with self.assertRaises(SpendInputLimitExceeded):
            too_small.reserve({"messages": messages})
        self.assertEqual(json.loads(too_small.path.read_text())["reservations"], {})

    def test_malformed_request_and_unknown_reservation_are_typed_errors(self):
        tracker = SpendTracker(self.root, configuration())
        for payload in ({}, {"messages": None}, {"messages": [float("nan")]}):
            with self.assertRaises(SpendAccountingError):
                tracker.reserve(payload)
        with self.assertRaises(SpendAccountingError):
            tracker.settle("missing", None)
        self.assertEqual(self.read_state()["charged_usd"], "0")
        for error in (SpendAccountingError, SpendInputLimitExceeded, SpendLimitExceeded):
            self.assertTrue(issubclass(error, ProviderError))

    def test_cap_may_change_only_above_already_charged_cost(self):
        tracker = SpendTracker(self.root, configuration())
        tracker.reserve(PAYLOAD)
        SpendTracker(self.root, configuration(max_usd="0.002248"))
        with self.assertRaises(SpendLimitExceeded):
            SpendTracker(self.root, configuration(max_usd="0.002247"))
        larger = SpendTracker(self.root, configuration(max_usd="0.02"))
        larger.reserve(PAYLOAD)
        self.assertEqual(self.read_state()["max_usd"], "0.02")

    def test_binding_refuses_model_prices_limits_provider_and_provenance_changes(self):
        SpendTracker(self.root, configuration())
        changed = {"model": "another", "provider": "openrouter", "base_url": "https://other.invalid/v1",
                   "pricing_source": "new sheet", "input_usd_per_million_tokens": "1.1",
                   "output_usd_per_million_tokens": "2.1", "max_input_tokens": 4096,
                   "max_output_tokens": 101, "output_token_parameter": "max_tokens"}
        for name, value in changed.items():
            with self.subTest(field=name), self.assertRaises(SpendAccountingError):
                SpendTracker(self.root, configuration(**{name: value}))
        SpendTracker(self.root, configuration(input_usd_per_million_tokens="1.000",
                                             base_url="https://example.invalid/v1/"))

    def test_token_overrun_charges_actual_cost_before_permanently_halting(self):
        tracker = SpendTracker(self.root, configuration(max_usd="0.002248"))
        reservation = tracker.reserve(PAYLOAD)
        with self.assertRaises(SpendAccountingError):
            tracker.settle(reservation, {"prompt_tokens": 3000, "completion_tokens": 200})
        state = self.read_state()
        self.assertEqual(state["charged_usd"], "0.0034")
        self.assertTrue(state["halted"])
        for instance in (tracker, SpendTracker(self.root, configuration(max_usd="1"))):
            with self.assertRaises(SpendAccountingError):
                instance.reserve(PAYLOAD)
            with self.assertRaises(SpendAccountingError):
                instance.settle(reservation, None)

    def test_input_overrun_stops_even_when_total_cost_is_below_reservation(self):
        tracker = SpendTracker(self.root, configuration())
        with self.assertRaises(SpendAccountingError):
            tracker.settle(tracker.reserve(PAYLOAD), {"prompt_tokens": 2049, "completion_tokens": 0})
        self.assertTrue(self.read_state()["halted"])
        self.assertEqual(self.read_state()["charged_usd"], "0.002049")

    def test_failed_atomic_replace_does_not_release_or_reserve_cost(self):
        tracker = SpendTracker(self.root, configuration())
        with patch("crest.optimizers.llm.spending.os.replace", side_effect=OSError("disk failed")):
            with self.assertRaises(SpendAccountingError):
                tracker.reserve(PAYLOAD)
        self.assertEqual(self.read_state()["charged_usd"], "0")
        self.assertEqual(tracker.reserve(PAYLOAD), "1")
        with patch("crest.optimizers.llm.spending.os.replace", side_effect=OSError("disk failed")):
            with self.assertRaises(SpendAccountingError):
                tracker.settle("1", {"prompt_tokens": 0, "completion_tokens": 0})
        self.assertEqual(self.read_state()["charged_usd"], "0.002248")

    def test_corrupt_or_inconsistent_state_is_never_reset(self):
        tracker = SpendTracker(self.root, configuration())
        tracker.reserve(PAYLOAD)
        state = self.read_state()
        state["charged_usd"] = "0"
        tracker.path.write_text(json.dumps(state))
        with self.assertRaises(SpendAccountingError):
            SpendTracker(self.root, configuration())
        tracker.path.write_text("not-json")
        with self.assertRaises(SpendAccountingError):
            SpendTracker(self.root, configuration())
        self.assertEqual(tracker.path.read_text(), "not-json")

    def test_missing_spend_state_with_old_paid_requests_refuses_reset(self):
        for location in ("requests", "memory/requests"):
            with self.subTest(location=location):
                root = self.root / location.replace("/", "-")
                requests = root / location
                requests.mkdir(parents=True)
                (requests / "000001.request.json").write_text("{}")
                with self.assertRaises(SpendAccountingError):
                    SpendTracker(root, configuration())
                self.assertFalse((root / "spend_state.json").exists())

    def test_invalid_settings_fail_before_creating_accounting(self):
        cases = [{"max_usd": "NaN"}, {"max_usd": -1}, {"max_usd": True},
                 {"input_usd_per_million_tokens": "Infinity"}, {"output_usd_per_million_tokens": -1},
                 {"input_usd_per_million_tokens": 0}, {"output_usd_per_million_tokens": 0},
                 {"max_input_tokens": 0}, {"max_output_tokens": True}, {"max_output_tokens": 1.5},
                 {"pricing_source": " "}, {"provider": None}]
        for update in cases:
            with self.subTest(update=update), self.assertRaises(SpendAccountingError):
                SpendTracker(self.root, configuration(**update))
        self.assertFalse((self.root / "spend_state.json").exists())


if __name__ == "__main__":
    unittest.main()
