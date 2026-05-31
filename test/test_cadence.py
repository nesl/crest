"""Tests for model-agnostic cadence resolution helpers."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from addict import Dict

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crest.cadence import resolve_batch_period_ms  # noqa: E402


class CadenceResolutionTests(unittest.TestCase):
    """Validate logical cadence resolution across config shapes."""

    def test_device_override_wins_over_dataset_sources(self) -> None:
        """Device latency-budget overrides should take highest precedence.

        Returns
        -------
        None
            Asserts the device override wins over metadata and config cadence.
        """
        result = resolve_batch_period_ms(
            Dict(batch_period_ms=2000, stride=20, sampling_rate_hz=100),
            dataset_metadata={"batch_period_ms": 1000},
            device_config=SimpleNamespace(latency_budget_ms=75),
        )

        self.assertEqual(result, 75.0)

    def test_metadata_batch_period_wins_over_config_sources(self) -> None:
        """Dataset metadata batch period should win over config cadence fields.

        Returns
        -------
        None
            Asserts loaded dataset metadata takes precedence over config fields.
        """
        result = resolve_batch_period_ms(
            Dict(batch_period_ms=2000, stride=20, sampling_rate_hz=100),
            dataset_metadata={"batch_period_ms": 1500},
            device_config=SimpleNamespace(latency_budget_ms=None),
        )

        self.assertEqual(result, 1500.0)

    def test_config_batch_period_wins_over_legacy_cadence(self) -> None:
        """Config batch period should win over legacy cadence.

        Returns
        -------
        None
            Asserts `dataset.params.batch_period_ms` wins over stride cadence.
        """
        result = resolve_batch_period_ms(
            SimpleNamespace(batch_period_ms=2000, stride=20, sampling_rate_hz=100),
            dataset_metadata={},
            device_config=SimpleNamespace(latency_budget_ms=None),
        )

        self.assertEqual(result, 2000.0)

    def test_legacy_odometry_cadence_still_derives_from_stride(self) -> None:
        """Legacy odometry configs should still derive cadence from stride.

        Returns
        -------
        None
            Asserts the odometry cadence formula remains unchanged.
        """
        result = resolve_batch_period_ms(
            Dict(stride=20, sampling_rate_hz=100),
            dataset_metadata={},
            device_config=SimpleNamespace(latency_budget_ms=None),
        )

        self.assertEqual(result, 200.0)

    def test_legacy_stride_can_come_from_metadata_and_rate_from_config(self) -> None:
        """Legacy stride may come from metadata while rate comes from config.

        Returns
        -------
        None
            Asserts mixed-source fallback is resolved independently per field.
        """
        result = resolve_batch_period_ms(
            Dict(sampling_rate_hz=100),
            dataset_metadata={"stride": 25},
            device_config=SimpleNamespace(latency_budget_ms=None),
        )

        self.assertEqual(result, 250.0)

    def test_legacy_stride_can_come_from_config_and_rate_from_metadata(self) -> None:
        """Legacy stride may come from config while rate comes from metadata.

        Returns
        -------
        None
            Asserts mixed-source fallback is resolved independently per field.
        """
        result = resolve_batch_period_ms(
            Dict(stride=25),
            dataset_metadata={"sampling_rate_hz": 50},
            device_config=SimpleNamespace(latency_budget_ms=None),
        )

        self.assertEqual(result, 500.0)

    def test_absent_batch_period_falls_through_to_legacy_fields(self) -> None:
        """Missing batch period should fall through to legacy cadence fields.

        Returns
        -------
        None
            Asserts absent fields do not block lower-precedence cadence.
        """
        result = resolve_batch_period_ms(
            Dict(stride=10, sampling_rate_hz=100),
            dataset_metadata={},
            device_config=SimpleNamespace(latency_budget_ms=None),
        )

        self.assertEqual(result, 100.0)

    def test_present_null_batch_period_fails_without_falling_through(self) -> None:
        """Explicit null batch periods should fail instead of falling through.

        Returns
        -------
        None
            Asserts present-invalid batch cadence blocks legacy fallback.
        """
        with self.assertRaisesRegex(ValueError, "dataset.params.batch_period_ms"):
            resolve_batch_period_ms(
                Dict(batch_period_ms=None, stride=10, sampling_rate_hz=100),
                dataset_metadata={},
                device_config=SimpleNamespace(latency_budget_ms=None),
            )

    def test_bool_device_override_fails_without_falling_through(self) -> None:
        """Boolean device overrides should fail before dataset fallback.

        Returns
        -------
        None
            Asserts invalid device overrides are not ignored.
        """
        with self.assertRaisesRegex(ValueError, "device.latency_budget_ms"):
            resolve_batch_period_ms(
                Dict(batch_period_ms=2000),
                dataset_metadata={},
                device_config=SimpleNamespace(latency_budget_ms=True),
            )

    def test_negative_metadata_batch_period_fails_without_falling_through(self) -> None:
        """Negative metadata batch periods should fail before config fallback.

        Returns
        -------
        None
            Asserts invalid metadata-owned cadence is not ignored.
        """
        with self.assertRaisesRegex(ValueError, "dataset.metadata.batch_period_ms"):
            resolve_batch_period_ms(
                Dict(batch_period_ms=2000),
                dataset_metadata={"batch_period_ms": -1},
                device_config=SimpleNamespace(latency_budget_ms=None),
            )

    def test_nonfinite_config_batch_period_fails_without_falling_through(self) -> None:
        """Non-finite config batch periods should fail before legacy fallback.

        Returns
        -------
        None
            Asserts invalid config-owned batch cadence is not ignored.
        """
        with self.assertRaisesRegex(ValueError, "dataset.params.batch_period_ms"):
            resolve_batch_period_ms(
                Dict(batch_period_ms=math.inf, stride=10, sampling_rate_hz=100),
                dataset_metadata={},
                device_config=SimpleNamespace(latency_budget_ms=None),
            )

    def test_zero_legacy_stride_fails(self) -> None:
        """Zero legacy stride should fail as an invalid cadence field.

        Returns
        -------
        None
            Asserts legacy cadence fields must be positive.
        """
        with self.assertRaisesRegex(ValueError, "dataset.params.stride"):
            resolve_batch_period_ms(
                Dict(stride=0, sampling_rate_hz=100),
                dataset_metadata={},
                device_config=SimpleNamespace(latency_budget_ms=None),
            )

    def test_missing_all_cadence_fields_fails_clearly(self) -> None:
        """Configs without batch or legacy cadence fields should fail clearly.

        Returns
        -------
        None
            Asserts a fully missing cadence contract raises ``ValueError``.
        """
        with self.assertRaisesRegex(ValueError, "stride"):
            resolve_batch_period_ms(
                Dict(),
                dataset_metadata={},
                device_config=SimpleNamespace(latency_budget_ms=None),
            )


if __name__ == "__main__":
    unittest.main()
