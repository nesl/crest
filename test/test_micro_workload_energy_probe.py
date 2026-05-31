"""Tests for the synthetic micro-workload energy probe helpers."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


def _load_probe_module():
    """Load the probe runner module from its script path.

    Returns
    -------
    module
        Imported probe runner module used by the focused unit tests.

    Raises
    ------
    RuntimeError
        If existing validation or execution checks fail.
    """
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "analysis_scripts" / "micro_workload_energy_probe" / "run_micro_workload_energy_probe.py"
    spec = importlib.util.spec_from_file_location("micro_workload_energy_probe_for_tests", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe = _load_probe_module()


class MicroWorkloadEnergyProbeTests(unittest.TestCase):
    """Validate workload schema, DUT telemetry defaults, and baseline math."""

    def test_poll_workload_is_supported_in_order(self):
        """Check workload ordering and firmware mode mapping.

        Returns
        -------
        None
            The test passes when workload constants match the firmware ABI.
        """
        self.assertEqual(probe.VALID_WORKLOADS, ("sleep", "wait", "poll", "float", "int"))
        self.assertEqual(probe.WORKLOAD_MODE, {"sleep": 0, "wait": 1, "poll": 2, "float": 3, "int": 4})

    def test_base_attempt_defaults_missing_dut_telemetry_to_minus_one(self):
        """Check default attempt sentinels for missing DUT telemetry.

        Returns
        -------
        None
            The test passes when telemetry defaults are explicit sentinel
            values rather than blanks.
        """
        attempt = probe.base_attempt(
            SimpleNamespace(window_ms=1000),
            probe.BoardSpec(token="ble", family="arduino", fqbn="arduino:mbed_nano:nano33ble"),
            "poll",
            1,
        )

        self.assertEqual(attempt["dut_iterations"], -1)
        self.assertEqual(attempt["dut_work_units"], -1)
        self.assertEqual(attempt["dut_elapsed_us"], -1)
        self.assertEqual(attempt["dut_cycles"], -1)
        self.assertEqual(attempt["dut_sleep_ms"], -1.0)
        self.assertEqual(attempt["dut_work_unit_label"], "")
        self.assertEqual(attempt["dut_sleep_mode"], "")

    def test_parse_dut_telemetry_accepts_prefixed_logs(self):
        """Check DUT telemetry parsing from stored diagnostic log lines.

        Returns
        -------
        None
            The test passes when optional stream prefixes do not affect parsed
            telemetry values.
        """
        telemetry = probe.parse_dut_telemetry(
            [
                "DUT: dut iterations output: 1024",
                "DUT: dut work units output: 8192",
                "DUT: dut work unit label output: fp_ops",
                "DUT: dut elapsed us output: 1001000",
                "DUT: dut cycles output: 600600000",
                "DUT: dut sleep ms output: 0.000",
                "DUT: dut sleep mode output: none",
                "DUT: micro workload run: ok",
            ]
        )

        self.assertEqual(telemetry["dut_iterations"], 1024)
        self.assertEqual(telemetry["dut_work_units"], 8192)
        self.assertEqual(telemetry["dut_work_unit_label"], "fp_ops")
        self.assertEqual(telemetry["dut_elapsed_us"], 1001000)
        self.assertEqual(telemetry["dut_cycles"], 600600000)
        self.assertEqual(telemetry["dut_sleep_ms"], 0.0)
        self.assertEqual(telemetry["dut_sleep_mode"], "none")

    def test_direct_serial_missing_telemetry_becomes_error(self):
        """Check direct-serial telemetry validation failure handling.

        Returns
        -------
        None
            The test passes when missing direct DUT telemetry marks the attempt
            as a parse failure.
        """
        attempt = {
            "workload": "float",
            "error_code": 0,
            "dut_iterations": -1,
            "dut_work_units": -1,
            "dut_elapsed_us": -1,
            "dut_work_unit_label": "",
        }

        probe.validate_direct_dut_telemetry(attempt, direct_serial=True)

        self.assertEqual(attempt["error_code"], 7)
        self.assertEqual(attempt["error_label"], "telemetry_parse_failed")

    def test_implausible_harness_window_becomes_error(self):
        """Check validation of the harness-measured trigger window.

        Returns
        -------
        None
            The test passes when an implausible trigger-high duration marks the
            attempt invalid.
        """
        attempt = {
            "error_code": 0,
            "error_label": "ok",
            "measured_harness_window_ms": 32.0,
        }

        probe.validate_harness_window_duration(attempt, 1000)

        self.assertEqual(attempt["error_code"], 11)
        self.assertEqual(attempt["error_label"], "window_duration_invalid")

    def test_aggregate_baselines_use_board_local_sleep_and_wait(self):
        """Check aggregate subtraction against board-local baselines.

        Returns
        -------
        None
            The test passes when phase and payload diagnostic fields use the
            expected local baseline rows.
        """
        attempts = [
            {"board": "ble", "workload": "sleep", "error_code": 0, "energy_mj_per_window": 100.0, "avg_power_mw": 100.0, "measured_harness_window_ms": 1000.0, "dut_work_units": 0},
            {"board": "ble", "workload": "wait", "error_code": 0, "energy_mj_per_window": 115.0, "avg_power_mw": 115.0, "measured_harness_window_ms": 1000.0, "dut_work_units": 1000},
            {"board": "ble", "workload": "poll", "error_code": 0, "energy_mj_per_window": 125.0, "avg_power_mw": 125.0, "measured_harness_window_ms": 1000.0, "dut_work_units": 5000},
            {"board": "ble", "workload": "float", "error_code": 0, "energy_mj_per_window": 150.0, "avg_power_mw": 150.0, "measured_harness_window_ms": 1000.0, "dut_work_units": 10000},
        ]

        rows = {(row["board"], row["workload"]): row for row in probe.summarize_attempts(attempts)}
        float_row = rows[("ble", "float")]

        self.assertEqual(float_row["energy_over_sleep_mj_mean"], 50.0)
        self.assertEqual(float_row["energy_over_poll_mj_mean"], 25.0)
        self.assertEqual(float_row["energy_over_wait_mj_mean"], 35.0)
        self.assertEqual(float_row["energy_per_work_unit_nj_mean"], 5000.0)
        self.assertEqual(float_row["payload_energy_per_work_unit_nj_mean"], 3500.0)

    def test_poll_above_compute_emits_warning_without_blocking_wait_payload(self):
        """Check poll diagnostic warnings without changing wait subtraction.

        Returns
        -------
        None
            The test passes when a high polling phase emits a warning while
            payload energy still uses the wait baseline.
        """
        attempts = [
            {"board": "m7", "workload": "sleep", "error_code": 0, "energy_mj_per_window": 100.0, "avg_power_mw": 100.0, "measured_harness_window_ms": 1000.0, "dut_work_units": 0},
            {"board": "m7", "workload": "wait", "error_code": 0, "energy_mj_per_window": 110.0, "avg_power_mw": 110.0, "measured_harness_window_ms": 1000.0, "dut_work_units": 1000},
            {"board": "m7", "workload": "poll", "error_code": 0, "energy_mj_per_window": 160.0, "avg_power_mw": 160.0, "measured_harness_window_ms": 1000.0, "dut_work_units": 5000},
            {"board": "m7", "workload": "float", "error_code": 0, "energy_mj_per_window": 140.0, "avg_power_mw": 140.0, "measured_harness_window_ms": 1000.0, "dut_work_units": 10000},
        ]

        rows = {(row["board"], row["workload"]): row for row in probe.summarize_attempts(attempts)}

        self.assertEqual(rows[("m7", "float")]["energy_over_poll_mj_mean"], -20.0)
        self.assertEqual(rows[("m7", "float")]["payload_energy_per_work_unit_nj_mean"], 3000.0)
        self.assertEqual(rows[("m7", "float")]["aggregate_warning"], "poll_exceeds_float")

    def test_stm32_telemetry_avoids_long_long_printf_format(self):
        """Check STM32 telemetry printing avoids unsupported printf formats.

        Returns
        -------
        None
            The test passes when 64-bit counters use the custom decimal
            printer instead of ``%llu``.
        """
        repo_root = Path(__file__).resolve().parents[1]
        runner_source = repo_root / "analysis_scripts" / "micro_workload_energy_probe" / "stm32_synthetic_dut_runner.c"
        source = runner_source.read_text()

        self.assertNotIn("%llu", source)
        self.assertIn("print_u64_output(\"dut iterations output: \", workload_telemetry.iterations);", source)
        self.assertIn("print_u64_output(\"dut work units output: \", workload_telemetry.work_units);", source)
        self.assertIn("print_u64_output(\"dut elapsed us output: \", elapsed_us);", source)


if __name__ == "__main__":
    unittest.main()
