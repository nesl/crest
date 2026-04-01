from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


def _load_module(module_name: str, relative_path: str):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


cadenced_portenta_h7 = _load_module(
    "cadenced_portenta_h7",
    "analysis_scripts/cadenced_portenta_h7/run_cadenced_portenta_h7.py",
)


class CadencedPortentaSummaryTests(unittest.TestCase):
    def test_summarize_group_counts_master_success(self):
        summary = cadenced_portenta_h7._summarize_group(
            core="cm7",
            phase=cadenced_portenta_h7.PHASE_BACK_TO_BACK,
            attempts=[
                {
                    "error_code": cadenced_portenta_h7.MASTER_SUCCESS_CODE,
                    "latency_ms": 95.0,
                    "harness_latency_ms": 96.0,
                    "energy_mj_per_inference": 4.0,
                },
                {
                    "error_code": 3,
                    "latency_ms": 70.0,
                    "harness_latency_ms": 71.0,
                    "energy_mj_per_inference": 3.0,
                },
            ],
            latency_budget_ms=100.0,
        )

        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(summary["failure_count"], 1)

    def test_summarize_group_excludes_failed_attempts_from_aggregates(self):
        summary = cadenced_portenta_h7._summarize_group(
            core="cm4",
            phase=cadenced_portenta_h7.PHASE_CADENCED,
            attempts=[
                {
                    "error_code": cadenced_portenta_h7.MASTER_SUCCESS_CODE,
                    "latency_ms": 125.0,
                    "harness_latency_ms": 126.0,
                    "energy_mj_per_inference": 8.0,
                    "avg_power_mw": 32.0,
                    "avg_current_ma": 6.0,
                    "bus_voltage_v": 5.0,
                    "idle_power_mw": 2.0,
                },
                {
                    "error_code": 3,
                    "latency_ms": 999.0,
                    "harness_latency_ms": 999.0,
                    "energy_mj_per_inference": 999.0,
                    "avg_power_mw": 999.0,
                    "avg_current_ma": 999.0,
                    "bus_voltage_v": 999.0,
                    "idle_power_mw": 999.0,
                },
            ],
            latency_budget_ms=100.0,
        )

        self.assertEqual(summary["over_budget_count"], 1)
        self.assertEqual(summary["latency_ms_n"], 1)
        self.assertAlmostEqual(summary["latency_ms_mean"], 125.0)
        self.assertAlmostEqual(summary["energy_mj_per_inference_mean"], 8.0)
        self.assertAlmostEqual(summary["avg_power_mw_mean"], 32.0)


if __name__ == "__main__":
    unittest.main()
