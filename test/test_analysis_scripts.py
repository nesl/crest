from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch


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
stm32_toy_ai_hil = _load_module(
    "stm32_toy_ai_hil",
    "analysis_scripts/stm32_example_project/run_stm32_toy_ai_hil.py",
)
stm32_cadenced_comparison = _load_module(
    "stm32_cadenced_comparison",
    "analysis_scripts/stm32_example_project/run_stm32_cadenced_comparison.py",
)
stm32_cpu_clock_sweep = _load_module(
    "stm32_cpu_clock_sweep",
    "analysis_scripts/stm32_example_project/run_stm32_cpu_clock_sweep.py",
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


class Stm32MeasuredRunsTests(unittest.TestCase):
    def test_write_phase_config_header_writes_measured_runs_and_clamps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            (project_root / "Inc").mkdir()

            header_path, changed = stm32_toy_ai_hil._write_phase_config_header(
                project_root=project_root,
                phase="cadenced",
                latency_budget_ms=200.0,
                measured_runs=100,
                cpu_clock_mhz=600,
                wake_margin_us=5000,
                min_sleep_us=5000,
            )

            self.assertTrue(changed)
            self.assertIn("#define TOY_AI_MEASURED_RUNS 100", header_path.read_text(encoding="utf-8"))

            header_path, changed = stm32_toy_ai_hil._write_phase_config_header(
                project_root=project_root,
                phase="cadenced",
                latency_budget_ms=200.0,
                measured_runs=0,
                cpu_clock_mhz=600,
                wake_margin_us=5000,
                min_sleep_us=5000,
            )

            self.assertTrue(changed)
            self.assertIn("#define TOY_AI_MEASURED_RUNS 1", header_path.read_text(encoding="utf-8"))

    def test_main_rejects_no_build_when_measured_runs_change_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            project_root = tmp_path / "project"
            inc_dir = project_root / "Inc"
            inc_dir.mkdir(parents=True)
            (inc_dir / "toy_ai_phase_config.h").write_text(
                "#ifndef TOY_AI_PHASE_CONFIG_H\n"
                "#define TOY_AI_PHASE_CONFIG_H\n\n"
                "#define TOY_AI_PHASE_BACK_TO_BACK 0\n"
                "#define TOY_AI_PHASE_CADENCED 1\n\n"
                "#define TOY_AI_SELECTED_PHASE TOY_AI_PHASE_CADENCED\n"
                "#define TOY_AI_LATENCY_BUDGET_MS 200\n"
                "#define TOY_AI_MEASURED_RUNS 10\n"
                "#define TOY_AI_CPU_CLOCK_MHZ 600\n"
                "#define TOY_AI_WAKE_MARGIN_US 5000\n"
                "#define TOY_AI_MIN_SLEEP_US 5000\n\n"
                "#endif /* TOY_AI_PHASE_CONFIG_H */\n",
                encoding="utf-8",
            )
            config_path = tmp_path / "nas_config.yaml"
            config_path.write_text("training:\n  nas_trials: 1\n  max_total_trials: 2\n", encoding="utf-8")
            output_path = tmp_path / "metrics.json"
            stage_output_root = tmp_path / "stage"
            stage_output_root.mkdir()

            argv = [
                "run_stm32_toy_ai_hil.py",
                "--project-root",
                str(project_root),
                "--config",
                str(config_path),
                "--output",
                str(output_path),
                "--stage-output-root",
                str(stage_output_root),
                "--reuse-staged-model",
                "--no-build",
                "--measured-runs",
                "100",
            ]

            with (
                patch.object(sys, "argv", argv),
                patch.object(stm32_toy_ai_hil, "_configure_logging"),
                patch.object(stm32_toy_ai_hil, "_resolve_required_tool_path", return_value=tmp_path / "tool"),
                patch.object(
                    stm32_toy_ai_hil,
                    "_validate_paths",
                    return_value=(project_root / "Debug", project_root / "Debug" / "toy_ai.elf"),
                ),
                patch.object(stm32_toy_ai_hil, "_read_staging_manifest", return_value={}),
            ):
                with self.assertRaises(stm32_toy_ai_hil.WorkflowError) as ctx:
                    stm32_toy_ai_hil.main()

            self.assertIn("--no-build cannot be used", str(ctx.exception))
            self.assertIn(
                "#define TOY_AI_MEASURED_RUNS 100",
                (inc_dir / "toy_ai_phase_config.h").read_text(encoding="utf-8"),
            )

    def test_cadenced_comparison_forwards_measured_runs_to_child(self):
        captured_cmd: list[str] = []

        def _fake_run(cmd, cwd, capture_output, text, check):
            del cwd, capture_output, text, check
            captured_cmd[:] = cmd
            output_path = Path(cmd[cmd.index("--output") + 1])
            output_path.write_text(json.dumps({"error_code": 1}), encoding="utf-8")
            output_path.with_name("metrics.diagnostics.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        args = Namespace(
            project_root=Path("/tmp/project"),
            config=Path("/tmp/config.yaml"),
            latency_budget_ms=200.0,
            measured_runs=100,
            cpu_clock_mhz=600,
            wake_margin_us=5000,
            min_sleep_us=5000,
            dut_port="/dev/ttyACM0",
            harness_port="/dev/ttyACM1",
            baud=115200,
            dut_ready_timeout=30.0,
            harness_ready_timeout=30.0,
            harness_done_timeout=10.0,
            harness_fqbn="arduino:mbed_nano:nano33ble",
            harness_auto_flash="once",
            harness_arm_pin=3,
            harness_trigger_pin=2,
            dut_arm_hold_ms=600,
            harness_stable_low_ms=50,
            weight_storage_mode="embedded",
            weights_flash_address="0x71000000",
            weights_memory_pool=Path("/tmp/nucleo_mypool.json"),
            gdb_port=61234,
            apid=0,
            server_ready_timeout=30.0,
            weights_external_loader=None,
            gdbserver=None,
            gdb=None,
            cubeprog_bin=None,
            jobs=None,
            reuse_staged_model=False,
            no_build=False,
            verbose=False,
            clean=False,
        )

        with patch.object(stm32_cadenced_comparison.subprocess, "run", side_effect=_fake_run):
            metrics, diagnostic = stm32_cadenced_comparison._run_phase_attempt(args, "cadenced", 1)

        self.assertEqual(metrics["error_code"], 1)
        self.assertTrue(diagnostic["ok"])
        self.assertIn("--measured-runs", captured_cmd)
        self.assertEqual(captured_cmd[captured_cmd.index("--measured-runs") + 1], "100")

    def test_cpu_clock_sweep_child_command_includes_measured_runs(self):
        args = Namespace(
            python_executable=Path("/usr/bin/python3"),
            project_root=Path("/tmp/project"),
            config=Path("/tmp/config.yaml"),
            stage_output_root=Path("/tmp/stage"),
            latency_budget_ms=200.0,
            measured_runs=100,
            wake_margin_us=5000,
            min_sleep_us=5000,
            dut_port="/dev/ttyACM0",
            harness_port="/dev/ttyACM1",
            baud=115200,
            serial_timeout=30.0,
            dut_ready_timeout=30.0,
            harness_ready_timeout=30.0,
            harness_done_timeout=10.0,
            harness_fqbn="arduino:mbed_nano:nano33ble",
            harness_auto_flash="once",
            harness_arm_pin=3,
            harness_trigger_pin=2,
            dut_arm_hold_ms=600,
            harness_stable_low_ms=50,
            weights_flash_address="0x71000000",
            weights_memory_pool=Path("/tmp/nucleo_mypool.json"),
            gdb_port=61234,
            apid=0,
            server_ready_timeout=30.0,
            weights_external_loader=None,
            gdbserver=None,
            gdb=None,
            cubeprog_bin=None,
            jobs=None,
            reuse_staged_model=False,
            no_build=False,
            verbose=False,
        )

        cmd = stm32_cpu_clock_sweep._build_child_command(
            args=args,
            scenario=stm32_cpu_clock_sweep.DEFAULT_SCENARIOS[1],
            cpu_clock_mhz=400,
            output_path=Path("/tmp/metrics.json"),
            reuse_staged_model=False,
            clean=False,
        )

        self.assertIn("--measured-runs", cmd)
        self.assertEqual(cmd[cmd.index("--measured-runs") + 1], "100")


if __name__ == "__main__":
    unittest.main()
