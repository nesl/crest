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


stm32_cadenced_comparison = _load_module(
    "stm32_cadenced_comparison_for_tests",
    "analysis_scripts/stm32_example_project/run_stm32_cadenced_comparison.py",
)
stm32_cpu_clock_sweep = _load_module(
    "stm32_cpu_clock_sweep_for_tests",
    "analysis_scripts/stm32_example_project/run_stm32_cpu_clock_sweep.py",
)


class Stm32RunnerWrapperTests(unittest.TestCase):
    def test_cadenced_comparison_raises_with_child_output_when_metrics_missing(self):
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
            verbose=False,
            clean=False,
        )

        def _fake_run(cmd, cwd, capture_output, text, check):
            del cmd, cwd, capture_output, text, check
            return subprocess.CompletedProcess([], 1, "child stdout", "child stderr")

        with patch.object(stm32_cadenced_comparison.subprocess, "run", side_effect=_fake_run):
            with self.assertRaises(RuntimeError) as ctx:
                stm32_cadenced_comparison._run_phase_attempt(args, "cadenced", 1)

        self.assertIn("producing metrics artifacts", str(ctx.exception))
        self.assertIn("child stderr", str(ctx.exception))

    def test_cadenced_comparison_returns_metrics_when_child_writes_artifacts(self):
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
            verbose=False,
            clean=False,
        )

        def _fake_run(cmd, cwd, capture_output, text, check):
            del cwd, capture_output, text, check
            output_path = Path(cmd[cmd.index("--output") + 1])
            output_path.write_text(json.dumps({"error_code": 1}), encoding="utf-8")
            output_path.with_name("metrics.diagnostics.json").write_text(
                json.dumps({"ok": True}), encoding="utf-8"
            )
            return subprocess.CompletedProcess(cmd, 1, "", "")

        with patch.object(stm32_cadenced_comparison.subprocess, "run", side_effect=_fake_run):
            metrics, diagnostic = stm32_cadenced_comparison._run_phase_attempt(args, "cadenced", 1)

        self.assertEqual(metrics["error_code"], 1)
        self.assertTrue(diagnostic["ok"])

    def test_cpu_clock_sweep_scales_cadenced_child_timeout(self):
        args = Namespace(
            python_executable=Path("/usr/bin/python3"),
            project_root=Path("/tmp/project"),
            config=Path("/tmp/config.yaml"),
            stage_output_root=Path("/tmp/stage"),
            latency_budget_ms=2000.0,
            measured_runs=20,
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

        self.assertIn("--serial-timeout", cmd)
        self.assertEqual(cmd[cmd.index("--serial-timeout") + 1], "50.0")

    def test_cpu_clock_sweep_preserves_larger_requested_timeout(self):
        args = Namespace(
            python_executable=Path("/usr/bin/python3"),
            project_root=Path("/tmp/project"),
            config=Path("/tmp/config.yaml"),
            stage_output_root=Path("/tmp/stage"),
            latency_budget_ms=200.0,
            measured_runs=10,
            wake_margin_us=5000,
            min_sleep_us=5000,
            dut_port="/dev/ttyACM0",
            harness_port="/dev/ttyACM1",
            baud=115200,
            serial_timeout=120.0,
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

        self.assertEqual(cmd[cmd.index("--serial-timeout") + 1], "120.0")

    def test_cpu_clock_sweep_reuses_staged_model_on_later_repeats(self):
        class _DummyTqdm:
            def __init__(self, *args, **kwargs):
                del args, kwargs

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                del exc_type, exc, tb
                return False

            def update(self, n):
                del n

            def set_postfix(self, **kwargs):
                del kwargs

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            project_root = tmp_path / "project"
            project_root.mkdir()
            config_path = tmp_path / "config.yaml"
            config_path.write_text("training: {}\n", encoding="utf-8")
            weights_memory_pool = tmp_path / "mypool.json"
            weights_memory_pool.write_text("{}\n", encoding="utf-8")
            results_root = tmp_path / "results"
            stage_output_root = tmp_path / "stage"

            captured_reuse_flags: list[tuple[str, bool]] = []

            def _fake_run(cmd, cwd, capture_output, text, check):
                del cwd, capture_output, text, check
                output_path = Path(cmd[cmd.index("--output") + 1])
                output_path.write_text(json.dumps({"error_code": 1}), encoding="utf-8")
                mode = cmd[cmd.index("--weight-storage-mode") + 1]
                captured_reuse_flags.append((mode, "--reuse-staged-model" in cmd))
                return subprocess.CompletedProcess(cmd, 0, "", "")

            argv = [
                "run_stm32_cpu_clock_sweep.py",
                "--project-root",
                str(project_root),
                "--config",
                str(config_path),
                "--results-root",
                str(results_root),
                "--stage-output-root",
                str(stage_output_root),
                "--python-executable",
                sys.executable,
                "--weights-memory-pool",
                str(weights_memory_pool),
                "--frequencies",
                "200",
                "--phases",
                "back_to_back",
                "--repeats",
                "3",
            ]

            with (
                patch.object(sys, "argv", argv),
                patch.object(stm32_cpu_clock_sweep.subprocess, "run", side_effect=_fake_run),
                patch.object(stm32_cpu_clock_sweep, "tqdm", _DummyTqdm),
            ):
                exit_code = stm32_cpu_clock_sweep.main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            captured_reuse_flags,
            [
                ("embedded", False),
                ("embedded", True),
                ("embedded", True),
                ("external_flash", False),
                ("external_flash", True),
                ("external_flash", True),
            ],
        )


if __name__ == "__main__":
    unittest.main()
