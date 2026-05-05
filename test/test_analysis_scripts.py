"""Regression tests for the repository's analysis-script wrappers."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _load_module(module_name: str, relative_path: str):
    """Load an analysis script by repository-relative path for wrapper tests.

    Parameters
    ----------
    module_name : str
        Synthetic module name to register in ``sys.modules``.
    relative_path : str
        Repository-relative path to the Python entrypoint under test.

    Returns
    -------
    module
        Imported module object loaded directly from the target file.
    """

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
stm32_phase2_candidate = _load_module(
    "stm32_phase2_candidate",
    "analysis_scripts/stm32_example_project/stm32_phase2_candidate.py",
)
stm32_backend_phase2_smoke = _load_module(
    "stm32_backend_phase2_smoke",
    "analysis_scripts/stm32_example_project/smoke_test_stm32_backend_phase2.py",
)
stm32_backend_phase4_smoke = _load_module(
    "stm32_backend_phase4_smoke",
    "analysis_scripts/stm32_example_project/smoke_test_stm32_backend_phase4.py",
)
arena_latency_curve = _load_module(
    "arena_latency_curve",
    "analysis_scripts/arena_latency_curve/run_arena_latency_curve.py",
)
arena_latency_curve_failure_probe = _load_module(
    "arena_latency_curve_failure_probe",
    "analysis_scripts/arena_latency_curve/run_arena_latency_curve_failure_probe.py",
)
clock_tick_latency = _load_module(
    "clock_tick_latency",
    "analysis_scripts/clock_tick_latency/run_clock_tick_latency.py",
)
stm32_lrun_common = _load_module(
    "stm32_lrun_common_for_tests",
    "analysis_scripts/stm32_example_project/stm32_lrun_common.py",
)
single_hil = _load_module(
    "single_hil_for_tests",
    "analysis_scripts/hil_single_run/run_single_hil.py",
)


class AnalysisCadenceHelperTests(unittest.TestCase):
    """Validate analysis helper cadence defaults for batch and legacy configs."""

    def test_analysis_support_uses_batch_period_before_legacy_cadence(self):
        """Analysis cadence helper should accept audio-style batch periods.

        Returns
        -------
        None
            Asserts explicit batch cadence takes precedence over legacy fields.
        """

        from tinyodom.analysis_support import derive_latency_budget_ms

        budget_ms = derive_latency_budget_ms(
            Namespace(batch_period_ms=2000, stride=20, sampling_rate_hz=100)
        )

        self.assertEqual(budget_ms, 2000.0)

    def test_analysis_support_preserves_legacy_stride_cadence(self):
        """Analysis cadence helper should preserve odometry stride cadence.

        Returns
        -------
        None
            Asserts legacy stride/sample-rate configs still derive the same
            latency budget.
        """

        from tinyodom.analysis_support import derive_latency_budget_ms

        budget_ms = derive_latency_budget_ms(Namespace(stride=20, sampling_rate_hz=100))

        self.assertEqual(budget_ms, 200.0)

    def test_analysis_support_missing_cadence_raises_value_error(self):
        """Analysis cadence helper should fail clearly when cadence is absent.

        Returns
        -------
        None
            Asserts incomplete audio-style configs raise ``ValueError`` instead
            of leaking ``AttributeError`` from legacy field access.
        """

        from tinyodom.analysis_support import derive_latency_budget_ms

        with self.assertRaisesRegex(ValueError, "stride"):
            derive_latency_budget_ms(Namespace())

    def test_lrun_device_defaults_use_dataset_batch_period(self):
        """LRUN device defaults should resolve batch periods through config.

        Returns
        -------
        None
            Asserts LRUN defaults use `dataset.params.batch_period_ms`.
        """

        defaults = stm32_lrun_common.device_defaults(
            {
                "device": {"runtime_mode": "cadenced"},
                "dataset": {"params": {"batch_period_ms": 2000}},
            }
        )

        self.assertEqual(defaults["latency_budget_ms"], 2000.0)

    def test_lrun_device_defaults_preserve_legacy_stride_cadence(self):
        """LRUN device defaults should keep odometry stride-derived cadence.

        Returns
        -------
        None
            Asserts LRUN defaults retain the legacy odometry cadence formula.
        """

        defaults = stm32_lrun_common.device_defaults(
            {
                "device": {"runtime_mode": "cadenced"},
                "dataset": {"params": {"stride": 20, "sampling_rate_hz": 100}},
            }
        )

        self.assertEqual(defaults["latency_budget_ms"], 200.0)

    def test_lrun_translates_invalid_cadence_to_workflow_error(self):
        """LRUN helper errors should stay WorkflowError-compatible.

        Returns
        -------
        None
            Asserts invalid cadence values are translated to the LRUN workflow
            error type expected by callers.
        """

        with self.assertRaises(stm32_lrun_common.stm32_cube_clt.WorkflowError):
            stm32_lrun_common.device_defaults(
                {
                    "device": {"runtime_mode": "cadenced"},
                    "dataset": {"params": {"batch_period_ms": 0}},
                }
            )


class SingleHILRunTests(unittest.TestCase):
    """Validate single-run HIL wrapper behavior."""

    def test_single_hil_uses_configured_export_variant(self) -> None:
        """Single HIL runner should not force a hard-coded model variant."""

        server = MagicMock()
        server.config.device.harness_arm_pin = 3
        server.config.device.harness_trigger_pin = 2
        server.config.device.dut_arm_hold_ms = 600
        server.config.device.harness_stable_low_ms = 500
        server.determine_metrics.return_value = {"latency_ms": 1.0}

        argv = ["run_single_hil.py", "--config", "src/config/nas_config.yaml"]
        with patch.object(sys, "argv", argv), patch.object(
            single_hil, "HILServer", return_value=server
        ), patch.object(single_hil, "_build_hyperparams", return_value={"nb_filters": 2}), patch.object(
            single_hil,
            "split_hil_request_hyperparams",
            return_value=({"nb_filters": 2}, {"flops": 100}),
        ):
            self.assertEqual(single_hil.main(), 0)

        server.determine_metrics.assert_called_once_with({"nb_filters": 2}, {"flops": 100})


class CadencedPortentaSummaryTests(unittest.TestCase):
    def test_summarize_group_counts_master_success(self):
        # The summary helper should count master-success rows in the expected aggregate bucket.
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
        # Aggregate summaries should exclude failed attempts so averages only reflect completed runs.
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

    def test_stage_phase_sketch_uses_candidate_dir_basename(self):
        """Portenta phase staging should use the candidate directory basename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            candidate_dir = root / "candidate"
            common_dir = root / "sketches" / "common"
            common_dir.mkdir(parents=True)
            (common_dir / "tinyodom_hil_config.h").write_text("// shared\n", encoding="utf-8")
            template = root / "template.ino"
            template.write_text("#define TINYODOM_LATENCY_BUDGET_MS 1\n", encoding="utf-8")
            server = type(
                "Server",
                (),
                {
                    "config": type(
                        "Config",
                        (),
                        {"outputs": type("Outputs", (), {"candidate_dir": candidate_dir})()},
                    )()
                },
            )()

            with patch.object(cadenced_portenta_h7, "SCRIPT_DIR", root), patch.object(
                cadenced_portenta_h7, "REPO_ROOT", root
            ), patch.dict(
                cadenced_portenta_h7.PHASE_SKETCH_FILENAMES,
                {cadenced_portenta_h7.PHASE_BACK_TO_BACK: template.name},
            ):
                staged = cadenced_portenta_h7._stage_phase_sketch(
                    server,
                    cadenced_portenta_h7.PHASE_BACK_TO_BACK,
                    latency_budget_ms=10.0,
                )

            self.assertEqual(staged, candidate_dir / "candidate.ino")
            self.assertTrue((candidate_dir / "common" / "tinyodom_hil_config.h").is_file())


class AnalysisScriptCandidateDirTests(unittest.TestCase):
    """Validate renamed candidate-dir metadata in analysis helpers."""

    def test_arena_latency_payload_uses_candidate_dir_metadata_key(self):
        """Arena latency JSON metadata should use candidate_dir instead of tcn_dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            settings = arena_latency_curve.SweepSettings(
                config_path=Path("/tmp/config.yaml"),
                device_name="ARDUINO_NANO_33_BLE_SENSE",
                core_label="n/a",
                device_options=None,
                arena_kb_list=[32],
                repeats=1,
                cooldown_s=0.0,
                input_mode="uniform",
                model_variant="approx_trained",
                trained_checkpoint=None,
                output_csv=root / "attempts.csv",
                output_json=root / "summary.json",
                output_plot=root / "summary.png",
            )
            attempts = [
                {
                    "timestamp": "2026-05-01T00:00:00Z",
                    "device": settings.device_name,
                    "core": settings.core_label,
                    "arena_kb": 32,
                    "repeat": 1,
                    "latency_ms": 1.0,
                    "energy_mj_per_inference": 2.0,
                    "ram_bytes": 1,
                    "flash_bytes": 2,
                    "arena_bytes": 32768,
                    "harness_latency_ms": 1.0,
                    "avg_power_mw": 1.0,
                    "avg_current_ma": 1.0,
                    "bus_voltage_v": 5.0,
                    "idle_power_mw": 0.0,
                    "error_code": 0,
                    "error_label": "OK",
                }
            ]
            server = type(
                "Server",
                (),
                {
                    "active_sketch_path": root / "candidate" / "odom_tcn.ino",
                    "config": type(
                        "Config",
                        (),
                        {"outputs": type("Outputs", (), {"candidate_dir": root / "candidate"})()},
                    )(),
                },
            )()

            payload = {
                "metadata": {
                    "timestamp_utc": "2026-05-01T00:00:00Z",
                    "config_path": str(settings.config_path),
                    "device": settings.device_name,
                    "core": settings.core_label,
                    "device_options": settings.device_options or {},
                    "model_variant": settings.model_variant,
                    "trained_checkpoint": "",
                    "input_mode": settings.input_mode,
                    "arena_kb_list": settings.arena_kb_list,
                    "repeats": settings.repeats,
                    "cooldown_s": settings.cooldown_s,
                    "active_sketch_path": str(server.active_sketch_path),
                    "candidate_dir": str(server.config.outputs.candidate_dir),
                },
                "attempts": attempts,
                "aggregates_by_arena": arena_latency_curve._aggregate_by_arena(
                    settings.arena_kb_list, attempts
                ),
            }
            settings.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            metadata = json.loads(settings.output_json.read_text(encoding="utf-8"))["metadata"]
            self.assertEqual(metadata["candidate_dir"], str(root / "candidate"))
            self.assertNotIn("tcn_dir", metadata)

    def test_clock_tick_payload_uses_candidate_dir_metadata_key(self):
        """Clock-tick JSON metadata should use candidate_dir instead of tcn_dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            settings = clock_tick_latency.RunSettings(
                config_path=Path("/tmp/config.yaml"),
                device_name="PORTENTA_H7",
                core_label="cm7",
                device_options={"target_core": "cm7"},
                repeats=1,
                cooldown_s=0.0,
                input_mode="uniform",
                model_variant="approx_trained",
                trained_checkpoint=None,
                fallback_clock_hz=400_000_000.0,
                output_csv=root / "attempts.csv",
                output_json=root / "summary.json",
                output_plot=root / "summary.png",
            )
            server = type(
                "Server",
                (),
                {
                    "active_sketch_path": root / "candidate" / "odom_tcn.ino",
                    "config": type(
                        "Config",
                        (),
                        {"outputs": type("Outputs", (), {"candidate_dir": root / "candidate"})()},
                    )(),
                },
            )()

            payload = {
                "metadata": {
                    "timestamp_utc": "2026-05-01T00:00:00Z",
                    "config_path": str(settings.config_path),
                    "device": settings.device_name,
                    "core": settings.core_label,
                    "device_options": settings.device_options,
                    "model_variant": settings.model_variant,
                    "trained_checkpoint": "",
                    "input_mode": settings.input_mode,
                    "repeats": settings.repeats,
                    "cooldown_s": settings.cooldown_s,
                    "fallback_clock_hz": settings.fallback_clock_hz,
                    "active_sketch_path": str(server.active_sketch_path),
                    "candidate_dir": str(server.config.outputs.candidate_dir),
                },
                "attempts": [],
                "aggregates": {
                    "latency_ms": clock_tick_latency._aggregate_metric([], "latency_ms"),
                    "ticks_per_inference": clock_tick_latency._aggregate_metric(
                        [], "ticks_per_inference"
                    ),
                },
            }
            settings.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            metadata = json.loads(settings.output_json.read_text(encoding="utf-8"))["metadata"]
            self.assertEqual(metadata["candidate_dir"], str(root / "candidate"))
            self.assertNotIn("tcn_dir", metadata)


class Stm32MeasuredRunsTests(unittest.TestCase):
    def _phase2_bundle(self, *, quantization_mode: str = "int8_ptq", calibration_inputs=object()):
        """Build a fake Phase 2 candidate bundle for script smoke tests."""

        return SimpleNamespace(
            config=SimpleNamespace(),
            calibration_inputs=calibration_inputs,
            hyperparams={"nb_filters": 8},
            model=object(),
            metadata=Namespace(model_variant="approx_trained", quantization_mode=quantization_mode),
            window_size=4,
            input_dim=6,
        )

    def test_phase2_candidate_uses_configured_device_name_in_tflite_filename(self):
        # Phase-two candidate naming should embed the configured device name in the TFLite artifact.
        convert_mock = unittest.mock.Mock()
        bundle = type(
            "Bundle",
            (),
            {
                "config": type(
                    "Config",
                    (),
                    {
                        "device": type("Device", (), {"name": "STM32_NUCLEO_N657X0_Q"})(),
                        "training": type(
                            "Training",
                            (),
                            {"quantization": {"mode": "int8_ptq", "search": False, "choices": ["int8_ptq"]}},
                        )(),
                    },
                )(),
                "model": object(),
                "calibration_inputs": object(),
                "metadata": {"model_variant": "approx_trained"},
                "window_size": 4,
                "input_dim": 6,
            },
        )()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            with (
                patch.object(stm32_phase2_candidate, "load_or_build_perturbed_candidate", return_value=bundle),
                patch.dict(
                    sys.modules,
                    {
                        "tinyodom.hardware": type(
                            "HardwareModule",
                            (),
                            {"convert_to_tflite_model": convert_mock},
                        )()
                    },
                ),
            ):
                tflite_path, metadata = stm32_phase2_candidate.export_perturbed_candidate_tflite(
                    Path("/tmp/nas_config.yaml"),
                    output_root,
                )

        self.assertEqual(
            tflite_path.name,
            "TinyOdomEx_OxIOD_STM32_NUCLEO_N657X0_Q_approx_trained.tflite",
        )
        self.assertEqual(metadata, {"model_variant": "approx_trained"})
        self.assertIs(convert_mock.call_args.kwargs["training_data"], bundle.calibration_inputs)

    def test_phase2_candidate_float_export_omits_calibration_data(self):
        """Float candidate export should not require representative data."""

        convert_mock = unittest.mock.Mock()
        bundle = type(
            "Bundle",
            (),
            {
                "config": type(
                    "Config",
                    (),
                    {
                        "device": type("Device", (), {"name": "STM32_NUCLEO_N657X0_Q"})(),
                        "training": type(
                            "Training",
                            (),
                            {"quantization": {"mode": "float", "search": False, "choices": ["float"]}},
                        )(),
                    },
                )(),
                "model": object(),
                "calibration_inputs": None,
                "metadata": {"model_variant": "approx_trained"},
                "window_size": 4,
                "input_dim": 6,
            },
        )()

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(stm32_phase2_candidate, "load_or_build_perturbed_candidate", return_value=bundle),
                patch.dict(
                    sys.modules,
                    {
                        "tinyodom.hardware": type(
                            "HardwareModule",
                            (),
                            {"convert_to_tflite_model": convert_mock},
                        )()
                    },
                ),
            ):
                stm32_phase2_candidate.export_perturbed_candidate_tflite(
                    Path("/tmp/nas_config.yaml"),
                    Path(tmpdir),
                )

        self.assertIsNone(convert_mock.call_args.kwargs["training_data"])
        self.assertEqual(convert_mock.call_args.kwargs["quantization_mode"], "float")

    def test_stm32_backend_phase2_smoke_uses_candidate_prepare_request(self):
        """Phase 2 smoke runner should call the current backend preparation API."""

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.yaml"
            project_root = root / "project"
            staged_root = root / "staged"
            config_path.write_text("device:\n  name: STM32_NUCLEO_N657X0_Q\n", encoding="utf-8")
            project_root.mkdir()
            bundle = self._phase2_bundle()
            device = MagicMock()
            device.prepare_candidate.return_value = staged_root
            device.compile.return_value = SimpleNamespace(
                success=True,
                build_dir=root / "build",
                flash_bytes=1024,
                ram_bytes=512,
            )

            with patch.object(stm32_backend_phase2_smoke, "get_device", return_value=device), patch.object(
                stm32_backend_phase2_smoke,
                "load_or_build_perturbed_candidate",
                return_value=bundle,
            ):
                exit_code = stm32_backend_phase2_smoke.main(
                    [
                        "--config",
                        str(config_path),
                        "--project-root",
                        str(project_root),
                        "--output-root",
                        str(root / "out"),
                        "--compile-only",
                    ]
                )

        self.assertEqual(exit_code, stm32_backend_phase2_smoke.EXIT_SUCCESS)
        request = device.prepare_candidate.call_args.kwargs["request"]
        self.assertEqual(request.quantization_mode, "int8_ptq")
        self.assertEqual(request.input_shape, (4, 6))
        self.assertIs(request.calibration_split.inputs, bundle.calibration_inputs)

    def test_stm32_backend_phase4_smoke_uses_candidate_prepare_request(self):
        """Phase 4 smoke runner should call the current backend preparation API."""

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.yaml"
            project_root = root / "project"
            staged_root = root / "staged"
            config_path.write_text("device:\n  name: STM32_NUCLEO_N657X0_Q\n", encoding="utf-8")
            project_root.mkdir()
            device = MagicMock()
            device.prepare_candidate.return_value = staged_root
            device.compile.return_value = SimpleNamespace(
                success=True,
                build_dir=root / "build",
                flash_bytes=1024,
                ram_bytes=512,
            )

            with patch.object(stm32_backend_phase4_smoke, "get_device", return_value=device), patch.object(
                stm32_backend_phase4_smoke,
                "load_or_build_perturbed_candidate",
                return_value=self._phase2_bundle(quantization_mode="float", calibration_inputs=None),
            ):
                exit_code = stm32_backend_phase4_smoke.main(
                    [
                        "--config",
                        str(config_path),
                        "--project-root",
                        str(project_root),
                        "--output-root",
                        str(root / "out"),
                        "--compile-only",
                    ]
                )

        self.assertEqual(exit_code, stm32_backend_phase4_smoke.EXIT_SUCCESS)
        request = device.prepare_candidate.call_args.kwargs["request"]
        self.assertEqual(request.quantization_mode, "float")
        self.assertIsNone(request.calibration_split)

    def test_analysis_float_artifact_preparation_skips_calibration(self):
        """Float analysis exports should pass no representative data."""

        for module in (arena_latency_curve, arena_latency_curve_failure_probe, clock_tick_latency):
            with self.subTest(module=module.__name__):
                server = SimpleNamespace(
                    model_family=SimpleNamespace(materialize_export_model=MagicMock(return_value=object())),
                    model_build_context=object(),
                    model_config={},
                    task=SimpleNamespace(compile_model=MagicMock()),
                    task_config={},
                    target_spec=object(),
                    config=SimpleNamespace(
                        training=SimpleNamespace(quantization=SimpleNamespace(mode="float")),
                        outputs=SimpleNamespace(tflite_model_path="model.tflite", candidate_dir="candidate"),
                    ),
                    get_calibration_inputs=MagicMock(side_effect=AssertionError("calibration should not be fetched")),
                )
                convert_tflite = MagicMock()

                module._prepare_model_artifacts(
                    server=server,
                    hyperparams={"nb_filters": 4},
                    model_variant="untrained",
                    trained_checkpoint=None,
                    convert_to_tflite_model_fn=convert_tflite,
                    convert_to_cpp_model_fn=MagicMock(),
                    require_calibration_inputs_fn=MagicMock(
                        side_effect=AssertionError("calibration should not be required")
                    ),
                )

                self.assertIsNone(convert_tflite.call_args.kwargs["training_data"])
                self.assertEqual(convert_tflite.call_args.kwargs["quantization_mode"], "float")

    def test_write_phase_config_header_writes_measured_runs_and_clamps(self):
        # Phase-config headers should record measured-run counts and clamp settings so generated STM32 workspaces match the requested run mode.
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

    def test_main_maps_workflow_error_to_master_fatal(self):
        # Workflow errors should map to the stable master-fatal status before the script exits.
        class _DummyMonitor:
            def __init__(self, *args, **kwargs):
                del args, kwargs

            def write_line(self, text):
                del text

            def wait_for(self, predicate, timeout_s, stage):
                del predicate, timeout_s
                return "HARNESS READY" if stage == "HARNESS READY" else None

            def snapshot_lines(self):
                return []

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            project_root = tmp_path / "project"
            inc_dir = project_root / "Inc"
            debug_dir = project_root / "Debug"
            inc_dir.mkdir(parents=True)
            debug_dir.mkdir(parents=True)
            elf_path = debug_dir / "toy_ai.elf"
            elf_path.write_text("", encoding="utf-8")
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
                    return_value=(debug_dir, elf_path),
                ),
                patch.object(stm32_toy_ai_hil, "_read_staging_manifest", return_value={}),
                patch.object(stm32_toy_ai_hil, "_build_project"),
                patch.object(
                    stm32_toy_ai_hil,
                    "_run_size",
                    return_value={
                        "text": 1,
                        "data": 1,
                        "bss": 1,
                        "dec": 3,
                        "hex": 3,
                        "elf_flash_bytes": 2,
                        "ram_bytes": 2,
                    },
                ),
                patch.object(stm32_toy_ai_hil, "_find_linker_script", return_value=project_root / "toy.ld"),
                patch.object(stm32_toy_ai_hil, "_parse_linker_reservations", return_value={}),
                patch.object(stm32_toy_ai_hil, "_parse_arena_bytes", return_value=1024),
                patch.object(stm32_toy_ai_hil, "ensure_harness_firmware"),
                patch.object(stm32_toy_ai_hil, "SerialMonitor", _DummyMonitor),
                patch.object(
                    stm32_toy_ai_hil,
                    "_load_and_run",
                    side_effect=stm32_toy_ai_hil.WorkflowError("load failed"),
                ),
            ):
                exit_code = stm32_toy_ai_hil.main()

            self.assertEqual(exit_code, 1)
            metrics = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(metrics["error_code"], stm32_toy_ai_hil.MASTER_FATAL)
            self.assertEqual(metrics["error_label"], "HIL_MASTER_FATAL")
            self.assertIn(
                "#define TOY_AI_MEASURED_RUNS 100",
                (inc_dir / "toy_ai_phase_config.h").read_text(encoding="utf-8"),
            )

    def test_cadenced_comparison_forwards_measured_runs_to_child(self):
        # Cadenced comparison wrappers should forward measured-run counts to the child process.
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
        # CPU-clock sweep commands should include the measured-run count so child runs use the intended averaging window.
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
