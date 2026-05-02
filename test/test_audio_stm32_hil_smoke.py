"""Tests for the STM32 audio HIL smoke utility."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from addict import Dict

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
SCRIPT_DIR = ROOT_DIR / "analysis_scripts" / "audio_stm32_hil_smoke"
for path in (SRC_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_audio_stm32_hil_smoke as smoke  # noqa: E402
from tinyodom.pipeline_types import DataSplit, ModelBuildContext, TargetSpec  # noqa: E402


class AudioSTM32HILSmokeTests(unittest.TestCase):
    """Validate the Phase 6 STM32 audio smoke runner."""

    def _config(self) -> Dict:
        """Build a minimal mutable audio STM32 config."""

        return Dict(
            config_path="config.yaml",
            outputs=Dict(
                candidate_dir="audio_dscnn",
                tflite_model_path="unused.tflite",
                checkpoint_path="checkpoint.keras",
            ),
            device=Dict(
                name="STM32_NUCLEO_N657X0_Q",
                hil=False,
                serial_port="/dev/ttyACM0",
                runtime_mode="back_to_back",
                measured_inference_runs=10,
                stm32=Dict(project_root="sketches/stm32/tinyodom_tcn_stm32_lrun"),
            ),
            training=Dict(energy_aware=False, quantization=True),
            model=Dict(params=Dict(export_variant="untrained")),
        )

    def _pipeline(self, *, input_shape: tuple[int, int] = (201, 64), calibration_rows: int = 2) -> SimpleNamespace:
        """Build a fake bootstrapped audio pipeline."""

        calibration = DataSplit(
            inputs=np.zeros((calibration_rows, *input_shape), dtype=np.float32),
            targets=np.arange(calibration_rows, dtype=np.int64),
            metadata={},
        )
        model = MagicMock(name="keras_model")
        model_family = MagicMock()
        model_family.default_seed_trial.return_value = {"base_channels": 16}
        model_family.decode_trial_hparams.return_value = {"base_channels": 16}
        model_family.materialize_export_model.return_value = model
        model_family.count_flops.return_value = 12345
        task = MagicMock()
        target_spec = TargetSpec(
            task_type="classification",
            output_names=["class_logits"],
            output_shapes=[(None, 10)],
            metadata={"num_classes": 10, "label_encoding": "class_index"},
        )
        return SimpleNamespace(
            selection={
                "model_family_name": "audio_dscnn",
                "task_name": "sound_classification",
                "model_config": Dict(family="audio_dscnn", params=Dict(export_variant="untrained"), search=Dict()),
            },
            model_build_context=ModelBuildContext(
                input_shape=input_shape,
                input_dtype="float32",
                target_spec=target_spec,
                dataset_metadata={"batch_period_ms": 2000},
                task_metadata={"num_classes": 10},
            ),
            bundle=SimpleNamespace(metadata={"batch_period_ms": 2000}, calibration=calibration),
            model_family=model_family,
            task=task,
            target_spec=target_spec,
        )

    def _args(self, output_dir: Path, **overrides) -> argparse.Namespace:
        """Build parsed-argument stand-ins for smoke tests."""

        values = {
            "config": "config.yaml",
            "output_dir": str(output_dir),
            "model_variant": None,
            "checkpoint_path": None,
            "serial_port": None,
            "harness_serial_port": None,
            "energy_aware": False,
            "runtime_mode": None,
            "measured_runs": None,
            "cpu_clock_mhz": None,
            "preflight_only": False,
            "prepare_only": False,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def _patch_preflight_dependencies(self, pipeline: SimpleNamespace | None = None):
        """Patch heavy preflight dependencies for one smoke test."""

        active_pipeline = self._pipeline() if pipeline is None else pipeline
        converter = MagicMock()
        converter.convert.return_value = b"real-tflite-bytes"
        return (
            active_pipeline,
            patch("run_audio_stm32_hil_smoke.ensure_audio_components_registered"),
            patch("run_audio_stm32_hil_smoke.load_config", return_value=self._config()),
            patch("run_audio_stm32_hil_smoke.bootstrap_pipeline", return_value=active_pipeline),
            patch(
                "run_audio_stm32_hil_smoke.tf.lite.TFLiteConverter.from_keras_model",
                return_value=converter,
            ),
        )

    def test_validate_args_rejects_invalid_values(self) -> None:
        """CLI validation should reject invalid Phase 6 overrides."""

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(SystemExit):
                smoke.parse_args(["--runtime-mode", "invalid"])
            with self.assertRaisesRegex(ValueError, "positive integer"):
                smoke.validate_args(self._args(Path(tmpdir), measured_runs=0))
            with self.assertRaisesRegex(ValueError, "--cpu-clock-mhz"):
                smoke.validate_args(self._args(Path(tmpdir), cpu_clock_mhz=123))
            with self.assertRaisesRegex(ValueError, "--checkpoint-path"):
                smoke.validate_args(self._args(Path(tmpdir), model_variant="trained"))

    def test_preflight_builds_request_and_writes_diagnostic_tflite(self) -> None:
        """Preflight should build the seeded model request without hardware calls."""

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            args = self._args(output_dir, preflight_only=True)
            pipeline, ensure_patch, config_patch, bootstrap_patch, converter_patch = self._patch_preflight_dependencies()
            with ensure_patch, config_patch, bootstrap_patch, converter_patch as converter_factory, patch(
                "run_audio_stm32_hil_smoke.get_microcontroller_device"
            ) as device_factory, patch(
                "run_audio_stm32_hil_smoke.HILServer"
            ) as server_cls:
                result = smoke.run_preflight(args)

            converter_factory.assert_called_once_with(pipeline.model_family.materialize_export_model.return_value)
            self.assertEqual(
                (output_dir / "preflight" / "audio_stm32_hil_smoke.tflite").read_bytes(),
                b"real-tflite-bytes",
            )
            self.assertEqual(result["runtime_metadata"]["timesteps"], 201)
            self.assertEqual(result["runtime_metadata"]["input_dim"], 64)
            self.assertEqual(result["runtime_metadata"]["flops"], 12345)
            self.assertEqual(result["input_source"], "precomputed_log_mel_features")
            self.assertEqual(result["pipeline_scope"], "classifier_inference_only")
            self.assertFalse(result["frontend_included"])
            self.assertIn("log-mel extraction", result["frontend_excluded_reason"])
            self.assertFalse(device_factory.called)
            self.assertFalse(server_cls.called)
            json.dumps(result)

    def test_build_preflight_uses_bundle_calibration_in_candidate_request(self) -> None:
        """Candidate preparation request should use the real cached calibration split."""

        with tempfile.TemporaryDirectory() as tmpdir:
            args = self._args(Path(tmpdir), preflight_only=True)
            pipeline, ensure_patch, config_patch, bootstrap_patch, converter_patch = self._patch_preflight_dependencies()
            with ensure_patch, config_patch, bootstrap_patch, converter_patch:
                preflight = smoke.build_preflight(args)

            self.assertIs(preflight.request.calibration_split, pipeline.bundle.calibration)
            self.assertEqual(preflight.request.input_shape, (201, 64))
            self.assertEqual(preflight.request.model_variant, "untrained")
            self.assertEqual(preflight.request.artifact_root, Path(tmpdir).resolve() / "candidates")

    def test_preflight_missing_cache_mentions_prepare_target(self) -> None:
        """Missing cache failures in preflight should point to the prepare target."""

        with tempfile.TemporaryDirectory() as tmpdir:
            args = self._args(Path(tmpdir), preflight_only=True)
            converter = MagicMock()
            converter.convert.return_value = b"unused"
            with patch("run_audio_stm32_hil_smoke.ensure_audio_components_registered"), patch(
                "run_audio_stm32_hil_smoke.load_config",
                return_value=self._config(),
            ), patch(
                "run_audio_stm32_hil_smoke.bootstrap_pipeline",
                side_effect=FileNotFoundError("cache missing"),
            ), patch(
                "run_audio_stm32_hil_smoke.tf.lite.TFLiteConverter.from_keras_model",
                return_value=converter,
            ):
                with self.assertRaisesRegex(FileNotFoundError, "make prepare-audio-dataset"):
                    smoke.run_preflight(args)

    def test_preflight_rejects_missing_calibration(self) -> None:
        """Preflight should require the cache-owned calibration split."""

        with tempfile.TemporaryDirectory() as tmpdir:
            args = self._args(Path(tmpdir), preflight_only=True)
            pipeline = self._pipeline(calibration_rows=0)
            pipeline.bundle.calibration = DataSplit(
                inputs=np.zeros((0, 201, 64), dtype=np.float32),
                targets=np.zeros((0,), dtype=np.int64),
                metadata={},
            )
            pipeline, ensure_patch, config_patch, bootstrap_patch, converter_patch = self._patch_preflight_dependencies(
                pipeline
            )
            with ensure_patch, config_patch, bootstrap_patch, converter_patch:
                with self.assertRaisesRegex(ValueError, "make prepare-audio-dataset"):
                    smoke.run_preflight(args)

    def test_preflight_rejects_non_phase6_audio_shape(self) -> None:
        """Preflight should enforce the fixed Phase 6 `(201, 64)` feature shape."""

        with tempfile.TemporaryDirectory() as tmpdir:
            args = self._args(Path(tmpdir), preflight_only=True)
            pipeline, ensure_patch, config_patch, bootstrap_patch, converter_patch = self._patch_preflight_dependencies(
                self._pipeline(input_shape=(100, 20))
            )
            with ensure_patch, config_patch, bootstrap_patch, converter_patch:
                with self.assertRaisesRegex(ValueError, "expects fixed log-mel input_shape"):
                    smoke.run_preflight(args)

    def test_prepare_only_calls_device_prepare_without_hil_runtime(self) -> None:
        """Prepare-only should call the STM32 backend directly and skip runtime collection."""

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            staged_root = output_dir / "staged"
            inc_dir = staged_root / "Appli" / "Inc"
            inc_dir.mkdir(parents=True)
            (inc_dir / "network.h").write_text(
                "#define AI_NETWORK_IN_1_SIZE_BYTES (12864)\n",
                encoding="utf-8",
            )
            device = MagicMock()
            device.prepare_candidate.return_value = staged_root
            args = self._args(output_dir, prepare_only=True, cpu_clock_mhz=600)
            pipeline, ensure_patch, config_patch, bootstrap_patch, converter_patch = self._patch_preflight_dependencies()
            with ensure_patch, config_patch, bootstrap_patch, converter_patch, patch(
                "run_audio_stm32_hil_smoke.get_microcontroller_device",
                return_value=device,
            ) as device_factory, patch(
                "run_audio_stm32_hil_smoke.HILServer"
            ) as server_cls:
                result = smoke.run_prepare_only(args)

            device_factory.assert_called_once()
            self.assertEqual(device_factory.call_args.kwargs["device_options"]["cpu_clock_mhz"], 600)
            device.prepare_candidate.assert_called_once()
            self.assertFalse(server_cls.called)
            self.assertTrue(result["prepared_network_input_ok"])
            self.assertEqual(result["prepared_network_input_bytes"], 12864)
            json.dumps(result)

    def test_prepare_input_validation_records_followup_on_mismatch(self) -> None:
        """Prepared input validation should flag mismatched generated byte counts."""

        with tempfile.TemporaryDirectory() as tmpdir:
            staged_root = Path(tmpdir) / "staged"
            inc_dir = staged_root / "Appli" / "Inc"
            inc_dir.mkdir(parents=True)
            (inc_dir / "network.h").write_text(
                "#define AI_NETWORK_IN_1_SIZE_BYTES (1234)\n",
                encoding="utf-8",
            )

            observed, ok, followups = smoke.validate_prepared_input_contract(
                staged_root,
                quantized_input=False,
            )

            self.assertEqual(observed, 1234)
            self.assertFalse(ok)
            self.assertIn("Phase 7", followups[0])

    def test_prepare_input_validation_accepts_quantized_audio_bytes(self) -> None:
        """Prepared input validation should accept int8 quantized audio input bytes."""

        with tempfile.TemporaryDirectory() as tmpdir:
            staged_root = Path(tmpdir) / "staged"
            inc_dir = staged_root / "Appli" / "Inc"
            inc_dir.mkdir(parents=True)
            (inc_dir / "network.h").write_text(
                "#define AI_NETWORK_IN_1_SIZE_BYTES (12864)\n",
                encoding="utf-8",
            )

            observed, ok, followups = smoke.validate_prepared_input_contract(
                staged_root,
                quantized_input=True,
            )

            self.assertEqual(observed, 12864)
            self.assertTrue(ok)
            self.assertEqual(followups, [])

    def test_full_hil_calls_server_with_runtime_metadata_and_overrides(self) -> None:
        """Full HIL mode should route through HILServer with explicit request metadata."""

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            args = self._args(
                output_dir,
                serial_port="/dev/ttyUSB0",
                harness_serial_port="/dev/ttyACM1",
                energy_aware=True,
                runtime_mode="cadenced",
                measured_runs=3,
                cpu_clock_mhz=800,
            )
            server = MagicMock()
            server.determine_metrics.return_value = {"latency_ms": 1.25}
            pipeline, ensure_patch, config_patch, bootstrap_patch, converter_patch = self._patch_preflight_dependencies()
            with ensure_patch, config_patch, bootstrap_patch, converter_patch, patch(
                "run_audio_stm32_hil_smoke.HILServer",
                return_value=server,
            ) as server_cls:
                result = smoke.run_full_hil(args)

            config_arg = server_cls.call_args.kwargs["config"]
            self.assertTrue(config_arg.device.hil)
            self.assertEqual(config_arg.device.serial_port, "/dev/ttyUSB0")
            self.assertEqual(config_arg.device.harness_serial_port, "/dev/ttyACM1")
            self.assertTrue(config_arg.training.energy_aware)
            self.assertEqual(config_arg.device.runtime_mode, "cadenced")
            self.assertEqual(config_arg.device.measured_inference_runs, 3)
            server.determine_metrics.assert_called_once()
            self.assertEqual(server.determine_metrics.call_args.kwargs["model_variant"], "untrained")
            self.assertEqual(
                server.determine_metrics.call_args.kwargs["device_options_overrides"],
                {"cpu_clock_mhz": 800},
            )
            self.assertEqual(result["metrics"], {"latency_ms": 1.25})
            json.dumps(result)

    def test_full_hil_accepts_master_success_code(self) -> None:
        """Full HIL summaries should treat HIL_MASTER_SUCCESS as success."""

        with tempfile.TemporaryDirectory() as tmpdir:
            args = self._args(Path(tmpdir))
            server = MagicMock()
            server.determine_metrics.return_value = {
                "error_code": 1,
                "error_label": "HIL_MASTER_SUCCESS",
                "runtime_mode": "cadenced",
            }
            pipeline, ensure_patch, config_patch, bootstrap_patch, converter_patch = self._patch_preflight_dependencies()
            with ensure_patch, config_patch, bootstrap_patch, converter_patch, patch(
                "run_audio_stm32_hil_smoke.HILServer",
                return_value=server,
            ):
                result = smoke.run_full_hil(args)

            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["metrics"]["error_label"], "HIL_MASTER_SUCCESS")

    def test_full_hil_marks_nonzero_error_code_blocked(self) -> None:
        """Full HIL summaries should fail the command when hardware upload fails."""

        with tempfile.TemporaryDirectory() as tmpdir:
            args = self._args(Path(tmpdir))
            server = MagicMock()
            server.determine_metrics.return_value = {
                "error_code": 6,
                "error_label": "HIL_MASTER_DEVICE_NOT_FOUND",
                "backend_error_kind": "upload",
            }
            pipeline, ensure_patch, config_patch, bootstrap_patch, converter_patch = self._patch_preflight_dependencies()
            with ensure_patch, config_patch, bootstrap_patch, converter_patch, patch(
                "run_audio_stm32_hil_smoke.HILServer",
                return_value=server,
            ):
                result = smoke.run_full_hil(args)

            self.assertEqual(result["status"], "blocked")
            self.assertEqual(result["metrics"]["error_code"], 6)


if __name__ == "__main__":
    unittest.main()
