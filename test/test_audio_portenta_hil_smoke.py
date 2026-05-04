"""Tests for the Arduino audio HIL smoke utility."""

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
SCRIPT_DIR = ROOT_DIR / "analysis_scripts" / "audio_portenta_hil_smoke"
for path in (SRC_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_audio_portenta_hil_smoke as smoke  # noqa: E402
from tinyodom.hardware import return_hardware_specs  # noqa: E402
from tinyodom.microcontrollers import resolve_device_options  # noqa: E402
from tinyodom.model import load_config  # noqa: E402
from tinyodom.pipeline_types import DataSplit, ModelBuildContext, TargetSpec  # noqa: E402


class AudioPortentaConfigTests(unittest.TestCase):
    """Validate the Phase 8 audio Portenta config."""

    def test_audio_portenta_config_selects_required_components_and_limits(self) -> None:
        """The checked-in config should resolve the required Phase 8 policy."""

        config = load_config(
            SRC_DIR / "config" / "nas_config_audio_portenta.yaml",
            task_metric_names={"accuracy"},
            training_only_task_metric_names=set(),
        )
        self.assertEqual(config.device.name, "PORTENTA_H7")
        self.assertEqual(config.device.runtime_mode, "back_to_back")
        self.assertEqual(config.device.portenta.target_core, "cm7")
        self.assertEqual(config.device.portenta.split, "75_25")
        self.assertFalse(config.training.energy_aware)
        self.assertEqual(config.training.input_mode, "uniform")
        self.assertEqual(config.dataset.name, "urbansound8k_mel")
        self.assertEqual(config.task.name, "sound_classification")
        self.assertEqual(config.model.family, "audio_dscnn")
        self.assertEqual(config.model.params.export_variant, "untrained")
        self.assertEqual(Path(config.outputs.candidate_dir).name, "audio_dscnn")
        self.assertEqual(config.outputs.artifact_stem, "TinyOdomEx_UrbanSound8K")

        options = resolve_device_options(config.device.name, config.device)
        ram_bytes, flash_bytes = return_hardware_specs(config.device.name, device_options=options)
        self.assertEqual(ram_bytes, 786432)
        self.assertEqual(flash_bytes, 1572864)

        terms = config.nas.score.params.terms
        self.assertEqual(terms[0].metric, "accuracy")
        self.assertEqual(float(terms[0].weight), 1.0)
        self.assertEqual(terms[1].metric, "ram_bytes")
        self.assertEqual(float(terms[1].weight), -0.10)
        self.assertEqual(terms[2].metric, "total_flash_bytes")
        self.assertEqual(float(terms[2].weight), -0.10)
        self.assertEqual(terms[3].type, "boundary")
        self.assertEqual(terms[3].metric, "latency_ms")
        self.assertEqual(float(terms[3].weight), 0.01)


class AudioPortentaHILSmokeTests(unittest.TestCase):
    """Validate the Phase 8 Arduino audio smoke runner."""

    def _config(self) -> Dict:
        """Build a minimal mutable audio Portenta config."""

        return Dict(
            config_path="config.yaml",
            outputs=Dict(
                candidate_dir="audio_dscnn",
                tflite_model_path="unused.tflite",
                checkpoint_path="checkpoint.keras",
            ),
            device=Dict(
                name="PORTENTA_H7",
                hil=False,
                runtime_mode="back_to_back",
                serial_port="/dev/ttyACM0",
                measured_inference_runs=10,
                portenta=Dict(target_core="cm7", split="75_25", security="none"),
            ),
            training=Dict(
                energy_aware=False,
                quantization=True,
                latency_proxy_max_flops=30_000_000.0,
                input_mode="uniform",
            ),
            model=Dict(params=Dict(export_variant="untrained")),
            dataset=Dict(params=Dict(batch_period_ms=2000)),
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
            "device_name": None,
            "target_core": None,
            "split": None,
            "measured_runs": None,
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
            patch("run_audio_portenta_hil_smoke.ensure_audio_components_registered"),
            patch("run_audio_portenta_hil_smoke.load_config", return_value=self._config()),
            patch("run_audio_portenta_hil_smoke.bootstrap_pipeline", return_value=active_pipeline),
            patch(
                "run_audio_portenta_hil_smoke.tf.lite.TFLiteConverter.from_keras_model",
                return_value=converter,
            ),
        )

    def test_preflight_builds_request_and_writes_diagnostic_tflite(self) -> None:
        """Preflight should build the seeded model request without hardware calls."""

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            args = self._args(output_dir, preflight_only=True)
            pipeline, ensure_patch, config_patch, bootstrap_patch, converter_patch = self._patch_preflight_dependencies()
            with ensure_patch, config_patch, bootstrap_patch, converter_patch as converter_factory, patch(
                "run_audio_portenta_hil_smoke.get_microcontroller_device"
            ) as device_factory, patch(
                "run_audio_portenta_hil_smoke.HILServer"
            ) as server_cls:
                result = smoke.run_preflight(args)

            converter_factory.assert_called_once_with(pipeline.model_family.materialize_export_model.return_value)
            self.assertEqual(
                (output_dir / "preflight" / "audio_portenta_hil_smoke.tflite").read_bytes(),
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

    def test_prepare_only_stages_audio_sketch_and_compile_only_metrics(self) -> None:
        """Prepare-only should stage `audio_dscnn.ino` and skip upload/runtime."""

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            staged_root = output_dir / "audio_dscnn"
            staged_root.mkdir(parents=True)
            (staged_root / "audio_dscnn.ino").write_text(
                "\n".join(
                    [
                        "#define TINYODOM_WINDOW_SIZE 201",
                        "#define TINYODOM_NUM_CHANNELS 64",
                        "#define TINYODOM_TENSOR_ARENA_BYTES (96 * 1024)",
                    ]
                ),
                encoding="utf-8",
            )
            device = MagicMock()
            device.prepare_candidate.return_value = staged_root
            args = self._args(output_dir, prepare_only=True)
            pipeline, ensure_patch, config_patch, bootstrap_patch, converter_patch = self._patch_preflight_dependencies()
            with ensure_patch, config_patch, bootstrap_patch, converter_patch, patch(
                "run_audio_portenta_hil_smoke.get_microcontroller_device",
                return_value=device,
            ) as device_factory, patch(
                "run_audio_portenta_hil_smoke.collect_metrics",
                return_value={"error_code": 0, "error_label": "HIL_ERROR_OK", "ram_bytes": 100, "flash_bytes": 200},
            ) as collect_mock:
                result = smoke.run_prepare_only(args)

            device_factory.assert_called_once()
            self.assertEqual(device_factory.call_args.kwargs["device_options"]["target_core"], "cm7")
            device.prepare_candidate.assert_called_once()
            request_arg = collect_mock.call_args.args[0]
            self.assertFalse(request_arg.hil_enabled)
            self.assertEqual(request_arg.window_size, 201)
            self.assertEqual(request_arg.input_dim, 64)
            self.assertEqual(result["candidate_root"], str(staged_root))
            self.assertEqual(result["staged_sketch_path"], str(staged_root / "audio_dscnn.ino"))
            self.assertEqual(result["staged_window_size"], 201)
            self.assertEqual(result["staged_num_channels"], 64)
            self.assertTrue(result["staged_shape_ok"])
            self.assertEqual(result["status"], "ok")
            json.dumps(result)

    def test_prepare_only_records_ble_blocker_evidence(self) -> None:
        """BLE prepare-only failures should preserve resource-pressure evidence."""

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            staged_root = output_dir / "audio_dscnn"
            staged_root.mkdir(parents=True)
            (staged_root / "audio_dscnn.ino").write_text(
                "#define TINYODOM_WINDOW_SIZE 201\n#define TINYODOM_NUM_CHANNELS 64\n",
                encoding="utf-8",
            )
            device = MagicMock()
            device.prepare_candidate.return_value = staged_root
            args = self._args(output_dir, prepare_only=True, device_name="ARDUINO_NANO_33_BLE_SENSE")
            metrics = {
                "error_code": 3,
                "error_label": "HIL_MASTER_RAM_OVERFLOW",
                "ram_bytes": 230000,
                "flash_bytes": 1000,
                "arena_bytes": 98304,
            }
            pipeline, ensure_patch, config_patch, bootstrap_patch, converter_patch = self._patch_preflight_dependencies()
            with ensure_patch, config_patch, bootstrap_patch, converter_patch, patch(
                "run_audio_portenta_hil_smoke.get_microcontroller_device",
                return_value=device,
            ), patch(
                "run_audio_portenta_hil_smoke.collect_metrics",
                return_value=metrics,
            ):
                result = smoke.run_prepare_only(args)

            self.assertEqual(result["device_name"], "ARDUINO_NANO_33_BLE_SENSE")
            self.assertEqual(result["status"], "blocked")
            self.assertIn("12864 int8 bytes", result["followups"][0])
            self.assertIn("ram_bytes=230000", result["followups"][0])


if __name__ == "__main__":
    unittest.main()
