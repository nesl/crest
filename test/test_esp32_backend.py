# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Hardware-free checks for the ESP32-S3 ESP-IDF backend."""

from pathlib import Path
import hashlib
import json
import os
import sys
import types
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import serial

import numpy as np
from addict import Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crest.devices import (
    CandidatePrepareRequest, CandidatePreparationError, CompileResult, DeviceMetrics, MeasureResult, UploadResult,
)
from crest.errors import (
    HIL_ERROR_INTEGRITY, HIL_ERROR_LATENCY, HIL_ERROR_OK, HIL_ERROR_UNDER_SIZED,
)
from crest.microcontrollers import get_device, list_device_specs, resolve_device_options
from crest.microcontrollers.esp32_idf import (
    ESP32IDFDevice,
    _validate_source_tflite_description,
    classify_idf_build_failure,
    parse_idf_size_output,
    resolve_esp32_idf_options,
)


class ESP32BackendTests(unittest.TestCase):
    def test_registry_constructs_backend_with_singleton_psram_arena(self) -> None:
        device = get_device("freenove_esp32_s3_wroom")
        self.assertIsInstance(device, ESP32IDFDevice)
        self.assertEqual(tuple(device.spec.arena_sizes_kb), (2048,))
        self.assertEqual(list_device_specs()["FREENOVE_ESP32_S3_WROOM"]["max_flash"], 4 * 1024 * 1024)

    def test_options_preserve_verified_harness_mapping_and_profile(self) -> None:
        options = resolve_esp32_idf_options(None)
        self.assertEqual(options.harness_trigger_gpio, 21)
        self.assertEqual(options.harness_arm_gpio, 47)
        self.assertEqual(options.optimization_profile, "debug")
        self.assertEqual(options.psram_bytes, 8 * 1024 * 1024)
        self.assertEqual(resolve_esp32_idf_options({"warmup_runs": 0}).warmup_runs, 0)

    def test_options_reject_fractional_arena_and_invalid_target(self) -> None:
        with self.assertRaises(ValueError):
            resolve_esp32_idf_options({"arena_sizes_kb": [2048.5]})
        with self.assertRaises(ValueError):
            resolve_esp32_idf_options({"target": "esp32"})
        for timeout in (True, float("nan"), float("inf"), 0):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                resolve_esp32_idf_options({"build_timeout_s": timeout})
        with self.assertRaises(ValueError):
            resolve_esp32_idf_options({"model_array_name": "other"})

    def test_config_options_can_be_passed_to_backend_without_internal_names(self) -> None:
        device_config = SimpleNamespace(
            runtime_mode="back_to_back",
            esp32=SimpleNamespace(arena_sizes_kb=[2048], warmup_runs=1),
        )
        options = resolve_device_options("FREENOVE_ESP32_S3_WROOM", device_config)
        for internal_name in (
            "model_output_dir", "model_source_name", "model_header_name", "model_array_name",
        ):
            self.assertNotIn(internal_name, options)
        device = get_device("FREENOVE_ESP32_S3_WROOM", device_options=options)
        self.assertIsInstance(device, ESP32IDFDevice)
        self.assertTrue(device.supports_tflite_artifact())

    def test_real_hil_capabilities_constructs_esp32_backend(self) -> None:
        from hil_server import HILServer
        with tempfile.TemporaryDirectory() as tempdir:
            config = Dict({
                "device": {"name": "FREENOVE_ESP32_S3_WROOM", "hil": True,
                           "runtime_mode": "back_to_back", "esp32": {"arena_sizes_kb": [2048]}},
                "dataset": {"name": "ferplus_rafdb_mixed", "params": {}},
                "task": {"name": "facial_expression_classification", "params": {}},
                "model": {"family": "fer_mini_xception",
                          "params": {"export_variant": "trained"}, "search": {}},
                "training": {"energy_aware": False},
                "outputs": {"models_dir": tempdir, "candidate_dir": tempdir},
                "network": {"max_artifact_bytes": 4194304},
            })
            server = HILServer(config=config)
            try:
                capabilities = server._artifact_capabilities()
            finally:
                server.socket.close(linger=0)
                server.context.term()
        self.assertEqual(capabilities["backend"]["name"], "FREENOVE_ESP32_S3_WROOM")
        self.assertTrue(capabilities["backend"]["supports_tflite_artifact"])

    def test_options_reject_capacities_beyond_verified_hardware(self) -> None:
        for options in (
            {"max_internal_ram_bytes": 512 * 1024 + 1},
            {"max_flash_bytes": 4 * 1024 * 1024 + 1},
            {"psram_bytes": 8 * 1024 * 1024 + 1},
            {"psram_bytes": 1024 * 1024, "arena_sizes_kb": [2048]},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                resolve_esp32_idf_options(options)

    def test_idf_size_and_overflow_parsing(self) -> None:
        flash, ram = parse_idf_size_output(
            "Total image size: 599232 bytes\nUsed static DRAM: 42000 bytes"
        )
        self.assertEqual((flash, ram), (599232, 42000))
        flash, ram = parse_idf_size_output(
            "│ IRAM │ 71975 │ 54.91 │ 59097 │ 131072 │\n"
            "│ DRAM │ 11812 │ 6.54 │ 168924 │ 180736 │\n"
            "Total image size: 412012 bytes"
        )
        self.assertEqual((flash, ram), (412012, 83787))
        flash, ram = parse_idf_size_output(
            "│ DIRAM │ 63059 │ 18.45 │ 278701 │ 341760 │\n"
            "│ IRAM │ 16384 │ 100.0 │ 0 │ 16384 │\n"
            "Total image size: 629937 bytes"
        )
        self.assertEqual((flash, ram), (629937, 79443))
        self.assertEqual(classify_idf_build_failure("region `dram0_0_seg' overflowed"), "ram")
        self.assertEqual(classify_idf_build_failure("app partition does not fit flash"), "flash")

    def test_compile_builds_incrementally_with_defines(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            project = Path(tempdir) / "project"
            project.mkdir()
            (project / "CMakeLists.txt").write_text("project(fake)\n")
            (project / "sdkconfig").write_text('CONFIG_IDF_TARGET="esp32s3"\n')
            device = ESP32IDFDevice(device_options={"project_root": project, "build_timeout_s": 2})
            completed = type("Completed", (), {"returncode": 0, "stdout": "Total image size: 123 bytes"})()
            with patch("crest.microcontrollers.esp32_idf.subprocess.run", side_effect=[completed, completed]) as run:
                result = device.compile(
                    sketch_path=project,
                    arena_kb=2048,
                    window_size=64,
                    num_channels=64,
                    build_defines={"CREST_INFERENCE_RUNS": 50},
                )
            self.assertTrue(result.success)
            self.assertEqual(result.flash_bytes, 123)
            command = run.call_args_list[0].args[0]
            self.assertIn("-DCREST_TENSOR_ARENA_BYTES=2097152", command)
            self.assertIn("-DCREST_INFERENCE_RUNS=50", command)
            self.assertEqual(command[-2:], ["build", "size"])

    def test_runtime_timeout_and_arena_failure_remain_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            project = root / "project"
            staged = root / "staged"
            (project / "main").mkdir(parents=True)
            (staged / "main").mkdir(parents=True)
            for name in ("model_data.cc", "model_data.h", "fixtures.h", "crest_manifest.h", "crest_candidate.tflite"):
                (staged / "main" / name).write_text(name)
            (staged / "main" / "crest_model_manifest.json").write_text("{}")
            device = ESP32IDFDevice(serial_port="/dev/fake", device_options={"project_root": project})
            compiled = CompileResult(True, "ok", 100, 50, None, project / "build", 2048 * 1024)
            cases = (
                (MeasureResult(None, None, [], {}), HIL_ERROR_LATENCY),
                (MeasureResult(None, "tensor arena AllocateTensors failed", [], {}), HIL_ERROR_UNDER_SIZED),
                (MeasureResult(None, None, [], {
                    "measurement_instrumentation_valid": False,
                    "measurement_failure_kind": "serial_io",
                }), HIL_ERROR_INTEGRITY),
            )
            for measured, expected in cases:
                with self.subTest(expected=expected), \
                     patch.object(device, "compile", return_value=compiled), \
                     patch.object(device, "upload", return_value=UploadResult(True, "ok")), \
                     patch.object(device, "measure", return_value=measured):
                    actual = device.evaluate(
                        dirpath=staged, arena_kb=2048, window_size=2, num_channels=2, run_hil=True
                    )
                self.assertEqual(actual.error_code, expected)

    def test_hil_controller_accepts_valid_over_budget_and_distinguishes_failures(self) -> None:
        from crest.errors import HIL_MASTER_ARENA_EXHAUSTED, HIL_MASTER_FATAL, HIL_MASTER_SUCCESS
        from crest.hardware import HIL_controller
        device = ESP32IDFDevice()
        cases = (
            (DeviceMetrics(100, 200, 3.55, 2048 * 1024, HIL_ERROR_OK,
                           {"measurement_instrumentation_valid": True}), HIL_MASTER_SUCCESS),
            (DeviceMetrics(100, 200, -1.0, 2048 * 1024, HIL_ERROR_LATENCY), HIL_MASTER_FATAL),
            (DeviceMetrics(100, 200, -1.0, 2048 * 1024, HIL_ERROR_UNDER_SIZED),
             HIL_MASTER_ARENA_EXHAUSTED),
        )
        with tempfile.TemporaryDirectory() as tempdir:
            for metrics, expected_master in cases:
                with self.subTest(expected_master=expected_master), \
                     patch.object(device, "evaluate", return_value=metrics):
                    result = HIL_controller(
                        dirpath=tempdir, chosen_device="FREENOVE_ESP32_S3_WROOM",
                        window_size=96, number_of_channels=96, run_hil=True, device=device,
                    )
                self.assertEqual(result[4], expected_master)
                if expected_master == HIL_MASTER_SUCCESS:
                    self.assertEqual(result[2], 3.55)
                    self.assertTrue(result[5]["measurement_instrumentation_valid"])

    def test_normal_hil_controller_uses_singleton_arena_compile_only(self) -> None:
        from crest.errors import HIL_MASTER_SUCCESS
        from crest.hardware import HIL_controller
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            project = root / "project"
            staged = root / "staged"
            (project / "main").mkdir(parents=True)
            (staged / "main").mkdir(parents=True)
            for name in ("model_data.cc", "model_data.h", "fixtures.h", "crest_manifest.h", "crest_candidate.tflite"):
                (staged / "main" / name).write_text(name)
            (staged / "main" / "crest_model_manifest.json").write_text("{}")
            device = ESP32IDFDevice(device_options={"project_root": project})
            compiled = CompileResult(True, "ok", 412012, 83787, None, project / "build", 2048 * 1024)
            with patch.object(device, "compile", return_value=compiled) as compile_backend:
                result = HIL_controller(
                    dirpath=staged, chosen_device="FREENOVE_ESP32_S3_WROOM",
                    window_size=2, number_of_channels=2, run_hil=False, device=device,
                )
            self.assertEqual(result[:5], (83787, 412012, -1.0, 2048 * 1024, HIL_MASTER_SUCCESS))
            self.assertEqual(compile_backend.call_count, 1)

    def test_upload_is_application_only(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            project = Path(tempdir)
            device = ESP32IDFDevice(serial_port="/dev/fake", device_options={"project_root": project})
            completed = type("Completed", (), {"returncode": 0, "stdout": "ok"})()
            with patch("crest.microcontrollers.esp32_idf.subprocess.run", return_value=completed) as run:
                result = device.upload(sketch_path=project, build_dir=project / "build", serial_port=None)
            self.assertTrue(result.success)
            self.assertEqual(run.call_args.args[0][-1], "app-flash")

    def test_same_variant_preparations_have_distinct_retained_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            project = root / "project"
            (project / "main").mkdir(parents=True)
            (project / "CMakeLists.txt").write_text("project(fake)\n")
            fixture_npz = root / "fixtures.npz"
            fixture_manifest = root / "fixtures.json"
            fixture_npz.write_bytes(b"fixture")
            fixture_manifest.write_text("{}")
            device = ESP32IDFDevice(device_options={
                "project_root": project, "fixture_npz_path": fixture_npz,
                "fixture_manifest_path": fixture_manifest,
            })
            contents = iter((b"first-model", b"second-model"))
            def fake_tflite(**kwargs):
                kwargs["output_name"].write_bytes(next(contents))
            def fake_cpp(model_path, output_dir, **kwargs):
                data = model_path.read_bytes()
                (output_dir / kwargs["source_name"]).write_text(
                    "{" + ",".join(f"0x{value:02x}" for value in data) + "}"
                )
                (output_dir / kwargs["header_name"]).write_text("extern const unsigned char g_model[];")
            fixture_evidence = {
                "fixture_sha256": "f" * 64, "fixture_manifest_sha256": "e" * 64,
                "fixture_manifest": {}, "fixture_count": 1, "fixture_bytes_per_input": 4,
                "fixture_ids": ["id"], "fixture_wire_ids": ["b64:aWQ"],
                "input_scale": 0.5, "input_zero_point": 0, "output_elements": 2,
            }
            request = CandidatePrepareRequest(
                config=SimpleNamespace(device=SimpleNamespace(measured_inference_runs=2)),
                model=object(), model_variant="same", artifact_root=root / "artifacts",
                tflite_model_path=root / "candidate.tflite", calibration_split=None,
                quantization_mode="float32", input_shape=(2, 2),
            )
            with patch("crest.hardware.convert_to_tflite_model", side_effect=fake_tflite), \
                 patch("crest.hardware.convert_to_cpp_model", side_effect=fake_cpp), \
                 patch("crest.microcontrollers.esp32_fixtures.stage_fixture_header", return_value=fixture_evidence):
                first = device.prepare_candidate(request=request)
                (first / "esp32_application.bin").write_bytes(b"first-app")
                second = device.prepare_candidate(request=request)
                (second / "esp32_application.bin").write_bytes(b"second-app")
            self.assertNotEqual(first, second)
            self.assertEqual((first / "main" / "crest_candidate.tflite").read_bytes(), b"first-model")
            self.assertEqual((second / "main" / "crest_candidate.tflite").read_bytes(), b"second-model")
            self.assertEqual((first / "esp32_application.bin").read_bytes(), b"first-app")
            self.assertEqual((second / "esp32_application.bin").read_bytes(), b"second-app")

    def test_harness_measurement_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            staged = root / "staged"
            (staged / "main").mkdir(parents=True)
            (staged / "main" / "crest_model_manifest.json").write_text("{}")
            device = ESP32IDFDevice(device_options={"project_root": root / "project"})
            result = device.evaluate(
                dirpath=staged, arena_kb=2048, run_hil=True, harness_serial_port="/dev/harness"
            )
            self.assertEqual(result.error_code, HIL_ERROR_INTEGRITY)
            self.assertFalse(result.power_metrics["measurement_instrumentation_valid"])

    def test_staging_failure_is_integrity_and_next_evaluation_recovers(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            project = root / "project"
            staged = root / "staged"
            (project / "main").mkdir(parents=True)
            (staged / "main").mkdir(parents=True)
            manifest = staged / "main" / "crest_model_manifest.json"
            manifest.write_text("not-json")
            device = ESP32IDFDevice(device_options={"project_root": project})
            failed = device.evaluate(
                dirpath=staged, arena_kb=2048, window_size=2, num_channels=2, run_hil=False
            )
            self.assertEqual(failed.error_code, HIL_ERROR_INTEGRITY)
            self.assertFalse(failed.power_metrics["measurement_instrumentation_valid"])
            manifest.write_text("{}")
            for name in ("model_data.cc", "model_data.h", "fixtures.h", "crest_manifest.h", "crest_candidate.tflite"):
                (staged / "main" / name).write_text(name)
            recovered_metrics = DeviceMetrics(1, 2, -1.0, 2048 * 1024, HIL_ERROR_OK)
            with patch("crest.devices.ArduinoDevice.evaluate", return_value=recovered_metrics):
                recovered = device.evaluate(
                    dirpath=staged, arena_kb=2048, window_size=2, num_channels=2, run_hil=False
                )
            self.assertEqual(recovered.error_code, HIL_ERROR_OK)

    def test_lock_timeout_is_integrity_but_unexpected_backend_bug_propagates(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            project = root / "project"
            staged = root / "staged"
            (project / "main").mkdir(parents=True)
            (staged / "main").mkdir(parents=True)
            (staged / "main" / "crest_model_manifest.json").write_text("{}")
            device = ESP32IDFDevice(device_options={"project_root": project})
            with patch("crest.microcontrollers.esp32_idf._acquire_lock", side_effect=TimeoutError("busy")):
                failed = device.evaluate(
                    dirpath=staged, arena_kb=2048, window_size=2, num_channels=2, run_hil=False
                )
            self.assertEqual(failed.error_code, HIL_ERROR_INTEGRITY)
            for name in ("model_data.cc", "model_data.h", "fixtures.h", "crest_manifest.h", "crest_candidate.tflite"):
                (staged / "main" / name).write_text(name)
            with patch("crest.devices.ArduinoDevice.evaluate", side_effect=RuntimeError("bug")):
                with self.assertRaisesRegex(RuntimeError, "bug"):
                    device.evaluate(
                        dirpath=staged, arena_kb=2048, window_size=2, num_channels=2, run_hil=False
                    )

    def test_evaluate_syncs_staging_under_workspace_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            project = root / "project"
            staged = root / "staged"
            (project / "main").mkdir(parents=True)
            (staged / "main").mkdir(parents=True)
            for name in ("model_data.cc", "model_data.h", "fixtures.h", "crest_manifest.h", "crest_candidate.tflite"):
                (staged / "main" / name).write_text(name)
            (staged / "main" / "crest_model_manifest.json").write_text("{}")
            device = ESP32IDFDevice(device_options={"project_root": project})
            metrics = DeviceMetrics(1, 2, 0.1, 2048 * 1024, 0)
            with patch("crest.devices.ArduinoDevice.evaluate", return_value=metrics) as evaluate:
                actual = device.evaluate(
                    dirpath=staged,
                    arena_kb=2048,
                    window_size=64,
                    num_channels=64,
                    run_hil=False,
                )
            self.assertEqual(actual, metrics)
            self.assertEqual((project / "main" / "model_data.cc").read_text(), "model_data.cc")
            self.assertEqual(evaluate.call_args.kwargs["dirpath"], project.resolve())

    def test_completed_measurement_without_valid_evidence_is_integrity_error(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            project = root / "project"
            staged = root / "staged"
            (project / "main").mkdir(parents=True)
            (staged / "main").mkdir(parents=True)
            for name in ("model_data.cc", "model_data.h", "fixtures.h", "crest_manifest.h", "crest_candidate.tflite"):
                (staged / "main" / name).write_text(name)
            (staged / "main" / "crest_model_manifest.json").write_text("{}")
            device = ESP32IDFDevice(serial_port="/dev/fake", device_options={"project_root": project})
            metrics = DeviceMetrics(1, 2, 0.1, 2048 * 1024, HIL_ERROR_OK, {})
            with patch("crest.devices.ArduinoDevice.evaluate", return_value=metrics):
                actual = device.evaluate(
                    dirpath=staged, arena_kb=2048, window_size=64, num_channels=64, run_hil=True
                )
            self.assertEqual(actual.error_code, HIL_ERROR_INTEGRITY)


class ESP32MeasurementEvidenceTests(unittest.TestCase):
    def _manifest(self) -> dict[str, object]:
        return {
            "fixture_ids": ["a", "b"],
            "fixture_count": 2,
            "measured_runs_per_fixture": 2,
            "warmup_runs_per_fixture": 1,
            "latency_aggregation": "median",
            "tflite_sha256": "a" * 64,
            "fixture_sha256": "b" * 64,
            "output_elements": 2,
        }

    def _valid_log(self) -> list[str]:
        return [
            f"ML_READY model_sha256={'a' * 64} fixture_sha256={'b' * 64} arena_used=100 arena_reserved=2097152 fixtures=2 warmups=1 repetitions=2 aggregation=median",
            "ML_SAMPLE fixture=0 id=a repetition=0 invoke_us=3500000",
            "ML_SAMPLE fixture=0 id=a repetition=1 invoke_us=3600000",
            "ML_OUTPUT fixture=0 id=a values=-1,2",
            "ML_SAMPLE fixture=1 id=b repetition=0 invoke_us=3400000",
            "ML_SAMPLE fixture=1 id=b repetition=1 invoke_us=3700000",
            "ML_OUTPUT fixture=1 id=b values=3,-4",
            f"MEASUREMENT_VALID model_sha256={'a' * 64} fixture_sha256={'b' * 64} samples=4",
            "runs: 4",
            "timer output: 3.550000000",
        ]

    def test_serial_disconnect_maps_to_controller_infrastructure_then_recovers(self) -> None:
        from crest.errors import HIL_MASTER_INFRASTRUCTURE, HIL_MASTER_SUCCESS
        from crest.hardware import HIL_controller
        device = ESP32IDFDevice()
        disconnected = DeviceMetrics(100, 200, -1.0, 2048 * 1024, HIL_ERROR_INTEGRITY, {
            "measurement_instrumentation_valid": False,
            "measurement_failure_kind": "serial_io",
        })
        recovered = DeviceMetrics(100, 200, 3.55, 2048 * 1024, HIL_ERROR_OK, {
            "measurement_instrumentation_valid": True,
        })
        with tempfile.TemporaryDirectory() as tempdir:
            with patch.object(device, "evaluate", side_effect=[disconnected, recovered]):
                failed_result = HIL_controller(
                    dirpath=tempdir, chosen_device="FREENOVE_ESP32_S3_WROOM",
                    window_size=96, number_of_channels=96, run_hil=True, device=device,
                )
                recovered_result = HIL_controller(
                    dirpath=tempdir, chosen_device="FREENOVE_ESP32_S3_WROOM",
                    window_size=96, number_of_channels=96, run_hil=True, device=device,
                )
        self.assertEqual(failed_result[4], HIL_MASTER_INFRASTRUCTURE)
        self.assertEqual(recovered_result[4], HIL_MASTER_SUCCESS)
        self.assertEqual(recovered_result[2], 3.55)

    def test_valid_over_budget_measurement_remains_instrumented(self) -> None:
        from crest.microcontrollers.esp32_measurement import validate_measurement_log
        evidence, diagnostics = validate_measurement_log(
            lines=self._valid_log(), manifest=self._manifest(), latency_s=3.55, psram_bytes=8 * 1024 * 1024
        )
        self.assertTrue(evidence["measurement_instrumentation_valid"])
        self.assertTrue(diagnostics["valid"])
        self.assertEqual(evidence["measurement_sample_count"], 4)

    def test_esp32_allocatetensors_boot_error_is_arena_failure(self) -> None:
        device = ESP32IDFDevice(serial_port="/dev/fake")
        device._active_manifest = self._manifest()
        failed = MeasureResult(
            None, None, ["Failed to resize buffer. Requested: 345488, available 47952",
                         "failed to allocate tensor arena: AllocateTensors failed"], {},
        )
        with patch("crest.devices.ArduinoDevice.measure", return_value=failed):
            measured = device.measure(
                serial_port="/dev/fake", baud_rate=115200, serial_timeout_s=10
            )
        self.assertIsNone(measured.latency_s)
        self.assertIn("Failed to resize buffer", measured.arena_error_line)
        self.assertFalse(measured.power_metrics["measurement_instrumentation_valid"])

    def test_measurement_record_write_failure_invalidates_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            device = ESP32IDFDevice(serial_port="/dev/fake")
            device._active_manifest = self._manifest()
            device._active_staged_root = Path(tempdir)
            successful = MeasureResult(3.55, None, self._valid_log(), {})
            with patch("crest.devices.ArduinoDevice.measure", return_value=successful), \
                 patch("pathlib.Path.write_text", side_effect=OSError("read-only")):
                measured = device.measure(
                    serial_port="/dev/fake", baud_rate=115200, serial_timeout_s=10
                )
            self.assertEqual(measured.latency_s, 3.55)
            self.assertFalse(measured.power_metrics["measurement_instrumentation_valid"])
            self.assertIn("read-only", measured.power_metrics["measurement_record_error"])

    def test_serial_failure_is_recoverable_on_next_measurement(self) -> None:
        device = ESP32IDFDevice(serial_port="/dev/fake")
        device._active_manifest = self._manifest()
        successful = MeasureResult(3.55, None, self._valid_log(), {})
        with patch(
            "crest.devices.ArduinoDevice.measure",
            side_effect=[serial.SerialException("disconnected"), successful],
        ):
            failed = device.measure(serial_port="/dev/fake", baud_rate=115200, serial_timeout_s=10)
            recovered = device.measure(serial_port="/dev/fake", baud_rate=115200, serial_timeout_s=10)
        self.assertIsNone(failed.latency_s)
        self.assertFalse(failed.power_metrics["measurement_instrumentation_valid"])
        self.assertEqual(recovered.latency_s, 3.55)
        self.assertTrue(recovered.power_metrics["measurement_instrumentation_valid"])

    def test_malformed_output_is_invalid_evidence_without_parser_error(self) -> None:
        from crest.microcontrollers.esp32_measurement import validate_measurement_log
        log = [line.replace("values=-1,2", "values=--,2") for line in self._valid_log()]
        evidence, diagnostics = validate_measurement_log(
            lines=log, manifest=self._manifest(), latency_s=3.55, psram_bytes=8 * 1024 * 1024
        )
        self.assertFalse(evidence["measurement_instrumentation_valid"])
        self.assertIn("output_count", diagnostics["reasons"])

    def test_missing_sample_invalidates_completed_measurement(self) -> None:
        from crest.microcontrollers.esp32_measurement import validate_measurement_log
        log = [line for line in self._valid_log() if "fixture=1 id=b repetition=1" not in line]
        evidence, diagnostics = validate_measurement_log(
            lines=log, manifest=self._manifest(), latency_s=3.55, psram_bytes=8 * 1024 * 1024
        )
        self.assertFalse(evidence["measurement_instrumentation_valid"])
        self.assertIn("sample_order", diagnostics["reasons"])


class ESP32SourceArtifactTests(unittest.TestCase):
    def _reference_root(self) -> Path:
        configured = os.environ.get("CREST_ESP32_REFERENCE_FIXTURES")
        if not configured:
            self.skipTest("CREST_ESP32_REFERENCE_FIXTURES is not configured")
        root = Path(configured)
        if not (root / "reference96_compile_fixture_untrained_int8.tflite").is_file():
            self.skipTest("ESP32 reference component artifact is unavailable")
        return root

    def _request(self, root: Path, source: Path, **overrides) -> CandidatePrepareRequest:
        values = {
            "config": SimpleNamespace(device=SimpleNamespace(measured_inference_runs=2)),
            "model": None, "model_variant": "supplied_tflite",
            "artifact_root": root / "artifacts", "tflite_model_path": root / "unused.tflite",
            "calibration_split": None, "quantization_mode": "int8_ptq",
            "input_shape": (96, 96), "checkpoint_path": None, "source_tflite_path": source,
        }
        values.update(overrides)
        return CandidatePrepareRequest(**values)

    def test_source_artifact_preserves_exact_bytes_without_export(self) -> None:
        source_root = self._reference_root()
        source = source_root / "reference96_compile_fixture_untrained_int8.tflite"
        if not source.is_file():
            self.skipTest("reference96 component artifact is unavailable")
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            project = root / "project"
            project.mkdir(); (project / "CMakeLists.txt").write_text("project(fake)\n")
            device = ESP32IDFDevice(device_options={
                "project_root": project,
                "fixture_npz_path": source_root / "representative_96.npz",
                "fixture_manifest_path": source_root / "representative_96.json",
            })
            with patch("crest.hardware.convert_to_tflite_model", side_effect=AssertionError("export called")) as export:
                staged = device.prepare_candidate(request=self._request(root, source))
            self.assertFalse(export.called)
            self.assertTrue(device.supports_tflite_artifact())
            self.assertEqual((staged / "main" / "crest_candidate.tflite").read_bytes(), source.read_bytes())
            manifest = json.loads((staged / "main" / "crest_model_manifest.json").read_text())
            self.assertEqual(manifest["preparation_mode"], "supplied_tflite")
            self.assertEqual(manifest["tflite_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())

    def test_source_conflicts_are_request_errors(self) -> None:
        source = self._reference_root() / "reference96_compile_fixture_untrained_int8.tflite"
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            project = root / "project"; project.mkdir(); (project / "CMakeLists.txt").write_text("project(fake)\n")
            device = ESP32IDFDevice(device_options={"project_root": project})
            for conflict in (
                {"model": object()}, {"calibration_split": object()}, {"checkpoint_path": "x"},
                {"quantization_mode": "float32"}, {"input_shape": (64, 64)},
            ):
                with self.subTest(conflict=next(iter(conflict))), self.assertRaises(CandidatePreparationError) as caught:
                    device.prepare_candidate(request=self._request(root, source, **conflict))
                self.assertEqual(caught.exception.error_kind, "request")

    def test_malformed_source_then_valid_source_recovers(self) -> None:
        source_root = self._reference_root()
        valid = source_root / "reference96_compile_fixture_untrained_int8.tflite"
        if not valid.is_file():
            self.skipTest("reference96 component artifact is unavailable")
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir); malformed = root / "bad.tflite"; malformed.write_bytes(b"not-tflite")
            project = root / "project"; project.mkdir(); (project / "CMakeLists.txt").write_text("project(fake)\n")
            device = ESP32IDFDevice(device_options={
                "project_root": project, "fixture_npz_path": source_root / "representative_96.npz",
                "fixture_manifest_path": source_root / "representative_96.json",
            })
            with self.assertRaises(CandidatePreparationError) as caught:
                device.prepare_candidate(request=self._request(root, malformed))
            self.assertEqual(caught.exception.error_kind, "request")
            staged = device.prepare_candidate(request=self._request(root, valid))
            self.assertEqual((staged / "main" / "crest_candidate.tflite").read_bytes(), valid.read_bytes())

    def test_source_tensor_contract_requires_batch_one(self) -> None:
        base = {
            "inputs": [{"shape": [1, 96, 96], "dtype": "int8",
                        "quantization": {"scales": [0.1], "zero_points": [0], "quantized_dimension": 0}}],
            "outputs": [{"shape": [1, 5], "dtype": "int8",
                         "quantization": {"scales": [0.1], "zero_points": [0], "quantized_dimension": 0}}],
        }
        _validate_source_tflite_description(base)
        for field, shape in (("inputs", [2, 96, 96]), ("outputs", [2, 5])):
            malformed = json.loads(json.dumps(base)); malformed[field][0]["shape"] = shape
            with self.subTest(field=field), self.assertRaises(ValueError):
                _validate_source_tflite_description(malformed)
        malformed = json.loads(json.dumps(base))
        malformed["outputs"][0]["quantization"]["scales"] = []
        with self.assertRaises(ValueError):
            _validate_source_tflite_description(malformed)


class ESP32FixtureTests(unittest.TestCase):
    def test_fixture_staging_preserves_logical_shape_and_order(self) -> None:
        from crest.microcontrollers.esp32_fixtures import stage_fixture_header
        from crest.hardware import _quantize_tflite_tensor  # load before substituting the interpreter module
        del _quantize_tflite_tensor

        class FakeInterpreter:
            input_shape = [1, 2, 2]
            output_shape = [1, 3]
            def __init__(self, *, model_path: str) -> None:
                self.model_path = model_path
            def allocate_tensors(self) -> None:
                pass
            def get_input_details(self):
                return [{"shape": np.array(self.input_shape), "dtype": np.int8, "quantization": (0.5, -1)}]
            def get_output_details(self):
                return [{"shape": np.array(self.output_shape), "dtype": np.int8}]

        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            npz = root / "fixtures.npz"
            model = root / "model.tflite"
            manifest = root / "fixtures.json"
            header = root / "fixtures.h"
            inputs = np.array([[[0.0, 0.5], [1.0, -0.5]], [[1.5, 2.0], [-1.0, -1.5]]], dtype=np.float32)
            ids = np.array(["first", "second"])
            np.savez(npz, inputs=inputs, ids=ids)
            model.write_bytes(b"fake")
            manifest.write_text(json.dumps({
                "npz_sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
                "inputs_shape": [2, 2, 2], "inputs_dtype": "float32",
                "ids": ["first", "second"],
            }))
            fake_tf = types.SimpleNamespace(lite=types.SimpleNamespace(Interpreter=FakeInterpreter))
            with patch.dict(sys.modules, {"tensorflow": fake_tf}):
                evidence = stage_fixture_header(
                    fixture_npz_path=npz, fixture_manifest_path=manifest,
                    tflite_model_path=model, output_path=header,
                )
            generated = header.read_text()
            self.assertEqual(evidence["fixture_ids"], ["first", "second"])
            self.assertEqual(evidence["fixture_wire_ids"], ["b64:Zmlyc3Q", "b64:c2Vjb25k"])
            self.assertEqual(evidence["output_elements"], 3)
            self.assertIn("kCrestFixtureCount = 2", generated)
            self.assertIn("{-1,0,1,-2}", generated)
            self.assertIn("b64:Zmlyc3Q", generated)
            for attr, shape in (("input_shape", [2, 2, 2]), ("output_shape", [2, 3])):
                FakeInterpreter.input_shape = [1, 2, 2]
                FakeInterpreter.output_shape = [1, 3]
                setattr(FakeInterpreter, attr, shape)
                with self.subTest(attr=attr), patch.dict(sys.modules, {"tensorflow": fake_tf}), \
                     self.assertRaises(ValueError):
                    stage_fixture_header(
                        fixture_npz_path=npz, fixture_manifest_path=manifest,
                        tflite_model_path=model, output_path=header,
                    )


class ESP32ArtifactIdentityTests(unittest.TestCase):
    def test_generated_source_and_application_must_contain_exact_model(self) -> None:
        from crest.microcontrollers.esp32_artifacts import (
            verify_application_contains_model, verify_generated_model_source,
        )
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            model = root / "model.tflite"
            source = root / "model_data.cc"
            app = root / "app.bin"
            model.write_bytes(b"\x01\x02\xfe")
            source.write_text("const unsigned char x[] = {0x01, 0x02, 0xfe};")
            app.write_bytes(b"prefix" + model.read_bytes() + b"suffix")
            digest = verify_generated_model_source(model_path=model, source_path=source)
            identity = verify_application_contains_model(app_path=app, model_path=model)
            self.assertEqual(identity["embedded_model_sha256"], digest)
            app.write_bytes(model.read_bytes() * 2)
            with self.assertRaises(ValueError):
                verify_application_contains_model(app_path=app, model_path=model)



if __name__ == "__main__":
    unittest.main()
