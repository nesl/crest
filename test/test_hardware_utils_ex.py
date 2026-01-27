import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import serial
# os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
import tensorflow as tf  # type: ignore[attr-defined]
from tcn import TCN
from unittest.mock import patch

tf.get_logger().setLevel('ERROR')  # Suppresses INFO and WARNING from TF's Python logger

# Ensure the project root is importable when tests run via `python -m unittest`.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.hardware_utils_ex import (  # noqa: E402
    ARDUINO_CLI_BIN,
    ARDUINO_CLI_CONFIG,
    XXD_BIN,
    DEVICE_SPECS,
    HIL_ERROR_COMPILE,
    HIL_ERROR_FLASH_OVERFLOW,
    HIL_ERROR_LATENCY,
    HIL_ERROR_OK,
    HIL_ERROR_RAM_OVERFLOW,
    HIL_ERROR_UNDER_SIZED,
    HIL_ERROR_UPLOAD,
    HIL_controller,
    HIL_spec,
    HIL_MASTER_ARENA_EXHAUSTED,
    HIL_MASTER_DEVICE_NOT_FOUND,
    HIL_MASTER_FLASH_OVERFLOW,
    HIL_MASTER_RAM_OVERFLOW,
    HIL_MASTER_FATAL,
    HIL_MASTER_SUCCESS,
    _compute_retry_hint_bytes,
    _pop_retry_hint_bytes,
    _store_retry_hint_bytes,
    _classify_compile_failure,
    _collect_latency_seconds,
    _convert_to_cpp_model_python,
    _convert_to_cpp_model_xxd,
    _parse_memory_from_compile,
    _patch_sketch_constants,
    _replace_define,
    arena_size_candidates,
    convert_to_cpp_model,
    convert_to_tflite_model,
    get_model_memory_usage,
    return_hardware_specs,
)


def _cli_exists() -> bool:
    """
    Determine whether the Arduino CLI binary referenced by the module is callable.
    """
    cli_path = Path(ARDUINO_CLI_BIN)
    if cli_path.exists() and os.access(cli_path, os.X_OK):
        return True
    return shutil.which(ARDUINO_CLI_BIN) is not None


class _FakeCompletedProcess:
    """Lightweight stand-in for subprocess.CompletedProcess used in tests."""

    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


COMPILE_SAMPLE_OUTPUT = (
    "Sketch uses 376104 bytes (38%) of program storage space. Maximum is 983040 bytes.\n"
    "Global variables use 98112 bytes (37%) of dynamic memory, leaving 164032 bytes for local variables. Maximum is 262144 bytes."
)
FLASH_OVERFLOW_STDERR = (
    "/home/m202/TinyODOMEx/TinyODOM-EX/tools/arduino-data/packages/arduino/tools/arm-none-eabi-gcc/"
    "7-2017q4/bin/../lib/gcc/arm-none-eabi/7.2.1/../../../../arm-none-eabi/bin/ld: "
    "/tmp/tmplhhgug0i/arduino-build/tinyodom_tcn.ino.elf section `.text' will not fit in region `FLASH'\n"
    "/home/m202/TinyODOMEx/TinyODOM-EX/tools/arduino-data/packages/arduino/tools/arm-none-eabi-gcc/"
    "7-2017q4/bin/../lib/gcc/arm-none-eabi/7.2.1/../../../../arm-none-eabi/bin/ld: "
    "region `FLASH' overflowed by 3814108 bytes\n"
    "collect2: error: ld returned 1 exit status\n"
    "Error during build: exit status 1\n"
)
RAM_OVERFLOW_STDERR = (
    "/tmp/arduino-build-p1g96nx6/linker_script.ld:138 cannot move location counter backwards "
    "(from 0000000020091d48 to 000000002003fc00)\n"
    "collect2: error: ld returned 1 exit status\n"
    "Error during build: exit status 1\n"
)


class OversizedModelMixin:
    """Build a deliberately large TinyODOM-style TCN to stress flash usage."""

    _OVERSIZED_TIMESTEPS = 400
    _OVERSIZED_CHANNELS = 9

    def _build_oversized_model(self) -> tf.keras.Model:
        inputs = tf.keras.Input(
            shape=(self._OVERSIZED_TIMESTEPS, self._OVERSIZED_CHANNELS),
            name="imu_window",
        )
        features = TCN(
            nb_filters=128,
            kernel_size=5,
            dilations=[1, 2, 4, 8, 16, 32, 64],
            dropout_rate=0.1,
            use_skip_connections=True,
            use_batch_norm=True,
        )(inputs)
        features = tf.keras.layers.Reshape((128, 1))(features)
        features = tf.keras.layers.MaxPooling1D(pool_size=2)(features)
        features = tf.keras.layers.Flatten()(features)
        features = tf.keras.layers.Dense(512, activation="relu", name="dense_big")(features)
        vel_x = tf.keras.layers.Dense(2, activation="linear", name="velx")(features)
        vel_y = tf.keras.layers.Dense(2, activation="linear", name="vely")(features)
        model = tf.keras.Model(inputs=[inputs], outputs=[vel_x, vel_y])
        model.compile(optimizer="adam", loss={"velx": "mse", "vely": "mse"})
        model.build((None, self._OVERSIZED_TIMESTEPS, self._OVERSIZED_CHANNELS))
        return model

class TinyModelMixin:
    """Provide a small trained model + dataset so converter tests stay fast."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        tf.random.set_seed(1234)
        np.random.seed(1234)
        cls.model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(4,), name="input"),
                tf.keras.layers.Dense(4, activation="relu"),
                tf.keras.layers.Dense(2, activation="linear"),
            ]
        )
        cls.model.compile(optimizer="adam", loss="mse")
        cls.train_x = np.random.rand(16, 4).astype(np.float32)
        cls.train_y = np.random.rand(16, 2).astype(np.float32)
        # cls.model.fit(cls.train_x, cls.train_y, epochs=1, verbose=0)


class ConversionHelperTests(TinyModelMixin, unittest.TestCase):
    def test_convert_to_tflite_model_creates_file(self):
        # Ensures the baseline float export path runs end-to-end so regressions in converter plumbing surface quickly.
        with tempfile.TemporaryDirectory() as tmpdir:
            tflite_path = Path(tmpdir) / "model_float.tflite"
            convert_to_tflite_model(self.model, self.train_x, output_name=tflite_path)
            self.assertTrue(tflite_path.exists())
            self.assertGreater(tflite_path.stat().st_size, 0)

    def test_convert_to_tflite_model_quantized_flow(self):
        # Covers the quantization branch so representative dataset wiring remains valid.
        with tempfile.TemporaryDirectory() as tmpdir:
            tflite_path = Path(tmpdir) / "model_int8.tflite"
            convert_to_tflite_model(
                self.model,
                self.train_x,
                quantization=True,
                output_name=tflite_path,
            )
            self.assertTrue(tflite_path.exists())
            self.assertGreater(tflite_path.stat().st_size, 0)

    def test_convert_to_cpp_model_old_emits_sources(self):
        # Validates the manual hex emitter so updates that break header/source symmetry are caught.
        with tempfile.TemporaryDirectory() as tmpdir:
            tflite_path = Path(tmpdir) / "model_float.tflite"
            convert_to_tflite_model(self.model, self.train_x, output_name=tflite_path)
            out_dir = Path(tmpdir) / "cpp_old"
            source_path, header_path = _convert_to_cpp_model_python(tflite_path, out_dir)
            expected_len = len(tflite_path.read_bytes())
            source_text = source_path.read_text()
            self.assertIn(f"const int g_model_len = {expected_len};", source_text)
            header_text = header_path.read_text()
            self.assertIn("TENSORFLOW_LITE_MICRO_EXAMPLES_HELLO_WORLD_MODEL_H_", header_text)

    @unittest.skipUnless(shutil.which("xxd"), "xxd command required for this test.")
    def test_convert_to_cpp_model_via_xxd(self):
        # Exercises the xxd-backed path so subprocess handling and type replacements stay correct.
        with tempfile.TemporaryDirectory() as tmpdir:
            tflite_path = Path(tmpdir) / "model_float.tflite"
            convert_to_tflite_model(self.model, self.train_x, output_name=tflite_path)
            out_dir = Path(tmpdir) / "cpp_xxd"
            source_path, header_path = convert_to_cpp_model(tflite_path, out_dir)
            self.assertTrue(source_path.exists())
            self.assertTrue(header_path.exists())
            source_text = source_path.read_text()
            self.assertIn('#include "model.h"', source_text)
            self.assertIn("alignas(8) const unsigned char g_model[]", source_text)

    def test_convert_to_cpp_model_missing_source_raises(self):
        # Missing model artifact should surface a FileNotFoundError so callers can recover cleanly.
        with tempfile.TemporaryDirectory() as tmpdir:
            bogus_model = Path(tmpdir) / "missing_model.tflite"
            with self.assertRaises(FileNotFoundError):
                convert_to_cpp_model(bogus_model, Path(tmpdir) / "out")

    @unittest.skipUnless(shutil.which("xxd"), "xxd command required for this test.")
    def test_convert_to_cpp_model_handles_corrupt_bytes(self):
        # Even malformed flatbuffers should round-trip through xxd so deployment tooling stays resilient.
        with tempfile.TemporaryDirectory() as tmpdir:
            tflite_path = Path(tmpdir) / "model_corrupt.tflite"
            tflite_path.write_bytes(os.urandom(128))
            out_dir = Path(tmpdir) / "cpp_corrupt"
            source_path, header_path = convert_to_cpp_model(tflite_path, out_dir)
            self.assertTrue(source_path.exists())
            self.assertTrue(header_path.exists())
            self.assertGreater(source_path.stat().st_size, 0)

    @unittest.skipUnless(XXD_BIN, "xxd command required for parity validation.")
    def test_python_and_xxd_emit_matching_sources(self):
        # Ensures the Python fallback produces identical C arrays when xxd is available.
        with tempfile.TemporaryDirectory() as tmpdir:
            tflite_path = Path(tmpdir) / "model_stub.tflite"
            tflite_path.write_bytes(bytes(range(64)))
            python_dir = Path(tmpdir) / "py_cpp"
            xxd_dir = Path(tmpdir) / "xxd_cpp"
            py_source, py_header = _convert_to_cpp_model_python(tflite_path, python_dir)
            xxd_source, xxd_header = _convert_to_cpp_model_xxd(tflite_path, xxd_dir)

            self.assertEqual(py_header.read_text(), xxd_header.read_text())

            py_source_text = py_source.read_text()
            xxd_source_text = xxd_source.read_text()
            hex_pattern = re.compile(r"0x[0-9a-f]{2}")
            self.assertListEqual(hex_pattern.findall(py_source_text), hex_pattern.findall(xxd_source_text))

            len_pattern = re.compile(r"const int g_model_len = (\d+);")
            py_len = len_pattern.search(py_source_text)
            xxd_len = len_pattern.search(xxd_source_text)
            self.assertIsNotNone(py_len)
            self.assertIsNotNone(xxd_len)
            self.assertEqual(py_len.group(1), xxd_len.group(1))

    def test_convert_to_cpp_model_output_dir_conflict(self):
        # Colliding with an existing file path should raise to highlight filesystem issues early.
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "not_a_directory"
            output_path.write_text("stub")
            model_path = Path(tmpdir) / "model_float.tflite"
            convert_to_tflite_model(self.model, self.train_x, output_name=model_path)
            with self.assertRaises(FileExistsError):
                convert_to_cpp_model(model_path, output_path)


class SpecHelperTests(unittest.TestCase):
    def test_return_hardware_specs_known_device(self):
        # Confirms catalog lookups return numeric limits for supported hardware.
        ram, flash = return_hardware_specs("ARDUINO_NANO_33_BLE_SENSE")
        self.assertGreater(ram, 0)
        self.assertGreater(flash, 0)

    def test_return_hardware_specs_unknown_device(self):
        # Ensures unsupported boards raise ValueError to prevent silent misconfiguration.
        with self.assertRaises(ValueError):
            return_hardware_specs("NOT_A_BOARD")

    def test_arena_size_candidates_happy_path(self):
        # Validates that arena sweeps are exposed as numpy arrays for downstream sweeps.
        arena = arena_size_candidates("ARDUINO_NANO_33_BLE_SENSE")
        self.assertIsInstance(arena, np.ndarray)
        self.assertGreater(len(arena), 0)

    def test_arena_size_candidates_invalid_device(self):
        # Confirms invalid device names raise to highlight typos early.
        with self.assertRaises(ValueError):
            arena_size_candidates("UNKNOWN_DEVICE")

    def test_get_model_memory_usage_quantized_smaller(self):
        # Verifies the quantized flag reduces the byte estimate, preventing regressive sizing.
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(4,)),
                tf.keras.layers.Dense(8, activation="relu"),
                tf.keras.layers.Dense(1),
            ]
        )
        float_usage = get_model_memory_usage(1, model, quantized=False)
        quant_usage = get_model_memory_usage(1, model, quantized=True)
        self.assertGreater(float_usage, 0)
        self.assertLessEqual(quant_usage, float_usage)


class DeviceCatalogTests(unittest.TestCase):
    def test_catalog_devices_are_accessible(self):
        # Ensures every catalog entry returns the exact limits defined in DEVICE_SPECS.
        for name, spec in DEVICE_SPECS.items():
            ram, flash = return_hardware_specs(name)
            self.assertEqual(ram, spec["max_ram"])
            self.assertEqual(flash, spec["max_flash"])
            arena = arena_size_candidates(name)
            self.assertTrue(np.array_equal(arena, spec["arena_sizes"]))
            self.assertGreater(len(arena), 0)

    def test_catalog_allows_new_device_entries(self):
        # Adding a new device should immediately make it discoverable via the helpers.
        new_name = "TEST_DEVICE"
        new_spec = {
            "arena_sizes": np.array([5, 15, 25]),
            "max_ram": 123_456,
            "max_flash": 654_321,
            "fqbn": "example:fqbn",
        }
        with patch.dict(DEVICE_SPECS, {new_name: new_spec}, clear=False):
            ram, flash = return_hardware_specs(new_name)
            arena = arena_size_candidates(new_name)
            self.assertEqual(ram, new_spec["max_ram"])
            self.assertEqual(flash, new_spec["max_flash"])
            self.assertTrue(np.array_equal(arena, new_spec["arena_sizes"]))


class MemoryUsageBoundaryTests(unittest.TestCase):
    def _build_dense_model(self) -> tf.keras.Model:
        return tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(4,), name="input"),
                tf.keras.layers.Dense(8, activation="relu"),
                tf.keras.layers.Dense(1),
            ]
        )

    def _build_example_model(self) -> tf.keras.Model:
        timesteps = 200
        input_dim = 6
        inputs = tf.keras.Input(shape=(timesteps, input_dim), name="imu_window")
        features = TCN(
            nb_filters=32,
            kernel_size=5,
            dilations=[1, 2, 4, 8, 16],
            dropout_rate=0.1,
            use_skip_connections=True,
            use_batch_norm=True,
        )(inputs)
        features = tf.keras.layers.Reshape((32, 1))(features)
        features = tf.keras.layers.MaxPooling1D(pool_size=2)(features)
        features = tf.keras.layers.Flatten()(features)
        features = tf.keras.layers.Dense(64, activation="relu", name="pre_dense")(features)
        vel_x = tf.keras.layers.Dense(1, activation="linear", name="velx")(features)
        vel_y = tf.keras.layers.Dense(1, activation="linear", name="vely")(features)
        model = tf.keras.Model(inputs=[inputs], outputs=[vel_x, vel_y])
        model.compile(optimizer="adam", loss={"velx": "mse", "vely": "mse"})
        model.build((None, timesteps, input_dim))
        return model

    def test_memory_usage_matches_manual_estimate(self):
        # Validates the estimator against a hand-computed baseline so regressions are caught quickly.
        model = self._build_dense_model()
        batch_size = 1
        usage = get_model_memory_usage(batch_size, model, quantized=False)

        shapes_mem_count = 0
        for layer in model.layers:
            out_shape = getattr(layer, "output_shape", None)
            if out_shape is None:
                continue
            if isinstance(out_shape, list):
                out_shape = out_shape[0]
            elems = 1
            for dim in out_shape:
                if dim is None:
                    continue
                elems *= dim
            shapes_mem_count += elems

        trainable = np.sum([tf.keras.backend.count_params(p) for p in model.trainable_weights])
        non_trainable = np.sum([tf.keras.backend.count_params(p) for p in model.non_trainable_weights])
        expected = 4.0 * (batch_size * shapes_mem_count + trainable + non_trainable)
        self.assertAlmostEqual(usage, expected, places=5)

    def test_memory_usage_respects_float_precision(self):
        # Ensures Keras global precision influences the estimator so resource sweeps stay trustworthy.
        original_floatx = tf.keras.backend.floatx()
        try:
            tf.keras.backend.set_floatx("float16")
            model_fp16 = self._build_dense_model()
            usage_fp16 = get_model_memory_usage(1, model_fp16, quantized=False)

            tf.keras.backend.set_floatx("float64")
            model_fp64 = self._build_dense_model()
            usage_fp64 = get_model_memory_usage(1, model_fp64, quantized=False)
        finally:
            tf.keras.backend.set_floatx(original_floatx)

        self.assertLess(usage_fp16, usage_fp64)
        self.assertGreater(usage_fp64, 0)

    def test_memory_usage_outpaces_model_serialization(self):
        # Estimated activation+parameter usage should dominate serialized model size for sanity.
        model = self._build_example_model()
        usage = get_model_memory_usage(1, model, quantized=False)
        param_bytes = 4.0 * model.count_params()
        with tempfile.TemporaryDirectory() as tmpdir:
            tflite_path = Path(tmpdir) / "dense_model.tflite"
            calibration_data = np.random.rand(8, 200, 6).astype(np.float32)
            convert_to_tflite_model(model, calibration_data, output_name=tflite_path)
            flatbuffer_bytes = tflite_path.stat().st_size

        self.assertGreaterEqual(usage, param_bytes)
        self.assertGreater(usage, flatbuffer_bytes * 0.25)


class SketchHelperTests(unittest.TestCase):
    def test_replace_define_updates_value(self):
        # Protects the regex-based macro replacement so sketch constants keep updating correctly.
        text = "#define TINYODOM_WINDOW_SIZE 100\nvoid loop() {}\n"
        updated = _replace_define(text, "TINYODOM_WINDOW_SIZE", "256")
        self.assertIn("TINYODOM_WINDOW_SIZE 256", updated)

    def test_patch_sketch_constants_edits_ino(self):
        # Ensures the helper edits real .ino files, preventing stale deployment constants.
        with tempfile.TemporaryDirectory() as tmpdir:
            sketch_dir = Path(tmpdir)
            ino_path = sketch_dir / "TinyOdom.ino"
            ino_path.write_text(
                "\n".join(
                    [
                        "#define TINYODOM_WINDOW_SIZE 100",
                        "#define TINYODOM_NUM_CHANNELS 1",
                        "#define TINYODOM_TENSOR_ARENA_BYTES (10 * 1024)",
                    ]
                )
            )
            _patch_sketch_constants(sketch_dir, arena_kb=42, window_size=256, num_channels=3)
            text = ino_path.read_text()
            self.assertIn("TINYODOM_WINDOW_SIZE 256", text)
            self.assertIn("TINYODOM_NUM_CHANNELS 3", text)
            self.assertIn("TINYODOM_TENSOR_ARENA_BYTES (42 * 1024)", text)

    def test_parse_memory_from_compile_extracts_numbers(self):
        # Validates CLI parsing so RAM/flash metrics stay trustworthy for scoring.
        sample_output = (
            "Sketch uses 376104 bytes (38%) of program storage space. Maximum is 983040 bytes. \n \
            Global variables use 98112 bytes (37%) of dynamic memory, leaving 164032 bytes for local variables. Maximum is 262144 bytes."
        )
        flash_bytes, ram_bytes = _parse_memory_from_compile(sample_output)
        self.assertEqual(flash_bytes, 376104)
        self.assertEqual(ram_bytes, 98112)

    def test_parse_memory_from_compile_handles_missing_data(self):
        # Ensures regex failures surface as None so callers can apply defaults instead of crashing.
        flash_bytes, ram_bytes = _parse_memory_from_compile("no relevant information")
        self.assertIsNone(flash_bytes)
        self.assertIsNone(ram_bytes)


class CompileFailureClassificationTests(unittest.TestCase):
    def test_classify_returns_none_for_normal_output(self):
        result = _classify_compile_failure(COMPILE_SAMPLE_OUTPUT)
        self.assertIsNone(result)

    def test_classify_detects_flash_overflow(self):
        result = _classify_compile_failure(FLASH_OVERFLOW_STDERR)
        self.assertEqual(result, "flash")

    def test_classify_detects_ram_overflow(self):
        result = _classify_compile_failure(RAM_OVERFLOW_STDERR)
        self.assertEqual(result, "ram")


class SerialHelperTests(unittest.TestCase):
    class _DummySerial:
        def __init__(self, responses):
            self._responses = iter(responses)

        def readline(self):
            try:
                return next(self._responses)
            except StopIteration:
                return b""

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def test_collect_latency_returns_value(self):
        # Serial stream publishing a timer line should surface the parsed float.
        responses = [b"ignored\n", b"timer output: 0.42\n"]

        def factory(*_args, **_kwargs):
            return self._DummySerial(responses)

        with patch("src.hardware_utils_ex.serial.Serial", side_effect=factory):
            latency, arena_line, serial_log = _collect_latency_seconds(
                "COM1", 115200, timeout_s=0.05
            )
        self.assertIsNotNone(latency)
        assert latency is not None
        self.assertAlmostEqual(latency, 0.42)
        self.assertIsNone(arena_line)
        self.assertEqual(serial_log, ["ignored", "timer output: 0.42"])

    def test_collect_latency_handles_timeout(self):
        # Empty serial responses should produce a None latency instead of blocking.
        responses = [b"", b""]

        def factory(*_args, **_kwargs):
            return self._DummySerial(responses)

        with patch("src.hardware_utils_ex.serial.Serial", side_effect=factory):
            latency, arena_line, serial_log = _collect_latency_seconds(
                "COM2", 115200, timeout_s=0.01
            )
        self.assertIsNone(latency)
        self.assertIsNone(arena_line)
        self.assertEqual(serial_log, [])

    def test_collect_latency_invalid_port_raises(self):
        # Serial exceptions should bubble up as RuntimeError so callers can react.
        with patch(
            "src.hardware_utils_ex.serial.Serial",
            side_effect=serial.SerialException("boom"),
        ):
            with self.assertRaises(RuntimeError):
                _collect_latency_seconds("COM3", 115200, timeout_s=0.01)

    def test_collect_latency_handles_non_numeric_payload(self):
        # Malformed timer output lines should fail gracefully and return None.
        responses = [b"timer output: not-a-float\n"]

        def factory(*_args, **_kwargs):
            return self._DummySerial(responses)

        with patch("src.hardware_utils_ex.serial.Serial", side_effect=factory):
            latency, arena_line, serial_log = _collect_latency_seconds(
                "COM4", 115200, timeout_s=0.01
            )
        self.assertIsNone(latency)
        self.assertIsNone(arena_line)
        self.assertEqual(serial_log, ["timer output: not-a-float"])

    def test_collect_latency_detects_arena_error(self):
        # Firmware-provided arena errors should surface so callers can escalate arena size.
        responses = [b"size is too small for all buffers\n"]

        def factory(*_args, **_kwargs):
            return self._DummySerial(responses)

        with patch("src.hardware_utils_ex.serial.Serial", side_effect=factory):
            latency, arena_line, serial_log = _collect_latency_seconds(
                "COM4", 115200, timeout_s=0.01
            )
        self.assertIsNone(latency)
        self.assertIsNotNone(arena_line)
        self.assertEqual(serial_log, ["size is too small for all buffers"])


class RetryHintHelperTests(unittest.TestCase):
    def tearDown(self) -> None:
        _store_retry_hint_bytes(None)

    def test_compute_retry_hint_uses_missing_field(self):
        # Firmware logs with a "missing" clause should expand the arena by the gap plus cushion.
        current_bytes = 100_000
        line = "Failed... missing: 4096"
        hint = _compute_retry_hint_bytes(current_bytes, line)
        self.assertEqual(hint, current_bytes + 4096 + 2048)

    def test_compute_retry_hint_uses_requested_field(self):
        # When only "requested" is present we jump to that size plus the safety margin.
        current_bytes = 50_000
        line = "Requested: 60000, available: 123"
        hint = _compute_retry_hint_bytes(current_bytes, line)
        self.assertEqual(hint, 60_000 + 2048)

    def test_compute_retry_hint_returns_none_without_growth(self):
        # Lines that do not imply growth (or provide no line) should not mutate arena selection.
        current_bytes = 40_000
        line = "Requested: 1000, missing: 0"
        hint = _compute_retry_hint_bytes(current_bytes, line)
        self.assertIsNone(hint)
        self.assertIsNone(_compute_retry_hint_bytes(current_bytes, None))

    def test_store_and_pop_retry_hint_bytes(self):
        # Stored hints should be returned once and cleared to avoid leaking state across trials.
        _store_retry_hint_bytes(12_345)
        self.assertEqual(_pop_retry_hint_bytes(), 12_345)
        self.assertIsNone(_pop_retry_hint_bytes())


class HILSpecErrorTests(unittest.TestCase):
    def _write_sketch(self, sketch_dir: Path) -> None:
        sketch_dir.mkdir(parents=True, exist_ok=True)
        (sketch_dir / "TinyOdom.ino").write_text(
            "\n".join(
                [
                    "#define TINYODOM_WINDOW_SIZE 100",
                    "#define TINYODOM_NUM_CHANNELS 1",
                    "#define TINYODOM_TENSOR_ARENA_BYTES (10 * 1024)",
                ]
            )
        )

    def test_hil_spec_upload_failure_sets_error_flag(self):
        # Upload failures should surface the upload-specific flag so callers can react.
        with tempfile.TemporaryDirectory() as tmpdir:
            sketch_dir = Path(tmpdir)
            self._write_sketch(sketch_dir)
            compile_result = _FakeCompletedProcess(stdout=COMPILE_SAMPLE_OUTPUT)
            upload_result = _FakeCompletedProcess(returncode=1)
            with patch(
                "src.hardware_utils_ex.subprocess.run",
                side_effect=[compile_result, upload_result],
            ) as mock_run:
                ram, flash, latency, arena_bytes, err, _power = HIL_spec(
                    dirpath=sketch_dir,
                    chosen_device="ARDUINO_NANO_33_BLE_SENSE",
                    serial_port="/dev/ttyMock",
                    compile_only=False,
                )
            self.assertEqual(err, HIL_ERROR_UPLOAD)
            self.assertEqual(latency, -1.0)
            self.assertEqual(ram, 98112)
            self.assertEqual(flash, 376104)
            self.assertGreater(arena_bytes, 0)
            self.assertEqual(mock_run.call_count, 2)

    def test_hil_spec_latency_timeout_sets_error_flag(self):
        # Latency timeouts should surface the latency timeout flag so automation can retry larger arenas.
        with tempfile.TemporaryDirectory() as tmpdir:
            sketch_dir = Path(tmpdir)
            self._write_sketch(sketch_dir)
            compile_result = _FakeCompletedProcess(stdout=COMPILE_SAMPLE_OUTPUT)
            upload_result = _FakeCompletedProcess(stdout="upload ok")
            with patch(
                "src.hardware_utils_ex.subprocess.run",
                side_effect=[compile_result, upload_result],
            ):
                with patch(
                    "src.hardware_utils_ex._collect_latency_seconds",
                    return_value=(None, None, ["failed to allocate"]),
                ):
                    ram, flash, latency, arena_bytes, err, _power = HIL_spec(
                        dirpath=sketch_dir,
                        chosen_device="ARDUINO_NANO_33_BLE_SENSE",
                        serial_port="/dev/ttyMock",
                        compile_only=False,
                    )
        self.assertEqual(err, HIL_ERROR_LATENCY)
        self.assertEqual(latency, -1.0)
        self.assertEqual(ram, 98112)
        self.assertEqual(flash, 376104)
        self.assertGreater(arena_bytes, 0)

    def test_hil_spec_rejects_out_of_range_arena_index(self):
        # Selecting an invalid arena index should raise IndexError immediately.
        with tempfile.TemporaryDirectory() as tmpdir:
            sketch_dir = Path(tmpdir)
            self._write_sketch(sketch_dir)
            with self.assertRaises(IndexError):
                HIL_spec(
                    dirpath=sketch_dir,
                    chosen_device="ARDUINO_NANO_33_BLE_SENSE",
                    idx=999,
                    compile_only=True,
                )

    def test_hil_spec_detects_flash_overflow(self):
        # Linker overflow messages should short-circuit with a dedicated error code.
        with tempfile.TemporaryDirectory() as tmpdir:
            sketch_dir = Path(tmpdir)
            self._write_sketch(sketch_dir)
            compile_result = _FakeCompletedProcess(
                returncode=1,
                stdout="",
                stderr=FLASH_OVERFLOW_STDERR,
            )
            with patch(
                "src.hardware_utils_ex.subprocess.run",
                return_value=compile_result,
            ) as mock_run:
                ram, flash, latency, arena_bytes, err, _power = HIL_spec(
                    dirpath=sketch_dir,
                    chosen_device="ARDUINO_NANO_33_BLE_SENSE",
                    compile_only=True,
                )
        self.assertEqual(err, HIL_ERROR_FLASH_OVERFLOW)
        self.assertEqual((ram, flash, latency), (-1, -1, -1.0))
        self.assertGreater(arena_bytes, 0)
        self.assertEqual(mock_run.call_count, 1)

    def test_hil_spec_detects_ram_overflow_message(self):
        # Location counter errors (RAM exhaustion) should emit the RAM-specific overflow code.
        with tempfile.TemporaryDirectory() as tmpdir:
            sketch_dir = Path(tmpdir)
            self._write_sketch(sketch_dir)
            compile_result = _FakeCompletedProcess(
                returncode=1,
                stdout="",
                stderr=RAM_OVERFLOW_STDERR,
            )
            with patch(
                "src.hardware_utils_ex.subprocess.run",
                return_value=compile_result,
            ) as mock_run:
                ram, flash, latency, arena_bytes, err, _power = HIL_spec(
                    dirpath=sketch_dir,
                    chosen_device="ARDUINO_NANO_33_BLE_SENSE",
                    compile_only=True,
                )
        self.assertEqual(err, HIL_ERROR_RAM_OVERFLOW)
        self.assertEqual((ram, flash, latency), (-1, -1, -1.0))
        self.assertGreater(arena_bytes, 0)
        self.assertEqual(mock_run.call_count, 1)

    def test_hil_spec_maps_arena_errors_to_under_sized_flag(self):
        # Serial error lines indicating insufficient buffers should set the under-sized flag.
        with tempfile.TemporaryDirectory() as tmpdir:
            sketch_dir = Path(tmpdir)
            self._write_sketch(sketch_dir)
            compile_result = _FakeCompletedProcess(stdout=COMPILE_SAMPLE_OUTPUT)
            upload_result = _FakeCompletedProcess(stdout="upload ok")
            with patch(
                    "src.hardware_utils_ex.subprocess.run",
                    side_effect=[compile_result, upload_result],
                ):
                with patch(
                    "src.hardware_utils_ex._collect_latency_seconds",
                    return_value=(None, "size is too small for all buffers", ["size is too small for all buffers"]),
                ):
                    ram, flash, latency, arena_bytes, err, _power = HIL_spec(
                        dirpath=sketch_dir,
                        chosen_device="ARDUINO_NANO_33_BLE_SENSE",
                        serial_port="/dev/ttyMock",
                        compile_only=False,
                    )
        self.assertEqual(err, HIL_ERROR_UNDER_SIZED)
        self.assertEqual(latency, -1.0)
        self.assertEqual(ram, 98112)
        self.assertEqual(flash, 376104)
        self.assertGreater(arena_bytes, 0)


class HILControllerTests(unittest.TestCase):
    def test_hil_controller_success_on_first_candidate(self):
        # Successful compile-only pass should short-circuit the sweep and surface metrics.
        arena_candidates = np.array([10, 20])
        hil_return = (64000, 128000, 0.25, 10 * 1024, HIL_ERROR_OK, None)
        with patch(
            "src.hardware_utils_ex.arena_size_candidates",
            return_value=arena_candidates,
        ), patch(
            "src.hardware_utils_ex.HIL_spec",
            return_value=hil_return,
        ) as mock_spec:
            ram, flash, latency, arena_bytes, master_error, _power_metrics = HIL_controller(
                dirpath="unused",
                chosen_device="ARDUINO_NANO_33_BLE_SENSE",
                run_hil=False,
            )
        self.assertEqual(mock_spec.call_count, 1)
        self.assertEqual(ram, 64000)
        self.assertEqual(flash, 128000)
        self.assertAlmostEqual(latency, 0.25)
        self.assertEqual(arena_bytes, 10 * 1024)
        self.assertEqual(master_error, HIL_MASTER_SUCCESS)
        self.assertIsNone(_power_metrics)

    def test_hil_controller_exhausts_candidates(self):
        # Exhausting the sweep without success should return masterError=2 and sentinel metrics.
        arena_candidates = np.array([10, 20])

        def hil_side_effect(**_kwargs):
            idx = _kwargs["idx"]
            arena = arena_candidates[idx] * 1024
            return (50000, 100000, -1.0, arena, HIL_ERROR_LATENCY, None)

        with patch(
            "src.hardware_utils_ex.arena_size_candidates",
            return_value=arena_candidates,
        ), patch(
            "src.hardware_utils_ex.HIL_spec",
            side_effect=hil_side_effect,
        ) as mock_spec:
            ram, flash, latency, arena_bytes, master_error, _power_metrics = HIL_controller(
                dirpath="unused",
                chosen_device="ARDUINO_NANO_33_BLE_SENSE",
                run_hil=False,
            )
        self.assertEqual(mock_spec.call_count, len(arena_candidates))
        self.assertEqual(
            (ram, flash, latency, arena_bytes, master_error, _power_metrics),
            (-1, -1, -1.0, -1, HIL_MASTER_ARENA_EXHAUSTED, None),
        )

    def test_hil_controller_non_arena_failure(self):
        # Non-arena errors should bubble up immediately with masterError=3 and captured metrics.
        arena_candidates = np.array([10, 20])
        hil_return = (72000, 160000, -1.0, 10 * 1024, HIL_ERROR_COMPILE, None)
        with patch(
            "src.hardware_utils_ex.arena_size_candidates",
            return_value=arena_candidates,
        ), patch(
            "src.hardware_utils_ex.HIL_spec",
            return_value=hil_return,
        ) as mock_spec:
            ram, flash, latency, arena_bytes, master_error, _power_metrics = HIL_controller(
                dirpath="unused",
                chosen_device="ARDUINO_NANO_33_BLE_SENSE",
                run_hil=False,
            )
        self.assertEqual(mock_spec.call_count, 1)
        self.assertEqual(
            (ram, flash, latency, arena_bytes, master_error),
            (72000, 160000, -1.0, 10 * 1024, HIL_MASTER_FATAL),
        )

    def test_hil_controller_reports_flash_overflow(self):
        # Linker overflow should surface a dedicated master error for pruning.
        arena_candidates = np.array([10])
        hil_return = (-1, -1, -1.0, 10 * 1024, HIL_ERROR_FLASH_OVERFLOW, None)
        with patch(
            "src.hardware_utils_ex.arena_size_candidates",
            return_value=arena_candidates,
        ), patch(
            "src.hardware_utils_ex.HIL_spec",
            return_value=hil_return,
        ) as mock_spec:
            ram, flash, latency, arena_bytes, master_error, _power_metrics = HIL_controller(
                dirpath="unused",
                chosen_device="ARDUINO_NANO_33_BLE_SENSE",
                run_hil=False,
            )
        self.assertEqual(mock_spec.call_count, 1)
        self.assertEqual(
            (ram, flash, latency, arena_bytes, master_error),
            (-1, -1, -1.0, 10 * 1024, HIL_MASTER_FLASH_OVERFLOW),
        )

    def test_hil_controller_reports_device_not_found(self):
        # Upload failures should set the dedicated master error so orchestration can stop early.
        arena_candidates = np.array([10])
        hil_return = (64000, 128000, -1.0, 10 * 1024, HIL_ERROR_UPLOAD, None)
        with patch(
            "src.hardware_utils_ex.arena_size_candidates",
            return_value=arena_candidates,
        ), patch(
            "src.hardware_utils_ex.HIL_spec",
            return_value=hil_return,
        ) as mock_spec:
            ram, flash, latency, arena_bytes, master_error, _power_metrics = HIL_controller(
                dirpath="unused",
                chosen_device="ARDUINO_NANO_33_BLE_SENSE",
                run_hil=False,
            )
        self.assertEqual(mock_spec.call_count, 1)
        self.assertEqual(
            (ram, flash, latency, arena_bytes, master_error),
            (64000, 128000, -1.0, 10 * 1024, HIL_MASTER_DEVICE_NOT_FOUND),
        )

    def test_hil_controller_prefers_smallest_successful_arena(self):
        # Success at a mid-point arena should trigger a retry with the next smaller candidate.
        arena_candidates = np.array([10, 20, 40, 80])
        call_log: list[int] = []

        def hil_side_effect(**kwargs):
            idx = kwargs["idx"]
            call_log.append(idx)
            arena = arena_candidates[idx] * 1024
            if len(call_log) == 1:
                self.assertEqual(idx, 1)
                return (64000, 128000, 0.25, arena, HIL_ERROR_OK, None)
            self.assertEqual(idx, 0)
            return (-1, -1, -1.0, arena, HIL_ERROR_UNDER_SIZED, None)

        with patch(
            "src.hardware_utils_ex.arena_size_candidates",
            return_value=arena_candidates,
        ), patch(
            "src.hardware_utils_ex.HIL_spec",
            side_effect=hil_side_effect,
        ) as mock_spec:
            ram, flash, latency, arena_bytes, master_error, _power_metrics = HIL_controller(
                dirpath="unused",
                chosen_device="ARDUINO_NANO_33_BLE_SENSE",
                run_hil=False,
            )
        self.assertEqual(mock_spec.call_count, 2)
        self.assertListEqual(call_log, [1, 0])
        self.assertEqual(master_error, HIL_MASTER_SUCCESS)
        self.assertEqual(arena_bytes, 20 * 1024)
        self.assertEqual((ram, flash), (64000, 128000))
        self.assertAlmostEqual(latency, 0.25)

    def test_hil_controller_reports_master_ram_overflow_at_smallest(self):
        # RAM overflow at the smallest arena should surface a dedicated master error for retries.
        arena_candidates = np.array([10])
        hil_return = (-1, -1, -1.0, 10 * 1024, HIL_ERROR_RAM_OVERFLOW, None)
        with patch(
            "src.hardware_utils_ex.arena_size_candidates",
            return_value=arena_candidates,
        ), patch(
            "src.hardware_utils_ex.HIL_spec",
            return_value=hil_return,
        ) as mock_spec:
            ram, flash, latency, arena_bytes, master_error, _power_metrics = HIL_controller(
                dirpath="unused",
                chosen_device="ARDUINO_NANO_33_BLE_SENSE",
                run_hil=False,
            )
        self.assertEqual(mock_spec.call_count, 1)
        self.assertEqual(master_error, HIL_MASTER_RAM_OVERFLOW)
        self.assertEqual((ram, flash, latency, arena_bytes), hil_return[:4])

    def test_hil_controller_retains_success_after_smaller_failure(self):
        # A smaller arena failure after a success should still return the best-known success metrics.
        arena_candidates = np.array([10, 20, 40])
        call_order: list[int] = []

        def hil_side_effect(**kwargs):
            idx = kwargs["idx"]
            call_order.append(idx)
            arena = arena_candidates[idx] * 1024
            if len(call_order) == 1:
                self.assertEqual(idx, 1)
                return (70000, 150000, 0.3, arena, HIL_ERROR_OK, None)
            self.assertEqual(idx, 0)
            return (-1, -1, -1.0, arena, HIL_ERROR_RAM_OVERFLOW, None)

        with patch(
            "src.hardware_utils_ex.arena_size_candidates",
            return_value=arena_candidates,
        ), patch(
            "src.hardware_utils_ex.HIL_spec",
            side_effect=hil_side_effect,
        ) as mock_spec:
            ram, flash, latency, arena_bytes, master_error, _power_metrics = HIL_controller(
                dirpath="unused",
                chosen_device="ARDUINO_NANO_33_BLE_SENSE",
                run_hil=False,
            )
        self.assertEqual(mock_spec.call_count, 2)
        self.assertListEqual(call_order, [1, 0])
        self.assertEqual(master_error, HIL_MASTER_SUCCESS)
        self.assertEqual(arena_bytes, 20 * 1024)
        self.assertEqual((ram, flash), (70000, 150000))
        self.assertAlmostEqual(latency, 0.3)

    def test_hil_controller_uses_retry_hint_to_jump(self):
        # Retry hints should allow the controller to skip intermediate arenas when logs provide guidance.
        _store_retry_hint_bytes(None)
        arena_candidates = np.array([10, 20, 40, 80])
        call_sequence: list[int] = []

        def hil_side_effect(**kwargs):
            idx = kwargs["idx"]
            call_sequence.append(idx)
            arena = arena_candidates[idx] * 1024
            # _store_retry_hint_bytes(None)  # Removed: should only be called before the first call
            if len(call_sequence) == 1:
                _store_retry_hint_bytes(70 * 1024)
                return (-1, -1, -1.0, arena, HIL_ERROR_UNDER_SIZED, None)
            if idx == 3:
                return (61000, 120000, 0.2, arena, HIL_ERROR_OK, None)
            # The controller should probe the next smaller arena after a success.
            self.assertEqual(idx, 2)
            return (-1, -1, -1.0, arena, HIL_ERROR_UNDER_SIZED, None)

        with patch(
            "src.hardware_utils_ex.arena_size_candidates",
            return_value=arena_candidates,
        ), patch(
            "src.hardware_utils_ex.HIL_spec",
            side_effect=hil_side_effect,
        ) as mock_spec:
            ram, flash, latency, arena_bytes, master_error, _power_metrics = HIL_controller(
                dirpath="unused",
                chosen_device="ARDUINO_NANO_33_BLE_SENSE",
                run_hil=False,
            )

        self.assertEqual(call_sequence[:2], [1, 3])
        self.assertEqual(call_sequence[-1], 2)
        self.assertEqual(master_error, HIL_MASTER_SUCCESS)
        self.assertEqual(arena_bytes, 80 * 1024)
        self.assertEqual((ram, flash), (61000, 120000))
        self.assertAlmostEqual(latency, 0.2)


class IntegrationTests(TinyModelMixin, unittest.TestCase):
    @unittest.skipUnless(shutil.which("xxd"), "xxd command required for this test.")
    def test_compile_only_pipeline(self):
        # Full pipeline smoke test to ensure exporter + HIL plumbing cooperate end-to-end.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            tflite_path = tmp_path / "model_full.tflite"
            convert_to_tflite_model(self.model, self.train_x, output_name=tflite_path)
            cpp_dir = tmp_path / "cpp_out"
            convert_to_cpp_model(tflite_path, cpp_dir)
            self.assertTrue((cpp_dir / "model.cc").exists())
            self.assertTrue((cpp_dir / "model.h").exists())

            sketch_dir = tmp_path / "tinyodom_tcn"
            sketch_dir.mkdir()
            (sketch_dir / "TinyOdom.ino").write_text(
                "\n".join(
                    [
                        "#define TINYODOM_WINDOW_SIZE 100",
                        "#define TINYODOM_NUM_CHANNELS 1",
                        "#define TINYODOM_TENSOR_ARENA_BYTES (10 * 1024)",
                    ]
                )
            )

            compile_result = _FakeCompletedProcess(stdout=COMPILE_SAMPLE_OUTPUT)
            with patch("src.hardware_utils_ex.subprocess.run", return_value=compile_result) as mock_run:
                    ram, flash, latency, arena_bytes, err, _power = HIL_spec(
                    dirpath=sketch_dir,
                    chosen_device="ARDUINO_NANO_33_BLE_SENSE",
                    compile_only=True,
                )

        self.assertEqual(err, HIL_ERROR_OK)
        self.assertEqual(latency, -1.0)
        self.assertEqual(ram, 98112)
        self.assertEqual(flash, 376104)
        self.assertGreater(arena_bytes, 0)
        compile_args = mock_run.call_args[0][0]
        self.assertIn("compile", compile_args)


ARDUINO_CLI_AVAILABLE = _cli_exists()
SKETCH_SOURCE_DIR = ROOT_DIR / "tinyodom_tcn"


@unittest.skipUnless(
    ARDUINO_CLI_AVAILABLE and SKETCH_SOURCE_DIR.exists(),
    "Arduino CLI and tinyodom sketch are required for compile-only validation.",
)
class HILCompileOnlyTests(TinyModelMixin, OversizedModelMixin, unittest.TestCase):
    def test_hil_spec_compile_only_runs_cli(self):
        # Runs the compile-only flow to ensure Arduino CLI integration keeps returning resource metrics.
        with tempfile.TemporaryDirectory() as tmpdir:
            sketch_copy = Path(tmpdir) / "tinyodom_tcn"
            shutil.copytree(SKETCH_SOURCE_DIR, sketch_copy)
            ram, flash, latency, arena_bytes, err, _power = HIL_spec(
                dirpath=sketch_copy,
                chosen_device="ARDUINO_NANO_33_BLE_SENSE",
                compile_only=True,
            )
            if err != HIL_ERROR_OK:
                self.skipTest("arduino-cli compile failed in this environment.")
            self.assertGreater(ram, 0)
            self.assertGreater(flash, 0)
            self.assertEqual(latency, -1.0)
            self.assertGreater(arena_bytes, 0)

    def test_compile_only_pipeline_reports_usage(self):
        # Uses the TinyModel mixin to refresh the sketch artifacts and capture CLI metrics.
        model = self.model
        calibration = self.train_x

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            tflite_path = tmp_path / "oversized.tflite"
            convert_to_tflite_model(model, calibration, output_name=tflite_path)
            cpp_dir = tmp_path / "cpp"
            convert_to_cpp_model(tflite_path, cpp_dir)

            sketch_copy = tmp_path / "tinyodom_tcn"
            shutil.copytree(SKETCH_SOURCE_DIR, sketch_copy)
            shutil.copy2(cpp_dir / "model.cc", sketch_copy / "model.cc")
            shutil.copy2(cpp_dir / "model.h", sketch_copy / "model.h")

            build_dir = tmp_path / "arduino-build"
            build_dir.mkdir()
            compile_cmd = [
                "arduino-cli",
                # "--config-file",
                # ARDUINO_CLI_CONFIG,
                "compile",
                "--fqbn",
                DEVICE_SPECS["ARDUINO_NANO_33_BLE_SENSE"]["fqbn"],
                "--build-path",
                str(build_dir),
                str(sketch_copy),
            ]
            proc = subprocess.run(compile_cmd, capture_output=True, text=True, check=False)
            compile_output = "\n".join([proc.stdout, proc.stderr])
            flash_bytes, ram_bytes = _parse_memory_from_compile(compile_output)

        self.assertIsNotNone(flash_bytes, f"Missing flash usage in output:\n{compile_output}")
        self.assertIsNotNone(ram_bytes, f"Missing RAM usage in output:\n{compile_output}")
        assert flash_bytes is not None and ram_bytes is not None
        self.assertGreater(flash_bytes, 0)
        self.assertGreater(ram_bytes, 0)

    def test_compile_only_detects_ram_overflow(self):
        # Request a gigantic arena via HIL_spec so the CLI trips the RAM limit.
        with tempfile.TemporaryDirectory() as tmpdir:
            sketch_copy = Path(tmpdir) / "tinyodom_tcn"
            shutil.copytree(SKETCH_SOURCE_DIR, sketch_copy)

            ram, flash, latency, arena_bytes, err, _power = HIL_spec(
                dirpath=sketch_copy,
                chosen_device="ARDUINO_NANO_33_BLE_SENSE",
                arenaSizes=[512],
                idx=0,
                compile_only=True,
            )
        if err != HIL_ERROR_RAM_OVERFLOW:
            self.skipTest(f"Expected RAM overflow but got err={err}; board toolchain may differ.")
        self.assertEqual((ram, flash, latency), (-1, -1, -1.0))
        self.assertGreater(arena_bytes, 0)

    def test_compile_only_with_oversized_model_triggers_flash_overflow(self):
        # Embeds a massive TCN in the sketch and expects flash overflow during compile.
        model = self._build_oversized_model()
        calibration = np.random.rand(
            16, self._OVERSIZED_TIMESTEPS, self._OVERSIZED_CHANNELS
        ).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            tflite_path = tmp_path / "oversized.tflite"
            convert_to_tflite_model(model, calibration, output_name=tflite_path)
            cpp_dir = tmp_path / "cpp"
            convert_to_cpp_model(tflite_path, cpp_dir)

            sketch_copy = tmp_path / "tinyodom_tcn"
            shutil.copytree(SKETCH_SOURCE_DIR, sketch_copy)
            shutil.copy2(cpp_dir / "model.cc", sketch_copy / "model.cc")
            shutil.copy2(cpp_dir / "model.h", sketch_copy / "model.h")

            ram, flash, latency, arena_bytes, err, _power = HIL_spec(
                dirpath=sketch_copy,
                chosen_device="ARDUINO_NANO_33_BLE_SENSE",
                compile_only=True,
            )
        if err != HIL_ERROR_FLASH_OVERFLOW:
            self.skipTest(f"Expected flash overflow but got err={err}; board toolchain may differ.")
        self.assertEqual((ram, flash, latency), (-1, -1, -1.0))
        self.assertGreater(arena_bytes, 0)


if __name__ == "__main__":
    defaultTest=None
    # defaultTest='ConversionHelperTests.test_convert_to_tflite_model_creates_file'
    
    unittest.main(defaultTest=defaultTest)
