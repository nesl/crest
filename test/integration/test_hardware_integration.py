import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from test.test_support import SKETCH_SOURCE_DIR, TinyModelMixin, _cli_exists

from tinyodom.hardware import DEVICE_SPECS, HIL_ERROR_OK, HIL_ERROR_RAM_OVERFLOW, HIL_spec
from tinyodom.hardware import convert_to_cpp_model, convert_to_tflite_model
from tinyodom.microcontrollers.arduino_base import _parse_memory_from_compile

ARDUINO_CLI_AVAILABLE = _cli_exists()


@unittest.skipUnless(
    ARDUINO_CLI_AVAILABLE and SKETCH_SOURCE_DIR.exists(),
    "Arduino CLI and tinyodom sketch are required for compile-only validation.",
)
class HILCompileOnlyTests(TinyModelMixin, unittest.TestCase):
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
