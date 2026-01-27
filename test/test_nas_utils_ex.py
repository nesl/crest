import csv
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tensorflow as tf

# Silence long TF logs during unit runs.
tf.get_logger().setLevel("ERROR")

# Ensure `src` is importable when the suite is launched via `python -m unittest`.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.nas_utils_ex import (
    DROP_RATE_CHOICES,
    collect_metrics,
    count_flops,
    load_config,
    log_trial,
)  # noqa: E402
from src.hardware_utils_ex import convert_to_cpp_model, convert_to_tflite_model  # noqa: E402, E501

try:  # Support both `python -m unittest test.test_*` and direct execution.
    from test.test_hardware_utils_ex import _cli_exists  # type: ignore  # noqa: E402
except ModuleNotFoundError:
    from test_hardware_utils_ex import _cli_exists  # type: ignore  # noqa: E402


class CountFlopsTests(unittest.TestCase):
    """Validate FLOP estimates produced by the NAS helpers."""

    def tearDown(self) -> None:
        # Prevent TF from accumulating graphs between tests.
        tf.keras.backend.clear_session()

    def test_deeper_model_has_more_flops(self) -> None:
        """A slightly larger dense stack should yield more FLOPs."""
        input_shape = (8,)

        small_model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=input_shape),
                tf.keras.layers.Dense(4, activation="relu"),
                tf.keras.layers.Dense(2, activation="linear"),
            ]
        )
        big_model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=input_shape),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(2, activation="linear"),
            ]
        )

        small_flops = count_flops(small_model, input_shape)
        big_flops = count_flops(big_model, input_shape)

        self.assertIsInstance(small_flops, int)
        self.assertGreater(small_flops, 0)
        self.assertGreater(big_flops, small_flops)


class CollectMetricsTests(unittest.TestCase):
    """Ensure controller plumbing and normalization behave as expected."""

    def test_proxy_metrics_normalize_none_values(self) -> None:
        """Proxy runs should convert None outputs into sentinel values."""

        def fake_controller(run_hil: bool, **kwargs):
            # Simulate a proxy flow that only reports flash usage.
            self.assertFalse(run_hil)
            self.assertEqual(kwargs["chosen_device"], "ARDUINO_NANO_33_BLE_SENSE")
            return (None, 4096, None, 2048, 0, None)

        with patch("src.nas_utils_ex.HIL_controller", fake_controller):
            metrics = collect_metrics(
                hil_enabled=False,
                flops=10_000_000,
                device_name="ARDUINO_NANO_33_BLE_SENSE",
                window_size=128,
                input_dim=6,
                dirpath=Path("tinyodom_tcn"),
                latency_proxy_max_flops=20_000_000,
                serial_port=None,
                latency_budget_ms=50.0,
            )

        self.assertEqual(metrics["ram_bytes"], -1)
        self.assertEqual(metrics["flash_bytes"], 4096)
        self.assertEqual(metrics["latency_ms"], -1)
        self.assertEqual(metrics["arena_bytes"], 2048)
        self.assertEqual(metrics["error_code"], 0)
        self.assertEqual(metrics["error_label"], "HIL_MASTER_PENDING")
        self.assertEqual(metrics["latency_budget_ms"], -1)

    def test_hil_metrics_report_latency_budget(self) -> None:
        """HIL runs should normalize latency using the provided budget."""

        def fake_controller(run_hil: bool, **kwargs):
            self.assertTrue(run_hil)
            self.assertEqual(kwargs["serial_port"], "ttyACM0")
            return (1024, 8192, 25.0, 4096, 0, None)

        with patch("src.nas_utils_ex.HIL_controller", fake_controller):
            metrics = collect_metrics(
                hil_enabled=True,
                flops=5_000_000,
                device_name="ARDUINO_NANO_33_BLE_SENSE",
                window_size=128,
                input_dim=6,
                dirpath=Path("tinyodom_tcn"),
                latency_proxy_max_flops=20_000_000,
                serial_port="ttyACM0",
                latency_budget_ms=50000.0,
            )

        self.assertEqual(metrics["latency_ms"], 25000.0)
        self.assertEqual(metrics["latency_budget_ms"], 50000.0)
        self.assertEqual(metrics["error_label"], "HIL_MASTER_PENDING")


class LoadSettingsTests(unittest.TestCase):
    """Verify the NAS configuration loader derives paths and validates input."""

    def test_load_settings_derives_expected_paths(self) -> None:
        """YAML entries should produce resolved paths and derived file names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "config.yaml"
            models_dir = tmp_path / "models_dir"
            tcn_dir = tmp_path / "tcn_dir"
            config_path.write_text(
                "\n".join(
                    [
                        "device:",
                        "  name: TEST_DEVICE",
                        "training:",
                        "  nas_trials: 5",
                        "outputs:",
                        f"  models_dir: \"{models_dir}\"",
                        f"  tcn_dir: \"{tcn_dir}\"",
                    ]
                )
            )

            settings = load_config(config_path=config_path)

            self.assertEqual(
                settings.outputs.model_name, "TinyOdomEx_OxIOD_TEST_DEVICE.tflite"
            )
            self.assertEqual(
                settings.outputs.checkpoint_name, "TinyOdomEx_OxIOD_TEST_DEVICE.keras"
            )
            self.assertTrue(settings.outputs.models_dir.is_dir())
            self.assertTrue(settings.outputs.tcn_dir.is_dir())
            self.assertEqual(
                settings.outputs.tflite_model_path,
                settings.outputs.models_dir / settings.outputs.model_name,
            )
            self.assertEqual(
                settings.outputs.checkpoint_path,
                settings.outputs.models_dir / settings.outputs.checkpoint_name,
            )
            self.assertEqual(settings.training.drop_rate_choices, DROP_RATE_CHOICES)  
    def test_load_settings_requires_sections(self) -> None:
        """Missing required sections should raise informative errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            cfg_missing_device = tmp_path / "missing_device.yaml"
            cfg_missing_device.write_text(
                "\n".join(
                    [
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  tcn_dir: \"{tmp_path / 'tcn'}\"",
                    ]
                )
            )

            with self.assertRaises(KeyError):
                load_config(config_path=cfg_missing_device)

            cfg_missing_outputs = tmp_path / "missing_outputs.yaml"
            cfg_missing_outputs.write_text(
                "\n".join(
                    [
                        "device:",
                        "  name: TEST_DEVICE",
                    ]
                )
            )

            with self.assertRaises(KeyError):
                load_config(config_path=cfg_missing_outputs)

    def test_load_settings_requires_training_section(self) -> None:
        """Training section should be mandatory for NAS runs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            cfg_missing_training = tmp_path / "missing_training.yaml"
            cfg_missing_training.write_text(
                "\n".join(
                    [
                        "device:",
                        "  name: TEST_DEVICE",
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  tcn_dir: \"{tmp_path / 'tcn'}\"",
                    ]
                )
            )

            with self.assertRaises(KeyError):
                load_config(config_path=cfg_missing_training)

    def test_load_settings_sets_default_max_total_trials(self) -> None:
        """max_total_trials should default to 2x the requested nas_trials when omitted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            cfg = tmp_path / "config.yaml"
            cfg.write_text(
                "\n".join(
                    [
                        "device:",
                        "  name: TEST_DEVICE",
                        "training:",
                        "  nas_trials: 10",
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  tcn_dir: \"{tmp_path / 'tcn'}\"",
                    ]
                )
            )

            settings = load_config(config_path=cfg)
            self.assertEqual(settings.training.max_total_trials, 20)

    def test_load_settings_missing_file(self) -> None:
        """Nonexistent config paths should raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_config(config_path=Path("does_not_exist.yaml"))


@unittest.skipUnless(_cli_exists(), "Arduino CLI not installed")
class CollectMetricsIntegrationTests(unittest.TestCase):
    """Run collect_metrics against the real controller (proxy mode)."""

    def test_proxy_flow_runs_end_to_end(self) -> None:
        sketch_src = ROOT_DIR / "tinyodom_tcn"
        with tempfile.TemporaryDirectory() as tmpdir:
            sketch_dir = Path(tmpdir) / "tinyodom_tcn"
            shutil.copytree(sketch_src, sketch_dir)

            # Build a tiny model so Arduino CLI compiles deterministically without overflow.
            inputs = tf.keras.Input(shape=(64, 3))
            x = tf.keras.layers.Conv1D(4, kernel_size=3, activation="relu")(inputs)
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
            outputs = tf.keras.layers.Dense(2, activation="linear")(x)
            model = tf.keras.Model(inputs, outputs)

            dummy_data = np.random.rand(8, 64, 3).astype(np.float32)
            tflite_path = sketch_dir / "model.tflite"
            convert_to_tflite_model(model, dummy_data, output_name=tflite_path)
            convert_to_cpp_model(
                tflite_path,
                sketch_dir,
                array_name="g_model",
                source_name="model.cc",
                header_name="model.h",
            )

            metrics = collect_metrics(
                hil_enabled=False,
                flops=5_000_000,
                device_name="ARDUINO_NANO_33_BLE_SENSE",
                window_size=200,
                input_dim=3,
                dirpath=sketch_dir,
                latency_proxy_max_flops=30_000_000,
                serial_port=None,
            )

        self.assertGreaterEqual(metrics["flash_bytes"], 0)
        self.assertGreaterEqual(metrics["ram_bytes"], -1)
        self.assertGreaterEqual(metrics["arena_bytes"], 0)
        self.assertEqual(metrics["latency_ms"], -1)
        self.assertEqual(metrics["latency_budget_ms"], -1)


class FakeTrial:
    def __init__(self):
        self.attrs = {}

    def set_user_attr(self, key, value):
        self.attrs[key] = value


class LogTrialTests(unittest.TestCase):
    HEADER = [
        "study_name",
        "timestamp_unix",
        "timestamp_readable",
        "score",
        "rmse_vel_x",
        "rmse_vel_y",
        "ram_bytes",
        "flash_bytes",
        "flops",
        "latency_ms",
        "energy_mj_per_inference",
        "avg_power_mw",
        "avg_current_ma",
        "bus_voltage_v",
        "nb_filters",
        "kernel_size",
        "dilations",
        "dropout_rate",
        "use_skip_connections",
        "norm_flag",
        "error_code",
        "error_label",
        "pruned",
        "prune_reason",
    ]

    def _sample_metrics(self):
        return {
            "ram_bytes": 1000,
            "flash_bytes": 2000,
            "latency_ms": 10,
            "latency_budget_ms": -1,
            "arena_bytes": 4096,
            "error_code": 0,
            "error_label": "HIL_MASTER_PENDING",
            "energy_mj_per_inference": 0.5,
            "avg_power_mw": 2.0,
            "avg_current_ma": 1.5,
            "bus_voltage_v": 3.3,
            "idle_power_mw": 2.0,
        }

    def _sample_hyperparams(self):
        return {
            "flops": 1_000_000,
            "nb_filters": 32,
            "kernel_size": 3,
            "dilations": [1, 2, 4],
            "dropout_rate": 0.1,
            "use_skip_connections": True,
            "norm_flag": True,
        }

    def test_log_trial_writes_header_and_row(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "log.csv"
            fake_trial = FakeTrial()
            metrics = self._sample_metrics()
            hyperparams = self._sample_hyperparams()

            with patch("src.nas_utils_ex.time.time", return_value=123.0), patch(
                "src.nas_utils_ex.time.strftime", return_value="01-02-1970 00:02:03"
            ):
                log_trial(
                    score=0.5,
                    rmse_vel_x=0.1,
                    rmse_vel_y=0.2,
                    metrics=metrics,
                    hyperparams=hyperparams,
                    trial=fake_trial,
                    log_file_name=str(log_path),
                )

            with log_path.open(newline="") as csvfile:
                rows = list(csv.reader(csvfile))

            self.assertEqual(rows[0], self.HEADER)
            header_index = {name: idx for idx, name in enumerate(self.HEADER)}
            self.assertEqual(rows[1][header_index["timestamp_unix"]], "123.0")
            self.assertEqual(
                rows[1][header_index["timestamp_readable"]], "01-02-1970 00:02:03"
            )
            self.assertEqual(float(rows[1][header_index["score"]]), 0.5)
            self.assertEqual(
                int(rows[1][header_index["ram_bytes"]]), metrics["ram_bytes"]
            )
            self.assertEqual(
                float(rows[1][header_index["latency_ms"]]), metrics["latency_ms"]
            )
            self.assertAlmostEqual(
                float(rows[1][header_index["energy_mj_per_inference"]]),
                metrics["energy_mj_per_inference"],
            )
            self.assertAlmostEqual(
                float(rows[1][header_index["avg_power_mw"]]), metrics["avg_power_mw"]
            )
            self.assertAlmostEqual(
                float(rows[1][header_index["avg_current_ma"]]), metrics["avg_current_ma"]
            )
            self.assertAlmostEqual(
                float(rows[1][header_index["bus_voltage_v"]]), metrics["bus_voltage_v"]
            )
            self.assertEqual(
                rows[1][header_index["error_label"]], metrics["error_label"]
            )
            self.assertEqual(rows[1][header_index["pruned"]], "False")
            self.assertEqual(rows[1][header_index["prune_reason"]], "")

            self.assertEqual(fake_trial.attrs["ram_bytes"], metrics["ram_bytes"])
            self.assertEqual(fake_trial.attrs["rmse_vel_x"], 0.1)
            self.assertEqual(fake_trial.attrs["rmse_vel_y"], 0.2)
            self.assertEqual(fake_trial.attrs["latency_budget_ms"], metrics["latency_budget_ms"])
            self.assertEqual(
                fake_trial.attrs["error_code_label"], metrics["error_label"]
            )
            self.assertEqual(
                fake_trial.attrs["energy_mj_per_inference"],
                metrics["energy_mj_per_inference"],
            )

    def test_log_trial_appends_without_duplicate_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "log.csv"
            metrics = self._sample_metrics()
            hyperparams = self._sample_hyperparams()

            fake_trial_one = FakeTrial()
            fake_trial_two = FakeTrial()

            log_trial(
                score=0.3,
                rmse_vel_x=0.05,
                rmse_vel_y=0.06,
                metrics=metrics,
                hyperparams=hyperparams,
                trial=fake_trial_one,
                log_file_name=str(log_path),
            )
            log_trial(
                score=0.2,
                rmse_vel_x=0.04,
                rmse_vel_y=0.05,
                metrics=metrics,
                hyperparams=hyperparams,
                trial=fake_trial_two,
                log_file_name=str(log_path),
            )

            with log_path.open(newline="") as csvfile:
                rows = list(csv.reader(csvfile))

            self.assertEqual(rows[0], self.HEADER)
            self.assertEqual(len(rows), 3)


if __name__ == "__main__":
    unittest.main()
