import csv
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tensorflow as tf
from addict import Dict

# Silence long TF logs during unit runs.
tf.get_logger().setLevel("ERROR")

# Ensure `src` is importable when the suite is launched via `python -m unittest`.
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tinyodom.model import (
    CollectMetricsRequest,
    DROP_RATE_CHOICES,
    HarnessConfig,
    apply_combined_perturbation,
    build_collect_metrics_request,
    collect_bn_layers,
    collect_non_bn_bias_layers,
    collect_metrics,
    count_flops,
    iter_layers,
    load_config,
    log_trial,
    validate_loaded_model_input_shape,
)  # noqa: E402
from tinyodom.hardware import convert_to_cpp_model, convert_to_tflite_model  # noqa: E402, E501

try:  # Support both `python -m unittest test.test_*` and direct execution.
    from test.test_hardware import _cli_exists  # type: ignore  # noqa: E402
except ModuleNotFoundError:
    from test_hardware import _cli_exists  # type: ignore  # noqa: E402


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


class ModelVariantHelperTests(unittest.TestCase):
    """Validate model-variant helper utilities used by HILServer."""

    def tearDown(self) -> None:
        tf.keras.backend.clear_session()

    def _build_perturbation_test_model(self) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=(16, 4))
        x = tf.keras.layers.Conv1D(8, kernel_size=3, use_bias=True, name="conv_bias")(inputs)
        x = tf.keras.layers.BatchNormalization(name="bn1")(x)
        x = tf.keras.layers.Conv1D(8, kernel_size=3, use_bias=False, name="conv_no_bias")(x)
        x = tf.keras.layers.BatchNormalization(name="bn2")(x)
        x = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)
        x = tf.keras.layers.Dense(4, use_bias=True, name="dense_bias")(x)
        outputs = tf.keras.layers.Dense(2, use_bias=True, name="dense_out")(x)
        return tf.keras.Model(inputs, outputs)

    def _snapshot_touched_tensors(self, model: tf.keras.Model) -> dict[str, np.ndarray]:
        snapshot: dict[str, np.ndarray] = {}
        for layer in collect_bn_layers(model):
            for attr in ("gamma", "beta", "moving_mean", "moving_variance"):
                tensor = getattr(layer, attr, None)
                if tensor is not None:
                    snapshot[f"{layer.name}.{attr}"] = np.array(tensor.numpy(), copy=True)
        for layer in collect_non_bn_bias_layers(model):
            bias = getattr(layer, "bias", None)
            if bias is not None:
                snapshot[f"{layer.name}.bias"] = np.array(bias.numpy(), copy=True)
        return snapshot

    def test_iter_layers_flattens_without_duplicates(self) -> None:
        inputs = tf.keras.Input(shape=(8,))
        shared_dense = tf.keras.layers.Dense(8, activation="relu", name="shared_dense")
        x = shared_dense(inputs)
        x = shared_dense(x)
        nested_stack = tf.keras.Sequential(
            [tf.keras.layers.Dense(8, name="nested_dense")],
            name="nested_stack",
        )
        x = nested_stack(x)
        outputs = tf.keras.layers.Dense(1, name="head_dense")(x)
        model = tf.keras.Model(inputs, outputs)

        layers = iter_layers(model)
        self.assertEqual(len(layers), len({id(layer) for layer in layers}))
        self.assertEqual(sum(layer is shared_dense for layer in layers), 1)
        self.assertTrue(any(layer.name == "nested_dense" for layer in layers))

    def test_collect_bn_layers_returns_only_batchnorm(self) -> None:
        inputs = tf.keras.Input(shape=(8,))
        x = tf.keras.layers.Dense(8, use_bias=True, name="dense1")(inputs)
        x = tf.keras.layers.BatchNormalization(name="bn1")(x)
        x = tf.keras.layers.Dense(4, use_bias=True, name="dense2")(x)
        outputs = tf.keras.layers.BatchNormalization(name="bn2")(x)
        model = tf.keras.Model(inputs, outputs)

        bn_layers = collect_bn_layers(model)
        self.assertEqual(len(bn_layers), 2)
        self.assertTrue(
            all(isinstance(layer, tf.keras.layers.BatchNormalization) for layer in bn_layers)
        )

    def test_collect_non_bn_bias_layers_excludes_bn(self) -> None:
        inputs = tf.keras.Input(shape=(16, 4))
        x = tf.keras.layers.Conv1D(8, kernel_size=3, use_bias=True, name="conv_bias")(inputs)
        x = tf.keras.layers.BatchNormalization(name="bn")(x)
        x = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)
        x = tf.keras.layers.Dense(4, use_bias=True, name="dense_bias")(x)
        outputs = tf.keras.layers.Dense(2, use_bias=False, name="dense_no_bias")(x)
        model = tf.keras.Model(inputs, outputs)

        layers = collect_non_bn_bias_layers(model)
        layer_names = {layer.name for layer in layers}
        self.assertSetEqual(layer_names, {"conv_bias", "dense_bias"})
        self.assertTrue(
            all(not isinstance(layer, tf.keras.layers.BatchNormalization) for layer in layers)
        )

    def test_apply_combined_perturbation_returns_expected_counts(self) -> None:
        model = self._build_perturbation_test_model()
        bn_touched, bias_touched = apply_combined_perturbation(model, seed=1337)
        self.assertEqual(bn_touched, 2)
        self.assertEqual(bias_touched, 3)

    def test_apply_combined_perturbation_is_deterministic_for_seed(self) -> None:
        seed = 2026
        model_a = self._build_perturbation_test_model()
        model_b = self._build_perturbation_test_model()

        apply_combined_perturbation(model_a, seed=seed)
        apply_combined_perturbation(model_b, seed=seed)
        snapshot_a = self._snapshot_touched_tensors(model_a)
        snapshot_b = self._snapshot_touched_tensors(model_b)

        self.assertSetEqual(set(snapshot_a), set(snapshot_b))
        self.assertGreater(len(snapshot_a), 0)
        for key, value_a in snapshot_a.items():
            np.testing.assert_allclose(value_a, snapshot_b[key], atol=0.0, rtol=0.0)

    def test_validate_loaded_model_input_shape_accepts_matching_shape(self) -> None:
        inputs = tf.keras.Input(shape=(20, 6))
        outputs = tf.keras.layers.Dense(4, name="dense")(inputs)
        model = tf.keras.Model(inputs, outputs)
        hyperparams = Dict(timesteps=20, input_dim=6)
        validate_loaded_model_input_shape(model, hyperparams)

    def test_validate_loaded_model_input_shape_rejects_mismatch(self) -> None:
        inputs = tf.keras.Input(shape=(20, 6))
        outputs = tf.keras.layers.Dense(4, name="dense")(inputs)
        model = tf.keras.Model(inputs, outputs)

        mismatched_hyperparams = (
            Dict(timesteps=21, input_dim=6),
            Dict(timesteps=20, input_dim=7),
        )
        for hyperparams in mismatched_hyperparams:
            with self.subTest(hyperparams=dict(hyperparams)):
                with self.assertRaises(ValueError):
                    validate_loaded_model_input_shape(model, hyperparams)

    def test_validate_loaded_model_input_shape_rejects_multi_input_model(self) -> None:
        input_a = tf.keras.Input(shape=(20, 6), name="input_a")
        input_b = tf.keras.Input(shape=(20, 6), name="input_b")
        outputs = tf.keras.layers.Add(name="sum")([input_a, input_b])
        model = tf.keras.Model([input_a, input_b], outputs)
        hyperparams = Dict(timesteps=20, input_dim=6)

        with self.assertRaises(ValueError):
            validate_loaded_model_input_shape(model, hyperparams)


class CollectMetricsTests(unittest.TestCase):
    """Ensure controller plumbing and normalization behave as expected."""

    def test_proxy_metrics_normalize_none_values(self) -> None:
        """Proxy runs should convert None outputs into sentinel values."""

        def fake_controller(run_hil: bool, **kwargs):
            # Simulate a proxy flow that only reports flash usage.
            self.assertFalse(run_hil)
            self.assertEqual(kwargs["chosen_device"], "ARDUINO_NANO_33_BLE_SENSE")
            return (None, 4096, None, 2048, 0, None)

        with patch("tinyodom.model.HIL_controller", fake_controller):
            request = CollectMetricsRequest(
                hil_enabled=False,
                energy_aware=False,
                flops=10_000_000,
                device_name="ARDUINO_NANO_33_BLE_SENSE",
                window_size=128,
                input_dim=6,
                dirpath=Path("tinyodom_tcn"),
                latency_proxy_max_flops=20_000_000,
                serial_port=None,
                latency_budget_ms=50.0,
            )
            metrics = collect_metrics(request)

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

        with patch("tinyodom.model.HIL_controller", fake_controller):
            request = CollectMetricsRequest(
                hil_enabled=True,
                energy_aware=False,
                flops=5_000_000,
                device_name="ARDUINO_NANO_33_BLE_SENSE",
                window_size=128,
                input_dim=6,
                dirpath=Path("tinyodom_tcn"),
                latency_proxy_max_flops=20_000_000,
                serial_port="ttyACM0",
                latency_budget_ms=50000.0,
            )
            metrics = collect_metrics(request)

        self.assertEqual(metrics["latency_ms"], 25000.0)
        self.assertEqual(metrics["latency_budget_ms"], 50000.0)
        self.assertEqual(metrics["error_label"], "HIL_MASTER_PENDING")

    def test_energy_aware_harness_fields_forwarded_to_controller(self) -> None:
        """Energy-aware requests should forward harness settings to HIL_controller."""

        harness = HarnessConfig(
            harness_serial_port="ttyACM1",
            harness_fqbn="arduino:mbed_nano:nano33ble",
            harness_auto_flash="once",
            harness_arm_pin=3,
            harness_trigger_pin=2,
            dut_arm_hold_ms=600,
            harness_stable_low_ms=500,
            harness_ready_timeout_s=5.0,
            harness_arm_timeout_s=0.0,
            harness_active_timeout_s=12.0,
            harness_done_timeout_s=3.6,
        )

        def fake_controller(run_hil: bool, **kwargs):
            self.assertTrue(run_hil)
            self.assertEqual(kwargs["harness_serial_port"], "ttyACM1")
            self.assertEqual(kwargs["harness_fqbn"], "arduino:mbed_nano:nano33ble")
            self.assertEqual(kwargs["harness_auto_flash"], "once")
            self.assertEqual(kwargs["harness_arm_pin"], 3)
            self.assertEqual(kwargs["harness_trigger_pin"], 2)
            self.assertEqual(kwargs["dut_arm_hold_ms"], 600)
            self.assertEqual(kwargs["harness_stable_low_ms"], 500)
            self.assertEqual(kwargs["harness_ready_timeout_s"], 5.0)
            self.assertEqual(kwargs["harness_arm_timeout_s"], 0.0)
            self.assertEqual(kwargs["harness_active_timeout_s"], 12.0)
            self.assertEqual(kwargs["harness_done_timeout_s"], 3.6)
            return (1024, 8192, 0.025, 4096, 0, None)

        with patch("tinyodom.model.HIL_controller", fake_controller):
            request = CollectMetricsRequest(
                hil_enabled=True,
                energy_aware=True,
                flops=5_000_000,
                device_name="ARDUINO_NANO_33_BLE_SENSE",
                window_size=128,
                input_dim=6,
                dirpath=Path("tinyodom_tcn"),
                latency_proxy_max_flops=20_000_000,
                serial_port="ttyACM0",
                latency_budget_ms=200.0,
                dut_ready_timeout_s=5.0,
                harness=harness,
            )
            metrics = collect_metrics(request)

        self.assertEqual(metrics["error_code"], 0)
        self.assertEqual(metrics["latency_budget_ms"], 200.0)

    def test_energy_aware_without_harness_raises(self) -> None:
        request = CollectMetricsRequest(
            hil_enabled=True,
            energy_aware=True,
            flops=5_000_000,
            device_name="ARDUINO_NANO_33_BLE_SENSE",
            window_size=128,
            input_dim=6,
            dirpath=Path("tinyodom_tcn"),
            latency_proxy_max_flops=20_000_000,
            serial_port="ttyACM0",
            latency_budget_ms=50000.0,
            harness=None,
        )

        with self.assertRaises(RuntimeError):
            collect_metrics(request)

    def test_collect_metrics_forwards_device_options(self) -> None:
        def fake_controller(run_hil: bool, **kwargs):
            self.assertFalse(run_hil)
            self.assertEqual(kwargs["device_options"]["target_core"], "cm7")
            self.assertEqual(kwargs["device_options"]["split"], "75_25")
            return (1024, 2048, None, 4096, 0, None)

        with patch("tinyodom.model.HIL_controller", fake_controller):
            request = CollectMetricsRequest(
                hil_enabled=False,
                energy_aware=False,
                flops=10_000_000,
                device_name="PORTENTA_H7",
                window_size=128,
                input_dim=6,
                dirpath=Path("tinyodom_tcn"),
                latency_proxy_max_flops=20_000_000,
                serial_port=None,
                latency_budget_ms=50.0,
                device_options={"target_core": "cm7", "split": "75_25"},
            )
            metrics = collect_metrics(request)

        self.assertEqual(metrics["error_code"], 0)

    def test_collect_metrics_portenta_cm4_runtime_requires_harness(self) -> None:
        request = CollectMetricsRequest(
            hil_enabled=True,
            energy_aware=False,
            flops=5_000_000,
            device_name="PORTENTA_H7",
            window_size=128,
            input_dim=6,
            dirpath=Path("tinyodom_tcn"),
            latency_proxy_max_flops=20_000_000,
            serial_port="ttyACM0",
            latency_budget_ms=200.0,
            device_options={"target_core": "cm4", "split": "50_50", "security": "none"},
        )

        with patch("tinyodom.model.HIL_controller") as controller_mock:
            with self.assertRaises(RuntimeError) as context:
                collect_metrics(request)

        self.assertIn("harness", str(context.exception).lower())
        controller_mock.assert_not_called()

    def test_collect_metrics_portenta_cm4_non_energy_forwards_harness(self) -> None:
        harness = HarnessConfig(
            harness_serial_port="ttyACM1",
            harness_fqbn="arduino:mbed_nano:nano33ble",
            harness_auto_flash="once",
            harness_arm_pin=3,
            harness_trigger_pin=2,
            dut_arm_hold_ms=600,
            harness_stable_low_ms=500,
            harness_ready_timeout_s=5.0,
            harness_arm_timeout_s=0.0,
            harness_active_timeout_s=12.0,
            harness_done_timeout_s=3.6,
        )

        def fake_controller(run_hil: bool, **kwargs):
            self.assertTrue(run_hil)
            self.assertEqual(kwargs["harness_serial_port"], "ttyACM1")
            self.assertEqual(kwargs["harness_done_timeout_s"], 3.6)
            return (1024, 2048, 0.01, 4096, 0, None)

        with patch("tinyodom.model.HIL_controller", fake_controller):
            request = CollectMetricsRequest(
                hil_enabled=True,
                energy_aware=False,
                flops=5_000_000,
                device_name="PORTENTA_H7",
                window_size=128,
                input_dim=6,
                dirpath=Path("tinyodom_tcn"),
                latency_proxy_max_flops=20_000_000,
                serial_port="ttyACM0",
                latency_budget_ms=200.0,
                harness=harness,
                device_options={"target_core": "cm4", "split": "50_50", "security": "none"},
            )
            metrics = collect_metrics(request)

        self.assertEqual(metrics["error_code"], 0)


class BuildCollectMetricsRequestTests(unittest.TestCase):
    """Validate config/hyperparameter mapping into CollectMetricsRequest."""

    def test_non_energy_aware_sets_harness_none(self) -> None:
        config = Dict(
            training=Dict(energy_aware=False, latency_proxy_max_flops=20_000_000),
            device=Dict(hil=True, name="ARDUINO_NANO_33_BLE_SENSE", serial_port="ttyACM0"),
            data=Dict(window_size=128),
            outputs=Dict(tcn_dir=Path("tinyodom_tcn")),
        )
        hyperparams = Dict(flops=123, input_dim=6)

        request = build_collect_metrics_request(config, hyperparams, latency_budget_ms=200.0)

        self.assertIsNone(request.harness)
        self.assertFalse(request.energy_aware)
        self.assertEqual(request.flops, 123)
        self.assertEqual(request.input_dim, 6)

    def test_energy_aware_populates_harness(self) -> None:
        config = Dict(
            training=Dict(energy_aware=True, latency_proxy_max_flops=20_000_000),
            device=Dict(
                hil=True,
                name="ARDUINO_NANO_33_BLE_SENSE",
                serial_port="ttyACM0",
                harness_serial_port="ttyACM1",
                harness_fqbn="arduino:mbed_nano:nano33ble",
                harness_auto_flash="once",
                harness_arm_pin=3,
                harness_trigger_pin=2,
                dut_arm_hold_ms=600,
                harness_stable_low_ms=500,
                harness_ready_timeout_s=5.0,
                harness_arm_timeout_s=5.0,
                harness_active_timeout_s=30.0,
                harness_done_timeout_s=5.0,
            ),
            data=Dict(window_size=128),
            outputs=Dict(tcn_dir=Path("tinyodom_tcn")),
        )
        hyperparams = Dict(flops=123, input_dim=6)

        request = build_collect_metrics_request(config, hyperparams, latency_budget_ms=200.0)

        self.assertTrue(request.energy_aware)
        self.assertIsInstance(request.harness, HarnessConfig)
        self.assertEqual(request.harness.harness_serial_port, "ttyACM1")
        self.assertEqual(request.harness.harness_arm_timeout_s, 5.0)

    def test_missing_dut_ready_timeout_uses_default(self) -> None:
        config = Dict(
            training=Dict(energy_aware=False, latency_proxy_max_flops=20_000_000),
            device=Dict(hil=True, name="ARDUINO_NANO_33_BLE_SENSE", serial_port="ttyACM0"),
            data=Dict(window_size=128),
            outputs=Dict(tcn_dir=Path("tinyodom_tcn")),
        )
        hyperparams = Dict(flops=123, input_dim=6)

        request = build_collect_metrics_request(config, hyperparams, latency_budget_ms=200.0)

        self.assertEqual(request.dut_ready_timeout_s, 5.0)

    def test_energy_aware_missing_harness_serial_port_raises(self) -> None:
        config = Dict(
            training=Dict(energy_aware=True, latency_proxy_max_flops=20_000_000),
            device=Dict(hil=True, name="ARDUINO_NANO_33_BLE_SENSE", serial_port="ttyACM0"),
            data=Dict(window_size=128),
            outputs=Dict(tcn_dir=Path("tinyodom_tcn")),
        )
        hyperparams = Dict(flops=123, input_dim=6)

        with self.assertRaises(RuntimeError):
            build_collect_metrics_request(config, hyperparams, latency_budget_ms=200.0)

    def test_portenta_cm4_runtime_missing_harness_serial_port_raises(self) -> None:
        config = Dict(
            training=Dict(energy_aware=False, latency_proxy_max_flops=20_000_000),
            device=Dict(
                hil=True,
                name="PORTENTA_H7",
                serial_port="ttyACM0",
                portenta=Dict(target_core="cm4", split="50_50", security="none"),
            ),
            data=Dict(window_size=128),
            outputs=Dict(tcn_dir=Path("tinyodom_tcn")),
        )
        hyperparams = Dict(flops=123, input_dim=6)

        with self.assertRaises(RuntimeError):
            build_collect_metrics_request(config, hyperparams, latency_budget_ms=200.0)

    def test_portenta_requires_target_core(self) -> None:
        config = Dict(
            training=Dict(energy_aware=False, latency_proxy_max_flops=20_000_000),
            device=Dict(hil=True, name="PORTENTA_H7", serial_port="ttyACM0", portenta=Dict()),
            data=Dict(window_size=128),
            outputs=Dict(tcn_dir=Path("tinyodom_tcn")),
        )
        hyperparams = Dict(flops=123, input_dim=6)

        with self.assertRaises(RuntimeError):
            build_collect_metrics_request(config, hyperparams, latency_budget_ms=200.0)

    def test_portenta_options_are_forwarded(self) -> None:
        config = Dict(
            training=Dict(energy_aware=False, latency_proxy_max_flops=20_000_000),
            device=Dict(
                hil=False,
                name="PORTENTA_H7",
                serial_port="ttyACM0",
                portenta=Dict(target_core="cm4", split="50_50", security="none"),
            ),
            data=Dict(window_size=128),
            outputs=Dict(tcn_dir=Path("tinyodom_tcn")),
        )
        hyperparams = Dict(flops=123, input_dim=6)

        request = build_collect_metrics_request(config, hyperparams, latency_budget_ms=200.0)

        self.assertEqual(request.device_options["target_core"], "cm4")
        self.assertEqual(request.device_options["split"], "50_50")
        self.assertEqual(request.device_options["security"], "none")

    def test_portenta_cm4_runtime_populates_harness_even_without_energy_aware(self) -> None:
        config = Dict(
            training=Dict(energy_aware=False, latency_proxy_max_flops=20_000_000),
            device=Dict(
                hil=True,
                name="PORTENTA_H7",
                serial_port="ttyACM0",
                harness_serial_port="ttyACM1",
                harness_fqbn="arduino:mbed_nano:nano33ble",
                harness_auto_flash="once",
                harness_arm_pin=3,
                harness_trigger_pin=2,
                dut_arm_hold_ms=600,
                harness_stable_low_ms=500,
                harness_ready_timeout_s=5.0,
                harness_arm_timeout_s=5.0,
                harness_active_timeout_s=30.0,
                harness_done_timeout_s=5.0,
                portenta=Dict(target_core="cm4", split="50_50", security="none"),
            ),
            data=Dict(window_size=128),
            outputs=Dict(tcn_dir=Path("tinyodom_tcn")),
        )
        hyperparams = Dict(flops=123, input_dim=6)

        request = build_collect_metrics_request(config, hyperparams, latency_budget_ms=200.0)

        self.assertIsNotNone(request.harness)
        self.assertEqual(request.harness.harness_serial_port, "ttyACM1")

    def test_device_name_is_normalized_before_request_build(self) -> None:
        config = Dict(
            training=Dict(energy_aware=False, latency_proxy_max_flops=20_000_000),
            device=Dict(
                hil=False,
                name="portenta_h7",
                portenta=Dict(target_core="cm7", split="75_25", security="none"),
            ),
            data=Dict(window_size=128),
            outputs=Dict(tcn_dir=Path("tinyodom_tcn")),
        )
        hyperparams = Dict(flops=123, input_dim=6)

        request = build_collect_metrics_request(config, hyperparams, latency_budget_ms=200.0)

        self.assertEqual(request.device_name, "PORTENTA_H7")
        self.assertEqual(request.device_options["target_core"], "cm7")

    def test_proxy_mode_allows_missing_serial_port(self) -> None:
        config = Dict(
            training=Dict(energy_aware=False, latency_proxy_max_flops=20_000_000),
            device=Dict(hil=False, name="ARDUINO_NANO_33_BLE_SENSE"),
            data=Dict(window_size=128),
            outputs=Dict(tcn_dir=Path("tinyodom_tcn")),
        )
        hyperparams = Dict(flops=123, input_dim=6)

        request = build_collect_metrics_request(config, hyperparams, latency_budget_ms=200.0)

        self.assertIsNone(request.serial_port)


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

    def test_load_settings_energy_aware_requires_harness_serial_port(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            cfg = tmp_path / "config.yaml"
            cfg.write_text(
                "\n".join(
                    [
                        "device:",
                        "  name: TEST_DEVICE",
                        "training:",
                        "  nas_trials: 5",
                        "  energy_aware: true",
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  tcn_dir: \"{tmp_path / 'tcn'}\"",
                    ]
                )
            )

            with self.assertRaises(RuntimeError):
                load_config(config_path=cfg)


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
                CollectMetricsRequest(
                    hil_enabled=False,
                    energy_aware=False,
                    flops=5_000_000,
                    device_name="ARDUINO_NANO_33_BLE_SENSE",
                    window_size=200,
                    input_dim=3,
                    dirpath=sketch_dir,
                    latency_proxy_max_flops=30_000_000,
                    serial_port=None,
                )
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

            with patch("tinyodom.model.time.time", return_value=123.0), patch(
                "tinyodom.model.time.strftime", return_value="01-02-1970 00:02:03"
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
