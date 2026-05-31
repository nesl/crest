"""Unit tests for NAS model orchestration and metric collection helpers."""

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

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
    CADENCED_CSV_FIELDS,
    DROP_RATE_CHOICES,
    TRIAL_LOG_STABLE_COLUMNS,
    TrialOutcome,
    _minimum_stm32_serial_timeout_s,
    _metric_unavailable,
    ScoreConfigEvaluationError,
    apply_combined_perturbation,
    collect_bn_layers,
    collect_non_bn_bias_layers,
    count_flops,
    evaluate_feasibility_rules,
    evaluate_score_config,
    iter_layers,
    load_config,
    log_trial,
    score_config_uses_training_metrics,
    validate_nas_policy_for_task,
    validate_model_input_shape,
    validate_loaded_model_input_shape,
)  # noqa: E402
from tinyodom.component_selection import resolve_component_selection  # noqa: E402
from tinyodom.hardware import convert_to_cpp_model, convert_to_tflite_model  # noqa: E402, E501
from tinyodom.hil_runtime import CollectMetricsRequest, HarnessConfig, build_collect_metrics_request, collect_metrics  # noqa: E402, E501
from tinyodom.microcontrollers import resolve_device_options  # noqa: E402
import tinyodom.model_families.odom_tcn as odom_tcn_module  # noqa: E402
from tinyodom.model_families.odom_tcn import OdomTCNFamily  # noqa: E402
from tinyodom.pipeline_types import ModelBuildContext, TargetSpec  # noqa: E402


class CountFlopsTests(unittest.TestCase):
    """Validate FLOP estimates produced by the NAS helpers."""

    def tearDown(self) -> None:
        # Prevent TF from accumulating graphs between tests.
        """Clean up test fixtures."""
        tf.keras.backend.clear_session()

    def test_deeper_model_has_more_flops(self) -> None:
        """A slightly larger dense stack should yield more FLOPs."""
        # The FLOP estimator should rise with model depth so proxy latency stays monotonic with model size.
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
        """Clean up test fixtures."""
        tf.keras.backend.clear_session()

    def _build_perturbation_test_model(self) -> tf.keras.Model:
        """Build a model that exercises both BN and non-BN perturbation paths.

        Returns
        -------
        tensorflow.keras.Model
            Model containing batch-normalization layers plus biased/non-biased
            layers so perturbation helpers can prove they touch the right
            tensors.
        """
        # Choose layers that cover both BatchNorm tensors and ordinary bias
        # tensors because the perturbation helper updates those groups
        # differently.
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
        """Capture the tensors that perturbation helpers are allowed to change.

        Parameters
        ----------
        model : tensorflow.keras.Model
            Model whose BN and non-BN bias tensors should be snapshotted.

        Returns
        -------
        dict[str, numpy.ndarray]
            Copies of the tensors keyed by ``layer_name.attribute``.
        """
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
        # Layer iteration should flatten nested and shared graphs without yielding duplicates so traversal helpers only touch each layer once.
        """Validate iter layers flattens without duplicates."""
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
        # BatchNorm collection should only return normalization layers so perturbation helpers do not rewrite unrelated weights.
        """Validate collect bn layers returns only batchnorm."""
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
        # Bias-layer collection should skip BatchNorm offsets so combined perturbations do not double-count normalization parameters.
        """Validate collect non bn bias layers excludes bn."""
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
        # Combined perturbations should report how many BatchNorm and bias tensors were touched so export instrumentation can validate the rewrite.
        """Validate apply combined perturbation returns expected counts."""
        model = self._build_perturbation_test_model()
        bn_touched, bias_touched = apply_combined_perturbation(model, seed=1337)
        self.assertEqual(bn_touched, 2)
        self.assertEqual(bias_touched, 3)

    def test_apply_combined_perturbation_is_deterministic_for_seed(self) -> None:
        # Seeded perturbations should be reproducible so approximate-export runs can be compared across executions.
        """Validate apply combined perturbation is deterministic for seed."""
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
        # Loaded-model shape validation should accept the exact hyperparameter shape the checkpoint was trained for.
        """Validate validate loaded model input shape accepts matching shape."""
        inputs = tf.keras.Input(shape=(20, 6))
        outputs = tf.keras.layers.Dense(4, name="dense")(inputs)
        model = tf.keras.Model(inputs, outputs)
        hyperparams = Dict(timesteps=20, input_dim=6)
        validate_loaded_model_input_shape(model, hyperparams)

    def test_validate_loaded_model_input_shape_rejects_mismatch(self) -> None:
        # Loaded-model shape validation should reject incompatible signatures before a mismatched checkpoint reaches inference.
        """Validate validate loaded model input shape rejects mismatch."""
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
        # Loaded-model shape validation should reject incompatible signatures before a mismatched checkpoint reaches inference.
        """Validate validate loaded model input shape rejects multi input model."""
        input_a = tf.keras.Input(shape=(20, 6), name="input_a")
        input_b = tf.keras.Input(shape=(20, 6), name="input_b")
        outputs = tf.keras.layers.Add(name="sum")([input_a, input_b])
        model = tf.keras.Model([input_a, input_b], outputs)
        hyperparams = Dict(timesteps=20, input_dim=6)

        with self.assertRaises(ValueError):
            validate_loaded_model_input_shape(model, hyperparams)

    def test_validate_model_input_shape_accepts_matching_shape(self) -> None:
        # Freshly built models should accept the expected logical input shape before export and scoring continue.
        """Validate validate model input shape accepts matching shape."""
        inputs = tf.keras.Input(shape=(20, 6))
        outputs = tf.keras.layers.Dense(4, name="dense")(inputs)
        model = tf.keras.Model(inputs, outputs)

        validate_model_input_shape(model, (20, 6))

    def test_validate_model_input_shape_rejects_missing_expected_shape(self) -> None:
        # Model-shape validation should fail fast when the caller cannot provide a usable logical input shape.
        """Validate validate model input shape rejects missing expected shape."""
        inputs = tf.keras.Input(shape=(20, 6))
        outputs = tf.keras.layers.Dense(4, name="dense")(inputs)
        model = tf.keras.Model(inputs, outputs)

        with self.assertRaises(ValueError):
            validate_model_input_shape(model, None)


class OdomTCNFamilyExportTests(unittest.TestCase):
    """Validate TinyODOM-family export materialization behavior."""

    def setUp(self) -> None:
        """Prepare test fixtures."""
        self.family = OdomTCNFamily()
        self.ctx = ModelBuildContext(
            input_shape=(32, 6),
            input_dtype="float32",
            target_spec=TargetSpec(
                task_type="regression",
                output_names=["velx", "vely"],
                output_shapes=[(1,), (1,)],
                metadata={},
            ),
        )

    def tearDown(self) -> None:
        """Clean up test fixtures."""
        tf.keras.backend.clear_session()

    def test_materialize_export_model_perturbs_approx_trained_variants(self) -> None:
        # Approx-trained exports should apply the deterministic perturbation pass before materialization so the approximate variant differs from the pristine build.
        """Validate materialize export model perturbs approx trained variants."""
        fake_model = object()
        model_config = Dict()

        with patch.object(self.family, "build_model", return_value=fake_model) as build_mock, patch.object(
            odom_tcn_module,
            "apply_combined_perturbation",
            return_value=(2, 3),
        ) as perturb_mock:
            materialized = self.family.materialize_export_model(
                {"nb_filters": 8},
                self.ctx,
                model_config,
                model_variant="approx_trained",
            )

        build_mock.assert_called_once_with({"nb_filters": 8}, self.ctx, model_config)
        perturb_mock.assert_called_once_with(model=fake_model, seed=1337)
        self.assertIs(materialized, fake_model)

    def test_materialize_export_model_trained_variant_delegates_to_loader(self) -> None:
        # Trained exports should delegate to the checkpoint loader so the materialized model reflects saved weights instead of a fresh build.
        """Validate materialize export model trained variant delegates to loader."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "trained.keras"
            checkpoint_path.write_text("placeholder", encoding="utf-8")
            model_config = Dict()

            with patch.object(self.family, "load_model", return_value="loaded-model") as load_mock:
                materialized = self.family.materialize_export_model(
                    {"nb_filters": 8},
                    self.ctx,
                    model_config,
                    model_variant="trained",
                    checkpoint_path=checkpoint_path,
                )

        load_mock.assert_called_once_with(checkpoint_path, self.ctx, model_config)
        self.assertEqual(materialized, "loaded-model")


class CollectMetricsTests(unittest.TestCase):
    """Ensure controller plumbing and normalization behave as expected."""

    def test_proxy_metrics_normalize_none_values(self) -> None:
        """Proxy runs should convert None outputs into sentinel values."""
        # Proxy-mode metrics should normalize missing controller outputs into stable sentinels so downstream scoring and logging stay schema-safe.

        def fake_controller(run_hil: bool, **kwargs):
            # Simulate a proxy flow that only reports flash usage.
            """Return fake controller metrics for the HIL request.

            Parameters
            ----------
            run_hil : bool
                HIL execution function replaced by the test double.
            **kwargs : dict[str, object]
                Keyword arguments forwarded to the helper.

            Returns
            -------
            object
                Metrics returned by the fake HIL controller.
            """
            self.assertFalse(run_hil)
            self.assertEqual(kwargs["chosen_device"], "ARDUINO_NANO_33_BLE_SENSE")
            return (None, 4096, None, 2048, 0, None)

        with patch("tinyodom.hil_runtime.HIL_controller", fake_controller):
            request = CollectMetricsRequest(
                hil_enabled=False,
                energy_aware=False,
                flops=10_000_000,
                device_name="ARDUINO_NANO_33_BLE_SENSE",
                window_size=128,
                input_dim=6,
                dirpath=Path("odom_tcn"),
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
        """HIL metrics should normalize latency using the provided budget."""
        # HIL latency should be normalized against the provided budget so the metrics payload carries both the observation and its target.

        def fake_controller(run_hil: bool, **kwargs):
            """Return fake controller metrics for the HIL request.

            Parameters
            ----------
            run_hil : bool
                HIL execution function replaced by the test double.
            **kwargs : dict[str, object]
                Keyword arguments forwarded to the helper.

            Returns
            -------
            object
                Metrics returned by the fake HIL controller.
            """
            self.assertTrue(run_hil)
            self.assertEqual(kwargs["serial_port"], "ttyACM0")
            return (1024, 8192, 25.0, 4096, 0, None)

        with patch("tinyodom.hil_runtime.HIL_controller", fake_controller):
            request = CollectMetricsRequest(
                hil_enabled=True,
                energy_aware=False,
                flops=5_000_000,
                device_name="ARDUINO_NANO_33_BLE_SENSE",
                window_size=128,
                input_dim=6,
                dirpath=Path("odom_tcn"),
                latency_proxy_max_flops=20_000_000,
                serial_port="ttyACM0",
                latency_budget_ms=50000.0,
            )
            metrics = collect_metrics(request)

        self.assertEqual(metrics["latency_ms"], 25000.0)
        self.assertEqual(metrics["latency_budget_ms"], 50000.0)
        self.assertEqual(metrics["error_label"], "HIL_MASTER_PENDING")

    def test_hil_metrics_forward_serial_timeout_to_controller(self) -> None:
        """Direct-serial requests should forward ``serial_timeout_s``."""
        # Direct-serial HIL requests should forward timeout and measured-run settings unchanged to the controller.

        def fake_controller(run_hil: bool, **kwargs):
            """Return fake controller metrics for the HIL request.

            Parameters
            ----------
            run_hil : bool
                HIL execution function replaced by the test double.
            **kwargs : dict[str, object]
                Keyword arguments forwarded to the helper.

            Returns
            -------
            object
                Metrics returned by the fake HIL controller.
            """
            self.assertTrue(run_hil)
            self.assertEqual(kwargs["serial_timeout_s"], 9.5)
            self.assertEqual(kwargs["measured_inference_runs"], 7)
            return (1024, 8192, 0.025, 4096, 0, None)

        with patch("tinyodom.hil_runtime.HIL_controller", fake_controller):
            request = CollectMetricsRequest(
                hil_enabled=True,
                energy_aware=False,
                flops=5_000_000,
                device_name="STM32_NUCLEO_N657X0_Q",
                window_size=128,
                input_dim=6,
                dirpath=Path("odom_tcn"),
                latency_proxy_max_flops=20_000_000,
                serial_port="ttyACM0",
                latency_budget_ms=200.0,
                serial_timeout_s=9.5,
                measured_inference_runs=7,
            )
            metrics = collect_metrics(request)

        self.assertEqual(metrics["error_code"], 0)

    def test_collect_metrics_preserves_backend_error_fields(self) -> None:
        """STM backend detail fields should survive final metric normalization."""
        # Backend-specific error fields should survive metric normalization so callers can surface the exact hardware-side failure.

        def fake_controller(run_hil: bool, **kwargs):
            """Return fake controller metrics for the HIL request.

            Parameters
            ----------
            run_hil : bool
                HIL execution function replaced by the test double.
            **kwargs : dict[str, object]
                Keyword arguments forwarded to the helper.

            Returns
            -------
            object
                Metrics returned by the fake HIL controller.
            """
            del kwargs
            self.assertTrue(run_hil)
            return (
                1024,
                8192,
                None,
                4096,
                3,
                {
                    "backend_error_kind": "runtime_timeout",
                    "backend_error_detail": "Timed out waiting for DUT READY.",
                    "external_flash_bytes": 2048,
                    "weight_storage_mode": "external_flash",
                },
            )

        with patch("tinyodom.hil_runtime.HIL_controller", fake_controller):
            request = CollectMetricsRequest(
                hil_enabled=True,
                energy_aware=False,
                flops=5_000_000,
                device_name="STM32_NUCLEO_N657X0_Q",
                window_size=128,
                input_dim=6,
                dirpath=Path("odom_tcn"),
                latency_proxy_max_flops=20_000_000,
                serial_port="ttyACM0",
                latency_budget_ms=200.0,
            )
            metrics = collect_metrics(request)

        self.assertEqual(metrics["backend_error_kind"], "runtime_timeout")
        self.assertEqual(metrics["backend_error_detail"], "Timed out waiting for DUT READY.")
        self.assertEqual(metrics["external_flash_bytes"], 2048)
        self.assertEqual(metrics["weight_storage_mode"], "external_flash")

    def test_collect_metrics_copies_cadenced_metrics_and_runtime_mode(self) -> None:
        """Cadenced STM32 extras should survive normalization into final metrics."""
        # Cadenced backend extras should survive normalization so scorers and logs can distinguish single-pass and cadenced runs.

        def fake_controller(run_hil: bool, **kwargs):
            """Return fake controller metrics for the HIL request.

            Parameters
            ----------
            run_hil : bool
                HIL execution function replaced by the test double.
            **kwargs : dict[str, object]
                Keyword arguments forwarded to the helper.

            Returns
            -------
            object
                Metrics returned by the fake HIL controller.
            """
            del kwargs
            self.assertTrue(run_hil)
            return (
                1024,
                8192,
                0.025,
                4096,
                0,
                {
                    "runtime_mode": "cadenced",
                    "cadenced_error_code": 0,
                    "cadenced_active_inference_latency_ms": 80.0,
                    "cadenced_window_latency_ms": 20000.0,
                    "cadenced_energy_mj_per_window": 1.25,
                    "cadenced_energy_mj_per_trial": 12.5,
                    "cadenced_rtc_sleep_ms": 1500.0,
                    "cadenced_deadline_miss_count": 0,
                    "cadenced_rtc_clock_source": "LSE",
                    "cadenced_timing_quality": "crystal",
                    "cadenced_stop_mode_variant": "system_stop_mainreg_wfi",
                },
            )

        with patch("tinyodom.hil_runtime.HIL_controller", fake_controller):
            request = CollectMetricsRequest(
                hil_enabled=True,
                energy_aware=False,
                flops=5_000_000,
                device_name="STM32_NUCLEO_N657X0_Q",
                window_size=128,
                input_dim=6,
                dirpath=Path("odom_tcn"),
                latency_proxy_max_flops=20_000_000,
                serial_port="ttyACM0",
                latency_budget_ms=200.0,
            )
            metrics = collect_metrics(request)

        self.assertEqual(metrics["runtime_mode"], "cadenced")
        self.assertAlmostEqual(metrics["cadenced_active_inference_latency_ms"], 80.0)
        self.assertAlmostEqual(metrics["cadenced_window_latency_ms"], 20000.0)
        self.assertAlmostEqual(metrics["cadenced_energy_mj_per_window"], 1.25)
        self.assertAlmostEqual(metrics["cadenced_energy_mj_per_trial"], 12.5)
        self.assertEqual(metrics["cadenced_deadline_miss_count"], 0)
        self.assertEqual(metrics["cadenced_error_label"], "HIL_ERROR_OK")

    def test_collect_metrics_back_to_back_mode_emits_cadenced_sentinels(self) -> None:
        """Single-pass mode should keep cadenced keys present with sentinel values."""
        # Back-to-back mode should still emit cadenced sentinel fields so downstream logs and scorers keep a stable schema.

        def fake_controller(run_hil: bool, **kwargs):
            """Return fake controller metrics for the HIL request.

            Parameters
            ----------
            run_hil : bool
                HIL execution function replaced by the test double.
            **kwargs : dict[str, object]
                Keyword arguments forwarded to the helper.

            Returns
            -------
            object
                Metrics returned by the fake HIL controller.
            """
            del kwargs
            self.assertTrue(run_hil)
            return (1024, 8192, 0.025, 4096, 0, {"runtime_mode": "back_to_back"})

        with patch("tinyodom.hil_runtime.HIL_controller", fake_controller):
            request = CollectMetricsRequest(
                hil_enabled=True,
                energy_aware=False,
                flops=5_000_000,
                device_name="STM32_NUCLEO_N657X0_Q",
                window_size=128,
                input_dim=6,
                dirpath=Path("odom_tcn"),
                latency_proxy_max_flops=20_000_000,
                serial_port="ttyACM0",
                latency_budget_ms=200.0,
            )
            metrics = collect_metrics(request)

        self.assertEqual(metrics["runtime_mode"], "back_to_back")
        self.assertEqual(metrics["cadenced_active_inference_latency_ms"], -1.0)
        self.assertIsNone(metrics["cadenced_rtc_clock_source"])
        self.assertEqual(metrics["cadenced_error_code"], -1)

    def test_collect_metrics_normalizes_runtime_mode_from_backend_payload(self) -> None:
        """Runtime mode should be normalized and default invalid values safely."""
        # Runtime-mode strings should normalize into one canonical vocabulary before scoring and CSV logging consume them.

        cases = [
            ("CADENCED", "cadenced"),
            ("  back_to_back  ", "back_to_back"),
            (None, "back_to_back"),
            ("", "back_to_back"),
            ("unexpected", "back_to_back"),
        ]

        for raw_runtime_mode, expected_runtime_mode in cases:
            with self.subTest(raw_runtime_mode=raw_runtime_mode):
                def fake_controller(run_hil: bool, **kwargs):
                    """Return fake controller metrics for the HIL request.

                    Parameters
                    ----------
                    run_hil : bool
                        HIL execution function replaced by the test double.
                    **kwargs : dict[str, object]
                        Keyword arguments forwarded to the helper.

                    Returns
                    -------
                    object
                        Metrics returned by the fake HIL controller.
                    """
                    del kwargs
                    self.assertTrue(run_hil)
                    return (1024, 8192, 0.025, 4096, 0, {"runtime_mode": raw_runtime_mode})

                with patch("tinyodom.hil_runtime.HIL_controller", fake_controller):
                    request = CollectMetricsRequest(
                        hil_enabled=True,
                        energy_aware=False,
                        flops=5_000_000,
                        device_name="STM32_NUCLEO_N657X0_Q",
                        window_size=128,
                        input_dim=6,
                        dirpath=Path("odom_tcn"),
                        latency_proxy_max_flops=20_000_000,
                        serial_port="ttyACM0",
                        latency_budget_ms=200.0,
                    )
                    metrics = collect_metrics(request)

                self.assertEqual(metrics["runtime_mode"], expected_runtime_mode)

    def test_energy_aware_harness_fields_forwarded_to_controller(self) -> None:
        """Energy-aware requests should forward harness settings to HIL controller."""
        # Energy-aware requests should forward the full harness contract so the controller measures the same timing window the config requested.

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
            """Return fake controller metrics for the HIL request.

            Parameters
            ----------
            run_hil : bool
                HIL execution function replaced by the test double.
            **kwargs : dict[str, object]
                Keyword arguments forwarded to the helper.

            Returns
            -------
            object
                Metrics returned by the fake HIL controller.
            """
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

        with patch("tinyodom.hil_runtime.HIL_controller", fake_controller):
            request = CollectMetricsRequest(
                hil_enabled=True,
                energy_aware=True,
                flops=5_000_000,
                device_name="ARDUINO_NANO_33_BLE_SENSE",
                window_size=128,
                input_dim=6,
                dirpath=Path("odom_tcn"),
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
        # Energy-aware HIL requests should fail immediately when the harness configuration is missing, because there is no safe fallback path.
        """Validate energy aware without harness raises."""
        request = CollectMetricsRequest(
            hil_enabled=True,
            energy_aware=True,
            flops=5_000_000,
            device_name="ARDUINO_NANO_33_BLE_SENSE",
            window_size=128,
            input_dim=6,
            dirpath=Path("odom_tcn"),
            latency_proxy_max_flops=20_000_000,
            serial_port="ttyACM0",
            latency_budget_ms=50000.0,
            harness=None,
        )

        with self.assertRaises(RuntimeError):
            collect_metrics(request)

    def test_collect_metrics_forwards_device_options(self) -> None:
        # Resolved device options should flow through collect_metrics so backend-specific board settings survive request normalization.
        """Validate collect metrics forwards device options."""
        def fake_controller(run_hil: bool, **kwargs):
            """Return fake controller metrics for the HIL request.

            Parameters
            ----------
            run_hil : bool
                HIL execution function replaced by the test double.
            **kwargs : dict[str, object]
                Keyword arguments forwarded to the helper.

            Returns
            -------
            object
                Metrics returned by the fake HIL controller.
            """
            self.assertFalse(run_hil)
            self.assertEqual(kwargs["device_options"]["target_core"], "cm7")
            self.assertEqual(kwargs["device_options"]["split"], "75_25")
            return (1024, 2048, None, 4096, 0, None)

        with patch("tinyodom.hil_runtime.HIL_controller", fake_controller):
            request = CollectMetricsRequest(
                hil_enabled=False,
                energy_aware=False,
                flops=10_000_000,
                device_name="PORTENTA_H7",
                window_size=128,
                input_dim=6,
                dirpath=Path("odom_tcn"),
                latency_proxy_max_flops=20_000_000,
                serial_port=None,
                latency_budget_ms=50.0,
                device_options={"target_core": "cm7", "split": "75_25"},
            )
            metrics = collect_metrics(request)

        self.assertEqual(metrics["error_code"], 0)

    def test_collect_metrics_portenta_cm4_runtime_requires_harness(self) -> None:
        # Portenta CM4 runtime mode depends on the harness channel because the DUT cannot report that path directly.
        """Validate collect metrics Portenta CM4 runtime requires harness."""
        request = CollectMetricsRequest(
            hil_enabled=True,
            energy_aware=False,
            flops=5_000_000,
            device_name="PORTENTA_H7",
            window_size=128,
            input_dim=6,
            dirpath=Path("odom_tcn"),
            latency_proxy_max_flops=20_000_000,
            serial_port="ttyACM0",
            latency_budget_ms=200.0,
            device_options={"target_core": "cm4", "split": "50_50", "security": "none"},
        )

        with patch("tinyodom.hil_runtime.HIL_controller") as controller_mock:
            with self.assertRaises(RuntimeError) as context:
                collect_metrics(request)

        self.assertIn("harness", str(context.exception).lower())
        controller_mock.assert_not_called()

    def test_collect_metrics_portenta_cm4_non_energy_forwards_harness(self) -> None:
        # Portenta CM4 runtime requests should still forward harness settings even when energy measurement itself is disabled.
        """Validate collect metrics Portenta CM4 non-energy forwards harness."""
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
            """Return fake controller metrics for the HIL request.

            Parameters
            ----------
            run_hil : bool
                HIL execution function replaced by the test double.
            **kwargs : dict[str, object]
                Keyword arguments forwarded to the helper.

            Returns
            -------
            object
                Metrics returned by the fake HIL controller.
            """
            self.assertTrue(run_hil)
            self.assertEqual(kwargs["harness_serial_port"], "ttyACM1")
            self.assertEqual(kwargs["harness_done_timeout_s"], 3.6)
            return (1024, 2048, 0.01, 4096, 0, None)

        with patch("tinyodom.hil_runtime.HIL_controller", fake_controller):
            request = CollectMetricsRequest(
                hil_enabled=True,
                energy_aware=False,
                flops=5_000_000,
                device_name="PORTENTA_H7",
                window_size=128,
                input_dim=6,
                dirpath=Path("odom_tcn"),
                latency_proxy_max_flops=20_000_000,
                serial_port="ttyACM0",
                latency_budget_ms=200.0,
                harness=harness,
                device_options={"target_core": "cm4", "split": "50_50", "security": "none"},
            )
            metrics = collect_metrics(request)

        self.assertEqual(metrics["error_code"], 0)

    def test_metric_unavailable_treats_negative_cadenced_sentinel_as_missing(self) -> None:
        # Cadenced sentinel values must be treated as unavailable so failed measurements do not look like real wins during scoring.
        """Validate metric unavailable treats negative cadenced sentinel as missing."""
        self.assertTrue(_metric_unavailable("cadenced_energy_mj_per_trial", -1.0))
        self.assertFalse(_metric_unavailable("cadenced_energy_mj_per_trial", 0.0))
        self.assertTrue(_metric_unavailable("cadenced_rtc_sleep_ms", -1.0))
        self.assertTrue(_metric_unavailable("cadenced_deadline_miss_count", -1))
        self.assertTrue(_metric_unavailable("cadenced_window_latency_ms", -1.0))


class BuildCollectMetricsRequestTests(unittest.TestCase):
    """Validate config/hyperparameter mapping into CollectMetricsRequest."""

    _DEFAULT_DIRPATH = Path("odom_tcn")

    def _build_request(
        self,
        config: Dict,
        runtime_metadata: Dict,
        *,
        latency_budget_ms: float = 200.0,
        dirpath: Path | None = None,
        device_options: dict | None | object = ...,
        hil_enabled: bool | None = None,
        energy_aware: bool | None = None,
        window_size: int | None = None,
        input_dim: int | None = None,
    ) -> CollectMetricsRequest:
        """Build a request with resolved backend options for test readability.

        Parameters
        ----------
        config : addict.Dict
            NAS/runtime configuration under test.
        runtime_metadata : addict.Dict
            Runtime metadata payload forwarded into request construction.

        Returns
        -------
        CollectMetricsRequest
            Request built with either explicit or auto-resolved backend options.
        """
        resolved_options = (
            resolve_device_options(str(config.device.name), config.device)
            if device_options is ...
            else device_options
        )
        return build_collect_metrics_request(
            config,
            runtime_metadata,
            latency_budget_ms=latency_budget_ms,
            dirpath=self._DEFAULT_DIRPATH if dirpath is None else dirpath,
            device_options=resolved_options,
            hil_enabled=hil_enabled,
            energy_aware=energy_aware,
            window_size=window_size,
            input_dim=input_dim,
        )

    def test_non_energy_aware_sets_harness_none(self) -> None:
        # Non-energy-aware requests should leave harness wiring unset so plain latency runs do not stage extra harness state.
        """Validate non-energy aware sets harness none."""
        config = Dict(
            training=Dict(energy_aware=False, latency_proxy_max_flops=20_000_000),
            device=Dict(hil=True, name="ARDUINO_NANO_33_BLE_SENSE", serial_port="ttyACM0"),
            dataset=Dict(params=Dict(window_size=128)),
            outputs=Dict(candidate_dir=Path("odom_tcn")),
        )
        hyperparams = Dict(flops=123, input_dim=6)

        request = self._build_request(config, hyperparams)

        self.assertIsNone(request.harness)
        self.assertFalse(request.energy_aware)
        self.assertEqual(request.flops, 123)
        self.assertEqual(request.input_dim, 6)
        self.assertEqual(request.measured_inference_runs, 10)

    def test_request_uses_configured_measured_inference_runs(self) -> None:
        # Request construction should honor the configured measured-run count so runtime measurements use the intended averaging window.
        """Validate request uses configured measured inference runs."""
        config = Dict(
            training=Dict(energy_aware=False, latency_proxy_max_flops=20_000_000),
            device=Dict(
                hil=True,
                name="ARDUINO_NANO_33_BLE_SENSE",
                serial_port="ttyACM0",
                measured_inference_runs=7,
            ),
            dataset=Dict(params=Dict(window_size=128)),
            outputs=Dict(candidate_dir=Path("odom_tcn")),
        )
        hyperparams = Dict(flops=123, input_dim=6)

        request = self._build_request(config, hyperparams)

        self.assertEqual(request.measured_inference_runs, 7)

    def test_explicit_window_size_and_input_dim_override_legacy_fallbacks(self) -> None:
        # Explicit request dimensions should win over legacy dataset-derived defaults so migrated configs do not silently change shape.
        """Validate explicit window size and input dim override legacy fallbacks."""
        config = Dict(
            training=Dict(energy_aware=False, latency_proxy_max_flops=20_000_000),
            device=Dict(hil=True, name="ARDUINO_NANO_33_BLE_SENSE", serial_port="ttyACM0"),
            dataset=Dict(params=Dict(window_size=128)),
            outputs=Dict(candidate_dir=Path("odom_tcn")),
        )
        hyperparams = Dict(flops=123, input_dim=6)

        request = self._build_request(
            config,
            hyperparams,
            window_size=32,
            input_dim=3,
        )

        self.assertEqual(request.window_size, 32)
        self.assertEqual(request.input_dim, 3)

    def test_build_request_accepts_plain_dict_config_and_runtime_metadata(self) -> None:
        # Request construction should accept plain dict payloads, not just addict.Dict wrappers.
        """Validate build request accepts plain dict config and runtime metadata."""
        config = {
            "training": {"energy_aware": False, "latency_proxy_max_flops": 20_000_000},
            "device": {"hil": True, "name": "ARDUINO_NANO_33_BLE_SENSE", "serial_port": "ttyACM0"},
            "dataset": {"params": {"window_size": 128}},
            "outputs": {"candidate_dir": Path("odom_tcn")},
        }
        runtime_metadata = {"flops": 123.0, "input_dim": 6}

        request = self._build_request(config, runtime_metadata, device_options={})

        self.assertEqual(request.window_size, 128)
        self.assertEqual(request.input_dim, 6)
        self.assertEqual(request.flops, 123.0)

    def test_build_request_raises_value_error_for_missing_runtime_input_dim(self) -> None:
        # Missing runtime input_dim should surface as ValueError instead of leaking AttributeError.
        """Validate build request raises value error for missing runtime input dim."""
        config = Dict(
            training=Dict(energy_aware=False, latency_proxy_max_flops=20_000_000),
            device=Dict(hil=True, name="ARDUINO_NANO_33_BLE_SENSE", serial_port="ttyACM0"),
            dataset=Dict(params=Dict(window_size=128)),
            outputs=Dict(candidate_dir=Path("odom_tcn")),
        )
        runtime_metadata = Dict(flops=123.0)

        with self.assertRaisesRegex(ValueError, "input_dim"):
            self._build_request(config, runtime_metadata, device_options={})

    def test_build_request_defaults_stm_serial_timeout(self) -> None:
        # STM32 requests should fall back to the backend minimum serial timeout so short configs cannot under-budget board bring-up.
        """Validate build request defaults stm serial timeout."""
        config = Dict(
            training=Dict(energy_aware=False, latency_proxy_max_flops=20_000_000),
            device=Dict(
                hil=True,
                name="STM32_NUCLEO_N657X0_Q",
                serial_port="ttyACM0",
                stm32=Dict(project_root=str(ROOT_DIR / "sketches" / "stm32" / "tinyodom_stm32_lrun")),
            ),
            dataset=Dict(params=Dict(window_size=128)),
            outputs=Dict(candidate_dir=Path("odom_tcn")),
        )
        hyperparams = Dict(flops=123, input_dim=6)

        request = self._build_request(
            config,
            hyperparams,
            device_options={"project_root": ROOT_DIR / "sketches" / "stm32" / "tinyodom_stm32_lrun"},
        )

        self.assertEqual(request.serial_timeout_s, 30.0)

    def test_build_request_scales_stm_serial_timeout_for_cadenced_runs(self) -> None:
        # Cadenced STM32 runs need a longer serial timeout so the second-pass window does not look like a board hang.
        """Validate build request scales stm serial timeout for cadenced runs."""
        config = Dict(
            training=Dict(energy_aware=False, latency_proxy_max_flops=20_000_000),
            device=Dict(
                hil=True,
                name="STM32_NUCLEO_N657X0_Q",
                runtime_mode="cadenced",
                serial_port="ttyACM0",
                measured_inference_runs=100,
                latency_budget_ms=2000.0,
                stm32=Dict(project_root=str(ROOT_DIR / "sketches" / "stm32" / "tinyodom_stm32_lrun")),
            ),
            dataset=Dict(params=Dict(window_size=128)),
            outputs=Dict(candidate_dir=Path("odom_tcn")),
        )
        hyperparams = Dict(flops=123, input_dim=6)

        request = self._build_request(
            config,
            hyperparams,
            latency_budget_ms=2000.0,
            device_options={"project_root": ROOT_DIR / "sketches" / "stm32" / "tinyodom_stm32_lrun"},
        )

        self.assertEqual(request.serial_timeout_s, 210.0)

    def test_build_request_preserves_larger_stm_serial_timeout(self) -> None:
        # Explicit STM32 serial timeouts should win when they already exceed the backend minimum, so caller tuning is not silently reduced.
        """Validate build request preserves larger stm serial timeout."""
        config = Dict(
            training=Dict(energy_aware=False, latency_proxy_max_flops=20_000_000),
            device=Dict(
                hil=True,
                name="STM32_NUCLEO_N657X0_Q",
                runtime_mode="back_to_back",
                serial_port="ttyACM0",
                serial_timeout_s=120.0,
                stm32=Dict(project_root=str(ROOT_DIR / "sketches" / "stm32" / "tinyodom_stm32_lrun")),
            ),
            dataset=Dict(params=Dict(window_size=128)),
            outputs=Dict(candidate_dir=Path("odom_tcn")),
        )
        hyperparams = Dict(flops=123, input_dim=6)

        request = self._build_request(
            config,
            hyperparams,
            device_options={"project_root": ROOT_DIR / "sketches" / "stm32" / "tinyodom_stm32_lrun"},
        )

        self.assertEqual(request.serial_timeout_s, 120.0)


class Stm32TimeoutHelperTests(unittest.TestCase):
    """Tests covering STM32 timeout helper behavior."""

    _DEFAULT_DIRPATH = Path("odom_tcn")

    def _build_request(
        self,
        config: Dict,
        runtime_metadata: Dict,
        *,
        latency_budget_ms: float = 200.0,
        dirpath: Path | None = None,
        device_options: dict | None | object = ...,
        hil_enabled: bool | None = None,
        energy_aware: bool | None = None,
    ) -> CollectMetricsRequest:
        """Build a request for STM32 timeout-specific request tests.

        Parameters
        ----------
        config : Dict
            Configuration object used by the helper.
        runtime_metadata : Dict
            Runtime metadata included in the HIL request.
        latency_budget_ms : float
            Latency budget in milliseconds for the HIL evaluation.
        dirpath : Path | None
            Directory path created by the temporary-directory helper.
        device_options : dict | None | object
            Device option overrides used for the HIL run.
        hil_enabled : bool | None
            HIL-enabled flag included in the request under test.
        energy_aware : bool | None
            Energy-aware flag included in the request under test.

        Returns
        -------
        CollectMetricsRequest
            Constructed request.
        """
        resolved_options = (
            resolve_device_options(str(config.device.name), config.device)
            if device_options is ...
            else device_options
        )
        return build_collect_metrics_request(
            config,
            runtime_metadata,
            latency_budget_ms=latency_budget_ms,
            dirpath=self._DEFAULT_DIRPATH if dirpath is None else dirpath,
            device_options=resolved_options,
            hil_enabled=hil_enabled,
            energy_aware=energy_aware,
        )

    def test_minimum_stm32_serial_timeout_is_30s_for_back_to_back(self) -> None:
        # Back-to-back STM32 runs should never budget less than the backend bring-up minimum for serial timeouts.
        """Validate minimum stm32 serial timeout is 30s for back to back."""
        self.assertEqual(
            _minimum_stm32_serial_timeout_s(
                runtime_mode="back_to_back",
                latency_budget_ms=200.0,
                measured_inference_runs=10,
            ),
            30.0,
        )

    def test_minimum_stm32_serial_timeout_scales_for_cadenced(self) -> None:
        # Cadenced STM32 runs should scale their serial timeout with window duration and measured-run count.
        """Validate minimum stm32 serial timeout scales for cadenced."""
        self.assertEqual(
            _minimum_stm32_serial_timeout_s(
                runtime_mode="cadenced",
                latency_budget_ms=2000.0,
                measured_inference_runs=100,
            ),
            210.0,
        )

    def test_energy_aware_populates_harness(self) -> None:
        # Energy-aware request building should materialize a full HarnessConfig so the controller sees the deployment wiring explicitly.
        """Validate energy aware populates harness."""
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
            dataset=Dict(params=Dict(window_size=128)),
            outputs=Dict(candidate_dir=Path("odom_tcn")),
        )
        hyperparams = Dict(flops=123, input_dim=6)

        request = self._build_request(config, hyperparams)

        self.assertTrue(request.energy_aware)
        self.assertIsInstance(request.harness, HarnessConfig)
        self.assertEqual(request.harness.harness_serial_port, "ttyACM1")
        self.assertEqual(request.harness.harness_arm_timeout_s, 5.0)

    def test_missing_dut_ready_timeout_uses_default(self) -> None:
        # Missing DUT-ready timeouts should fall back to the documented default so configs do not need to restate the common case.
        """Validate missing dut ready timeout uses default."""
        config = Dict(
            training=Dict(energy_aware=False, latency_proxy_max_flops=20_000_000),
            device=Dict(hil=True, name="ARDUINO_NANO_33_BLE_SENSE", serial_port="ttyACM0"),
            dataset=Dict(params=Dict(window_size=128)),
            outputs=Dict(candidate_dir=Path("odom_tcn")),
        )
        hyperparams = Dict(flops=123, input_dim=6)

        request = self._build_request(config, hyperparams)

        self.assertEqual(request.dut_ready_timeout_s, 5.0)

    def test_energy_aware_missing_harness_serial_port_raises(self) -> None:
        # Energy-aware configs must provide a harness serial port because the measurement path cannot recover that wiring later.
        """Validate energy aware missing harness serial port raises."""
        config = Dict(
            training=Dict(energy_aware=True, latency_proxy_max_flops=20_000_000),
            device=Dict(hil=True, name="ARDUINO_NANO_33_BLE_SENSE", serial_port="ttyACM0"),
            dataset=Dict(params=Dict(window_size=128)),
            outputs=Dict(candidate_dir=Path("odom_tcn")),
        )
        hyperparams = Dict(flops=123, input_dim=6)

        with self.assertRaises(RuntimeError):
            self._build_request(config, hyperparams)

    def test_portenta_cm4_runtime_missing_harness_serial_port_raises(self) -> None:
        # Portenta CM4 runtime mode must have a harness serial port because the CM4 path relies on harness-mediated execution.
        """Validate Portenta CM4 runtime missing harness serial port raises."""
        config = Dict(
            training=Dict(energy_aware=False, latency_proxy_max_flops=20_000_000),
            device=Dict(
                hil=True,
                name="PORTENTA_H7",
                serial_port="ttyACM0",
                portenta=Dict(target_core="cm4", split="50_50", security="none"),
            ),
            dataset=Dict(params=Dict(window_size=128)),
            outputs=Dict(candidate_dir=Path("odom_tcn")),
        )
        hyperparams = Dict(flops=123, input_dim=6)

        with self.assertRaises(RuntimeError):
            self._build_request(
                config,
                hyperparams,
                device_options={"target_core": "cm4", "split": "50_50", "security": "none"},
            )

    def test_resolve_device_options_validates_portenta_target_core(self) -> None:
        # Portenta device-option resolution should require an explicit target core so the build never guesses between CM4 and CM7.
        """Validate resolve device options validates Portenta target core."""
        config = Dict(
            training=Dict(energy_aware=False, latency_proxy_max_flops=20_000_000),
            device=Dict(hil=True, name="PORTENTA_H7", serial_port="ttyACM0", portenta=Dict()),
            dataset=Dict(params=Dict(window_size=128)),
            outputs=Dict(candidate_dir=Path("odom_tcn")),
        )
        with self.assertRaises(ValueError):
            resolve_device_options(str(config.device.name), config.device)

    def test_portenta_options_are_forwarded(self) -> None:
        # Portenta runtime options should survive request construction so downstream hardware helpers see the selected split and security mode.
        """Validate Portenta options are forwarded."""
        config = Dict(
            training=Dict(energy_aware=False, latency_proxy_max_flops=20_000_000),
            device=Dict(
                hil=False,
                name="PORTENTA_H7",
                serial_port="ttyACM0",
                portenta=Dict(target_core="cm4", split="50_50", security="none"),
            ),
            dataset=Dict(params=Dict(window_size=128)),
            outputs=Dict(candidate_dir=Path("odom_tcn")),
        )
        hyperparams = Dict(flops=123, input_dim=6)

        request = self._build_request(config, hyperparams)

        self.assertEqual(request.device_options["target_core"], "cm4")
        self.assertEqual(request.device_options["split"], "50_50")
        self.assertEqual(request.device_options["security"], "none")

    def test_portenta_cm4_runtime_populates_harness_even_without_energy_aware(self) -> None:
        # Portenta CM4 runtime mode should still materialize the harness path even when energy accounting is off.
        """Validate Portenta CM4 runtime populates harness even without energy aware."""
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
            dataset=Dict(params=Dict(window_size=128)),
            outputs=Dict(candidate_dir=Path("odom_tcn")),
        )
        hyperparams = Dict(flops=123, input_dim=6)

        request = self._build_request(
            config,
            hyperparams,
            device_options={"target_core": "cm4", "split": "50_50", "security": "none"},
        )

        self.assertIsNotNone(request.harness)
        self.assertEqual(request.harness.harness_serial_port, "ttyACM1")

    def test_device_name_is_normalized_before_request_build(self) -> None:
        # Request construction should normalize device names before looking up backend-specific defaults.
        """Validate device name is normalized before request build."""
        config = Dict(
            training=Dict(energy_aware=False, latency_proxy_max_flops=20_000_000),
            device=Dict(
                hil=False,
                name="portenta_h7",
                portenta=Dict(target_core="cm7", split="75_25", security="none"),
            ),
            dataset=Dict(params=Dict(window_size=128)),
            outputs=Dict(candidate_dir=Path("odom_tcn")),
        )
        hyperparams = Dict(flops=123, input_dim=6)

        request = self._build_request(config, hyperparams)

        self.assertEqual(request.device_name, "PORTENTA_H7")
        self.assertEqual(request.device_options["target_core"], "cm7")

    def test_proxy_mode_allows_missing_serial_port(self) -> None:
        # Proxy-mode requests should allow the serial port to stay unset because no hardware round trip is expected.
        """Validate proxy mode allows missing serial port."""
        config = Dict(
            training=Dict(energy_aware=False, latency_proxy_max_flops=20_000_000),
            device=Dict(hil=False, name="ARDUINO_NANO_33_BLE_SENSE"),
            dataset=Dict(params=Dict(window_size=128)),
            outputs=Dict(candidate_dir=Path("odom_tcn")),
        )
        hyperparams = Dict(flops=123, input_dim=6)

        request = self._build_request(config, hyperparams)

        self.assertIsNone(request.serial_port)

    def test_stm_request_uses_caller_supplied_dirpath_and_options(self) -> None:
        """Ensure generic request building uses caller-supplied STM fields.

        Returns
        -------
        None
        """
        # STM request construction should preserve caller-supplied staging paths and board options instead of recomputing them.
        config = Dict(
            training=Dict(energy_aware=True, latency_proxy_max_flops=20_000_000),
            device=Dict(
                hil=True,
                name="STM32_NUCLEO_N657X0_Q",
                serial_port="ttyACM0",
                stm32=Dict(
                    project_root=Path("/tmp/stm32_fsbl"),
                    gdb_port=61234,
                    apid=1,
                    server_ready_timeout_s=15.0,
                ),
            ),
            dataset=Dict(params=Dict(window_size=128)),
            outputs=Dict(candidate_dir=Path("odom_tcn")),
        )
        hyperparams = Dict(flops=123, input_dim=6)

        request = self._build_request(
            config,
            hyperparams,
            dirpath=Path("/tmp/stm32_fsbl"),
            device_options={"project_root": Path("/tmp/stm32_fsbl"), "cpu_clock_mhz": 400},
            energy_aware=False,
            hil_enabled=False,
        )

        self.assertEqual(request.device_name, "STM32_NUCLEO_N657X0_Q")
        self.assertEqual(request.dirpath, Path("/tmp/stm32_fsbl"))
        self.assertFalse(request.energy_aware)
        self.assertIsNone(request.harness)
        self.assertEqual(request.device_options["cpu_clock_mhz"], 400)

    def test_resolve_device_options_defaults_stm_template_root(self) -> None:
        """Ensure STM config resolution no longer requires an explicit project root.

        Returns
        -------
        None
        """
        # STM32 option resolution should synthesize the default template root so callers are not forced to provide it explicitly.
        config = Dict(
            training=Dict(energy_aware=False, latency_proxy_max_flops=20_000_000),
            device=Dict(hil=False, name="STM32_NUCLEO_N657X0_Q"),
            dataset=Dict(params=Dict(window_size=128)),
            outputs=Dict(candidate_dir=Path("odom_tcn")),
        )

        resolved = resolve_device_options(str(config.device.name), config.device)

        self.assertIn("project_root", resolved)
        self.assertEqual(resolved["cpu_clock_mhz"], 600)

    def test_resolve_device_options_supports_full_canonical_stm_backend_block(self) -> None:
        """Ensure the full canonical STM config surface resolves predictably.

        Returns
        -------
        None
        """
        # Canonical STM32 backend blocks should round-trip through option resolution without losing fields.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            template_root = tmp_path / "stm32_project" / "FSBL"
            (template_root / "Debug").mkdir(parents=True)
            (template_root / "Debug" / "makefile").write_text("# makefile\n", encoding="utf-8")
            weights_memory_pool = tmp_path / "nucleo_mypool.json"
            weights_memory_pool.write_text("{}\n", encoding="utf-8")
            weights_external_loader = tmp_path / "mx25um51245g.stldr"
            weights_external_loader.write_text("loader\n", encoding="utf-8")

            config = Dict(
                training=Dict(energy_aware=False, latency_proxy_max_flops=20_000_000),
                device=Dict(
                    hil=False,
                    name="STM32_NUCLEO_N657X0_Q",
                    stm32=Dict(
                        template_root=template_root,
                        gdb_port=61235,
                        apid=2,
                        server_ready_timeout_s=20.0,
                        weight_storage_mode="external_flash",
                        weights_flash_address="0x71000000",
                        weights_memory_pool=weights_memory_pool,
                        weights_external_loader=weights_external_loader,
                        max_external_flash_bytes=123456,
                    ),
                ),
                dataset=Dict(params=Dict(window_size=128)),
                outputs=Dict(candidate_dir=Path("odom_tcn")),
            )

            resolved = resolve_device_options(str(config.device.name), config.device)

        self.assertEqual(resolved["project_root"], template_root.resolve())
        self.assertEqual(resolved["gdb_port"], 61235)
        self.assertEqual(resolved["apid"], 2)
        self.assertEqual(resolved["server_ready_timeout_s"], 20.0)
        self.assertEqual(resolved["cpu_clock_mhz"], 600)
        self.assertEqual(resolved["weight_storage_mode"], "external_flash")
        self.assertEqual(resolved["weights_flash_address"], "0x71000000")
        self.assertEqual(resolved["weights_memory_pool"], weights_memory_pool.resolve())
        self.assertEqual(resolved["weights_external_loader"], weights_external_loader.resolve())
        self.assertEqual(resolved["max_external_flash_bytes"], 123456)

    def test_resolve_device_options_accepts_unmaterialized_custom_stm_root(self) -> None:
        """Ensure custom STM roots resolve before the LRUN workspace is materialized."""
        # Custom STM32 project roots should resolve before the workspace is materialized so staging can create them later.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            unresolved_root = tmp_path / "stm32_project"
            config = Dict(
                training=Dict(energy_aware=False, latency_proxy_max_flops=20_000_000),
                device=Dict(
                    hil=False,
                    name="STM32_NUCLEO_N657X0_Q",
                    stm32=Dict(project_root=unresolved_root),
                ),
                dataset=Dict(params=Dict(window_size=128)),
                outputs=Dict(candidate_dir=Path("odom_tcn")),
            )

            resolved = resolve_device_options(str(config.device.name), config.device)

        self.assertEqual(resolved["project_root"], unresolved_root.resolve())

    def test_resolve_device_options_normalizes_custom_lrun_project_root_without_layout(self) -> None:
        """Ensure custom LRUN roots infer the dev-boot layout automatically."""
        # Custom LRUN roots should normalize cleanly even before the standard folder layout exists on disk.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            project_root = tmp_path / "tinyodom_stm32_lrun"
            for required_dir in (
                project_root / "FSBL",
                project_root / "Appli",
                project_root / "STM32CubeIDE" / "Boot" / "Debug",
                project_root / "STM32CubeIDE" / "AppS" / "Debug",
            ):
                required_dir.mkdir(parents=True)
            (project_root / "STM32CubeIDE" / "Boot" / "Debug" / "makefile").write_text(
                "# makefile\n",
                encoding="utf-8",
            )
            (project_root / "STM32CubeIDE" / "AppS" / "Debug" / "makefile").write_text(
                "# makefile\n",
                encoding="utf-8",
            )
            config = Dict(
                training=Dict(energy_aware=False, latency_proxy_max_flops=20_000_000),
                device=Dict(
                    hil=False,
                    name="STM32_NUCLEO_N657X0_Q",
                    stm32=Dict(
                        project_root=project_root,
                        gdb_port=61235,
                    ),
                ),
                dataset=Dict(params=Dict(window_size=128)),
                outputs=Dict(candidate_dir=Path("odom_tcn")),
            )

            resolved = resolve_device_options(str(config.device.name), config.device)

        self.assertEqual(resolved["project_root"], project_root.resolve())
        self.assertEqual(resolved["gdb_port"], 61235)


class ScoreEvaluationHelperTests(unittest.TestCase):
    """Ensure score evaluation stays aligned with the supported NAS path."""

    def test_evaluate_score_config_uses_effective_energy_flag_from_metrics(self) -> None:
        """STM Phase 1 should score on latency when upstream energy is disabled."""
        # Scoring should trust the effective energy flag reported by metrics so STM32 phase-one proxy runs do not pretend energy was measured.
        score_config = Dict(
            type="scoring-function",
            metrics=Dict(),
            params=Dict(
                terms=[
                    Dict(type="weighted", metric="latency_ms", weight=-1.0),
                ]
            ),
        )
        metrics = {
            "energy_aware": False,
            "energy_mj_per_inference": -1.0,
            "latency_ms": 12.5,
            "hil_enabled": True,
            "error_code": 0,
            "ram_bytes": 128,
            "flash_bytes": 256,
        }
        hyperparams = Dict(flops=1_000)

        scoring_result = evaluate_score_config(
            metrics=metrics,
            hyperparams=hyperparams,
            score_config=score_config,
        )

        self.assertEqual(scoring_result.score, -12.5)

    def test_evaluate_score_config_supports_normalized_weighted_terms(self) -> None:
        """Normalized weighted terms should use the provided device limits."""
        # Normalized score terms should use the active device limits so scoring never falls back to stale reference values.
        score_config = Dict(
            type="scoring-function",
            metrics=Dict(),
            params=Dict(
                terms=[
                    Dict(
                        type="normalized-weighted",
                        metric="ram_bytes",
                        weight=0.5,
                        reference=Dict(type="metric", metric="max_ram_bytes"),
                    ),
                    Dict(
                        type="normalized-weighted",
                        metric="flash_bytes",
                        weight=-2.0,
                        reference=Dict(type="metric", metric="max_flash_bytes"),
                    ),
                ]
            ),
        )
        metrics = {
            "energy_aware": False,
            "energy_mj_per_inference": -1.0,
            "latency_ms": 12.5,
            "hil_enabled": True,
            "error_code": 0,
            "ram_bytes": 128,
            "flash_bytes": 256,
            "max_ram_bytes": 1_024.0,
            "max_flash_bytes": 2_048.0,
        }
        hyperparams = Dict(flops=1_000)

        scoring_result = evaluate_score_config(
            metrics=metrics,
            hyperparams=hyperparams,
            score_config=score_config,
        )

        self.assertAlmostEqual(scoring_result.score, -0.1875)

    def test_evaluate_score_config_supports_energy_budget_from_power_metric(self) -> None:
        """Energy budget derived metrics should resolve from power and duration."""
        # Derived energy budgets should resolve from power and duration so score terms can express device-level energy envelopes.
        score_config = Dict(
            type="scoring-function",
            metrics=Dict(
                energy_budget_mj=Dict(
                    type="energy-budget-from-power",
                    power_mw=Dict(type="literal", value=100.0),
                    duration_ms=Dict(type="metric", metric="latency_budget_ms"),
                )
            ),
            params=Dict(
                terms=[
                    Dict(
                        type="target",
                        metric="energy_mj_per_inference",
                        weight=0.15,
                        reference=Dict(type="metric", metric="energy_budget_mj"),
                    ),
                ]
            ),
        )
        metrics = {
            "energy_aware": True,
            "energy_mj_per_inference": 3.0,
            "latency_ms": 12.5,
            "latency_budget_ms": 20.0,
            "hil_enabled": True,
            "error_code": 0,
            "ram_bytes": 128,
            "flash_bytes": 256,
        }
        hyperparams = Dict(flops=1_000)

        scoring_result = evaluate_score_config(
            metrics=metrics,
            hyperparams=hyperparams,
            score_config=score_config,
        )

        self.assertAlmostEqual(scoring_result.score, -0.15)

    def test_evaluate_score_config_supports_multiobjective_custom_metrics(self) -> None:
        """Multi-objective evaluation should honor arbitrary task metric names."""
        # Generic task metrics should flow through score evaluation without requiring the old no-training wrapper.
        score_config = Dict(
            type="multi-objective",
            metrics=Dict(),
            params=Dict(
                objectives=[
                    Dict(metric="signed_bias", direction="maximize"),
                    Dict(metric="signed_offset", direction="maximize"),
                ]
            ),
        )
        metrics = {"signed_bias": -1.0, "signed_offset": -1.0, "latency_ms": 12.5}
        hyperparams = Dict(flops=1_000)

        result = evaluate_score_config(
            metrics=metrics,
            hyperparams=hyperparams,
            score_config=score_config,
            task_nonnegative_metric_names=set(),
        )

        self.assertIsNone(result.score)
        self.assertEqual(result.objective_values, [-1.0, -1.0])

    def test_evaluate_score_config_raises_dedicated_exception_for_unavailable_score_metric(self) -> None:
        """Unavailable score metrics should raise ScoreConfigEvaluationError."""
        # Unavailable score metrics should raise the dedicated evaluation error so NAS pruning can treat them as configuration problems.
        score_config = Dict(
            type="scoring-function",
            metrics=Dict(),
            params=Dict(
                terms=[
                    Dict(
                        type="target",
                        metric="energy_mj_per_inference",
                        weight=0.15,
                        reference=Dict(type="metric", metric="latency_budget_ms"),
                    ),
                ]
            ),
        )
        metrics = {
            "energy_aware": True,
            "energy_mj_per_inference": 3.0,
            "latency_ms": 12.5,
            "latency_budget_ms": -1.0,
            "hil_enabled": True,
            "error_code": 0,
            "ram_bytes": 128,
            "flash_bytes": 256,
        }
        hyperparams = Dict(flops=1_000)

        with self.assertRaises(ScoreConfigEvaluationError):
            evaluate_score_config(
                metrics=metrics,
                hyperparams=hyperparams,
                score_config=score_config,
            )

    def test_evaluate_score_config_matches_scalar_scoring_semantics(self) -> None:
        """Public score evaluation helper should preserve scalar score behavior."""
        # The public score-evaluation helper should stay aligned with the scalar path used during real trial execution.
        score_config = Dict(
            type="scoring-function",
            metrics=Dict(),
            params=Dict(
                terms=[
                    Dict(type="weighted", metric="rmse_total", weight=-1.0),
                    Dict(type="weighted", metric="flops", weight=-0.001),
                ]
            ),
        )
        metrics = {"rmse_total": 0.5, "latency_ms": 10.0}
        hyperparams = Dict(flops=1_000)

        result = evaluate_score_config(
            metrics=metrics,
            hyperparams=hyperparams,
            score_config=score_config,
        )

        self.assertAlmostEqual(result.score, -1.5)
        self.assertEqual(result.objective_names, ["score"])


class LoadSettingsTests(unittest.TestCase):
    """Verify the NAS configuration loader derives paths and validates input."""

    @staticmethod
    def _score_lines(include_quantization: bool = True) -> list[str]:
        """Return the minimal scoring-function YAML block shared by loader tests.

        Parameters
        ----------
        include_quantization : bool
            Whether quantization metadata should be included.

        Returns
        -------
        list[str]
            Rendered score lines for the provided summary.
        """
        lines = []
        if include_quantization:
            lines.extend(
                [
                    "  quantization:",
                    "    mode: int8_ptq",
                    "    search: false",
                    "    choices: [int8_ptq]",
                ]
            )
        lines.extend(
            [
                "nas:",
                "  score:",
                "    type: scoring-function",
                "    params:",
                "      terms:",
                "        - type: weighted",
                "          metric: flops",
                "          weight: -1.0",
            ]
        )
        return lines

    def test_load_settings_derives_expected_paths(self) -> None:
        """YAML entries should produce resolved paths and derived file names."""
        # Config loading should derive the expected model, checkpoint, and artifact paths from the selected device and output roots.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "config.yaml"
            models_dir = tmp_path / "models_dir"
            candidate_dir = tmp_path / "candidate_dir"
            config_path.write_text(
                "\n".join(
                    [
                        "device:",
                        "  name: TEST_DEVICE",
                        "training:",
                        "  nas_trials: 5",
                        *self._score_lines(),
                        "outputs:",
                        f"  models_dir: \"{models_dir}\"",
                        f"  candidate_dir: \"{candidate_dir}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                    ]
                )
            )

            settings = load_config(config_path=config_path)

            self.assertEqual(
                settings.outputs.model_name, "TinyOdomEx_Test_TEST_DEVICE.tflite"
            )
            self.assertEqual(
                settings.outputs.checkpoint_name, "TinyOdomEx_Test_TEST_DEVICE.keras"
            )
            self.assertTrue(settings.outputs.models_dir.is_dir())
            self.assertTrue(settings.outputs.candidate_dir.is_dir())
            self.assertEqual(
                settings.outputs.tflite_model_path,
                settings.outputs.models_dir / settings.outputs.model_name,
            )
            self.assertEqual(
                settings.outputs.checkpoint_path,
                settings.outputs.models_dir / settings.outputs.checkpoint_name,
            )
            self.assertEqual(settings.training.drop_rate_choices, DROP_RATE_CHOICES)  

    def test_load_settings_accepts_fixed_int8_quantization(self) -> None:
        """A fixed int8 PTQ quantization block should normalize cleanly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "device:",
                        "  name: STM32_NUCLEO_N657X0_Q",
                        "training:",
                        "  nas_trials: 5",
                        "  quantization:",
                        "    mode: int8_ptq",
                        "    search: false",
                        "    choices: [int8_ptq]",
                        *self._score_lines(include_quantization=False),
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'candidate'}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                    ]
                )
            )

            settings = load_config(config_path=config_path)

            self.assertEqual(settings.training.quantization.mode, "int8_ptq")
            self.assertFalse(settings.training.quantization.search)
            self.assertEqual(settings.training.quantization.choices, ["int8_ptq"])

    def test_load_settings_accepts_searchable_quantization_on_supported_board(self) -> None:
        """Arduino-backed supported boards may search float and int8 PTQ exports."""
        for device_lines in (
            [
                "  name: PORTENTA_H7",
                "  portenta:",
                "    target_core: cm7",
                "    split: 75_25",
                "    security: none",
            ],
            ["  name: ARDUINO_NANO_33_BLE_SENSE"],
        ):
            with self.subTest(device=device_lines[0]):
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_path = Path(tmpdir)
                    config_path = tmp_path / "config.yaml"
                    config_path.write_text(
                        "\n".join(
                            [
                                "device:",
                                *device_lines,
                                "training:",
                                "  nas_trials: 5",
                                "  quantization:",
                                "    mode: int8_ptq",
                                "    search: true",
                                "    choices: [float, int8_ptq]",
                                *self._score_lines(include_quantization=False),
                                "outputs:",
                                f"  models_dir: \"{tmp_path / 'models'}\"",
                                f"  candidate_dir: \"{tmp_path / 'candidate'}\"",
                                "  artifact_stem: \"TinyOdomEx_Test\"",
                            ]
                        )
                    )

                    settings = load_config(config_path=config_path)

                    self.assertTrue(settings.training.quantization.search)
                    self.assertEqual(settings.training.quantization.choices, ["float", "int8_ptq"])

    def test_load_settings_rejects_invalid_quantization_configs(self) -> None:
        """Quantization must use the new mapping shape and supported choices."""
        cases = {
            "missing": [],
            "legacy_boolean": ["  quantization: true"],
            "empty_mapping": ["  quantization: {}"],
            "empty_choices": [
                "  quantization:",
                "    mode: int8_ptq",
                "    search: false",
                "    choices: []",
            ],
            "mode_not_in_choices": [
                "  quantization:",
                "    mode: float",
                "    search: false",
                "    choices: [int8_ptq]",
            ],
        }
        for label, quantization_lines in cases.items():
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_path = Path(tmpdir)
                    config_path = tmp_path / "config.yaml"
                    config_path.write_text(
                        "\n".join(
                            [
                                "device:",
                                "  name: ARDUINO_NANO_33_BLE_SENSE",
                                "training:",
                                "  nas_trials: 5",
                                *quantization_lines,
                                *self._score_lines(include_quantization=False),
                                "outputs:",
                                f"  models_dir: \"{tmp_path / 'models'}\"",
                                f"  candidate_dir: \"{tmp_path / 'candidate'}\"",
                                "  artifact_stem: \"TinyOdomEx_Test\"",
                            ]
                        )
                    )

                    with self.assertRaises((KeyError, ValueError)):
                        load_config(config_path=config_path)

    def test_load_settings_rejects_missing_candidate_dir(self) -> None:
        """Configuration loading should reject outputs without candidate_dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "device:",
                        "  name: TEST_DEVICE",
                        "training:",
                        "  nas_trials: 5",
                        *self._score_lines(),
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                    ]
                )
            )

            with self.assertRaisesRegex(ValueError, "outputs.candidate_dir"):
                load_config(config_path=config_path)

    def test_load_settings_rejects_missing_artifact_stem(self) -> None:
        """Configuration loading should require an artifact stem."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "device:",
                        "  name: TEST_DEVICE",
                        "training:",
                        "  nas_trials: 5",
                        *self._score_lines(),
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'candidate'}\"",
                    ]
                )
            )

            with self.assertRaisesRegex(ValueError, "outputs.artifact_stem"):
                load_config(config_path=config_path)

    def test_load_settings_rejects_yaml_artifact_filenames(self) -> None:
        """Configuration loading should reject authored artifact filenames."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "device:",
                        "  name: TEST_DEVICE",
                        "training:",
                        "  nas_trials: 5",
                        *self._score_lines(),
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'candidate'}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                        "  model_name: \"manual.tflite\"",
                    ]
                )
            )

            with self.assertRaisesRegex(ValueError, "outputs.model_name"):
                load_config(config_path=config_path)

    def test_load_settings_rejects_yaml_checkpoint_name(self) -> None:
        """Configuration loading should reject authored checkpoint filenames."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "device:",
                        "  name: TEST_DEVICE",
                        "training:",
                        "  nas_trials: 5",
                        *self._score_lines(),
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'candidate'}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                        "  checkpoint_name: \"manual.keras\"",
                    ]
                )
            )

            with self.assertRaisesRegex(ValueError, "outputs.model_name"):
                load_config(config_path=config_path)

    def test_load_settings_rejects_invalid_artifact_stems(self) -> None:
        """Artifact stems should reject blank and non-string YAML values."""
        cases = {
            "blank": "\"   \"",
            "integer": "123",
        }
        for label, yaml_value in cases.items():
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_path = Path(tmpdir)
                    config_path = tmp_path / "config.yaml"
                    config_path.write_text(
                        "\n".join(
                            [
                                "device:",
                                "  name: TEST_DEVICE",
                                "training:",
                                "  nas_trials: 5",
                                *self._score_lines(),
                                "outputs:",
                                f"  models_dir: \"{tmp_path / 'models'}\"",
                                f"  candidate_dir: \"{tmp_path / 'candidate'}\"",
                                f"  artifact_stem: {yaml_value}",
                            ]
                        )
                    )

                    with self.assertRaisesRegex(ValueError, "outputs.artifact_stem"):
                        load_config(config_path=config_path)

    def test_load_settings_rejects_path_like_artifact_stem(self) -> None:
        """Artifact stems should be filename stems, not paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "device:",
                        "  name: TEST_DEVICE",
                        "training:",
                        "  nas_trials: 5",
                        *self._score_lines(),
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'candidate'}\"",
                        "  artifact_stem: \"nested/TinyOdomEx_Test\"",
                    ]
                )
            )

            with self.assertRaisesRegex(ValueError, "outputs.artifact_stem"):
                load_config(config_path=config_path)

    def test_shipped_configs_use_artifact_stem_and_export_variant(self) -> None:
        """All shipped configs should load with the Phase 5 artifact schema."""
        cases = [
            ("nas_config_stm32.yaml", "odom_tcn", "approx_trained"),
            ("nas_config_ble.yaml", "odom_tcn", "approx_trained"),
            ("nas_config_portenta.yaml", "odom_tcn", "approx_trained"),
            ("nas_config_audio_stm32.yaml", "audio_dscnn", "untrained"),
            ("nas_config_audio_portenta.yaml", "audio_dscnn", "untrained"),
            ("nas_config_flops_rmse.yaml", "odom_tcn", "approx_trained"),
            ("nas_config_memory_proxy.yaml", "odom_tcn", "approx_trained"),
        ]
        for filename, family, export_variant in cases:
            with self.subTest(filename=filename):
                config_path = ROOT_DIR / "src/config" / filename
                text = config_path.read_text(encoding="utf-8")
                self.assertIn("artifact_stem:", text)
                self.assertNotIn("model_name:", text)
                self.assertNotIn("checkpoint_name:", text)
                settings = load_config(config_path=config_path)
                selection = resolve_component_selection(settings)

                self.assertEqual(settings.model.family, family)
                self.assertEqual(selection["model_config"]["params"].export_variant, export_variant)

    def test_shipped_configs_use_production_training_budgets(self) -> None:
        """Checked-in example configs should use production training budgets."""
        expected_budgets = {
            "nas_config_stm32.yaml": 250,
            "nas_config_ble.yaml": 150,
            "nas_config_portenta.yaml": 150,
            "nas_config_audio_stm32.yaml": 200,
            "nas_config_audio_portenta.yaml": 200,
            "nas_config_flops_rmse.yaml": 150,
            "nas_config_memory_proxy.yaml": 150,
        }
        for filename, expected_trials in expected_budgets.items():
            with self.subTest(filename=filename):
                settings = load_config(config_path=ROOT_DIR / "src/config" / filename)
                self.assertEqual(settings.training["nas_epochs"], 55)
                self.assertEqual(settings.training["model_epochs"], 990)
                self.assertEqual(settings.training["nas_trials"], expected_trials)
                self.assertEqual(settings.training["nas_multiobjective_population_size"], 50)
                self.assertEqual(settings.training["max_total_trials"], 300)
                if filename.startswith("nas_config_audio"):
                    self.assertEqual(settings.model.search, {})

    def test_config_readme_lists_audio_config_and_artifact_stem(self) -> None:
        """Config documentation should mention the audio config and new stem key."""
        readme = (ROOT_DIR / "src/config/README.md").read_text(encoding="utf-8")

        self.assertIn("nas_config_audio_stm32.yaml", readme)
        self.assertIn("nas_config_flops_rmse.yaml", readme)
        self.assertIn("nas_config_memory_proxy.yaml", readme)
        self.assertIn("artifact_stem", readme)
        self.assertIn("export_variant", readme)
        self.assertIn("compile_when_hil_disabled", readme)

    def test_audio_stm32_config_derives_audio_artifact_names(self) -> None:
        """The audio STM32 config should derive artifact names from its stem."""
        settings = load_config(config_path=ROOT_DIR / "src/config/nas_config_audio_stm32.yaml")

        self.assertEqual(settings.model.family, "audio_dscnn")
        self.assertEqual(settings.model.params.export_variant, "untrained")
        self.assertEqual(
            settings.outputs.model_name,
            "TinyOdomEx_UrbanSound8K_STM32_NUCLEO_N657X0_Q.tflite",
        )
        self.assertEqual(
            settings.outputs.checkpoint_name,
            "TinyOdomEx_UrbanSound8K_STM32_NUCLEO_N657X0_Q.keras",
        )
        self.assertEqual(settings.training.quantization.mode, "int8_ptq")
        self.assertFalse(settings.training.quantization.search)
        self.assertEqual(settings.training.quantization.choices, ["int8_ptq"])

    def test_audio_stm32_config_resolves_audio_components(self) -> None:
        """The audio STM32 config should resolve the audio component stack."""
        settings = load_config(config_path=ROOT_DIR / "src/config/nas_config_audio_stm32.yaml")

        selection = resolve_component_selection(settings)

        self.assertEqual(selection["dataset_name"], "urbansound8k_mel")
        self.assertEqual(selection["task_name"], "sound_classification")
        self.assertEqual(selection["model_family_name"], "audio_dscnn")
        self.assertEqual(selection["model_config"]["params"].export_variant, "untrained")
        self.assertNotIn("evaluation", settings)
        self.assertEqual(settings.task.params.evaluation.protocol, "fixed_split")
        self.assertEqual(settings.task.params.evaluation.fold_rotation.test_folds, list(range(1, 11)))

    def test_load_settings_validates_fold_rotation_config(self) -> None:
        """Fold-rotation reporting config should require explicit cache roots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "device:",
                        "  name: TEST_DEVICE",
                        "training:",
                        "  nas_trials: 5",
                        "  quantization:",
                        "    mode: int8_ptq",
                        "    search: false",
                        "    choices: [int8_ptq]",
                        "dataset:",
                        "  name: urbansound8k_mel",
                        "  params:",
                        "    cache_dir: cache/fixed",
                        "    batch_period_ms: 2000",
                        "task:",
                        "  name: sound_classification",
                        "  params:",
                        "    evaluation:",
                        "      protocol: fold_rotation",
                        "      fold_rotation:",
                        "        test_folds: [1, 2, 10]",
                        *self._score_lines(include_quantization=False),
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'candidate'}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                    ]
                )
            )

            with self.assertRaisesRegex(ValueError, "fold_rotation_cache_dir"):
                load_config(config_path=config_path)

            text = config_path.read_text(encoding="utf-8").replace(
                "    batch_period_ms: 2000",
                "    batch_period_ms: 2000\n    fold_rotation_cache_dir: cache/fold_rotation",
            )
            config_path.write_text(text, encoding="utf-8")
            settings = load_config(config_path=config_path)

        self.assertEqual(settings.task.params.evaluation.protocol, "fold_rotation")
        self.assertEqual(settings.task.params.evaluation.fold_rotation.test_folds, [1, 2, 10])

    def test_load_settings_rejects_fold_rotation_multiobjective(self) -> None:
        """Fold rotation should fail fast for multi-objective NAS."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "device:",
                        "  name: TEST_DEVICE",
                        "training:",
                        "  nas_trials: 5",
                        "  quantization:",
                        "    mode: int8_ptq",
                        "    search: false",
                        "    choices: [int8_ptq]",
                        "dataset:",
                        "  name: urbansound8k_mel",
                        "  params:",
                        "    cache_dir: cache/fixed",
                        "    fold_rotation_cache_dir: cache/fold_rotation",
                        "    batch_period_ms: 2000",
                        "task:",
                        "  name: sound_classification",
                        "  params:",
                        "    evaluation:",
                        "      protocol: fold_rotation",
                        "nas:",
                        "  score:",
                        "    type: multi-objective",
                        "    params:",
                        "      objectives:",
                        "        - metric: accuracy",
                        "          direction: maximize",
                        "  prune:",
                        "    rules: []",
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'candidate'}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                    ]
                )
            )

            with self.assertRaisesRegex(ValueError, "single-objective"):
                load_config(config_path=config_path)

    def test_audio_stm32_score_validates_against_classification_metrics(self) -> None:
        """Audio NAS scoring should use classification metrics, not RMSE terms."""
        settings = load_config(config_path=ROOT_DIR / "src/config/nas_config_audio_stm32.yaml")

        validate_nas_policy_for_task(
            settings,
            task_metric_names={"loss", "accuracy", "macro_f1"},
            training_only_task_metric_names=set(),
        )

        settings.nas.score.params.terms = [Dict(type="weighted", metric="rmse_total", weight=-1.0)]
        with self.assertRaisesRegex(ValueError, "rmse_total"):
            validate_nas_policy_for_task(
                settings,
                task_metric_names={"loss", "accuracy", "macro_f1"},
                training_only_task_metric_names=set(),
            )

    def test_load_settings_rejects_removed_tcn_dir(self) -> None:
        """Configuration loading should reject the removed tcn_dir output key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "device:",
                        "  name: TEST_DEVICE",
                        "training:",
                        "  nas_trials: 5",
                        *self._score_lines(),
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  tcn_dir: \"{tmp_path / 'legacy'}\"",
                    ]
                )
            )

            with self.assertRaisesRegex(ValueError, "outputs.tcn_dir"):
                load_config(config_path=config_path)

    def test_load_settings_requires_sections(self) -> None:
        """Missing required sections should raise informative errors."""
        # Missing top-level sections should fail during config load so malformed YAML never reaches the NAS loop.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            cfg_missing_device = tmp_path / "missing_device.yaml"
            cfg_missing_device.write_text(
                "\n".join(
                    [
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'tcn'}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                        *self._score_lines(),
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
                        *self._score_lines(),
                    ]
                )
            )

            with self.assertRaises(KeyError):
                load_config(config_path=cfg_missing_outputs)

    def test_load_settings_requires_training_section(self) -> None:
        """Training section should be mandatory for NAS runs."""
        # NAS configs must include a training section because search defaults depend on it.
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
                        f"  candidate_dir: \"{tmp_path / 'tcn'}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                        *self._score_lines(),
                    ]
                )
            )

            with self.assertRaises(KeyError):
                load_config(config_path=cfg_missing_training)

    def test_load_settings_sets_default_max_total_trials(self) -> None:
        """max_total_trials should default to 2x the requested nas_trials when omitted."""
        # The loader should default max_total_trials from nas_trials so retry budget stays predictable in minimal configs.
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
                        "  quantization:",
                        "    mode: int8_ptq",
                        "    search: false",
                        "    choices: [int8_ptq]",
                        *self._score_lines(),
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'tcn'}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                    ]
                )
            )

            settings = load_config(config_path=cfg)
            self.assertEqual(settings.training.max_total_trials, 20)

    def test_load_settings_defaults_measured_inference_runs(self) -> None:
        # Omitted measured inference runs should fall back to the documented loader defaults.
        """Validate load settings defaults measured inference runs."""
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
                        "  quantization:",
                        "    mode: int8_ptq",
                        "    search: false",
                        "    choices: [int8_ptq]",
                        *self._score_lines(),
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'tcn'}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                    ]
                )
            )

            settings = load_config(config_path=cfg)

        self.assertEqual(settings.device.measured_inference_runs, 10)

    def test_load_settings_defaults_runtime_mode_and_latency_budget_override(self) -> None:
        # Omitted runtime mode and latency budget override should fall back to the documented loader defaults.
        """Validate load settings defaults runtime mode and latency budget override."""
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
                        "  quantization:",
                        "    mode: int8_ptq",
                        "    search: false",
                        "    choices: [int8_ptq]",
                        *self._score_lines(),
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'tcn'}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                    ]
                )
            )

            settings = load_config(config_path=cfg)

        self.assertEqual(settings.device.runtime_mode, "back_to_back")
        self.assertIsNone(settings.device.latency_budget_ms)
        self.assertEqual(settings.device.compile_when_hil_disabled, "auto")

    def test_load_settings_accepts_compile_when_hil_disabled_modes(self) -> None:
        """The non-HIL compile policy should normalize supported YAML shapes."""
        cases = {
            "auto": "auto",
            "true": "true",
            "false": "false",
            "true_bool": True,
            "false_bool": False,
        }
        for label, raw_value in cases.items():
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_path = Path(tmpdir)
                    cfg = tmp_path / "config.yaml"
                    value_text = str(raw_value).lower() if isinstance(raw_value, bool) else f"\"{raw_value}\""
                    cfg.write_text(
                        "\n".join(
                            [
                                "device:",
                                "  name: TEST_DEVICE",
                                f"  compile_when_hil_disabled: {value_text}",
                                "training:",
                                "  nas_trials: 10",
                                "  quantization:",
                                "    mode: int8_ptq",
                                "    search: false",
                                "    choices: [int8_ptq]",
                                *self._score_lines(include_quantization=False),
                                "outputs:",
                                f"  models_dir: \"{tmp_path / 'models'}\"",
                                f"  candidate_dir: \"{tmp_path / 'tcn'}\"",
                                "  artifact_stem: \"TinyOdomEx_Test\"",
                            ]
                        )
                    )

                    settings = load_config(config_path=cfg)

                expected = str(raw_value).lower() if isinstance(raw_value, bool) else raw_value
                self.assertEqual(settings.device.compile_when_hil_disabled, expected)

    def test_load_settings_rejects_invalid_compile_when_hil_disabled(self) -> None:
        """Invalid non-HIL compile policy values should fail during config load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            cfg = tmp_path / "config.yaml"
            cfg.write_text(
                "\n".join(
                    [
                        "device:",
                        "  name: TEST_DEVICE",
                        "  compile_when_hil_disabled: sometimes",
                        "training:",
                        "  nas_trials: 10",
                        "  quantization:",
                        "    mode: int8_ptq",
                        "    search: false",
                        "    choices: [int8_ptq]",
                        *self._score_lines(include_quantization=False),
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'tcn'}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                    ]
                )
            )

            with self.assertRaisesRegex(ValueError, "device.compile_when_hil_disabled"):
                load_config(config_path=cfg)

    def test_load_settings_accepts_runtime_mode_and_latency_budget_override(self) -> None:
        # Runtime mode and latency budget override should remain a supported config shape.
        """Validate load settings accepts runtime mode and latency budget override."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            cfg = tmp_path / "config.yaml"
            cfg.write_text(
                "\n".join(
                    [
                        "device:",
                        "  name: TEST_DEVICE",
                        "  runtime_mode: cadenced",
                        "  latency_budget_ms: 37.5",
                        "training:",
                        "  nas_trials: 10",
                        *self._score_lines(),
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'tcn'}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                    ]
                )
            )

            settings = load_config(config_path=cfg)

        self.assertEqual(settings.device.runtime_mode, "cadenced")
        self.assertEqual(settings.device.latency_budget_ms, 37.5)

    def test_load_settings_rejects_invalid_measured_inference_runs(self) -> None:
        # Invalid invalid measured inference runs should fail during config load so unsupported NAS settings never reach execution.
        """Validate load settings rejects invalid measured inference runs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            cfg = tmp_path / "config.yaml"
            cfg.write_text(
                "\n".join(
                    [
                        "device:",
                        "  name: TEST_DEVICE",
                        "  measured_inference_runs: 0",
                        "training:",
                        "  nas_trials: 10",
                        *self._score_lines(),
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'tcn'}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                    ]
                )
            )

            with self.assertRaises(ValueError):
                load_config(config_path=cfg)

    def test_load_settings_rejects_legacy_stm_runtime_mode_path(self) -> None:
        # Invalid legacy STM32 runtime mode path should fail during config load so unsupported NAS settings never reach execution.
        """Validate load settings rejects legacy stm runtime mode path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            cfg = tmp_path / "config.yaml"
            cfg.write_text(
                "\n".join(
                    [
                        "device:",
                        "  name: STM32_NUCLEO_N657X0_Q",
                        "  stm32:",
                        "    runtime_mode: cadenced",
                        "training:",
                        "  nas_trials: 10",
                        "  quantization:",
                        "    mode: int8_ptq",
                        "    search: false",
                        "    choices: [int8_ptq]",
                        *self._score_lines(),
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'tcn'}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                    ]
                )
            )

            with self.assertRaisesRegex(ValueError, "device.stm32.runtime_mode"):
                load_config(config_path=cfg)

    def test_load_settings_rejects_cadenced_portenta_non_uniform_input(self) -> None:
        # Invalid cadenced Portenta non uniform input should fail during config load so unsupported NAS settings never reach execution.
        """Validate load settings rejects cadenced Portenta non uniform input."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            cfg = tmp_path / "config.yaml"
            cfg.write_text(
                "\n".join(
                    [
                        "device:",
                        "  name: PORTENTA_H7",
                        "  runtime_mode: cadenced",
                        "  portenta:",
                        "    target_core: cm7",
                        "training:",
                        "  nas_trials: 10",
                        "  quantization:",
                        "    mode: int8_ptq",
                        "    search: false",
                        "    choices: [int8_ptq]",
                        "  input_mode: representative",
                        *self._score_lines(),
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'tcn'}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                    ]
                )
            )

            with self.assertRaisesRegex(ValueError, "only supports training.input_mode='uniform'"):
                load_config(config_path=cfg)

    def test_load_settings_accepts_normalized_weighted_term(self) -> None:
        """Normalized weighted terms should validate typed references."""
        # Normalized weighted term should remain a supported config shape.
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
                        "  quantization:",
                        "    mode: int8_ptq",
                        "    search: false",
                        "    choices: [int8_ptq]",
                        "nas:",
                        "  score:",
                        "    type: scoring-function",
                        "    params:",
                        "      terms:",
                        "        - type: normalized-weighted",
                        "          metric: ram_bytes",
                        "          weight: 0.01",
                        "          reference:",
                        "            type: metric",
                        "            metric: max_ram_bytes",
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'tcn'}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                    ]
                )
            )

            settings = load_config(config_path=cfg)

        self.assertEqual(settings.nas.score.params.terms[0].type, "normalized-weighted")
        self.assertEqual(settings.nas.score.params.terms[0].reference.metric, "max_ram_bytes")

    def test_load_settings_rejects_non_positive_normalized_weight_reference(self) -> None:
        """Literal normalized references must be strictly positive."""
        # Invalid non positive normalized weight reference should fail during config load so unsupported NAS settings never reach execution.
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
                        "  quantization:",
                        "    mode: int8_ptq",
                        "    search: false",
                        "    choices: [int8_ptq]",
                        "nas:",
                        "  score:",
                        "    type: scoring-function",
                        "    params:",
                        "      terms:",
                        "        - type: normalized-weighted",
                        "          metric: ram_bytes",
                        "          weight: 0.01",
                        "          reference:",
                        "            type: literal",
                        "            value: 0.0",
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'tcn'}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                    ]
                )
            )

            with self.assertRaisesRegex(ValueError, "greater than zero"):
                load_config(config_path=cfg)

    def test_load_settings_accepts_energy_budget_from_power_metric(self) -> None:
        """Energy budgets derived from power should validate cleanly."""
        # Energy budget from power metric should remain a supported config shape.
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
                        "  quantization:",
                        "    mode: int8_ptq",
                        "    search: false",
                        "    choices: [int8_ptq]",
                        "nas:",
                        "  score:",
                        "    type: scoring-function",
                        "    metrics:",
                        "      energy_budget_mj:",
                        "        type: energy-budget-from-power",
                        "        power_mw:",
                        "          type: literal",
                        "          value: 100.0",
                        "        duration_ms:",
                        "          type: metric",
                        "          metric: latency_budget_ms",
                        "    params:",
                        "      terms:",
                        "        - type: target",
                        "          metric: energy_mj_per_inference",
                        "          weight: 0.15",
                        "          reference:",
                        "            type: metric",
                        "            metric: energy_budget_mj",
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'tcn'}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                    ]
                )
            )

            settings = load_config(config_path=cfg)

        self.assertEqual(settings.nas.score.metrics.energy_budget_mj.type, "energy-budget-from-power")
        self.assertEqual(settings.nas.score.metrics.energy_budget_mj.duration_ms.metric, "latency_budget_ms")

    def test_load_settings_rejects_non_positive_energy_budget_duration(self) -> None:
        """Energy budget duration literals must stay strictly positive."""
        # Invalid non positive energy budget duration should fail during config load so unsupported NAS settings never reach execution.
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
                        "  quantization:",
                        "    mode: int8_ptq",
                        "    search: false",
                        "    choices: [int8_ptq]",
                        "nas:",
                        "  score:",
                        "    type: scoring-function",
                        "    metrics:",
                        "      energy_budget_mj:",
                        "        type: energy-budget-from-power",
                        "        power_mw:",
                        "          type: literal",
                        "          value: 100.0",
                        "        duration_ms:",
                        "          type: literal",
                        "          value: 0.0",
                        "    params:",
                        "      terms:",
                        "        - type: target",
                        "          metric: energy_mj_per_inference",
                        "          weight: 0.15",
                        "          reference:",
                        "            type: metric",
                        "            metric: energy_budget_mj",
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'tcn'}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                    ]
                )
            )

            with self.assertRaisesRegex(ValueError, "greater than zero"):
                load_config(config_path=cfg)

    def test_load_settings_rejects_legacy_top_level_score(self) -> None:
        # Invalid legacy top level score should fail during config load so unsupported NAS settings never reach execution.
        """Validate load settings rejects legacy top level score."""
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
                        "  quantization:",
                        "    mode: int8_ptq",
                        "    search: false",
                        "    choices: [int8_ptq]",
                        "score:",
                        "  type: scoring-function",
                        "  params:",
                        "    terms:",
                        "      - type: weighted",
                        "        metric: flops",
                        "        weight: -1.0",
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'tcn'}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                    ]
                )
            )

            with self.assertRaisesRegex(KeyError, "nas.score"):
                load_config(config_path=cfg)

    def test_load_settings_accepts_scalar_prune_rules(self) -> None:
        # Scalar prune rules should remain a supported config shape.
        """Validate load settings accepts scalar prune rules."""
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
                        "  quantization:",
                        "    mode: int8_ptq",
                        "    search: false",
                        "    choices: [int8_ptq]",
                        "nas:",
                        "  score:",
                        "    type: scoring-function",
                        "    params:",
                        "      terms:",
                        "        - type: weighted",
                        "          metric: flops",
                        "          weight: -1.0",
                        "  prune:",
                        "    rules:",
                        "      - rule: latency_budget",
                        "        metric: latency_ms",
                        "        condition: gt",
                        "        reference:",
                        "          type: metric",
                        "          metric: latency_budget_ms",
                        "        reason: Latency exceeds budget",
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'tcn'}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                    ]
                )
            )

            settings = load_config(config_path=cfg)

        self.assertEqual(settings.nas.prune.rules[0].rule, "latency_budget")
        self.assertEqual(settings.nas.prune.rules[0].condition, "gt")

    def test_load_settings_accepts_empty_scalar_prune_rules(self) -> None:
        # Empty scalar prune rules should remain a supported config shape.
        """Validate load settings accepts empty scalar prune rules."""
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
                        "  quantization:",
                        "    mode: int8_ptq",
                        "    search: false",
                        "    choices: [int8_ptq]",
                        "nas:",
                        "  score:",
                        "    type: scoring-function",
                        "    params:",
                        "      terms:",
                        "        - type: weighted",
                        "          metric: flops",
                        "          weight: -1.0",
                        "  prune:",
                        "    rules: []",
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'tcn'}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                    ]
                )
            )

            settings = load_config(config_path=cfg)

        self.assertEqual(settings.nas.prune.rules, [])

    def test_load_settings_accepts_prune_rules_for_multiobjective_score(self) -> None:
        """Multi-objective score configs should accept pre-fit feasibility gates.

        Returns
        -------
        None
            Asserts valid multi-objective prune rules pass task-aware
            validation.
        """
        # Multi-objective prune rules are post-build gates, not Optuna pruning,
        # so valid pre-fit metrics should pass task-aware validation.
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
                        "  quantization:",
                        "    mode: int8_ptq",
                        "    search: false",
                        "    choices: [int8_ptq]",
                        "nas:",
                        "  score:",
                        "    type: multi-objective",
                        "    params:",
                        "      objectives:",
                        "        - metric: rmse_total",
                        "          direction: minimize",
                        "        - metric: latency_ms",
                        "          direction: minimize",
                        "  prune:",
                        "    rules:",
                        "      - rule: latency_budget",
                        "        metric: latency_ms",
                        "        condition: gt",
                        "        reference:",
                        "          type: metric",
                        "          metric: latency_budget_ms",
                        "        reason: Latency exceeds deployment budget",
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'tcn'}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                    ]
                )
            )

            settings = load_config(
                config_path=cfg,
                task_metric_names={"rmse_total"},
                training_only_task_metric_names={"rmse_total"},
            )

        self.assertEqual(settings.nas.prune.rules[0].rule, "latency_budget")
        self.assertEqual(settings.nas.prune.rules[0].metric, "latency_ms")

    def test_load_settings_defers_prune_rules_that_depend_on_training_metrics_until_task_validation(self) -> None:
        """Task-dependent prune validation should run after the task contract is known."""
        # Prune rules that depend on task-owned training metrics should survive generic config load and fail only once task-aware validation runs.
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
                        "  quantization:",
                        "    mode: int8_ptq",
                        "    search: false",
                        "    choices: [int8_ptq]",
                        "nas:",
                        "  score:",
                        "    type: scoring-function",
                        "    metrics:",
                        "      total_error:",
                        "        type: add",
                        "        metrics:",
                        "          - rmse_total",
                        "          - flops",
                        "    params:",
                        "      terms:",
                        "        - type: weighted",
                        "          metric: flops",
                        "          weight: -1.0",
                        "  prune:",
                        "    rules:",
                        "      - metric: total_error",
                        "        condition: gt",
                        "        reference:",
                        "          type: literal",
                        "          value: 1.0",
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'tcn'}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                    ]
                )
            )

            settings = load_config(config_path=cfg)

            with self.assertRaisesRegex(ValueError, "training-only"):
                validate_nas_policy_for_task(
                    settings,
                    task_metric_names={"rmse_vel_x", "rmse_vel_y", "rmse_total"},
                    training_only_task_metric_names={"rmse_vel_x", "rmse_vel_y", "rmse_total"},
                )

    def test_load_settings_defers_unknown_metric_terms_until_task_validation(self) -> None:
        """Generic config load should not guess whether a metric is task-owned."""
        # Unknown-looking metric names should remain intact through generic load and fail only when a concrete task contract excludes them.
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
                        "  quantization:",
                        "    mode: int8_ptq",
                        "    search: false",
                        "    choices: [int8_ptq]",
                        "nas:",
                        "  score:",
                        "    type: scoring-function",
                        "    params:",
                        "      terms:",
                        "        - type: weighted",
                        "          metric: weight_storage_mode",
                        "          weight: 1.0",
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'tcn'}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                    ]
                )
            )

            settings = load_config(config_path=cfg)

        self.assertEqual(settings.nas.score.params.terms[0].metric, "weight_storage_mode")
        with self.assertRaisesRegex(ValueError, "unknown metric"):
            validate_nas_policy_for_task(
                settings,
                task_metric_names={"custom_metric"},
                training_only_task_metric_names=set(),
            )

    def test_load_settings_accepts_custom_task_metric_in_score_term(self) -> None:
        """Task-aware validation should allow caller-supplied task metrics."""
        # Custom task metric in score term should remain a supported config shape.
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
                        "  quantization:",
                        "    mode: int8_ptq",
                        "    search: false",
                        "    choices: [int8_ptq]",
                        "nas:",
                        "  score:",
                        "    type: scoring-function",
                        "    params:",
                        "      terms:",
                        "        - type: weighted",
                        "          metric: custom_metric",
                        "          weight: 1.0",
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'tcn'}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                    ]
                )
            )

            settings = load_config(
                config_path=cfg,
                task_metric_names={"custom_metric"},
                training_only_task_metric_names=set(),
            )

        self.assertEqual(settings.nas.score.params.terms[0].metric, "custom_metric")

    def test_load_settings_accepts_unknown_custom_task_metric_without_task_context(self) -> None:
        """Generic config load should preserve potential task metrics verbatim."""
        # Unknown task metric names should remain intact until a concrete task contract is available.
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
                        "  quantization:",
                        "    mode: int8_ptq",
                        "    search: false",
                        "    choices: [int8_ptq]",
                        "nas:",
                        "  score:",
                        "    type: scoring-function",
                        "    params:",
                        "      terms:",
                        "        - type: weighted",
                        "          metric: custom_metric",
                        "          weight: 1.0",
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'tcn'}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                    ]
                )
            )

            settings = load_config(config_path=cfg)

        self.assertEqual(settings.nas.score.params.terms[0].metric, "custom_metric")

    def test_validate_nas_policy_for_task_rejects_unknown_custom_task_metric(self) -> None:
        """Task-aware validation should reject undeclared task metric names."""
        # Once the task contract is known, undeclared custom metric names should fail immediately.
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
                        "  quantization:",
                        "    mode: int8_ptq",
                        "    search: false",
                        "    choices: [int8_ptq]",
                        "nas:",
                        "  score:",
                        "    type: scoring-function",
                        "    params:",
                        "      terms:",
                        "        - type: weighted",
                        "          metric: custom_metric",
                        "          weight: 1.0",
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'tcn'}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                    ]
                )
            )

            settings = load_config(config_path=cfg)

        with self.assertRaisesRegex(ValueError, "unknown metric"):
            validate_nas_policy_for_task(
                settings,
                task_metric_names={"different_metric"},
                training_only_task_metric_names=set(),
            )

    def test_validate_nas_policy_for_task_accepts_audio_classification_metrics(self) -> None:
        """Task-aware validation should accept audio metrics without RMSE.

        Returns
        -------
        None
            Asserts `accuracy` and `macro_f1` are sufficient task metrics for
            a classification score config.
        """
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
                        "  quantization:",
                        "    mode: int8_ptq",
                        "    search: false",
                        "    choices: [int8_ptq]",
                        "nas:",
                        "  score:",
                        "    type: scoring-function",
                        "    params:",
                        "      terms:",
                        "        - type: weighted",
                        "          metric: accuracy",
                        "          weight: 1.0",
                        "        - type: weighted",
                        "          metric: macro_f1",
                        "          weight: 0.1",
                        "  prune:",
                        "    rules: []",
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'audio'}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                    ]
                )
            )

            settings = load_config(config_path=cfg)
            validated = validate_nas_policy_for_task(
                settings,
                task_metric_names={"loss", "accuracy", "macro_f1"},
                training_only_task_metric_names=set(),
            )

        self.assertEqual(validated.nas.score.params.terms[0].metric, "accuracy")

    def test_load_settings_accepts_derived_metric_that_references_custom_task_metric(self) -> None:
        """Derived score metrics may depend on caller-supplied task metrics."""
        # Derived metric that references custom task metric should remain a supported config shape.
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
                        "  quantization:",
                        "    mode: int8_ptq",
                        "    search: false",
                        "    choices: [int8_ptq]",
                        "nas:",
                        "  score:",
                        "    type: scoring-function",
                        "    metrics:",
                        "      combined_metric:",
                        "        type: add",
                        "        metrics:",
                        "          - custom_metric",
                        "          - flops",
                        "    params:",
                        "      terms:",
                        "        - type: weighted",
                        "          metric: combined_metric",
                        "          weight: 1.0",
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'tcn'}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                    ]
                )
            )

            settings = load_config(
                config_path=cfg,
                task_metric_names={"custom_metric"},
                training_only_task_metric_names=set(),
            )

        self.assertEqual(settings.nas.score.metrics.combined_metric.metrics, ["custom_metric", "flops"])

    def test_load_settings_rejects_derived_metric_name_that_collides_with_task_metric(self) -> None:
        """Derived metric names may not redefine caller-supplied task metrics."""
        # Invalid derived metric name that collides with task metric should fail during config load so unsupported NAS settings never reach execution.
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
                        "  quantization:",
                        "    mode: int8_ptq",
                        "    search: false",
                        "    choices: [int8_ptq]",
                        "nas:",
                        "  score:",
                        "    type: scoring-function",
                        "    metrics:",
                        "      custom_metric:",
                        "        type: add",
                        "        metrics:",
                        "          - flops",
                        "          - latency_ms",
                        "    params:",
                        "      terms:",
                        "        - type: weighted",
                        "          metric: custom_metric",
                        "          weight: 1.0",
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'tcn'}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                    ]
                )
            )

            with self.assertRaisesRegex(
                ValueError,
                "redefine a built-in or task-declared metric",
            ):
                load_config(
                    config_path=cfg,
                    task_metric_names={"custom_metric"},
                    training_only_task_metric_names=set(),
                )

    def test_load_settings_defaults_training_only_task_metrics_to_empty_when_task_metrics_are_supplied(self) -> None:
        """Custom task metrics should not silently inherit odometry training-only defaults."""
        # Omitted training only task metrics to empty when task metrics are supplied should fall back to the documented loader defaults.
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
                        "  quantization:",
                        "    mode: int8_ptq",
                        "    search: false",
                        "    choices: [int8_ptq]",
                        "nas:",
                        "  score:",
                        "    type: scoring-function",
                        "    params:",
                        "      terms:",
                        "        - type: weighted",
                        "          metric: custom_metric",
                        "          weight: 1.0",
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'tcn'}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                    ]
                )
            )

            settings = load_config(
                config_path=cfg,
                task_metric_names={"custom_metric"},
            )

        self.assertEqual(settings.nas.score.params.terms[0].metric, "custom_metric")

    def test_load_settings_rejects_task_metric_overlap_with_infrastructure_metrics(self) -> None:
        """Task metrics may not reuse reserved infrastructure metric names."""
        # Invalid task metric overlap with infrastructure metrics should fail during config load so unsupported NAS settings never reach execution.
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
                        *self._score_lines(),
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'tcn'}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                    ]
                )
            )

            with self.assertRaisesRegex(ValueError, "overlaps a reserved infrastructure metric name"):
                load_config(
                    config_path=cfg,
                    task_metric_names={"latency_ms"},
                    training_only_task_metric_names=set(),
                )

    def test_load_settings_rejects_training_only_task_metrics_outside_task_metric_set(self) -> None:
        """Training-only task metrics must be a subset of the task metric set."""
        # Invalid training only task metrics outside task metric set should fail during config load so unsupported NAS settings never reach execution.
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
                        *self._score_lines(),
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'tcn'}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                    ]
                )
            )

            with self.assertRaisesRegex(ValueError, "must be a subset"):
                load_config(
                    config_path=cfg,
                    task_metric_names={"custom_metric"},
                    training_only_task_metric_names={"other_metric"},
                )

    def test_load_settings_accepts_custom_task_metric_in_prune_rules(self) -> None:
        """Prune validation should allow supplied non-training-only task metrics."""
        # Custom task metric in prune rules should remain a supported config shape.
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
                        "  quantization:",
                        "    mode: int8_ptq",
                        "    search: false",
                        "    choices: [int8_ptq]",
                        "nas:",
                        "  score:",
                        "    type: scoring-function",
                        "    params:",
                        "      terms:",
                        "        - type: weighted",
                        "          metric: flops",
                        "          weight: -1.0",
                        "  prune:",
                        "    rules:",
                        "      - rule: custom_task_gate",
                        "        metric: custom_metric",
                        "        condition: gt",
                        "        reference:",
                        "          type: literal",
                        "          value: 0.0",
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'tcn'}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                    ]
                )
            )

            settings = load_config(
                config_path=cfg,
                task_metric_names={"custom_metric"},
                training_only_task_metric_names=set(),
            )

        self.assertEqual(settings.nas.prune.rules[0].metric, "custom_metric")

    def test_validate_nas_policy_for_task_rejects_unknown_multiobjective_prune_metric(self) -> None:
        """Task-aware validation should reject undeclared multi-objective gate metrics.

        Returns
        -------
        None
            Asserts generic load preserves the metric and task-aware validation
            rejects it.
        """
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
                        "  quantization:",
                        "    mode: int8_ptq",
                        "    search: false",
                        "    choices: [int8_ptq]",
                        "nas:",
                        "  score:",
                        "    type: multi-objective",
                        "    params:",
                        "      objectives:",
                        "        - metric: flops",
                        "          direction: minimize",
                        "  prune:",
                        "    rules:",
                        "      - rule: custom_task_gate",
                        "        metric: custom_metric",
                        "        condition: gt",
                        "        reference:",
                        "          type: literal",
                        "          value: 0.0",
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'tcn'}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                    ]
                )
            )

            settings = load_config(config_path=cfg)

        self.assertEqual(settings.nas.prune.rules[0].metric, "custom_metric")
        with self.assertRaisesRegex(ValueError, "unknown metric"):
            validate_nas_policy_for_task(
                settings,
                task_metric_names={"different_metric"},
                training_only_task_metric_names=set(),
            )

    def test_load_settings_rejects_prune_rules_that_use_custom_training_only_task_metrics(self) -> None:
        """Prune rules may not directly read task metrics that need training."""
        # Custom training-only task metrics cannot appear in prune rules because those values are unavailable before fit() runs.
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
                        "  quantization:",
                        "    mode: int8_ptq",
                        "    search: false",
                        "    choices: [int8_ptq]",
                        "nas:",
                        "  score:",
                        "    type: scoring-function",
                        "    params:",
                        "      terms:",
                        "        - type: weighted",
                        "          metric: flops",
                        "          weight: -1.0",
                        "  prune:",
                        "    rules:",
                        "      - rule: custom_task_gate",
                        "        metric: custom_metric",
                        "        condition: gt",
                        "        reference:",
                        "          type: literal",
                        "          value: 0.0",
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'tcn'}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                    ]
                )
            )

            with self.assertRaisesRegex(ValueError, "training-only"):
                load_config(
                    config_path=cfg,
                    task_metric_names={"custom_metric"},
                    training_only_task_metric_names={"custom_metric"},
                )

    def test_load_settings_rejects_multiobjective_prune_rules_with_training_only_metrics(self) -> None:
        """Multi-objective prune rules may not read post-training task metrics.

        Returns
        -------
        None
            Asserts direct training-only task metrics are invalid in
            multi-objective gates.
        """
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
                        "  quantization:",
                        "    mode: int8_ptq",
                        "    search: false",
                        "    choices: [int8_ptq]",
                        "nas:",
                        "  score:",
                        "    type: multi-objective",
                        "    params:",
                        "      objectives:",
                        "        - metric: flops",
                        "          direction: minimize",
                        "  prune:",
                        "    rules:",
                        "      - rule: custom_task_gate",
                        "        metric: custom_metric",
                        "        condition: gt",
                        "        reference:",
                        "          type: literal",
                        "          value: 0.0",
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'tcn'}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                    ]
                )
            )

            with self.assertRaisesRegex(ValueError, "training-only"):
                load_config(
                    config_path=cfg,
                    task_metric_names={"custom_metric"},
                    training_only_task_metric_names={"custom_metric"},
                )

    def test_load_settings_rejects_multiobjective_prune_reference_with_training_only_metrics(self) -> None:
        """Multi-objective prune references may not read post-training task metrics.

        Returns
        -------
        None
            Asserts reference metrics that need training are invalid in
            multi-objective gates.
        """
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
                        "  quantization:",
                        "    mode: int8_ptq",
                        "    search: false",
                        "    choices: [int8_ptq]",
                        "nas:",
                        "  score:",
                        "    type: multi-objective",
                        "    params:",
                        "      objectives:",
                        "        - metric: flops",
                        "          direction: minimize",
                        "  prune:",
                        "    rules:",
                        "      - rule: custom_task_gate",
                        "        metric: flops",
                        "        condition: gt",
                        "        reference:",
                        "          type: metric",
                        "          metric: custom_metric",
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'tcn'}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                    ]
                )
            )

            with self.assertRaisesRegex(ValueError, "training-only"):
                load_config(
                    config_path=cfg,
                    task_metric_names={"custom_metric"},
                    training_only_task_metric_names={"custom_metric"},
                )

    def test_load_settings_rejects_prune_rules_that_depend_on_custom_training_only_task_metrics(self) -> None:
        """Prune rules may not depend indirectly on task metrics that need training."""
        # Derived prune metrics cannot close over training-only task signals because prune decisions happen before training.
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
                        "  quantization:",
                        "    mode: int8_ptq",
                        "    search: false",
                        "    choices: [int8_ptq]",
                        "nas:",
                        "  score:",
                        "    type: scoring-function",
                        "    metrics:",
                        "      combined_metric:",
                        "        type: add",
                        "        metrics:",
                        "          - custom_metric",
                        "          - flops",
                        "    params:",
                        "      terms:",
                        "        - type: weighted",
                        "          metric: flops",
                        "          weight: -1.0",
                        "  prune:",
                        "    rules:",
                        "      - rule: custom_task_gate",
                        "        metric: combined_metric",
                        "        condition: gt",
                        "        reference:",
                        "          type: literal",
                        "          value: 0.0",
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'tcn'}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                    ]
                )
            )

            with self.assertRaisesRegex(ValueError, "training-only"):
                load_config(
                    config_path=cfg,
                    task_metric_names={"custom_metric"},
                    training_only_task_metric_names={"custom_metric"},
                )

    def test_load_settings_rejects_multiobjective_prune_rules_with_training_dependent_derived_metrics(self) -> None:
        """Multi-objective prune rules may not indirectly depend on training metrics.

        Returns
        -------
        None
            Asserts derived metrics that depend on training-only metrics are
            invalid in multi-objective gates.
        """
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
                        "  quantization:",
                        "    mode: int8_ptq",
                        "    search: false",
                        "    choices: [int8_ptq]",
                        "nas:",
                        "  score:",
                        "    type: multi-objective",
                        "    metrics:",
                        "      combined_metric:",
                        "        type: add",
                        "        metrics:",
                        "          - custom_metric",
                        "          - flops",
                        "    params:",
                        "      objectives:",
                        "        - metric: flops",
                        "          direction: minimize",
                        "  prune:",
                        "    rules:",
                        "      - rule: custom_task_gate",
                        "        metric: combined_metric",
                        "        condition: gt",
                        "        reference:",
                        "          type: literal",
                        "          value: 0.0",
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'tcn'}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                    ]
                )
            )

            with self.assertRaisesRegex(ValueError, "training-only"):
                load_config(
                    config_path=cfg,
                    task_metric_names={"custom_metric"},
                    training_only_task_metric_names={"custom_metric"},
                )

    def test_load_settings_accepts_cadenced_sleep_metric_in_score_terms(self) -> None:
        # Cadenced sleep metric in score terms should remain a supported config shape.
        """Validate load settings accepts cadenced sleep metric in score terms."""
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
                        "  quantization:",
                        "    mode: int8_ptq",
                        "    search: false",
                        "    choices: [int8_ptq]",
                        "nas:",
                        "  score:",
                        "    type: scoring-function",
                        "    params:",
                        "      terms:",
                        "        - type: weighted",
                        "          metric: cadenced_rtc_sleep_ms",
                        "          weight: -1.0",
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'tcn'}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                    ]
                )
            )

            settings = load_config(config_path=cfg)

        self.assertEqual(settings.nas.score.params.terms[0].metric, "cadenced_rtc_sleep_ms")

    def test_load_settings_accepts_cadenced_deadline_metric_in_prune_rules(self) -> None:
        # Cadenced deadline metric in prune rules should remain a supported config shape.
        """Validate load settings accepts cadenced deadline metric in prune rules."""
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
                        "  quantization:",
                        "    mode: int8_ptq",
                        "    search: false",
                        "    choices: [int8_ptq]",
                        "nas:",
                        "  score:",
                        "    type: scoring-function",
                        "    params:",
                        "      terms:",
                        "        - type: weighted",
                        "          metric: latency_ms",
                        "          weight: -1.0",
                        "  prune:",
                        "    rules:",
                        "      - rule: deadline_budget",
                        "        metric: cadenced_deadline_miss_count",
                        "        condition: gt",
                        "        reference:",
                        "          type: literal",
                        "          value: 0",
                        "        reason: deadline misses exceed budget",
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'tcn'}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                    ]
                )
            )

            settings = load_config(config_path=cfg)

        self.assertEqual(settings.nas.prune.rules[0].metric, "cadenced_deadline_miss_count")

    def test_load_settings_accepts_cadenced_error_code_in_score_terms(self) -> None:
        # Cadenced error code in score terms should remain a supported config shape.
        """Validate load settings accepts cadenced error code in score terms."""
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
                        "  quantization:",
                        "    mode: int8_ptq",
                        "    search: false",
                        "    choices: [int8_ptq]",
                        "nas:",
                        "  score:",
                        "    type: scoring-function",
                        "    params:",
                        "      terms:",
                        "        - type: weighted",
                        "          metric: cadenced_error_code",
                        "          weight: -1.0",
                        "  prune:",
                        "    rules: []",
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'tcn'}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                    ]
                )
            )

            settings = load_config(config_path=cfg)

        self.assertEqual(settings.nas.score.params.terms[0].metric, "cadenced_error_code")

    def test_load_settings_accepts_cadenced_error_code_in_prune_rules(self) -> None:
        # Cadenced error code in prune rules should remain a supported config shape.
        """Validate load settings accepts cadenced error code in prune rules."""
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
                        "  quantization:",
                        "    mode: int8_ptq",
                        "    search: false",
                        "    choices: [int8_ptq]",
                        "nas:",
                        "  score:",
                        "    type: scoring-function",
                        "    params:",
                        "      terms:",
                        "        - type: weighted",
                        "          metric: latency_ms",
                        "          weight: -1.0",
                        "  prune:",
                        "    rules:",
                        "      - rule: cadenced_phase_ok",
                        "        metric: cadenced_error_code",
                        "        condition: gt",
                        "        reference:",
                        "          type: literal",
                        "          value: 0",
                        "        reason: cadenced phase failed",
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'tcn'}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                    ]
                )
            )

            settings = load_config(config_path=cfg)

        self.assertEqual(settings.nas.prune.rules[0].metric, "cadenced_error_code")

    def test_load_settings_accepts_feasibility_rules(self) -> None:
        """Valid pre-training feasibility rules should normalize from config."""
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
                        "  quantization:",
                        "    mode: int8_ptq",
                        "    search: false",
                        "    choices: [int8_ptq]",
                        "nas:",
                        "  score:",
                        "    type: scoring-function",
                        "    params:",
                        "      terms:",
                        "        - type: weighted",
                        "          metric: flops",
                        "          weight: -1.0",
                        "  feasibility:",
                        "    train_if_infeasible: true",
                        "    rules:",
                        "      - rule: latency_budget",
                        "        metric: latency_ms",
                        "        condition: gt",
                        "        reference:",
                        "          type: metric",
                        "          metric: latency_budget_ms",
                        "        reason: Latency exceeds budget",
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'tcn'}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                    ]
                )
            )

            settings = load_config(config_path=cfg)

        self.assertTrue(settings.nas.feasibility.train_if_infeasible)
        self.assertEqual(settings.nas.feasibility.rules[0].rule, "latency_budget")
        self.assertEqual(settings.nas.feasibility.rules[0].reference.metric, "latency_budget_ms")

    def test_validate_nas_policy_rejects_training_only_feasibility_metric(self) -> None:
        """Feasibility rules must not depend on post-training task metrics."""
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
                        "  quantization:",
                        "    mode: int8_ptq",
                        "    search: false",
                        "    choices: [int8_ptq]",
                        "nas:",
                        "  score:",
                        "    type: scoring-function",
                        "    params:",
                        "      terms:",
                        "        - type: weighted",
                        "          metric: flops",
                        "          weight: -1.0",
                        "  feasibility:",
                        "    rules:",
                        "      - rule: rmse_gate",
                        "        metric: rmse_total",
                        "        condition: gt",
                        "        reference:",
                        "          type: literal",
                        "          value: 1.0",
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'tcn'}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                    ]
                )
            )

            settings = load_config(config_path=cfg)
            with self.assertRaisesRegex(ValueError, "training-only metric 'rmse_total'"):
                validate_nas_policy_for_task(
                    settings,
                    task_metric_names={"rmse_total"},
                    training_only_task_metric_names={"rmse_total"},
                )

    def test_evaluate_feasibility_rules_computes_signed_constraints(self) -> None:
        """Feasibility rules should emit signed Optuna constraints in order."""
        score_config = Dict(type="scoring-function", metrics=Dict(), params=Dict())
        feasibility_config = Dict(
            rules=[
                Dict(rule="gt", metric="latency_ms", condition="gt", reference=Dict(type="literal", value=10.0), reason=""),
                Dict(rule="gte", metric="latency_ms", condition="gte", reference=Dict(type="literal", value=12.0), reason=""),
                Dict(rule="lt", metric="latency_ms", condition="lt", reference=Dict(type="literal", value=20.0), reason=""),
                Dict(rule="lte", metric="latency_ms", condition="lte", reference=Dict(type="literal", value=12.0), reason=""),
            ]
        )

        result = evaluate_feasibility_rules(
            metrics={"latency_ms": 12.0},
            hyperparams=Dict(flops=1),
            score_config=score_config,
            feasibility_config=feasibility_config,
        )

        self.assertFalse(result.feasible)
        self.assertEqual(result.status, "infeasible")
        self.assertGreater(result.constraints[0], 0.0)
        self.assertGreater(result.constraints[1], 0.0)
        self.assertGreater(result.constraints[2], 0.0)
        self.assertGreater(result.constraints[3], 0.0)
        self.assertEqual(result.first_violation["rule"], "gt")

    def test_load_settings_missing_file(self) -> None:
        """Nonexistent config paths should raise FileNotFoundError."""
        # Missing config files should fail immediately instead of producing a partially initialized runtime.
        with self.assertRaises(FileNotFoundError):
            load_config(config_path=Path("does_not_exist.yaml"))

    def test_load_settings_does_not_apply_backend_specific_harness_validation(self) -> None:
        # Config loading should stay backend-agnostic and leave harness compatibility checks to the runtime-specific layers.
        """Validate load settings does not apply backend specific harness validation."""
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
                        "  quantization:",
                        "    mode: int8_ptq",
                        "    search: false",
                        "    choices: [int8_ptq]",
                        *self._score_lines(),
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'tcn'}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                    ]
                )
            )

            settings = load_config(config_path=cfg)
            self.assertTrue(settings.training.energy_aware)

    def test_resolve_device_options_normalizes_stm_backend_block(self) -> None:
        """Ensure STM backend option normalization moved into the resolver.

        Returns
        -------
        None
        """
        # STM32 backend blocks should normalize into one canonical option shape before request construction uses them.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            project_root = tmp_path / "stm32_project" / "FSBL"
            (project_root / "Debug").mkdir(parents=True)
            (project_root / "Debug" / "makefile").write_text("# makefile\n")
            cfg = tmp_path / "stm32.yaml"
            cfg.write_text(
                "\n".join(
                    [
                        "device:",
                        "  name: STM32_NUCLEO_N657X0_Q",
                        "  stm32:",
                        f"    template_root: \"{project_root}\"",
                        "    gdb_port: 61235",
                        "    apid: 2",
                        "    server_ready_timeout_s: 20.0",
                        "training:",
                        "  nas_trials: 5",
                        "  energy_aware: true",
                        "  quantization:",
                        "    mode: int8_ptq",
                        "    search: false",
                        "    choices: [int8_ptq]",
                        *self._score_lines(),
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'tcn'}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                    ]
                )
            )

            settings = load_config(config_path=cfg)
            resolved = resolve_device_options(str(settings.device.name), settings.device)

            self.assertEqual(resolved["project_root"], project_root.resolve())
            self.assertEqual(resolved["gdb_port"], 61235)
            self.assertEqual(resolved["apid"], 2)
            self.assertEqual(resolved["server_ready_timeout_s"], 20.0)
            self.assertEqual(resolved["cpu_clock_mhz"], 600)

    def test_load_settings_accepts_null_cpu_clock_options(self) -> None:
        # Null CPU-clock option lists should preserve the board default instead of inventing a search space.
        """Validate load settings accepts null cpu clock options."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            cfg = tmp_path / "config.yaml"
            cfg.write_text(
                "\n".join(
                    [
                        "device:",
                        "  name: TEST_DEVICE",
                        "  cpu_clock_mhz_options: null",
                        "training:",
                        "  nas_trials: 5",
                        "  quantization:",
                        "    mode: int8_ptq",
                        "    search: false",
                        "    choices: [int8_ptq]",
                        *self._score_lines(),
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'tcn'}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                    ]
                )
            )

            settings = load_config(config_path=cfg)

        self.assertIsNone(settings.device.cpu_clock_mhz_options)

    def test_load_settings_rejects_boolean_cpu_clock_options(self) -> None:
        # Invalid boolean CPU clock options should fail during config load so unsupported NAS settings never reach execution.
        """Validate load settings rejects boolean cpu clock options."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            cfg = tmp_path / "config.yaml"
            cfg.write_text(
                "\n".join(
                    [
                        "device:",
                        "  name: TEST_DEVICE",
                        "  cpu_clock_mhz_options: [true, 400]",
                        "training:",
                        "  nas_trials: 5",
                        "  quantization:",
                        "    mode: int8_ptq",
                        "    search: false",
                        "    choices: [int8_ptq]",
                        *self._score_lines(),
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'tcn'}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                    ]
                )
            )

            with self.assertRaises(ValueError):
                load_config(config_path=cfg)

    def test_load_settings_rejects_unsupported_stm_cpu_clock_options(self) -> None:
        # Invalid unsupported STM32 CPU clock options should fail during config load so unsupported NAS settings never reach execution.
        """Validate load settings rejects unsupported stm cpu clock options."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            cfg = tmp_path / "config.yaml"
            cfg.write_text(
                "\n".join(
                    [
                        "device:",
                        "  name: STM32_NUCLEO_N657X0_Q",
                        "  cpu_clock_mhz_options: [250, 400]",
                        "training:",
                        "  nas_trials: 5",
                        "  quantization:",
                        "    mode: int8_ptq",
                        "    search: false",
                        "    choices: [int8_ptq]",
                        *self._score_lines(),
                        "outputs:",
                        f"  models_dir: \"{tmp_path / 'models'}\"",
                        f"  candidate_dir: \"{tmp_path / 'tcn'}\"",
                        "  artifact_stem: \"TinyOdomEx_Test\"",
                    ]
                )
            )

            with self.assertRaisesRegex(ValueError, "must be one of"):
                load_config(config_path=cfg)


class ScoreConfigTrainingDependencyTests(unittest.TestCase):
    """Validate task-aware training-metric detection helpers."""

    def test_score_config_uses_training_metrics_detects_multilevel_derived_dependency(self) -> None:
        # Derived score terms should count as training-dependent even when the training-only metric is hidden behind another derived metric.
        """Validate score config uses training metrics detects multilevel derived dependency."""
        score_config = Dict(
            type="scoring-function",
            metrics=Dict(
                level_one=Dict(type="add", metrics=["level_two", "flops"]),
                level_two=Dict(type="add", metrics=["custom_training_metric", "latency_ms"]),
            ),
            params=Dict(terms=[Dict(type="weighted", metric="level_one", weight=1.0)]),
        )

        self.assertTrue(
            score_config_uses_training_metrics(
                score_config,
                training_only_metric_names={"custom_training_metric"},
            )
        )

    def test_score_config_uses_training_metrics_detects_typed_reference_dependency(self) -> None:
        # Reference metrics should also mark the score config as training-dependent when they ultimately read a training-only value.
        """Validate score config uses training metrics detects typed reference dependency."""
        score_config = Dict(
            type="scoring-function",
            metrics=Dict(
                custom_reference_metric=Dict(
                    type="add",
                    metrics=["custom_training_metric", "flops"],
                )
            ),
            params=Dict(
                terms=[
                    Dict(
                        type="normalized-weighted",
                        metric="flops",
                        weight=1.0,
                        reference=Dict(type="metric", metric="custom_reference_metric"),
                    )
                ]
            ),
        )

        self.assertTrue(
            score_config_uses_training_metrics(
                score_config,
                training_only_metric_names={"custom_training_metric"},
            )
        )

    def test_score_config_uses_training_metrics_returns_false_for_non_training_metrics(self) -> None:
        # Pure deployment metrics should not mark the score config as training-dependent.
        """Validate score config uses training metrics returns false for non training metrics."""
        score_config = Dict(
            type="scoring-function",
            metrics=Dict(
                combined_metric=Dict(type="add", metrics=["custom_metric", "flops"]),
            ),
            params=Dict(terms=[Dict(type="weighted", metric="combined_metric", weight=1.0)]),
        )

        self.assertFalse(
            score_config_uses_training_metrics(
                score_config,
                training_only_metric_names=set(),
            )
        )

    def test_score_config_uses_training_metrics_keeps_default_odometry_behavior(self) -> None:
        # Default odometry objectives should still count as training-dependent because they rely on RMSE outputs from fit().
        """Validate score config uses training metrics keeps default odometry behavior."""
        score_config = Dict(
            type="multi-objective",
            metrics=Dict(),
            params=Dict(objectives=[Dict(metric="rmse_total", direction="minimize")]),
        )

        self.assertTrue(score_config_uses_training_metrics(score_config))

class FakeTrial:
    """Small Optuna-trial stand-in used by CSV logging tests."""

    def __init__(self):
        """Initialize an attribute store mirroring Optuna's user attrs."""
        self.attrs = {}

    def set_user_attr(self, key, value):
        """Persist one user attribute exactly as Optuna would expose it.

        Parameters
        ----------
        key : object
            Dictionary key recorded by the test double.
        value : object
            Value recorded by the test double.
        """
        self.attrs[key] = value


class LogTrialTests(unittest.TestCase):
    """Tests covering log trial behavior."""

    HEADER = [
        *TRIAL_LOG_STABLE_COLUMNS,
        "metric__rmse_total",
        "metric__rmse_vel_x",
        "metric__rmse_vel_y",
        "hparam__dilations",
        "hparam__dropout_rate",
        "hparam__kernel_size",
        "hparam__nb_filters",
        "hparam__norm_flag",
        "hparam__use_skip_connections",
    ]

    def _sample_metrics(self):
        """Return a representative metric payload for trial-log tests.

        Returns
        -------
        object
            Representative metrics dictionary for a logged trial.
        """
        return {
            "ram_bytes": 1000,
            "flash_bytes": 2000,
            "external_flash_bytes": 3000,
            "weight_storage_mode": "external_flash",
            "weight_bytes": 4096,
            "activation_bytes": 8192,
            "memory_traffic_bytes": 12288,
            "memory_proxy_dtype_bytes": 1,
            "memory_proxy_warning_count": 4,
            "rmse_total": 0.3,
            "latency_ms": 10,
            "latency_budget_ms": -1,
            "arena_bytes": 4096,
            "error_code": 0,
            "error_label": "HIL_MASTER_PENDING",
            "energy_mj_per_inference": 0.5,
            "avg_power_mw": 2.0,
            "avg_current_ma": 1.5,
            "bus_voltage_v": 3.3,
            "cpu_clock_mhz_requested": 400,
            "clock_hz": 399_000_000.0,
            "idle_power_mw": 2.0,
            "runtime_mode": "back_to_back",
            "cadenced_error_code": -1,
            "cadenced_error_label": None,
            "cadenced_active_inference_latency_ms": -1.0,
            "cadenced_window_latency_ms": -1.0,
            "cadenced_energy_mj_per_window": -1.0,
            "cadenced_energy_mj_per_trial": -1.0,
            "cadenced_rtc_sleep_ms": -1.0,
            "cadenced_deadline_miss_count": -1,
        }

    def _sample_hyperparams(self):
        """Return a representative hyperparameter payload for trial-log tests.

        Returns
        -------
        object
            Representative hyperparameter dictionary for a logged trial.
        """
        return {
            "flops": 1_000_000,
            "weight_bytes": 4096,
            "activation_bytes": 8192,
            "memory_traffic_bytes": 12288,
            "memory_proxy_dtype_bytes": 1,
            "memory_proxy_warning_count": 4,
            "nb_filters": 32,
            "kernel_size": 3,
            "dilations": [1, 2, 4],
            "dropout_rate": 0.1,
            "use_skip_connections": True,
            "norm_flag": True,
        }

    def _sample_trial_outcome(
        self,
        *,
        score: float | None = 0.5,
        objective_names: list[str] | None = None,
        objective_values: list[float] | None = None,
        objective_directions: list[str] | None = None,
        artifact_summary: dict[str, object] | None = None,
        task_metrics: dict[str, object] | None = None,
        hyperparams: dict[str, object] | None = None,
    ) -> TrialOutcome:
        """Build a representative ``TrialOutcome`` for CSV logging tests.

        Parameters
        ----------
        score : float | None
            Optimization score recorded in the fake trial summary.
        objective_names : list[str] | None
            Objective names stored in the fake trial summary.
        objective_values : list[float] | None
            Objective values stored in the fake trial summary.
        objective_directions : list[str] | None
            Objective directions stored in the fake trial summary.
        artifact_summary : dict[str, object] | None
            Artifact summary included in the sample outcome.
        task_metrics : dict[str, object] | None
            Task metrics included in the sample outcome.
        hyperparams : dict[str, object] | None
            Hyperparameters included in the sample outcome.

        Returns
        -------
        TrialOutcome
            Representative trial outcome used by logging tests.
        """
        if objective_names is None:
            objective_names = ["score"]
        if objective_values is None:
            objective_values = [score if score is not None else 0.0]
        if objective_directions is None:
            objective_directions = ["maximize"]
        if task_metrics is None:
            task_metrics = {
                "rmse_vel_x": 0.1,
                "rmse_vel_y": 0.2,
                "rmse_total": 0.3,
            }
        if hyperparams is None:
            hyperparams = self._sample_hyperparams()
        return TrialOutcome(
            score=score,
            objective_names=objective_names,
            objective_values=objective_values,
            objective_directions=objective_directions,
            task_metrics=task_metrics,
            hyperparams=hyperparams,
            artifact_summary=artifact_summary,
        )

    def test_log_trial_writes_header_and_row(self):
        # The first trial log entry should write both the stable header and a fully populated row so post-run CSV tools can parse the file immediately.
        """Validate log trial writes header and row."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "log.csv"
            fake_trial = FakeTrial()
            metrics = self._sample_metrics()
            trial_outcome = self._sample_trial_outcome(
                artifact_summary={"plot_path": "plots/demo.png"}
            )

            with patch("tinyodom.model.time.time", return_value=123.0), patch(
                "tinyodom.model.time.strftime", return_value="01-02-1970 00:02:03"
            ):
                log_trial(
                    trial_outcome=trial_outcome,
                    metrics=metrics,
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
                int(rows[1][header_index["external_flash_bytes"]]),
                metrics["external_flash_bytes"],
            )
            self.assertEqual(
                rows[1][header_index["weight_storage_mode"]],
                metrics["weight_storage_mode"],
            )
            self.assertEqual(int(rows[1][header_index["weight_bytes"]]), metrics["weight_bytes"])
            self.assertEqual(
                int(rows[1][header_index["activation_bytes"]]),
                metrics["activation_bytes"],
            )
            self.assertEqual(
                int(rows[1][header_index["memory_traffic_bytes"]]),
                metrics["memory_traffic_bytes"],
            )
            self.assertEqual(
                int(rows[1][header_index["memory_proxy_dtype_bytes"]]),
                metrics["memory_proxy_dtype_bytes"],
            )
            self.assertEqual(
                int(rows[1][header_index["memory_proxy_warning_count"]]),
                metrics["memory_proxy_warning_count"],
            )
            self.assertEqual(
                float(rows[1][header_index["latency_ms"]]), metrics["latency_ms"]
            )
            self.assertEqual(
                float(rows[1][header_index["latency_budget_ms"]]),
                metrics["latency_budget_ms"],
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
                int(rows[1][header_index["cpu_clock_mhz_requested"]]),
                metrics["cpu_clock_mhz_requested"],
            )
            self.assertAlmostEqual(
                float(rows[1][header_index["clock_hz"]]), metrics["clock_hz"]
            )
            self.assertEqual(
                rows[1][header_index["error_label"]], metrics["error_label"]
            )
            self.assertEqual(rows[1][header_index["score_type"]], "scoring-function")
            self.assertEqual(
                rows[1][header_index["artifact_summary_json"]],
                '{"plot_path": "plots/demo.png"}',
            )
            self.assertEqual(rows[1][header_index["pruned"]], "False")
            self.assertEqual(rows[1][header_index["prune_reason"]], "")
            self.assertEqual(rows[1][header_index["prune_rule"]], "")
            self.assertEqual(rows[1][header_index["runtime_mode"]], "back_to_back")
            self.assertEqual(rows[1][header_index["cadenced_window_latency_ms"]], "-1.0")
            self.assertEqual(rows[1][header_index["cadenced_energy_mj_per_window"]], "-1.0")
            self.assertEqual(rows[1][header_index["cadenced_energy_mj_per_trial"]], "-1.0")
            self.assertEqual(rows[1][header_index["cadenced_error_code"]], "-1")
            self.assertEqual(rows[1][header_index["metric__rmse_vel_x"]], "0.1")
            self.assertEqual(rows[1][header_index["metric__rmse_vel_y"]], "0.2")
            self.assertEqual(rows[1][header_index["metric__rmse_total"]], "0.3")
            self.assertEqual(rows[1][header_index["hparam__nb_filters"]], "32")
            self.assertEqual(rows[1][header_index["hparam__kernel_size"]], "3")
            self.assertEqual(
                fake_trial.attrs["cadenced_window_latency_ms"],
                metrics["cadenced_window_latency_ms"],
            )
            self.assertEqual(
                fake_trial.attrs["cadenced_energy_mj_per_window"],
                metrics["cadenced_energy_mj_per_window"],
            )
            self.assertEqual(
                fake_trial.attrs["cadenced_energy_mj_per_trial"],
                metrics["cadenced_energy_mj_per_trial"],
            )

            self.assertEqual(fake_trial.attrs["ram_bytes"], metrics["ram_bytes"])
            self.assertEqual(
                fake_trial.attrs["external_flash_bytes"],
                metrics["external_flash_bytes"],
            )
            self.assertEqual(
                fake_trial.attrs["weight_storage_mode"],
                metrics["weight_storage_mode"],
            )
            self.assertEqual(fake_trial.attrs["weight_bytes"], metrics["weight_bytes"])
            self.assertEqual(fake_trial.attrs["activation_bytes"], metrics["activation_bytes"])
            self.assertEqual(
                fake_trial.attrs["memory_traffic_bytes"],
                metrics["memory_traffic_bytes"],
            )
            self.assertEqual(
                fake_trial.attrs["memory_proxy_dtype_bytes"],
                metrics["memory_proxy_dtype_bytes"],
            )
            self.assertEqual(
                fake_trial.attrs["memory_proxy_warning_count"],
                metrics["memory_proxy_warning_count"],
            )
            self.assertNotIn("hparam__memory_traffic_bytes", fake_trial.attrs)
            self.assertEqual(fake_trial.attrs["task_metrics"], trial_outcome.task_metrics)
            self.assertEqual(fake_trial.attrs["metric__rmse_vel_x"], 0.1)
            self.assertEqual(fake_trial.attrs["metric__rmse_vel_y"], 0.2)
            self.assertEqual(fake_trial.attrs["metric__rmse_total"], 0.3)
            self.assertEqual(fake_trial.attrs["hyperparameters"], trial_outcome.hyperparams)
            self.assertEqual(fake_trial.attrs["hparam__nb_filters"], 32)
            self.assertEqual(fake_trial.attrs["artifact_summary"], {"plot_path": "plots/demo.png"})
            self.assertEqual(fake_trial.attrs["latency_budget_ms"], metrics["latency_budget_ms"])
            self.assertEqual(
                fake_trial.attrs["cpu_clock_mhz_requested"],
                metrics["cpu_clock_mhz_requested"],
            )
            self.assertEqual(fake_trial.attrs["clock_hz"], metrics["clock_hz"])
            self.assertEqual(
                fake_trial.attrs["error_code_label"], metrics["error_label"]
            )
            self.assertEqual(
                fake_trial.attrs["energy_mj_per_inference"],
                metrics["energy_mj_per_inference"],
            )
            self.assertEqual(fake_trial.attrs["prune_rule"], "")
            self.assertEqual(fake_trial.attrs["runtime_mode"], "back_to_back")

    def test_log_trial_appends_without_duplicate_header(self):
        # Appending later trials should preserve a single CSV header so repeated runs stay spreadsheet-friendly.
        """Validate log trial appends without duplicate header."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "log.csv"
            metrics = self._sample_metrics()

            fake_trial_one = FakeTrial()
            fake_trial_two = FakeTrial()

            log_trial(
                trial_outcome=self._sample_trial_outcome(
                    score=0.3,
                    objective_values=[0.3],
                    task_metrics={
                        "rmse_vel_x": 0.05,
                        "rmse_vel_y": 0.06,
                        "rmse_total": 0.11,
                    },
                ),
                metrics=metrics,
                trial=fake_trial_one,
                log_file_name=str(log_path),
            )
            log_trial(
                trial_outcome=self._sample_trial_outcome(
                    score=0.2,
                    objective_values=[0.2],
                    task_metrics={
                        "rmse_vel_x": 0.04,
                        "rmse_vel_y": 0.05,
                        "rmse_total": 0.09,
                    },
                ),
                metrics=metrics,
                trial=fake_trial_two,
                log_file_name=str(log_path),
            )

            with log_path.open(newline="") as csvfile:
                rows = list(csv.reader(csvfile))

            self.assertEqual(rows[0], self.HEADER)
            self.assertEqual(len(rows), 3)

    def test_log_trial_expands_dynamic_header_and_backfills_prior_rows(self):
        # CSV logging should backfill older rows when new dynamic columns appear so one run does not corrupt the whole trial history.
        """Validate log trial expands dynamic header and backfills prior rows."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "log.csv"
            metrics = self._sample_metrics()

            log_trial(
                trial_outcome=self._sample_trial_outcome(
                    task_metrics={"rmse_total": 0.3},
                    hyperparams={"flops": 1_000_000, "kernel_size": 3},
                ),
                metrics=metrics,
                trial=FakeTrial(),
                log_file_name=str(log_path),
            )
            log_trial(
                trial_outcome=self._sample_trial_outcome(
                    task_metrics={"rmse_total": 0.2, "custom_accuracy": 0.9},
                    hyperparams={
                        "flops": 1_000_000,
                        "kernel_size": 3,
                        "nb_filters": 32,
                    },
                ),
                metrics=metrics,
                trial=FakeTrial(),
                log_file_name=str(log_path),
            )

            with log_path.open(newline="") as csvfile:
                rows = list(csv.DictReader(csvfile))

            self.assertEqual(
                list(rows[0].keys()),
                [
                    *TRIAL_LOG_STABLE_COLUMNS,
                    "metric__custom_accuracy",
                    "metric__rmse_total",
                    "hparam__kernel_size",
                    "hparam__nb_filters",
                ],
            )
            self.assertEqual(rows[0]["metric__custom_accuracy"], "")
            self.assertEqual(rows[0]["hparam__nb_filters"], "")
            self.assertEqual(rows[1]["metric__custom_accuracy"], "0.9")
            self.assertEqual(rows[1]["hparam__nb_filters"], "32")

    def test_log_trial_marks_single_objective_multiobjective_runs_correctly(self):
        # Single-objective multi-objective runs should still mark their score type correctly so downstream dashboards do not misclassify them.
        """Validate log trial marks single objective multiobjective runs correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "log.csv"
            fake_trial = FakeTrial()
            metrics = self._sample_metrics()

            log_trial(
                trial_outcome=self._sample_trial_outcome(
                    score=None,
                    objective_names=["rmse_total"],
                    objective_values=[0.3],
                    objective_directions=["minimize"],
                    task_metrics={
                        "rmse_vel_x": 0.1,
                        "rmse_vel_y": 0.2,
                        "rmse_total": 0.3,
                    },
                ),
                metrics=metrics,
                trial=fake_trial,
                log_file_name=str(log_path),
            )

            with log_path.open(newline="") as csvfile:
                rows = list(csv.reader(csvfile))

            header_index = {name: idx for idx, name in enumerate(self.HEADER)}
            self.assertEqual(rows[1][header_index["score"]], "")
            self.assertEqual(rows[1][header_index["score_type"]], "multi-objective")
            self.assertEqual(fake_trial.attrs["score_type"], "multi-objective")


if __name__ == "__main__":
    unittest.main()
