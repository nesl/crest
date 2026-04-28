import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import tensorflow as tf
from addict import Dict

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tinyodom.datasets.oxiod import OxIODDataset  # noqa: E402
from tinyodom.model import DILATION_CANDIDATES  # noqa: E402
from tinyodom.model_families.tinyodom_tcn import TinyOdomTCNFamily  # noqa: E402
from tinyodom.pipeline_types import DataSplit, DatasetBundle, ModelBuildContext, TargetSpec  # noqa: E402
from tinyodom.tasks.odometry_regression import OdometryRegressionTask  # noqa: E402


def _make_legacy_split(
    *,
    num_windows: int = 3,
    window_size: int = 4,
    channels: int = 10,
) -> SimpleNamespace:
    inputs = np.arange(num_windows * window_size * channels, dtype=np.float32).reshape(
        num_windows, window_size, channels
    )
    return SimpleNamespace(
        inputs=inputs,
        disp=np.arange(num_windows, dtype=np.float32),
        heading=np.arange(num_windows, dtype=np.float32) + 0.5,
        position=np.zeros((num_windows, window_size, 2), dtype=np.float32),
        x0=[0.0],
        y0=[1.0],
        size_of_each=[num_windows],
        x_vel=np.arange(num_windows, dtype=np.float32),
        y_vel=np.arange(num_windows, dtype=np.float32) + 10.0,
        head_s=np.ones(num_windows, dtype=np.float32),
        head_c=np.zeros(num_windows, dtype=np.float32),
        inputs_orig=np.zeros((num_windows * window_size, channels), dtype=np.float32),
    )


class OxIODDatasetTests(unittest.TestCase):
    """Validate the Phase 2 OxIOD dataset adapter."""

    def setUp(self) -> None:
        self.dataset = OxIODDataset()
        self.config = Dict(
            directory="data/oxiod/",
            sampling_rate_hz=100,
            window_size=200,
            stride=20,
        )

    def test_load_maps_legacy_splits_into_dataset_bundle(self) -> None:
        train_split = _make_legacy_split()
        val_split = _make_legacy_split()
        test_split = _make_legacy_split()

        with patch(
            "tinyodom.datasets.oxiod.import_oxiod_dataset",
            side_effect=[train_split, val_split, test_split],
        ) as load_mock:
            bundle = self.dataset.load(self.config)

        self.assertEqual(load_mock.call_count, 3)
        self.assertEqual(bundle.train.targets["velx"].tolist(), train_split.x_vel.tolist())
        self.assertEqual(bundle.train.targets["vely"].tolist(), train_split.y_vel.tolist())
        self.assertIn("position", bundle.train.metadata)
        self.assertEqual(bundle.input_shape, train_split.inputs.shape[1:])
        self.assertEqual(bundle.input_dtype, str(train_split.inputs.dtype))
        self.assertEqual(bundle.metadata["input_dim"], train_split.inputs.shape[2])
        self.assertIsNone(bundle.calibration)

    def test_train_and_validation_calls_preserve_legacy_loader_kwargs(self) -> None:
        train_split = _make_legacy_split()
        val_split = _make_legacy_split()
        test_split = _make_legacy_split()

        with patch(
            "tinyodom.datasets.oxiod.import_oxiod_dataset",
            side_effect=[train_split, val_split, test_split],
        ) as load_mock:
            self.dataset.load(self.config)

        train_call = load_mock.call_args_list[0]
        val_call = load_mock.call_args_list[1]
        test_call = load_mock.call_args_list[2]

        for call in (train_call, val_call):
            self.assertTrue(call.kwargs["useMagnetometer"])
            self.assertTrue(call.kwargs["useStepCounter"])
            self.assertEqual(call.kwargs["AugmentationCopies"], 0)
            self.assertEqual(
                call.kwargs["sub_folders"],
                ["handbag/", "handheld/", "pocket/", "running/", "slow_walking/", "trolley/"],
            )

        self.assertNotIn("useMagnetometer", test_call.kwargs)
        self.assertNotIn("useStepCounter", test_call.kwargs)
        self.assertNotIn("AugmentationCopies", test_call.kwargs)

    def test_missing_calibration_windows_keeps_bundle_calibration_empty(self) -> None:
        with patch(
            "tinyodom.datasets.oxiod.import_oxiod_dataset",
            side_effect=[_make_legacy_split(), _make_legacy_split(), _make_legacy_split()],
        ):
            bundle = self.dataset.load(self.config)

        self.assertIsNone(bundle.calibration)
        self.assertIs(self.dataset.make_calibration_data(bundle, self.config), bundle.train)

    def test_configured_calibration_windows_loads_capped_calibration_split(self) -> None:
        config = Dict(
            directory="data/oxiod/",
            sampling_rate_hz=100,
            window_size=200,
            stride=20,
            calibration_windows=7,
        )
        train_split = _make_legacy_split()
        val_split = _make_legacy_split()
        test_split = _make_legacy_split()
        calibration_split = _make_legacy_split(num_windows=2)

        with patch(
            "tinyodom.datasets.oxiod.import_oxiod_dataset",
            side_effect=[train_split, val_split, test_split, calibration_split],
        ) as load_mock:
            bundle = self.dataset.load(config)

        self.assertIsNotNone(bundle.calibration)
        self.assertEqual(load_mock.call_args_list[3].kwargs["max_windows"], 7)
        self.assertEqual(bundle.calibration.targets["velx"].tolist(), calibration_split.x_vel.tolist())
        self.assertIs(self.dataset.make_calibration_data(bundle, config), bundle.calibration)


class OdometryRegressionTaskTests(unittest.TestCase):
    """Validate the Phase 2 odometry regression task adapter."""

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.task = OdometryRegressionTask(
            checkpoint_path=Path(self.tempdir.name) / "best.keras",
            early_stopping_patience=17,
        )
        self.train_split = DataSplit(
            inputs=np.zeros((3, 4, 10), dtype=np.float32),
            targets={
                "velx": np.array([0.0, 1.0, 2.0], dtype=np.float32),
                "vely": np.array([10.0, 11.0, 12.0], dtype=np.float32),
            },
        )
        self.val_split = DataSplit(
            inputs=np.ones((3, 4, 10), dtype=np.float32),
            targets={
                "velx": np.array([0.0, 1.0, 2.0], dtype=np.float32),
                "vely": np.array([10.0, 11.0, 12.0], dtype=np.float32),
            },
        )
        self.bundle = DatasetBundle(train=self.train_split, val=self.val_split, input_shape=(4, 10))
        self.target_spec = TargetSpec(
            task_type="regression",
            output_names=["velx", "vely"],
            output_shapes=[(1,), (1,)],
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()
        tf.keras.backend.clear_session()

    def test_build_target_spec_returns_two_head_regression_contract(self) -> None:
        target_spec = self.task.build_target_spec(self.bundle, {})

        self.assertEqual(target_spec.task_type, "regression")
        self.assertEqual(target_spec.output_names, ["velx", "vely"])
        self.assertEqual(target_spec.output_shapes, [(1,), (1,)])

    def test_metric_contract_returns_expected_rmse_names(self) -> None:
        contract = self.task.metric_contract(self.target_spec, {})

        self.assertEqual(contract.available_metric_names, {"rmse_vel_x", "rmse_vel_y", "rmse_total"})
        self.assertEqual(contract.training_only_metric_names, {"rmse_vel_x", "rmse_vel_y", "rmse_total"})
        self.assertEqual(contract.nonnegative_metric_names, {"rmse_vel_x", "rmse_vel_y", "rmse_total"})
        self.assertEqual(contract.primary_metric_names, {"rmse_total"})

    def test_compile_model_uses_adam_and_legacy_loss_map(self) -> None:
        model = MagicMock()

        self.task.compile_model(model, {}, self.target_spec)

        model.compile.assert_called_once()
        _, kwargs = model.compile.call_args
        self.assertEqual(kwargs["loss"], {"velx": "mse", "vely": "mse"})
        self.assertIsInstance(kwargs["optimizer"], tf.keras.optimizers.Adam)

    def test_make_fit_plan_builds_expected_wiring_and_callbacks(self) -> None:
        fit_plan = self.task.make_fit_plan(self.bundle, {}, self.target_spec)

        self.assertTrue(np.array_equal(fit_plan.fit_kwargs["x"], self.train_split.inputs))
        self.assertEqual(len(fit_plan.fit_kwargs["y"]), 2)
        self.assertEqual(fit_plan.monitor_metric, "val_loss")
        self.assertEqual(len(fit_plan.callbacks), 2)
        self.assertIsInstance(fit_plan.callbacks[0], tf.keras.callbacks.ModelCheckpoint)
        self.assertIsInstance(fit_plan.callbacks[1], tf.keras.callbacks.EarlyStopping)
        self.assertEqual(fit_plan.callbacks[0].filepath, str(Path(self.tempdir.name) / "best.keras"))
        self.assertEqual(fit_plan.callbacks[1].patience, 17)

    def test_make_fit_plan_requires_validation_split(self) -> None:
        bundle = DatasetBundle(train=self.train_split, val=None)

        with self.assertRaisesRegex(ValueError, "requires a validation split"):
            self.task.make_fit_plan(bundle, {}, self.target_spec)

    def test_evaluate_uses_legacy_prediction_ordering(self) -> None:
        model = MagicMock()
        predictions = [
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([12.0, 13.0, 14.0], dtype=np.float32),
        ]
        model.predict.return_value = predictions

        result = self.task.evaluate(model, self.val_split, {}, self.target_spec)

        model.predict.assert_called_once_with(self.val_split.inputs)
        self.assertAlmostEqual(result.metrics["rmse_vel_x"], 1.0)
        self.assertAlmostEqual(result.metrics["rmse_vel_y"], 2.0)
        self.assertAlmostEqual(result.metrics["rmse_total"], 3.0)
        self.assertIs(result.predictions, predictions)


class _DummyTrial:
    """Small Optuna-like trial stub for model-family tests."""

    def suggest_int(self, name, low, high):
        if name == "nb_filters":
            assert (low, high) == (2, 63)
            return 8
        if name == "kernel_size":
            assert (low, high) == (2, 15)
            return 5
        if name == "dilations_index":
            assert (low, high) == (0, len(DILATION_CANDIDATES) - 1)
            return 2
        raise AssertionError(f"Unexpected suggest_int call: {name}, {low}, {high}")

    def suggest_categorical(self, name, choices):
        if name == "dropout_rate":
            return choices[1]
        if name == "use_skip_connections":
            return True
        if name == "norm_flag":
            return False
        raise AssertionError(f"Unexpected suggest_categorical call: {name}, {choices}")


class TinyOdomTCNFamilyTests(unittest.TestCase):
    """Validate the Phase 2 TinyODOM TCN model family adapter."""

    def setUp(self) -> None:
        self.family = TinyOdomTCNFamily()
        self.ctx = ModelBuildContext(
            input_shape=(20, 6),
            input_dtype="float32",
            target_spec=TargetSpec(
                task_type="regression",
                output_names=["velx", "vely"],
                output_shapes=[(1,), (1,)],
            ),
        )

    def tearDown(self) -> None:
        tf.keras.backend.clear_session()

    def test_sample_hparams_matches_legacy_search_surface(self) -> None:
        hparams = self.family.sample_hparams(_DummyTrial(), self.ctx, {})

        self.assertEqual(
            set(hparams),
            {"nb_filters", "kernel_size", "dropout_rate", "use_skip_connections", "norm_flag", "dilations"},
        )
        self.assertEqual(hparams["dilations"], DILATION_CANDIDATES[2])

    def test_build_model_wraps_legacy_payload_with_addict_dict(self) -> None:
        hparams = {
            "nb_filters": 8,
            "kernel_size": 5,
            "dropout_rate": 0.1,
            "use_skip_connections": True,
            "norm_flag": False,
            "dilations": [1, 2, 4],
        }

        with patch(
            "tinyodom.model_families.tinyodom_tcn.build_tinyodom_model",
            return_value=MagicMock(),
        ) as build_mock:
            self.family.build_model(hparams, self.ctx, {})

        build_mock.assert_called_once()
        legacy_payload = build_mock.call_args.args[0]
        self.assertIsInstance(legacy_payload, Dict)
        self.assertEqual(legacy_payload.timesteps, 20)
        self.assertEqual(legacy_payload.input_dim, 6)

    def test_build_model_preserves_legacy_output_names(self) -> None:
        hparams = {
            "nb_filters": 8,
            "kernel_size": 5,
            "dropout_rate": 0.1,
            "use_skip_connections": True,
            "norm_flag": False,
            "dilations": [1, 2, 4],
        }

        model = self.family.build_model(hparams, self.ctx, {})

        self.assertEqual(model.output_names, ["velx", "vely"])

    def test_custom_objects_returns_tcn_mapping(self) -> None:
        self.assertIn("TCN", self.family.custom_objects())

    def test_validate_hparams_rejects_missing_required_keys(self) -> None:
        hparams = {
            "nb_filters": 8,
            "kernel_size": 5,
            "dropout_rate": 0.1,
            "use_skip_connections": True,
            "norm_flag": False,
        }

        with self.assertRaisesRegex(ValueError, "requires hyperparameters"):
            self.family.validate_hparams(hparams, self.ctx, {})
