# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the Phase 2 dataset, task, and model-family adapters."""

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

from crest.datasets.oxiod import OxIODDataset  # noqa: E402
from crest.model_families.odom_tcn import DILATION_CANDIDATES, OdomTCNFamily  # noqa: E402
from crest.pipeline_types import DataSplit, DatasetBundle, ModelBuildContext, TargetSpec  # noqa: E402
from crest.tasks.odometry_regression import OdometryRegressionTask  # noqa: E402


def _make_legacy_split(
    *,
    num_windows: int = 3,
    window_size: int = 4,
    channels: int = 10,
) -> SimpleNamespace:
    """Synthesize the legacy split payload consumed by the Phase 2 adapters.

    Parameters
    ----------
    num_windows : int, optional
        Number of synthetic windows to expose.
    window_size : int, optional
        Number of timesteps per window.
    channels : int, optional
        Feature dimension per timestep.

    Returns
    -------
    types.SimpleNamespace
        Legacy split object with the fields the adapter layer expects from the
        old data loader contract.
    """
    # Mirror the legacy loader's field layout closely so adapter tests verify
    # translation logic rather than a custom synthetic shape.
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
        """Prepare test fixtures."""
        self.dataset = OxIODDataset()
        self.config = Dict(
            directory="data/oxiod/",
            sampling_rate_hz=100,
            window_size=200,
            stride=20,
        )

    def test_load_maps_legacy_splits_into_dataset_bundle(self) -> None:
        # The OxIOD dataset adapter should map the legacy split loader output into the new DatasetBundle shape.
        """Validate load maps legacy splits into dataset bundle."""
        train_split = _make_legacy_split()
        val_split = _make_legacy_split()
        test_split = _make_legacy_split()

        with patch(
            "crest.datasets.oxiod.import_oxiod_dataset",
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
        # The adapter should forward the legacy loader kwargs unchanged into each split load.
        """Validate train and validation calls preserve legacy loader kwargs."""
        train_split = _make_legacy_split()
        val_split = _make_legacy_split()
        test_split = _make_legacy_split()

        with patch(
            "crest.datasets.oxiod.import_oxiod_dataset",
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
        # Missing calibration windows should leave the bundle calibration split empty instead of fabricating one.
        """Validate missing calibration windows keeps bundle calibration empty."""
        with patch(
            "crest.datasets.oxiod.import_oxiod_dataset",
            side_effect=[_make_legacy_split(), _make_legacy_split(), _make_legacy_split()],
        ):
            bundle = self.dataset.load(self.config)

        self.assertIsNone(bundle.calibration)
        self.assertIs(self.dataset.make_calibration_data(bundle, self.config), bundle.train)

    def test_configured_calibration_windows_loads_capped_calibration_split(self) -> None:
        # Configured calibration windows should load the capped calibration split from the adapter.
        """Validate configured calibration windows loads capped calibration split."""
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
            "crest.datasets.oxiod.import_oxiod_dataset",
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
        """Prepare test fixtures."""
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
        """Clean up test fixtures."""
        self.tempdir.cleanup()
        tf.keras.backend.clear_session()

    def test_build_target_spec_returns_two_head_regression_contract(self) -> None:
        # The odometry task should expose the expected two-head regression target contract.
        """Validate build target spec returns two head regression contract."""
        target_spec = self.task.build_target_spec(self.bundle, {})

        self.assertEqual(target_spec.task_type, "regression")
        self.assertEqual(target_spec.output_names, ["velx", "vely"])
        self.assertEqual(target_spec.output_shapes, [(1,), (1,)])

    def test_metric_contract_returns_expected_rmse_names(self) -> None:
        # The odometry task's metric contract should keep the expected RMSE metric names.
        """Validate metric contract returns expected rmse names."""
        contract = self.task.metric_contract(self.target_spec, {})

        self.assertEqual(contract.available_metric_names, {"rmse_vel_x", "rmse_vel_y", "rmse_total"})
        self.assertEqual(contract.training_only_metric_names, {"rmse_vel_x", "rmse_vel_y", "rmse_total"})
        self.assertEqual(contract.nonnegative_metric_names, {"rmse_vel_x", "rmse_vel_y", "rmse_total"})
        self.assertEqual(contract.primary_metric_names, {"rmse_total"})

    def test_compile_model_uses_adam_and_legacy_loss_map(self) -> None:
        # Task compilation should preserve the legacy Adam optimizer and loss mapping.
        """Validate compile model uses adam and legacy loss map."""
        model = MagicMock()

        self.task.compile_model(model, {}, self.target_spec)

        model.compile.assert_called_once()
        _, kwargs = model.compile.call_args
        self.assertEqual(kwargs["loss"], {"velx": "mse", "vely": "mse"})
        self.assertIsInstance(kwargs["optimizer"], tf.keras.optimizers.Adam)

    def test_build_fit_plan_builds_expected_wiring_and_callbacks(self) -> None:
        # The fit-plan helper should wire datasets, callbacks, and checkpointing the way the task expects.
        """Validate build fit plan builds expected wiring and callbacks."""
        fit_plan = self.task.build_fit_plan(
            self.bundle,
            {},
            self.target_spec,
            mode="search",
            combine_train_val=False,
        )

        self.assertTrue(np.array_equal(fit_plan.fit_kwargs["x"], self.train_split.inputs))
        self.assertEqual(len(fit_plan.fit_kwargs["y"]), 2)
        self.assertEqual(fit_plan.monitor_metric, "val_loss")
        self.assertEqual(len(fit_plan.callbacks), 2)
        self.assertIsInstance(fit_plan.callbacks[0], tf.keras.callbacks.ModelCheckpoint)
        self.assertIsInstance(fit_plan.callbacks[1], tf.keras.callbacks.EarlyStopping)
        self.assertEqual(fit_plan.callbacks[0].filepath, str(Path(self.tempdir.name) / "best.keras"))
        self.assertEqual(fit_plan.callbacks[1].patience, 17)

    def test_build_fit_plan_requires_validation_split_when_not_combining(self) -> None:
        # The fit-plan helper should reject attempts to train without a validation split.
        """Validate build fit plan requires validation split when not combining."""
        bundle = DatasetBundle(train=self.train_split, val=None)

        with self.assertRaisesRegex(ValueError, "requires a validation split"):
            self.task.build_fit_plan(
                bundle,
                {},
                self.target_spec,
                mode="search",
                combine_train_val=False,
            )

    def test_build_fit_plan_combines_train_and_val_for_final_mode(self) -> None:
        # Final-fit plans should merge train and validation data and switch monitoring to training loss.
        """Validate build fit plan combines train and val for final mode."""
        fit_plan = self.task.build_fit_plan(
            self.bundle,
            {},
            self.target_spec,
            mode="final",
            combine_train_val=True,
        )

        self.assertEqual(fit_plan.monitor_metric, "loss")
        self.assertNotIn("validation_data", fit_plan.fit_kwargs)
        self.assertEqual(fit_plan.fit_kwargs["x"].shape[0], 6)
        self.assertEqual(fit_plan.fit_kwargs["y"][0].shape[0], 6)
        self.assertEqual(fit_plan.fit_kwargs["y"][1].shape[0], 6)

    def test_history_component_keys_return_velocity_loss_pairs(self) -> None:
        # Odometry closeout plots should be driven by task-owned history-key declarations.
        """Validate history component keys return velocity loss pairs."""
        self.assertEqual(
            self.task.history_component_keys(self.target_spec),
            [("velx_loss", "val_velx_loss"), ("vely_loss", "val_vely_loss")],
        )

    def test_generate_closeout_artifacts_restores_odometry_trajectory_outputs(self) -> None:
        # The built-in odometry task should still emit trajectory metrics and plots during closeout.
        """Validate generate closeout artifacts restores odometry trajectory outputs."""
        length = 4
        vx = np.full((length, 1), 0.5, dtype=np.float32)
        vy = np.full((length, 1), 0.5, dtype=np.float32)
        test_split = DataSplit(
            inputs=np.zeros((length, 4, 10), dtype=np.float32),
            targets={"velx": vx, "vely": vy},
            metadata={
                "size_of_each": [length],
                "x0": [0.0],
                "y0": [0.0],
            },
        )
        bundle = DatasetBundle(
            train=self.train_split,
            val=self.val_split,
            test=test_split,
            input_shape=(4, 10),
            metadata={"sampling_rate_hz": 100, "window_size": 2, "stride": 1},
        )

        class FakeModel:
            """Fake model used by artifact and prediction tests."""

            def predict(self, _inputs):
                """Run predict.

                Parameters
                ----------
                _inputs : object
                    Input tensor batch passed to the fake model.

                Returns
                -------
                object
                    Predictions emitted by the fake model.
                """
                return [vx, vy]

        artifacts = self.task.generate_closeout_artifacts(
            FakeModel(),
            bundle,
            {},
            self.target_spec,
            output_dir=Path(self.tempdir.name) / "closeout",
        )

        self.assertAlmostEqual(artifacts["ate_mean"], 0.0)
        self.assertTrue(Path(artifacts["trajectory_metrics_path"]).is_file())
        self.assertEqual(len(artifacts["plots"]), 1)
        self.assertTrue(Path(artifacts["plots"][0]).is_file())

    def test_evaluate_uses_legacy_prediction_ordering(self) -> None:
        # Task evaluation should preserve the legacy prediction ordering used by downstream odometry metrics.
        """Validate evaluate uses legacy prediction ordering."""
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
        """Run suggest int.

        Parameters
        ----------
        name : object
            Parameter name requested by the fake sampler.
        low : object
            Lower bound used by the fake sampler.
        high : object
            Upper bound used by the fake sampler.

        Returns
        -------
        object
            Selected integer value from the fake sampler.

        Raises
        ------
        AssertionError
            If existing validation or execution checks fail.
        """
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
        """Run suggest categorical.

        Parameters
        ----------
        name : object
            Parameter name requested by the fake sampler.
        choices : object
            Candidate values available to the fake sampler.

        Returns
        -------
        object
            Selected categorical value from the fake sampler.

        Raises
        ------
        AssertionError
            If existing validation or execution checks fail.
        """
        if name == "dropout_rate":
            return choices[1]
        if name == "use_skip_connections":
            return True
        if name == "norm_flag":
            return False
        raise AssertionError(f"Unexpected suggest_categorical call: {name}, {choices}")


class OdomTCNFamilyTests(unittest.TestCase):
    """Validate the Phase 2 Odom TCN model family adapter."""

    def setUp(self) -> None:
        """Prepare test fixtures."""
        self.family = OdomTCNFamily()
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
        """Clean up test fixtures."""
        tf.keras.backend.clear_session()

    def test_sample_hparams_matches_legacy_search_surface(self) -> None:
        # The model family should keep exposing the legacy hyperparameter search surface.
        """Validate sample hparams matches legacy search surface."""
        hparams = self.family.sample_hparams(_DummyTrial(), self.ctx, {})

        self.assertEqual(
            set(hparams),
            {"nb_filters", "kernel_size", "dropout_rate", "use_skip_connections", "norm_flag", "dilations"},
        )
        self.assertEqual(hparams["dilations"], DILATION_CANDIDATES[2])

    def test_dilation_candidates_preserve_legacy_search_surface(self) -> None:
        # The family-owned dilation table should preserve the original CREST search surface exactly.
        """Validate dilation candidates preserve legacy search surface."""
        self.assertEqual(len(DILATION_CANDIDATES), 465)
        self.assertEqual(DILATION_CANDIDATES[0], [1, 2, 4])
        self.assertEqual(DILATION_CANDIDATES[107], [1, 4, 8, 64])

    def test_build_model_passes_plain_hparams_into_local_builder(self) -> None:
        # Model construction should use the family-owned builder with plain hyperparameters.
        """Validate build model passes plain hparams into local builder."""
        hparams = {
            "nb_filters": 8,
            "kernel_size": 5,
            "dropout_rate": 0.1,
            "use_skip_connections": True,
            "norm_flag": False,
            "dilations": [1, 2, 4],
        }

        with patch(
            "crest.model_families.odom_tcn.build_odom_tcn_model",
            return_value=MagicMock(),
        ) as build_mock:
            self.family.build_model(hparams, self.ctx, {})

        build_mock.assert_called_once()
        build_hparams = build_mock.call_args.args[0]
        self.assertIsInstance(build_hparams, dict)
        self.assertEqual(build_hparams["timesteps"], 20)
        self.assertEqual(build_hparams["input_dim"], 6)

    def test_build_model_preserves_legacy_output_names(self) -> None:
        # Model construction should preserve the legacy output names expected by the original CREST heads.
        """Validate build model preserves legacy output names."""
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
        # The TCN family should keep returning the expected custom-objects mapping.
        """Validate custom objects returns tcn mapping."""
        self.assertIn("TCN", self.family.custom_objects())

    def test_decode_trial_hparams_expands_dilations_index(self) -> None:
        # Persisted trial params should be decoded by the family rather than by NAS orchestration.
        """Validate decode trial hparams expands dilations index."""
        decoded = self.family.decode_trial_hparams(
            {
                "nb_filters": 8,
                "kernel_size": 5,
                "dropout_rate": 0.1,
                "use_skip_connections": True,
                "norm_flag": False,
                "dilations_index": 2,
            },
            self.ctx,
            {},
        )

        self.assertNotIn("dilations_index", decoded)
        self.assertEqual(decoded["dilations"], DILATION_CANDIDATES[2])

    def test_default_seed_trial_matches_raw_trial_surface(self) -> None:
        # The family should own the default persisted trial seed for new studies.
        """Validate default seed trial matches raw trial surface."""
        seed = self.family.default_seed_trial(self.ctx, {})

        self.assertIsNotNone(seed)
        self.assertEqual(seed["dilations_index"], 107)
        self.assertEqual(
            self.family.decode_trial_hparams(seed, self.ctx, {})["dilations"],
            [1, 4, 8, 64],
        )

    def test_count_flops_returns_positive_estimate(self) -> None:
        # The family should expose a working FLOP counter through the model-family contract.
        """Validate count flops returns positive estimate."""
        hparams = {
            "nb_filters": 8,
            "kernel_size": 5,
            "dropout_rate": 0.1,
            "use_skip_connections": True,
            "norm_flag": False,
            "dilations": [1, 2, 4],
        }
        model = self.family.build_model(hparams, self.ctx, {})

        flops = self.family.count_flops(model, self.ctx, {})

        self.assertIsInstance(flops, int)
        self.assertGreater(flops, 0)

    def test_estimate_static_memory_returns_positive_tcn_proxy(self) -> None:
        # The OdomTCN override should count real weights and infer custom TCN internal activation traffic.
        """Validate estimate static memory returns positive tcn proxy."""
        hparams = {
            "nb_filters": 8,
            "kernel_size": 5,
            "dropout_rate": 0.1,
            "use_skip_connections": True,
            "norm_flag": False,
            "dilations": [1, 2, 4],
        }
        model = self.family.build_model(hparams, self.ctx, {})

        estimate = self.family.estimate_static_memory(
            model,
            self.ctx,
            {},
            quantization_mode="int8_ptq",
        )

        self.assertEqual(estimate.dtype_bytes, 1)
        self.assertGreater(estimate.weight_bytes, 0)
        self.assertGreater(estimate.activation_bytes, 0)
        self.assertEqual(
            estimate.memory_traffic_bytes,
            estimate.weight_bytes + estimate.activation_bytes,
        )
        self.assertGreater(estimate.warning_count, 0)

    def test_validate_hparams_rejects_missing_required_keys(self) -> None:
        # Hyperparameter validation should reject missing required keys before model construction starts.
        """Validate validate hparams rejects missing required keys."""
        hparams = {
            "nb_filters": 8,
            "kernel_size": 5,
            "dropout_rate": 0.1,
            "use_skip_connections": True,
            "norm_flag": False,
        }

        with self.assertRaisesRegex(ValueError, "requires hyperparameters"):
            self.family.validate_hparams(hparams, self.ctx, {})
