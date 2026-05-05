"""Tests for the sound classification task adapter."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import tensorflow as tf
from addict import Dict

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tinyodom.datasets.urbansound8k_common import CLASS_NAMES, LABEL_ENCODING  # noqa: E402
from tinyodom.pipeline_types import DataSplit, DatasetBundle  # noqa: E402
from tinyodom.runtime_bootstrap import instantiate_task_component  # noqa: E402
from tinyodom.tasks.odometry_regression import OdometryRegressionTask  # noqa: E402
from tinyodom.tasks.sound_classification import SoundClassificationTask  # noqa: E402


class PredictOnlyModel:
    """Minimal model double exposing deterministic logits."""

    def __init__(self, logits: np.ndarray) -> None:
        """Store logits returned by `predict`.

        Parameters
        ----------
        logits : numpy.ndarray
            Logits returned from prediction calls.
        """

        self.logits = np.asarray(logits, dtype=np.float32)
        self.predict_calls = 0

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Return configured logits while recording call count.

        Parameters
        ----------
        inputs : numpy.ndarray
            Input batch. Only the row count is checked by tests.

        Returns
        -------
        numpy.ndarray
            Configured logits.
        """

        del inputs
        self.predict_calls += 1
        return self.logits


def _task(tmp_path: Path) -> SoundClassificationTask:
    """Build one task instance for tests.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory used for checkpoint paths.

    Returns
    -------
    SoundClassificationTask
        Configured task instance.
    """

    return SoundClassificationTask(checkpoint_path=tmp_path / "checkpoint.keras")


def _bundle(
    num_classes: int = len(CLASS_NAMES),
    *,
    include_test: bool = True,
    class_names: list[str] | None = None,
    label_encoding: str = LABEL_ENCODING,
) -> DatasetBundle:
    """Build a small classification dataset bundle.

    Parameters
    ----------
    num_classes : int, optional
        Number of classes represented in metadata.
    include_test : bool, optional
        Whether to attach a test split.
    class_names : list[str] | None, optional
        Optional class-name metadata override.
    label_encoding : str, optional
        Optional label-encoding metadata override.

    Returns
    -------
    DatasetBundle
        Bundle with flat integer class-index targets.
    """

    inputs = np.zeros((3, 201, 64), dtype=np.float32)
    targets = np.asarray([0, 1, 1], dtype=np.int64)
    split = DataSplit(inputs=inputs, targets=targets)
    return DatasetBundle(
        train=split,
        val=split,
        test=split if include_test else None,
        input_shape=(201, 64),
        input_dtype="float32",
        metadata={
            "num_classes": num_classes,
            "class_names": list(CLASS_NAMES) if class_names is None else class_names,
            "label_encoding": label_encoding,
        },
    )


class SoundClassificationTaskTests(unittest.TestCase):
    """Validate the audio classification task contract."""

    def tearDown(self) -> None:
        """Clear TensorFlow state after each test.

        Returns
        -------
        None
            Releases Keras graph state accumulated by model tests.
        """

        tf.keras.backend.clear_session()

    def test_target_spec_and_metric_contract(self) -> None:
        """Target and metric contracts should match the audio plan.

        Returns
        -------
        None
            Asserts logits target metadata and task metric names.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            task = _task(Path(tmpdir))
            target_spec = task.build_target_spec(_bundle(), Dict())
            contract = task.metric_contract(target_spec, Dict())

        self.assertEqual(task.name, "sound_classification")
        self.assertEqual(target_spec.output_names, ["class_logits"])
        self.assertEqual(target_spec.output_shapes, [(10,)])
        self.assertTrue(target_spec.metadata["from_logits"])
        self.assertEqual(contract.available_metric_names, {"loss", "accuracy", "macro_f1"})
        self.assertEqual(contract.training_only_metric_names, set())

    def test_target_spec_rejects_metadata_class_drift(self) -> None:
        """Target construction should reject non-UrbanSound8K metadata drift.

        Returns
        -------
        None
            Asserts class count, class names, and label encoding are fixed.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            task = _task(Path(tmpdir))
            with self.assertRaisesRegex(ValueError, "num_classes"):
                task.build_target_spec(_bundle(num_classes=9), Dict())
            with self.assertRaisesRegex(ValueError, "class_names"):
                task.build_target_spec(_bundle(class_names=list(reversed(CLASS_NAMES))), Dict())
            with self.assertRaisesRegex(ValueError, "class_index"):
                task.build_target_spec(_bundle(label_encoding="one_hot"), Dict())

    def test_compile_model_uses_logits_loss_and_adam(self) -> None:
        """Compile should use Adam and sparse CE from logits.

        Returns
        -------
        None
            Asserts compile-time optimizer, loss, and metric choices.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            task = _task(Path(tmpdir))
            target_spec = task.build_target_spec(_bundle(), Dict())
            model = tf.keras.Sequential(
                [
                    tf.keras.layers.Input(shape=(201, 64)),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(10),
                ]
            )
            task.compile_model(model, Dict(), target_spec)

        self.assertIsInstance(model.optimizer, tf.keras.optimizers.Adam)
        self.assertAlmostEqual(float(model.optimizer.learning_rate.numpy()), 0.001)
        self.assertTrue(model.loss.from_logits)
        self.assertEqual(model.metrics[1].name, "compile_metrics")

    def test_fit_plan_uses_flat_integer_targets(self) -> None:
        """Fit plans should pass flat class-index targets directly.

        Returns
        -------
        None
            Asserts search and final fit wiring use array targets.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            task = _task(Path(tmpdir))
            bundle = _bundle()
            target_spec = task.build_target_spec(bundle, Dict())
            search_plan = task.build_fit_plan(
                bundle,
                Dict(),
                target_spec,
                mode="search",
                combine_train_val=False,
            )
            final_plan = task.build_fit_plan(
                bundle,
                Dict(),
                target_spec,
                mode="final",
                combine_train_val=True,
            )

        self.assertIs(search_plan.fit_kwargs["y"], bundle.train.targets)
        self.assertIn("validation_data", search_plan.fit_kwargs)
        self.assertEqual(search_plan.monitor_metric, "val_loss")
        self.assertEqual(final_plan.fit_kwargs["y"].shape[0], 6)
        self.assertNotIn("validation_data", final_plan.fit_kwargs)
        self.assertEqual(final_plan.monitor_metric, "loss")

    def test_validate_model_outputs_accepts_logits_and_rejects_bad_shapes(self) -> None:
        """Output validation should require one logits tensor.

        Returns
        -------
        None
            Asserts logits pass while bad class counts and multiple outputs
            fail.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            task = _task(Path(tmpdir))
            target_spec = task.build_target_spec(_bundle(), Dict())
            good_model = tf.keras.Sequential(
                [
                    tf.keras.layers.Input(shape=(201, 64)),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(10),
                ]
            )
            bad_shape = tf.keras.Sequential(
                [
                    tf.keras.layers.Input(shape=(201, 64)),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(9),
                ]
            )
            inputs = tf.keras.Input(shape=(201, 64))
            flat = tf.keras.layers.Flatten()(inputs)
            multi_output = tf.keras.Model(
                inputs=inputs,
                outputs=[tf.keras.layers.Dense(10)(flat), tf.keras.layers.Dense(10)(flat)],
            )

            task.validate_model_outputs(good_model, target_spec)
            with self.assertRaisesRegex(ValueError, "shape"):
                task.validate_model_outputs(bad_shape, target_spec)
            with self.assertRaisesRegex(ValueError, "shape"):
                task.validate_model_outputs(SimpleNamespace(output_shape=(1, 10), layers=[]), target_spec)
            with self.assertRaisesRegex(ValueError, "one model output"):
                task.validate_model_outputs(multi_output, target_spec)

    def test_validate_model_outputs_rejects_softmax_probabilities(self) -> None:
        """Output validation should reject structural softmax probabilities.

        Returns
        -------
        None
            Asserts final Dense softmax and final Softmax layer are rejected.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            task = _task(Path(tmpdir))
            target_spec = task.build_target_spec(_bundle(), Dict())
            dense_softmax = tf.keras.Sequential(
                [
                    tf.keras.layers.Input(shape=(201, 64)),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(10, activation="softmax"),
                ]
            )
            softmax_layer = tf.keras.Sequential(
                [
                    tf.keras.layers.Input(shape=(201, 64)),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(10),
                    tf.keras.layers.Softmax(),
                ]
            )

            with self.assertRaisesRegex(ValueError, "logits"):
                task.validate_model_outputs(dense_softmax, target_spec)
            with self.assertRaisesRegex(ValueError, "logits"):
                task.validate_model_outputs(softmax_layer, target_spec)

    def test_evaluate_computes_json_safe_metrics_in_one_predict_call(self) -> None:
        """Evaluation should compute metrics manually from one logits pass.

        Returns
        -------
        None
            Asserts predictions, macro-F1 zero-support behavior, and JSON
            serializability.
        """

        logits = np.full((3, 10), -4.0, dtype=np.float32)
        logits[0, 0] = 4.0
        logits[1, 2] = 4.0
        logits[2, 1] = 4.0
        model = PredictOnlyModel(logits)
        with tempfile.TemporaryDirectory() as tmpdir:
            task = _task(Path(tmpdir))
            bundle = _bundle()
            target_spec = task.build_target_spec(bundle, Dict())
            result = task.evaluate(model, bundle.val, Dict(), target_spec)
            prediction_result = task.evaluate_predictions(logits, bundle.val, Dict(), target_spec)

        self.assertEqual(model.predict_calls, 1)
        self.assertEqual(result.predictions, [0, 2, 1])
        self.assertEqual(prediction_result.predictions, result.predictions)
        self.assertEqual(prediction_result.metrics, result.metrics)
        self.assertAlmostEqual(result.metrics["accuracy"], 2.0 / 3.0)
        self.assertAlmostEqual(result.metrics["macro_f1"], (1.0 + (2.0 / 3.0)) / 10.0)
        json.dumps(result.metrics)
        json.dumps(result.artifacts)
        json.dumps(result.predictions)

    def test_generate_closeout_artifacts_writes_json_for_test_split(self) -> None:
        """Closeout should write a compact JSON-safe test summary.

        Returns
        -------
        None
            Asserts closeout artifact path and JSON payload.
        """

        logits = np.full((3, 10), -4.0, dtype=np.float32)
        logits[:, 0] = 4.0
        model = PredictOnlyModel(logits)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            task = _task(tmp_path)
            bundle = _bundle()
            target_spec = task.build_target_spec(bundle, Dict())
            closeout = task.generate_closeout_artifacts(
                model,
                bundle,
                Dict(),
                target_spec,
                output_dir=tmp_path / "closeout",
            )
            payload = json.loads(Path(closeout["sound_classification_metrics_path"]).read_text())

        self.assertIn("confusion_matrix", closeout)
        self.assertIn("metrics", payload)
        json.dumps(closeout)

    def test_generate_closeout_artifacts_returns_empty_without_test_split(self) -> None:
        """Closeout should be a no-op when no test split exists.

        Returns
        -------
        None
            Asserts no file is written when `bundle.test` is absent.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            task = _task(tmp_path)
            bundle = _bundle(include_test=False)
            target_spec = task.build_target_spec(bundle, Dict())
            closeout = task.generate_closeout_artifacts(
                PredictOnlyModel(np.zeros((3, 10), dtype=np.float32)),
                bundle,
                Dict(),
                target_spec,
                output_dir=tmp_path / "closeout",
            )

            self.assertEqual(closeout, {})
            self.assertFalse((tmp_path / "closeout" / "sound_classification_metrics.json").exists())

    def test_bootstrap_rejects_non_integer_patience_values(self) -> None:
        """Task bootstrap should reject lossy patience coercion.

        Returns
        -------
        None
            Asserts bools, floats, and strings fail before task construction.
        """

        config = SimpleNamespace(outputs=SimpleNamespace(checkpoint_path=Path("checkpoint.keras")))
        invalid_values = [True, 3.5, "3"]
        for value in invalid_values:
            with self.subTest(value=value), patch(
                "tinyodom.runtime_bootstrap.task_registry.get",
                return_value=SoundClassificationTask,
            ):
                with self.assertRaisesRegex(ValueError, "early_stopping_patience"):
                    instantiate_task_component(
                        "sound_classification",
                        config,
                        Dict(early_stopping_patience=value),
                    )

    def test_bootstrap_accepts_integer_patience_for_odometry(self) -> None:
        """Task bootstrap should preserve integer patience for odometry.

        Returns
        -------
        None
            Asserts the existing odometry task still constructs with integer
            patience.
        """

        config = SimpleNamespace(outputs=SimpleNamespace(checkpoint_path=Path("checkpoint.keras")))
        with patch(
            "tinyodom.runtime_bootstrap.task_registry.get",
            return_value=OdometryRegressionTask,
        ):
            task = instantiate_task_component(
                "odometry_regression",
                config,
                Dict(early_stopping_patience=40),
            )

        self.assertEqual(task.early_stopping_patience, 40)


if __name__ == "__main__":
    unittest.main()
