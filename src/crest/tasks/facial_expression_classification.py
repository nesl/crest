# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Five-class FER+/RAF-DB probability-classification task."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np
import tensorflow as tf

from ..datasets.ferplus_rafdb_mixed import CLASS_NAMES, LABEL_SOURCE, SOURCES, equal_source_weights
from ..component_selection import RUNTIME_TRAINING_BATCH_SIZE_KEY, cfg_get as _cfg_get
from ..interfaces import TaskABC
from ..pipeline_types import DataSplit, DatasetBundle, EvaluationResult, FitPlan, TargetSpec, TaskMetricContract

COMPONENT_NAME = "facial_expression_classification"
PROBABILITY_SUM_TOLERANCE = 5.0 / 256.0


def _config_keys(config: Any) -> set[str]:
    """Return explicit keys from a mapping-like config object."""
    if isinstance(config, Mapping):
        return set(config)
    items = getattr(config, "items", None)
    if callable(items):
        return {str(key) for key, _ in items()}
    return set(vars(config)) if hasattr(config, "__dict__") else set()


def _positive_integer(value: Any, *, field_name: str) -> int:
    """Validate a non-boolean positive integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return int(value)


def _positive_number(value: Any, *, field_name: str) -> float:
    """Validate a finite positive numeric value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a positive finite number.")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field_name} must be a positive finite number.")
    return result


def build_augmentation(seed: int) -> tf.keras.Sequential:
    """Build the reference pixel-space augmentation pipeline."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.Rescaling(127.5, offset=127.5),
            tf.keras.layers.RandomFlip("horizontal", seed=seed),
            tf.keras.layers.RandomRotation(0.028, fill_mode="nearest", seed=seed + 1),
            tf.keras.layers.RandomTranslation(0.08, 0.08, fill_mode="nearest", seed=seed + 2),
            tf.keras.layers.RandomContrast(0.15, seed=seed + 3),
            tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0),
        ],
        name="pixel_space_augmentation",
    )


def _normalize_probabilities(predictions: Any, *, expected_rows: int) -> np.ndarray:
    """Validate dequantized probability output and normalize it for CCE."""
    if isinstance(predictions, (list, tuple)):
        if len(predictions) != 1:
            raise ValueError("Facial-expression evaluation expects one probability output.")
        predictions = predictions[0]
    probabilities = np.asarray(predictions, dtype=np.float32)
    if probabilities.shape != (expected_rows, len(CLASS_NAMES)):
        raise ValueError("Facial-expression probabilities must have shape (N, 5).")
    if not np.isfinite(probabilities).all():
        raise ValueError("Facial-expression probabilities must be finite.")
    quantum = 1.0 / 256.0
    if np.any(probabilities < -quantum) or np.any(probabilities > 1.0 + quantum):
        raise ValueError("Facial-expression probabilities are outside the quantized probability range.")
    clipped = np.clip(probabilities, 0.0, 1.0)
    sums = clipped.sum(axis=1, keepdims=True)
    if np.any(np.abs(sums - 1.0) > PROBABILITY_SUM_TOLERANCE):
        raise ValueError("Facial-expression probability rows do not have a valid quantized sum.")
    if np.any(sums <= 0.0):
        raise ValueError("Facial-expression probability rows must have positive mass.")
    return clipped / sums


def _classification_report(labels: np.ndarray, probabilities: np.ndarray) -> tuple[dict[str, float], dict[str, Any]]:
    """Compute prediction CCE, accuracy, macro-F1, confusion, and per-class values."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    predicted = np.argmax(probabilities, axis=1).astype(np.int64)
    matrix = np.zeros((5, 5), dtype=np.int64)
    np.add.at(matrix, (labels, predicted), 1)
    true_positive = np.diag(matrix).astype(np.float64)
    predicted_count = matrix.sum(axis=0)
    support = matrix.sum(axis=1)
    precision = np.divide(true_positive, predicted_count, out=np.zeros(5), where=predicted_count != 0)
    recall = np.divide(true_positive, support, out=np.zeros(5), where=support != 0)
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(5),
        where=(precision + recall) != 0,
    )
    epsilon = np.finfo(np.float32).eps
    loss = float(-np.log(np.clip(probabilities[np.arange(len(labels)), labels], epsilon, 1.0)).mean())
    metrics = {
        "loss": loss,
        "accuracy": float(np.mean(predicted == labels)),
        "macro_f1": float(np.mean(f1)),
    }
    artifacts = {
        "loss_semantics": "prediction categorical cross-entropy; excludes model regularization",
        "class_names": list(CLASS_NAMES),
        "confusion_matrix_true_rows_predicted_columns": matrix.tolist(),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(CLASS_NAMES)
        },
    }
    return metrics, artifacts


def _source_reports(labels: np.ndarray, probabilities: np.ndarray, sources: np.ndarray) -> dict[str, Any]:
    """Compute diagnostic reports for FER+ and RAF-DB separately."""
    reports: dict[str, Any] = {}
    for source in SOURCES:
        selected = np.asarray(sources) == source
        if not np.any(selected):
            reports[source] = None
            continue
        metrics, artifacts = _classification_report(labels[selected], probabilities[selected])
        reports[source] = {**metrics, **artifacts}
    available = [reports[source]["macro_f1"] for source in SOURCES if reports[source] is not None]
    return {
        "sources": reports,
        "equal_source_mean_macro_f1": float(np.mean(available)) if len(available) == len(SOURCES) else None,
    }


class _ValidationMetrics(tf.keras.callbacks.Callback):
    """Add pooled and source validation diagnostics to epoch logs."""

    def __init__(self, split: DataSplit, batch_size: int) -> None:
        super().__init__()
        self.inputs = np.asarray(split.inputs)
        self.labels = np.asarray(split.targets)
        self.sources = np.asarray(split.metadata["sources"])
        self.batch_size = batch_size
        self.best_macro_f1 = -float("inf")

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        """Evaluate diagnostics without changing validation-loss selection."""
        del epoch
        logs = logs if logs is not None else {}
        rows = [
            np.asarray(self.model(self.inputs[start : start + self.batch_size], training=False))
            for start in range(0, len(self.labels), self.batch_size)
        ]
        probabilities = _normalize_probabilities(np.concatenate(rows), expected_rows=len(self.labels))
        metrics, _ = _classification_report(self.labels, probabilities)
        source_reports = _source_reports(self.labels, probabilities, self.sources)
        logs["val_macro_f1"] = metrics["macro_f1"]
        logs["val_report_accuracy"] = metrics["accuracy"]
        logs["val_equal_source_mean_macro_f1"] = source_reports["equal_source_mean_macro_f1"]
        for source in SOURCES:
            report = source_reports["sources"][source]
            if report is not None:
                logs[f"val_{source}_macro_f1"] = report["macro_f1"]
        self.best_macro_f1 = max(self.best_macro_f1, metrics["macro_f1"])
        logs["val_best_macro_f1"] = self.best_macro_f1


class FacialExpressionClassificationTask(TaskABC):
    """Probability-output facial-expression classification task."""

    def __init__(self, *, checkpoint_path: Path, early_stopping_patience: int = 30) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.early_stopping_patience = _positive_integer(
            early_stopping_patience, field_name="early_stopping_patience"
        )

    @property
    def name(self) -> str:
        """Return the stable registry key."""
        return COMPONENT_NAME

    def validate_config(self, task_config: Any) -> None:
        """Validate optimizer, augmentation, callback, and derived batch settings."""
        allowed = {
            "early_stopping_patience",
            "learning_rate",
            "lr_patience",
            "augmentation_seed",
            "report_validation_macro_f1",
            RUNTIME_TRAINING_BATCH_SIZE_KEY,
        }
        unknown = sorted(_config_keys(task_config) - allowed)
        if unknown:
            raise ValueError(f"Facial-expression task.params has unsupported keys: {', '.join(unknown)}.")
        if _cfg_get(task_config, "early_stopping_patience", None) is not None:
            _positive_integer(
                _cfg_get(task_config, "early_stopping_patience"),
                field_name="task.params.early_stopping_patience",
            )
        _positive_number(
            _cfg_get(task_config, "learning_rate", 0.001),
            field_name="task.params.learning_rate",
        )
        _positive_integer(
            _cfg_get(task_config, "lr_patience", 8),
            field_name="task.params.lr_patience",
        )
        seed = _cfg_get(task_config, "augmentation_seed", 2026)
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("task.params.augmentation_seed must be a nonnegative integer.")
        report = _cfg_get(task_config, "report_validation_macro_f1", True)
        if not isinstance(report, bool):
            raise ValueError("task.params.report_validation_macro_f1 must be a real boolean.")
        runtime_batch = _cfg_get(task_config, RUNTIME_TRAINING_BATCH_SIZE_KEY, None)
        if runtime_batch is not None:
            _positive_integer(runtime_batch, field_name=RUNTIME_TRAINING_BATCH_SIZE_KEY)

    def build_target_spec(self, bundle: DatasetBundle, task_config: Any) -> TargetSpec:
        """Build the fixed five-class probability target contract."""
        del task_config
        if bundle.metadata.get("label_source") != LABEL_SOURCE:
            raise ValueError("Facial-expression task requires the verified mixed label source.")
        if bundle.metadata.get("class_names") != list(CLASS_NAMES):
            raise ValueError("Facial-expression class order does not match the task contract.")
        training_targets = np.asarray(bundle.train.metadata.get("training_targets"))
        if training_targets.shape != (len(bundle.train.targets), 5):
            raise ValueError("Facial-expression training soft targets are missing or misaligned.")
        return TargetSpec(
            task_type="classification",
            output_names=["probabilities"],
            output_shapes=[(5,)],
            metadata={
                "num_classes": 5,
                "class_names": list(CLASS_NAMES),
                "label_encoding": "conditional_soft_training_hard_evaluation",
                "from_logits": False,
            },
        )

    def metric_contract(self, target_spec: TargetSpec, task_config: Any) -> TaskMetricContract:
        """Declare pooled prediction CCE, accuracy, and macro-F1 metrics."""
        del target_spec, task_config
        names = {"loss", "accuracy", "macro_f1"}
        return TaskMetricContract(
            available_metric_names=set(names),
            training_only_metric_names=set(),
            nonnegative_metric_names=set(names),
            primary_metric_names={"accuracy", "macro_f1"},
        )

    def compile_model(self, model: Any, task_config: Any, target_spec: TargetSpec) -> None:
        """Compile the probability model with reference Adam and categorical CE."""
        del target_spec
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=_positive_number(
                    _cfg_get(task_config, "learning_rate", 0.001),
                    field_name="task.params.learning_rate",
                )
            ),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
        )

    @staticmethod
    def _fit_arrays(
        bundle: DatasetBundle,
        *,
        combine_train_val: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Resolve external inputs, training targets, weights, and source membership."""
        train_inputs = np.asarray(bundle.train.inputs)
        train_targets = np.asarray(bundle.train.metadata["training_targets"], dtype=np.float32)
        train_sources = np.asarray(bundle.train.metadata["sources"])
        train_weights = np.asarray(bundle.train.sample_weights, dtype=np.float32)
        if not combine_train_val:
            return train_inputs, train_targets, train_weights, train_sources
        if bundle.val is None:
            raise ValueError("Final combined facial-expression training requires validation data.")
        val_labels = np.asarray(bundle.val.targets, dtype=np.int64)
        inputs = np.concatenate([train_inputs, np.asarray(bundle.val.inputs)], axis=0)
        targets = np.concatenate(
            [train_targets, np.eye(5, dtype=np.float32)[val_labels]], axis=0
        )
        sources = np.concatenate([train_sources, np.asarray(bundle.val.metadata["sources"])])
        return inputs, targets, equal_source_weights(sources), sources

    def build_fit_plan(
        self,
        bundle: DatasetBundle,
        task_config: Any,
        target_spec: TargetSpec,
        *,
        mode: Literal["search", "final"],
        combine_train_val: bool,
    ) -> FitPlan:
        """Build the prebatched augmentation pipeline and validation callbacks."""
        del target_spec
        if mode not in {"search", "final"}:
            raise ValueError("Facial-expression fit mode must be 'search' or 'final'.")
        batch_size = _positive_integer(
            _cfg_get(task_config, RUNTIME_TRAINING_BATCH_SIZE_KEY, None),
            field_name=RUNTIME_TRAINING_BATCH_SIZE_KEY,
        )
        inputs, targets, weights, _ = self._fit_arrays(
            bundle, combine_train_val=combine_train_val
        )
        augmentation = build_augmentation(int(_cfg_get(task_config, "augmentation_seed", 2026)))
        training = tf.data.Dataset.from_tensor_slices((inputs, targets, weights))
        training = training.shuffle(len(inputs), seed=int(_cfg_get(task_config, "augmentation_seed", 2026)))
        training = training.batch(batch_size, drop_remainder=False)

        def augment_batch(batch_inputs: Any, batch_targets: Any, batch_weights: Any) -> tuple[Any, Any, Any]:
            channel_inputs = tf.expand_dims(batch_inputs, axis=-1)
            augmented = augmentation(channel_inputs, training=True)
            restored = tf.squeeze(tf.clip_by_value(augmented, -1.0, 1.0), axis=-1)
            return restored, batch_targets, batch_weights

        training = training.map(augment_batch, num_parallel_calls=1, deterministic=True).prefetch(1)
        monitor = "loss" if combine_train_val else "val_loss"
        callbacks: list[Any] = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(self.checkpoint_path),
                monitor=monitor,
                mode="min",
                save_best_only=True,
                verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=self.early_stopping_patience,
                mode="min",
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=monitor,
                patience=_positive_integer(
                    _cfg_get(task_config, "lr_patience", 8),
                    field_name="task.params.lr_patience",
                ),
                factor=0.5,
                min_lr=1e-6,
            ),
            tf.keras.callbacks.TerminateOnNaN(),
        ]
        fit_kwargs: dict[str, Any] = {"x": training}
        if not combine_train_val:
            if bundle.val is None:
                raise ValueError("Facial-expression search training requires validation data.")
            val_targets = np.eye(5, dtype=np.float32)[np.asarray(bundle.val.targets, dtype=np.int64)]
            validation = tf.data.Dataset.from_tensor_slices((bundle.val.inputs, val_targets))
            validation = validation.batch(batch_size, drop_remainder=False).prefetch(1)
            fit_kwargs["validation_data"] = validation
            if bool(_cfg_get(task_config, "report_validation_macro_f1", True)):
                callbacks.insert(0, _ValidationMetrics(bundle.val, batch_size))
        return FitPlan(fit_kwargs=fit_kwargs, callbacks=callbacks, monitor_metric=monitor)

    def validate_model_outputs(self, model: Any, target_spec: TargetSpec) -> None:
        """Require one named five-class softmax probability tensor."""
        output_shape = getattr(model, "output_shape", None)
        if isinstance(output_shape, list):
            if len(output_shape) != 1:
                raise ValueError("Facial-expression models require exactly one output.")
            output_shape = output_shape[0]
        if tuple(output_shape or ()) != (None, int(target_spec.metadata["num_classes"])):
            raise ValueError("Facial-expression model output must have shape (None, 5).")
        if list(getattr(model, "output_names", [])) != target_spec.output_names:
            raise ValueError("Facial-expression model output must be named 'probabilities'.")
        layers = list(getattr(model, "layers", []) or [])
        if not layers:
            raise ValueError("Facial-expression models require a structural softmax output.")
        final = layers[-1]
        activation = getattr(final, "activation", None)
        if not isinstance(final, tf.keras.layers.Softmax) and getattr(activation, "__name__", "") != "softmax":
            raise ValueError("Facial-expression models must emit softmax probabilities.")

    def evaluate_predictions(
        self,
        predictions: Any,
        split: DataSplit,
        task_config: Any,
        target_spec: TargetSpec,
    ) -> EvaluationResult:
        """Evaluate pooled hard labels and retain source-specific diagnostics."""
        del task_config, target_spec
        labels = np.asarray(split.targets, dtype=np.int64).reshape(-1)
        if len(labels) == 0 or np.any((labels < 0) | (labels >= 5)):
            raise ValueError("Facial-expression evaluation labels are invalid.")
        normalized = _normalize_probabilities(predictions, expected_rows=len(labels))
        raw_payload = predictions[0] if isinstance(predictions, (list, tuple)) and len(predictions) == 1 else predictions
        raw_probabilities = np.asarray(raw_payload, dtype=np.float32)
        metrics, artifacts = _classification_report(labels, normalized)
        sources = np.asarray(split.metadata.get("sources"))
        if sources.shape == (len(labels),):
            artifacts.update(_source_reports(labels, normalized, sources))
        return EvaluationResult(
            metrics=metrics,
            artifacts=artifacts,
            predictions=raw_probabilities,
        )

    def evaluate(
        self,
        model: Any,
        split: DataSplit,
        task_config: Any,
        target_spec: TargetSpec,
    ) -> EvaluationResult:
        """Evaluate a Keras probability model on one prepared split."""
        predictions = model.predict(split.inputs)
        return self.evaluate_predictions(predictions, split, task_config, target_spec)
