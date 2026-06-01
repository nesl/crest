# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Sound classification task adapter for cached audio features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam

from ..interfaces import TaskABC
from ..pipeline_types import DataSplit, DatasetBundle, EvaluationResult, FitPlan, TargetSpec, TaskMetricContract
from ..datasets.urbansound8k_common import CLASS_NAMES, LABEL_ENCODING


def _require_positive_integer(value: Any, *, field_name: str) -> int:
    """Validate one positive integer configuration value.

    Parameters
    ----------
    value : Any
        Raw value to validate.
    field_name : str
        Human-readable field name for errors.

    Returns
    -------
    int
        Validated integer value.

    Raises
    ------
    ValueError
        If the value is boolean, non-integer, or non-positive.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be a positive integer.")
    if value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return int(value)


def _as_logits_array(predictions: Any) -> np.ndarray:
    """Normalize Keras prediction output to a logits array.

    Parameters
    ----------
    predictions : Any
        Raw value returned by ``model.predict(...)``.

    Returns
    -------
    numpy.ndarray
        Two-dimensional logits array.

    Raises
    ------
    ValueError
        If predictions are not a single 2D output.
    """
    if isinstance(predictions, (list, tuple)):
        if len(predictions) != 1:
            raise ValueError("SoundClassificationTask expects one logits prediction output.")
        predictions = predictions[0]
    logits = np.asarray(predictions, dtype=np.float32)
    if logits.ndim != 2:
        raise ValueError("SoundClassificationTask predictions must be a 2D logits array.")
    return logits


def _sparse_cross_entropy_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    """Compute mean sparse categorical cross-entropy from logits.

    Parameters
    ----------
    logits : numpy.ndarray
        Logits with shape ``(N, num_classes)``.
    labels : numpy.ndarray
        Integer labels with shape ``(N,)``.

    Returns
    -------
    float
        Mean cross-entropy loss.
    """
    if logits.shape[0] == 0:
        return 0.0
    max_logits = np.max(logits, axis=1, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(logits - max_logits), axis=1)) + max_logits.reshape(-1)
    losses = logsumexp - logits[np.arange(labels.shape[0]), labels]
    return float(np.mean(losses))


def _classification_artifacts(
    labels: np.ndarray,
    predicted_ids: np.ndarray,
    *,
    num_classes: int,
    class_names: list[str],
) -> dict[str, Any]:
    """Build JSON-safe confusion and per-class classification artifacts.

    Parameters
    ----------
    labels : numpy.ndarray
        Integer ground-truth labels.
    predicted_ids : numpy.ndarray
        Integer predicted class IDs.
    num_classes : int
        Number of classes in the classification task.
    class_names : list[str]
        Ordered class names.

    Returns
    -------
    dict[str, Any]
        JSON-safe confusion matrix and per-class metrics.
    """
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, predicted in zip(labels, predicted_ids):
        confusion[int(truth), int(predicted)] += 1

    support = confusion.sum(axis=1)
    predicted_counts = confusion.sum(axis=0)
    true_positive = np.diag(confusion)

    per_class_accuracy: list[float] = []
    per_class_f1: list[float] = []
    for class_index in range(num_classes):
        recall = (
            float(true_positive[class_index] / support[class_index])
            if support[class_index] > 0
            else 0.0
        )
        precision = (
            float(true_positive[class_index] / predicted_counts[class_index])
            if predicted_counts[class_index] > 0
            else 0.0
        )
        f1 = (
            float(2.0 * precision * recall / (precision + recall))
            if (precision + recall) > 0.0
            else 0.0
        )
        per_class_accuracy.append(recall)
        per_class_f1.append(f1)

    return {
        "confusion_matrix": confusion.astype(int).tolist(),
        "per_class_counts": support.astype(int).tolist(),
        "per_class_accuracy": [float(value) for value in per_class_accuracy],
        "per_class_f1": [float(value) for value in per_class_f1],
        "class_names": list(class_names),
    }


def _extract_single_output_shape(model: Any) -> tuple[Any, ...]:
    """Return the single output shape for a Keras-like model.

    Parameters
    ----------
    model : Any
        Model exposing ``output_shape``.

    Returns
    -------
    tuple[Any, ...]
        Single output shape.

    Raises
    ------
    ValueError
        If the model exposes zero or multiple outputs.
    """
    output_shape = getattr(model, "output_shape", None)
    if isinstance(output_shape, list):
        if len(output_shape) != 1:
            raise ValueError("SoundClassificationTask requires exactly one model output.")
        output_shape = output_shape[0]
    if output_shape is None:
        raise ValueError("SoundClassificationTask requires model.output_shape.")
    return tuple(output_shape)


def _has_probability_final_layer(model: Any) -> bool:
    """Return whether the model appears to end in softmax probabilities.

    Parameters
    ----------
    model : Any
        Keras-like model exposing ``layers``.

    Returns
    -------
    bool
        ``True`` when the final layer structurally applies softmax.
    """
    layers = list(getattr(model, "layers", []) or [])
    if not layers:
        return False
    final_layer = layers[-1]
    if isinstance(final_layer, tf.keras.layers.Softmax):
        return True
    activation = getattr(final_layer, "activation", None)
    return getattr(activation, "__name__", "") == "softmax"


class SoundClassificationTask(TaskABC):
    """Single-output logits classification task for audio features."""

    def __init__(
        self,
        *,
        checkpoint_path: Path,
        early_stopping_patience: int = 40,
    ) -> None:
        """Initialize task-owned callback configuration.

        Parameters
        ----------
        checkpoint_path : pathlib.Path
            Checkpoint destination used by the task-owned callback.
        early_stopping_patience : int, optional
            Positive integer early-stopping patience.
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.early_stopping_patience = _require_positive_integer(
            early_stopping_patience,
            field_name="early_stopping_patience",
        )

    @property
    def name(self) -> str:
        """Return the stable registry-facing task name.

        Returns
        -------
        str
            Stable task identifier.
        """
        return "sound_classification"

    def validate_config(self, task_config: Any) -> None:
        """Validate sound-classification task configuration.

        Parameters
        ----------
        task_config : Any
            Task-local configuration subtree.

        Returns
        -------
        None
            Raises when optional patience config is invalid.
        """
        getter = getattr(task_config, "get", None)
        raw_patience = getter("early_stopping_patience", None) if callable(getter) else getattr(task_config, "early_stopping_patience", None)
        if raw_patience is not None:
            _require_positive_integer(raw_patience, field_name="task.params.early_stopping_patience")

    def build_target_spec(
        self,
        bundle: DatasetBundle,
        task_config: Any,
    ) -> TargetSpec:
        """Build the logits classification target contract.

        Parameters
        ----------
        bundle : DatasetBundle
            Loaded dataset bundle.
        task_config : Any
            Task-local configuration subtree.

        Returns
        -------
        TargetSpec
            Single-output logits classification contract.

        Raises
        ------
        ValueError
            If existing validation or execution checks fail.
        """
        del task_config
        class_names = list(CLASS_NAMES)
        num_classes = len(class_names)
        if bundle.metadata.get("num_classes") != num_classes:
            raise ValueError("SoundClassificationTask requires UrbanSound8K num_classes metadata.")
        if list(bundle.metadata.get("class_names", [])) != class_names:
            raise ValueError("SoundClassificationTask requires UrbanSound8K class_names metadata.")
        if bundle.metadata.get("label_encoding") != LABEL_ENCODING:
            raise ValueError("SoundClassificationTask requires class_index labels.")
        return TargetSpec(
            task_type="classification",
            output_names=["class_logits"],
            output_shapes=[(num_classes,)],
            metadata={
                "num_classes": num_classes,
                "class_names": class_names,
                "label_encoding": LABEL_ENCODING,
                "from_logits": True,
            },
        )

    def metric_contract(
        self,
        target_spec: TargetSpec,
        task_config: Any,
    ) -> TaskMetricContract:
        """Describe metrics produced by the sound-classification task.

        Parameters
        ----------
        target_spec : TargetSpec
            Task-owned target specification.
        task_config : Any
            Task-local configuration subtree.

        Returns
        -------
        TaskMetricContract
            Metric declaration for loss, accuracy, and macro-F1.
        """
        del target_spec, task_config
        metric_names = {"loss", "accuracy", "macro_f1"}
        return TaskMetricContract(
            available_metric_names=set(metric_names),
            training_only_metric_names=set(),
            nonnegative_metric_names=set(metric_names),
            primary_metric_names={"accuracy", "macro_f1"},
        )

    def compile_model(
        self,
        model: Any,
        task_config: Any,
        target_spec: TargetSpec,
    ) -> None:
        """Compile a logits classification model.

        Parameters
        ----------
        model : Any
            Keras model to compile.
        task_config : Any
            Task-local configuration subtree.
        target_spec : TargetSpec
            Task-owned target specification.

        Returns
        -------
        None
            Mutates ``model`` by applying optimizer, loss, and metrics.
        """
        del task_config, target_spec
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=SparseCategoricalCrossentropy(from_logits=True),
            metrics=[SparseCategoricalAccuracy(name="accuracy")],
        )

    def build_fit_plan(
        self,
        bundle: DatasetBundle,
        task_config: Any,
        target_spec: TargetSpec,
        *,
        mode: Literal["search", "final"],
        combine_train_val: bool,
    ) -> FitPlan:
        """Build task-owned Keras fit wiring.

        Parameters
        ----------
        bundle : DatasetBundle
            Loaded dataset bundle.
        task_config : Any
            Task-local configuration subtree.
        target_spec : TargetSpec
            Task-owned target specification.
        mode : {"search", "final"}
            Runner phase requesting the fit plan.
        combine_train_val : bool
            Whether validation data should be merged into training.

        Returns
        -------
        FitPlan
            Fit kwargs and task-owned callbacks.

        Raises
        ------
        ValueError
            If existing validation or execution checks fail.
        """
        del task_config, target_spec
        if mode not in {"search", "final"}:
            raise ValueError("SoundClassificationTask build_fit_plan mode must be 'search' or 'final'.")

        train_inputs = bundle.train.inputs
        train_targets = bundle.train.targets
        validation_data = None
        monitor_metric = "val_loss"

        if combine_train_val:
            monitor_metric = "loss"
            if bundle.val is not None:
                train_inputs = np.concatenate([bundle.train.inputs, bundle.val.inputs], axis=0)
                train_targets = np.concatenate([bundle.train.targets, bundle.val.targets], axis=0)
        elif bundle.val is None:
            raise ValueError("SoundClassificationTask requires validation data when combine_train_val=False.")
        else:
            validation_data = (bundle.val.inputs, bundle.val.targets)

        checkpoint = ModelCheckpoint(
            filepath=str(self.checkpoint_path),
            monitor=monitor_metric,
            mode="min",
            verbose=1,
            save_best_only=True,
        )
        early_stop = EarlyStopping(
            monitor=monitor_metric,
            patience=self.early_stopping_patience,
            mode="min",
            verbose=1,
            restore_best_weights=True,
        )

        fit_kwargs = {
            "x": train_inputs,
            "y": train_targets,
            "shuffle": True,
        }
        if validation_data is not None:
            fit_kwargs["validation_data"] = validation_data
        return FitPlan(
            fit_kwargs=fit_kwargs,
            callbacks=[checkpoint, early_stop],
            monitor_metric=monitor_metric,
        )

    def validate_model_outputs(
        self,
        model: Any,
        target_spec: TargetSpec,
    ) -> None:
        """Validate that a model emits one logits tensor.

        Parameters
        ----------
        model : Any
            Built Keras model.
        target_spec : TargetSpec
            Task-owned target specification.

        Returns
        -------
        None
            Raises when output shape or final activation is incompatible.

        Raises
        ------
        ValueError
            If existing validation or execution checks fail.
        """
        output_shape = _extract_single_output_shape(model)
        num_classes = int(target_spec.metadata["num_classes"])
        if len(output_shape) != 2 or output_shape[0] is not None or int(output_shape[-1]) != num_classes:
            raise ValueError(
                "SoundClassificationTask requires one output with shape "
                f"(None, {num_classes})."
            )
        if _has_probability_final_layer(model):
            raise ValueError("SoundClassificationTask requires logits, not softmax probabilities.")

    def evaluate(
        self,
        model: Any,
        split: DataSplit,
        task_config: Any,
        target_spec: TargetSpec,
    ) -> EvaluationResult:
        """Evaluate one split from logits in a single forward pass.

        Parameters
        ----------
        model : Any
            Keras model to evaluate.
        split : DataSplit
            Split containing inputs and integer class-index targets.
        task_config : Any
            Task-local configuration subtree.
        target_spec : TargetSpec
            Task-owned target specification.

        Returns
        -------
        EvaluationResult
            JSON-safe metrics, artifacts, and predicted class IDs.
        """
        predictions = model.predict(split.inputs)
        return self.evaluate_predictions(predictions, split, task_config, target_spec)

    def evaluate_predictions(
        self,
        predictions: Any,
        split: DataSplit,
        task_config: Any,
        target_spec: TargetSpec,
    ) -> EvaluationResult:
        """Evaluate classification metrics from normalized logits.

        Parameters
        ----------
        predictions : Any
            Logit prediction payload.
        split : DataSplit
            Split containing integer class-index targets.
        task_config : Any
            Task-local configuration subtree.
        target_spec : TargetSpec
            Task-owned target specification.

        Returns
        -------
        EvaluationResult
            JSON-safe metrics, artifacts, and predicted class IDs.

        Raises
        ------
        ValueError
            If existing validation or execution checks fail.
        """
        del task_config
        labels = np.asarray(split.targets, dtype=np.int64).reshape(-1)
        logits = _as_logits_array(predictions)
        if logits.shape[0] != labels.shape[0]:
            raise ValueError("SoundClassificationTask prediction count does not match labels.")
        num_classes = int(target_spec.metadata["num_classes"])
        if logits.shape[1] != num_classes:
            raise ValueError("SoundClassificationTask logits class count does not match target spec.")
        if labels.size and (np.any(labels < 0) or np.any(labels >= num_classes)):
            raise ValueError("SoundClassificationTask labels are outside the target class range.")

        predicted_ids = np.argmax(logits, axis=1).astype(np.int64)
        accuracy = float(np.mean(predicted_ids == labels)) if labels.shape[0] > 0 else 0.0
        loss = _sparse_cross_entropy_from_logits(logits, labels)
        artifacts = _classification_artifacts(
            labels,
            predicted_ids,
            num_classes=num_classes,
            class_names=list(target_spec.metadata["class_names"]),
        )
        macro_f1 = float(np.mean(artifacts["per_class_f1"])) if num_classes > 0 else 0.0
        return EvaluationResult(
            metrics={
                "loss": float(loss),
                "accuracy": float(accuracy),
                "macro_f1": float(macro_f1),
            },
            artifacts=artifacts,
            predictions=[int(value) for value in predicted_ids.tolist()],
        )

    def generate_closeout_artifacts(
        self,
        model: Any,
        dataset_bundle: DatasetBundle,
        task_config: Any,
        target_spec: TargetSpec,
        *,
        output_dir: Path,
    ) -> dict[str, Any]:
        """Write compact JSON closeout metrics for the test split.

        Parameters
        ----------
        model : Any
            Final model to evaluate.
        dataset_bundle : DatasetBundle
            Dataset bundle backing the run.
        task_config : Any
            Task-local configuration subtree.
        target_spec : TargetSpec
            Task-owned target specification.
        output_dir : pathlib.Path
            Directory where the closeout JSON file is written.

        Returns
        -------
        dict[str, Any]
            JSON-safe artifact summary, or an empty dict when no test split
            exists.
        """
        if dataset_bundle.test is None:
            return {}
        output_dir.mkdir(parents=True, exist_ok=True)
        evaluation = self.evaluate(model, dataset_bundle.test, task_config, target_spec)
        payload = {
            "metrics": dict(evaluation.metrics),
            "artifacts": dict(evaluation.artifacts or {}),
            "predictions": list(evaluation.predictions or []),
        }
        output_path = output_dir / "sound_classification_metrics.json"
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return {
            "sound_classification_metrics_path": str(output_path),
            **payload["artifacts"],
        }
