# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Task-owned training, metrics, and evaluation contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from ..pipeline_types import (
    DataSplit, DatasetBundle, EvaluationResult, FitPlan, TargetSpec, TaskMetricContract,
)

if TYPE_CHECKING:
    import tensorflow as tf


class TaskABC(ABC):
    """Abstract task contract for training, evaluation, and metrics.

    Implementations define the task-owned target contract, metric declaration,
    compile behavior, fit wiring, and evaluation logic. By default,
    :attr:`name` returns the implementing class name,
    :meth:`validate_config` performs no validation, and
    :meth:`validate_model_outputs` performs no output checks.
    """

    @property
    def name(self) -> str:
        """Return a human-readable task identifier.

        Returns
        -------
        str
            Default identifier derived from the implementing class name.
        """
        return type(self).__name__

    @abstractmethod
    def build_target_spec(
        self,
        bundle: DatasetBundle,
        task_config: Any,
    ) -> TargetSpec:
        """Build the concrete target/output contract for one task.

        Parameters
        ----------
        bundle : DatasetBundle
            Normalized dataset package.
        task_config : Any
            Task-local configuration subtree.

        Returns
        -------
        TargetSpec
            Concrete target and output contract for model construction and
            validation. This is the task-owned contract returned by the task
            implementation for downstream task logic.
        """
    @abstractmethod
    def metric_contract(
        self,
        target_spec: TargetSpec,
        task_config: Any,
    ) -> TaskMetricContract:
        """Describe task-defined metric availability.

        Parameters
        ----------
        target_spec : TargetSpec
            Task-owned target specification.
        task_config : Any
            Task-local configuration subtree.

        Returns
        -------
        TaskMetricContract
            Task-defined metric declaration consumed by orchestration code.
        """
    @abstractmethod
    def compile_model(
        self,
        model: tf.keras.Model,
        task_config: Any,
        target_spec: TargetSpec,
    ) -> None:
        """Apply task-owned compile semantics to a model.

        Parameters
        ----------
        model : tf.keras.Model
            Uncompiled model returned by a model family.
        task_config : Any
            Task-local configuration subtree.
        target_spec : TargetSpec
            Task-owned target specification.

        Returns
        -------
        None
            The task implementation applies compile configuration to the passed
            model using ``task_config`` and ``target_spec``.
        """
    @abstractmethod
    def build_fit_plan(
        self,
        bundle: DatasetBundle,
        task_config: Any,
        target_spec: TargetSpec,
        *,
        mode: Literal["search", "final"],
        combine_train_val: bool,
    ) -> FitPlan:
        """Build task-owned ``model.fit(...)`` wiring.

        Parameters
        ----------
        bundle : DatasetBundle
            Normalized dataset package.
        task_config : Any
            Task-local configuration subtree.
        target_spec : TargetSpec
            Task-owned target specification.
        mode : {"search", "final"}
            Runner phase requesting the fit plan.
        combine_train_val : bool
            Whether the task should fold the validation split into training
            for this plan.

        Returns
        -------
        FitPlan
            Task-owned fit wiring excluding runner-injected schedule policy.
        """
    def history_component_keys(
        self,
        target_spec: TargetSpec,
    ) -> list[tuple[str, str]]:
        """Return optional per-output history key pairs for closeout plots.

        Parameters
        ----------
        target_spec : TargetSpec
            Task-owned target specification.

        Returns
        -------
        list[tuple[str, str]]
            Ordered ``(train_key, val_key)`` pairs to plot when present in a
            Keras history dictionary. The default implementation returns an
            empty list so generic tasks do not need to expose component-loss
            plots.
        """
        del target_spec
        return []

    def generate_closeout_artifacts(
        self,
        model: tf.keras.Model,
        dataset_bundle: DatasetBundle,
        task_config: Any,
        target_spec: TargetSpec,
        *,
        output_dir: Path,
    ) -> dict[str, Any]:
        """Generate optional task-owned closeout artifacts after final training.

        Parameters
        ----------
        model : tf.keras.Model
            Final trained model available for task-specific closeout work.
        dataset_bundle : DatasetBundle
            Dataset bundle backing the current run.
        task_config : Any
            Task-local configuration subtree.
        target_spec : TargetSpec
            Task-owned target specification.
        output_dir : Path
            Directory where task-owned artifacts should be written.

        Returns
        -------
        dict[str, Any]
            Structured task-owned closeout payload. Tasks may return paths,
            metrics, or small JSON-serializable summaries. The default
            implementation returns an empty mapping so tasks opt into extra
            closeout artifacts explicitly.
        """
        del model, dataset_bundle, task_config, target_spec, output_dir
        return {}

    @abstractmethod
    def evaluate_predictions(
        self,
        predictions: Any,
        split: DataSplit,
        task_config: Any,
        target_spec: TargetSpec,
    ) -> EvaluationResult:
        """Evaluate normalized predictions for one split.

        Parameters
        ----------
        predictions : Any
            Prediction payload already normalized into the task's expected
            shape.
        split : DataSplit
            Dataset split to evaluate.
        task_config : Any
            Task-local configuration subtree.
        target_spec : TargetSpec
            Task-owned target specification.

        Returns
        -------
        EvaluationResult
            Structured evaluation output.
        """
    @abstractmethod
    def evaluate(
        self,
        model: tf.keras.Model,
        split: DataSplit,
        task_config: Any,
        target_spec: TargetSpec,
    ) -> EvaluationResult:
        """Evaluate a model on one split and return structured results.

        Parameters
        ----------
        model : tf.keras.Model
            Compiled or already-trained model.
        split : DataSplit
            Dataset split to evaluate.
        task_config : Any
            Task-local configuration subtree.
        target_spec : TargetSpec
            Task-owned target specification.

        Returns
        -------
        EvaluationResult
            Structured evaluation output.
        """
    def validate_config(self, task_config: Any) -> None:
        """Validate task-local configuration.

        Parameters
        ----------
        task_config : Any
            Task-local configuration subtree.

        Returns
        -------
        None
            The default implementation intentionally performs no validation and
            accepts any config object.
        """
        del task_config

    def validate_model_outputs(
        self,
        model: tf.keras.Model,
        target_spec: TargetSpec,
    ) -> None:
        """Validate model outputs against the task contract.

        Parameters
        ----------
        model : tf.keras.Model
            Built model to validate.
        target_spec : TargetSpec
            Task-owned target specification.

        Returns
        -------
        None
            The default implementation intentionally performs no checks and
            exists as an override hook for task-specific output validation.
        """
        del model, target_spec
