# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Shared payload contracts passed between dataset, task, and model code.

This module centralizes the lightweight dataclasses that move structured
payloads across the CREST pipeline. The payload fields intentionally stay
permissive so dataset, task, and model-family implementations can exchange
arrays, tensors, and other runtime-specific objects without forcing the
pipeline onto one concrete backend representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(eq=False)
class DataSplit:
    """Container for one dataset partition.

    This dataclass is a passive transport container. It groups split payloads
    without validating, coercing, or copying them on construction.

    Parameters
    ----------
    inputs : Any
        Model-ready inputs for one split. The exact runtime type is
        intentionally left broad so dataset implementations can supply arrays,
        tensors, or other Keras-compatible structures.
    targets : Any
        Split targets interpreted by the owning task.
    sample_weights : Any | None, optional
        Optional sample-weight payload forwarded to task or training code.
    metadata : dict[str, Any], optional
        Split-local metadata. Defaults to an empty dictionary.

    Attributes
    ----------
    inputs : Any
        Model-ready inputs for one split. The exact runtime type is
        intentionally left broad so dataset implementations can supply arrays,
        tensors, or other Keras-compatible structures.
    targets : Any
        Split targets interpreted by the owning task.
    sample_weights : Any | None
        Optional sample-weight payload forwarded to task or training code.
    metadata : dict[str, Any]
        Split-local metadata. Defaults to an empty dictionary.
    """

    inputs: Any
    targets: Any
    sample_weights: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(eq=False)
class DatasetBundle:
    """Normalized dataset package returned by a dataset implementation.

    A valid bundle always includes a training split. Validation, test, and
    calibration splits are optional attachments populated only when the dataset
    can supply them.

    Parameters
    ----------
    train : DataSplit
        Training split.
    val : DataSplit | None, optional
        Validation split when available.
    test : DataSplit | None, optional
        Test split when available.
    calibration : DataSplit | None, optional
        Calibration split used for export-time representative data when needed.
    input_shape : tuple[int, ...] | None, optional
        Shape of one logical input example, excluding the batch dimension.
    input_dtype : str | None, optional
        Input dtype label expected by the model family.
    metadata : dict[str, Any], optional
        Dataset-wide metadata carried into later pipeline stages.

    Attributes
    ----------
    train : DataSplit
        Training split.
    val : DataSplit | None
        Validation split when available.
    test : DataSplit | None
        Test split when available.
    calibration : DataSplit | None
        Calibration split used for export-time representative data when needed.
    input_shape : tuple[int, ...] | None
        Shape of one logical input example, excluding the batch dimension.
    input_dtype : str | None
        Input dtype label expected by the model family.
    metadata : dict[str, Any]
        Dataset-wide metadata carried into later pipeline stages.
    """

    train: DataSplit
    val: DataSplit | None = None
    test: DataSplit | None = None
    calibration: DataSplit | None = None
    input_shape: tuple[int, ...] | None = None
    input_dtype: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(eq=False)
class TargetSpec:
    """Concrete target and output contract for one task.

    Parameters
    ----------
    task_type : str
        Stable task-type label, such as ``"regression"`` or
        ``"classification"``.
    output_names : list[str]
        Ordered output names produced by the model.
    output_shapes : list[Any]
        Ordered output-shape descriptors matching ``output_names``.
    metadata : dict[str, Any], optional
        Task-owned metadata required by model construction or evaluation.

    Attributes
    ----------
    task_type : str
        Stable task-type label, such as ``"regression"`` or
        ``"classification"``.
    output_names : list[str]
        Ordered output names produced by the model.
    output_shapes : list[Any]
        Ordered output-shape descriptors matching ``output_names``.
    metadata : dict[str, Any]
        Task-owned metadata required by model construction or evaluation.
    """

    task_type: str
    output_names: list[str]
    output_shapes: list[Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(eq=False)
class ModelBuildContext:
    """Normalized build-time context supplied to a model family.

    This context is the model-family-facing handoff assembled from dataset and
    task outputs. Unlike :class:`DatasetBundle`, it carries only the build-time
    contract needed to construct a model, not the raw split payloads.

    Parameters
    ----------
    input_shape : tuple[int, ...] | None
        Shape of one logical input example, excluding the batch dimension.
    input_dtype : str | None
        Input dtype label expected by the model family.
    target_spec : TargetSpec
        Concrete target/output contract produced by the task.
    dataset_metadata : dict[str, Any], optional
        Dataset-owned metadata relevant to model construction.
    task_metadata : dict[str, Any], optional
        Task-owned metadata relevant to model construction.

    Attributes
    ----------
    input_shape : tuple[int, ...] | None
        Shape of one logical input example, excluding the batch dimension.
    input_dtype : str | None
        Input dtype label expected by the model family.
    target_spec : TargetSpec
        Concrete target/output contract produced by the task.
    dataset_metadata : dict[str, Any]
        Dataset-owned metadata relevant to model construction.
    task_metadata : dict[str, Any]
        Task-owned metadata relevant to model construction.
    """

    input_shape: tuple[int, ...] | None
    input_dtype: str | None
    target_spec: TargetSpec
    dataset_metadata: dict[str, Any] = field(default_factory=dict)
    task_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(eq=False)
class FitPlan:
    """Task-owned training plan for one ``model.fit(...)`` call.

    Parameters
    ----------
    fit_kwargs : dict[str, Any], optional
        Keyword arguments forwarded to ``model.fit(...)`` that describe
        task-owned data wiring, such as ``x``, ``y``, and ``validation_data``.
        Runner-owned schedule fields such as ``epochs`` and ``batch_size`` are
        intentionally excluded and should be injected by orchestration code.
    callbacks : list[Any], optional
        Callback objects or callback-like payloads owned by the task.
    monitor_metric : str | None, optional
        Optional monitor metric name used by task-owned callbacks.

    Attributes
    ----------
    fit_kwargs : dict[str, Any]
        Keyword arguments forwarded to ``model.fit(...)`` that describe
        task-owned data wiring, such as ``x``, ``y``, and ``validation_data``.
        Runner-owned schedule fields such as ``epochs`` and ``batch_size`` are
        intentionally excluded and should be injected by orchestration code.
    callbacks : list[Any]
        Callback objects or callback-like payloads owned by the task.
    monitor_metric : str | None
        Optional monitor metric name used by task-owned callbacks.
    """

    fit_kwargs: dict[str, Any] = field(default_factory=dict)
    callbacks: list[Any] = field(default_factory=list)
    monitor_metric: str | None = None


@dataclass(eq=False)
class EvaluationResult:
    """Structured task evaluation output.

    Parameters
    ----------
    metrics : dict[str, Any], optional
        Flat metric dictionary consumed by NAS, reporting, or other shared
        orchestration logic.
    artifacts : dict[str, Any] | None, optional
        Optional flat structured payload describing task-owned evaluation
        artifacts. This field is metadata-oriented and does not imply that
        files must be written during every evaluation call.
    predictions : Any | None, optional
        Optional predictions or prediction-like payloads needed by later
        task-owned analysis code.

    Attributes
    ----------
    metrics : dict[str, Any]
        Flat metric dictionary consumed by NAS, reporting, or other shared
        orchestration logic.
    artifacts : dict[str, Any] | None
        Optional flat structured payload describing task-owned evaluation
        artifacts. This field is metadata-oriented and does not imply that
        files must be written during every evaluation call.
    predictions : Any | None
        Optional predictions or prediction-like payloads needed by later
        task-owned analysis code.
    """

    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] | None = None
    predictions: Any | None = None


@dataclass(eq=False)
class TaskMetricContract:
    """Task-defined metric declaration shared with orchestration code.

    Parameters
    ----------
    available_metric_names : set[str], optional
        Metric names the task can produce.
    training_only_metric_names : set[str], optional
        Metric names available only after training has completed.
    nonnegative_metric_names : set[str], optional
        Task metric names guaranteed to stay nonnegative. Shared score and
        prune validation uses this set to interpret negative sentinels as
        unavailable values.
    primary_metric_names : set[str], optional
        Informational headline metrics exposed by the task. These names do not
        override runner-owned objective configuration.

    Attributes
    ----------
    available_metric_names : set[str]
        Metric names the task can produce.
    training_only_metric_names : set[str]
        Metric names available only after training has completed.
    nonnegative_metric_names : set[str]
        Task metric names guaranteed to stay nonnegative. Shared score and
        prune validation uses this set to interpret negative sentinels as
        unavailable values.
    primary_metric_names : set[str]
        Informational headline metrics exposed by the task. These names do not
        override runner-owned objective configuration.
    """

    available_metric_names: set[str] = field(default_factory=set)
    training_only_metric_names: set[str] = field(default_factory=set)
    nonnegative_metric_names: set[str] = field(default_factory=set)
    primary_metric_names: set[str] = field(default_factory=set)
