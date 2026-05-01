"""Shared runtime bootstrap helpers for modular dataset/task/model-family wiring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .component_selection import cfg_get, resolve_component_selection
from .model import validate_nas_policy_for_task
from .pipeline_types import DatasetBundle, ModelBuildContext, TargetSpec, TaskMetricContract
from .registry import dataset_registry, model_family_registry, task_registry


@dataclass(frozen=True)
class BootstrappedPipeline:
    """Resolved modular runtime state for one config.

    Parameters
    ----------
    selection : dict[str, Any]
        Resolved component names and local config payloads.
    dataset : Any
        Instantiated dataset component.
    task : Any
        Instantiated task component.
    model_family : Any
        Instantiated model-family component.
    bundle : DatasetBundle
        Loaded dataset bundle.
    target_spec : TargetSpec
        Task-owned target specification.
    metric_contract : TaskMetricContract
        Task-owned metric declaration.
    model_build_context : ModelBuildContext
        Normalized build context shared with the model family.
    """

    selection: dict[str, Any]
    dataset: Any
    task: Any
    model_family: Any
    bundle: DatasetBundle
    target_spec: TargetSpec
    metric_contract: TaskMetricContract
    model_build_context: ModelBuildContext


def instantiate_task_component(
    task_name: str,
    config: Any,
    task_config: Any,
    *,
    checkpoint_path: Path | None = None,
    early_stopping_patience: int | None = None,
) -> Any:
    """Instantiate one task using the explicit runtime constructor contract.

    Parameters
    ----------
    task_name : str
        Registered task name.
    config : Any
        Loaded runtime configuration.
    task_config : Any
        Task-local configuration subtree.
    checkpoint_path : Path | None, optional
        Explicit checkpoint path override.
    early_stopping_patience : int | None, optional
        Explicit early-stopping patience override.

    Returns
    -------
    Any
        Instantiated task component.
    """

    task_cls = task_registry.get(task_name)
    resolved_checkpoint_path = (
        Path(config.outputs.checkpoint_path)
        if checkpoint_path is None
        else Path(checkpoint_path)
    )
    resolved_patience = int(
        cfg_get(task_config, "early_stopping_patience", 40)
        if early_stopping_patience is None
        else early_stopping_patience
    )
    return task_cls(
        checkpoint_path=resolved_checkpoint_path,
        early_stopping_patience=resolved_patience,
    )


def build_model_context(
    bundle: DatasetBundle,
    target_spec: TargetSpec,
) -> ModelBuildContext:
    """Build the normalized model-family context for one dataset/task pair.

    Parameters
    ----------
    bundle : DatasetBundle
        Dataset bundle backing the active run.
    target_spec : TargetSpec
        Task-owned target specification.

    Returns
    -------
    ModelBuildContext
        Normalized build context for model-family operations.
    """

    return ModelBuildContext(
        input_shape=bundle.input_shape,
        input_dtype=bundle.input_dtype,
        target_spec=target_spec,
        dataset_metadata=dict(bundle.metadata),
        task_metadata=dict(target_spec.metadata),
    )


def bootstrap_pipeline(
    config: Any,
    *,
    dataset: Any | None = None,
    bundle: DatasetBundle | None = None,
    checkpoint_path: Path | None = None,
    early_stopping_patience: int | None = None,
) -> BootstrappedPipeline:
    """Bootstrap the modular dataset/task/model-family pipeline for one config.

    Parameters
    ----------
    config : Any
        Loaded runtime configuration.
    dataset : Any | None, optional
        Pre-instantiated dataset component to reuse.
    bundle : DatasetBundle | None, optional
        Preloaded dataset bundle to reuse instead of reloading the dataset.
    checkpoint_path : Path | None, optional
        Explicit checkpoint path override for task construction.
    early_stopping_patience : int | None, optional
        Explicit early-stopping patience override for task construction.

    Returns
    -------
    BootstrappedPipeline
        Fully resolved modular runtime state, including task metric
        validation for the active NAS policy.
    """

    selection = resolve_component_selection(config)
    dataset_cls = dataset_registry.get(selection["dataset_name"])
    if dataset is None:
        dataset = dataset_cls()
    dataset.validate_config(selection["dataset_config"])
    if bundle is None:
        bundle = dataset.load(selection["dataset_config"])

    task = instantiate_task_component(
        selection["task_name"],
        config,
        selection["task_config"],
        checkpoint_path=checkpoint_path,
        early_stopping_patience=early_stopping_patience,
    )
    model_family_cls = model_family_registry.get(selection["model_family_name"])
    model_family = model_family_cls()

    task.validate_config(selection["task_config"])
    model_family.validate_config(selection["model_config"])

    target_spec = task.build_target_spec(bundle, selection["task_config"])
    metric_contract = task.metric_contract(target_spec, selection["task_config"])
    validate_nas_policy_for_task(
        config,
        task_metric_names=metric_contract.available_metric_names,
        training_only_task_metric_names=metric_contract.training_only_metric_names,
    )
    model_build_context = build_model_context(bundle, target_spec)
    return BootstrappedPipeline(
        selection=selection,
        dataset=dataset,
        task=task,
        model_family=model_family,
        bundle=bundle,
        target_spec=target_spec,
        metric_contract=metric_contract,
        model_build_context=model_build_context,
    )
