"""Built-in component registration for the modular TinyODOM pipeline."""

from __future__ import annotations

from .datasets.oxiod import OxIODDataset
from .model_families.tinyodom_tcn import TinyOdomTCNFamily
from .registry import dataset_registry, model_family_registry, task_registry
from .tasks.odometry_regression import OdometryRegressionTask


def ensure_builtin_components_registered() -> None:
    """Register the built-in TinyODOM components exactly once.

    Returns
    -------
    None
        The function mutates the module-level registries only when a built-in
        component has not already been registered by name.
    """

    if "oxiod" not in dataset_registry:
        dataset_registry.register("oxiod", OxIODDataset)
    if "odometry_regression" not in task_registry:
        task_registry.register("odometry_regression", OdometryRegressionTask)
    if "tinyodom_tcn" not in model_family_registry:
        model_family_registry.register("tinyodom_tcn", TinyOdomTCNFamily)
