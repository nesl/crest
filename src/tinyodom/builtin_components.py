"""Built-in component registration for the modular TinyODOM pipeline.

This module centralizes registration of the repository-owned dataset, task,
and model-family implementations that ship with TinyODOM.
"""

from __future__ import annotations

from .datasets.oxiod import OxIODDataset
from .datasets.urbansound8k_mel import UrbanSound8KMelDataset
from .model_families.audio_dscnn import AudioDSCNNFamily
from .model_families.odom_tcn import OdomTCNFamily
from .registry import dataset_registry, model_family_registry, task_registry
from .tasks.odometry_regression import OdometryRegressionTask
from .tasks.sound_classification import SoundClassificationTask


def ensure_builtin_components_registered() -> None:
    """Register the built-in TinyODOM components exactly once.

    Returns
    -------
    None
        The function mutates the module-level registries only when a built-in
        component has not already been registered by name. Existing registry
        entries are left untouched so repeated calls are idempotent.
    """

    if "oxiod" not in dataset_registry:
        dataset_registry.register("oxiod", OxIODDataset)
    if "urbansound8k_mel" not in dataset_registry:
        dataset_registry.register("urbansound8k_mel", UrbanSound8KMelDataset)
    if "odometry_regression" not in task_registry:
        task_registry.register("odometry_regression", OdometryRegressionTask)
    if "sound_classification" not in task_registry:
        task_registry.register("sound_classification", SoundClassificationTask)
    if "odom_tcn" not in model_family_registry:
        model_family_registry.register("odom_tcn", OdomTCNFamily)
    if "audio_dscnn" not in model_family_registry:
        model_family_registry.register("audio_dscnn", AudioDSCNNFamily)
