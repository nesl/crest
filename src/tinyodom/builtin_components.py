"""Built-in component registration for the modular TinyODOM pipeline.

This module centralizes registration of the repository-owned dataset, task,
and model-family implementations that ship with TinyODOM.
"""

from __future__ import annotations

from .registry import dataset_registry, model_family_registry, task_registry


def ensure_audio_components_registered() -> None:
    """Register only the built-in audio components.

    Returns
    -------
    None
        Registers the UrbanSound8K mel dataset, sound-classification task, and
        audio DS-CNN family if they are not already present.

    Notes
    -----
    This intentionally avoids importing OxIOD so audio-only preparation and
    smoke paths do not pull in unrelated pandas/giotto dependencies.
    """

    if "urbansound8k_mel" not in dataset_registry:
        from .datasets.urbansound8k_mel import UrbanSound8KMelDataset

        dataset_registry.register("urbansound8k_mel", UrbanSound8KMelDataset)
    if "sound_classification" not in task_registry:
        from .tasks.sound_classification import SoundClassificationTask

        task_registry.register("sound_classification", SoundClassificationTask)
    if "audio_dscnn" not in model_family_registry:
        from .model_families.audio_dscnn import AudioDSCNNFamily

        model_family_registry.register("audio_dscnn", AudioDSCNNFamily)


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
        from .datasets.oxiod import OxIODDataset

        dataset_registry.register("oxiod", OxIODDataset)
    if "odometry_regression" not in task_registry:
        from .tasks.odometry_regression import OdometryRegressionTask

        task_registry.register("odometry_regression", OdometryRegressionTask)
    if "odom_tcn" not in model_family_registry:
        from .model_families.odom_tcn import OdomTCNFamily

        model_family_registry.register("odom_tcn", OdomTCNFamily)
    ensure_audio_components_registered()
