<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->

# Datasets

Dataset adapters are the entry point between local data artifacts and the shared CREST training/runtime pipeline.

No hardware is required to implement or test a dataset adapter.
Hardware is only needed later when a selected dataset/task/model combination is evaluated through a microcontroller backend.

For the broader source map, see [`../../README.md`](../../README.md).
For dataset preparation commands and local cache layout, see [`../../../data/dataset_download_and_splits/README.md`](../../../data/dataset_download_and_splits/README.md).

## What This Layer Owns

Datasets are responsible for the following.

- loading raw or cached data from `dataset.params`
- validating dataset-local configuration
- normalizing train, validation, test, and calibration splits into [`DatasetBundle`](../pipeline_types.py)
- publishing input shape, input dtype, and metadata used by tasks and model families
- selecting representative calibration data for export-time quantization.

Datasets do not own target semantics, model outputs, losses, scores, HIL requests, or board-specific staging.

## Shipped Implementations

- [`oxiod.py`](oxiod.py) Registered as `oxiod`.
  Wraps the OxIOD split loader, normalizes the odometry splits into `DatasetBundle`, and uses the training split for calibration unless `calibration_windows` specifies a cap.
- [`urbansound8k_mel.py`](urbansound8k_mel.py) Registered as `urbansound8k_mel`.
  Loads deterministic cached UrbanSound8K log-mel tensors, validates cache metadata, supports fixed-split and fold-rotation cache loading, and exposes audio metadata to the task/model layers.
- [`urbansound8k_common.py`](urbansound8k_common.py) Shared UrbanSound8K cache constants and split metadata used by the adapter and preparation path.
- [`ferplus_rafdb_mixed.py`](ferplus_rafdb_mixed.py) Registered as `ferplus_rafdb_mixed`.
  Verifies and loads the prepared native-resolution FER+/RAF-DB cache, preserves source and stable-ID metadata, performs the deployed nearest-floor grayscale preprocessing, and selects deterministic training-only calibration and representative fixtures.

## Key Files

- [`../interfaces/dataset.py`](../interfaces/dataset.py) Defines `DatasetABC`.
- [`../pipeline_types.py`](../pipeline_types.py) Defines `DatasetBundle` and `DataSplit`.
- [`../builtin_components.py`](../builtin_components.py) Registers shipped datasets under stable config names.
- [`../registry.py`](../registry.py) Owns `dataset_registry`.
- [`../component_selection.py`](../component_selection.py) Resolves `dataset.name` and `dataset.params` from config.
- [`../../config/README.md`](../../config/README.md) Documents the config surface that selects datasets.

## Runtime Flow

The dataset path runs before task and model-family construction.

1. Entry points register built-ins through [`../builtin_components.py`](../builtin_components.py).
2. [`../component_selection.py`](../component_selection.py) resolves `dataset.name` and `dataset.params`.
3. Runtime bootstrap instantiates the selected `DatasetABC` implementation.
4. `validate_config(...)` checks dataset-local configuration.
5. `load(...)` returns a normalized `DatasetBundle`.
6. The selected task consumes the bundle to build the target/output contract.
7. Export paths may call `make_calibration_data(...)` to select representative data for quantization.

## Add A New Dataset

Use [`oxiod.py`](oxiod.py) or [`urbansound8k_mel.py`](urbansound8k_mel.py) as the concrete starting point.

1. Add a new adapter module under [`src/crest/datasets/`](.).
2. Implement `DatasetABC` from [`../interfaces/dataset.py`](../interfaces/dataset.py).
3. Return a `DatasetBundle` with normalized `DataSplit` objects.
4. Validate all required `dataset.params` fields in `validate_config(...)`.
5. Keep dataset-specific preprocessing, cache validation, and split selection inside the dataset adapter or its preparation script.
6. Register the adapter in [`../builtin_components.py`](../builtin_components.py) with `dataset_registry.register(...)`.
7. Select it in config with `dataset.name` and put adapter-local knobs under `dataset.params`.
8. Add hardware-free tests for validation, loading, metadata, and calibration behavior.

Components are registered explicitly and are not discovered from files in this directory.
Dataset hooks receive `dataset.params`, rather than the root config.

## Contract In Practice

`DatasetABC` exposes these hooks.

- `name` Optional property.
  Returns the human-readable dataset identifier.
  Shipped adapters override it with the stable registry key.
- `load(dataset_config)` Required.
  Loads data and returns a `DatasetBundle` containing train, validation, test, optional calibration data, input shape, input dtype, and metadata.
- `validate_config(dataset_config)` Optional but expected for shipped adapters.
  Raise clear `ValueError`s for missing paths, invalid numeric parameters, or incompatible cache metadata.
- `make_calibration_data(bundle, dataset_config)` Optional.
  Returns representative data for export/quantization.
  Use this when the adapter needs capped, deterministic, or dataset-specific calibration behavior.

The bundle should be deterministic for the same local data and config.
If a dataset preparation step owns random crops or folds, document the seed and cache contract in the preparation README.

## Minimal Adapter Example

Run this hardware-free example from `src/` with NumPy installed.
It shows the bundle and calibration contracts without downloading a dataset.
The synthetic splits are for adapter bring-up.
Real experiments need separately prepared, representative partitions.

```python
import numpy as np

from crest.interfaces import DatasetABC
from crest.pipeline_types import DataSplit, DatasetBundle

class TinyDataset(DatasetABC):
    @property
    def name(self):
        return "tiny"

    def validate_config(self, dataset_config):
        if dataset_config.get("samples", 8) < 2:
            raise ValueError("samples must be at least 2")

    def load(self, dataset_config):
        self.validate_config(dataset_config)
        count = dataset_config.get("samples", 8)
        inputs = np.arange(count * 3, dtype=np.float32).reshape(count, 3)
        targets = inputs.sum(axis=1, keepdims=True)
        train = DataSplit(inputs[:-1], targets[:-1])
        val = DataSplit(inputs[-1:], targets[-1:])
        return DatasetBundle(
            train=train,
            val=val,
            calibration=DataSplit(inputs[:2], targets[:2]),
            input_shape=(3,),
            input_dtype="float32",
        )

adapter = TinyDataset()
dataset_params = {"samples": 8}
bundle = adapter.load(dataset_params)
assert bundle.train.inputs.shape == (7, 3)
assert adapter.make_calibration_data(bundle, dataset_params).inputs.shape == (2, 3)
```

After registering `TinyDataset` as `tiny` in `builtin_components.py`, select it with this fragment.

```yaml
dataset:
  name: tiny
  params:
    samples: 8
```

This example supplies scalar regression targets.
Select or implement a compatible task before using it in a complete pipeline.

## Definition Of Done

Verify the adapter with hardware-free success and representative failure cases.

- Missing paths, invalid numeric parameters, incompatible cache schemas, and unknown split names fail with clear errors.
- Repeated loads return the same splits, input shape/dtype, and task-required metadata.
- Calibration tests confirm the intended split, cap, and deterministic selection.
- A config using the registered `dataset.name` resolves the adapter and loads its bundle.
- Preparation/cache documentation and NumPy-style docstrings match the implementation.

## Troubleshooting

If dataset bring-up fails, check these layers in order.

1. Preparation.
   Does the raw dataset or generated cache exist at the configured path?
2. Config.
   Does `dataset.name` match the registered key and do all `dataset.params` fields match the adapter contract?
3. Splits.
   Are train, validation, test, and calibration splits present and nonempty?
4. Metadata.
   Does the adapter publish the input shape/dtype and task-required metadata?
5. Calibration.
   Does `make_calibration_data(...)` return the intended split for export?
