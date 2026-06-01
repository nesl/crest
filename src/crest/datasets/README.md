<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->

# Datasets

Dataset adapters are the entry point between local data artifacts and the
shared CREST training/runtime pipeline.

Hardware requirement: no hardware is required to implement or test a dataset
adapter. Hardware is only needed later when a selected dataset/task/model
combination is evaluated through a microcontroller backend.

For the broader source map, see [`../../README.md`](../../README.md). For
dataset preparation commands and local cache layout, see
[`../../../data/dataset_download_and_splits/README.md`](../../../data/dataset_download_and_splits/README.md).

## What This Layer Owns

Datasets own:

- loading raw or cached data from `dataset.params`;
- validating dataset-local configuration;
- normalizing train, validation, test, and calibration splits into
  [`DatasetBundle`](../pipeline_types.py);
- publishing input shape, input dtype, and metadata used by tasks and model
  families;
- selecting representative calibration data for export-time quantization.

Datasets do not own target semantics, model outputs, losses, scores, HIL
requests, or board-specific staging.

## Shipped Implementations

- [`oxiod.py`](oxiod.py)
  Registered as `oxiod`. Wraps the legacy OxIOD split loader, normalizes the
  odometry splits into `DatasetBundle`, and preserves the current calibration
  fallback behavior used by the modular pipeline.
- [`urbansound8k_mel.py`](urbansound8k_mel.py)
  Registered as `urbansound8k_mel`. Loads deterministic cached UrbanSound8K
  log-mel tensors, validates cache metadata, supports fixed-split and
  fold-rotation cache loading, and exposes audio metadata to the task/model
  layers.
- [`urbansound8k_common.py`](urbansound8k_common.py)
  Shared UrbanSound8K cache constants and split metadata used by the adapter
  and preparation path.

## Key Files

- [`../interfaces.py`](../interfaces.py)
  Defines `DatasetABC`.
- [`../pipeline_types.py`](../pipeline_types.py)
  Defines `DatasetBundle` and `DataSplit`.
- [`../builtin_components.py`](../builtin_components.py)
  Registers shipped datasets under stable config names.
- [`../registry.py`](../registry.py)
  Owns `dataset_registry`.
- [`../component_selection.py`](../component_selection.py)
  Resolves `dataset.name` and `dataset.params` from config.
- [`../../config/README.md`](../../config/README.md)
  Documents the config surface that selects datasets.

## Runtime Flow

The dataset path runs before task and model-family construction:

1. Entry points register built-ins through
   [`../builtin_components.py`](../builtin_components.py).
2. [`../component_selection.py`](../component_selection.py) resolves
   `dataset.name` and `dataset.params`.
3. Runtime bootstrap instantiates the selected `DatasetABC` implementation.
4. `validate_config(...)` checks dataset-local configuration.
5. `load(...)` returns a normalized `DatasetBundle`.
6. The selected task consumes the bundle to build the target/output contract.
7. Export paths may call `make_calibration_data(...)` to select representative
   data for quantization.

## Add A New Dataset

Use [`oxiod.py`](oxiod.py) or [`urbansound8k_mel.py`](urbansound8k_mel.py) as
the concrete starting point.

1. Add a new adapter module under [`src/crest/datasets/`](.).
2. Implement `DatasetABC` from [`../interfaces.py`](../interfaces.py).
3. Return a `DatasetBundle` with normalized `DataSplit` objects.
4. Validate all required `dataset.params` fields in `validate_config(...)`.
5. Keep dataset-specific preprocessing, cache validation, and split selection
   inside the dataset adapter or its preparation script.
6. Register the adapter in [`../builtin_components.py`](../builtin_components.py)
   with `dataset_registry.register(...)`.
7. Select it in config with `dataset.name` and put adapter-local knobs under
   `dataset.params`.
8. Add hardware-free tests for validation, loading, metadata, and calibration
   behavior.

## Contract In Practice

`DatasetABC` exposes three hooks:

- `name`
  Optional property. Returns the human-readable dataset identifier; shipped
  adapters override it with the stable registry key.
- `load(dataset_config)`
  Required. Loads data and returns a `DatasetBundle` containing train,
  validation, test, optional calibration data, input shape, input dtype, and
  metadata.
- `validate_config(dataset_config)`
  Optional but expected for shipped adapters. Raise clear `ValueError`s for
  missing paths, invalid numeric parameters, or incompatible cache metadata.
- `make_calibration_data(bundle, dataset_config)`
  Optional. Returns representative data for export/quantization. Use this when
  the adapter needs capped, deterministic, or dataset-specific calibration
  behavior.

The bundle should be deterministic for the same local data and config. If a
dataset preparation step owns random crops or folds, document the seed and cache
contract in the preparation README.

## Registration And Config Selection

Registration is explicit. Dataset adapters are not auto-discovered from this
directory.

The shipped pattern is:

1. Define the adapter class in this package.
2. Import it in [`../builtin_components.py`](../builtin_components.py).
3. Register it under a stable key with `dataset_registry.register(...)`.
4. Select that key through `dataset.name`.
5. Put adapter-owned options under `dataset.params`.

## Contributor Checklist

- Keep target/output semantics in the task layer, not in the dataset adapter.
- Keep model architecture assumptions out of dataset loading.
- Validate paths, numeric parameters, cache schemas, and split names locally.
- Preserve deterministic split and calibration behavior.
- Add or update NumPy-style docstrings for changed functions, classes,
  dataclasses, and tests.
- Add hardware-free tests before validating any downstream HIL flow.

## Definition Of Done

A dataset adapter is ready when:

1. `validate_config(...)` catches missing or invalid dataset parameters.
2. `load(...)` returns deterministic `DatasetBundle` and `DataSplit` objects.
3. Metadata includes the fields needed by the paired task/model-family path.
4. Calibration behavior is explicit and tested.
5. The adapter is registered and selectable with `dataset.name`.
6. The README/config docs describe any required preparation or cache layout.
7. Hardware-free tests cover success and representative failure paths.

## Troubleshooting

If dataset bring-up fails, check these layers in order:

1. Preparation:
   Does the raw dataset or generated cache exist at the configured path?
2. Config:
   Does `dataset.name` match the registered key and do all `dataset.params`
   fields match the adapter contract?
3. Splits:
   Are train, validation, test, and calibration splits present and nonempty?
4. Metadata:
   Does the adapter publish the input shape/dtype and task-required metadata?
5. Calibration:
   Does `make_calibration_data(...)` return the intended split for export?
