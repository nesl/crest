<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->

# CREST Python package

`crest` implements the reusable pipeline behind the [source entry points](../README.md).
Install and configure the environment using the [repository guide](../../README.md).
Package exports are lazy; importing `crest` alone does not initialize TensorFlow or hardware.

| Area | Responsibility |
| --- | --- |
| [`interfaces.py`](interfaces.py), [`pipeline_types.py`](pipeline_types.py) | Component contracts and shared request/result payloads |
| [`registry.py`](registry.py), [`builtin_components.py`](builtin_components.py), [`component_selection.py`](component_selection.py) | Component registration and config-based selection |
| [`datasets`](datasets/README.md), [`tasks`](tasks/README.md), [`model_families`](model_families/README.md) | Data loading, task-specific training/evaluation, and architectures |
| [`optimizers`](optimizers/README.md) | Native and explicit candidate proposals, descriptors, and history |
| [`hardware.py`](hardware.py), [`hil`](hil/README.md), [`microcontrollers`](microcontrollers/README.md) | Compile/deploy coordination, serial protocol, and device backends |
| [`search`](search/README.md) | Progress artifacts and training randomness policy |
| [`model.py`](model.py), [`model_metrics.py`](model_metrics.py), [`data.py`](data.py) | Shared model, metric, and data utilities |
| [`runtime_bootstrap.py`](runtime_bootstrap.py) | Task-aware component setup, model context, and pipeline validation |

Use canonical imports such as `from crest.pipeline_types import ModelBuildContext`.
The orchestration entry point is [`nas_model_client.py`](../nas_model_client.py).
Configuration examples and supported settings live in [`config`](../config/README.md).
Heavy dependencies are declared in [`pyproject.toml`](../../pyproject.toml); individual
runtime modules may require TensorFlow, Optuna, serial libraries, or board toolchains.

For contributions, keep dataset/task/family/device responsibilities in their component
layers and extend the matching tests in [`test`](../../test). Run pytest from the
repository root; its configuration places this checkout's `src` on the import path.
Hardware execution requires the configured target and, for harness measurements, the
measurement harness; pure contract and configuration checks do not.
