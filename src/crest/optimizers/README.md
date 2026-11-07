<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->

# Optimizer contracts

`optuna` is the default native optimizer: it requests a `NativeRound`, and Optuna
samples parameters inside the objective. Explicit optimizers return an
`ExplicitRound` of complete raw parameter mappings and provenance. Both run
through the same objective, feasibility checks, trial budget, and callbacks.

[`runner.py`](runner.py) validates an entire explicit batch before reserving any
trials, then evaluates it through Optuna. Candidates must match the active
[`search-space descriptor`](search_space.py) exactly. Reservations are individual
database writes; resume drains existing WAITING trials before requesting new work.
[`history.py`](history.py) supplies immutable proposal/history records.

[`study_state.py`](study_state.py) owns persisted optimizer identity, strict resume
checks, legacy adoption, and interrupted-trial recovery. Identity is checked before
initialization and persisted after successful initialization. Changing optimizer
settings on resume must not silently reinterpret an existing study.

[`context.py`](context.py) assembles bounded, deterministic semantic context for
explicit proposers. Its helpers in [`contexts/`](contexts/) cover
[data/task](contexts/data_task.py), [device](contexts/device.py),
[runtime](contexts/runtime.py), [STM32](contexts/stm32.py), and
[value normalization](contexts/values.py). Architectural hints are priors;
measured results take precedence. Native sampling skips descriptor/context setup.

See [`OptimizerABC`](../interfaces/optimizer.py), the round and budget types in
[`pipeline_types.py`](../pipeline_types.py), the
[model-family contract](../model_families/README.md), and the
[optimizer configuration reference](../../config/README.md#optimizer).
Progress reporting is shared across optimizers; see
[search utilities](../search/README.md).
