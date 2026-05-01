# Source Guide

This directory contains the Python entry points and core library code for the
TinyODOM-EX training, hardware-in-the-loop, and deployment flow.

This guide maps the training, HIL, and extension surfaces under `src/`.

Related docs:

- Repository setup, dataset preparation, and operator workflow live in
  [`../README.md`](../README.md).
- Config block meanings, NAS policy shape, and runtime config caveats live in
  [`config/README.md`](config/README.md).
- Microcontroller and hardware-backend bring-up live in
  [`tinyodom/microcontrollers/README.md`](tinyodom/microcontrollers/README.md).
- Known rough edges from the component/model-family refactor are tracked in
  [`../things_forgotten_in_the_model_refactor.md`](../things_forgotten_in_the_model_refactor.md).

## Top-Level Entry Points

- [`nas_model_client.py`](nas_model_client.py)
  Runs NAS, training, scoring, artifact export, and the client side of the
  hardware-in-the-loop workflow.
- [`hil_server.py`](hil_server.py)
  Runs the ZeroMQ HIL server that materializes models, stages device-specific
  candidates, and returns compile/runtime metrics.
- [`config/nas_config.yaml`](config/nas_config.yaml)
  Default runtime configuration for device selection, dataset parameters,
  training controls, NAS scoring/pruning, outputs, logging, and network
  settings.

## Package Map

The `tinyodom/` package holds the reusable implementation behind the entry
points above.

- [`tinyodom/component_selection.py`](tinyodom/component_selection.py)
  Resolves the active dataset, task, and model-family selections from config.
- [`tinyodom/registry.py`](tinyodom/registry.py)
  Defines the string-keyed registries for datasets, tasks, and model families.
- [`tinyodom/builtin_components.py`](tinyodom/builtin_components.py)
  Registers the built-in dataset, task, and model-family implementations.
- [`tinyodom/interfaces.py`](tinyodom/interfaces.py)
  Defines the core abstraction contracts:
  `DatasetABC`, `TaskABC`, and `ModelFamilyABC`.
- [`tinyodom/pipeline_types.py`](tinyodom/pipeline_types.py)
  Defines shared typed payloads passed between the modular pipeline layers.
- [`tinyodom/datasets/`](tinyodom/datasets)
  Dataset adapters. Today this repo ships the built-in
  [`oxiod.py`](tinyodom/datasets/oxiod.py) adapter.
- [`tinyodom/tasks/`](tinyodom/tasks)
  Task adapters. Today this repo ships the built-in
  [`odometry_regression.py`](tinyodom/tasks/odometry_regression.py) task.
- [`tinyodom/model_families/`](tinyodom/model_families)
  Model-family implementations. Today this repo ships the built-in
  [`tinyodom_tcn.py`](tinyodom/model_families/tinyodom_tcn.py) family.
- [`tinyodom/microcontrollers/`](tinyodom/microcontrollers)
  Hardware backends and backend registry/factory logic. See
  [`tinyodom/microcontrollers/README.md`](tinyodom/microcontrollers/README.md)
  for bring-up details.
- [`tinyodom/model.py`](tinyodom/model.py)
  Shared runtime helpers for config loading, score evaluation, and generic
  metric normalization.
- [`tinyodom/runtime_bootstrap.py`](tinyodom/runtime_bootstrap.py)
  Shared task-aware bootstrap path used by both `nas_model_client.py` and
  `hil_server.py` to resolve component selection, instantiate dataset/task/
  family components, and validate NAS policy against the active task contract.
- [`tinyodom/hil_runtime.py`](tinyodom/hil_runtime.py)
  Runtime-owned HIL request construction and metric collection helpers used by
  the HIL server and related tests.
- [`tinyodom/devices.py`](tinyodom/devices.py)
  Shared device dataclasses and the `DeviceInterface` contract used by
  hardware backends.
- [`tinyodom/hardware.py`](tinyodom/hardware.py)
  Legacy/shared hardware utility layer used by the current HIL/NAS flow.

## Shared Abstractions

The modular runtime is built around three main interfaces plus a small set of
typed payloads shared across orchestration code.

- `DatasetABC` in [`tinyodom/interfaces.py`](tinyodom/interfaces.py)
  Loads raw data and returns a normalized [`DatasetBundle`](tinyodom/pipeline_types.py)
  with train/validation/test/calibration splits plus dataset metadata.
- `TaskABC` in [`tinyodom/interfaces.py`](tinyodom/interfaces.py)
  Defines the target/output contract, task-owned fitting behavior, evaluation
  behavior, and the metric contract that shared NAS/scoring code consumes.
- `ModelFamilyABC` in [`tinyodom/interfaces.py`](tinyodom/interfaces.py)
  Samples hyperparameters, builds models, validates family-local config, and
  materializes the export variant passed into HIL.
- Shared typed payloads in [`tinyodom/pipeline_types.py`](tinyodom/pipeline_types.py)
  carry the normalized information exchanged between those layers:
  `DatasetBundle`, `TargetSpec`, `ModelBuildContext`, `FitPlan`,
  `EvaluationResult`, and `TaskMetricContract`.

## End-to-End Flow

At a high level, the source tree is wired like this:

1. `nas_model_client.py` or `hil_server.py` loads
   [`config/nas_config.yaml`](config/nas_config.yaml) through shared helpers in
   [`tinyodom/model.py`](tinyodom/model.py).
2. The entry point calls
   `ensure_builtin_components_registered()` from
   [`tinyodom/builtin_components.py`](tinyodom/builtin_components.py) so the
   default dataset, task, and model family are available by name.
3. The entry point runs the shared bootstrap in
   [`tinyodom/runtime_bootstrap.py`](tinyodom/runtime_bootstrap.py), which
   resolves component selection, instantiates the selected dataset/task/model
   family, derives the target spec, and validates `nas.score` / `nas.prune`
   against the task metric contract.
4. The selected dataset adapter loads data and produces a normalized
   `DatasetBundle`.
5. The selected task adapter builds the target contract and training/evaluation
   behavior.
6. The selected model family samples hyperparameters, builds models, and
   materializes export variants.
7. When hardware metrics are needed, the HIL path builds a normalized request
   through [`tinyodom/hil_runtime.py`](tinyodom/hil_runtime.py), then the
   selected microcontroller backend stages, compiles, uploads, and measures
   one candidate.
8. Shared scoring, pruning, and result-shaping code combines task metrics and
   backend metrics into the values used by NAS and reporting.

That split is important:

- Dataset/task/model-family code owns the ML-side behavior.
- Microcontroller backend code owns the build/upload/runtime measurement path.
- The top-level scripts own orchestration and policy.

NAS and HIL both use the same component-selection path. The same config-driven
dataset/task/model-family selection is resolved before the code branches into
training/NAS behavior or board/backend behavior.

## Component Selection

The current modular selection surface is resolved in
[`tinyodom/component_selection.py`](tinyodom/component_selection.py).

The main config knobs are:

- `dataset.name`
  Selects the dataset adapter.
- `dataset.params`
  Required dataset-local config block.
- `task.name`
  Selects the task adapter.
- `task.params`
  Optional task-local config block.
- `model.family`
  Selects the model family.
- `model.params`
  Model-family-local configuration.
- `model.search`
  Model-family-local search-space configuration.

`dataset`, `task`, and `model` are required blocks. The older top-level
`data` fallback is not part of the supported config contract anymore.

See [`config/README.md`](config/README.md) for the current shipped config shape.

## Registration Model

TinyODOM-EX does not currently auto-discover components from the filesystem.
The registry model is explicit:

- [`tinyodom/registry.py`](tinyodom/registry.py) defines
  `dataset_registry`, `task_registry`, and `model_family_registry`.
- [`tinyodom/builtin_components.py`](tinyodom/builtin_components.py) registers
  the built-in components under their stable string keys.
- The entry points call `ensure_builtin_components_registered()` before they
  resolve component names from config.

If you add a new dataset, task, or model family, it is not available until
something registers it under the name you intend to use in config.

Known extension limits and refactor leftovers are tracked in
[`../things_forgotten_in_the_model_refactor.md`](../things_forgotten_in_the_model_refactor.md).

## Shared Scoring And Trial Outputs

The shared score/logging path no longer assumes one fixed TinyODOM model shape
or one fixed metric schema.

### Task Metric Declaration

[`TaskMetricContract`](tinyodom/pipeline_types.py) is the task-owned
declaration shared with orchestration code. It tells the shared pipeline:

- which metrics the task can produce
- which metrics only exist after full training
- which metrics are guaranteed nonnegative
- which metrics are the task's primary headline metrics

This is how the task layer advertises metrics such as `rmse_total` without
hardcoding them into the shared NAS/logging layer.

### Task-Owned Fit And Closeout Hooks

`TaskABC` now owns both fit-plan construction and task-specific closeout hooks.

- `build_fit_plan(...)`
  Returns the `FitPlan` used by NAS-time training and final retraining,
  including the `combine_train_val=True` path.
- `history_component_keys(...)`
  Advertises which per-output training curves should be plotted for the active
  task.
- `generate_closeout_artifacts(...)`
  Produces task-specific closeout artifacts without forcing generic NAS code to
  assume odometry trajectory reporting.

### Score Resolution

[`ScoreEvaluationResult`](tinyodom/model.py) is the shared representation of
the resolved score/objective values for one trial.

- Scalar runs set `score` and leave the objective lists as the configured
  single-objective projection.
- Multi-objective runs leave `score` as `None` and instead carry ordered
  objective names, values, and directions.

### Generic Trial Outcome

[`TrialOutcome`](tinyodom/model.py) is the generic payload written to CSV and
mirrored into Optuna trial attributes. It carries:

- resolved scalar/objective values
- `task_metrics`
- resolved trial `hyperparams`
- optional task-owned `artifact_summary`

The shared logging code does not need to know which task or model family
produced those values.

### CSV And Optuna Logging

[`log_trial(...)`](tinyodom/model.py) writes a stable infrastructure column set
plus dynamic task/hyperparameter columns.

Stable shared columns include:

- study/timestamp fields
- shared hardware metrics such as RAM, flash, latency, power, energy, and
  error codes
- score/objective metadata (`score_type`, `objective_*_json`)
- pruning metadata
- `artifact_summary_json`
- cadenced runtime telemetry fields when present

Dynamic columns are added per trial outcome:

- task metrics become `metric__{name}`
- hyperparameters become `hparam__{name}`

The same information is also mirrored into `trial.user_attrs`, including the
fully expanded `metric__*` and `hparam__*` keys.

## Extension Paths

### Add A New Model Family

If you mean a new trainable/exportable model family, this is the primary
extension path in the current codebase.

Key files:

- [`tinyodom/interfaces.py`](tinyodom/interfaces.py)
  `ModelFamilyABC` defines the contract.
- [`tinyodom/model_families/tinyodom_tcn.py`](tinyodom/model_families/tinyodom_tcn.py)
  Concrete example of the built-in family.
- [`tinyodom/builtin_components.py`](tinyodom/builtin_components.py)
  Built-in registration.
- [`tinyodom/component_selection.py`](tinyodom/component_selection.py)
  Explicit config selection for dataset, task, and model-family components.
- [`hil_server.py`](hil_server.py) and
  [`nas_model_client.py`](nas_model_client.py)
  Entry-point orchestration that consumes the selected family.

Typical steps:

1. Add a new module under [`tinyodom/model_families/`](tinyodom/model_families).
2. Implement a `ModelFamilyABC` subclass.
3. Register it in the appropriate registry path.
   The built-in pattern today is
   [`tinyodom/builtin_components.py`](tinyodom/builtin_components.py).
4. Set `model.family` in config to the registered name.
5. Put family-local knobs under `model.params` and `model.search` when needed.
6. Verify export/materialization semantics if the family needs custom model
   loading, custom objects, or variant handling.

Important boundary:

- Model families own model construction and export-oriented materialization.
- Hardware backends own candidate staging, compile, upload, and runtime
  measurement.

### Add A New Dataset

Key files:

- [`tinyodom/interfaces.py`](tinyodom/interfaces.py) for `DatasetABC`
- [`tinyodom/datasets/oxiod.py`](tinyodom/datasets/oxiod.py) as the built-in
  example
- [`tinyodom/builtin_components.py`](tinyodom/builtin_components.py) for the
  current registration pattern

Typical steps:

1. Add a new dataset adapter under [`tinyodom/datasets/`](tinyodom/datasets).
2. Implement `DatasetABC`.
3. Register it under a stable string key.
4. Select it with `dataset.name`.
5. Put dataset-local knobs under `dataset.params`.

### Add A New Task

Key files:

- [`tinyodom/interfaces.py`](tinyodom/interfaces.py) for `TaskABC`
- [`tinyodom/tasks/odometry_regression.py`](tinyodom/tasks/odometry_regression.py)
  as the built-in example
- [`tinyodom/builtin_components.py`](tinyodom/builtin_components.py) for the
  current registration pattern

Typical steps:

1. Add a new task adapter under [`tinyodom/tasks/`](tinyodom/tasks).
2. Implement `TaskABC`.
3. Register it under a stable string key.
4. Select it with `task.name`.
5. Keep task-owned training, evaluation, and metric-contract logic inside the
   task adapter.

### Add A New Microcontroller Or Hardware Backend

Key files:

- [`tinyodom/microcontrollers/README.md`](tinyodom/microcontrollers/README.md)
  for the bring-up guide
- [`tinyodom/devices.py`](tinyodom/devices.py) for `DeviceInterface`
- [`tinyodom/microcontrollers/__init__.py`](tinyodom/microcontrollers/__init__.py)
  for backend registration/factory plumbing

That path is intentionally separate from model-family work. Board bring-up,
toolchain integration, upload flow, runtime telemetry, and backend-owned
device options belong to the microcontroller backend layer, not to
`ModelFamilyABC`.

Important caveat:

- a new non-Arduino backend must be wired into the registry metadata in
  [`tinyodom/microcontrollers/__init__.py`](tinyodom/microcontrollers/__init__.py)
  or it is unreachable except through the Arduino FQBN fallback path

## NAS Policy And HIL Details

For the current scoring, pruning, and runtime knobs, use:

- [`config/README.md`](config/README.md) for the config reference
- [`config/nas_config.yaml`](config/nas_config.yaml) for the default config shape
- [`tinyodom/model.py`](tinyodom/model.py) for score/prune evaluation and HIL
  request construction
- [`hil_server.py`](hil_server.py) for the HIL-side request handling and
  backend failure shaping
