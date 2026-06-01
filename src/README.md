# Source Guide

This directory contains the Python entry points and core library code for the
CREST training, hardware-in-the-loop, and deployment flow.

This guide maps the training, HIL, and extension surfaces under `src/`.

Related docs:

- Repository setup, dataset preparation, and operator workflow live in
  [`../README.md`](../README.md).
- Config block meanings, NAS policy shape, and runtime config caveats live in
  [`config/README.md`](config/README.md).
- Microcontroller and hardware-backend bring-up live in
  [`crest/microcontrollers/README.md`](crest/microcontrollers/README.md).

## Where To Start

| Goal | Start Here | Hardware Requirement |
|------|------------|----------------------|
| Run or modify NAS/training orchestration | [`nas_model_client.py`](nas_model_client.py) and [`crest/runtime_bootstrap.py`](crest/runtime_bootstrap.py) | No hardware required when `device.hil: false` |
| Run or modify HIL server behavior | [`hil_server.py`](hil_server.py) and [`crest/hil_runtime.py`](crest/hil_runtime.py) | Development board required for execution; HIL harness required for energy measurement |
| Add a dataset | [`crest/datasets/README.md`](crest/datasets/README.md) | No hardware required for adapter development |
| Add a task | [`crest/tasks/README.md`](crest/tasks/README.md) | No hardware required for adapter development |
| Add a model family | [`crest/model_families/README.md`](crest/model_families/README.md) | No hardware required for model construction; hardware needed for backend validation |
| Add board support | [`crest/microcontrollers/README.md`](crest/microcontrollers/README.md) | Matching board and toolchain required |

## Top-Level Entry Points

- [`nas_model_client.py`](nas_model_client.py)
  Runs NAS, training, scoring, artifact export, and the client side of the
  hardware-in-the-loop workflow.
- [`hil_server.py`](hil_server.py)
  Runs the ZeroMQ HIL server that materializes models, stages device-specific
  candidates, and returns compile/runtime metrics.
- [`pareto_hil_replay.py`](pareto_hil_replay.py)
  Replays already-logged Pareto-front NAS candidates through a target HIL
  config and writes replay artifacts.

## Package Map

The `crest/` package holds the reusable implementation behind the entry
points above.

- [`crest/component_selection.py`](crest/component_selection.py)
  Resolves the active dataset, task, and model-family selections from config.
- [`crest/registry.py`](crest/registry.py)
  Defines the string-keyed registries for datasets, tasks, and model families.
- [`crest/builtin_components.py`](crest/builtin_components.py)
  Registers the built-in dataset, task, and model-family implementations.
- [`crest/interfaces.py`](crest/interfaces.py)
  Defines the core abstraction contracts:
  `DatasetABC`, `TaskABC`, and `ModelFamilyABC`.
- [`crest/pipeline_types.py`](crest/pipeline_types.py)
  Defines shared typed payloads passed between the modular pipeline layers.
- [`crest/datasets/`](crest/datasets)
  Dataset adapters. Built-ins include [`oxiod.py`](crest/datasets/oxiod.py)
  and [`urbansound8k_mel.py`](crest/datasets/urbansound8k_mel.py). See
  [`crest/datasets/README.md`](crest/datasets/README.md) for the contributor
  guide.
- [`crest/tasks/`](crest/tasks)
  Task adapters. Built-ins include
  [`odometry_regression.py`](crest/tasks/odometry_regression.py) and
  [`sound_classification.py`](crest/tasks/sound_classification.py). See
  [`crest/tasks/README.md`](crest/tasks/README.md) for the contributor guide.
- [`crest/model_families/`](crest/model_families)
  Model-family implementations. Built-ins include
  [`odom_tcn.py`](crest/model_families/odom_tcn.py) and
  [`audio_dscnn.py`](crest/model_families/audio_dscnn.py). See
  [`crest/model_families/README.md`](crest/model_families/README.md) for the
  contributor guide.
- [`crest/microcontrollers/`](crest/microcontrollers)
  Hardware backends and backend registry/factory logic. See
  [`crest/microcontrollers/README.md`](crest/microcontrollers/README.md)
  for bring-up details.
- [`crest/model.py`](crest/model.py)
  Shared runtime helpers for config loading, score evaluation, and generic
  metric normalization.
- [`crest/runtime_bootstrap.py`](crest/runtime_bootstrap.py)
  Shared task-aware bootstrap path used by both `nas_model_client.py` and
  `hil_server.py` to resolve component selection, instantiate dataset/task/
  family components, and validate NAS policy against the active task contract.
- [`crest/hil_runtime.py`](crest/hil_runtime.py)
  Runtime-owned HIL request construction and metric collection helpers used by
  the HIL server and related tests.
- [`crest/pareto_replay.py`](crest/pareto_replay.py)
  Reusable Pareto replay selection, request reconstruction, resume, manifest,
  and result-writing logic behind `pareto_hil_replay.py`.
- [`crest/devices.py`](crest/devices.py)
  Shared device dataclasses and the `DeviceInterface` contract used by
  hardware backends.
- [`crest/hardware.py`](crest/hardware.py)
  Legacy/shared hardware utility layer used by the current HIL/NAS flow.

## Shared Abstractions

The modular runtime is built around three main interfaces plus a small set of
typed payloads shared across orchestration code.

- `DatasetABC` in [`crest/interfaces.py`](crest/interfaces.py)
  Loads raw data and returns a normalized [`DatasetBundle`](crest/pipeline_types.py)
  with train/validation/test/calibration splits plus dataset metadata.
- `TaskABC` in [`crest/interfaces.py`](crest/interfaces.py)
  Defines the target/output contract, task-owned fitting behavior, evaluation
  behavior, and the metric contract that shared NAS/scoring code consumes.
- `ModelFamilyABC` in [`crest/interfaces.py`](crest/interfaces.py)
  Samples hyperparameters, builds models, validates family-local config, and
  materializes the export variant passed into HIL.
- Shared typed payloads in [`crest/pipeline_types.py`](crest/pipeline_types.py)
  carry the normalized information exchanged between those layers:
  `DatasetBundle`, `TargetSpec`, `ModelBuildContext`, `FitPlan`,
  `EvaluationResult`, and `TaskMetricContract`.

## End-to-End Flow

At a high level, the source tree is wired like this:

1. `nas_model_client.py` or `hil_server.py` loads
   [`config/nas_config_stm32.yaml`](config/nas_config_stm32.yaml) through shared helpers in
   [`crest/model.py`](crest/model.py).
2. The entry point calls
   `ensure_builtin_components_registered()` from
   [`crest/builtin_components.py`](crest/builtin_components.py) so the
   default dataset, task, and model family are available by name.
3. The entry point runs the shared bootstrap in
   [`crest/runtime_bootstrap.py`](crest/runtime_bootstrap.py), which
   resolves component selection, instantiates the selected dataset/task/model
   family, derives the target spec, and validates `nas.score`, `nas.prune`,
   and `nas.feasibility` against the task metric contract.
4. The selected dataset adapter loads data and produces a normalized
   `DatasetBundle`.
5. The selected task adapter builds the target contract and training/evaluation
   behavior.
6. The selected model family samples hyperparameters, builds models, and
   materializes export variants.
7. When hardware metrics are needed, the HIL path builds a normalized request
   through [`crest/hil_runtime.py`](crest/hil_runtime.py), then the
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
[`crest/component_selection.py`](crest/component_selection.py).

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

CREST does not currently auto-discover components from the filesystem.
The registry model is explicit:

- [`crest/registry.py`](crest/registry.py) defines
  `dataset_registry`, `task_registry`, and `model_family_registry`.
- [`crest/builtin_components.py`](crest/builtin_components.py) registers
  the built-in components under their stable string keys.
- The entry points call `ensure_builtin_components_registered()` before they
  resolve component names from config.

If you add a new dataset, task, or model family, it is not available until
something registers it under the name you intend to use in config.

## Shared Scoring And Trial Outputs

The shared score/logging path no longer assumes one fixed model shape
or one fixed metric schema.

### Task Metric Declaration

[`TaskMetricContract`](crest/pipeline_types.py) is the task-owned
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

[`ScoreEvaluationResult`](crest/model.py) is the shared representation of
the resolved score/objective values for one trial.

- Scalar runs set `score` and leave the objective lists as the configured
  single-objective projection.
- Multi-objective runs leave `score` as `None` and instead carry ordered
  objective names, values, and directions.

### Generic Trial Outcome

[`TrialOutcome`](crest/model.py) is the generic payload written to CSV and
mirrored into Optuna trial attributes. It carries:

- resolved scalar/objective values
- `task_metrics`
- resolved trial `hyperparams`
- optional task-owned `artifact_summary`

The shared logging code does not need to know which task or model family
produced those values.

### CSV And Optuna Logging

[`log_trial(...)`](crest/model.py) writes a stable infrastructure column set
plus dynamic task/hyperparameter columns.

Stable shared columns include:

- study/timestamp fields
- shared hardware metrics such as RAM, flash, latency, power, energy, and
  error codes
- score/objective metadata (`score_type`, `objective_*_json`)
- pruning metadata
- feasibility metadata and signed Optuna constraints
- `artifact_summary_json`
- cadenced runtime telemetry fields when present

Dynamic columns are added per trial outcome:

- task metrics become `metric__{name}`
- hyperparameters become `hparam__{name}`

The same information is also mirrored into `trial.user_attrs`, including the
fully expanded `metric__*` and `hparam__*` keys.

## Pareto HIL Replay

`pareto_hil_replay.py` does not run NAS or retrain models. It reads a source
NAS CSV/config, reconstructs the logged candidate payloads, selects the valid
Pareto front, and reuses the same HIL server/backend request path that normal
hardware scoring uses.

Use `--dry-run` as a hardware-free preflight; it writes `manifest.json`,
`replay_requests.jsonl`, and `replay_results.csv` without instantiating
`HILServer`. For hardware execution, the command forwards reconstructed
`family_hparams`, `runtime_metadata`, `quantization_mode`, optional preserved
device options, and optional CLI model/checkpoint overrides to HIL.

`--resume` skips payload keys already present in `replay_results.csv` with
`completed` or `dry_run` status. Do not reuse a dry-run output directory for a
hardware replay with `--resume` unless skipping those candidates is intentional.

## Extension Paths

Contributor standards for new implementation work:

- Make breaking config or API changes explicit in the relevant README and
  config examples.
- Avoid temporary compatibility paths unless they have a tracked owner and
  removal point.
- Keep new helpers small, readable, locally validated, and documented with
  NumPy-style docstrings for changed functions, classes, dataclasses, and
  tests.
- Keep non-obvious logic commented, especially feature caching, score/cadence
  derivation, export/materialization, and hardware staging decisions.
- Keep code testable without hardware where possible; hardware flows should
  expose non-hardware preflight checks for config, model construction, export,
  and generated requests.
- Pin or justify dependency additions in the implementation plan that adds
  them.

### Add A New Model Family

If you mean a new trainable/exportable model family, this is the primary
extension path in the current codebase.

Key files:

- [`crest/interfaces.py`](crest/interfaces.py)
  `ModelFamilyABC` defines the contract.
- [`crest/model_families/odom_tcn.py`](crest/model_families/odom_tcn.py)
  Concrete example of the built-in odometry TCN family.
- [`crest/model_families/audio_dscnn.py`](crest/model_families/audio_dscnn.py)
  Concrete example of a built-in classification family over cached log-mel
  tensors.
- [`crest/builtin_components.py`](crest/builtin_components.py)
  Built-in registration.
- [`crest/component_selection.py`](crest/component_selection.py)
  Explicit config selection for dataset, task, and model-family components.
- [`hil_server.py`](hil_server.py) and
  [`nas_model_client.py`](nas_model_client.py)
  Entry-point orchestration that consumes the selected family.

Typical steps:

1. Add a new module under [`crest/model_families/`](crest/model_families).
2. Implement a `ModelFamilyABC` subclass.
3. Register it in the appropriate registry path.
   The built-in pattern today is
   [`crest/builtin_components.py`](crest/builtin_components.py).
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

- [`crest/datasets/README.md`](crest/datasets/README.md)
  for the full contributor guide
- [`crest/interfaces.py`](crest/interfaces.py) for `DatasetABC`
- [`crest/datasets/oxiod.py`](crest/datasets/oxiod.py) and
  [`crest/datasets/urbansound8k_mel.py`](crest/datasets/urbansound8k_mel.py)
  as built-in examples
- [`crest/builtin_components.py`](crest/builtin_components.py) for the
  current registration pattern

Typical steps:

1. Add a new dataset adapter under [`crest/datasets/`](crest/datasets).
2. Implement `DatasetABC`.
3. Register it under a stable string key.
4. Select it with `dataset.name`.
5. Put dataset-local knobs under `dataset.params`.

### Add A New Task

Key files:

- [`crest/tasks/README.md`](crest/tasks/README.md)
  for the full contributor guide
- [`crest/interfaces.py`](crest/interfaces.py) for `TaskABC`
- [`crest/tasks/odometry_regression.py`](crest/tasks/odometry_regression.py)
  and [`crest/tasks/sound_classification.py`](crest/tasks/sound_classification.py)
  as built-in examples
- [`crest/builtin_components.py`](crest/builtin_components.py) for the
  current registration pattern

Typical steps:

1. Add a new task adapter under [`crest/tasks/`](crest/tasks).
2. Implement `TaskABC`.
3. Register it under a stable string key.
4. Select it with `task.name`.
5. Keep task-owned training, evaluation, and metric-contract logic inside the
   task adapter.

### Add A New Microcontroller Or Hardware Backend

Key files:

- [`crest/microcontrollers/README.md`](crest/microcontrollers/README.md)
  for the bring-up guide
- [`crest/devices.py`](crest/devices.py) for `DeviceInterface`
- [`crest/microcontrollers/__init__.py`](crest/microcontrollers/__init__.py)
  for backend registration/factory plumbing

That path is intentionally separate from model-family work. Board bring-up,
toolchain integration, upload flow, runtime telemetry, and backend-owned
device options belong to the microcontroller backend layer, not to
`ModelFamilyABC`.

Important caveat:

- A new non-Arduino backend must be wired into the registry metadata in
  [`crest/microcontrollers/__init__.py`](crest/microcontrollers/__init__.py)
  or it is unreachable except through the Arduino FQBN fallback path.

## NAS Policy And HIL Details

For the current scoring, pruning, and runtime knobs, use:

- [`config/README.md`](config/README.md) for the config reference
- [`config/nas_config_stm32.yaml`](config/nas_config_stm32.yaml) for the STM32 config shape
- [`crest/model.py`](crest/model.py) for score/prune/feasibility
  evaluation and HIL request construction
- [`hil_server.py`](hil_server.py) for the HIL-side request handling and
  backend failure shaping
