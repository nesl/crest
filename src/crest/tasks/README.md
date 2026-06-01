# Tasks

Task adapters define the application semantics that sit between dataset bundles
and model families.

Hardware requirement: no hardware is required to implement or test a task
adapter. Hardware is only needed when the resulting dataset/task/model
combination is exported and measured through a microcontroller backend.

For the broader source map, see [`../../README.md`](../../README.md). For the
config surface that selects tasks, see [`../../config/README.md`](../../config/README.md).

## What This Layer Owns

Tasks own:

- target/output contracts derived from a `DatasetBundle`;
- task-local config validation;
- Keras compile behavior, losses, and metrics;
- `model.fit(...)` wiring and final-training fit plans;
- metric contracts consumed by shared scoring and pruning;
- evaluation of model predictions on dataset splits;
- optional closeout artifacts such as summaries, confusion matrices, or plots.

Tasks do not own dataset loading, architecture search spaces, export
materialization, HIL requests, or board-specific measurement.

## Shipped Implementations

- [`odometry_regression.py`](odometry_regression.py)
  Registered as `odometry_regression`. Defines the two-output OxIOD velocity
  regression contract, task-owned fit wiring, RMSE metrics, and odometry
  closeout artifacts.
- [`sound_classification.py`](sound_classification.py)
  Registered as `sound_classification`. Defines the single-output logits
  classification contract for cached audio features, classification metrics,
  fixed-split reporting, and optional fold-rotation evaluation.

## Key Files

- [`../interfaces.py`](../interfaces.py)
  Defines `TaskABC`.
- [`../pipeline_types.py`](../pipeline_types.py)
  Defines `TargetSpec`, `FitPlan`, `EvaluationResult`, and
  `TaskMetricContract`.
- [`../builtin_components.py`](../builtin_components.py)
  Registers shipped tasks under stable config names.
- [`../registry.py`](../registry.py)
  Owns `task_registry`.
- [`../component_selection.py`](../component_selection.py)
  Resolves `task.name` and `task.params` from config.
- [`../../config/README.md`](../../config/README.md)
  Documents task selection and task-owned config blocks.

## Runtime Flow

The task path runs after dataset loading and before model-family construction:

1. Entry points register built-ins through
   [`../builtin_components.py`](../builtin_components.py).
2. Runtime bootstrap instantiates the selected task with runner-owned callback
   paths and patience values.
3. `validate_config(...)` checks `task.params`.
4. `build_target_spec(...)` derives the output contract from the dataset
   bundle.
5. `metric_contract(...)` declares task metrics to shared scoring and pruning.
6. The selected model family builds a model against the task target spec.
7. `compile_model(...)` applies task-owned losses and optimizer setup.
8. `build_fit_plan(...)` provides training and final-training `model.fit(...)`
   inputs, targets, validation data, callbacks, and monitor keys.
9. `evaluate(...)` and `evaluate_predictions(...)` produce structured metrics
   for reporting and scoring.

## Add A New Task

Use [`odometry_regression.py`](odometry_regression.py) or
[`sound_classification.py`](sound_classification.py) as the concrete starting
point.

1. Add a new task module under [`src/crest/tasks/`](.).
2. Implement `TaskABC` from [`../interfaces.py`](../interfaces.py).
3. Define a stable task key and return it from `name`.
4. Validate `task.params` in `validate_config(...)`.
5. Build a `TargetSpec` that accurately describes model outputs.
6. Declare every task metric used by `nas.score`, `nas.prune`, or
   `nas.feasibility` in `metric_contract(...)`.
7. Keep task-specific training, evaluation, and closeout logic inside the task
   adapter.
8. Register the task in [`../builtin_components.py`](../builtin_components.py)
   with `task_registry.register(...)`.
9. Select it in config with `task.name` and put task-local knobs under
   `task.params`.
10. Add hardware-free tests for target spec, fit-plan wiring, metrics, and
    evaluation behavior.

## Contract In Practice

`TaskABC` exposes these main hooks:

- `name`
  Optional property. Returns the human-readable task identifier; shipped tasks
  override it with the stable registry key.
- `build_target_spec(bundle, task_config)`
  Required. Returns the task-owned output names, shapes, task type, and
  metadata used by model families and output validation.
- `metric_contract(target_spec, task_config)`
  Required. Declares available, training-only, nonnegative, and primary
  metrics so shared NAS/scoring code can validate policies.
- `compile_model(model, task_config, target_spec)`
  Required. Applies task-owned loss, optimizer, and compile-time metric setup.
- `build_fit_plan(...)`
  Required. Returns the inputs, targets, validation data, callbacks, and
  monitor key used by NAS-time and final training.
- `evaluate(...)` and `evaluate_predictions(...)`
  Required. Produce `EvaluationResult` payloads from a model or normalized
  predictions.
- `history_component_keys(...)`, `generate_closeout_artifacts(...)`, and
  `validate_model_outputs(...)`
  Optional. Use these when the task needs per-output history plots,
  task-specific final artifacts, or structural output checks.

## Registration And Config Selection

Registration is explicit. Task adapters are not auto-discovered from this
directory.

The shipped pattern is:

1. Define the task class in this package.
2. Import it in [`../builtin_components.py`](../builtin_components.py).
3. Register it under a stable key with `task_registry.register(...)`.
4. Select that key through `task.name`.
5. Put task-owned options under `task.params`.

The runtime expects task constructors to use the explicit keyword-only
constructor contract documented in [`../../config/README.md`](../../config/README.md).

## Contributor Checklist

- Keep dataset loading in the dataset layer and architecture/search logic in
  the model-family layer.
- Keep every metric name stable and declare it in `metric_contract(...)`.
- Validate task config locally before model construction.
- Keep fit-plan wiring deterministic and explicit for search and final modes.
- Add or update NumPy-style docstrings for changed functions, classes,
  dataclasses, and tests.
- Add hardware-free tests before running any HIL validation.

## Definition Of Done

A task adapter is ready when:

1. `validate_config(...)` catches invalid task parameters.
2. `build_target_spec(...)` describes the model outputs accurately.
3. `metric_contract(...)` declares all metrics used by config policies.
4. `compile_model(...)`, `build_fit_plan(...)`, and evaluation hooks work for
   both NAS-time and final-training paths.
5. Optional closeout artifacts are deterministic and written under the
   runner-provided output directory.
6. The task is registered and selectable with `task.name`.
7. Hardware-free tests cover target spec, fit-plan, evaluation, and failure
   behavior.

## Troubleshooting

If task bring-up fails, check these layers in order:

1. Config:
   Does `task.name` match the registered key and do `task.params` validate?
2. Target spec:
   Do output names and shapes match the selected model family?
3. Metrics:
   Are score, prune, and feasibility metric names declared by the task?
4. Fit plan:
   Are inputs, targets, validation data, callbacks, and monitor keys valid for
   both search and final modes?
5. Evaluation:
   Does the task handle model predictions and split targets in the same order
   used during training?
