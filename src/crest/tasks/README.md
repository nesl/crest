<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->

# Tasks

Task adapters define the application semantics that sit between dataset bundles and model families.

No hardware is required to implement or test a task adapter.
Hardware is only needed when the resulting dataset/task/model combination is exported and measured through a microcontroller backend.

For the broader source map, see [`../../README.md`](../../README.md).
For the config surface that selects tasks, see [`../../config/README.md`](../../config/README.md).
For implementation work, start with [extension steps](#add-a-new-task), [hook contracts](#contract-in-practice), and the [worked example](#concrete-task-usage).

## What This Layer Owns

Tasks are responsible for the following.

- target/output contracts derived from a `DatasetBundle`
- task-local config validation
- Keras compile behavior, losses, and metrics
- `model.fit(...)` wiring and final-training fit plans
- metric contracts consumed by shared scoring and pruning
- evaluation of model predictions on dataset splits
- optional closeout artifacts such as summaries, confusion matrices, or plots.

Tasks do not own dataset loading, architecture search spaces, export materialization, HIL requests, or board-specific measurement.

## Shipped Implementations

- [`odometry_regression.py`](odometry_regression.py) Registered as `odometry_regression`.
  Defines the two-output OxIOD velocity regression contract, task-owned fit wiring, RMSE metrics, and odometry closeout artifacts.
- [`sound_classification.py`](sound_classification.py) Registered as `sound_classification`.
  Defines the single-output logits classification contract for cached audio features, classification metrics, fixed-split reporting, and optional fold-rotation evaluation.

## Key Files

- [`../interfaces/task.py`](../interfaces/task.py) Defines `TaskABC`.
- [`../pipeline_types.py`](../pipeline_types.py) Defines `TargetSpec`, `FitPlan`, `EvaluationResult`, and `TaskMetricContract`.
- [`../builtin_components.py`](../builtin_components.py) Registers shipped tasks under stable config names.
- [`../registry.py`](../registry.py) Owns `task_registry`.
- [`../component_selection.py`](../component_selection.py) Resolves `task.name` and `task.params` from config.
- [`../../config/README.md`](../../config/README.md) Documents task selection and task-owned config blocks.

## Runtime Flow

The task path runs after dataset loading and before model-family construction.

1. Entry points register built-ins through [`../builtin_components.py`](../builtin_components.py).
2. Runtime bootstrap instantiates the selected task with runner-owned callback paths and patience values.
3. `validate_config(...)` checks `task.params`.
4. `build_target_spec(...)` derives the output contract from the dataset bundle.
5. `metric_contract(...)` declares task metrics to shared scoring and pruning.
6. The selected model family builds a model against the task target spec.
7. `compile_model(...)` applies task-owned losses and optimizer setup.
8. `build_fit_plan(...)` provides training and final-training `model.fit(...)` inputs, targets, validation data, callbacks, and monitor keys.
9. `evaluate(...)` and `evaluate_predictions(...)` produce structured metrics for reporting and scoring.

## Add A New Task

A task supplies the target specification, compiles the model, owns fit inputs and callbacks, and evaluates predictions.

1. Add a `TaskABC` subclass under `src/crest/tasks/`. Accept the keyword arguments `checkpoint_path` and `early_stopping_patience` that `instantiate_task_component` passes to the constructor.
2. Validate `task.params` and derive a `TargetSpec` from the dataset bundle. Ensure its ordered output names and shapes match the tensors the [model family](../model_families/README.md) builds and the targets passed to Keras.
3. Implement `compile_model` and `build_fit_plan`. Put inputs, targets, and any validation data in `FitPlan.fit_kwargs` and task callbacks in `FitPlan.callbacks`. The runner supplies scheduling arguments such as epochs and batch size.
4. Implement `evaluate` and `evaluate_predictions`. Declare the task metrics used in `nas.score`, `nas.prune`, and `nas.feasibility` in `metric_contract`. Use the same names in returned metrics and policies.
5. Return a stable key from `name`. Register the class with `task_registry` in [`builtin_components.py`](../builtin_components.py) and select the key under `task.name`. Set task-specific options under `task.params`.

For a worked example, `SoundClassificationTask` validates UrbanSound8K class metadata and declares a single `class_logits` output with one value per class. It compiles with sparse categorical cross-entropy configured for logits and declares `loss`, `accuracy`, and `macro_f1` for evaluation. The odometry task in [`odometry_regression.py`](odometry_regression.py) shows the two-output case.

Check target specifications and fit plans in both search and final training modes. Pass small, known predictions to `evaluate_predictions` to test metric names and values without training a model. See [`test_sound_classification_task.py`](../../../test/test_sound_classification_task.py).

## Contract In Practice

`TaskABC` exposes these main hooks.

- `name` Optional property.
  Returns the human-readable task identifier.
  Shipped tasks override it with the stable registry key.
- `build_target_spec(bundle, task_config)` Required.
  Returns the task-owned output names, shapes, task type, and metadata used by model families and output validation.
- `metric_contract(target_spec, task_config)` Required.
  Declares available, training-only, nonnegative, and primary metrics so shared NAS/scoring code can validate policies.
- `compile_model(model, task_config, target_spec)` Required.
  Applies task-owned loss, optimizer, and compile-time metric setup.
- `build_fit_plan(...)` Required.
  Returns the inputs, targets, validation data, callbacks, and monitor key used by NAS-time and final training.
- `evaluate(...)` and `evaluate_predictions(...)` Required.
  Produce `EvaluationResult` payloads from a model or normalized predictions.
- `history_component_keys(...)`, `generate_closeout_artifacts(...)`, and `validate_model_outputs(...)` Optional.
  Use these when the task needs per-output history plots, task-specific final artifacts, or structural output checks.

## Concrete Task Usage

The shipped sound task can validate its target and metric contracts and evaluate predictions without training a model or accessing hardware.
Run this example from `src/` with NumPy installed.

```python
from pathlib import Path
import numpy as np

from crest.datasets.urbansound8k_common import CLASS_NAMES, LABEL_ENCODING
from crest.pipeline_types import DataSplit, DatasetBundle
from crest.tasks.sound_classification import SoundClassificationTask

split = DataSplit(
    inputs=np.zeros((2, 8, 8, 1), dtype=np.float32),
    targets=np.array([0, 1], dtype=np.int64),
)
bundle = DatasetBundle(
    train=split,
    input_shape=(8, 8, 1),
    input_dtype="float32",
    metadata={
        "num_classes": len(CLASS_NAMES),
        "class_names": list(CLASS_NAMES),
        "label_encoding": LABEL_ENCODING,
    },
)
task = SoundClassificationTask(checkpoint_path=Path("checkpoints/example.keras"))
task_params = {"early_stopping_patience": 5}
task.validate_config(task_params)
target_spec = task.build_target_spec(bundle, task_params)
contract = task.metric_contract(target_spec, task_params)
logits = np.zeros((2, len(CLASS_NAMES)), dtype=np.float32)
logits[0, 0] = logits[1, 1] = 10.0
result = task.evaluate_predictions(logits, split, task_params, target_spec)
assert result.metrics["accuracy"] == 1.0
assert set(result.metrics) <= contract.available_metric_names
```

The example uses synthetic inputs and labels to exercise the task boundary.
It does not prepare an UrbanSound8K cache or train a classifier.
No checkpoint is written by these calls.
In a full run, select the registered task with the following configuration.

```yaml
task:
  name: sound_classification
  params:
    early_stopping_patience: 5
```

Pair this fragment with the `urbansound8k_mel` dataset and `audio_dscnn` family in a complete config.
The runner supplies constructor callback paths and patience.
A new task subclass must implement all required hooks in [`TaskABC`](../interfaces/task.py).
This shipped-task example is a quick check of their input/output contracts rather than a complete subclass skeleton.

## Definition Of Done

Verify the task with hardware-free success and representative failure cases.

- Invalid `task.params` fail before model construction.
- Output names and shapes match the paired model family, and configured score, prune, and feasibility metrics are declared.
- Compile, fit-plan, and evaluation checks cover both NAS-time and final-training modes, including deterministic callbacks and monitor keys.
- Optional closeout artifacts are deterministic and stay under the runner-provided output directory.
- A config using the registered `task.name` resolves the adapter.
- NumPy-style docstrings match changed functions, classes, dataclasses, and tests.

## Troubleshooting

If task bring-up fails, check these layers in order.

1. Config.
   Does `task.name` match the registered key and do `task.params` validate?
2. Target spec.
   Do output names and shapes match the selected model family?
3. Metrics.
   Are score, prune, and feasibility metric names declared by the task?
4. Fit plan.
   Are inputs, targets, validation data, callbacks, and monitor keys valid for both search and final modes?
5. Evaluation.
   Does the task handle model predictions and split targets in the same order used during training?
