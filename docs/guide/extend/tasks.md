# Add a task

A task supplies the target specification, compiles the model, owns fit inputs and callbacks, and evaluates predictions. See [Task API](../api/tasks.md).

1. Add a `TaskABC` subclass under `src/crest/tasks/`. Accept the keyword arguments `checkpoint_path` and `early_stopping_patience` that `instantiate_task_component` passes to the constructor.
2. Validate `task.params` and derive a `TargetSpec` from the dataset bundle. Ensure its ordered output names and shapes match the tensors the [model family](models.md) builds and the targets passed to Keras.
3. Implement `compile_model` and `build_fit_plan`. Put inputs, targets, and any validation data in `FitPlan.fit_kwargs` and task callbacks in `FitPlan.callbacks`. The runner supplies scheduling arguments such as epochs and batch size.
4. Implement `evaluate` and `evaluate_predictions`. Declare the task metrics used in `nas.score`, `nas.prune`, and `nas.feasibility` in `metric_contract`. Use the same names in returned metrics and policies.
5. Register the class with `task_registry` in `src/crest/builtin_components.py` and select the key under `task.name`.

For a worked example, `SoundClassificationTask` validates UrbanSound8K class metadata and declares a single `class_logits` output with one value per class. It compiles with sparse categorical cross-entropy configured for logits and declares `loss`, `accuracy`, and `macro_f1` for evaluation. The odometry task in `src/crest/tasks/odometry_regression.py` shows the two-output case.

Check target specifications and fit plans in both search and final training modes. Pass small, known predictions to `evaluate_predictions` to test metric names and values without training a model. See `test/test_sound_classification_task.py`.
