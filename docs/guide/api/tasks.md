# Task API

See [Add a task](../extend/tasks.md) for the constructor and metric declaration conventions.

::: crest.interfaces.TaskABC
    options:
      show_root_heading: true
      members: [build_target_spec, metric_contract, compile_model, build_fit_plan, evaluate, evaluate_predictions, validate_config, validate_model_outputs]

::: crest.pipeline_types.TargetSpec
    options:
      show_root_heading: true

::: crest.pipeline_types.TaskMetricContract
    options:
      show_root_heading: true

::: crest.pipeline_types.FitPlan
    options:
      show_root_heading: true

::: crest.pipeline_types.EvaluationResult
    options:
      show_root_heading: true

::: crest.runtime_bootstrap.instantiate_task_component
    options:
      show_root_heading: true
