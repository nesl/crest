# Workload and training

A dataset loads the inputs, a task defines the outputs and metrics, and a model family defines which architectures CREST can explore. All three are selected explicitly.

| Config field | OxIOD desktop example | Meaning |
| --- | --- | --- |
| `dataset.name` | `oxiod` | Adapter that prepares the input windows |
| `task.name` | `odometry_regression` | Target definition and evaluation contract |
| `model.family` | `odom_tcn` | Model builder and search space |

## Set a training budget

`training.nas_epochs` controls training per search candidate. `training.model_epochs` controls final retraining. In a constrained study, `training.nas_trials` is the target number of feasible completed trials and `training.max_total_trials` caps all attempts. Set both budgets deliberately when board time is scarce.

??? info "Component fields and example YAML"
<!-- config-section `dataset`, `task`, and `model` -->

??? info "Training and quantization fields"
<!-- config-section `training` -->

??? info "UrbanSound8K fold rotation"
<!-- config-section Audio fold rotation -->

[Extend a component](../extend/index.md) · [Choose objectives](search.md)
