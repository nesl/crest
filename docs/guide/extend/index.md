# Extend CREST

Extend the component that owns the change you want to make.

| Change | Guide |
| --- | --- |
| Load a different dataset or supply representative calibration samples | [Add a dataset](datasets.md) |
| Define targets, losses, metrics, or evaluation | [Add a task](tasks.md) |
| Define an architecture, search space, or export variant | [Add a model family](models.md) |
| Build, flash, or measure a model on a new board | [Add a board](boards.md) |

The software pipeline resolves the dataset, task, and model family by name. The dataset returns a bundle, the task turns that bundle into a target specification, and the model family builds against that specification. These three components use [explicit registries](../api/registry.md). Boards use the [microcontroller factory](../api/devices.md).

Start with the built-in component closest to your change. Check its contract with small fixtures before running a full NAS study. See the [API reference](../api/index.md) for the interfaces and the data they exchange.
