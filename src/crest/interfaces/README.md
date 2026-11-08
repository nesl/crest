<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->

# Component interfaces

These abstract base classes define how interchangeable CREST components connect
to the shared pipeline. Each module owns one contract; concrete implementations
live in their corresponding component packages.

| Module | Contract | Implementations |
| --- | --- | --- |
| [`dataset.py`](dataset.py) | `DatasetABC`: loading, validation, calibration data | [Datasets](../datasets/README.md) |
| [`task.py`](task.py) | `TaskABC`: training setup, metrics, evaluation | [Tasks](../tasks/README.md) |
| [`model_family.py`](model_family.py) | `ModelFamilyABC`: architecture sampling, construction, export | [Model families](../model_families/README.md) |
| [`optimizer.py`](optimizer.py) | `OptimizerABC`: initialization, proposal rounds, resume identity | [Optimizers](../optimizers/README.md) |

Import the contracts through the package exports:

```python
from crest.interfaces import DatasetABC, TaskABC, ModelFamilyABC, OptimizerABC
from crest.interfaces import OptimizerBudgetExhausted
```

`OptimizerBudgetExhausted` lives beside the optimizer contract and signals that an
optimizer-owned budget cannot fund another round. Shared request/result records
remain in [`pipeline_types.py`](../pipeline_types.py).

The contracts can be inspected without loading TensorFlow or device backends.
Method docstrings specify required overrides and default behavior; use the linked
component guides when adding an implementation.

## Required hooks and defaults

The ABCs distinguish methods every implementation must supply from optional hooks
with shared defaults. Implement the abstract methods first, then override defaults
only when the component needs different behavior.

| Contract | Required methods |
| --- | --- |
| `DatasetABC` | `load` |
| `TaskABC` | `build_target_spec`, `metric_contract`, `compile_model`, `build_fit_plan`, `evaluate`, `evaluate_predictions` |
| `ModelFamilyABC` | `sample_hparams`, `build_model` |
| `OptimizerABC` | `propose_round` |

Some optional hooks become necessary for a particular execution mode. For example,
an explicit proposer needs the model family's `trial_search_space` declaration,
while native sampling can work without it. The component guides explain these
requirements, validation hooks, and available defaults.

### Example: inspect an implementation's obligations

This works without importing TensorFlow or creating any dataset, model, or provider:

```python
from crest.interfaces import ModelFamilyABC

print(sorted(ModelFamilyABC.__abstractmethods__))
# ['build_model', 'sample_hparams']
```

A subclass that leaves an abstract method unimplemented cannot be instantiated.
Completing those methods is the minimum Python requirement, not proof that the
component is compatible with a particular task, dataset, or deployment mode.

## Add a component

1. Choose the contract matching the responsibility you are implementing. Dataset
   code loads data; task code defines training/evaluation; model-family code builds
   architectures; optimizer code proposes experiments.
2. Follow the corresponding guide's example and implement the required hooks:
   [dataset](../datasets/README.md), [task](../tasks/README.md),
   [model family](../model_families/README.md), or
   [optimizer](../optimizers/README.md).
3. Register the implementation class under a stable key in the appropriate
   [registry](../registry.py), and select it through `dataset.name`, `task.name`,
   `model.family`, or `optimizer.type`. Registration is explicit; adding a Python
   file alone does not make a component selectable.
4. Test the component with small inputs and invalid configurations before a full
   search. Use the component guide's definition of done to cover its runtime paths.

## Change an interface

1. Read the existing method contract and its implementations before adding or
   changing a hook. Put shared input/output records in
   [`pipeline_types.py`](../pipeline_types.py), rather than coupling interfaces to
   concrete implementations.
2. Decide whether a new hook can have a useful default. Making it abstract requires
   every existing subclass to implement it, including third-party components.
3. Update the module docstrings, affected implementations, and their component
   guides together. Keep package-level imports in `__init__.py` working.
4. Run interface compatibility tests and the affected component tests. Run the full
   suite when changing shared method signatures or defaults.

## Definition of done

An interface change is ready when:

- Required hooks, return types, defaults, and failure behavior are documented.
- Existing public imports still resolve to the same exported classes.
- Importing the contracts does not initialize ML runtimes or concrete backends.
- Existing implementations still instantiate and their behavior is covered by
  tests; any intentional compatibility change has an explicit migration path.
- Examples, registration instructions, and links match the current source layout.

For package import and default-hook checks, run from the repository root:

```bash
python -m pytest test/test_interfaces_package.py test/test_modularity_scaffolding.py
```

## Troubleshooting

- **Cannot instantiate an abstract class:** inspect `__abstractmethods__` to find
  hooks the subclass has not implemented.
- **Unknown component name:** check registration and the matching config selector;
  importing an interface does not register implementations.
- **Unexpected runtime imports:** keep framework types under `TYPE_CHECKING` and
  runtime-dependent default-hook imports inside the method that needs them.
- **A component works alone but fails in a study:** check the shared payloads and
  the selected task/model/optimizer requirements, not only the abstract-method list.
