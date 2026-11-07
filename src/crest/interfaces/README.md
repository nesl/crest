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

Existing imports remain supported through the package exports:

```python
from crest.interfaces import DatasetABC, TaskABC, ModelFamilyABC, OptimizerABC
from crest.interfaces import OptimizerBudgetExhausted
```

`OptimizerBudgetExhausted` lives beside the optimizer contract and signals that an
optimizer-owned budget cannot fund another round. Shared request/result records
remain in [`pipeline_types.py`](../pipeline_types.py).

Importing these contracts does not load TensorFlow or concrete component backends.
Model-dependent default hooks import their runtime helpers when called. Method
docstrings specify required overrides and default behavior; use the linked
component guides when adding an implementation.
