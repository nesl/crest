<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->

# Source Guide

This directory contains the Python entry points and core library code for the CREST training, hardware-in-the-loop, and deployment flow.

Related docs

- Repository setup, dataset preparation, and operator workflow live in [`../README.md`](../README.md).
- Config block meanings, NAS policy shape, and runtime config caveats live in [`config/README.md`](config/README.md).
- The complete optimizer contract, evidence, budgeting, and restart behavior live in the [optimizer system guide](crest/optimizers/README.md).
- Microcontroller and hardware-backend bring-up live in [`crest/microcontrollers/README.md`](crest/microcontrollers/README.md).

For installation and runnable workflows, see [Getting started](GETTING_STARTED.md).

## Where To Start

| Goal | Start Here | Hardware Requirement |
|------|------------|----------------------|
| Run or modify NAS/training orchestration | [`nas_model_client.py`](nas_model_client.py) and [`crest/runtime_bootstrap.py`](crest/runtime_bootstrap.py) | No hardware required when `device.hil: false` |
| Add a proposer or inspect the shared optimizer flow | [Optimizer system guide](crest/optimizers/README.md) and [shared contracts](#shared-contracts) | No hardware required for contract checks |
| Modify LLM candidate generation | [`crest/optimizers/llm/`](crest/optimizers/llm) and [optimizer config](config/README.md#optimizer) | No hardware required for fake-provider tests |
| Run or modify HIL server behavior | [`hil_server.py`](hil_server.py) and [`crest/hil/runtime.py`](crest/hil/runtime.py) | Development board required for execution. HIL harness required for energy measurement |
| Add a dataset | [`crest/datasets/README.md`](crest/datasets/README.md) | No hardware required for adapter development |
| Add a task | [`crest/tasks/README.md`](crest/tasks/README.md) | No hardware required for adapter development |
| Add a model family | [`crest/model_families/README.md`](crest/model_families/README.md) | No hardware required for model construction. Hardware needed for backend validation |
| Add board support | [`crest/microcontrollers/README.md`](crest/microcontrollers/README.md) | Matching board and toolchain required |

## Top-Level Entry Points

- [`nas_model_client.py`](nas_model_client.py) Runs NAS, training, scoring, artifact export, and the client side of the hardware-in-the-loop workflow.
- [`hil_server.py`](hil_server.py) Runs the ZeroMQ HIL server that materializes models, stages device-specific candidates, and returns compile/runtime metrics.
- [`pareto_hil_replay.py`](pareto_hil_replay.py) Replays already-logged Pareto-front NAS candidates through a target HIL config and writes replay artifacts.

## Package Map

| Area | Responsibility |
| --- | --- |
| [`crest/component_selection.py`](crest/component_selection.py), [`registry.py`](crest/registry.py), [`builtin_components.py`](crest/builtin_components.py) | Resolve configured names and register components |
| [Shared contracts](#shared-contracts), [`pipeline_types.py`](crest/pipeline_types.py) | Component contracts and shared records |
| [`crest/datasets/`](crest/datasets/README.md) | Data loading, splits, and calibration data |
| [`crest/tasks/`](crest/tasks/README.md) | Training, evaluation, and task metrics |
| [`crest/model_families/`](crest/model_families/README.md) | Search spaces, model construction, and export variants |
| [`crest/optimizers/`](crest/optimizers/README.md) | Candidate proposals and shared round execution |
| [`crest/search/`](crest/optimizers/README.md#progress-and-reproducibility) | Progress reporting and training randomness |
| [`crest/hil/`](#hil-request-and-serial-helpers), [`microcontrollers/`](crest/microcontrollers/README.md) | Hardware requests, measurement, and board backends |
| [`crest/model.py`](crest/model.py), [`runtime_bootstrap.py`](crest/runtime_bootstrap.py) | Configuration, scoring, and component initialization |
| [`crest/pareto_replay.py`](crest/pareto_replay.py) | Candidate selection and request reconstruction for Pareto HIL replay |
| [`crest/devices.py`](crest/devices.py), [`hardware.py`](crest/hardware.py) | Device contracts and shared hardware utilities |

## Shared Contracts

The `crest` package implements the reusable pipeline behind the source entry points.
Use canonical imports such as `from crest.pipeline_types import ModelBuildContext`.
Dependencies are declared in [`../pyproject.toml`](../pyproject.toml).
Runtime modules may require TensorFlow, Optuna, serial libraries, or board toolchains.

| Module | Contract | Implementation guide |
| --- | --- | --- |
| [`dataset.py`](crest/interfaces/dataset.py) | `DatasetABC` loads and validates data and supplies calibration inputs | [Datasets](crest/datasets/README.md) |
| [`task.py`](crest/interfaces/task.py) | `TaskABC` owns training, evaluation, and metrics | [Tasks](crest/tasks/README.md) |
| [`model_family.py`](crest/interfaces/model_family.py) | `ModelFamilyABC` samples, builds, and exports architectures | [Model families](crest/model_families/README.md) |
| [`optimizer.py`](crest/interfaces/optimizer.py) | `OptimizerABC` owns proposal rounds and resume identity | [Optimizers](crest/optimizers/README.md) |

Import contracts through the package exports.

```python
from crest.interfaces import DatasetABC, TaskABC, ModelFamilyABC, OptimizerABC
from crest.interfaces import OptimizerBudgetExhausted
```

`OptimizerBudgetExhausted` signals that an optimizer-owned budget cannot fund another round.
Shared input/output records live in [`crest/pipeline_types.py`](crest/pipeline_types.py), including splits, target specifications, model contexts, evaluation results, history, budgets, and proposals.
The contracts can be inspected without loading TensorFlow or device backends.
Method docstrings specify required overrides and default behavior.

### Required Hooks And Defaults

| Contract | Required methods |
| --- | --- |
| `DatasetABC` | `load` |
| `TaskABC` | `build_target_spec`, `metric_contract`, `compile_model`, `build_fit_plan`, `evaluate`, `evaluate_predictions` |
| `ModelFamilyABC` | `sample_hparams`, `build_model` |
| `OptimizerABC` | `propose_round` |

Implement abstract methods first and override defaults only when needed.
Some execution modes require optional hooks.
An explicit proposer needs the model family's `trial_search_space` declaration, while native sampling can work without it.

### Inspect An Implementation's Obligations

This example needs no dataset, model, provider, or TensorFlow import.

```python
from crest.interfaces import ModelFamilyABC

print(sorted(ModelFamilyABC.__abstractmethods__))
# ['build_model', 'sample_hparams']
```

A subclass with an unimplemented abstract method cannot be instantiated.
Implementing those methods alone does not establish compatibility with a selected task, dataset, or deployment mode.

## End-to-End Flow

1. An entry point loads configuration and resolves the selected dataset, task, model family, and optimizer.
2. The dataset supplies splits and metadata.
   The task defines outputs, fitting, evaluation, and available metrics.
3. The optimizer requests native sampling or proposes explicit parameters.
   The shared runner manages trials and budgets and evaluates candidates through the same objective.
4. The model family builds and exports each candidate.
   When hardware metrics are needed, the selected backend stages, compiles, uploads, and measures it.
5. Shared scoring combines task and hardware metrics for search and reporting.

NAS and the HIL server share component initialization through [`runtime_bootstrap.py`](crest/runtime_bootstrap.py).

## Component Selection

The current modular selection surface is resolved in [`crest/component_selection.py`](crest/component_selection.py).

Use the [configuration reference](config/README.md) for component selectors, required blocks, defaults, and component-local parameters.

## Registration Model

Registration is explicit.
Adding a Python file does not make a component selectable.
Built-ins are registered in [`crest/builtin_components.py`](crest/builtin_components.py).
Register custom classes under stable keys in [`crest/registry.py`](crest/registry.py) before loading configuration.
Select them through `dataset.name`, `task.name`, `model.family`, or `optimizer.type`.
Follow the matching component guide's extension steps and completion checks, including small inputs and invalid configurations before a full search.

Native sampling and explicit LLM proposals use the same objective, feasibility rules, and evaluation budget.
The runner validates complete explicit batches before queueing them.
See the [optimizer guide](crest/optimizers/README.md) for the proposal contract and [LLM guide](crest/optimizers/llm/README.md) for provider examples.

## Shared Scoring And Trial Outputs

### Task Metric Declaration

Tasks declare their available metrics through `TaskMetricContract`.
Shared scoring validates objective, pruning, and feasibility settings against that declaration.
See the [task contract](crest/tasks/README.md#contract-in-practice).

### Task-Owned Fit And Closeout Hooks

Tasks own training plans, evaluation, and final reporting artifacts.
Their [contributor guide](crest/tasks/README.md) explains these hooks and includes a concrete example.

### Score Resolution

Scoring supports scalar values and ordered multi-objective values/directions.
See the [NAS configuration reference](config/README.md#nas) for score policies.

### Generic Trial Outcome

Trial outcomes combine score values, task metrics, hyperparameters, hardware measurements, and optional task artifacts.
They are written to CSV and Optuna.

### CSV And Optuna Logging

CSV output includes study identity, objectives, hardware measurements, pruning, and feasibility status.
Task metrics use `metric__{name}` columns and hyperparameters use `hparam__{name}`.
The corresponding values also appear in Optuna trial attributes.
For analysis, use the [trial-outcome guide](../analysis_scripts/appendix_trial_outcomes/README.md).

## Pareto HIL Replay

`pareto_hil_replay.py` does not run NAS or retrain models.
It reads a source NAS CSV/config, reconstructs the logged candidate payloads, selects the valid Pareto front, and reuses the same HIL server/backend request path that normal hardware scoring uses.

Use `--dry-run` as a hardware-free preflight.
It writes `manifest.json`, `replay_requests.jsonl`, and `replay_results.csv` without accessing hardware.
Review these requests before running against a board.
Hardware execution uses the saved candidate parameters and any model/checkpoint overrides you supply.

`--resume` skips payload keys already present in `replay_results.csv` with `completed` or `dry_run` status.
Do not reuse a dry-run output directory for a hardware replay with `--resume` unless skipping those candidates is intentional.

## Extension Paths

Contributor standards for new implementation work

- Make breaking config or API changes explicit in the relevant README and config examples.
- Avoid temporary compatibility paths unless they have a tracked owner and removal point.
- Keep new helpers small, readable, locally validated, and documented with NumPy-style docstrings for changed functions, classes, dataclasses, and tests.
- Keep non-obvious logic commented, especially feature caching, score/cadence derivation, export/materialization, and hardware staging decisions.
- Keep code testable without hardware where possible.
  Hardware flows should expose non-hardware preflight checks for config, model construction, export, and generated requests.
- Pin dependencies and explain additions in the relevant package documentation.

## Change An Interface

1. Read the method contract and all affected implementations before changing a hook.
   Put shared records in [`crest/pipeline_types.py`](crest/pipeline_types.py) and keep interfaces independent of concrete implementations.
2. Prefer a useful default when possible.
   A new abstract hook requires every subclass, including third-party components, to implement it.
3. Update module docstrings, implementations, component guides, and migration instructions together.
   Preserve package exports in `crest/interfaces/__init__.py`.
4. Run interface compatibility and affected component tests from the repository root.
   Run the full suite when changing shared signatures or defaults.

### Completion Checks

- Required hooks, return types, defaults, and failure behavior are documented.
- Public imports resolve to the exported classes without initializing ML runtimes or concrete backends.
- Existing implementations instantiate and have behavior covered by tests.
- Intentional compatibility changes have an explicit migration path.
- Examples, registration instructions, and links match the source layout.

```bash
python -m pytest test/test_interfaces_package.py test/test_modularity_scaffolding.py
```

### Contract Troubleshooting

- If an abstract class cannot instantiate, inspect `__abstractmethods__` for missing hooks.
- If a component name is unknown, check registration and its config selector.
  Importing a contract does not register implementations.
- If importing contracts loads a runtime unexpectedly, keep framework types under `TYPE_CHECKING` and runtime-dependent imports inside the default hook that needs them.
- If a component works alone but fails in a study, inspect shared payloads and the selected task, family, and optimizer requirements.

## NAS Policy And HIL Details

For the current scoring, pruning, and runtime knobs, use the following guides.

- [`config/README.md`](config/README.md) for the config reference
- [`config/nas_config_stm32.yaml`](config/nas_config_stm32.yaml) for the STM32 config shape
- [`crest/model.py`](crest/model.py) for score/prune/feasibility evaluation and HIL request construction
- [`hil_server.py`](hil_server.py) for the HIL-side request handling and backend failure shaping

## HIL Request And Serial Helpers

HIL helpers connect runtime configuration to the hardware controller and provide the DUT/harness handshake used by device backends.
[`crest/hil/runtime.py`](crest/hil/runtime.py) builds measurement requests and normalizes metrics.
[`crest/hil/protocol.py`](crest/hil/protocol.py) parses serial messages and synchronizes the DUT and harness.
Import serial helpers from `crest.hil.protocol` and request/metric helpers from `crest.hil.runtime`.
Start the server with `python src/hil_server.py`.
