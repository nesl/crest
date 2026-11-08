<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->

# Source Guide

This directory contains the Python entry points and core library code for the
CREST training, hardware-in-the-loop, and deployment flow.

Related docs:

- Repository setup, dataset preparation, and operator workflow live in
  [`../README.md`](../README.md).
- Config block meanings, NAS policy shape, and runtime config caveats live in
  [`config/README.md`](config/README.md).
- The complete optimizer contract, evidence, budgeting, and restart behavior live
  in the [optimizer system guide](crest/optimizers/README.md).
- Microcontroller and hardware-backend bring-up live in
  [`crest/microcontrollers/README.md`](crest/microcontrollers/README.md).

For installation and runnable workflows, see [Getting started](GETTING_STARTED.md).

## Where To Start

| Goal | Start Here | Hardware Requirement |
|------|------------|----------------------|
| Run or modify NAS/training orchestration | [`nas_model_client.py`](nas_model_client.py) and [`crest/runtime_bootstrap.py`](crest/runtime_bootstrap.py) | No hardware required when `device.hil: false` |
| Add a proposer or inspect the shared optimizer flow | [Optimizer system guide](crest/optimizers/README.md) and [`crest/interfaces/README.md`](crest/interfaces/README.md) | No hardware required for contract checks |
| Modify LLM candidate generation | [`crest/optimizers/llm/`](crest/optimizers/llm) and [optimizer config](config/README.md#optimizer) | No hardware required for fake-provider tests |
| Run or modify HIL server behavior | [`hil_server.py`](hil_server.py) and [`crest/hil/runtime.py`](crest/hil/runtime.py) | Development board required for execution; HIL harness required for energy measurement |
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

| Area | Responsibility |
| --- | --- |
| [`crest/component_selection.py`](crest/component_selection.py), [`registry.py`](crest/registry.py), [`builtin_components.py`](crest/builtin_components.py) | Resolve configured names and register components |
| [`crest/interfaces/`](crest/interfaces/README.md), [`pipeline_types.py`](crest/pipeline_types.py) | Component contracts and shared records |
| [`crest/datasets/`](crest/datasets/README.md) | Data loading, splits, and calibration data |
| [`crest/tasks/`](crest/tasks/README.md) | Training, evaluation, and task metrics |
| [`crest/model_families/`](crest/model_families/README.md) | Search spaces, model construction, and export variants |
| [`crest/optimizers/`](crest/optimizers/README.md) | Candidate proposals and shared round execution |
| [`crest/search/`](crest/search/README.md) | Progress reporting and training randomness |
| [`crest/hil/`](crest/hil/README.md), [`microcontrollers/`](crest/microcontrollers/README.md) | Hardware requests, measurement, and board backends |
| [`crest/model.py`](crest/model.py), [`runtime_bootstrap.py`](crest/runtime_bootstrap.py) | Configuration, scoring, and component initialization |
| [`crest/pareto_replay.py`](crest/pareto_replay.py) | Candidate selection and request reconstruction for Pareto HIL replay |
| [`crest/devices.py`](crest/devices.py), [`hardware.py`](crest/hardware.py) | Device contracts and shared hardware utilities |

## Shared Abstractions

The four component interfaces separate data loading (`DatasetABC`), task behavior
(`TaskABC`), model construction (`ModelFamilyABC`), and proposal strategy
(`OptimizerABC`). The [interface guide](crest/interfaces/README.md) lists required
hooks, optional defaults, and implementation examples.

[`pipeline_types.py`](crest/pipeline_types.py) defines the records they exchange,
including dataset splits, target specifications, model contexts, evaluation
results, trial history, budgets, and candidate proposals.

## End-to-End Flow

1. An entry point loads configuration and resolves the selected dataset, task,
   model family, and optimizer.
2. The dataset supplies splits and metadata; the task defines outputs, fitting,
   evaluation, and available metrics.
3. The optimizer requests native sampling or proposes explicit parameters. The
   shared runner manages trials and budgets and evaluates candidates through the
   same objective.
4. The model family builds and exports each candidate. When hardware metrics are
   needed, the selected backend stages, compiles, uploads, and measures it.
5. Shared scoring combines task and hardware metrics for search and reporting.

NAS and the HIL server share component initialization through
[`runtime_bootstrap.py`](crest/runtime_bootstrap.py).

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
- `optimizer.type`
  Selects the registered proposer for the NAS client; defaults to `optuna`.

`dataset`, `task`, and `model` are required blocks.

See [`config/README.md`](config/README.md) for the current shipped config shape.

## Candidate Generation

`optimizer.type` defaults to `optuna`; `llm_generator` proposes explicit candidates.
Both use the shared objective, feasibility rules, and evaluation budget. The
runner validates complete explicit batches before queueing them.

Use the [optimizer guide](crest/optimizers/README.md) for the proposal contract,
[configuration reference](config/README.md#optimizer) for settings and resume
rules, and [LLM guide](crest/optimizers/llm/README.md) for provider setup and examples.

## Registration Model

Registration is explicit; placing a file in a component directory does not make
it available through configuration. Built-ins are registered in
[`crest/builtin_components.py`](crest/builtin_components.py). Register custom
components under stable names before loading configuration; each component guide
below shows the registration and configuration steps.

## Shared Scoring And Trial Outputs

### Task Metric Declaration

Tasks declare their available metrics through `TaskMetricContract`. Shared scoring
validates objective, pruning, and feasibility settings against that declaration.
See the [task contract](crest/tasks/README.md#contract-in-practice).

### Task-Owned Fit And Closeout Hooks

Tasks own training plans, evaluation, and final reporting artifacts. Their
[contributor guide](crest/tasks/README.md) explains these hooks and includes a
concrete example.

### Score Resolution

Scoring supports scalar values and ordered multi-objective values/directions.
See the [NAS configuration reference](config/README.md#nas) for score policies.

### Generic Trial Outcome

Trial outcomes combine score values, task metrics, hyperparameters, hardware
measurements, and optional task artifacts. They are written to CSV and Optuna.

### CSV And Optuna Logging

CSV output includes study identity, objectives, hardware measurements, pruning,
and feasibility status. Task metrics use `metric__{name}` columns and
hyperparameters use `hparam__{name}`. The corresponding values also appear in
Optuna trial attributes. For analysis, use the
[trial-outcome guide](../analysis_scripts/appendix_trial_outcomes/README.md).

## Pareto HIL Replay

`pareto_hil_replay.py` does not run NAS or retrain models. It reads a source
NAS CSV/config, reconstructs the logged candidate payloads, selects the valid
Pareto front, and reuses the same HIL server/backend request path that normal
hardware scoring uses.

Use `--dry-run` as a hardware-free preflight; it writes `manifest.json`,
`replay_requests.jsonl`, and `replay_results.csv` without accessing hardware.
Review these requests before running against a board. Hardware execution uses
the saved candidate parameters and any model/checkpoint overrides you supply.

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
- Pin dependencies and explain additions in the relevant package documentation.

### Add A New Model Family

Follow the [model-family guide](crest/model_families/README.md) for required hooks,
registration, raw search-space declarations, a worked example, and completion
checks. Model families own architecture and export variants; hardware backends
own staging, compilation, upload, and measurement.

### Add A New Dataset

Follow the [dataset guide](crest/datasets/README.md) for normalized splits,
calibration data, validation, registration, and a runnable adapter example.

### Add A New Task

Follow the [task guide](crest/tasks/README.md) for target specifications,
metric declarations, fitting, evaluation, and task-specific reporting.

### Add A New Microcontroller Or Hardware Backend

Follow the [backend guide](crest/microcontrollers/README.md) for Arduino, STM32,
and new-vendor bring-up, configuration, and hardware acceptance checks.

## NAS Policy And HIL Details

For the current scoring, pruning, and runtime knobs, use:

- [`config/README.md`](config/README.md) for the config reference
- [`config/nas_config_stm32.yaml`](config/nas_config_stm32.yaml) for the STM32 config shape
- [`crest/model.py`](crest/model.py) for score/prune/feasibility
  evaluation and HIL request construction
- [`hil_server.py`](hil_server.py) for the HIL-side request handling and
  backend failure shaping

## HIL import paths

HIL helpers live in `crest.hil`: import serial helpers from
`crest.hil.protocol` and request/metric helpers from `crest.hil.runtime`.
Start the server with `python src/hil_server.py`.
