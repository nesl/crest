<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->

# Case-study experiments

This directory contains portable copies of saved experiment configurations, organized by the current paper's case-study numbering.
Each YAML records its source and original file hash.
Dataset, device, network, output, and firmware paths are adjusted for this checkout, while search objectives, budgets, precision choices, and proposal policies retain their saved values.
Use these for new studies after preparing data and setting device ports using the [getting-started guide](../../GETTING_STARTED.md).
Use the [general example configs](../README.md#example-configs) as the simpler starting point for a new application.

These configurations do not recreate historical firmware, trained models, or saved measurements.
Use fresh study names and output directories rather than resuming an archived study with changed paths or runtime versions.

## Case study 1 STM32 runtime schedule comparison

All six configurations use OxIOD and the STM32 N657.
Continuous and periodic identify the energy objective, not necessarily the measurement mode.
The 100 and 500 ms searches collect both measurement phases even when scoring continuous energy.

| Release period | Continuous-energy objective | Periodic-energy objective |
| --- | --- | --- |
| 100 ms | [Continuous](cs1/stm32_continuous_100ms.yaml) | [Periodic](cs1/stm32_periodic_100ms.yaml) |
| 200 ms | [Continuous](cs1/stm32_continuous_200ms.yaml) | [Periodic](cs1/stm32_periodic_200ms.yaml) |
| 500 ms | [Continuous](cs1/stm32_continuous_500ms.yaml) | [Periodic](cs1/stm32_periodic_500ms.yaml) |

The 100 and 500 ms campaigns use int8 and a 256-attempt cap.
The older 200 ms configurations search float and int8 and retain their saved 600-attempt cap, although the paper analyzes only the first 256 attempts.
The 200 ms periodic campaign used earlier firmware with per-slot UART output and is excluded from the paper's model and selection analyses.
Current firmware runs cannot reproduce that historical measurement protocol from YAML alone.

## Case study 2 OxIOD TCN across targets

The two proxy searches use mixed float/int8 precision and require no board.
The four measured-energy searches require the corresponding target and measurement harness.

| Experiment | Configuration |
| --- | --- |
| FLOPs proxy | [OxIOD FLOPs](cs2/oxiod_flops_proxy.yaml) |
| Memory-traffic proxy | [OxIOD memory](cs2/oxiod_memory_proxy.yaml) |
| Nano 33 BLE Sense measured energy | [BLE](cs2/oxiod_ble_measured_energy.yaml) |
| Portenta H7 CM4 measured energy | [CM4](cs2/oxiod_portenta_m4_measured_energy.yaml) |
| Portenta H7 CM7 measured energy | [CM7](cs2/oxiod_portenta_m7_measured_energy.yaml) |
| STM32 N657 measured energy and clock search | [STM32](cs2/oxiod_stm32_measured_energy.yaml) |

These are the saved paper campaigns, including their precision choices and budgets.
The [general FLOPs and memory examples](../README.md#example-configs) use different precision settings.
For cross-target evaluation of selected candidates, use [Pareto HIL replay](../../README.md#pareto-hil-replay).

## Case study 3 UrbanSound8K DS-CNN application scoring

All six campaigns score validation macro-F1 and continuous energy using the same application score.
Portenta TPE collects continuous measurements only, while the other campaigns also collect periodic measurements.
Their recorded elapsed times therefore include different measurement work.

| Target | TPE | Memory | History |
| --- | --- | --- | --- |
| Portenta H7 CM7 | [TPE](cs3/audio_portenta_tpe.yaml) | [Memory](cs3/audio_portenta_memory.yaml) | [History](cs3/audio_portenta_history.yaml) |
| STM32 N657 | [TPE](cs3/audio_stm32_tpe.yaml) | [Memory](cs3/audio_stm32_memory.yaml) | [History](cs3/audio_stm32_history.yaml) |

Memory and History use the same [15 warm-up candidates](cs3/warmup_params.json) before provider calls.
Keep that file beside the YAMLs.
For live credentials and provider access, follow the [LLM guide](../../crest/optimizers/llm/README.md#run-an-experiment).
The saved model choice is retained, but provider availability and nondeterministic responses can affect a new run.
