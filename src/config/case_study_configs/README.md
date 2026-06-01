<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->

# Case Study Configs

These configs preserve the run axes used by the CREST case studies. Use the
top-level configs in `src/config/` as general starting points; use this folder
when you want the case-study device, workload, runtime-mode, and scoring
combinations.

Each YAML includes its intended launch command near the top. Run commands from
the repository root with the `crest` environment active.

## Case Study 1: OxIOD TCN Across Targets

Hardware requirement: target development board and HIL harness required for
the measured-energy configs in this folder.

- [`nas_config_case1_2_ble33_b2b_oxiod.yaml`](nas_config_case1_2_ble33_b2b_oxiod.yaml)
  Arduino Nano 33 BLE Sense, OxIOD, back-to-back runtime.
- [`nas_config_case1_3_portenta_m7_b2b_oxiod.yaml`](nas_config_case1_3_portenta_m7_b2b_oxiod.yaml)
  Portenta H7 CM7, OxIOD, back-to-back runtime.
- [`nas_config_case1_4_portenta_m4_b2b_oxiod.yaml`](nas_config_case1_4_portenta_m4_b2b_oxiod.yaml)
  Portenta H7 CM4, OxIOD, back-to-back runtime.
- [`nas_config_case1_5_stm32_b2b_oxiod.yaml`](nas_config_case1_5_stm32_b2b_oxiod.yaml)
  NUCLEO-N657X0-Q, OxIOD, back-to-back runtime.

The desktop proxy examples for the same OxIOD path live one directory up:
[`../nas_config_flops_rmse.yaml`](../nas_config_flops_rmse.yaml) and
[`../nas_config_memory_proxy.yaml`](../nas_config_memory_proxy.yaml). Those
proxy configs do not require hardware.

## Case Study 2: STM32 Runtime Schedule Comparison

Hardware requirement: NUCLEO-N657X0-Q board required; HIL harness required for
energy-measured runs.

- [`nas_config_case2_1_stm32_b2b_oxiod.yaml`](nas_config_case2_1_stm32_b2b_oxiod.yaml)
  NUCLEO-N657X0-Q, OxIOD, continuous back-to-back runtime.
- [`nas_config_case2_2_stm32_cadenced_oxiod.yaml`](nas_config_case2_2_stm32_cadenced_oxiod.yaml)
  NUCLEO-N657X0-Q, OxIOD, cadenced sensing-window runtime with cadence-aware
  feasibility constraints.

## Case Study 3: UrbanSound8K DS-CNN Application Scoring

Hardware requirement: target board and HIL harness required as shipped. For
hardware-free training, use these configs as templates with HIL and measured
scoring disabled.

- [`nas_config_case3_1_portenta_m7_audio.yaml`](nas_config_case3_1_portenta_m7_audio.yaml)
  Portenta H7 CM7, UrbanSound8K cached log-mel inputs, DS-CNN family.
- [`nas_config_case3_2_stm32_audio.yaml`](nas_config_case3_2_stm32_audio.yaml)
  NUCLEO-N657X0-Q, UrbanSound8K cached log-mel inputs, DS-CNN family.

For the full config schema and scoring policy reference, see
[`../README.md`](../README.md).
