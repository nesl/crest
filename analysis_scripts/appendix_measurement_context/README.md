<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->
# Appendix measurement context

Use the [micro-workload probe](#micro-workload-energy-probe) to collect synthetic measurement-context data.
These fixed-duration phase measurements are separate from the model-specific [CS1 waiting-power experiments](../cs1/README.md#collect-new-measurements).
Jump to [hardware setup](#hardware-setup), [commands](#quick-start), or [outputs](#outputs).
Run one experiment at a time on a DUT/harness.

## Micro-workload energy probe

The probe uses CREST's HIL harness and INA228 telemetry to measure sleep, wait, polling, floating-point, and integer-compute windows on supported boards.

A target development board and CREST HIL harness are required.
Follow the [common command setup](../README.md).

For general HIL setup and backend details, see [hardware guide](../../src/crest/microcontrollers/README.md).
For the STM32 LRUN workspace used by the STM32 probe path, see [STM32 workspace guide](../../sketches/stm32/crest_stm32_lrun/README.md).

### Contents

Probe sources live under `experiments/micro_workload_energy_probe/` relative to this guide.

- `run_micro_workload_energy_probe.py` Host-side runner.
  It compiles and uploads DUT firmware, prepares the HIL harness, runs the requested board/workload/repeat matrix, validates telemetry, and writes per-attempt plus aggregate outputs.
- `micro_workload_probe.ino` Arduino DUT sketch used for Nano 33 BLE and Portenta H7 targets.
- `stm32_synthetic_dut_runner.c` STM32 LRUN DUT runner staged into the shipped STM32 template for the synthetic workloads.
- `results/` Default local output directory, ignored by git.
  Supply external output paths as in the examples to keep generated results outside the checkout.

### Supported Targets

Board tokens are selected with `--boards`.

- `stm32`.
  STM32 NUCLEO-N657X0-Q using the shipped STM32 LRUN template with external-flash App programming.
- `portenta_m4`.
  Arduino Portenta H7 M4 core.
- `portenta_m7`.
  Arduino Portenta H7 M7 core.
- `ble`.
  Arduino Nano 33 BLE Sense using `arduino:mbed_nano:nano33ble`.

There is no separate core argument.
The board token selects target behavior.

### Workloads

Workloads are selected with `--workloads`.

- `sleep`.
  Idle/sleep baseline for the requested window.
- `wait`.
  Low-impact block/timer wait baseline for payload diagnostics.
- `poll`.
  Active tight polling/spinning without a compute payload.
- `float`.
  Block-loop structure plus single-precision floating-point multiply/add recurrences.
- `int`.
  Block-loop structure plus unsigned integer multiply/add and XOR/shift recurrences.

The DUT raises the harness trigger at the beginning of the requested window and lowers it at the end.
Serial telemetry is emitted only after the trigger-high measurement window has ended so serial printing does not pollute the measured phase.

### Hardware Setup

The probe reuses the standard CREST HIL harness contract.

- Arduino DUT trigger.
  D2
- Arduino DUT arm.
  D3, active-low
- STM32 DUT trigger.
  PD0
- STM32 DUT arm.
  PE9, active-low

Before running a probe, confirm the following requirements.

- the target board is connected and visible at the serial port used by the selected board token
- the HIL harness is connected at the harness serial port
- Arduino CLI setup has been completed for Arduino targets
- STM32CubeCLT, the repo-local STM32 setup, signing tools, and CubeProgrammer path have been completed for the `stm32` target.
  The synthetic STM32 probe does not require ST Edge AI runtime libraries or `stedgeai` code generation
- the harness wiring matches the DUT arm/trigger pins listed above.

### Quick Start

Run a short STM32 smoke attempt.

```bash
python analysis_scripts/appendix_measurement_context/experiments/micro_workload_energy_probe/run_micro_workload_energy_probe.py \
  --boards stm32 \
  --workloads sleep \
  --window-ms 200 \
  --repeats 1 \
  --output-json /path/to/results/stm32_smoke.json \
  --output-csv /path/to/results/stm32_smoke.csv
```

Run the full supported workload matrix for one-second windows.

```bash
python analysis_scripts/appendix_measurement_context/experiments/micro_workload_energy_probe/run_micro_workload_energy_probe.py \
  --boards stm32 portenta_m4 portenta_m7 ble \
  --workloads sleep wait poll float int \
  --window-ms 1000 \
  --repeats 3 \
  --output-json /path/to/results/workload_matrix.json \
  --output-csv /path/to/results/workload_matrix.csv
```

Reuse an already-flashed harness and write explicit outputs.

```bash
python analysis_scripts/appendix_measurement_context/experiments/micro_workload_energy_probe/run_micro_workload_energy_probe.py \
  --boards ble \
  --workloads wait poll float int \
  --skip-harness-flash \
  --output-json /path/to/results/ble_probe.json \
  --output-csv /path/to/results/ble_probe.csv
```

Run 50 one-second STM32 repeats.

```bash
python analysis_scripts/appendix_measurement_context/experiments/micro_workload_energy_probe/run_micro_workload_energy_probe.py \
  --boards stm32 \
  --workloads sleep wait poll float int \
  --window-ms 1000 \
  --repeats 50 \
  --output-json /path/to/results/stm32_1s_x50.json \
  --output-csv /path/to/results/stm32_1s_x50.csv
```

Use `--help` for the full option list.

### Outputs

Completed attempts are saved incrementally, so their results remain available if a later attempt fails.

Passing a directory to `--output-json` or `--output-csv` is allowed.
The runner places a timestamped default filename inside that directory.
The final JSON contains the following.

- `metadata`.
  Config paths, board/workload matrix, ports, harness policy, STM32 staging details, requested window, aggregate CSV path, and streaming JSONL path.
- `attempts`.
  One row per board/workload/repeat.
- `aggregates`.
  Grouped by board/workload.

The attempt CSV is written incrementally to `--output-csv`.
A paired aggregate CSV is written at finalization beside it with `_aggregates` appended to the filename stem.
A streaming JSONL file is written beside the final JSON with `_attempts.jsonl` appended to the filename stem.

Each attempt row includes the following.

- `timestamp_utc`
- `board`
- `workload`
- `repeat`
- `requested_window_ms`
- `measured_harness_window_ms`
- `energy_mj_per_window`
- `avg_power_mw`
- `avg_current_ma`
- `bus_voltage_v`
- `idle_baseline_mw`
- `dut_iterations`
- `dut_work_units`
- `dut_work_unit_label`
- `dut_elapsed_us`
- `dut_cycles`
- `dut_sleep_ms`
- `dut_sleep_mode`
- `error_code`
- `error_label`
- `serial_log_path`
- `build_metadata`

Aggregates include mean, standard deviation, minimum, maximum, and count for energy, power, measured harness window duration, DUT work units, elapsed time, cycles, and sleep residency.
Derived aggregate fields subtract the board-local `sleep` mean for phase/residency cost and the board-local `wait` mean for payload diagnostics.
