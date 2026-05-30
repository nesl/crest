# Micro Workload Energy Probe

Synthetic phase-energy probes for TinyODOM-compatible MCU targets. The probe
uses the existing TinyODOM hardware-in-the-loop harness and INA228 telemetry
path to measure fixed-duration deployment windows occupied by simple phases:
sleep/idle, light wait, tight polling, floating-point compute, and integer
compute.

## Contents

- `run_micro_workload_energy_probe.py`
  Host-side runner. It compiles and uploads DUT firmware, prepares the HIL
  harness, runs the requested board/workload/repeat matrix, validates telemetry,
  and writes per-attempt plus aggregate outputs.
- `micro_workload_probe.ino`
  Arduino DUT sketch used for Nano 33 BLE and Portenta H7 targets.
- `stm32_synthetic_dut_runner.c`
  STM32 LRUN DUT runner staged into the production STM32 template for the
  synthetic workloads.
- `results/`
  Generated local outputs. This directory is intentionally ignored by git.

## Supported Targets

Board tokens are selected with `--boards`:

- `stm32`: STM32 NUCLEO-N657X0-Q using the production STM32 LRUN template,
  clock setup, RTC setup, signing, external-flash App programming, and debug
  load path.
- `portenta_m4`: Arduino Portenta H7 M4 core, internally `target_core=cm4`,
  including the existing CM7 boot-helper/runtime preparation path.
- `portenta_m7`: Arduino Portenta H7 M7 core, internally `target_core=cm7`.
- `ble`: Arduino Nano 33 BLE Sense using `arduino:mbed_nano:nano33ble`.

There is no separate core argument. The board token selects target behavior.

## Workloads

Workloads are selected with `--workloads`:

- `sleep`: idle/sleep baseline for the requested window.
- `wait`: low-impact block/timer wait baseline for payload diagnostics.
- `poll`: active tight polling/spinning without a compute payload.
- `float`: block-loop structure plus single-precision floating-point
  multiply/add recurrences.
- `int`: block-loop structure plus unsigned integer multiply/add and XOR/shift
  recurrences.

The DUT raises the harness trigger at the beginning of the requested window and
lowers it at the end. Serial telemetry is emitted only after the trigger-high
measurement window has ended so serial printing does not pollute the measured
phase.

## Hardware Setup

The probe reuses the standard TinyODOM HIL harness contract:

- Arduino DUT trigger: D2
- Arduino DUT arm: D3, active-low
- STM32 DUT trigger: PD0
- STM32 DUT arm: PE9, active-low

The harness emits the normal INA228 telemetry lines, including:

- `runs:`
- `energy output (mJ):`
- `avg power output (mW):`
- `avg current output (mA):`
- `bus voltage output (V):`
- `idle power baseline (mW):`
- `harness timer output:`
- `DONE`

The host parses these lines through the existing TinyODOM helpers rather than a
separate measurement protocol.

## Quick Start

Run a short STM32 smoke attempt:

```bash
python analysis_scripts/micro_workload_energy_probe/run_micro_workload_energy_probe.py \
  --boards stm32 \
  --workloads sleep \
  --window-ms 200 \
  --repeats 1
```

Run the full supported workload matrix for one-second windows:

```bash
python analysis_scripts/micro_workload_energy_probe/run_micro_workload_energy_probe.py \
  --boards stm32 portenta_m4 portenta_m7 ble \
  --workloads sleep wait poll float int \
  --window-ms 1000 \
  --repeats 3
```

Reuse an already-flashed harness and write explicit outputs:

```bash
python analysis_scripts/micro_workload_energy_probe/run_micro_workload_energy_probe.py \
  --boards ble \
  --workloads wait poll float int \
  --skip-harness-flash \
  --output-json analysis_scripts/micro_workload_energy_probe/results/ble_probe.json \
  --output-csv analysis_scripts/micro_workload_energy_probe/results/ble_probe.csv
```

Run 50 one-second STM32 repeats:

```bash
python analysis_scripts/micro_workload_energy_probe/run_micro_workload_energy_probe.py \
  --boards stm32 \
  --workloads sleep wait poll float int \
  --window-ms 1000 \
  --repeats 50 \
  --output-json analysis_scripts/micro_workload_energy_probe/results/stm32_1s_x50.json \
  --output-csv analysis_scripts/micro_workload_energy_probe/results/stm32_1s_x50.csv
```

Use `--help` for the full option list.

## Outputs

The runner opens output files before the first hardware attempt and streams each
completed attempt to CSV and JSONL immediately. This protects long hardware
runs from losing completed rows if later attempts fail.

Passing a directory to `--output-json` or `--output-csv` is allowed; the runner
places a timestamped default filename inside that directory. The final JSON
contains:

- `metadata`: config paths, board/workload matrix, ports, harness policy, STM32
  staging details, requested window, aggregate CSV path, and streaming JSONL
  path.
- `attempts`: one row per board/workload/repeat.
- `aggregates`: grouped by board/workload.

The attempt CSV is written incrementally to `--output-csv`. A paired aggregate
CSV is written at finalization beside it with `_aggregates` appended to the
filename stem. A streaming JSONL file is written beside the final JSON with
`_attempts.jsonl` appended to the filename stem.

Each attempt row includes:

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

Aggregates include mean, standard deviation, minimum, maximum, and count for
energy, power, measured harness window duration, DUT work units, elapsed time,
cycles, and sleep residency. Derived aggregate fields subtract the board-local
`sleep` mean for phase/residency cost and the board-local `wait` mean for
payload diagnostics.

## Reused TinyODOM Code

This analysis script reuses:

- `src/tinyodom/hil_protocol.py` for harness priming and DONE waits.
- `src/tinyodom/microcontrollers/arduino_base.py` for Arduino compile/upload,
  harness compile/upload, and INA228 telemetry parsing.
- `src/tinyodom/microcontrollers/arduino_portenta_h7.py` for Portenta board
  options and CM4 boot-helper behavior.
- `src/tinyodom/microcontrollers/arduino_ble33.py` for BLE FQBN/device
  metadata.
- `analysis_scripts/stm32_example_project/stm32_lrun_common.py` for STM32 LRUN
  workspace build, signing, FSBL copy-window update, and external image
  programming.
- `sketches/stm32/tinyodom_stm32_lrun/` as the staged STM32 production
  template.
- `sketches/common/tinyodom_hil_config.h` and the existing harness sketch
  protocol.
