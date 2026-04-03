# Analysis Scripts

This folder collects small, self-contained utilities for HIL experiments and
hardware checks. It is intentionally separate from the core training/runtime
code so you can run diagnostics without touching the main workflow.

Recent Portenta H7 analysis packages in this folder rely on shared DUT clock
telemetry emitted by `sketches/common/tinyodom_clock_telemetry.h`. When the
target core exposes a DWT cycle counter, the DUT can now report:

- `clock_hz`
- `dwt_cycles_per_inference`

## Folders

- `cadenced_portenta_h7/`
  - Runs Portenta H7 perturbed-model experiments for:
    - back-to-back (10 consecutive windows)
    - cadenced (one window every latency budget with idle sleep)
  - Produces JSON + CSV summaries across `cm7` and `cm4`.
  - See `analysis_scripts/cadenced_portenta_h7/README.md`.

- `portenta_baseline_load/`
  - Synthetic baseline timing/energy test for Portenta H7 using harness
    telemetry.
  - Compares `heavy` (busy-loop 10 x 200 ms) vs `sleep` (`delay(200)` for
    10 iterations) across `cm7` and `cm4`.
  - Produces JSON + CSV summaries.
  - See `analysis_scripts/portenta_baseline_load/README.md`.

- `hil_noise_analysis/`
  - Scripts for running multi-mode HIL noise scans, exporting representative
    input data, and analyzing the resulting CSV outputs.
  - See `analysis_scripts/hil_noise_analysis/README.md` for details.

- `ina228_check/`
  - A minimal Arduino sketch that verifies INA228 readings over I2C.
  - Prints bus voltage, power, and computed current every second.

- `hil_single_run/`
  - Runs a single HIL controller pass and prints the metrics.
  - Useful as a quick “does the board/toolchain still work?” sanity check.
  - Surfaces clock telemetry fields too when the underlying DUT sketch reports
    them.

- `arena_latency_curve/`
  - Runs fixed-arena HIL sweeps and records latency/energy per arena size.
  - Produces attempt CSV, aggregated JSON, and a dual-axis latency+energy PNG.
  - Supports BLE and Portenta H7 (`cm7` / `cm4`).
  - See `analysis_scripts/arena_latency_curve/README.md`.

- `clock_tick_latency/`
  - Runs repeated perturbed-model Portenta H7 HIL attempts and exports
    `latency_ms` + `ticks_per_inference`.
  - Produces attempt CSV, aggregates JSON, and latency-vs-ticks scatter PNG.
  - See `analysis_scripts/clock_tick_latency/README.md`.

- `stm32_blink_example_project/`
  - Working STM32N6 CubeIDE blink package for `NUCLEO-N657X0-Q`.
  - Started from a fresh STM32CubeIDE project and was then adapted from the
    STM32CubeN6 Template FSBL sources.
  - Now serves as the first STM32-oriented stepping stone toward a future
    TinyODOM backend, with both GUI bring-up notes and a researched CLI path.
  - See `analysis_scripts/stm32_blink_example_project/README.md` and
    `memory_stm.md`.

## Running the INA228 check

These commands assume you have run `./setup_arduino.sh` and that your board is
Arduino Nano 33 BLE

1. Compile:

```bash
tools/bin/arduino-cli compile \
  --fqbn arduino:mbed_nano:nano33ble \
  --config-file tools/arduino-cli.yaml \
  analysis_scripts/ina228_check
```

2. Upload:

```bash
tools/bin/arduino-cli upload \
  --fqbn arduino:mbed_nano:nano33ble \
  --port /dev/ttyACM0 \
  --config-file tools/arduino-cli.yaml \
  analysis_scripts/ina228_check
```

3. Monitor serial output:

```bash
tools/bin/arduino-cli monitor \
  -p /dev/ttyACM0 \
  --config-file tools/arduino-cli.yaml \
  --config baudrate=115200
```

If your port differs, replace `/dev/ttyACM0` with the value from:

```bash
tools/bin/arduino-cli board list --config-file tools/arduino-cli.yaml
```
