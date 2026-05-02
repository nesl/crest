# Analysis Scripts

This folder collects small, self-contained utilities for HIL experiments and
hardware checks. It is intentionally separate from the core training/runtime
code so you can run diagnostics without touching the main workflow.

Current script contract:

- HIL analysis runners should derive runtime dimensions from
  `HILServer.get_runtime_dimensions()` when they are intentionally exercising
  the live HIL server path. Preflight/export-only scripts may derive dimensions
  from `BootstrappedPipeline.model_build_context` so they can validate config,
  model construction, and request payloads without constructing a server.
- Export-oriented runners should read representative inputs from
  `HILServer.get_calibration_inputs()` or the bootstrapped dataset bundle,
  rather than from ad hoc `training_data` aliases.
- HIL request scripts should call
  `HILServer.determine_metrics(family_hparams, runtime_metadata, ...)` with
  the structured request shape introduced by the runtime refactor.
- Fixed TinyODOM helper scripts should route through
  `tinyodom.analysis_support` instead of importing build/FLOP helpers from
  `tinyodom.model`.

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
    input data, analyzing the resulting CSV outputs, and running the staged
    `epoch_sweep/` checkpoint workflow.
  - See `analysis_scripts/hil_noise_analysis/README.md` for details.

- `ina228_check/`
  - A minimal Arduino sketch that verifies INA228 readings over I2C.
  - Prints bus voltage, power, and computed current every second.

- `hil_single_run/`
  - Runs a single HIL controller pass and prints the metrics.
  - Useful as a quick “does the board/toolchain still work?” sanity check.
  - Surfaces clock telemetry fields too when the underlying DUT sketch reports
    them.
  - Also contains the toy GPIO harness validation runner used to debug D2/D3
    handshake wiring.

- `arena_latency_curve/`
  - Runs fixed-arena HIL sweeps and records latency/energy per arena size.
  - Produces attempt CSV, aggregated JSON, and a dual-axis latency+energy PNG.
  - Supports BLE and Portenta H7 (`cm7` / `cm4`).
  - Includes a companion failure-probe runner for narrowing arena bounds and
    forced-model-size experiments.
  - See `analysis_scripts/arena_latency_curve/README.md`.

- `clock_tick_latency/`
  - Runs repeated perturbed-model Portenta H7 HIL attempts and exports
    `latency_ms` + `ticks_per_inference`.
  - Produces attempt CSV, aggregates JSON, and latency-vs-ticks scatter PNG.
  - See `analysis_scripts/clock_tick_latency/README.md`.

- `audio_stm32_hil_smoke/`
  - Runs the Phase 6 STM32 smoke path for `audio_dscnn` over cached
    UrbanSound8K log-mel feature tensors.
  - Supports hardware-free preflight, STM32 prepare-only staging, and full HIL
    timing through the existing STM32 backend.
  - Reports classifier inference over precomputed features only; it does not
    include firmware-side microphone capture or log-mel feature extraction.
  - See `analysis_scripts/audio_stm32_hil_smoke/README.md`.

- `stm32_example_project/`
  - STM32N6 HIL package for `NUCLEO-N657X0-Q`.
  - Contains the blink bring-up project, the toy AI FSBL project, and the full
    HIL runner (`run_stm32_toy_ai_hil.py`) that stages a perturbed TinyODOM
    model, builds the firmware, optionally programs weights to external NOR
    flash, and collects energy + latency metrics via the Arduino harness.
  - Also includes the host-only ST Edge AI Phase 0 probe, the standalone
    staging helper, the back-to-back vs cadenced comparison runner, the
    archival CPU-clock sweep runner, the plotting helper for archived sweeps,
    and the smoke-test script.
  - See `analysis_scripts/stm32_example_project/README.md`.

- `toy_gpio_dut/` and `toy_gpio_harness/`
  - Minimal Arduino sketches used by `hil_single_run/run_toy_hil.py`.
  - Intended for GPIO-only timing validation without the full TinyODOM runtime.

## Running the INA228 check

These commands assume you have already run `make arduino-setup` and installed
the Arduino core package needed for the board you are using. The examples below
use the Nano 33 BLE FQBN because that is the common harness-board setup in this
repo.

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
