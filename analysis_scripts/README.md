# Analysis Scripts

This folder contains public-facing analysis utilities, paper-figure generators,
and small hardware smoke checks that sit outside the core TinyODOM runtime.

## Folders

- `audio_desktop_smoke/`
  - Hardware-free UrbanSound8K audio DS-CNN training smoke test.
  - Writes checkpoint, history, and metrics artifacts under
    `models/audio_desktop_smoke/`.

- `audio_portenta_hil_smoke/`
  - Arduino-backed audio DS-CNN preflight, prepare-only, and full HIL smoke path
    for Portenta H7 and BLE over cached UrbanSound8K log-mel tensors.

- `audio_stm32_hil_smoke/`
  - STM32 audio DS-CNN preflight, prepare-only, and full HIL smoke path over
    cached UrbanSound8K log-mel tensors.

- `compare_pareto_front_calcs/`
  - Generic CSV-derived Pareto-front comparison helper with explicit
    quality/cost columns, feasibility filters, matching rules, and reduction
    denominators.

- `cs3_audio_sensitivity/`
  - Case Study 3 post-hoc score-sensitivity analysis over audio NAS logs.

- `ina228_check/`
  - Minimal Arduino sketch for verifying INA228 voltage, power, and current
    telemetry over I2C.

- `micro_workload_energy_probe/`
  - Synthetic phase-energy probe for TinyODOM-compatible MCU targets, reusing
    the existing HIL harness and INA228 telemetry path.

- `paper_plots/`
  - Publication figure generators plus sidecar plotted-point and summary
    outputs from explicit input CSV/replay paths.

## Running The INA228 Check

These commands assume `make arduino-setup` has already installed the Arduino
core package for the board you are using. The examples below use the Nano 33
BLE FQBN because that is the common harness-board setup in this repo.

Compile:

```bash
tools/bin/arduino-cli compile \
  --fqbn arduino:mbed_nano:nano33ble \
  --config-file tools/arduino-cli.yaml \
  analysis_scripts/ina228_check
```

Upload:

```bash
tools/bin/arduino-cli upload \
  --fqbn arduino:mbed_nano:nano33ble \
  --port /dev/ttyACM0 \
  --config-file tools/arduino-cli.yaml \
  analysis_scripts/ina228_check
```

Monitor:

```bash
tools/bin/arduino-cli monitor \
  -p /dev/ttyACM0 \
  --config-file tools/arduino-cli.yaml \
  --config baudrate=115200
```
