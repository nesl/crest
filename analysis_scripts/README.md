# Analysis Scripts

This folder collects small, self-contained utilities for HIL experiments and
hardware checks. It is intentionally separate from the core training/runtime
code so you can run diagnostics without touching the main workflow.

## Folders

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
