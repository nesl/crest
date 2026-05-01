# Single HIL Run

This folder provides a minimal, single-pass HIL sanity check. It builds a
fixed TinyODOM model, runs the HIL controller once, and prints the returned
metrics (latency/energy/RAM/flash/etc.).

The current scripts use the modular runtime surface:

- runtime dimensions come from `HILServer.get_runtime_dimensions()`
- representative export inputs come from `HILServer.get_calibration_inputs()`
- the HIL request is split into `family_hparams` and `runtime_metadata`
  before calling `HILServer.determine_metrics(...)`

When the active DUT sketch emits clock telemetry, the returned metrics can also
include `clock_hz` and `dwt_cycles_per_inference`.

## Usage

```bash
python analysis_scripts/hil_single_run/run_single_hil.py
```

Optional overrides:

```bash
python analysis_scripts/hil_single_run/run_single_hil.py --input-mode uniform
python analysis_scripts/hil_single_run/run_single_hil.py --input-mode representative
python analysis_scripts/hil_single_run/run_single_hil.py --output analysis_scripts/hil_single_run/last_run.json
python analysis_scripts/hil_single_run/run_single_hil.py --harness-arm-pin 3 --harness-trigger-pin 2
python analysis_scripts/hil_single_run/run_single_hil.py --dut-arm-hold-ms 600 --harness-stable-low-ms 500
```

## Perturbed One-Run Variant

Run exactly one HIL pass with the BN+bias perturbed model variant:

- Forces `energy_aware = true`
- Forces `input_mode = uniform`
- Forces `model_variant = approx_trained`

```bash
python analysis_scripts/hil_single_run/run_single_hil_perturbed.py
```

Optional overrides:

```bash
python analysis_scripts/hil_single_run/run_single_hil_perturbed.py --output analysis_scripts/hil_single_run/last_run_perturbed.json
python analysis_scripts/hil_single_run/run_single_hil_perturbed.py --harness-arm-pin 3 --harness-trigger-pin 2
python analysis_scripts/hil_single_run/run_single_hil_perturbed.py --dut-arm-hold-ms 600 --harness-stable-low-ms 500
```

## Toy GPIO test

Use the minimal DUT/harness GPIO test to validate D2 wiring without TFLite or
handshake logic. This script flashes the toy sketches, then streams serial
output from both boards to the terminal and to log files.

```bash
python analysis_scripts/hil_single_run/run_toy_hil.py
```

Optional overrides:

```bash
python analysis_scripts/hil_single_run/run_toy_hil.py --dut-port /dev/ttyACM0 --harness-port /dev/ttyACM1
python analysis_scripts/hil_single_run/run_toy_hil.py --dut-fqbn arduino:mbed_nano:nano33ble
python analysis_scripts/hil_single_run/run_toy_hil.py --harness-fqbn arduino:mbed_nano:nano33ble
python analysis_scripts/hil_single_run/run_toy_hil.py --skip-dut-flash
python analysis_scripts/hil_single_run/run_toy_hil.py --skip-harness-flash
python analysis_scripts/hil_single_run/run_toy_hil.py --cycles 10 --sleep-seconds 2
```

Default behavior:
- Harness is flashed once and monitored continuously.
- DUT is flashed 5 times (one pulse per flash) with a 5-second pause between cycles.
- Stop with Ctrl-C.

Toy timing notes:
- D3 is an active-low arm line; the harness only arms when D3 is LOW and D2 is LOW.
- Harness requires a 500 ms stable-low window before arming.
- DUT holds D3 LOW for 600 ms before driving D2 HIGH.
- Harness auto-disarms after each pulse and waits for D3 HIGH again.
- The runner waits up to 5 seconds for `HARNESS: DONE` after each DUT cycle.

Lessons learned from toy validation:
- Bootloader/upload noise is real: during DUT flash, D2 can go HIGH even when no real inference is running, so D2-only measurement causes false captures.
- Hardware gating with a separate arm line works: D3 active-low gating plus a stable-low window prevented upload-time false pulses, and auto-disarm prevented double-counting.
- Margin between DUT and harness windows matters: DUT arm hold time must be longer than harness stable-low requirement; toy settings (DUT 600 ms, harness 500 ms) were stable across repeated cycles.
- Host-side lifecycle management is required: stop DUT serial monitoring before reflashing DUT, keep harness monitoring alive across all cycles, and wait for per-cycle `HARNESS: DONE`.
- Keep per-board logs: separate DUT and harness logs made race conditions and monitor-thread bugs obvious and debuggable.
- Expected timing behavior: harness and DUT pulse durations were close but not identical; a small consistent offset is expected from independent clocks and timestamp points.
- Direct carryover to real HIL design: keep explicit arming/disarming semantics, keep per-attempt DUT/harness completion barriers, and treat harness telemetry as valid only inside a known attempt window.

Production migration checks (D3 arm + D2 trigger):
- During DUT upload, harness can observe transient `D2 HIGH` but should stay disarmed while `D3` is HIGH.
- For a valid attempt, expect: DUT `DUT READY` -> DUT `timer output:` -> harness `DONE`.
- The harness should report one `DONE` per DUT attempt and run-counts should match.
- DUT/harness timing should be close with a stable offset; large drift suggests wiring or gating regressions.
- Use CLI overrides in `run_single_hil.py` to quickly test alternate pin/timing combinations before editing YAML.

Sketches used by the toy script live under:
- `analysis_scripts/toy_gpio_dut/toy_gpio_dut.ino`
- `analysis_scripts/toy_gpio_harness/toy_gpio_harness.ino`

Logs are written to:
- `analysis_scripts/hil_single_run/dut_log.txt`
- `analysis_scripts/hil_single_run/harness_log.txt`

Press Ctrl-C to stop the monitoring loop.
