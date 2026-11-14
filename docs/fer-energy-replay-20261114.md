# Saved Pareto energy replay — host-validated setup, November 14, 2026

The implementation and host setup are validated. **No hardware was flashed or
measured.** Physical energy validation remains mandatory before the full replay.
No training or model conversion was performed. No energy estimator was fitted.

Development checkout: `/tmp/crest-energy-replay-dev`, branch
`hci/saved-artifact-energy-replay`, based on audited HCI revision
`b31b358826f0d50bb2e2b35ff47758dfa57062d4`. The original integration checkout,
main branch, source study/database/results, existing HIL/tunnel sessions and
m202-GPU training jobs were not modified. Only completed source CSV/config files
were copied read-only from m202-GPU to `source/` for local validation.

## Inputs and validation performed

Source study: `fer_joint_75e_20261111_195454`. All 13 staged TFLite SHA-256 values
and byte lengths were reverified. Existing Pareto selection on all 125 source
rows selects exactly trials 114, 23, 66, 108, 86, 29, 68, 41, 79, 121, 76, 50, 71
(the CLI preserves CSV order, so trial 23 is first).

- 282 tests and 32 subtests passed, with no skipped tests in the final run.
  Includes existing replay, ESP32, shared hardware, HIL server and artifact
  transport regressions; exact artifact selection/hash/resume/failure tests;
  both real fixture resolutions; strict missing/invalid energy checks; and
  execution of the real harness state machine against simulated GPIO/power.
- All 13 exact artifacts passed executable TFLite validation and fixture staging
  through existing CREST code. Both fixture files passed their recorded SHA-256,
  shape, dtype, IDs and order checks.
- Trial 114 (64×64) ESP32 build: 491,713 image bytes; 79,443 static RAM bytes.
  Trial 71 (96×96): 623,145 image bytes; 79,443 static RAM bytes. Each retained
  application contains its exact model bytes once. A 2 MiB PSRAM arena is separate
  from static internal RAM. `build-records.json` links retained build identities.
- ESP-IDF v5.5.1, debug/ESP-NN, 240 MHz, original memory/partition settings.
  Resolved sdkconfig SHA-256 remains
  `d065c95d6f38504fbb3f49d25793923097902752f9b3acedb8aee82428935150`, matching the
  original latency profile. The five-second task watchdog and 20-ms yields are
  unchanged. Runner source/application hashes identify the new trigger behavior.
- Nano 33 BLE harness compilation passed: 104,448 program bytes, 46,296 RAM bytes.
  The binary in `harness-build/` has **not** been installed.
- Final config loaded successfully. Actual HIL request preparation for trial 23
  (64) and trial 71 (96) succeeded with model construction, conversion, serial
  access and hardware collection explicitly blocked. These were preflights,
  not measurements. Final-path CLI dry-run selected all 13 candidates.

Evidence: `validation/tests.log`, `validation/firmware-build.log`,
`validation/harness-build.log`, `validation/dry-run/`,
`validation/runtime-preflight.log`, `build-records.json`, and
`preflight-candidates/`. Build and runtime-preflight Python scripts are retained
under `validation/` for reproduction. The original draft is archived at
`validation/target-energy.original.yaml`; `target-energy.yaml` replaces it based
on config validation, both-resolution staging/builds and runtime preflight.
This does **not** certify its physical measurement setup. The manifest's original
staging-only status is retained because energy measurements do not yet exist.

## Remaining physical gates — do not skip

Both devices were visible as `/dev/ttyACM1` (ESP32) and `/dev/ttyACM0` (harness).
No serial opener was reported by `fuser` at inspection, but `crest-hil` and
`crest-tunnel` remain running. This is not a board reservation. The owner must
establish exclusive availability without interrupting training/HIL requests.
Do not kill unrelated processes or stop the existing sessions as part of this
runbook. Wait for an available board window if they still serve other work.

Before installing the harness or running the pilot, verify:

1. GPIO21 → D2, GPIO47 → D3, common ground, safe logic levels, active-low arm and
   high trigger semantics. Confirm twenty distinct trigger pulses for ten
   fixtures × two measured invocations, and no pulses during warmups or copying.
2. The complete intended ESP32 supply passes through the sensor. Identify and
   remove or account for alternate power feeds (including USB) that bypass it.
   Record the measurement boundary and whether the regulator is included.
3. Installed INA228/shunt identity and independently checked calibration.
   Source defaults are 0.015 ohm, 0.30 A, ADC range 40.96 mV; verify current
   headroom, offset/gain, voltage, saturation and a known reference load. Do not
   treat the software defaults or unchanged wiring as verified calibration.
4. The new opt-in `WINDOWS 1` harness binary is installed with matching settings.
   It uses 280-us current, 150-us voltage and temperature conversions, averaging
   one, and immediate power-down/read at falling edges. Verify sensor integration
   against trigger edges with a scope/logic analyzer and known-duration load.
   Quantify polling/I2C/conversion edge bias, especially for the fastest model.
   Host duration checks are coarse rejection checks, not electrical calibration.
5. Each invocation fits the unchanged watchdog; no resets, missed pulses,
   harness errors, stale identity, or invalid/zero energy. Compare pilot latency
   to the original profile and record differences; do not silently change clocks,
   optimization or watchdog settings to obtain a result.

Energy is **measured total monitored-rail energy**, including board/background
power during the invocation windows, not idle-subtracted or CPU-only energy.
The sensor's finite conversion/edge uncertainty still needs physical validation.
The existing logger has aggregate telemetry only; it cannot retain raw current
traces. No raw traces were captured. Save external scope traces separately if
available, along with calibration/settings/photos, firmware hashes, all serial
telemetry, timing evidence and variability. Do not publish an estimator here.

## Owner commands (not executed during preparation)

Start a new shell in tmux on **blinky**, leaving existing sessions alone:

```bash
tmux new-session -s crest-energy-pilot
```

Inside that shell:

```bash
cd /tmp/crest-energy-replay-dev
export REPLAY_ROOT=/home/joe/Projects/crest-fer-energy-replay-20261114
export REPLAY_PY=/home/joe/miniforge3/envs/tinyodomex/bin/python
export IDF_TOOLS_PATH=/tmp/hci-esp32-bringup/idf-tools
export PATH=/home/joe/miniforge3/bin:/tmp/hci-esp32-bringup/host-tools/bin:$PATH
source /tmp/hci-esp32-bringup/esp-idf/export.sh
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=''
set -o pipefail
hostname
ls -l /dev/serial/by-id/
tmux list-panes -a -F '#S #{pane_pid} #{pane_current_command}'
fuser /dev/ttyACM0 /dev/ttyACM1
```

`fuser` returning no openers is insufficient while another service may submit
work: first establish the exclusive board window described above. Save physical
validation records under `$REPLAY_ROOT/validation/hardware/` before acceptance.
After validating power/calibration and approving installation, upload the
prepared **harness only** (this command flashes hardware):

```bash
/home/joe/Projects/tinyodom-ex/tools/bin/arduino-cli upload \
  --config-file /home/joe/Projects/tinyodom-ex/tools/arduino-cli.yaml \
  --fqbn arduino:mbed_nano:nano33ble \
  --port /dev/serial/by-id/usb-Arduino_Nano_33_BLE_DD2379DECFEA6AE6-if00 \
  --input-dir "$REPLAY_ROOT/harness-build" \
  "$REPLAY_ROOT/harness" \
  2>&1 | tee "$REPLAY_ROOT/validation/harness-install.log"
```

If calibration constants need changing, update the shared `crest_power.h`,
copy `firmware/common/*.h` and `firmware/harness/harness.ino` to the isolated
`$REPLAY_ROOT/harness/` workspace, and rebuild before installation:

```bash
cp firmware/harness/harness.ino "$REPLAY_ROOT/harness/"
cp firmware/common/*.h "$REPLAY_ROOT/harness/common/"
/home/joe/Projects/tinyodom-ex/tools/bin/arduino-cli compile \
  --config-file /home/joe/Projects/tinyodom-ex/tools/arduino-cli.yaml \
  --fqbn arduino:mbed_nano:nano33ble \
  --build-path "$REPLAY_ROOT/harness-build" \
  --build-property 'build.extra_flags=-DCREST_HARNESS_ARM_PIN=3 -DCREST_HARNESS_TRIGGER_PIN=2 -DCREST_HARNESS_ARM_TIMEOUT_MS=30000 -DCREST_HARNESS_ACTIVE_TIMEOUT_MS=180000' \
  "$REPLAY_ROOT/harness" 2>&1 | tee "$REPLAY_ROOT/validation/harness-build.log"
```

One-model pilot: **trial 23, 64×64**, original validation macro-F1
0.6742388123912452, original median latency 196.274 ms, exact model SHA-256
`e83995cd3359fe2c35cfb7877b7153f0b02e02470d455b5563c48eb3b7aed82d`.
This command builds/app-flashes the ESP32 and starts the first measurement:

```bash
"$REPLAY_PY" src/pareto_hil_replay.py \
  --source-run-dir "$REPLAY_ROOT/source/fer_joint_75e_20261111_195454" \
  --source-csv "$REPLAY_ROOT/source/fer_joint_75e_20261111_195454/log_NAS_source.csv" \
  --artifact-manifest "$REPLAY_ROOT/artifact-manifest.json" \
  --target-config "$REPLAY_ROOT/target-energy.yaml" \
  --max-candidates 1 --resume \
  --output-dir "$REPLAY_ROOT/pilot/r1" \
  2>&1 | tee "$REPLAY_ROOT/validation/pilot-r1.log"
```

Expect one completed result with `target__error_code=1`,
`target__measurement_instrumentation_valid=True`,
`target__measurement_energy_valid=True`, correct model/fixture hashes,
20 ordered invocation timings, ten output vectors, 20 observed energy windows,
positive finite mJ/inference, no harness errors and a `DONE` marker. Inspect the
linked `target__measurement_record` JSON for serial telemetry and detailed
validation. Its enclosing candidate directory retains the application and
build identity. The replay CSV retains original source metrics independently.
Failure must be investigated, not treated as latency-only success. An installed
old harness fails `WINDOWS 1` negotiation before `START`; no automatic reflash
or fallback repairs it.

After inspecting r1 and physical traces, repeat the same pilot command with
`pilot/r2` and `pilot/r3` (and distinct log names). Each has ten warmups outside
measurement and twenty measured invocations. Report per-session energy and
latency, mean/standard deviation/range across sessions, and per-invocation
latency variability. Aggregate energy telemetry does not provide individual
invocation energies. Reusing a completed output directory with `--resume` skips
it; use separate directories for repeatability measurements.

Only after all physical pilot gates pass, use a separate tmux shell:

```bash
tmux new-session -s crest-energy-full
```

Repeat the environment setup above in that shell, then run the full front:

```bash
for REPETITION in 1 2 3; do
  "$REPLAY_PY" src/pareto_hil_replay.py \
    --source-run-dir "$REPLAY_ROOT/source/fer_joint_75e_20261111_195454" \
    --source-csv "$REPLAY_ROOT/source/fer_joint_75e_20261111_195454/log_NAS_source.csv" \
    --artifact-manifest "$REPLAY_ROOT/artifact-manifest.json" \
    --target-config "$REPLAY_ROOT/target-energy.yaml" \
    --resume --output-dir "$REPLAY_ROOT/full/r$REPETITION" \
    2>&1 | tee "$REPLAY_ROOT/validation/full-r$REPETITION.log" || break
done
```

Each successful repetition must contain thirteen source-trial/hash-linked
results and 260 measured invocations. Validate 96×96 trial 71 identity/timing as
well as 64×64 candidates; the one-model pilot does not prove 96×96 on-board
behavior. Keep every failed attempt and its diagnostics. No existing HIL server
or network port is needed: the replay uses HILServer directly in the local
process without starting its network listener.
