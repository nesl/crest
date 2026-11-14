# Saved-artifact Pareto replay and ESP32 energy

`src/pareto_hil_replay.py --artifact-manifest FILE` reuses the existing feasible
Pareto selection and supplied-TFLite backend. It does not train, load checkpoints,
or convert models. The manifest has a `candidates` list with `source_study`,
`source_trial`, `staged_tflite_path`, `tflite_sha256`, `byte_length`, and optional
`resolution`. Relative artifact paths resolve against the manifest directory.
Selected CSV rows must carry the original hash in `artifact_summary_json` (or
`tflite_sha256`) and the study in `study_name` (otherwise the source directory
name). Trial IDs come from the manifest; CSV row positions are never trial IDs.
If `trial_number` is present it must also match. Missing, ambiguous, malformed,
or hash-mismatched artifacts fail before HIL. Executability and hashes are
checked using the existing artifact reader, including again before execution.

Each separate replay result retains source study, original trial ID, hash,
source metrics, target metrics and request identity. Exact replay resume keys
include these identities, target config contents, fixture contents, firmware
sources, resolved sdkconfig and dependency lock when present. They distinguish
preflight from execution and do not accept legacy architecture-only keys.
Failed exact measurements return a nonzero CLI status and remain retryable.
The original source CSV/database/models are read-only. The legacy reconstruction
path remains available for HIL-first studies; its existing return/resume behavior
is unchanged. Source feasibility flags are now respected during selection.

## Why a harness protocol extension is necessary

The original protocol treats the first falling trigger edge as the end of one
continuous batch, using a compiled invocation count. Merely pulsing the ESP32
trigger for every invocation would measure only the first pulse. Keeping one
pulse around the batch would include warmups, UART output, input copies and
watchdog yields. Re-arming and emitting telemetry for each invocation would add
600 ms waits and substantially change the workload cadence.

The optional `WINDOWS 1` / `WINDOWS READY 1` negotiation therefore adds multiple
trigger windows within the existing active-low arm interval. This is a necessary
harness session contract extension, not a replacement artifact transport or
energy logger. The default `PING` handshake retains the legacy single-pulse
protocol. Unsupported installed firmware fails negotiation before `START`.
Existing boards and callers default to the old protocol. The ESP32 uses the
existing shared handshake, parser, power accumulator and metrics, requests
exclusive serial access, and disables automatic harness flashing. It requires
explicit operator installation of the correct harness firmware.

GPIO21 (harness D2) is high only immediately around timed `Invoke()`. GPIO47
(harness D3) stays low for the session, then high to finish. Allocation, input
copying, warmups, printing, and 20 ms task yields are outside every pulse.
Watchdog settings, yields, debug optimization, ESP-NN and 240 MHz clock remain
unchanged. A single invocation must still fit within the existing five-second
task watchdog interval; this must be confirmed for the deployed models.

The harness counts completed pulses, accumulates their energy and durations,
and emits one aggregate record after disarm. Thus ten fixtures with two measured
repetitions normalize by **20**, not two. `energy_mj_per_inference` is total
monitored-rail energy divided by observed pulses. The additional
`measurement_energy_total_mj` is that per-invocation value times the pulse count.
DUT median latency remains independent of the harness mean window duration.
The host checks counts and gross duration alignment (2% or 2 ms per invocation),
but that check does not establish electrical accuracy.

## Measurement boundary and calibration

Energy includes board/regulator/memory/background overhead supplied through the
monitored rail during invocation windows. It is total energy, **not
idle-subtracted** and not a CPU-only estimate. The idle baseline is retained as
telemetry but not subtracted. The sensor observes the actual power boundary,
which the owner must verify; USB or another feed bypassing the shunt invalidates
that interpretation.

The shared INA228 implementation returns joules and emits mJ. Invocation mode
uses the existing 0.015-ohm shunt and 0.30-A calibration range, 280-us current,
150-us bus and 150-us temperature conversions, but changes averaging from 64 to
1. It powers down at each falling edge instead of using the legacy triggered
conversion plus 20-ms flush wait. Legacy averaging and flush remain unchanged.
Actual settings are emitted in the serial evidence (conversion codes correspond
to the installed Adafruit enums). The resistor value, sensor range, operating
voltage, current headroom, gain/offset, and lack of saturation must be checked
against the physical setup and a reference load before accepting results.

GPIO polling, I2C transaction time and finite conversion intervals cause edge
uncertainty. This implementation does **not** prove exact sensor integration
alignment. Scope/logic-analyzer comparison and a known-load timing test are
mandatory before claiming measured classifier-window energy. Reject a pilot
with missed pulses, appreciable edge bias, or inconsistent timing; do not hide
these with a latency-only result or an estimator. If settings change during
calibration, rebuild the harness and retain its source/binary hashes and settings.

The current INA228 logger supplies aggregate telemetry, not raw current/voltage
traces or a sample-count stream. `measurement_energy_raw_trace_available` is
false. Retain serial telemetry, pulse counts/durations, individual DUT timings,
model/fixture hashes, application build evidence, calibration records and scope
captures separately. Do not label aggregate telemetry as a raw trace.

## November 14 FER handoff on blinky

The prepared local runbook is
`/home/joe/Projects/crest-fer-energy-replay-20261114/README.md`. It contains the
exact environment, isolated paths, one-model pilot and later full replay
commands. The setup is host-validated only; no hardware was flashed or measured
while preparing it. The full 13-model run is gated on successful physical pilot
validation. No energy estimator is fitted or published.
