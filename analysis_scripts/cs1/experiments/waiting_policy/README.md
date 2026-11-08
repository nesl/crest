<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->
# CS1: waiting-policy experiments

Replay prepared STM32 continuous, cadenced awake-tail and cadenced STOP-tail images in randomized pair blocks. Preparation and planning open no hardware. This recipe imports successful native build metadata; it does not export models, patch firmware or build STOP-tail firmware. Supply separately prepared images containing the selected `rtc_wall_v4_dwt_active_stop_evidence_awake_tail` or `rtc_wall_v4_dwt_active_stop_evidence_stop_tail` protocol. The repository's awake-only firmware cannot produce STOP-tail images without a separately controlled preparation workflow.

```sh
python -m analysis_scripts.cs1.experiments.waiting_policy.runner prepare --manifest /path/prepared.json --output /path/new-output
python -m analysis_scripts.cs1.experiments.waiting_policy.runner plan --study /path/new-output --rounds 5 --seed 42
python -m analysis_scripts.cs1.experiments.waiting_policy.runner run --study /path/new-output --rounds 5 --seed 42 --device-config /path/device.json --board-ready
```

The JSON manifest contains `schema_version: 1`, `variants: [...]` and `blocks: [["continuous", "awake", "stop"]]`. Blocks partition variant names exactly once. Each variant has `name`, `candidate`, `phase` (`back_to_back` or `cadenced`), `policy` (`awake` or `stop`), `runs`, `clock_mhz`, `period_ms`, native `device_options`, `window_size`, `num_channels`, `project`, `phase_header` (relative to project), `phase_header_sha256`, `app_elf_sha256`, and the native successful `compile_result` dictionary. Project and compile-result paths resolve relative to the manifest. Use 10 measured runs, 5000 us wake margin and minimum sleep for this protocol. Continuous uses the awake image; the phase header cadence is still checked.

The accepted `compile_result` schema is the native [CompileResult dataclass](../../../../src/crest/devices.py). Preserve the actual successful return value from `STM32NucleoN657X0QDevice.compile(...)`, serialized with `dataclasses.asdict` and paths converted to strings (as in [repeatability preparation](../repeatability/experiment.py)). Do not reconstruct build metadata from filenames. Its fields include `success`, `log`, memory/overflow diagnostics, `build_dir`, `boot_build_dir`, `boot_elf_path`, `app_build_dir`, `app_elf_path`, `signed_app_bin_path`, and `fsbl_copy_window_bytes`; retain all fields from the native result. `success` must be true, `overflow_kind` empty, and `app_elf_path` must identify the hashed protocol image. `device_options` uses the native [STM32 options schema](../../../../src/crest/microcontrollers/stm32_nucleo_n657x0.py); preparation validates image/header identity, not the feasibility of arbitrary options.

Use complete prepared native projects in their original build locations, including boot/application binaries, weights, native storage sidecar and external loader. No project inventory or relocation is performed. Preparation records absolute paths and hashes; files remain external inputs. Native execution updates programming-cache fields in the storage sidecar. Keep separate prepared projects for each condition and avoid other processes mutating them. Weights size/hash, ELF protocol/hash and phase-header hash are checked; runtime phase-header rewriting and compilation are disabled. Both application and weights must be programmed on every accepted measurement.

A minimal device JSON uses native defaults for omitted pins/timeouts:

```json
{
  "device_id": "lab-device",
  "runtime_kwargs": {
    "serial_port": "/dev/ttyACM0",
    "harness_serial_port": "/dev/ttyACM1",
    "baud_rate": 115200
  }
}
```

Replace ports and identity for your setup. For custom wiring/timeouts, use the exact `_evaluate_single_phase` keyword schema in the [native backend](../../../../src/crest/microcontrollers/stm32_nucleo_n657x0.py).

Device JSON contains a canonical `device_id` and `runtime_kwargs` with explicit `serial_port`, `harness_serial_port`, baud rate, pins and timeouts expected by the native backend. Install matching harness firmware beforehand; harness auto-flashing is disabled. Run only one experiment at a time on a DUT/harness.

Accepted measurements resume with the original schedule and device settings. Failures preserve raw evidence and stop; inspect a failed/interrupted run before creating a new output directory. There is no automatic retry system. Acceptance retains policy-specific timing, energy, protocol, programming and sleep/deadline checks. Raw metrics and DUT/harness logs are retained in each result for paired offline analysis. The output study contains `study.json`, `schedule.json`, and `measurements/<id>/result.json` with `attempt-01/` raw records and logs. There is no policy-replay analysis subcommand. [CS1 policy analysis](../../waiting_policy.py) expects a JSON list with top-level `power_metrics`, accepted status, ten runs, and its own candidate/round/shape grouping; this runner nests metrics under `evidence`. It does not accept the study directory or result files directly. No adapter is supplied.
