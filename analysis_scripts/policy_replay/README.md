<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->
# Policy replay recipe

Replay prepared STM32 continuous, cadenced awake-tail and cadenced STOP-tail images in randomized pair blocks. Preparation and planning open no hardware. This recipe imports successful native build metadata; it does not export models, patch firmware or build STOP-tail firmware. Supply separately prepared images containing the selected `rtc_wall_v4_dwt_active_stop_evidence_awake_tail` or `rtc_wall_v4_dwt_active_stop_evidence_stop_tail` protocol. The repository's awake-only firmware cannot produce STOP-tail images without a separately controlled preparation workflow.

```sh
python -m analysis_scripts.policy_replay.replay prepare --manifest /path/prepared.json --output /path/new-output
python -m analysis_scripts.policy_replay.replay plan --study /path/new-output --rounds 5 --seed 42
python -m analysis_scripts.policy_replay.replay run --study /path/new-output --rounds 5 --seed 42 --device-config /path/device.json --board-ready
```

The JSON manifest contains `schema_version: 1`, `variants: [...]` and `blocks: [["continuous", "awake", "stop"]]`. Blocks partition variant names exactly once. Each variant has `name`, `candidate`, `phase` (`back_to_back` or `cadenced`), `policy` (`awake` or `stop`), `runs`, `clock_mhz`, `period_ms`, native `device_options`, `window_size`, `num_channels`, `project`, `phase_header` (relative to project), `phase_header_sha256`, `app_elf_sha256`, and the native successful `compile_result` dictionary. Project and compile-result paths resolve relative to the manifest. Use 10 measured runs, 5000 us wake margin and minimum sleep for the original protocol. Continuous uses the awake image; the phase header cadence is still checked.

Use complete prepared native projects in their original build locations, including boot/application binaries, weights, native storage sidecar and external loader. No project inventory or relocation is performed. Preparation records absolute paths and hashes; files remain external inputs. Native execution updates programming-cache fields in the storage sidecar. Keep separate prepared projects for each condition and avoid other processes mutating them. Weights size/hash, ELF protocol/hash and phase-header hash are checked; runtime phase-header rewriting and compilation are disabled. Both application and weights must be programmed on every accepted measurement.

Device JSON contains a canonical `device_id` and `runtime_kwargs` with explicit `serial_port`, `harness_serial_port`, baud rate, pins and timeouts expected by the native backend. Install matching harness firmware beforehand; harness auto-flashing is disabled. Use the same device ID across all experiment runners. `--lock-dir`, `CREST_DEVICE_LOCK_DIR`, or the shared default select the cross-runner device lock.

Accepted measurements resume with the original schedule and device settings. Failures preserve raw evidence and stop; inspect a failed/interrupted run before creating a new output directory. There is no automatic retry system. Acceptance retains policy-specific timing, energy, protocol, programming and sleep/deadline checks. Raw metrics and DUT/harness logs are retained in each result for paired offline analysis. No hardware execution has been validated here.
