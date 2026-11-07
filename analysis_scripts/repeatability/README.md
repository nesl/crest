<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->
# Paired repeatability experiments

This package runs **new hardware experiments**, with a frozen randomized schedule and compile-once STM32 artifacts. It does not train models. For offline reanalysis of saved measurements, see [CS1 offline analysis](../cs1_offline/README.md).

Run from the repository root:

```sh
python -m analysis_scripts.repeatability prepare --manifest /path/candidates.json --output /path/new-study --rounds 4 --seed 17
python -m analysis_scripts.repeatability build-artifacts --study /path/new-study --config /path/device-and-data.yaml
python -m analysis_scripts.repeatability run --study /path/new-study --device-id lab-device --dut-port /path/dut --harness-port /path/harness --board-ready
```

The caller supplies all candidate identities, architectures, clocks, input dimensions, cadence, run count, and initialization seed. The manifest has this shape; replace the architecture objects with valid CREST architecture specifications:

```json
{
  "prospective_protocol": "rtc_wall_v4_dwt_active_stop_evidence_awake_tail",
  "timesteps": 16,
  "input_dim": 3,
  "period_ms": 250,
  "measured_runs": 4,
  "initialization_seed": 7,
  "candidates": [
    {"candidate_id": "a", "architecture": {}, "clock_mhz": 80},
    {"candidate_id": "b", "architecture": {}, "clock_mhz": 160}
  ],
  "pairs": [{"pair_id": "pair", "candidate_a": "a", "candidate_b": "b"}]
}
```

Each candidate belongs to exactly one pair. Each round measures both continuous and cadenced phases for every candidate. AB/BA order is balanced across even round counts, with one randomized extra order for odd counts. `--max-attempts` bounds retries across invocations; `run --max-evaluations` bounds work per invocation. Accepted evaluations are skipped on restart. Exceptions and uncertain interruptions require inspection before resuming; evidence is retained.

`build-artifacts` lazily uses the repository's `hil_server.HILServer` model/calibration preparation and native STM32 export/compiler APIs. It initializes weights deterministically, performs int8 PTQ, and compiles each phase without device evaluation. It requires the configured calibration dataset, TensorFlow/NumPy and CREST dependencies, firmware template, ST Edge AI, and ARM/STM32 toolchain. Missing dependencies fail explicitly. `bind-artifacts --artifact-root /path/bundles` can instead bind existing verified bundles before any attempt.

Prepared bundle manifests hash model inputs, compiled artifacts, backend code, and tools. Absolute generated paths make a prepared bundle host/path bound: build a new bundle after moving it. Runtime programming caches and sidecars are changed only in disposable attempt workspaces; frozen inputs are verified again after each phase. Acceptance requires matching raw native records, successful app and weight programming, protocol/clock/run identity, cadence timing, and deadline/STOP checks.

All hardware runners sharing a DUT/harness must use the **same `--device-id`** and shared lock root. `--lock-dir` or `CREST_DEVICE_LOCK_DIR` overrides the common per-user default. Port aliases do not establish device identity. The shared lock covers the entire hardware run. Harness firmware must already match the manifest's run count; this runner disables automatic harness flashing. Diagnostic profiling is outside this protocol.

Safe checks:

```sh
python -m analysis_scripts.repeatability --help
python -m pytest -q analysis_scripts/repeatability/test_runner.py
```
