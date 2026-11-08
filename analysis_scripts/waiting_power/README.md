<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->
# Waiting-power experiments

This package builds and runs **new STM32 hardware experiments** and analyzes their recorded evidence.

Run from the repository root, providing a candidate manifest with `candidates`, `timesteps`, and `input_dim` (candidate entries need `candidate_id`, `architecture`, and `clock_mhz`):

```sh
python -m analysis_scripts.waiting_power build --manifest /path/candidates.json --candidate a --config /path/device-and-data.yaml --output /path/new-study --clocks 200 400 --period-ms 250 --initialization-seed 7 --dry-run
```

Use the complete candidate manifest in [repeatability](../repeatability/README.md); `--candidate a` selects its first entry. The architecture must match the family in the [STM32 configuration](../../src/config/nas_config_stm32.yaml); adapt its data, project and toolchain paths before building. Accepted architecture keys are defined by [OdomTCNFamily](../../src/crest/model_families/odom_tcn.py), and configuration fields by the [configuration guide](../../src/config/README.md).

Remove `--dry-run` to export initialized int8 PTQ weights once and compile continuous/ten-run/short-run variants for every clock. `--low-runs` selects the short cadenced window (1–9). The ten-run reference and firmware protocol are measurement-protocol constraints. The release period is frozen in study metadata and never inferred during measurement or analysis. `--firmware` selects the source used for protocol and provenance checks; the configured backend project must use the same protocol. Build requires the explicit configuration's calibration dataset, TensorFlow/NumPy and CREST dependencies, firmware, ST Edge AI, and STM32/ARM build tools. Missing dependencies are reported as failures; no substitutes are supplied.

```sh
python -m analysis_scripts.waiting_power run --study /path/new-study --rounds 4 --seed 17 --clocks 200 400 --device-id lab-device --dut-port /path/dut --harness-port /path/harness --board-ready
python -m analysis_scripts.waiting_power analyze --study /path/new-study --output /path/analysis
```

`run --dry-run` prints the schedule without opening devices. Real runs program app/weights and automatically update the harness run count. Grouped scheduling reduces harness reflashes; shuffled scheduling is also available. The grouped schedule ends with a ten-run block. After an interrupted run, check the harness configuration before another experiment.

The repeatability, waiting-power and policy-replay runners using the same DUT/harness must use the **same `--device-id`** and lock root. `--lock-dir` or `CREST_DEVICE_LOCK_DIR` overrides the common per-user root. Accepted measurements are skipped on restart; schedule changes are refused. Exceptions and interrupted attempts require inspection. Retries remain bounded by `--max-attempts`.

The runner reuses compiled
variants, checks phase headers, clears native programming caches so app and
weights are programmed, and validates native acceptance flags, protocol, clock,
run count, STOP entries, and deadlines. CREST writes programming metadata into
the built workspace. Keep builds at their generated paths and rebuild after
changing their inputs.

Offline `analyze` reads recorded results only. It solves `W_k = k E + (T-C)[P_busy + (k-1)P_stop]`, with energies in mJ and times in seconds; output power is mW. The default active time uses DWT cycles with RTC fallback (`--latency-source rtc` selects RTC). Only complete accepted continuous/short-window/ten-run triples at the same round and clock enter the fit. Missing or rejected members are excluded; invalid active times outside `0 < C < T` are reported and skipped. `P_stop_eff_mw` folds wake margin, recovery and polling into the inter-release STOP interval. `P_stop_sleep_only_mw` instead subtracts those awake intervals at fitted busy power and divides by recorded requested sleep time; it is unavailable when that time is zero. `release_mj_measured` is the observed awake-tail energy per release. `release_mj_stop_tail` is a derived correction using the effective fit, not a measured STOP-tail deployment.

Build metadata and variants live under `--output`; runs retain `measurements/<id>/result.json` and attempt evidence there. Analysis writes `raw_measurements.csv`, `waiting_power_per_round.csv`, `waiting_power_by_clock.csv`, and `summary.json` into `--output` (default: `<study>/analysis`). This `analyze` command is the compatible entrypoint for these native results. [CS1 calibration](../cs1/calibration.py) requires separately shaped probe/candidate/latency tables; these output CSVs are not direct inputs and no adapter is supplied. Diagnostic profiling remains separate from these measurements.

```sh
python -m analysis_scripts.waiting_power --help
python -m pytest -q analysis_scripts/waiting_power/test_runner.py
```
