<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->
# Waiting-power experiments

This package builds and runs **new STM32 hardware experiments** and analyzes their recorded evidence.

Run from the repository root, providing a candidate manifest with `candidates`, `timesteps`, and `input_dim` (candidate entries need `candidate_id`, `architecture`, and `clock_mhz`).

```sh
python -m analysis_scripts.cs1.experiments.waiting_power build --manifest /path/candidates.json --candidate a --config /path/device-and-data.yaml --output /path/new-study --clocks 200 400 --period-ms 250 --initialization-seed 7 --dry-run
```

Use the complete candidate manifest in [repeatability](../repeatability/README.md).
`--candidate a` selects its first entry.
The architecture must match the family in the [STM32 configuration](../../../../src/config/nas_config_stm32.yaml).
Adapt its data, project and toolchain paths before building.
Accepted architecture keys are defined by [OdomTCNFamily](../../../../src/crest/model_families/odom_tcn.py), and configuration fields by the [configuration guide](../../../../src/config/README.md).

Remove `--dry-run` to quantize the model to int8 and compile continuous, ten-run cadenced, and short cadenced variants for each clock.
`--low-runs` selects the short window (1–9).
Keep the selected release period and firmware protocol consistent throughout the experiment.
Build requires the configured calibration dataset, CREST runtime, firmware, ST Edge AI, and STM32/ARM build tools.

```sh
python -m analysis_scripts.cs1.experiments.waiting_power run --study /path/new-study --rounds 4 --seed 17 --clocks 200 400 --dut-port /path/dut --harness-port /path/harness --board-ready
python -m analysis_scripts.cs1.experiments.waiting_power analyze --study /path/new-study --output /path/analysis
```

`run --dry-run` prints the schedule without opening devices.
Real runs program app/weights and automatically update the harness run count.
Grouped scheduling reduces harness reflashes.
Shuffled scheduling is also available.
The grouped schedule ends with a ten-run block.
After an interrupted run, check the harness configuration before another experiment.

Run only one experiment at a time on a DUT/harness.
Accepted measurements are skipped on restart.
Schedule changes are refused.
Exceptions and interrupted attempts require inspection.
Retries remain bounded by `--max-attempts`.

Keep builds at their generated paths and rebuild after changing their inputs.

Offline `analyze` reads recorded results only.
It solves `W_k = k E + (T-C)[P_busy + (k-1)P_stop]`, with energies in mJ and times in seconds.
Output power is mW.
The default active time uses DWT cycles with RTC fallback (`--latency-source rtc` selects RTC).
Only complete accepted continuous/short-window/ten-run triples at the same round and clock enter the fit.
Missing or rejected members are excluded.
Invalid active times outside `0 < C < T` are reported and skipped.
`P_stop_eff_mw` folds wake margin, recovery and polling into the inter-release STOP interval.
`P_stop_sleep_only_mw` instead subtracts those awake intervals at fitted busy power and divides by recorded requested sleep time.
It is unavailable when that time is zero.
`release_mj_measured` is the observed awake-tail energy per release.
`release_mj_stop_tail` is a derived correction using the effective fit, not a measured STOP-tail deployment.

Measurements and logs are saved under `measurements/<id>/`.
Analysis writes `raw_measurements.csv`, `waiting_power_per_round.csv`, `waiting_power_by_clock.csv`, and `summary.json` to `--output` (default `<study>/analysis`).
Use this recipe’s `analyze` command for these results.
[CS1 calibration](../../README.md) requires different input tables.

```sh
python -m analysis_scripts.cs1.experiments.waiting_power --help
python -m pytest -q analysis_scripts/cs1/experiments/waiting_power/test_runner.py
```
