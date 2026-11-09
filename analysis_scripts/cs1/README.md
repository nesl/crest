<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->
# CS1 waiting power and model selection

The analysis commands below use saved energy measurements without training or hardware access.
Hardware experiment recipes appear under [Collect new measurements](#collect-new-measurements).
The saved-measurement analyses require Python, NumPy, pandas and SciPy.
Inputs and generated results stay outside this package.
The saved-measurement analysis commands support `--help` and record input hashes.
Use [saved-measurement analysis](#analyze-saved-measurements) for existing data or [collection setup](#collect-new-measurements) before choosing a hardware recipe.

| Script | Purpose |
| --- | --- |
| `calibration.py` | Two-period waiting fit and candidate predictions |
| `heldout_prediction.py` | Graph-separated prediction, bootstrap and selection |
| `raw_selection_baseline.py` | Raw energy/latency selection baselines |
| `paired_transfer.py` | Paired prediction residuals and timing substitutions |
| `waiting_policy.py` | Paired awake/STOP-policy contrasts |
| `repeats.py` | Search-pair repeats and estimated STOP correction |
| `direct_session.py` | Direct contrasts, waiting slopes and accounting |
| `scaling_periods.py` | Clock scaling and measured energy optima |

Follow the [common command setup](../README.md).
Relative input/output paths resolve from the current working directory.
Use explicit input/output paths for new datasets.
Prediction supports configurable folds, repeats and quality ceilings.
Supply pairings in external JSON files, with each list entry shaped as `{"left": "candidate_a", "right": "candidate_b"}`.
Calibration assumes one/ten-release awake-final windows.
Its periods and excluded pools are explicit options.
Input column names follow the saved CREST measurement tables.
Graph grouping currently supports TCN logs.

## Analyze saved measurements

```sh
python analysis_scripts/cs1/calibration.py \
  --probe-long /path/to/probe_500.csv --probe-short /path/to/probe_100.csv \
  --candidates /path/to/candidates.csv --latencies /path/to/latencies.csv \
  --periods-ms 500 100 --output-dir /path/to/results/calibration
python analysis_scripts/cs1/heldout_prediction.py \
  --predictions /path/to/results/calibration/predictions.csv \
  --observations /path/to/observations.csv \
  --archive-root /path/to/candidate-logs \
  --output-dir /path/to/results/heldout
python analysis_scripts/cs1/raw_selection_baseline.py \
  --predictions /path/to/results/calibration/predictions.csv \
  --output-dir /path/to/results/raw_selection
python analysis_scripts/cs1/waiting_policy.py \
  --energy-records /path/to/policy_energy_records.json \
  --pairs /path/to/policy_pairs.json --output-dir /path/to/results/policy
python analysis_scripts/cs1/repeats.py \
  --repeated-observations /path/to/repeated_observations.csv \
  --joint-fit /path/to/joint_fit.csv --pairs /path/to/repeat_pairs.json \
  --output-dir /path/to/results/repeats
```

The calibration command writes `calibration.csv`, `predictions.csv`, `probe_joint_fit_per_round.csv`, and input provenance.
Use `--fit-decimals` to choose coefficient export rounding and repeat `--exclude-pool` to exclude pools.
The held-out analysis uses candidate source logs for graph grouping.
Its `--archive-root` relocates saved source paths when needed.
Repeat analysis takes a frozen joint-fit CSV with one row per `clock_mhz` and columns `P_busy_mw_mean`, `P_stop_mw_mean`, and `K_mj_mean`.
Keep pairing files alongside their input datasets.
Results include CSVs, input hashes and short reports.

Selection regret uses the lowest saved energy in the same eligible bank.
Repeated folds/ceilings are correlated.
Paired intervals describe within-session repeats.
Estimated STOP intervals condition on the calibration.
These scripts regenerate analysis results from saved observations, not new hardware measurements.

## Collect new measurements

**The three CS1 hardware recipes below have not been updated for the current STM32 runtime and are incompatible with it.**
For new board runs, start with the current [HIL workflow](../../src/GETTING_STARTED.md#running-nas-and-hil).
Use these recipes only with a compatible historical runtime and matching firmware protocol.

The [repeatability](#paired-repeatability-experiments), [waiting-power](#waiting-power-experiments), and [waiting-policy](#waiting-policy-experiments) recipes run new STM32 experiments.
Each recipe describes preparation, hardware requirements, outputs, and the inputs needed for subsequent analysis.
Run commands from the repository root in the `crest` environment and reserve the DUT/harness for one experiment at a time.
Supply the configured device ports and acknowledge board readiness with `--board-ready`.
Builds require the calibration dataset, CREST runtime, firmware template, ST Edge AI, and STM32/ARM toolchain when a recipe builds model artifacts.
Keep prepared builds at their original paths with inputs unchanged, and rebuild after moving them or changing model, configuration, or toolchain.

### Paired repeatability experiments

Run paired STM32 measurements with a fixed randomized schedule and reusable compiled models.
This recipe initializes model weights.
It does not train models.

```sh
python -m analysis_scripts.cs1.experiments.repeatability prepare --manifest /path/candidates.json --output /path/new-study --rounds 4 --seed 17
python -m analysis_scripts.cs1.experiments.repeatability build-artifacts --study /path/new-study --config /path/device-and-data.yaml
python -m analysis_scripts.cs1.experiments.repeatability run --study /path/new-study --dut-port /path/dut --harness-port /path/harness --board-ready
```

The caller supplies all candidate identities, architectures, clocks, input dimensions, cadence, run count, and initialization seed.
This small manifest uses the supported `odom_tcn` architecture schema.

```json
{
  "prospective_protocol": "rtc_wall_v4_dwt_active_stop_evidence_awake_tail",
  "timesteps": 16,
  "input_dim": 3,
  "period_ms": 250,
  "measured_runs": 4,
  "initialization_seed": 7,
  "candidates": [
    {"candidate_id": "a", "architecture": {"nb_filters": 8, "kernel_size": 3, "dropout_rate": 0.0, "use_skip_connections": true, "norm_flag": false, "dilations": [1, 2]}, "clock_mhz": 200},
    {"candidate_id": "b", "architecture": {"nb_filters": 8, "kernel_size": 3, "dropout_rate": 0.0, "use_skip_connections": true, "norm_flag": false, "dilations": [1, 2]}, "clock_mhz": 400}
  ],
  "pairs": [{"pair_id": "pair", "candidate_a": "a", "candidate_b": "b"}]
}
```

Architecture keys are defined by [OdomTCNFamily.validate_hparams](../../src/crest/model_families/odom_tcn.py).
Use a configuration selecting that family, such as the [STM32 OxIOD configuration](../../src/config/nas_config_stm32.yaml), adapted to your calibration data, dimensions, project and toolchain paths.
The [configuration guide](../../src/config/README.md) defines the accepted YAML settings.
Clocks and cadence must also suit your board and workload.
These values do not establish hardware feasibility.

Each candidate belongs to exactly one pair.
Each round measures both continuous and cadenced phases for every candidate.
AB/BA order is balanced across even round counts, with one randomized extra order for odd counts.
`--max-attempts` bounds retries across invocations.
`run --max-evaluations` bounds work per invocation.
Accepted evaluations are skipped on restart.
Exceptions and uncertain interruptions require inspection before resuming.
Evidence is retained.

`build-artifacts` initializes weights, quantizes models to int8, and compiles each measurement phase.
To use existing prepared bundles, run `bind-artifacts --artifact-root /path/bundles` before starting measurements.

Harness firmware must already match the manifest's run count.
This runner disables automatic harness flashing.
Diagnostic profiling is outside this protocol.

Find compiled bundles under `artifacts/<candidate>/` and measurements, results, and logs under `evaluations/<round>/<candidate>/attempt-*/`.
This recipe has no analysis subcommand.
To use [CS1 repeatability analysis](#analyze-saved-measurements), first convert the measurements to its documented CSV inputs.
It does not read this study directory directly.

Safe checks

```sh
python -m analysis_scripts.cs1.experiments.repeatability --help
python -m pytest -q analysis_scripts/cs1/experiments/repeatability/test_runner.py
```

### Waiting-power experiments

This package builds and runs **new STM32 hardware experiments** and analyzes their recorded evidence.

Provide a candidate manifest with `candidates`, `timesteps`, and `input_dim` (candidate entries need `candidate_id`, `architecture`, and `clock_mhz`).

```sh
python -m analysis_scripts.cs1.experiments.waiting_power build --manifest /path/candidates.json --candidate a --config /path/device-and-data.yaml --output /path/new-study --clocks 200 400 --period-ms 250 --initialization-seed 7 --dry-run
```

Use the complete candidate manifest in [repeatability](#paired-repeatability-experiments).
`--candidate a` selects its first entry.
The architecture must match the family in the [STM32 configuration](../../src/config/nas_config_stm32.yaml).
Adapt its data, project and toolchain paths before building.
Accepted architecture keys are defined by [OdomTCNFamily](../../src/crest/model_families/odom_tcn.py), and configuration fields by the [configuration guide](../../src/config/README.md).

Remove `--dry-run` to quantize the model to int8 and compile continuous, ten-run cadenced, and short cadenced variants for each clock.
`--low-runs` selects the short window (1–9).
Keep the selected release period and firmware protocol consistent throughout the experiment.

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

Accepted measurements are skipped on restart.
Schedule changes are refused.
Exceptions and interrupted attempts require inspection.
Retries remain bounded by `--max-attempts`.

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
[CS1 calibration](#analyze-saved-measurements) requires different input tables.

```sh
python -m analysis_scripts.cs1.experiments.waiting_power --help
python -m pytest -q analysis_scripts/cs1/experiments/waiting_power/test_runner.py
```

### Waiting-policy experiments

Compare prepared STM32 continuous, cadenced awake-tail, and cadenced STOP-tail deployments in randomized blocks.
Preparation and planning do not access hardware.
Supply separately built images with the appropriate `rtc_wall_v4_dwt_active_stop_evidence_awake_tail` or `rtc_wall_v4_dwt_active_stop_evidence_stop_tail` protocol.
This recipe does not build or modify firmware.

```sh
python -m analysis_scripts.cs1.experiments.waiting_policy.runner prepare --manifest /path/prepared.json --output /path/new-output
python -m analysis_scripts.cs1.experiments.waiting_policy.runner plan --study /path/new-output --rounds 5 --seed 42
python -m analysis_scripts.cs1.experiments.waiting_policy.runner run --study /path/new-output --rounds 5 --seed 42 --device-config /path/device.json --board-ready
```

The JSON manifest contains `schema_version: 1`, `variants: [...]` and `blocks: [["continuous", "awake", "stop"]]`.
Blocks partition variant names exactly once.
Each variant has `name`, `candidate`, `phase` (`back_to_back` or `cadenced`), `policy` (`awake` or `stop`), `runs`, `clock_mhz`, `period_ms`, native `device_options`, `window_size`, `num_channels`, `project`, `phase_header` (relative to project), `phase_header_sha256`, `app_elf_sha256`, and the native successful `compile_result` dictionary.
Project and compile-result paths resolve relative to the manifest.
Use 10 measured runs, 5000 us wake margin and minimum sleep for this protocol.
Continuous uses the awake image.
The phase header cadence is still checked.

For `compile_result`, preserve the successful [CompileResult](../../src/crest/devices.py) returned by the build, serialized as a dictionary with paths converted to strings.
See [repeatability preparation](experiments/repeatability/experiment.py) for an example.
Use the [STM32 backend options](../../src/crest/microcontrollers/stm32_nucleo_n657x0.py) for `device_options`.

Keep a separate, complete prepared project for each condition at its original build location, including binaries, weights, storage metadata, and external loader.
Leave these projects unchanged between preparation and measurement.

A minimal device JSON uses native defaults for omitted pins/timeouts.

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

Replace the ports and identity for your setup.
The [STM32 backend](../../src/crest/microcontrollers/stm32_nucleo_n657x0.py) documents optional wiring and timeout settings.

Install matching harness firmware before running.
Automatic harness flashing is disabled.

Restart with the original schedule and device settings to skip accepted measurements.
Failures preserve evidence and stop without automatic retries.
Inspect an interrupted or failed run before starting a new output directory.
Results are saved in `measurements/<id>/result.json`, with raw records and logs under `attempt-01/`.
This recipe has no analysis subcommand.
[CS1 policy analysis](#analyze-saved-measurements) requires its documented input format and does not read this study directory directly.
