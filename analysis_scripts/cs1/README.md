<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->
# CS1: waiting power and model selection

Reusable analysis of saved energy measurements; no training or hardware access.
Requires Python, NumPy, pandas and SciPy. Inputs and generated results stay outside
this package. Each command supports `--help` and records input hashes.

| Script | Purpose |
| --- | --- |
| `calibration.py` | Two-period waiting fit and candidate predictions |
| `heldout_prediction.py` | Graph-separated prediction, bootstrap and selection |
| `raw_selection_baseline.py` | Raw energy/latency selection baselines |
| `paired_transfer.py` | Paired prediction residuals and timing substitutions |
| `policy_replay.py` | Paired awake/STOP-policy contrasts |
| `repeats.py` | Search-pair repeats and estimated STOP correction |
| `direct_session.py` | Direct contrasts, waiting slopes and accounting |
| `scaling_periods.py` | Clock scaling and measured energy optima |

Run the commands below from the repository root. Relative input/output paths
resolve from the current working directory. Use explicit input/output paths for new datasets. Prediction supports configurable
folds, repeats and quality ceilings. Supply pairings in external JSON files, with
each list entry shaped as `{"left": "candidate_a", "right": "candidate_b"}`.
Calibration assumes one/ten-release awake-final windows;
its periods and excluded pools are explicit options. Input column names follow
the saved CREST measurement tables. Graph grouping currently supports TCN logs.

## Explicit-input analysis

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
python analysis_scripts/cs1/policy_replay.py \
  --energy-records /path/to/policy_energy_records.json \
  --pairs /path/to/policy_pairs.json --output-dir /path/to/results/policy
python analysis_scripts/cs1/repeats.py \
  --repeated-observations /path/to/repeated_observations.csv \
  --joint-fit /path/to/joint_fit.csv --pairs /path/to/repeat_pairs.json \
  --output-dir /path/to/results/repeats
```

The calibration command writes `calibration.csv`, `predictions.csv`,
`probe_joint_fit_per_round.csv`, and input provenance. Use `--fit-decimals` to
choose coefficient export rounding and repeat `--exclude-pool` to exclude pools.
The held-out analysis uses candidate source logs for graph grouping; its
`--archive-root` relocates saved source paths when needed. Repeat analysis takes
a frozen joint-fit CSV with one row per `clock_mhz` and columns
`P_busy_mw_mean`, `P_stop_mw_mean`, and `K_mj_mean`.
Keep pairing files alongside their input datasets. Results include CSVs, input
hashes and short reports.

Selection regret uses the lowest saved energy in the same eligible bank. Repeated
folds/ceilings are correlated. Paired intervals describe within-session repeats;
estimated STOP intervals condition on the calibration. These scripts regenerate
analysis results from saved observations, not new hardware measurements.
