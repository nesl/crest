<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->
# CS1 offline analysis

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

Use explicit input/output paths for new datasets. Prediction supports configurable
folds, repeats and quality ceilings. Supply pairings in external JSON files, with
each list entry shaped as `{"left": "candidate_a", "right": "candidate_b"}`.
Calibration assumes one/ten-release awake-final windows;
its periods and excluded pools are explicit options. Input column names follow
the saved CREST measurement tables. Graph grouping currently supports TCN logs.

## Submitted-paper reproduction

```sh
python analysis_scripts/cs1_offline/reproduce_paper.py \
  --archive-root /path/to/research-archive --output-dir /path/to/results \
  --policy-pairs /path/to/policy_pairs.json \
  --repeat-pairs /path/to/repeat_pairs.json
```

Keep the pairing files with the research archive and use the original pairings
when checking the paper results. The recipe selects the original data files and
settings, including three-decimal calibration export rounding. Add `--source-root` if the original candidate logs
have moved. `verify_paper.py --results /path/to/results` checks printed values and
counts; only this recipe/checker requires the submitted dataset. Results include
CSVs, input hashes, short reports and `paper_checks.csv`.

Selection regret uses the lowest saved energy in the same eligible bank. Repeated
folds/ceilings are correlated. Paired intervals describe within-session repeats;
estimated STOP intervals condition on the calibration. These scripts regenerate
analysis results from saved observations, not new hardware measurements.
