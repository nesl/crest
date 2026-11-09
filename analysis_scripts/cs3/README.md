<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->
# CS3 proposal strategies and evaluation

Offline analysis of supplied Optuna databases, saved responses and evaluations.
Requires Python and NumPy.
CSV sweeps also require pandas.
No training, inference or provider calls.
Keep outputs outside the checkout.

| Script | Output |
| --- | --- |
| `campaign.py` | Endpoint attempts, trial time, saved tokens and feasible observations |
| `scoring.py` | Exact selection and winning-pool weight intervals |
| `score_sensitivity.py` | Discrete weight/energy-scale sweeps and quality/energy reference selections |
| `evaluation.py` | Exact-parameter repeat variability and saved final-test metrics |
| `search_history.py` | Completion-prefix decoded-network trajectories, raw/decoded repeat dispersion and saved proposal accounting |
| `replay.py` | Source-selection identity checks and destination energy ranks |

```sh
python analysis_scripts/cs3/campaign.py --config /path/to/campaigns.json \
  --endpoint 200 --output-dir /path/to/results/campaign
python analysis_scripts/cs3/scoring.py \
  --observations /path/to/results/campaign/feasible_observations.csv \
  --output-dir /path/to/results/scoring
```

The config has a `campaigns` list.
Each entry provides `name`, `board`, `method`, `database`, and optionally `study_id`, `response_root`, and `ledger`.
Point `response_root` at the active optimizer directory, excluding backups.
Memory summaries beneath it are included.
Optional `mapping` changes field names.

Use `evaluation.py --group LABEL DATABASE TEST_METRICS_JSON --endpoint 200 --output-dir DIR` (repeat `--group` for other campaigns).
Use `replay.py --observations CSV --replay DESTINATION=CSV --output-dir DIR` for each supplied replay.
All scripts support `--help`.
Endpoints, score weights and energy scales are configurable where applicable.

Trial time sums terminal attempts through the feasible endpoint, including FAIL and PRUNED attempts.
API totals cover saved responses, which may span a different time window.
Missing usage is reported, not imputed.
Neither is campaign wall time.
Final-test metrics are read from saved evaluations, not recomputed from predictions.
Their schema does not establish historical fold/checkpoint policy.

Re-scoring collected trials does not simulate another search.
Replay carries source validation quality and reports recorded model provenance, without implying trained-checkpoint portability.
Exact score intervals retain their boundary inclusion rules and deterministic tie-breaking.

Use `search_history.py --history-db DATABASE --output DIR` for completion-prefix trajectories.
Select `--study-name NAME` when needed and optionally supply `--ledger-dir DIR`.
It requires the repository's `crest` package (or the local `src` tree).
Its decoded-network grouping differs from `evaluation.py`'s exact parameter dictionaries, and its raw-repeat dispersion remains separately labeled.

## Score sensitivity from CSV logs

Use `scoring.py` for exact weight intervals from campaign observations.
Use `score_sensitivity.py` to compare selections at specified weights and energy scales in your own CSV logs.

```sh
python analysis_scripts/cs3/score_sensitivity.py \
  --run "board-a=/path/to/trials.csv" \
  --budget-mj 200 --budget-mj 400 --lambda 0 --lambda 0.1 \
  --baseline-budget-mj 400 --baseline-lambda 0.1 \
  --output-dir /path/to/results/sensitivity
```

Repeat `--run` for other candidate pools.
Inputs may be CSVs or directories.
For directories, the tool chooses a usable top-level CSV.
Column detection is automatic.
To select columns explicitly, add options such as `--quality-col validation_macro_f1 --energy-col energy_mj_per_inference --trial-id-col trial_id --status-col state --feasible-col feasible`.
Use `--help` for the remaining column and accepted-value options.

Both tools score `quality - weight * energy / scale`.
The scale is a scoring reference, not a feasibility limit.
Sweeps require finite quality and energy, apply status/feasibility filters when those columns are present, and require a finite source score when its column is present.
They use floating-point scores, then quality descending, energy ascending, and string trial ID ascending to resolve ties.
Exact intervals use rational arithmetic and numeric trial IDs, explicit filters, and a configurable latency limit.
Match candidate populations and account for these tie rules before comparing their selections.

Sweeps write `selections.csv`, `summary.json`, `summary.md`, and `manifest.json`.
The manifest records selected input files, columns, filters, and sweep settings.
