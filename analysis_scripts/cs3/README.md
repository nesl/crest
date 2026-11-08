<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->
# CS3: proposal strategies and evaluation

Offline analysis of supplied Optuna databases, saved responses and evaluations.
Requires Python and NumPy. No training, inference or provider calls. Keep outputs
outside the checkout.

| Script | Output |
| --- | --- |
| `campaign.py` | Endpoint attempts, trial time, saved tokens and feasible observations |
| `scoring.py` | Exact selection and winning-pool weight intervals |
| `evaluation.py` | Exact-parameter repeat variability and saved final-test metrics |
| `search_history.py` | Completion-prefix decoded-network trajectories, raw/decoded repeat dispersion and saved proposal accounting |
| `replay.py` | Source-selection identity checks and destination energy ranks |
| `verify_paper.py` | Optional reference checks on regenerated results |

```sh
python analysis_scripts/cs3/campaign.py --config /path/to/campaigns.json \
  --endpoint 200 --output-dir /path/to/results/campaign
python analysis_scripts/cs3/scoring.py \
  --observations /path/to/results/campaign/feasible_observations.csv \
  --output-dir /path/to/results/scoring
```

The config has a `campaigns` list. Each entry provides `name`, `board`,
`method`, `database`, and optionally `study_id`, `response_root`, and `ledger`.
Point `response_root` at the active optimizer directory, excluding backups;
Memory summaries beneath it are included. Optional `mapping` changes field names.

Use `evaluation.py --group LABEL DATABASE TEST_METRICS_JSON --endpoint 200
--output-dir DIR` (repeat `--group` for other campaigns). Use `replay.py
--observations CSV --replay DESTINATION=CSV --output-dir DIR` for each supplied
replay. All scripts support `--help`; endpoints, score weights and energy scales
are configurable where applicable.

Trial time sums terminal attempts through the feasible endpoint, including FAIL
and PRUNED attempts. API totals cover saved responses, which may span a different
time window; missing usage is reported, not imputed. Neither is campaign wall
time. Final-test metrics are read from saved evaluations, not recomputed from
predictions; their schema does not establish historical fold/checkpoint policy.

Re-scoring collected trials does not simulate another search. Replay carries
source validation quality and reports recorded model provenance, without
implying trained-checkpoint portability. Exact score intervals retain their
boundary inclusion rules and deterministic tie-breaking.

Use `search_history.py --history-db DATABASE --output DIR` for completion-prefix
trajectories; select `--study-name NAME` when needed and optionally supply
`--ledger-dir DIR`. It requires the repository's `crest` package (or the local
`src` tree). Its decoded-network grouping differs from `evaluation.py`'s exact
parameter dictionaries, and its raw-repeat dispersion remains separately labeled.
