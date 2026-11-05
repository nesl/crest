# CS3 scoring and campaign audit

Offline analysis of supplied Optuna databases, saved responses and evaluations.
Requires Python and NumPy. No training, inference or provider calls. Research
data, responses and private path manifests are not distributed; keep outputs
outside the checkout.

| Script | Output |
| --- | --- |
| `campaign.py` | Endpoint attempts, trial time, saved tokens and feasible observations |
| `scoring.py` | Exact selection and winning-pool weight intervals |
| `evaluation.py` | Exact-parameter repeat variability and saved final-test metrics |
| `replay.py` | Source-selection identity checks and destination energy ranks |
| `verify_paper.py` | Optional reference checks on regenerated results |

```sh
python analysis_scripts/cs3_audit/campaign.py --config /path/to/campaigns.json \
  --endpoint 200 --output-dir /path/to/results/campaign
python analysis_scripts/cs3_audit/scoring.py \
  --observations /path/to/results/campaign/feasible_observations.csv \
  --output-dir /path/to/results/scoring
```

The private config has a `campaigns` list. Each entry provides `name`, `board`,
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
