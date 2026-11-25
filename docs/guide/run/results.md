# Read study results

Start in `<outputs.models_dir>/<study_name>/`. The shipped examples use `models/` as the root. Smoke tests add the device name to the study directory.

| File | Use it to… |
| --- | --- |
| Copied YAML | Recover the workload, search budget, objectives, and measurement settings. CLI smoke settings are applied at runtime. |
| `optuna.db` or `optuna_smoke_test.db` | Inspect persistent trial state, values, parameters, and user attributes. |
| CSV named by `outputs.log_file_name` | Read the evaluation log, including hardware metrics, objective metadata, and `metric__*` and `hparam__*` columns. |
| `trials.csv` | Read the Optuna trial data frame exported at full-search closeout. It includes the trial number and state. |
| `<study_name>_pareto.csv` in `outputs.models_dir` | Find the exported front after a full multi-objective search. |
| `train_history.json` and `summary.json` | Review scalar-run retraining and evaluation when finalization succeeds. |

Candidate build and staging files use `outputs.candidate_dir`. Keep completed study artifacts together with the source revision and any evidence needed for subsequent analysis.

## Check which trials count

- Compare the attempted trials to the configured attempt cap and the feasible completions to the requested target. A full search can hit the cap before reaching the target.
- Check Optuna state alongside feasibility, prune reason, and hardware status. `COMPLETE` means Optuna received an objective outcome. Deployment feasibility depends on the configured checks.
- Verify each reused objective is finite and recorded for the intended runtime phase. When comparing continuous and cadenced energy, keep the measurement window, cadence, units, and filters explicit.
- For a multi-objective study, review the directions before interpreting front points. For a scalar study, use the summary and held-out evaluation artifacts to distinguish search-time selection from final evaluation.

To extend a compatible study, retain the YAML, study name, and database and increase the total budgets. Start a new study when changing the optimizer, objectives, or deployment settings. [Configuration reference](https://github.com/nesl/crest/blob/main/src/config/README.md).

[Reproduce an analysis or replay selected candidates](reproduce.md).
