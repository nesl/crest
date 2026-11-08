<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->
# Appendix training-budget sensitivity

To collect new training trajectories, use the [training-budget experiment](#training-budget-experiment).

## Analyze saved checkpoint evaluations

Analyze saved checkpoint evaluations and training histories.
No training, TensorFlow or hardware access is required.
Install NumPy, SciPy and PyYAML.

```sh
python analysis_scripts/appendix_training_budget/analyze.py \
  --panels /path/to/batch-a /path/to/batch-b \
  --reference-budget 300 --seed 17 --output-dir /path/to/results
```

Each batch contains `manifest.json`, `execution.json`, source/effective YAML configs, and `<candidate>/seed_<seed>/` with `history.json`, `results.json`, `checkpoints.json`, and `complete.json`.
Batches must describe compatible runs.
Candidate counts, budgets and seeds come from the manifests.
Use `--help` for options.

Outputs.
`summary.json` (correlations, fronts, shortlist regret and training time), `checkpoint_metrics.csv`, and a plot-ready CSV per seed.
Keep generated outputs outside the checkout.
Training times sum trajectory prefixes.
They exclude later checkpoint evaluation and hardware measurement.

Pass `figure10_training_budget.csv` to the existing training-budget plotter using `--input-csv` to regenerate the figures.
These analyses use saved evaluations.
They do not rerun inference.
The selected panel and fixed historical energy do not establish convergence or search-wide ranking reliability.

## Training-budget experiment

`experiments/epoch_sensitivity.py` trains selected candidates from a saved OxIOD/TCN back-to-back search run and compares their performance at different epoch budgets.
It requires explicit source, dataset, and output paths.
It does not run search or access hardware.

Run from the repository root in the CREST environment.
Preparation needs PyYAML.
Training needs the full CREST runtime, and analysis needs Matplotlib.

```bash
python analysis_scripts/appendix_training_budget/experiments/epoch_sensitivity.py prepare \
  --source-run-dir /path/to/saved-run \
  --output-dir /path/to/new-experiment \
  --count 4 --budgets 15,30,55,100,200,300 --seeds 17

python analysis_scripts/appendix_training_budget/experiments/epoch_sensitivity.py run \
  --experiment-dir /path/to/new-experiment \
  --dataset-dir /path/to/oxiod --execute

python analysis_scripts/appendix_training_budget/experiments/epoch_sensitivity.py analyze \
  --experiment-dir /path/to/new-experiment
```

Preparation selects evenly spaced energy positions on the valid deduplicated source Pareto front, including its extremes.
Use `--source-rows 0 2 4` instead of `--count` for an explicit panel that can include dominated candidates.
Rows are zero-based CSV data-row indices, not trial IDs.
`--source-csv` and `--source-config` override file discovery.
`--expected-source-sha256` checks the source CSV before creating output.
Inspect the prepared manifest before training.
An existing output directory is refused.

Each candidate/seed follows one continuous training trajectory.
At every budget, the recipe compares the best validation-loss checkpoint so far (`budget_only`) with the checkpoint selected by the original early-stopping policy (`original_policy`).
The latter stays fixed once that policy would have stopped.
Evaluation uses validation TFLite `rmse_total` in the candidate’s original float/INT8 mode.
Calibration uses training data.
The held-out test split is never loaded.

Budgets must include 55 and a greater reference budget.
The retained comparison selects the front at budget 55 under the original policy and measures that subset's reference-budget performance against the whole panel.
The maximum budget is a reference, not a convergence claim.
Source energy stays fixed at its historical measured value, so these are conditional fronts rather than new checkpoint-specific deployment measurements.
Front-only selection cannot identify slow starters outside the panel.
One seed gives descriptive evidence.

The experiment directory contains configurations, candidate/seed histories, checkpoints, exports, `checkpoint_metrics.csv`, `summary.json`, and per-seed figures.
Analysis requires a complete, successful panel.
Keep the dataset and source artifacts unchanged, and use the same Git revision for preparation and execution.
GPU bitwise determinism is not guaranteed.

`run` requires `--execute`.
Use `--skip-completed` only with the same manifest and dataset.
Partial trajectories cannot resume.
Preserve partial outputs and start a new directory.
Training runs analysis automatically on completion.
The `analyze` command above can regenerate it.
The [training-budget appendix analysis](#analyze-saved-checkpoint-evaluations) uses separate input tables.

```bash
python -m pytest -q test/test_training_experiment.py
```
