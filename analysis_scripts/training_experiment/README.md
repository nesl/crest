<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->

# Training-budget experiment

`epoch_sensitivity.py` trains a selected panel from a saved OxIOD/TCN
back-to-back search run. It reuses CREST's candidate reconstruction, dataset
loader, model family, task fit plan, checkpoint callbacks and TFLite evaluation.
This is an experiment recipe with explicit source, dataset and output paths;
it does not run search, compile firmware or access hardware.

Run from the repository root in the CREST environment. Preparation needs PyYAML;
training needs the full CREST runtime, and analysis needs Matplotlib.

```bash
python analysis_scripts/training_experiment/epoch_sensitivity.py prepare \
  --source-run-dir /path/to/saved-run \
  --output-dir /path/to/new-experiment \
  --count 4 --budgets 15,30,55,100,200,300 --seeds 17

python analysis_scripts/training_experiment/epoch_sensitivity.py run \
  --experiment-dir /path/to/new-experiment \
  --dataset-dir /path/to/oxiod --execute

python analysis_scripts/training_experiment/epoch_sensitivity.py analyze \
  --experiment-dir /path/to/new-experiment
```

Preparation selects evenly spaced energy positions on the valid deduplicated
source Pareto front, including its extremes. Use `--source-rows 0 2 4` instead
of `--count` for an explicit panel that can include dominated candidates. Rows
are zero-based CSV data-row indices, not trial IDs. `--source-csv` and
`--source-config` override file discovery; `--expected-source-sha256` checks the
source CSV before creating output. Inspect the prepared manifest before training.
An existing output directory is refused.

Each candidate/seed follows one continuous trajectory. At every budget the
runner saves the lowest-`val_loss` checkpoint so far (`budget_only`) and the
checkpoint selected by the original early-stopping policy (`original_policy`).
The original policy freezes after virtual stopping while the live trajectory
continues. Evaluation happens after fitting, with each candidate's original
float/INT8 mode, using validation TFLite `rmse_total`. Calibration uses training
data; the held-out test split is never loaded. Identical checkpoint hashes share
validation evaluation within a trajectory.

Budgets must include 55 and a greater reference budget: the retained comparison
selects the front at budget 55 under the original policy and measures that
subset's reference-budget performance against the whole panel. The maximum
budget is a reference, not a convergence claim. Source energy stays fixed at
its historical measured value, so these are conditional fronts rather than
new checkpoint-specific deployment measurements. Front-only selection cannot
identify slow starters outside the panel; one seed gives descriptive evidence.

The experiment directory contains source/effective configurations, manifest and
execution identity, candidate/seed histories, checkpoints, exports, completion
or failure records, `checkpoint_metrics.csv`, `summary.json` and per-seed figures.
Analysis refuses incomplete or failed panels. Source artifacts stay read-only.
Execution requires the same Git revision as preparation and records train/valid
split-list hashes (not every sensor file). Keep the dataset immutable. Seeds
are set, but GPU bitwise determinism is not guaranteed.

`run` requires `--execute` before runtime imports or dataset access. Use
`--skip-completed` only with the same manifest and dataset identity; partial
trajectories are refused. There is no automatic mid-trajectory resume. Preserve
partial outputs and prepare a new directory for a documented restart. Training
also runs the single-panel analysis on completion. For offline analysis across
multiple completed batches, see [training budget](../training_budget/README.md).

```bash
python -m pytest -q test/test_training_experiment.py
```

Tests use synthetic sources, manually driven callbacks and fake fit events;
they do not launch training or hardware work. TensorFlow-dependent checks skip
when its runtime is unavailable.
