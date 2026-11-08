<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->
# Appendix: training-budget sensitivity

Analyze saved checkpoint evaluations and training histories. No training,
TensorFlow or hardware access is required. Install NumPy, SciPy and PyYAML.

```sh
python analysis_scripts/appendix_training_budget/analyze.py \
  --panels /path/to/batch-a /path/to/batch-b \
  --reference-budget 300 --seed 17 --output-dir /path/to/results
```

Each batch contains `manifest.json`, `execution.json`, source/effective YAML
configs, and `<candidate>/seed_<seed>/` with `history.json`, `results.json`,
`checkpoints.json`, and `complete.json`. Batches must describe compatible runs.
Candidate counts, budgets and seeds come from the manifests; use `--help` for
options. The submitted CS2 Portenta CM7 panel combines all four batches: 50 candidates.

Outputs: `summary.json` (correlations, fronts, shortlist regret and training time),
`checkpoint_metrics.csv`, and a plot-ready CSV per seed. Keep outputs private and
outside the checkout. Training times sum trajectory prefixes; they exclude later
checkpoint evaluation and hardware measurement.

For the submitted panel, run `verify_paper.py --summary /path/to/results/summary.json`.
Pass `figure10_training_budget.csv` to the existing training-budget plotter
using `--input-csv` to regenerate the figures. These checks reproduce results from saved
evaluations; they do not rerun inference. The selected panel and fixed historical
energy do not establish convergence or search-wide ranking reliability.
