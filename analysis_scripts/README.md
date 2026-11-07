<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->

# Analysis Scripts

This folder contains public-facing analysis utilities and publication-figure
generators that sit outside the core CREST runtime.

Run these scripts from the repository root with the `crest` Conda environment
active unless a package README says otherwise. Most scripts consume existing
NAS, replay, or measurement artifacts and do not run NAS or touch hardware.

The recipes below run training or collect hardware measurements. Hardware
measurement commands require explicit device identity and board readiness;
training requires `--execute`.
Offline analysis and publication scripts consume saved artifacts separately.

## Experiment recipes

- [Paired repeatability](repeatability/README.md): frozen randomized paired
  schedules, compile-once artifacts, and native measurement acceptance.
- [Waiting power](waiting_power/README.md): continuous and cadenced run-count
  probes with a separate offline power-analysis command.
- [Policy replay](policy_replay/README.md): native prepared-project policy comparisons
  with explicit deployment configuration.
- [Micro-workload energy probe](micro_workload_energy_probe/README.md): synthetic
  phase-energy measurements through the CREST HIL harness.

- [Training budget experiment](training_experiment/README.md): configurable
  OxIOD/TCN continuous training-budget recipe with historical energy inputs;
  training requires `--execute` and does not use HIL.

The repeatability, waiting-power, and policy runners share a physical-device
lock. Use the same device ID and lock root across runners. See each
runner's README for build dependencies and measurement safeguards.

## Case studies and appendices

| Folder | Analysis |
| --- | --- |
| [`cs1/`](cs1/README.md) | Waiting-power calibration, held-out prediction, selection, and paired replay |
| [`cs3/`](cs3/README.md) | Search trajectories, campaign effort, score sensitivity, saved evaluation, and replay |
| [`appendix_evaluation_cost/`](appendix_evaluation_cost/README.md) | Candidate evaluation timing and host/device cost breakdowns |
| [`appendix_training_budget/`](appendix_training_budget/README.md) | Checkpoint policies, ranking sensitivity, and training time |
| [`appendix_trial_outcomes/`](appendix_trial_outcomes/README.md) | Recorded trial outcomes and analysis populations |

Each guide lists inputs, commands, and outputs.

Other utilities include [Pareto comparison](compare_pareto_front_calcs/README.md),
the [hardware energy probe](micro_workload_energy_probe/README.md), and
[publication plots](paper_plots/README.md). The hardware probe requires a target
board and measurement harness; the analysis tools read saved artifacts.

The publication-plot guide includes the figure map, input schemas, reproduction
commands, and numeric sidecars.

## Case Study 3: proposal strategies and search effort

These tools analyze externally supplied study artifacts for the paper's audio
proposal-strategy comparison.

- `cs3/search_history.py` summarizes recorded trial outcomes, architecture
  coverage, proposal provenance, score progress and recorded API wall time.
  These support the CS3 comparison of search behavior and effort; they do not
  establish optimizer superiority across unmatched experiments.

Run from the repository root, using your own study artifacts:

```bash
PYTHONPATH=src python analysis_scripts/cs3/search_history.py \
  --history-db /path/to/study/optuna.db \
  --study-name audio_search \
  --ledger-dir /path/to/study/llm_optimizer \
  --output /path/to/analysis-output

```

`--ledger-dir` is optional for analysis without LLM request records. The analyzer
reads saved artifacts without launching search, contacting a provider, or accessing
hardware. Keep generated outputs outside the source checkout.

Search refreshes score or hypervolume plots automatically under each study's
`progress/` directory. To regenerate them from a saved database, use the
[shared progress Python API](../src/crest/search/README.md).
