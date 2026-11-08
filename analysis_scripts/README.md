<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->

# Experiments and analysis

Use these tools to collect CREST measurements, analyze saved results, and
reproduce the paper's data figures. Supply your own datasets, study databases,
logs, and output paths. Run commands from the repository root with the `crest`
Conda environment active unless a guide specifies otherwise.

## 1. Collect measurements

| Tool | Purpose |
| --- | --- |
| [Paired repeatability](cs1/experiments/repeatability/README.md) | Randomized paired measurements using compiled STM32 artifacts |
| [Waiting power](cs1/experiments/waiting_power/README.md) | Continuous and cadenced probes for waiting-power calibration |
| [Waiting-policy experiments](cs1/experiments/waiting_policy/README.md) | Compare prepared continuous, awake-final, and STOP-final deployments |
| [Micro-workload probe](appendix_measurement_context/experiments/micro_workload_energy_probe/README.md) | Fixed-duration board measurements for sleep and compute phases |
| [Training budget](appendix_training_budget/experiments/README.md) | Train and evaluate candidates at different checkpoint budgets |

Hardware runners require a target board, measurement harness, and device setup
from their guides. Run only one experiment at a time on a DUT/harness. The
repeatability, waiting-power, and policy runners require explicit serial ports
and `--board-ready`. Training requires `--execute` and does not use HIL.

The [measurement-context guide](appendix_measurement_context/README.md)
introduces the synthetic workload probes. The CS1 hardware recipes use the
measurement protocol specified in their guides; they have not been updated for
the current STM32 runtime. For new board runs, start with the main
[HIL workflow](../src/GETTING_STARTED.md#running-nas-and-hil).

## 2. Analyze saved results

| Guide | Questions it answers |
| --- | --- |
| [CS1](cs1/README.md) | How do waiting power, clock, and execution policy affect prediction and selection? |
| [CS3](cs3/README.md) | How do proposal strategies, score weights, and target replay affect search outcomes? |
| [Evaluation cost appendix](appendix_evaluation_cost/README.md) | Where is time spent in candidate evaluation? |
| [Training budget appendix](appendix_training_budget/README.md) | How do training budgets affect rankings and fronts? |
| [Trial outcomes appendix](appendix_trial_outcomes/README.md) | Which attempts passed and which observations enter each analysis? |
| [Pareto comparison](compare_pareto_front_calcs/README.md) | How do two measured fronts compare under explicit matching rules? |

These analyses read saved inputs without training, contacting providers, or
accessing hardware. Each guide describes inputs, commands, and outputs. Keep
generated results outside the source checkout.

For example, summarize an existing audio study:

```sh
PYTHONPATH=src python analysis_scripts/cs3/search_history.py \
  --history-db /path/to/study/optuna.db \
  --study-name audio_search \
  --ledger-dir /path/to/study/llm_optimizer \
  --output /path/to/results/search_history
```

Omit `--ledger-dir` when there are no LLM request records. Search also refreshes
score or hypervolume plots under each study's `progress/` directory; the
[progress API](../src/crest/search/README.md) can regenerate them offline.

## 3. Generate figures

The [figure catalog](paper_plots/README.md) maps the paper's data figures to
scripts, required inputs, and runnable commands. Generators read saved results
and write figures plus numeric sidecars for checking the plotted values.

For example, plot the CS1 clock comparison from paired policy measurements:

```sh
python analysis_scripts/paper_plots/plot_case1_clock_reversal.py \
  --input-csv /path/to/results/policy_per_round.csv \
  --output-dir /path/to/figures
```
