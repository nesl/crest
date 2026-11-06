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
## Offline analysis and publication tools

- `trial_outcomes/`: read-only final-trial outcome classification, separate
  population flags, and aggregate checks; see its [README](trial_outcomes/README.md).

- `cs3_audit/`: reusable exact scoring, campaign effort, saved evaluation and
  replay checks; see its [README](cs3_audit/README.md).

- `timing_attribution/`: saved client/server timing reconciliation and appendix
  checks; see its [README](timing_attribution/README.md).

- `training_budget/`: reusable checkpoint-policy and training-time analysis;
  see its [README](training_budget/README.md) for inputs and paper checks.

- `cs1_offline/`: offline calibration, prediction, selection and paired replay
  analysis, with a separate [submitted-paper recipe](cs1_offline/README.md).

- `compare_pareto_front_calcs/`
  - **Hardware requirement: no hardware required.** Generic CSV-derived
    Pareto-front comparison helper with explicit quality/cost columns,
    feasibility filters, matching rules, and reduction denominators.

- `cs3_audio_sensitivity/`
  - **Hardware requirement: no hardware required.** Case Study 3 post-hoc
    score-sensitivity analysis over audio NAS logs.

- `paper_plots/`: existing publication figure tools; see its README.

## Case Study 3: proposal strategies and search effort

These tools analyze externally supplied study artifacts for the paper's audio
proposal-strategy comparison. This directory contains code, not study databases,
prompt captures, measurements, cost reports or generated figures.

- `analyze_cs3_search.py` summarizes recorded trial outcomes, architecture
  coverage, proposal provenance, score progress and recorded API wall time.
  These support the CS3 comparison of search behavior and effort; they do not
  establish optimizer superiority across unmatched experiments.
- `plot_cs3_progress.py` plots feasible scores and the incumbent against trial
  number and elapsed time, supporting inspection of CS3 search progress. The
  study name, input directory and output directory must be supplied explicitly.
  It reads `optuna.db` within the study directory and requires a scalar study
  with one maximized objective.

Run from the repository root, using your own study artifacts:

```bash
PYTHONPATH=src python analysis_scripts/analyze_cs3_search.py \
  --history-db /path/to/study/optuna.db \
  --study-name audio_search \
  --ledger-dir /path/to/study/llm_optimizer \
  --output /path/to/analysis-output

python analysis_scripts/plot_cs3_progress.py \
  --study-dir /path/to/study \
  --study-name audio_search \
  --warmup-count 15 \
  --output-dir /path/to/figures
```

`--ledger-dir` is optional for analysis without LLM request records. Set
`--warmup-count` to the actual fixed warm-up length (default zero). The plotting
tool writes into `progress/` beneath the specified output directory. Both tools
read saved artifacts without launching search, contacting a provider or accessing
hardware. Keep their generated outputs outside the source checkout.

For the current paper figure-to-script mapping, required input schemas, and
reproduction commands, see [Publication Figure Scripts](paper_plots/README.md).

For a study configured without feasibility rules, add `--feasibility-disabled`
to `plot_cs3_progress.py`. Its default requires affirmative feasibility evidence;
explicitly rejected trials are excluded in either mode.
