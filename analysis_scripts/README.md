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

- `paper_plots/`
  - **Hardware requirement: no hardware required.** Publication figure
    generators for all eight submitted-paper data plots, with explicit external
    inputs and plotted-point/summary sidecars. Private data manifests are not
    distributed. The
    [figure catalog](paper_plots/README.md) gives the canonical scripts and
    commands; superseded draft generators remain in Git history.

## Case Study 3: proposal strategies and search effort

These tools analyze externally supplied study artifacts for the paper's audio
proposal-strategy comparison.

- `analyze_cs3_search.py` summarizes recorded trial outcomes, architecture
  coverage, proposal provenance, score progress and recorded API wall time.
  These support the CS3 comparison of search behavior and effort; they do not
  establish optimizer superiority across unmatched experiments.

Run from the repository root, using your own study artifacts:

```bash
PYTHONPATH=src python analysis_scripts/analyze_cs3_search.py \
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

For the current paper figure-to-script mapping, required input schemas, and
reproduction commands, see [Publication Figure Scripts](paper_plots/README.md).
