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
The micro-workload energy probe is the exception: it stages a synthetic
workload and uses the CREST HIL harness.

## Folders

- `compare_pareto_front_calcs/`
  - **Hardware requirement: no hardware required.** Generic CSV-derived
    Pareto-front comparison helper with explicit quality/cost columns,
    feasibility filters, matching rules, and reduction denominators.

- `cs3_audio_sensitivity/`
  - **Hardware requirement: no hardware required.** Case Study 3 post-hoc
    score-sensitivity analysis over audio NAS logs.

- `micro_workload_energy_probe/`
  - **Hardware requirement: development board and HIL harness required.**
    Synthetic phase-energy probe for CREST-compatible MCU targets, reusing the
    existing HIL harness and INA228 telemetry path.

- `paper_plots/`
  - **Hardware requirement: no hardware required.** Publication figure
    generators plus sidecar plotted-point and summary outputs from explicit
    input CSV/replay paths.

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
