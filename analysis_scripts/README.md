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
workload and uses the CREST HIL harness. The CS1 epoch pilot can train models
when explicitly invoked with `run --execute`, but never accesses hardware.

## Folders

- `cs1_epoch_sensitivity/`
  - **Hardware requirement: no hardware required. Training is opt-in.**
    Four-point original-Pareto-front OxIOD training-budget pilot, reusing
    CREST's model/task/evaluation code and holding historical energy fixed.
    See its README for the limited within-front interpretation.

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
