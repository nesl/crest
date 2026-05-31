# Analysis Scripts

This folder contains public-facing analysis utilities and paper-figure
generators that sit outside the core TinyODOM runtime.

## Folders

- `compare_pareto_front_calcs/`
  - Generic CSV-derived Pareto-front comparison helper with explicit
    quality/cost columns, feasibility filters, matching rules, and reduction
    denominators.

- `cs3_audio_sensitivity/`
  - Case Study 3 post-hoc score-sensitivity analysis over audio NAS logs.

- `micro_workload_energy_probe/`
  - Synthetic phase-energy probe for TinyODOM-compatible MCU targets, reusing
    the existing HIL harness and INA228 telemetry path.

- `paper_plots/`
  - Publication figure generators plus sidecar plotted-point and summary
    outputs from explicit input CSV/replay paths.
