# Cadenced Portenta H7 Analysis

This analysis package compares two runtime regimes on Portenta H7 using the
perturbed TinyODOM model path:

1. `back_to_back`: 10 consecutive invokes in one measurement window.
2. `cadenced`: one invoke every `latency_budget_ms`, with light idle sleep
   (`__WFI`) between release slots.

It runs both `cm7` and `cm4` and writes `JSON + CSV` summaries.

## What This Script Does

- Forces the perturbed model variant (`approx_trained` alias path).
- Forces energy-aware runtime.
- Uses shared TinyODOM sketches so the same DUT path can emit harness power
  telemetry and, where available, DUT clock telemetry for follow-on analyses.
- Runs the four combinations:
  - `cm7 + back_to_back`
  - `cm7 + cadenced`
  - `cm4 + back_to_back`
  - `cm4 + cadenced`
- Repeats each combination `--repeats` times (default: `1`).

## Notes About CM4

- CM4 runs are harness-only for runtime telemetry in this repo.
- Because CM4 runtime collection is harness-based here, this package should be
  treated as an energy/latency comparison, not a clock-cycle study.
- The script uses the existing TinyODOM runtime flow; no `src/` edits are
  required for this analysis package.

## Usage

From repo root:

```bash
python analysis_scripts/cadenced_portenta_h7/run_cadenced_portenta_h7.py
```

Common overrides:

```bash
python analysis_scripts/cadenced_portenta_h7/run_cadenced_portenta_h7.py \
  --config src/config/nas_config_stm32.yaml \
  --repeats 3 \
  --cores cm7 cm4 \
  --latency-budget-ms 200 \
  --output-json analysis_scripts/cadenced_portenta_h7/results/latest.json \
  --output-csv analysis_scripts/cadenced_portenta_h7/results/latest.csv
```

## Outputs

JSON contains:

- `metadata`
- `attempts` (one record per run)
- `aggregates` (grouped by `core x phase`, includes mean/std/min/max)

CSV contains one row per attempt and includes derived columns:

- `phase_a_energy_mj_per_inference`
- `phase_a_latency_ms_per_window`
- `phase_b_energy_mj_per_window`
- `phase_b_latency_ms_per_window`

## Sketch Templates

This folder includes phase-specific sketch templates staged by the runner into
the active TinyODOM build folder before each attempt:

- `tinyodom_inference_back_to_back.ino`
- `tinyodom_inference_cadenced.ino`
