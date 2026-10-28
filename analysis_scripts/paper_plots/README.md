<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->

# Publication Figure Scripts

This directory contains the Python scripts used to generate CREST publication
figures. The plotting entry points are intentionally command-line driven:
input CSV paths, replay directories, output directories, and output filename
stems are provided explicitly instead of being inferred from local workstation
paths.

Hardware requirement: no hardware required. These scripts consume existing NAS
and replay artifacts; they do not run NAS, retrain models, replay hardware, or
start a HIL server.

The scripts write publication figures plus small sidecar files, such as plotted
points and summaries, that make it easier to audit what was drawn. Rendering
constants live in the scripts so the published figures can be regenerated
without changing font sizes, marker sizes, spacing, axis limits, or DPI.

## Requirements

Use the repository environment:

```bash
conda env create -f environment.yml -n crest
conda activate crest
```

For plot-only reproduction, the required Python packages are:

- `matplotlib`
- `numpy`
- `pandas`
- `Pillow` (CS3 image validation)
- `scipy` (training-budget rank correlation)

Run commands from the repository root unless noted otherwise.
Plot-only use does not require `make install` because the scripts use
repository-relative imports and explicit input paths. Run the full root setup
when you also need NAS, HIL, or package import behavior outside these scripts.

## Current paper figures

The current manuscript uses the following data-plot generators. Inputs are
external artifacts; no observations, study databases, captured prompts, or
generated plots are bundled here. Keep output directories outside the checkout.
The architecture diagrams and hardware photographs are authored artwork rather
than data plots.

| Paper figure | Script | External input |
| --- | --- | --- |
| CS1 measured-energy and proxy fronts | `plot_case1_combined_fronts_v3.py` | Measured NAS logs and FLOPs/memory replay directories. |
| CS2 continuous/cadenced energy ranks | `plot_case2_energy_ranks_corrected.py` | Paired-energy TSV. |
| CS2 selection fronts appendix | `plot_case2_selection_context.py` | Paired-selection JSON. |
| CS3 proposal comparison | `plot_case3_proposal_log.py` | Feasible-observation and endpoint CSVs. |
| Training-budget appendix | `plot_training_budget.py` (reconstructed renderer) | Matched checkpoint-evaluation JSON files. |
| Timing appendix | `plot_appendix_timing_analysis.py` | Reconciled timing tables for both boards. |

`prepare_case1_memory_completed.py` supplies the CS1 population-filtering step:
it selects the first requested completed source rows **before** computing the
proxy Pareto front, then filters already-recorded hardware replays. It performs
no new hardware measurements.

## Other figure generators

| Script | Purpose |
| --- | --- |
| `plot_stm32_oxiod_cadenced_motivation.py` | Continuous vs cadenced STM32 motivation plot. |
| `plot_case1_combined_fronts_v2.py` | Case Study 1 measured-energy, FLOPs replay, and memory-traffic replay fronts. |
| `plot_case2_b2b_cadenced_fronts.py` | Case Study 2 continuous/cadenced cross-runtime comparison. |
| `plot_case3_audio_selection_tradeoff.py` | Case Study 3 audio score-selection tradeoff. |
| `plot_case3_audio_transfer.py` | Case Study 3 cross-board transfer figures. |
| `plot_combined_replay_fronts.py` | Generic replay-vs-CREST front comparison helper used by Case Study 1. |

## Output Convention

Every script takes an explicit output directory and filename stem. A typical
run writes one or more image files plus CSV or text sidecars:

```bash
mkdir -p /path/to/figures
```

Use `--help` to inspect the required inputs for each script:

```bash
python -B analysis_scripts/paper_plots/plot_case3_audio_selection_tradeoff.py --help
```

For non-plot Pareto-front comparison calculations, see
`analysis_scripts/compare_pareto_front_calcs/`.

## Examples

The examples below use placeholder paths so they can be adapted to user artifacts.

### Case Study 1

`--target` controls both the input paths and subplot order. Repeat it once per
target:

```bash
python -B analysis_scripts/paper_plots/plot_case1_combined_fronts_v2.py \
  --output-dir /path/to/figures \
  --basename case1_combined_fronts \
  --layout row \
  --target "Target A=path/to/measured_run_a,path/to/flops_replay_a,path/to/memory_replay_a" \
  --target "Target B=path/to/measured_run_b,path/to/flops_replay_b,path/to/memory_replay_b"
```

Each `MEASURED_RUN_DIR` must contain a `log_NAS_*.csv` file. Each replay path
may be either a replay output directory or an explicit `replay_results.csv`
file.

### Case Study 2

```bash
python -B analysis_scripts/paper_plots/plot_case2_b2b_cadenced_fronts.py \
  --study-points-csv path/to/case2_study_points.csv \
  --overlay-points-csv path/to/cadenced_front_replayed_b2b_points.csv \
  --b2b-on-cadenced-replay-csv path/to/b2b_front_replayed_cadenced/replay_results.csv \
  --cadenced-log-csv path/to/cadenced_nas_log.csv \
  --output-dir /path/to/figures \
  --plot-name case2_cross_runtime
```

### Case Study 3 Selection

```bash
python -B analysis_scripts/paper_plots/plot_case3_audio_selection_tradeoff.py \
  --portenta-log path/to/portenta_audio_nas_log.csv \
  --stm-log path/to/stm32_audio_nas_log.csv \
  --output-dir /path/to/figures \
  --output-stem case3_audio_selection_tradeoff \
  --v2
```

### Case Study 3 Transfer

```bash
python -B analysis_scripts/paper_plots/plot_case3_audio_transfer.py \
  --stm-log path/to/stm32_audio_nas_log.csv \
  --portenta-log path/to/portenta_audio_nas_log.csv \
  --stm-on-portenta-replay path/to/stm_selected_on_portenta/replay_results.csv \
  --portenta-on-stm-replay path/to/portenta_selected_on_stm32/replay_results.csv \
  --output-dir /path/to/figures \
  --score-progress-stem case3_score_progress \
  --transfer-stem case3_cross_board_transfer \
  --transfer-v2-stem case3_cross_board_transfer_compact \
  --transfer-points-stem case3_cross_board_transfer_points
```

## Current figure inputs and commands

### CS1 completed-population fronts

The source CSV retains completed-trial order and has an
`objective_values_json` column containing `[RMSE, proxy]`. Each replay directory
contains `replay_results.csv` with `source_row_index` and a `manifest.json`.

```bash
python analysis_scripts/paper_plots/prepare_case1_memory_completed.py \
  --source-csv /path/to/memory_source.csv \
  --replay 'board_a=/path/to/memory_replay' \
  --completed-count 150 --output-dir /path/to/filtered_replays

python analysis_scripts/paper_plots/plot_case1_combined_fronts_v3.py \
  --target 'Target A=/path/to/measured_run,/path/to/flops_replay,/path/to/filtered_replays/board_a' \
  --layout row --output-dir /path/to/figures --basename case1_fronts
```

Repeat `--target` in the desired panel order. `--no-replay-target LABEL` hides
proxy overlays for a measured-only panel; `--clock-target LABEL` displays its
measured configurations by clock frequency. The same source population must be
used for comparisons across replay targets.

### CS2 ranks and selection

Rank TSV columns: `period_ms`, `study`, `trial`,
`continuous_mj_per_inference`, `cadenced_mj_per_release`. Study values are
`continuous` and `cadenced`; the paper compares 100, 200, and 500 ms periods.
Ranks are computed within each period and study, averaging exact ties; pairwise
reversal statistics exclude pairs tied in either measure.

Selection input is a JSON list with `study`, `period_ms`, `trial`, `rmse`, `continuous`,
and `cadenced` fields. Both fronts are drawn against cadenced energy so their
selection consequences can be compared within each pool and period.

```bash
python analysis_scripts/paper_plots/plot_case2_energy_ranks_corrected.py \
  --input /path/to/paired_energy.tsv --output-dir /path/to/figures
python analysis_scripts/paper_plots/plot_case2_selection_context.py \
  --input /path/to/paired_selection.json --output-dir /path/to/figures
```

### CS3 proposal strategies

Observation CSV columns: `board`, `method`, `trial`, `f1`, `energy_mj`, `score`,
`latency_ms`. Boards are `Portenta` and `STM32`; methods are `Optuna`, `Memory`,
and `History`. Scores follow the paper's `f1 - energy_mj / 4000` rule. The
endpoint CSV has `study` and `best_individual` columns, with lowercase
`board_method` keys. Endpoint values verify the score-selected stars.

```bash
python analysis_scripts/paper_plots/plot_case3_proposal_log.py \
  --input /path/to/observations.csv --endpoints /path/to/endpoints.csv \
  --trials-per-group 200 --output-dir /path/to/figures
```

All observations, including repeated architectures, remain in the scatter plot.
The 400 mJ line represents a scoring allocation.

### Timing appendix

The input directory holds STM32 timing tables, with matching Portenta tables in
`portenta_m7/`. Each contains:

- `breakdown.json`: `conditions` keyed by `continuous`, `cadenced`, `flops`,
  and `memory`; records include `trained_candidates`, `epochs`, `categories_s`,
  `client_process_s`, and HIL `hil_detail_s`.
- `proxy_times.csv`: `stage` and `inclusive_s`, with stages `proxy.flops` and
  `proxy.memory_estimate`.
- `arena_search.json`: `conditions` for continuous/cadenced HIL, including
  `failed_probe_measurement_s` and `candidates`.

The renderer checks the paper protocol (ten candidates per condition, 550
epochs), finite proxy samples, and timing-category reconciliation. These are
protocol constraints, not bundled measurements.

```bash
python analysis_scripts/paper_plots/plot_appendix_timing_analysis.py \
  --data-dir /path/to/timing_tables --output-dir /path/to/figures
```

### Training-budget appendix

`plot_training_budget.py` is a **reconstructed renderer**, not the recovered
original final plotting script and not a pixel-exact reproduction. It computes
the appendix's two analyses directly from checkpoint evaluations: Spearman
rank correlation against the largest budget and Pareto fronts minimizing
validation error and fixed historical energy.

Each supplied JSON file is a list of records with `candidate_id`, `seed`,
`budget`, `policy`, `status`, `energy_policy`, `energy_mj_per_inference`, and
`metrics.rmse_total`. The renderer selects `policy="budget_only"` and the
requested seed, requires every candidate at every requested budget, and checks
that historical energy stays fixed. Checkpoints must already have been selected
by lowest validation loss available within each budget; the renderer neither
selects nor evaluates checkpoints.

```bash
python analysis_scripts/paper_plots/plot_training_budget.py \
  --results /path/to/panel/row_*/seed_17/results.json \
  --seed 17 --output-dir /path/to/figures
```

The default budgets are 15, 30, 55, 100, 200, and 300 epochs, with the
55-epoch search budget highlighted. Styling matches the other paper generators:
DejaVu Sans typography, blue/orange highlights, light grids, and distinct front
markers. Exports include a 7.05 × 3.5 inch PNG at 400 dpi and vector PDF/SVG;
text bounds are checked before saving. Repeated configurations must not be
duplicated across input files. The script writes figures and a computed JSON
summary outside the checkout.
