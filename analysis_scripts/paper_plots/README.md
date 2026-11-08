<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->

# Publication Figure Scripts

Seven generators cover the seven data figures (Figures 5–11) in the CREST paper.
Figure 10 has two image panels, giving eight plot assets in total.
They read saved measurements and evaluations.
They do not train models, launch searches, contact providers, or access hardware.
Diagrams and photographs are separate artwork.

The scripts provide standalone CLIs, explicit external paths, and numeric audit sidecars.
Figures use white backgrounds, DejaVu Sans, consistent series colors, and vector PDF exports.
PNG exports default to 400 dpi.
Layouts, labels, colors and export bounds follow the submitted assets.

Start with [environment and inputs](#environment-and-inputs), then find the generator in the [figure manifest](#submitted-figure-manifest).
Use the task sections for [CS1](#cs1-clock-reversal-ranks-and-model-transfer), [CS2 replay](#cs2-proxy-candidates-replayed-on-target-hardware), [CS3](#cs3-proposal-strategies), [training budgets](#training-budget-appendix), or [timing](#timing-appendix).

## Environment and inputs

Run from the repository root with the repository Conda environment, or a plot-only environment containing `matplotlib`, `numpy`, `pandas`, `scipy`, and `Pillow`.
No CREST installation or hardware toolchain is needed.

Study data and input manifests must be supplied separately.
Use the schemas below and replace example `/path/to/...` values with your archive locations.
Keep outputs and provenance reports outside the source checkout.
All commands support `--help`.

## Submitted figure manifest

| Figure | Manuscript asset | Generator | Input |
| --- | --- | --- | --- |
| 5 | `cs1_clock_reversal.png` | `plot_case1_clock_reversal.py` | Paired policy rounds |
| 6 | `cs1_energy_ranks.png` | `plot_case1_energy_ranks.py` | Joined candidate table |
| 7 | `cs2_proxy_replay_fronts.png` | `plot_case2_proxy_replay_fronts.py` | Measured NAS logs and proxy replays |
| 8 | `cs3_proposal_comparison.png` | `plot_case3_proposal_comparison.py` | Feasible observations and endpoint table |
| 9 | `cs1_model_transfer.png` | `plot_case1_model_transfer.py` | Joined candidate table |
| 10, top | `cs2_training_budget_ranking.png` | `plot_training_budget.py` | Checkpoint evaluations or panel CSV |
| 10, bottom | `cs2_training_budget_fronts.png` | `plot_training_budget.py` | Same input |
| 11 | `appendix_timing_analysis.png` | `plot_appendix_timing_analysis.py` | Reconciled timing tables for both boards |

Figures 1–4 are the runtime-schedule diagram, framework diagram, NAS-flow illustration, and hardware-harness artwork.
They are not data plots and have no Python generators here.

PNG, PDF, and SVG exports are formats of the same figures, not additional plots.
Numeric sidecars record the plotted data and summaries.

Outputs use these basenames directly.
The CS1, CS2 and CS3 generators also write plotted-point CSVs and numeric summaries.
Training writes correlations, front membership and input provenance.
Timing writes aggregated values and a caption.

## CS1 clock reversal, ranks and model transfer

```bash
CS1_DATA=/path/to/cs1_tables
FIGURE_OUTPUT=/path/to/figures
python analysis_scripts/paper_plots/plot_case1_clock_reversal.py \
  --input-csv "$CS1_DATA/final_slot_v1/policy_per_round.csv" --output-dir "$FIGURE_OUTPUT"
python analysis_scripts/paper_plots/plot_case1_energy_ranks.py \
  --input-csv "$CS1_DATA/selection_regret/joined_candidates.csv" --output-dir "$FIGURE_OUTPUT"
python analysis_scripts/paper_plots/plot_case1_model_transfer.py \
  --input-csv "$CS1_DATA/selection_regret/joined_candidates.csv" --output-dir "$FIGURE_OUTPUT"
```

Policy columns include `candidate`, `clock_mhz`, `period_ms`, `round`, `E_stop_mj`, `E_inf_mj`, `C_s`.
Compare `continuous_203` and `continuous_137` at 300 MHz against trial 203 at 800 MHz.
The paper has five paired rounds at 100/500 ms.
Round IDs pair observations across candidates and periods.
Endpoint intervals use paired Student-t uncertainty.
Lines interpolate only between measured periods.
Interpolated sidecar rows are marked.
Out-of-domain crossings are null.

Candidate columns include `pool`, `trial`, `study`, `period_ms`, `continuous_mj`, `scheduled_mj`.
Transfer also requires `E_pred`.
Study codes `continuous` and `cadenced` display as continuous and periodic search.
Ranks use all **905** paper candidates, independently within each period/study, averaging ties.
Transfer excludes `200ms_cadenced`, leaving **756**, and plots measured minus predicted release energy.
Do not apply that exclusion to the rank figure.

## CS2 proxy candidates replayed on target hardware

The submitted comparison uses **16 unique mixed-precision candidates per proxy**, replayed on each Arduino target.

```bash
CS2_MEASUREMENTS=/path/to/measured_nas_runs
CS2_REPLAYS=/path/to/mixed_proxy_replays
FIGURE_OUTPUT=/path/to/figures
python analysis_scripts/paper_plots/plot_case2_proxy_replay_fronts.py \
  --target "BLE33=$CS2_MEASUREMENTS/ble33,$CS2_REPLAYS/new_flops_ble.csv,$CS2_REPLAYS/new_memory_ble.csv" \
  --target "Portenta M4=$CS2_MEASUREMENTS/portenta_m4,$CS2_REPLAYS/new_flops_m4.csv,$CS2_REPLAYS/new_memory_m4.csv" \
  --target "Portenta M7=$CS2_MEASUREMENTS/portenta_m7,$CS2_REPLAYS/new_flops.csv,$CS2_REPLAYS/new_memory.csv" \
  --target "STM32 N657=$CS2_MEASUREMENTS/stm32_n657,$CS2_REPLAYS/stm32_flops,$CS2_REPLAYS/stm32_memory" \
  --no-replay-target 'STM32 N657' --layout row --dpi 400 --output-dir "$FIGURE_OUTPUT"
```

Each measured directory uses the first sorted `log_NAS_*.csv`, falling back to `trials.csv`.
Replay arguments are CSVs or directories containing `replay_results.csv`.
Measured points need finite positive inference energy and finite RMSE.
Replay points also need `replay_status=completed` and a successful `target__error_code`.
See `load_crest_points` and `load_replay_points` in [`replay_front_support.py`](replay_front_support.py) for column schemas, status filters, and latency handling.
STM32 replay inputs remain required by the common loader but are omitted from its measured-only panel.
The 200 ms threshold changes marker fill, not Pareto membership.
Arduino-panel counts refer to the FLOPs replay front.
`--clock-target LABEL` is an optional frequency-marker view, not enabled in the final-paper recipe.

## CS3 proposal strategies

```bash
CS3_DATA=/path/to/cs3_tables
python analysis_scripts/paper_plots/plot_case3_proposal_comparison.py \
  --input "$CS3_DATA/cs3_trial_score_audit_20261023.csv" \
  --endpoints "$CS3_DATA/cs3_campaign_endpoints_20261023.csv" \
  --trials-per-group 200 --output-dir /path/to/figures
```

Observation columns include `board`, `method`, `trial`, `f1`, `energy_mj`, `score`, `latency_ms`.
Boards.
`Portenta`, `STM32`.
Methods include `Optuna`, `Memory`, `History`.
`Optuna` displays as **TPE**.
Endpoint columns include `study`, `best_individual`, with lowercase `board_method` keys.
Validation checks all 1,200 observations, unique trial IDs per group, `score = f1 - energy_mj / 4000`, and selected scores against the endpoint table.
Tied maximum scores select the first input row.
Repeated architectures remain in the scatter.
The 400 mJ line is a scoring allocation, not a measured limit.

## Training-budget appendix

```bash
python analysis_scripts/paper_plots/plot_training_budget.py \
  --input-csv /path/to/training_tables/figure10_training_budget.csv \
  --output-dir /path/to/figures
```

CSV columns include `candidate_id`, `budget`, `rmse_total`, `energy_mj`, `source`.
The frozen panel has 50 candidates at 15/30/55/100/200/300 epochs.
The renderer checks a matched panel and constant energy, computes Spearman agreement with 300 epochs, and minimizes error/energy for each Pareto front.
It exports both panels.
Seed 17 and best-validation-loss checkpoint selection are reported archive provenance, not independently encoded or verified by the CSV.

The validated JSON interface also accepts saved checkpoint evaluations.

```bash
python analysis_scripts/paper_plots/plot_training_budget.py \
  --results /path/to/panel/row_*/seed_17/results.json \
  --seed 17 --output-dir /path/to/figures
```

Each JSON contains a list with `candidate_id`, `seed`, `budget`, `policy`, `status`, `energy_policy`, `energy_mj_per_inference`, and `metrics.rmse_total`.
The loader selects `policy="budget_only"` for the requested seed and validates a complete matched panel and fixed historical energy.
Checkpoints must already be selected/evaluated.
The frozen CSV path reproduces the submitted panels exactly in its recorded environment.
JSON inputs reproduce them only if they encode the same panel.
Neither path implies convergence at 300 epochs.

## Timing appendix

```bash
python analysis_scripts/paper_plots/plot_appendix_timing_analysis.py \
  --data-dir /path/to/reconciled_timing_tables --output-dir /path/to/figures
```

The input root holds STM32 tables and a `portenta_m7/` child with matching files.

- `breakdown.json`.
  `conditions` keyed by `continuous`, `cadenced`, `flops`, `memory`, with `trained_candidates`, `epochs`, `categories_s`, `client_process_s`, and HIL `hil_detail_s`.
- `proxy_times.csv`.
  `stage`, `inclusive_s`, with stages `proxy.flops` and `proxy.memory_estimate`.
- `arena_search.json`.
  `conditions` keyed by continuous/cadenced, containing `failed_probe_measurement_s` and `candidates`.

Duration fields are in seconds.
For the category and HIL-detail keys consumed by the renderer, see `load_panel` in [`plot_appendix_timing_analysis.py`](plot_appendix_timing_analysis.py).
The renderer checks ten candidates per condition, 550 training epochs, finite proxy samples, and timing reconciliation.
Input code `cadenced` displays as **Periodic HIL**.
The layout depicts cost attribution, not execution order.

## Maintenance

```bash
python -m pytest -q test/test_paper_plot_inputs.py test/test_paper_plot_cs1.py
```

Changes to filtering, ranking, pairing or layout should also be rendered using the frozen external inputs.
Compare numeric sidecars as well as images.
Font and rendering versions can change raster bytes without changing plotted values.
