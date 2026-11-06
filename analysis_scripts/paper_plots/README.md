<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->

# Publication Figure Scripts

These generators cover the eight data plots in the submitted CREST paper.
They read saved measurements and evaluations; they do not train models, launch
searches, contact providers, or access hardware. Diagrams and photographs are
separate artwork.

The scripts follow the original plotting tools' conventions: standalone CLIs,
explicit external paths, NumPy-style docstrings, and numeric audit sidecars.
Figures use white backgrounds, DejaVu Sans, consistent series colors, and
vector PDF exports. PNG exports default to 400 dpi. Layouts, labels, colors and export bounds follow
the submitted assets. All eight PNGs were verified pixel-for-pixel against those
assets with the rendering environments below.

## Environment and inputs

Run from the repository root with the repository Conda environment, or a
plot-only environment containing `matplotlib`, `numpy`, `pandas`, `scipy`, and
`Pillow`. No CREST installation or hardware toolchain is needed.

Research data, private input manifests, and generated images are not distributed.
Supply your own inputs using the schemas below. Keep any generated provenance
reports with your private results, outside the source checkout.
Replace the example `/path/to/...` values with your archive locations and keep
outputs outside the source checkout. All commands support `--help`.

## Submitted figure manifest

| Manuscript asset | Generator | Input |
| --- | --- | --- |
| `cs1_clock_reversal.png` | `plot_case1_clock_reversal.py` | Paired policy rounds |
| `cs1_energy_ranks.png` | `plot_case1_energy_ranks.py` | Joined candidate table |
| `cs1_model_transfer.png` | `plot_case1_model_transfer.py` | Same joined table |
| `cs2_proxy_replay_fronts.png` | `plot_case2_proxy_replay_fronts.py` | Measured NAS logs and proxy replays |
| `cs3_proposal_comparison.png` | `plot_case3_proposal_comparison.py` | Feasible observations and endpoint table |
| `cs2_training_budget_ranking.png` | `plot_training_budget.py` | Checkpoint evaluations or frozen panel CSV |
| `cs2_training_budget_fronts.png` | Same training generator | Same input |
| `appendix_timing_analysis.png` | `plot_appendix_timing_analysis.py` | Reconciled timing tables for both boards |

Outputs use these basenames directly. The CS1, CS2 and CS3 generators also
write plotted-point CSVs and numeric summaries. Training writes correlations,
front membership and input provenance; timing writes aggregated values and a
caption. `replay_front_support.py` contains the shared replay-loading, Pareto
and outcome-accounting functions imported by CS2. It has no plotting CLI or
Matplotlib dependency; obsolete plotting entry points have been removed.

## CS1: clock reversal, ranks and model transfer

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

Policy columns: `candidate`, `clock_mhz`, `period_ms`, `round`, `E_stop_mj`,
`E_inf_mj`, `C_s`. Compare `continuous_203` and `continuous_137` at 300 MHz
against trial 203 at 800 MHz. The paper has five paired rounds at 100/500 ms.
Round IDs pair observations across candidates and periods. Endpoint intervals
use paired Student-t uncertainty; lines interpolate only between measured
periods. Interpolated sidecar rows are marked; out-of-domain crossings are null.

Candidate columns: `pool`, `trial`, `study`, `period_ms`, `continuous_mj`,
`scheduled_mj`; transfer also requires `E_pred`. Study codes `continuous` and
`cadenced` display as continuous and periodic search. Ranks use all **905**
paper candidates, independently within each period/study, averaging ties.
Transfer excludes `200ms_cadenced`, leaving **756**, and plots measured minus
predicted release energy. Do not apply that exclusion to the rank figure.

## CS2: proxy candidates replayed on target hardware

The submitted comparison uses **16 unique mixed-precision candidates per
proxy**, replayed on each Arduino target. Earlier completed-row filters and
smaller memory-proxy populations are superseded. Historical archive names
still say `case1`; the current manuscript calls this CS2.

```bash
CS2_ARCHIVE=/path/to/CREST_case1
CS2_REPLAYS=/path/to/mixed_proxy_replays
FIGURE_OUTPUT=/path/to/figures
python analysis_scripts/paper_plots/plot_case2_proxy_replay_fronts.py \
  --target "BLE33=$CS2_ARCHIVE/models/OxIOD_BLE33_B2B_case1_2_t1,$CS2_REPLAYS/new_flops_ble.csv,$CS2_REPLAYS/new_memory_ble.csv" \
  --target "Portenta M4=$CS2_ARCHIVE/models/OxIOD_PORTENTA_M4_B2B_case1_4_t5,$CS2_REPLAYS/new_flops_m4.csv,$CS2_REPLAYS/new_memory_m4.csv" \
  --target "Portenta M7=$CS2_ARCHIVE/models/OxIOD_PORTENTA_M7_B2B_case1_3_t1,$CS2_REPLAYS/new_flops.csv,$CS2_REPLAYS/new_memory.csv" \
  --target "STM32 N657=$CS2_ARCHIVE/models/OxIOD_STM32_B2B_case1_5_t1,$CS2_ARCHIVE/replays/data/flops_case1_1_t3_on_stm32_cadenced,$CS2_ARCHIVE/replays/data/memory_case1_1_on_stm32_b2b_v2" \
  --no-replay-target 'STM32 N657' --layout row --dpi 400 --output-dir "$FIGURE_OUTPUT"
```

Each measured directory contains one `log_NAS_*.csv`; replay arguments are CSVs
or directories containing `replay_results.csv`. STM32 replay inputs remain
required by the common loader but are omitted from its measured-only panel.
The 200 ms threshold changes marker fill, not Pareto membership. Arduino-panel
counts refer to the FLOPs replay front. `--clock-target LABEL` is an optional
frequency-marker view, not enabled in the final-paper recipe.

## CS3: proposal strategies

```bash
CS3_DATA=/path/to/cs3_tables
python analysis_scripts/paper_plots/plot_case3_proposal_comparison.py \
  --input "$CS3_DATA/cs3_trial_score_audit_20261023.csv" \
  --endpoints "$CS3_DATA/cs3_campaign_endpoints_20261023.csv" \
  --trials-per-group 200 --output-dir /path/to/figures
```

Observation columns: `board`, `method`, `trial`, `f1`, `energy_mj`, `score`,
`latency_ms`. Boards: `Portenta`, `STM32`; methods: `Optuna`, `Memory`, `History`.
`Optuna` displays as **TPE**. Endpoint columns: `study`, `best_individual`, with
lowercase `board_method` keys. Validation checks all 1,200 observations, unique
trial IDs per group, `score = f1 - energy_mj / 4000`, and selected scores against
the endpoint table. Repeated architectures remain in the scatter. The 400 mJ
line is a scoring allocation, not a measured limit.

## Training-budget appendix

```bash
python analysis_scripts/paper_plots/plot_training_budget.py \
  --input-csv /path/to/training_tables/figure10_training_budget.csv \
  --output-dir /path/to/figures
```

CSV columns: `candidate_id`, `budget`, `rmse_total`, `energy_mj`, `source`.
The frozen panel has 50 candidates at 15/30/55/100/200/300 epochs. The renderer
checks a matched panel and constant energy, computes Spearman agreement with
300 epochs, and minimizes error/energy for each Pareto front. It exports both
panels. Seed 17 and best-validation-loss checkpoint selection are reported
archive provenance, not independently encoded or verified by the CSV.

The alternative validated JSON interface remains available:

```bash
python analysis_scripts/paper_plots/plot_training_budget.py \
  --results /path/to/panel/row_*/seed_17/results.json \
  --seed 17 --output-dir /path/to/figures
```

Each JSON contains a list with `candidate_id`, `seed`, `budget`, `policy`,
`status`, `energy_policy`, `energy_mj_per_inference`, and `metrics.rmse_total`.
The loader selects `policy="budget_only"` for the requested seed and validates a
complete matched panel and fixed historical energy. Checkpoints must already
be selected/evaluated. This JSON workflow was reconstructed; standalone styling
was recovered from the local paper renderer. The frozen CSV path reproduces the submitted panels exactly in its recorded
environment. JSON inputs reproduce them only if they encode the same panel;
neither path implies convergence at 300 epochs.

## Timing appendix

```bash
python analysis_scripts/paper_plots/plot_appendix_timing_analysis.py \
  --data-dir /path/to/reconciled_timing_tables --output-dir /path/to/figures
```

The input root holds STM32 tables and a `portenta_m7/` child with matching files:

- `breakdown.json`: `conditions` keyed by `continuous`, `cadenced`, `flops`,
  `memory`, with `trained_candidates`, `epochs`, `categories_s`,
  `client_process_s`, and HIL `hil_detail_s`.
- `proxy_times.csv`: `stage`, `inclusive_s`, with stages `proxy.flops` and
  `proxy.memory_estimate`.
- `arena_search.json`: continuous/cadenced conditions containing
  `failed_probe_measurement_s` and `candidates`.

The renderer checks ten candidates per condition, 550 training epochs, finite
proxy samples, and timing reconciliation. Input code `cadenced` displays as
**Periodic HIL**. The layout depicts cost attribution, not execution order.

## Maintenance

Superseded motivation, cross-runtime, original CS3 selection/transfer, old
rank/selection-context and v2 front generators were removed, along with the
obsolete completed-population preparation script. Historical implementations
remain in Git history. Keep draft variants out of the current-paper catalog.

```bash
python -m pytest -q test/test_paper_plot_inputs.py test/test_paper_plot_cs1.py
```

Changes to filtering, ranking, pairing or layout should also be rendered using
the frozen external inputs. Compare numeric sidecars as well as images; font
and rendering versions can change raster bytes without changing plotted values.


## Submitted-image verification

The submitted PNG metadata records two Matplotlib versions. Use the matching
rendering environment when exact pixels matter; a version change can alter text
metrics and layout even when the plotted data are identical. The following
macOS environments reproduced all eight assets with zero differing RGBA pixels:

| Figures | Python | Matplotlib | NumPy | pandas | SciPy | Pillow |
| --- | --- | --- | --- | --- | --- | --- |
| CS1, CS2 proxy, CS3, timing | 3.13.5 | 3.10.0 | 2.1.3 | 2.2.3 | 1.15.3 | 11.1.0 |
| Two training-budget panels | 3.12.14 | 3.11.2 | 2.5.3 | Not required | 1.16.2 | 12.3.0 |

Keep the submitted images external and immutable. Collect the eight regenerated
PNGs in one output directory, then run:

```bash
python analysis_scripts/paper_plots/verify_paper_figures.py \
  --reference-dir /path/to/submitted/assets \
  --generated-dir /path/to/regenerated/figures \
  --report /path/to/verification.json
```

The checker compares dimensions and every decoded RGBA pixel, reports file
hashes and difference metrics, and exits nonzero for missing or changed images.
It ignores PNG metadata/compression differences. It does not establish data
provenance; the input manifest and numeric sidecars serve that separate purpose.
