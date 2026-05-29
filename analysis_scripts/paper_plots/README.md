# Paper Plot Scripts

This directory contains the Python scripts used to generate the figures for the
paper. The plotting entry points are intentionally command-line driven: input
CSV paths, replay directories, output directories, and output filename stems are
provided explicitly instead of being inferred from local workstation paths.

The scripts write publication figures plus small sidecar files, such as plotted
points and summaries, that make it easier to audit what was drawn. Rendering
constants live in the scripts so the published figures can be regenerated
without changing font sizes, marker sizes, spacing, axis limits, or DPI.

## Requirements

Use the repository environment:

```bash
conda env create -f environment.yml
conda activate tinyodomex
```

For plot-only reproduction, the required Python packages are:

- `matplotlib`
- `numpy`
- `pandas`

Some scripts import helper modules from `analysis_scripts/hil_replay`, so run
commands from the repository root unless noted otherwise.

## Scripts

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
run writes a PNG, a PDF, and one or more CSV or text sidecars:

```bash
mkdir -p outputs/paper_plots
```

Use `--help` to inspect the required inputs for each script:

```bash
python -B analysis_scripts/paper_plots/plot_case3_audio_selection_tradeoff.py --help
```

## Examples

The examples below use placeholder paths so they can be adapted to released
artifacts, local experiment outputs, or regenerated replay data.

### Case Study 1

`--target` controls both the input paths and subplot order. Repeat it once per
target:

```bash
python -B analysis_scripts/paper_plots/plot_case1_combined_fronts_v2.py \
  --output-dir outputs/paper_plots \
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
  --output-dir outputs/paper_plots \
  --plot-name case2_cross_runtime
```

### Case Study 3 Selection

```bash
python -B analysis_scripts/paper_plots/plot_case3_audio_selection_tradeoff.py \
  --portenta-log path/to/portenta_audio_nas_log.csv \
  --stm-log path/to/stm32_audio_nas_log.csv \
  --output-dir outputs/paper_plots \
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
  --output-dir outputs/paper_plots \
  --score-progress-stem case3_score_progress \
  --transfer-stem case3_cross_board_transfer \
  --transfer-v2-stem case3_cross_board_transfer_compact \
  --transfer-points-stem case3_cross_board_transfer_points
```

## Local Reproduction Notes

The file `README_local_only.md` contains local paper-13 commands that reference
the repository's current internal directory layout. It is useful while preparing
the paper branch, but the commands there are not intended to be stable public
API for released artifacts.

## Development Checks

Before changing the plotting scripts, capture the current figure outputs. After
the change, regenerate the same figures into a scratch directory and compare the
image dimensions and any expected checksums for the artifacts you are updating.

Basic checks:

```bash
python -B -m py_compile analysis_scripts/paper_plots/*.py
python -B analysis_scripts/paper_plots/plot_stm32_oxiod_cadenced_motivation.py --help
python -B analysis_scripts/paper_plots/plot_case1_combined_fronts_v2.py --help
python -B analysis_scripts/paper_plots/plot_case2_b2b_cadenced_fronts.py --help
python -B analysis_scripts/paper_plots/plot_case3_audio_selection_tradeoff.py --help
python -B analysis_scripts/paper_plots/plot_case3_audio_transfer.py --help
python -B analysis_scripts/paper_plots/plot_combined_replay_fronts.py --help
git diff --check -- analysis_scripts/paper_plots
```
