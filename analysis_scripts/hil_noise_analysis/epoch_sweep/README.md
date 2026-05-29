# Epoch Sweep Experiment

This folder contains a two-step experiment flow:

1. Train a fixed hyperparameter model in staged epoch increments and export
   per-stage artifacts.
2. Evaluate those stage checkpoints on HIL and collect per-run metrics.

## Files

- `train_epoch_sweep.py`
  - GPU-side staged training script.
  - Saves checkpoints at stage boundaries (`50`, `100`, ... by default).
  - Applies global early stopping across the full run.
  - Exports quantized TFLite per saved checkpoint and writes graph stats to CSV.

- `hil_epoch_sweep_scan.py`
  - HIL-side checkpoint sweep script.
  - Reads training CSV, runs HIL metrics per checkpoint, and writes per-run CSV.
  - Optionally writes checkpoint-level summary CSV.

- `audit_fresh_untrained_tflite.py`
  - Builds a fresh untrained fixed-architecture model and exports quantized TFLite.
  - Extracts and prints op histogram stats (`op_count`, `ADD` count, per-op histogram).
  - Appends a `fresh_untrained_audit` row to the epoch-sweep training stats CSV (unless disabled).

## Default Output Location

- `analysis_scripts/hil_noise_analysis/epoch_sweep/artifacts/`

## GPU (Overnight) Training Run

```bash
python analysis_scripts/hil_noise_analysis/epoch_sweep/train_epoch_sweep.py \
  --config src/config/nas_config_stm32.yaml \
  --out-dir analysis_scripts/hil_noise_analysis/epoch_sweep/artifacts \
  --artifact-prefix noise_scan_epoch_sweep \
  --max-epochs 500 \
  --stage-size 50 \
  --patience 40 \
  --min-delta 0.0 \
  --verbose-fit
```

### GPU Training Run (Explicit Output Paths)

```bash
python analysis_scripts/hil_noise_analysis/epoch_sweep/train_epoch_sweep.py \
  --config src/config/nas_config_stm32.yaml \
  --out-dir analysis_scripts/hil_noise_analysis/epoch_sweep/artifacts \
  --plots-dir analysis_scripts/hil_noise_analysis/epoch_sweep/artifacts/plots \
  --csv-path analysis_scripts/hil_noise_analysis/epoch_sweep/artifacts/epoch_sweep_training_stats.csv \
  --artifact-prefix noise_scan_epoch_sweep \
  --max-epochs 500 \
  --stage-size 50 \
  --patience 40 \
  --min-delta 0.0 \
  --verbose-fit
```

Expected primary output:

- `analysis_scripts/hil_noise_analysis/epoch_sweep/artifacts/epoch_sweep_training_stats.csv`
- `analysis_scripts/hil_noise_analysis/epoch_sweep/artifacts/noise_scan_epoch_sweep_training_manifest.json`
- Per-checkpoint metadata JSON files (for example `noise_scan_epoch_sweep_epoch_50.json`)

## Copy to HIL Host

Copy the artifacts directory (or at minimum the training CSV plus checkpoint files):

```bash
scp -r analysis_scripts/hil_noise_analysis/epoch_sweep/artifacts <hil_host>:<repo>/analysis_scripts/hil_noise_analysis/epoch_sweep/
```

## HIL Sweep Run (Morning)

```bash
python analysis_scripts/hil_noise_analysis/epoch_sweep/hil_epoch_sweep_scan.py \
  --config src/config/nas_config_stm32.yaml \
  --training-csv analysis_scripts/hil_noise_analysis/epoch_sweep/artifacts/epoch_sweep_training_stats.csv \
  --runs 1 \
  --input-modes uniform \
  --cooldown 0
```

Verbose logging:

```bash
python analysis_scripts/hil_noise_analysis/epoch_sweep/hil_epoch_sweep_scan.py \
  --config src/config/nas_config_stm32.yaml \
  --training-csv analysis_scripts/hil_noise_analysis/epoch_sweep/artifacts/epoch_sweep_training_stats.csv \
  --runs 1 \
  --input-modes uniform \
  --verbose
```

Explicit output paths + checkpoint remap (useful when CSV paths came from another machine):

```bash
python analysis_scripts/hil_noise_analysis/epoch_sweep/hil_epoch_sweep_scan.py \
  --config src/config/nas_config_stm32.yaml \
  --training-csv analysis_scripts/hil_noise_analysis/epoch_sweep/artifacts/epoch_sweep_training_stats.csv \
  --checkpoint-root analysis_scripts/hil_noise_analysis/epoch_sweep/artifacts \
  --csv-path analysis_scripts/hil_noise_analysis/epoch_sweep/artifacts/epoch_sweep_hil_metrics.csv \
  --summary-csv-path analysis_scripts/hil_noise_analysis/epoch_sweep/artifacts/epoch_sweep_hil_summary.csv \
  --runs 1 \
  --input-modes uniform
```

Expected primary outputs:

- `analysis_scripts/hil_noise_analysis/epoch_sweep/artifacts/epoch_sweep_hil_metrics.csv`
- `analysis_scripts/hil_noise_analysis/epoch_sweep/artifacts/epoch_sweep_hil_summary.csv`

## Useful Optional Flags

- `train_epoch_sweep.py`
  - `--verbose-fit` to print per-epoch logs.
  - `--calibration-windows-override N` for quick smoke tests with smaller splits.

- `hil_epoch_sweep_scan.py`
  - `--csv-path` to override per-run output CSV path.
  - `--summary-csv-path` to override grouped summary output CSV path.
  - `--checkpoint-root DIR` to remap checkpoint paths when training CSV paths are from a different host.
  - `--epoch-filter "50,100-200"` to evaluate only selected checkpoints.
  - `--energy-aware` to force energy-aware sketch selection.
  - `--verbose` to print INFO logs during checkpoint evaluation and HIL protocol steps.
