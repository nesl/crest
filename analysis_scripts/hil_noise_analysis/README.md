# HIL Noise Analysis

This folder contains scripts and docs for running HIL noise scans and analyzing
how input distributions affect latency and energy measurements.

## Scripts

- `hil_energy_noise_scan.py`
  - Runs a multi-mode (`uniform` / `representative` / `real`) HIL noise scan.
  - Runs one or more model variants (`trained_50ep`, `untrained`) against each input mode.
  - Re-syncs the Arduino sketch variant for each input mode and records per-run metrics.
  - Writes a CSV with `model_variant`, `input_mode`, and per-run metrics.

- `train_noise_scan_model.py`
  - GPU-side utility to train the fixed noise-scan architecture for up to 50 epochs
    with NAS-style early stopping (`patience=40`).
  - Exports a `.keras` checkpoint plus metadata JSON for transfer to the HIL host.
  - Can optionally export a `.tflite` copy for audit/debug.
  - Automatically falls back to a numpy-only data loader if `tinyodom.data` (pandas-based)
    is unavailable in the environment.

- `noise_scan_model_spec.py`
  - Shared fixed hyperparameter spec for the noise scan architecture
    (`nb_filters=10`, `kernel_size=12`, `dilations=[1,4,8,64]`, etc.).
  - Used by both scan and training scripts to keep architecture definitions aligned.

- `oxiod_input_profile.py`
  - Loads OxIOD windows using the same data loader and config parameters used by
    `hil_server.py` / `nas_model_client.py`.
  - Prints per-channel statistics to compare the real dataset to uniform `[0, 5]` inputs.
  - Can export `sketches/analysis_sketches/tinyodom_tcn_input_data.h`, which embeds:
    - Per-channel mean/std/min/max values.
    - A fixed set of real windows (default 10) for the real-data sketch.

- `hil_energy_noise_analysis.py`
  - Analyzes the noise scan CSV and writes summary stats plus plots.
  - If `model_variant` is present, outputs a grouped summary by
    (`model_variant`, `input_mode`) in addition to the legacy input-mode summary.
  - Outputs to `analysis_scripts/hil_noise_analysis/analysis_output/` by default.

## Test sketch variants

The energy-aware analysis sketches live under `sketches/analysis_sketches/`:

- `tinyodom_tcn_energy_uniform.ino` (uniform [0,5] inputs)
- `tinyodom_tcn_energy_representative.ino` (synthetic inputs using dataset mean/std + clamping)
- `tinyodom_tcn_energy_real_data.ino` (fixed real dataset windows)

The generated header lives alongside them:

- `sketches/analysis_sketches/tinyodom_tcn_input_data.h`

## Config selection

Set the following in `src/nas_config.yaml` to choose a variant when `energy_aware: true`:

- `input_mode: "standard"` uses `sketches/tinyodom_tcn_energy.ino`
- `input_mode: "uniform"` uses `sketches/analysis_sketches/tinyodom_tcn_energy_uniform.ino`
- `input_mode: "representative"` uses `sketches/analysis_sketches/tinyodom_tcn_energy_representative.ino`
- `input_mode: "real"` uses `sketches/analysis_sketches/tinyodom_tcn_energy_real_data.ino`

## Two-Machine Workflow (GPU Train + HIL Scan)

```bash
# 1) Optional: regenerate representative/real input header
python analysis_scripts/hil_noise_analysis/oxiod_input_profile.py --split train --export-header sketches/analysis_sketches/tinyodom_tcn_input_data.h --real-window-count 10

# 2) On the GPU host, train and package the fixed 50-epoch artifact
python analysis_scripts/hil_noise_analysis/train_noise_scan_model.py \
  --config src/nas_config.yaml \
  --epochs 50 \
  --out-dir analysis_scripts/hil_noise_analysis/artifacts \
  --artifact-prefix noise_scan_50ep

# 3) Copy artifacts from GPU host to HIL host
scp analysis_scripts/hil_noise_analysis/artifacts/noise_scan_50ep.keras <hil_host>:<repo>/analysis_scripts/hil_noise_analysis/artifacts/
scp analysis_scripts/hil_noise_analysis/artifacts/noise_scan_50ep.json <hil_host>:<repo>/analysis_scripts/hil_noise_analysis/artifacts/

# 4) On the HIL host, run trained vs untrained scan across the three input modes
python analysis_scripts/hil_noise_analysis/hil_energy_noise_scan.py \
  --model-variants trained_50ep,untrained \
  --input-modes uniform,representative,real \
  --trained-checkpoint analysis_scripts/hil_noise_analysis/artifacts/noise_scan_50ep.keras \
  --trained-meta analysis_scripts/hil_noise_analysis/artifacts/noise_scan_50ep.json \
  --csv-path hil_energy_noise_scan.csv

# 5) Analyze grouped results
python analysis_scripts/hil_noise_analysis/hil_energy_noise_analysis.py --csv hil_energy_noise_scan.csv
```

## Key CLI Flags

- `hil_energy_noise_scan.py`
  - `--model-variants` (default: `trained_50ep,untrained`)
  - `--trained-checkpoint` (required when any `trained*` variant is requested)
  - `--trained-meta` (optional metadata JSON)
  - Use `--model-variants untrained` if you only want legacy untrained behavior.

- `train_noise_scan_model.py`
  - `--epochs` (default: `50`)
  - `--out-dir` (default: `analysis_scripts/hil_noise_analysis/artifacts`)
  - `--artifact-prefix` (default: `noise_scan_50ep`)
  - `--export-tflite` (optional)
  - If pandas import fails with a GLIBCXX/libstdc++ mismatch, the script falls back automatically.
