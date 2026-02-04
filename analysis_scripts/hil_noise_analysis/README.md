# HIL Noise Analysis

This folder contains scripts and docs for running HIL noise scans and analyzing
how input distributions affect latency and energy measurements.

## Scripts

- `hil_energy_noise_scan.py`
  - Runs a multi-mode (standard / uniform / representative / real) HIL noise scan.
  - Re-syncs the Arduino sketch variant for each input mode and records per-run metrics.
  - Writes a CSV that includes an `input_mode` column so you can compare distributions.

- `oxiod_input_profile.py`
  - Loads OxIOD windows using the same data loader and config parameters used by
    `hil_server.py` / `nas_model_client.py`.
  - Prints per-channel statistics to compare the real dataset to uniform `[0, 5]` inputs.
  - Can export `sketches/analysis_sketches/tinyodom_tcn_input_data.h`, which embeds:
    - Per-channel mean/std/min/max values.
    - A fixed set of real windows (default 10) for the real-data sketch.

- `hil_energy_noise_analysis.py`
  - Analyzes the noise scan CSV and writes summary stats plus plots.
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

## Typical usage

```bash
python analysis_scripts/hil_noise_analysis/oxiod_input_profile.py --split train --export-header sketches/analysis_sketches/tinyodom_tcn_input_data.h --real-window-count 10
python analysis_scripts/hil_noise_analysis/hil_energy_noise_scan.py
python analysis_scripts/hil_noise_analysis/hil_energy_noise_analysis.py --csv hil_energy_noise_scan.csv
```
