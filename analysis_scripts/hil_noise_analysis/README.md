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
  - Can export `sketches/analysis_sketches/tinyodom_input_data.h`, which embeds:
    - Per-channel mean/std/min/max values.
    - A fixed set of real windows (default 10) for the real-data sketch.

- `hil_energy_noise_analysis.py`
  - Analyzes the noise scan CSV and writes summary stats plus plots.
  - If `model_variant` is present, outputs a grouped summary by
    (`model_variant`, `input_mode`) in addition to the legacy input-mode summary.
  - Outputs to `analysis_scripts/hil_noise_analysis/analysis_output/` by default.

- `op_transition_probe.py`
  - Builds controlled model variants to probe when extra TFLite ops appear.
  - Exports float/int8 TFLite artifacts per variant and records file size, total ops,
  and ADD-op count.
  - Writes a detailed CSV plus a compact text summary under
    `op_transition_probe_output/` (or caller-provided output directory).

- `epoch_sweep/`
  - Contains the staged checkpoint workflow:
    - `train_epoch_sweep.py` for GPU-side staged training and artifact export
    - `hil_epoch_sweep_scan.py` for HIL-side checkpoint evaluation
    - `audit_fresh_untrained_tflite.py` for appending a fresh-untrained TFLite
      audit row to the training CSV
  - See `analysis_scripts/hil_noise_analysis/epoch_sweep/README.md`.

## Test sketch variants

Energy-aware input modes map to the following sketches:

- `sketches/tinyodom_inference_energy.ino` (uniform [0,5] inputs)
- `sketches/analysis_sketches/tinyodom_inference_representative.ino` (synthetic inputs using dataset mean/std + clamping)
- `sketches/analysis_sketches/tinyodom_inference_real_data.ino` (fixed real dataset windows)

The generated header lives alongside them:

- `sketches/analysis_sketches/tinyodom_input_data.h`

## Config selection

Set the following in `src/config/nas_config.yaml` to choose a variant when `energy_aware: true`:

- `input_mode: "uniform"` uses `sketches/tinyodom_inference_energy.ino`
- `input_mode: "representative"` uses `sketches/analysis_sketches/tinyodom_inference_representative.ino`
- `input_mode: "real"` uses `sketches/analysis_sketches/tinyodom_inference_real_data.ino`

## Input Mode Definitions

- `uniform`: fills the model input window with random values in `[0, 5]`.
- `representative`: uses synthetic values shaped by OxIOD channel statistics (mean/std with clamping).
- `real`: replays fixed real OxIOD windows embedded in `tinyodom_input_data.h`.

## Two-Machine Workflow (GPU Train + HIL Scan)

```bash
# 1) Optional: regenerate representative/real input header
python analysis_scripts/hil_noise_analysis/oxiod_input_profile.py --split train --export-header sketches/analysis_sketches/tinyodom_input_data.h --real-window-count 10

# 2) On the GPU host, train and package the fixed 50-epoch artifact
python analysis_scripts/hil_noise_analysis/train_noise_scan_model.py \
  --config src/config/nas_config.yaml \
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

## Epoch Sweep Workflow

Use the staged checkpoint flow when you want per-epoch HIL telemetry instead of
just a trained-vs-untrained comparison:

```bash
# 1) Train and export checkpoint stages on the GPU host
python analysis_scripts/hil_noise_analysis/epoch_sweep/train_epoch_sweep.py \
  --config src/config/nas_config.yaml \
  --out-dir analysis_scripts/hil_noise_analysis/epoch_sweep/artifacts \
  --artifact-prefix noise_scan_epoch_sweep

# 2) Optional: append a fresh-untrained TFLite audit row to the training CSV
python analysis_scripts/hil_noise_analysis/epoch_sweep/audit_fresh_untrained_tflite.py

# 3) Run HIL metrics across staged checkpoints
python analysis_scripts/hil_noise_analysis/epoch_sweep/hil_epoch_sweep_scan.py \
  --config src/config/nas_config.yaml \
  --training-csv analysis_scripts/hil_noise_analysis/epoch_sweep/artifacts/epoch_sweep_training_stats.csv \
  --runs 1 \
  --input-modes uniform
```

The full staged workflow, optional flags, and artifact layout are documented in
`analysis_scripts/hil_noise_analysis/epoch_sweep/README.md`.

## Key CLI Flags

- `hil_energy_noise_scan.py`
  - `--runs` (default: `20`)
  - `--cooldown` (default: `20`)
  - `--model-variants` (default: `untrained`)
  - `--trained-checkpoint` (required when any `trained*` variant is requested)
  - `--trained-meta` (optional metadata JSON)
  - `--energy-aware` to force the energy-aware sketch path for the scan
  - Use `--model-variants untrained` if you only want legacy untrained behavior.

- `train_noise_scan_model.py`
  - `--epochs` (default: `50`)
  - `--out-dir` (default: `analysis_scripts/hil_noise_analysis/artifacts`)
  - `--artifact-prefix` (default: `noise_scan_50ep`)
  - `--export-tflite` (optional)
  - If pandas import fails with a GLIBCXX/libstdc++ mismatch, the script falls back automatically.

- `oxiod_input_profile.py`
  - `--export-header` to rewrite `sketches/analysis_sketches/tinyodom_input_data.h`
  - `--export-stats-csv` for per-channel summary CSV output
  - `--real-window-count` (default: `10`) to control the embedded real-window set

- `op_transition_probe.py`
  - `--variants` to choose the exact perturbation sequence to export
  - `--quant-modes float,int8` (default) to compare both conversion paths
  - `--trained-checkpoint` when the `trained_checkpoint` variant is requested

## Key Finding: Op-Count Transition

From the `op_transition_probe.py` experiments:

- Fresh untrained models produce fewer exported ops than trained checkpoints.
- BN-only perturbation closes part of the gap.
- Non-BN-bias-only perturbation also closes part of the gap.
- Perturbing both BN (gamma/beta/moving stats) and non-BN biases matches trained op counts.

Observed op counts (`float` and `int8` showed the same progression):

| Variant | Ops | ADD ops |
|---|---:|---:|
| `fresh_untrained` | 69 | 4 |
| `bn_full_perturbed` | 75 | 10 |
| `non_bn_bias_perturbed` | 75 | 10 |
| `bn_full_plus_non_bn_bias_perturbed` | 81 | 16 |
| `trained_checkpoint` | 81 | 16 |

Detailed write-up:

- `analysis_scripts/hil_noise_analysis/FINDINGS_bn_bias_op_transition.md`

Raw summaries:

- `analysis_scripts/hil_noise_analysis/op_transition_probe_output/op_transition_probe_summary.txt`
- `analysis_scripts/hil_noise_analysis/op_transition_probe_output_bias_cmp/op_transition_probe_summary.txt`
