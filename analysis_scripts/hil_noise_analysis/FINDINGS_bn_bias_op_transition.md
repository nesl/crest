# Findings: BN/Bias Op-Count Transition

## Summary

The trained-vs-untrained op-count gap is not explained by "randomly initialized kernels" alone.
The transition is strongly tied to BatchNorm state and non-BN bias values.

Key result:

- `fresh_untrained` exported at `69 ops / 4 ADD`
- `trained_checkpoint` exported at `81 ops / 16 ADD`
- `approx_trained` (legacy alias: `bn_full_plus_non_bn_bias_perturbed`) also exported at `81 ops / 16 ADD`

This means training is not the only path to the higher-op graph form; parameter state is.

## What We Tested

We used `analysis_scripts/hil_noise_analysis/op_transition_probe.py` to generate controlled variants:

- `fresh_untrained`
- `bn_gamma_beta_perturbed`
- `bn_moving_stats_perturbed`
- `bn_full_perturbed`
- `bn_calibrated_no_train`
- `non_bn_bias_perturbed`
- `bn_full_plus_non_bn_bias_perturbed`
- `trained_checkpoint`

## Results

### Probe 1: BN-only sweep (`op_transition_probe_output`)

From `analysis_scripts/hil_noise_analysis/op_transition_probe_output/op_transition_probe_summary.txt`:

| Variant | Float ops | Float ADD | Int8 ops | Int8 ADD |
|---|---:|---:|---:|---:|
| `fresh_untrained` | 69 | 4 | 69 | 4 |
| `bn_gamma_beta_perturbed` | 75 | 10 | 75 | 10 |
| `bn_moving_stats_perturbed` | 75 | 10 | 75 | 10 |
| `bn_full_perturbed` | 75 | 10 | 75 | 10 |
| `bn_calibrated_no_train` | 75 | 10 | 75 | 10 |

Interpretation:

- BN perturbation consistently adds `+6 ops / +6 ADD` over fresh untrained.
- BN alone does not reach trained-level op counts.

### Probe 2: BN + bias comparison (`op_transition_probe_output_bias_cmp`)

From `analysis_scripts/hil_noise_analysis/op_transition_probe_output_bias_cmp/op_transition_probe_summary.txt`:

| Variant | Float ops | Float ADD | Int8 ops | Int8 ADD |
|---|---:|---:|---:|---:|
| `fresh_untrained` | 69 | 4 | 69 | 4 |
| `bn_full_perturbed` | 75 | 10 | 75 | 10 |
| `non_bn_bias_perturbed` | 75 | 10 | 75 | 10 |
| `bn_full_plus_non_bn_bias_perturbed` | 81 | 16 | 81 | 16 |
| `trained_checkpoint` | 81 | 16 | 81 | 16 |

Interpretation:

- Non-BN bias perturbation alone gives the same partial jump as BN-only (`69 -> 75`).
- Combining BN and non-BN bias perturbations reproduces the trained op-count pattern (`69 -> 81`).

## Practical Conclusion

1. Untrained models start with random kernels, but many fold-friendly/default BN/bias settings still reduce exported op count.
2. The missing ops appear when BN state and non-BN biases move away from those fold-friendly defaults.
3. Deterministic perturbation (`approx_trained`, legacy alias `bn_full_plus_non_bn_bias_perturbed`) is a valid no-training way to reproduce trained-like op structure.

## Related Findings From Other Analysis Studies

### HIL trained-vs-untrained gap is large and consistent

From `analysis_scripts/hil_noise_analysis/hil_energy_noise_analysis_trained_vs_untrained_v2/summary_by_model_variant_input_mode.csv`:

- Uniform mode:
  - trained latency mean: `265.29025 ms`
  - untrained latency mean: `153.01330 ms`
  - absolute delta: `112.27695 ms` (`+73.38%` for trained)
- Uniform mode energy:
  - trained energy mean: `12.3382329 mJ/inference`
  - untrained energy mean: `7.2981447 mJ/inference`
  - absolute delta: `5.0400882 mJ` (`+69.06%` for trained)

Sample coverage in the paired v2 scan was balanced (`20` runs per model/input group), from:

- `analysis_scripts/hil_noise_analysis/hil_energy_noise_scan_trained_vs_untrained_v2.csv`

### Input mode has a much smaller effect than model state

From the same v2 summary:

- Trained (`trained_50ep`) latency spread across input modes (`uniform/representative/real`): `0.64675 ms`.
- Untrained latency spread across input modes: `0.33925 ms`.
- Trained energy spread across input modes: `0.0561033 mJ`.
- Untrained energy spread across input modes: `0.0477542 mJ`.

This is much smaller than the trained-vs-untrained deltas (`112.27695 ms` latency, `5.0400882 mJ` energy).

### Uniform input is intentionally synthetic versus OxIOD statistics

From `analysis_scripts/hil_noise_analysis/oxiod_stats.csv`:

- Most real-data channels are not concentrated in `[0, 5]`.
- Fraction of samples in `[0,5]` by channel ranges from very low values (for example `acc_y: 0.0112`, `mag_y: 0.0641`) to moderate values (for example `gyr_y: 0.5644`).
- Only `step` is effectively fully in `[0,5]` (`frac_in_0_5 = 1.0`, binary).

Interpretation:

- `uniform` mode is useful as a controlled stress/input baseline, but it is not distribution-matched to OxIOD sensor channels.

### Approx-trained runtime is close to trained runtime on uniform input

From:

- `analysis_scripts/hil_noise_analysis/hil_energy_noise_scan_bn_full_plus_non_bn_bias_perturbed_uniform.csv`
- `analysis_scripts/hil_noise_analysis/hil_energy_noise_analysis_trained_vs_untrained_v2/summary_by_model_variant_input_mode.csv`

Observed means:

- `approx_trained` (legacy alias run): latency `265.50220 ms`, energy `12.346289 mJ`, avg power `46.86585 mW`.
- `trained_50ep` (uniform): latency `265.29025 ms`, energy `12.3382329 mJ`, avg power `46.90655 mW`.

Difference is small:

- latency delta: `+0.21195 ms`
- energy delta: `+0.0080561 mJ`
- avg power delta: `-0.04070 mW`

### Epoch sweep confirms op-count stability after checkpoint training starts

From `analysis_scripts/hil_noise_analysis/epoch_sweep/artifacts/epoch_sweep_training_stats.csv`:

- For all trained checkpoints in the sweep (`50` through `373` epochs):
  - quantized TFLite bytes stayed constant at `41608`
  - quantized op count stayed constant at `81`
  - quantized ADD count stayed constant at `16`
- `fresh_untrained_audit` in the same CSV remained at `34768 bytes`, `69 ops`, `4 ADD`.

Training quality still improved while graph stats stayed fixed:

- best validation loss improved from `0.085080639` at the 50-epoch stage to `0.056600865` (best epoch `333`), about `33.47%` lower.
- early stop occurred at epoch `373` (`noise_scan_epoch_sweep_training_manifest.json`).

Current caveat:

- HIL epoch-sweep runtime CSV coverage is incomplete in this snapshot
  (`epoch_sweep_hil_metrics.csv` has one populated row and `epoch_sweep_hil_metrics_100_200.csv` is empty), so no robust cross-epoch HIL latency trend can be concluded yet.

## Reproducibility Notes

- Combined perturbation is implemented in `src/hil_server.py` under variant `approx_trained` (legacy aliases: `representative`, `bn_full_plus_non_bn_bias_perturbed`).
- The perturbation uses a fixed RNG seed (`1337`) for repeatable values.
- This finding covers graph/op composition; latency/power still require HIL measurement runs.

## Raw Evidence

- `analysis_scripts/hil_noise_analysis/op_transition_probe_output/op_transition_probe_summary.txt`
- `analysis_scripts/hil_noise_analysis/op_transition_probe_output_bias_cmp/op_transition_probe_summary.txt`
- `analysis_scripts/hil_noise_analysis/hil_energy_noise_analysis_trained_vs_untrained_v2/summary_by_model_variant_input_mode.csv`
- `analysis_scripts/hil_noise_analysis/hil_energy_noise_scan_trained_vs_untrained_v2.csv`
- `analysis_scripts/hil_noise_analysis/hil_energy_noise_scan_bn_full_plus_non_bn_bias_perturbed_uniform.csv`
- `analysis_scripts/hil_noise_analysis/oxiod_stats.csv`
- `analysis_scripts/hil_noise_analysis/epoch_sweep/artifacts/epoch_sweep_training_stats.csv`
- `analysis_scripts/hil_noise_analysis/epoch_sweep/artifacts/noise_scan_epoch_sweep_training_manifest.json`
- `analysis_scripts/hil_noise_analysis/epoch_sweep/artifacts/epoch_sweep_hil_metrics.csv`
- `analysis_scripts/hil_noise_analysis/epoch_sweep/artifacts/epoch_sweep_hil_metrics_100_200.csv`
