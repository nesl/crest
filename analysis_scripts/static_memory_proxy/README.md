# Static Memory Proxy

Offline prototype for adding a second cheap proxy line beside FLOPs for Case 1 OdomTCN analysis. This does not modify NAS training, configs, export, flashing, or HIL. It reads logged trial CSVs, rebuilds each OdomTCN candidate from logged hyperparameters, and writes an augmented CSV with static memory proxy columns.

## Proxy Definition

The script estimates static memory traffic as:

```text
memory_traffic_bytes =
  sum_layers(input_activation_bytes + layer_weight_bytes + output_activation_bytes)
```

The prototype assumes batch size 1 and deployment dtype bytes:

```text
float    -> 4 bytes
int8_ptq -> 1 byte
```

If a row does not expose quantization mode, the script defaults to `int8_ptq` and records that in `proxy_quantization_mode_source`.

## Outputs

The augmented CSV keeps all original columns and adds:

```text
proxy_weight_bytes
proxy_activation_bytes
proxy_memory_traffic_bytes
proxy_dtype_bytes
proxy_warning_count
proxy_quantization_mode
proxy_quantization_mode_source
```

Use `--include-layer-details` to also write `proxy_layer_details_json`.

## Run Case 1 STM32

From the repository root:

```bash
python analysis_scripts/static_memory_proxy/compute_static_memory_proxy.py \
  --config src/config/nas_config_case1_5_stm32_b2b_oxiod.yaml \
  --trials-csv models/OxIOD_STM32_B2B_case1_5_t1/trials.csv \
  --output-csv analysis_scripts/static_memory_proxy/stm32_trials_with_memory_proxy.csv \
  --plot \
  --plot-dir analysis_scripts/static_memory_proxy
```

This writes:

```text
analysis_scripts/static_memory_proxy/stm32_trials_with_memory_proxy.csv
analysis_scripts/static_memory_proxy/stm32_trials_with_memory_proxy_flops_vs_memory_traffic.png
analysis_scripts/static_memory_proxy/stm32_trials_with_memory_proxy_rmse_total_vs_memory_traffic.png
analysis_scripts/static_memory_proxy/stm32_trials_with_memory_proxy_energy_mj_per_inference_vs_memory_traffic.png
```

For multiple input CSVs, use `--output-dir` instead of `--output-csv`.

## Plots

With `--plot`, the script writes scatter plots for available columns:

```text
flops vs proxy_memory_traffic_bytes
rmse_total vs proxy_memory_traffic_bytes
energy_mj_per_inference vs proxy_memory_traffic_bytes
```

It also prints Spearman and Kendall rank correlations for each plotted pair. These are rank correlations, so they measure monotonic ordering rather than exact linear fit.

## Warning Count

`proxy_warning_count` is the number of layer-level estimates where the script could not read exact symbolic activation tensors directly from Keras and used an inference path. OdomTCN uses the custom `TCN` layer; its residual blocks and child layers are visible, and weights are counted directly, but nested child input/output tensors are not exposed cleanly by Keras after build. The script infers those internal activation shapes from timestep count and channel/filter counts and marks them as warnings.

A nonzero warning count does not mean the row failed. It means part of the estimate is architecture-aware static inference rather than a direct Keras tensor-shape read.

## Limitations

This is not predicted latency and not measured energy. It ignores cache behavior, DMA, tiling, operator fusion, im2col or temporary buffers, allocator overhead, flash/SRAM placement, alignment, backend-specific rereads, and kernel implementation details. Use it as a ranking proxy to compare against FLOPs and measured energy before deciding whether to wire it into NAS.
