# Reproduce an experiment

Choose whether you need to recalculate a result, rerun a search configuration, or take new measurements of saved candidates. Each operation starts from different inputs.

## Recalculate from saved results

Use the original CSVs, database, or replay results required by the analysis. The [analysis catalog](https://github.com/nesl/crest/tree/main/analysis_scripts) describes how to supply these files. The [figure catalog](https://github.com/nesl/crest/tree/main/analysis_scripts/paper_plots) lists figure-specific commands and output checks.

For example, compare two completed OxIOD measured-energy runs with the same objective units and schema. Replace both paths with your saved `trials.csv` files. If the studies logged deployment feasibility, also filter each input to the feasible rows before comparison.

```bash
python analysis_scripts/compare_pareto_front_calcs/compare_pareto_fronts.py \
  --source-csv /path/to/baseline/trials.csv \
  --target-csv /path/to/comparison/trials.csv \
  --source-quality-col values_rmse_total --target-quality-col values_rmse_total \
  --source-cost-col values_energy_mj_per_inference \
  --target-cost-col values_energy_mj_per_inference \
  --source-status-col state --target-status-col state \
  --source-status-values COMPLETE --target-status-values COMPLETE \
  --match-rule nearest-quality --reduction-denominator source \
  --output-dir front_comparison
```

Check `front_comparison/manifest.json` for the selected columns, row counts, filters, and comparison settings. Inspect the quality gaps and match IDs in `matches.csv` before using the aggregated reductions.

## Rerun a case-study configuration

The [case-study configuration index](https://github.com/nesl/crest/tree/main/src/config/case_study_configs) maps YAMLs to STM32 schedule comparisons, cross-target OxIOD proxy and measured-energy searches, and UrbanSound8K studies. Use the YAML and launch command from the revision that defines your experiment. Prepare the [dataset](setup.md) and, for measured runs, follow [hardware setup](hardware.md). Set local paths and ports while keeping the experiment settings you intend to compare. Choose a fresh study name for new results.

Archived firmware, original checkpoints and measurements, and provider behavior can all affect reproducibility. For a historical comparison, retain the source revision, config, study artifacts, and measurement protocol used for each run.

## Replay selected candidates on a target

Start with a saved multi-objective source run directory containing its YAML and NAS evaluation CSV. Replace the source path with that directory. The example uses target settings from the Nano 33 BLE YAML and previews one selected candidate.

```bash
python src/pareto_hil_replay.py \
  --source-run-dir /path/to/source-study \
  --target-config src/config/nas_config_ble.yaml \
  --device-option-policy target-default \
  --max-candidates 1 --dry-run --output-dir models/replays/ble_preview
```

Check `manifest.json` for the source selection and `replay_requests.jsonl` for the target, quantization, and request settings. A dry run stages request artifacts locally and requires no connected board.

For measurements, run the command on the prepared device host, use a target YAML with the actual ports, drop `--dry-run`, and choose a fresh output directory. The replay tool invokes the HIL implementation locally. It reconstructs logged candidate requests without NAS or retraining. Supply the appropriate checkpoint when the intended model variant needs one. Verify `replay_status` and the target metrics in `replay_results.csv`. Use `--resume` on the hardware results directory to skip completed requests. [Replay source guide](https://github.com/nesl/crest/blob/main/src/README.md#pareto-hil-replay).
