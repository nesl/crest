# Pareto Front Comparison Calculations

This folder contains calculation-only scripts for comparing CSV-derived Pareto
fronts. They consume existing NAS or replay artifacts; they do not run NAS,
retrain models, or replay hardware.

## Generic Front Comparison

`compare_pareto_fronts.py` compares two CSV-derived Pareto fronts using explicit
column names and matching semantics. It is intended for users who rerun NAS or
replay experiments and want to compare the resulting fronts without relying on
paper-specific paths.

```bash
python -B analysis_scripts/compare_pareto_front_calcs/compare_pareto_fronts.py \
  --source-csv path/to/source/replay_results.csv \
  --target-csv path/to/target/trials.csv \
  --source-quality-col source__metric__rmse_total \
  --source-cost-col target__energy_mj_per_inference \
  --target-quality-col values_rmse_total \
  --target-cost-col values_energy_mj_per_inference \
  --source-status-col replay_status \
  --source-status-values completed \
  --target-status-col state \
  --target-status-values COMPLETE \
  --source-filter target__latency_ms le 200 \
  --match-rule nearest-quality \
  --reduction-direction target-vs-source \
  --reduction-denominator source \
  --output-dir outputs/front_compare/example
```

The command writes:

- `manifest.json`: command, input paths, columns, filters, matching rule,
  denominator, formulas, and row counts.
- `source_front.csv` and `target_front.csv`: recomputed non-dominated fronts.
- `matches.csv`: source-front to target-front matches, with source/target row
  IDs, quality values, cost values, quality gap, oriented cost delta, fallback
  status, and source/target denominator reduction percentages.
- `summary.json` and `summary.md`: dominance counts and median reductions.

Reduction orientation is explicit. Reductions are numeric cost deltas; with the
default `--cost-direction minimize`, positive values are energy/cost reductions.
With `--reduction-direction target-vs-source`, positive reductions mean the
target front has lower numeric cost:

```text
oriented_cost_delta = source_cost - target_cost
source_denominator_reduction = (source_cost - target_cost) / source_cost
target_denominator_reduction = (source_cost - target_cost) / target_cost
```

With `--reduction-direction source-vs-target`, positive reductions mean the
source front has lower numeric cost:

```text
oriented_cost_delta = target_cost - source_cost
source_denominator_reduction = (target_cost - source_cost) / source_cost
target_denominator_reduction = (target_cost - source_cost) / target_cost
```

Use `--match-rule nearest-quality` for nearest-front RMSE-style comparisons.
Use `--match-rule equal-or-better-quality` when each source-front point should
match the best-cost target-front point at equal-or-better quality, falling back
to nearest quality only when no such target point exists.

Use repeatable `--source-filter COLUMN OP VALUE` and `--target-filter COLUMN OP
VALUE` options for explicit numeric feasibility gates, such as deadline-miss or
latency-budget audits. Supported operations are `lt`, `le`, `eq`, `ge`, `gt`,
and `ne`.

By default, status filters accept `COMPLETE`, `completed`, `success`,
`succeeded`, `1`, and `true` when a status column is supplied. The tool filters
blank, non-finite, sentinel-sized, and nonpositive minimized cost values. Use
`--sentinel-abs-threshold` and `--allow-nonpositive-cost` only when those
defaults do not match the artifact being audited.
