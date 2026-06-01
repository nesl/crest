# CS3 Audio Score Sensitivity

This folder contains a calculation-only score-sensitivity utility for
CSV-derived NAS candidates. It consumes existing run logs and writes
reproducibility artifacts; it does not train models, run NAS, export models, or
touch hardware.

The utility was used for the Case Study 3 audio-selection analysis, but the
formula and CLI are generic for CSV logs that contain quality and
energy-per-inference columns.

The scoring rule is:

```text
score = quality - lambda * energy / energy_budget_mj
```

Candidate selection is deterministic: score descending, quality descending,
energy ascending, then trial ID as a string ascending.

## Requirements

- The `crest` Conda environment, or Python with `pandas` installed.
- Existing CSV artifacts from a NAS or replay run.
- No hardware access. This tool only reads CSV files and writes summary
  artifacts.

## Usage

```bash
python -B analysis_scripts/cs3_audio_sensitivity/score_sensitivity.py \
  --run "portable-board=path/to/portable_run_dir_or_trials.csv" \
  --run "low-power-board=path/to/low_power_run_dir_or_trials.csv" \
  --output-dir analysis_scripts/cs3_audio_sensitivity/results/example
```

Run inputs may be CSV files or directories. Directory inputs search only
top-level `*.csv` files, skip lock files, and select the best usable log with a
deterministic heuristic. The default sweeps are:

- budgets: `100, 200, 300, 400, 600, 800, 1200`
- lambdas: `0, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30`
- baseline: `lambda=0.10`, `energy_budget_mj=400`

Each input must contain a quality column and an energy-per-inference column.
Optional columns can provide trial IDs, latency, status, feasibility, source
scores, and architecture parameters.

Column detection is automatic by default. Use explicit columns when artifacts
use different names:

```bash
python -B analysis_scripts/cs3_audio_sensitivity/score_sensitivity.py \
  --run "run-a=path/to/run_a.csv" \
  --run "run-b=path/to/run_b.csv" \
  --quality-col validation_macro_f1 \
  --energy-col energy_mj_per_inference \
  --trial-id-col trial_id \
  --latency-col latency_ms \
  --status-col state \
  --feasible-col feasible \
  --source-score-col value_score \
  --arch-col params_depth \
  --arch-col params_width \
  --output-dir analysis_scripts/cs3_audio_sensitivity/results/explicit-columns
```

Override sweep values with repeatable flags:

```bash
python -B analysis_scripts/cs3_audio_sensitivity/score_sensitivity.py \
  --run "run-a=path/to/run_a.csv" \
  --output-dir analysis_scripts/cs3_audio_sensitivity/results/custom-sweep \
  --budget-mj 200 \
  --budget-mj 400 \
  --lambda 0 \
  --lambda 0.1 \
  --baseline-budget-mj 400 \
  --baseline-lambda 0.1
```

## Outputs

The command writes all generated artifacts under the explicit output directory:

- `manifest.json`: command, formula, sweeps, accepted filter values, output
  fields, selected CSVs, column mappings, discovery diagnostics, and candidate
  counts.
- `summary.json`: machine-readable baseline, reference, budget-sweep, and
  lambda-sweep selections.
- `selections.csv`: flat selection table.
- `summary.md`: human-readable summary tables.

`selections.csv` uses this stable schema:

```text
section, run_label, setting, lambda, energy_budget_mj, trial_id, candidate_id,
quality, energy, latency, score, matches_baseline, architecture_fingerprint
```

Filtering preserves the original analysis behavior. Rows must have finite
quality and energy. When a status column is detected or configured, rows must
match the accepted status values. When a feasible column is detected or
configured, rows must match the accepted feasible values. When a source-score
column is detected or configured, rows must have a finite source score.
