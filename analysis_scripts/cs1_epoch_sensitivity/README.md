# CS1 epoch sensitivity: four-point original-front pilot

This is a deliberately small **within-front pilot**, not a fresh NAS search or
proof that 55 epochs preserves the best candidates from the entire search space.
It selects four energy-spaced members of an existing measured-energy OxIOD/TCN
Pareto front, trains new trajectories, and compares validation error at increasing
budgets. The four points are selected from the source log; there were no four
previously identified trial IDs to hard-code.

The hypothesis is that early training retains useful tradeoffs among these
candidates. Record reversals, failures, and negative results too. Restricting the
panel to original-front candidates **cannot detect slow starters outside it**.
The prepared manifest freezes all selection decisions before new training.

## Scope and reuse

- Existing `crest.pareto_replay`: parsing, objective resolution, validity checks,
  dominance, payload reconstruction and deduplication. Additional guards reject
  negative RMSE and nonpositive energy/latency sentinels.
- Existing OxIOD split loader and adapter: identical training/validation feature
  preparation, with no held-out test split loaded.
- Existing component bootstrap, Odom model family, task compilation and search
  fit plan; original batch size and per-candidate quantization are retained.
- Existing task ModelCheckpoint and EarlyStopping callbacks: stopping is observed
  virtually so the live trajectory can continue without restoring old weights.
- Existing NAS client's TFLite export, subprocess predictions and task evaluation.

No existing production file needs modification. This script never starts a HIL
server, opens a board connection, compiles firmware, flashes hardware, runs Optuna
search, or modifies source-run artifacts.

**Energy is held at each candidate's historical measured value.** Figures and
tables explicitly label this assumption. They are conditional error–energy
fronts, not newly measured checkpoint-specific deployment fronts. Original
`approx_trained` HIL artifacts need not match newly exported trained checkpoints.
Keep the checkpoints/exports for a later, separately authorized hardware replay.

## Commands (examples only; nothing is scheduled)

Run from the branch checkout using the existing CREST Python/Conda environment.
`prepare` needs PyYAML but does not import TensorFlow, load a dataset or train.
`run` requires CREST's full dependencies; `analyze` also needs Matplotlib.

### 1. Prepare only

```bash
python analysis_scripts/cs1_epoch_sensitivity/epoch_front_pilot.py prepare \
  --source-run-dir /home/joseph/Projects/tinyodom-ex/models/OxIOD_PORTENTA_M7_B2B_case1_3_t1 \
  --output-dir /home/joseph/Projects/tinyodom-ex/experiments/cs1_epoch_front_pilot \
  --count 4 --budgets 15,30,55,100,200,300 --seeds 17
```

Selection sorts the valid, deduplicated original front by measured energy and
takes evenly spaced positions, including both extremes. It fails if fewer than
four eligible points exist; it never silently substitutes dominated candidates.
The manifest records the entire selected source rows, config/log hashes, source
row indices, original-front count, code revision, budgets and seeds. CSV row
indices are **not** presented as Optuna trial numbers when the log lacks those.

Inspect `manifest.json` before agreeing to train. An existing output directory is
never overwritten. Use `--source-csv`/`--source-config` for explicit filenames.
The pilot retains all valid HIL points, with original latency saved; it does not
silently turn the 200 ms annotation into a new selection constraint.

### 2. Train/evaluate later, only after explicit authorization

```bash
python analysis_scripts/cs1_epoch_sensitivity/epoch_front_pilot.py run \
  --experiment-dir /home/joseph/Projects/tinyodom-ex/experiments/cs1_epoch_front_pilot \
  --dataset-dir /home/joseph/Projects/tinyodom-ex/data/oxiod \
  --execute
```

Without `--execute`, `run` raises before runtime imports or dataset access.
Nothing invokes this command automatically when the branch is checked out.
Four candidates × 300 epochs is up to 1,200 model-epochs for one seed; this is
not an elapsed-time estimate. Exports/validation add cost. Start with one seed;
choose extra complete seed blocks before outcome inspection.

Training is one continuous trajectory per candidate/seed. At each budget save:

- `budget_only`: the lowest-validation-loss checkpoint seen by that budget.
- `original_policy`: the checkpoint available under CREST's original patience
  rule and that cap. After virtual early stopping, this policy's selected weights
  stay frozen even if the continuing trajectory later improves.

The original task's checkpoint selection criterion remains `val_loss`, not the
best observed TFLite RMSE. Each selected checkpoint is subsequently evaluated
using TFLite validation `rmse_total` (sum of x/y velocity RMSE), in its original
float/INT8 mode. Evaluation occurs after fitting, so checkpoint loading cannot
reset the active training model or its optimizer. Calibration uses training data.
Identical checkpoint hashes reuse validation exports/results within a trajectory.

The 300-epoch endpoint is a reference budget, not a convergence claim or the
original 990-epoch final-retraining procedure. Validation is never combined with
training. Historical 55-epoch numbers are selection provenance, not substituted
for the newly trained trajectories' own 55-epoch measurements.

## Outputs and interpretation

- `manifest.json`, source/effective configs and `execution.json`: frozen selection,
  source hashes, dataset train/validation-list hashes, code/runtime versions,
  dimensions and seeds. Split-list hashes do not hash every sensor data file;
  keep the dataset immutable for the experiment.
- `row_NNNN/seed_N/`: numeric epoch history, virtual stopping state, best/last and
  uniquely named budget checkpoints, evaluation results and status/failure files.
- `exports/`: retained TFLite models used for validation.
- `checkpoint_metrics.csv`, `summary.json` and one two-panel figure per seed.

The figure shows budget-specific conditional fronts and compares the **reference
performance** of the 55-policy-selected subset with the reference front of all
four panel candidates. The summary reports RMSE regret at each panel energy
level, shortlist size and coverage failures. If all four remain nondominated,
zero shortlist regret is weak evidence: no choices were actually eliminated.

```bash
python analysis_scripts/cs1_epoch_sensitivity/epoch_front_pilot.py analyze \
  --experiment-dir /home/joseph/Projects/tinyodom-ex/experiments/cs1_epoch_front_pilot
```

Analysis refuses incomplete panels rather than reporting only successful
trajectories. It never omits candidates because they weaken the hypothesis.
One seed supports a descriptive within-panel finding, not seed robustness,
universal 55-epoch adequacy or full-search fidelity.

## Safety and interruptions

Source logs/configurations and old checkpoints are read-only. All new outputs
are under the explicit experiment root. The prepared code/config identity must
match at execution. `--skip-completed` can skip fully completed trajectories
under the same manifest and dataset split hashes; it refuses any partial
trajectory. There is **no automatic mid-trajectory resume**: saved model optimizer
state alone does not establish exact RNG/shuffle/callback restoration. Preserve
failed/partial runs and prepare a new output directory for a documented restart.

Python/TensorFlow seeds are set, but GPU determinism is not guaranteed. Do not
claim bitwise repeatability. No API keys, LLM provider or external service is used.

## Local tests

```bash
python -m pytest -q test/test_cs1_epoch_front_pilot.py
```

The tests use synthetic CSVs and manually driven callbacks; they do not train
on OxIOD or operate hardware. The real Keras callback test is skipped if
TensorFlow is unavailable. Run it in the CREST environment before a real study.
