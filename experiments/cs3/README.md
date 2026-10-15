# CS3 Luna preparation

Use branch `codex/llm-generator-mvp`. `nas_config_cs3_luna.yaml` is a campaign
configuration with bounded API/HIL preflight passed; campaign launch still requires authorization.
Run from the repository root and use study name
`UrbanSound8K_DSCNN_STM32_AUDIO_case3_llm_t1` when launch is authorized.

The first 15 attempts replay the raw configurations in `warmup_params.json`,
freshly evaluated through the shared runner. These are baseline trials 0–14;
trial 6 was historically pruned for an infrastructure error and remains included.
No historical scores are supplied to the optimizer. `baseline_provenance.json`
records source hashes and validation; `extract_baseline.py --help` describes
reproduction from read-only baseline artifacts. Historical source supports a
15-trial startup, but the exact deployed revision was not archived.

Scientific settings come from the archived baseline YAML, including its
`export_variant: untrained`, 55 NAS epochs, 990 final epochs, fixed int8 PTQ,
2000 ms cadence, score, and 200-feasible/300-attempt budget. Changes outside the
optimizer are the new output names and the existing STM32 project-directory
rename to `sketches/stm32/crest_stm32_lrun` and the relocated memory-pool file
`src/config/stm32_nucleo_mypool.json`. The current measurement implementation
includes subsequent CS2 fixes; paired warm-up results must assess drift.

Luna uses medium reasoning with temperature omitted, standard service tier, five
candidates per batch, the latest 15 trials and five anchors. Memory starts after
15 terminal attempts and updates every 15; every accepted version is retained in
`<models_dir>/<study>/llm_optimizer/memory/snapshots/`. Archive the entire ledger,
including requests, responses, events, state, and `spend_state.json`. Suspended
summaries also produce a run-log warning; inspect and correct the cause before
explicitly resetting the circuit with workers stopped.

The initial local spending cap is $5, shared by generation, memory, repairs, and
HTTP retries. Each request is limited to 8,192 output tokens including reasoning,
with a conservative 100,000-token input allowance. The configured input rate
($0.125/M) includes the documented cache-write premium above ordinary input
($0.10/M); output is $0.50/M. Rates were checked on 2026-10-13 against the
[official Luna page](https://developers.openai.com/api/docs/models/gpt-6-luna).
The cap relies on those standard, short-context, nonregional prices and provider
limits; it is not an invoice guarantee. A full request reserves $0.016596 before
sending, then settles valid usage. Unknown usage retains the reservation.
Preserve spending state on resume; only the dollar cap can be raised in place.

This first campaign compares with historical TPE only; it is a feasibility and
search-behavior study, not evidence of optimizer superiority. Use paired warm-up
results to assess drift. Phase policy v2 follows the greater fraction of the two
stopping budgets: specification at 70 feasible trials or 105 attempts, and
finalization at 170 feasible trials or 255 attempts, whichever occurs first.
Warm-up counts toward these totals. The policy changes prompt guidance only and
is recorded in optimizer identity; existing v1 LLM studies cannot silently resume
under v2.

The offline smoke passed: 15 fixed plus 20 generated synthetic trials, summaries
at 15/30, restarts after warm-up and trial 30, and an interrupted batch with three
queued candidates retained. Findings, snapshots, and provenance survive without
trial replay. Reproduce with `PYTHONPATH=src python -m pytest -q
-p no:cacheprovider test/test_cs3_offline_campaign.py`. This uses a fake provider
and synthetic objective with network access blocked; it does not test model
quality, API acceptance, audio training, or hardware.

Offline cost estimation is complete; see [cost_estimate.md](cost_estimate.md).
CS3 bounds pending prompt history and catch-up summary chunks to 15 records with
`memory.max_pending_trials: 15`. Suspended memory retains the newest 15 pending
records outside the recent 15 and reports omissions; saved evidence is preserved.
The updated backlog proxy fits the 100,000 input guard at 51,109 guard units (semantic context v2).

The updated-context user-run API smoke passed on 2026-10-14: two requests,
$0.00170525 at configured accounting rates. The one-candidate hardware preflight
also passed at `fa1381a`, with ten accepted runs in each measurement phase and
zero cadence deadline misses. See the CS3 plan for costs and evidence references.
Neither result authorizes a full campaign. Semantic context v3 corrects the
energy wording: trial-level `energy_mj_per_inference` is back-to-back energy;
cadenced release energy is separate telemetry. No scoring behavior changed.

The complete manual hardware-preflight commands and stop conditions are in
[hardware_preflight.md](hardware_preflight.md). The config generator prepares
files only; it does not start hardware, training, or a campaign.

The corrected `api_smoke.py` now uses the campaign's shared search-space and
semantic-context builders, including parameter semantics and a versioned MCU
profile: configured clock/runtime, memory-region meanings, external weights,
cadence, and measurement scope. NPU use and usable weights capacity are explicitly
unverified; measured observations take precedence over architectural priors.
Use `--render-only --output /path/to/new-directory` to save both complete prompts
without a provider connection or API key. Scores/states remain synthetic and
dataset metadata is reconstructed offline. The preview uses a fake empty memory
update. The paid smoke tested MCU context v2; the v3 energy-wording correction
is checked offline without another paid request.

The stopped launch exposed summary length/schema failures. The revised config
holds back the newest 15 terminal trials (`memory.lag_trials: 15`): first summary
at 30 attempts, then every 15 eligible records. Repair feedback reports actual
character counts; missing `retire` means no retirements. The memory-v2 identity
requires a fresh campaign rather than silently resuming the preserved old run.
This policy change is being validated offline; no restart is authorized.


### Architecture rationales and automatic progress plots

The draft CS3 config enables `optimizer.llm.require_rationale: true`. New Luna
candidates use `{"params": {"...": "raw search values"}, "rationale": "short design hypothesis"}`.
The rationale must be nonempty and at most 2,000 characters. It is saved in
`crest_proposal.rationale` on the Optuna trial and in
`llm_optimizer/returned_candidates.jsonl`. Historical proposals and random
fallbacks are not assigned invented Luna explanations.

Single-objective maximize runs now refresh `progress/score_vs_time.{png,pdf}`,
`progress/score_vs_trial.{png,pdf}`, and `progress/score_progress.csv` under the
study directory after terminal attempts and on resume/interruption. These show
feasible scores, the incumbent, and the warmup boundary. Failed/pruned attempts
retain their positions; elapsed time includes API calls and operator downtime.
Plotting errors are logged without stopping NAS. The read-only standalone tool
`analysis_scripts/plot_cs3_progress.py` regenerates the same artifacts.

Existing studies require an explicit, stopped-writer contract migration before
turning on rationales. `python -m crest.optimizers.llm.rationale_migration --help`
describes the backup-first helper. Generate both signatures through the normal
runner identity code, retain all other settings, and preserve memory/spend files.
The helper permits only candidate schema 1 to 2 plus `require_rationale: true`,
rejects RUNNING trials, and records the historical trial boundary. Old queued
proposals, if any, remain explicitly grandfathered without retrospective reasons.

## Next-study full-history mode

The opt-in `nas_config_cs3_next_study.yaml` uses Responses, exact columnar
history, decoded deduplication, finite Python-assigned coverage targets, and
bounded refill with generated memory disabled. It is prepared locally only;
do not use it to resume an existing study. See
[next_study_smoke.md](next_study_smoke.md) for offline prompt review and the
explicitly opt-in $0.05 Luna capability test, which never trains or uses hardware.

## Seeded replacement study (prepared, not launched)

The next-study draft targets a fresh `cs3_luna_production_v4` output directory.
`training.random_seed: 0` resets Python, NumPy and TensorFlow before model
construction and search fitting, independently of optimizer and trial number.
`training.deterministic_ops` remains false: enabling it failed on the GPU's
unequal-stride depthwise convolution kernel during an isolated reproduction.
Seeds are controlled, but GPU execution is not guaranteed bitwise deterministic.
The policy is stored in each trial and the study identity; final
fits save a `.randomness.json` sidecar. Changing that policy requires a new
study. The optimizer's `random_seed` is separate. Exact replay still requires
matching data, software and hardware; a single fixed seed does not estimate
training variability or reconstruct historical unseeded runs.

Full-history prompts now balance capacity against the actual score tradeoff;
coverage ties use stable hashed cell ordering instead of shallow-first ordering.
The draft requires rerunning all 15 warmups, not importing their old outcomes.
Completed v2 outputs are archived under `cs3_luna_production_v2_memory_complete`
on both machines, including the GPU's `tinyodom-ex/models` copy. Internal
database study names and comparison-report labels remain unchanged.

Host TFLite evaluation retries XNNPACK-specific runtime errors once with built-in
kernels on the same flatbuffer, restarting the entire split. Other errors and
failed retries still fail evaluation. Each NAS trial keeps its own validation
flatbuffer; interpreter policy and fallback reasons are recorded in trial attrs
and an `.evaluation.json` sidecar. This does not change MCU execution or use
Keras predictions as a substitute for quantized predictions.
