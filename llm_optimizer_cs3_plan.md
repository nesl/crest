# Plan: LLM optimizer on the CS3 STM32 study

Draft, updated 2026-10-13 after offline CS3 preparation on `codex/llm-generator-mvp`. No CS3 LLM campaign
has been run. This document is an implementation plan, not launch authorization.

## Preparation status (steps 1–3)

- Spending/output safeguards, Luna request compatibility, and memory-suspension
  run-log warnings are implemented on the LLM branch with offline tests.
- `experiments/cs3/warmup_params.json` contains validated baseline trials 0–14;
  `baseline_provenance.json` records SQLite/CSV/YAML hashes and startup evidence.
  Trial 6 was historically PRUNED for an infrastructure failure; its parameters
  remain in order. The exact deployed baseline revision was not archived.
- `experiments/cs3/nas_config_cs3_luna.yaml` preserves archived scientific settings,
  including `export_variant: untrained`, with relocated STM32 project/memory-pool paths and new
  output names. It selects Luna/medium, standard tier, omitted temperature,
  memory on at 15/15, recent15/anchors5, batches5, a $5 local cap, 100,000 input
  allowance and 8,192 output tokens. Pricing assumptions and limitations are in
  `experiments/cs3/README.md`.
- Preparation steps 1–6 were offline. The user subsequently ran the bounded
  paid API smoke successfully; no training, hardware preflight, deployment, or
  campaign launch has been performed as part of this preparation.
  Step 5 offline campaign smoke now passes: 15 fixed plus 20 generated trials,
  summaries at 15/30, restart after warm-up, a partially evaluated batch, and
  trial 30. Synthetic objectives and blocked network access were used.
  Step 6 is complete: see `experiments/cs3/cost_estimate.md`. Offline proxy
  costs are about $0.11–$0.38 at 300 attempts before retries, depending on
  output/reasoning use. The suspended backlog issue is resolved by setting
  `memory.max_pending_trials: 15`; the updated proxy fits at 48,860/100,000
  input guard units. The user-run API smoke passed; next is deployment verification and bounded HIL preflight.

## Question

Can an LLM, given a measured warm-up, drive a real hardware-in-the-loop NAS on the
STM32 N657 to a result comparable with TPE, under the same evaluation pipeline? How
does it search (validity, repeats, energy, exploration)?

Paper framing: CREST keeps the evaluation boundary fixed (training, quantization,
HIL measurement, scoring, logging) and swaps only the source of candidates. This
extends CS3 (or becomes a short fourth study). The paper's wording "keeps the
optimizer fixed" would become "keeps the evaluation boundary fixed".

## Design

| | Baseline (exists) | LLM run (new) |
|---|---|---|
| Study | `UrbanSound8K_DSCNN_STM32_AUDIO_case3_2_t1` | new name, e.g. `..._case3_llm_t1` |
| Board | STM32 N657, cadenced 2000 ms, int8 PTQ, external-flash weights | identical |
| Training | 55 epochs, UrbanSound8K log-mel, DS-CNN, same splits | identical |
| Score | macro F1 - 0.1 x energy / 400 mJ; latency-budget feasibility | identical |
| Budget | `nas_trials: 200`, `max_total_trials: 300` | identical |
| First 15 attempts | historical configurations; verify actual startup boundary | **the same 15 configurations, same order**, freshly measured and trained |
| Trials 15+ | TPE (multivariate, constraints) | LLM proposals, 5 per request |

Proposed LLM settings: `gpt-6-luna` (verify the provider model ID, supported request
fields, and pricing before use), standard tier, memory on with summaries every 15 terminal attempts, `recent_trial_window: 15` (first request sees
the whole warm-up), `anchor_count: 5`, semantic context on, temperature omitted
unless the model accepts it, one repair attempt then CREST's random fallback.

Agreed scope: one new LLM campaign compared with historical TPE only, with no
new TPE run or replicated searches. This establishes feasibility and search
behavior, not optimizer superiority. A fresh TPE campaign remains possible later.
The 15 warm-up attempts count toward the existing 200-feasible/300-attempt budget;
they are not 15 extra free evaluations. Cadenced telemetry is collected, but the
score remains based on `energy_mj_per_inference`, not energy per cadence release.

## What we compare (fixed before the run)

Primary:
- Best scored candidate within the same stopping policy (200 feasible completions, at most 300 attempts) and best score against
  hours and against trial number (baseline best: 0.763 at trial 186, 14.0 h).
- Report the observed score gap. A proposed 0.03 practical tolerance is not a
  statistical equivalence/superiority test; the earlier 0.03-0.05 noise estimate
  needs source verification and a definition of what was repeated.
- The selected model's held-out fold-10 result, as CS3 reports it.

Secondary (how it searches):
- Valid proposal rate, repair and random-fallback counts, duplicate rate.
- Distance of each proposal to the nearest earlier trial (as in
  `adhoc/targets_novelty.py`): does it explore or stay local?
- Energy and FLOPs of proposals against TPE's; the replay found LLMs ignore energy.
- Latency rejections, API cost, and request time.

Measurement-drift check: the 15 shared warm-up configurations are measured in
both runs. Compare paired energy, latency, quality, and score separately: training
variation and hardware measurement drift are different effects. Agreement on 15
configurations cannot certify equivalence across the search space. Report runtime
changes and drift; do not apply a blanket score correction. Include attempts,
feasible completions, and elapsed time including provider waits in comparisons.
TPE adapts after each trial; the LLM receives feedback between batches of five.

Hypotheses from the replay, stated in advance: the LLM gets ahead early
(trials 15-60); it may plateau later because it sees only 15 recent records plus
5 best; it proposes higher-energy models than TPE.

Optional later: cross-board replay of the LLM's selected model on the Portenta
(existing CS3 replay workflow), and a Portenta LLM run to test whether it learns
the memory limit (28 arena rejections in the Portenta baseline).

## Implementation status and single branch

Use **`codex/llm-generator-mvp` at `1012d03`** for all CS3 implementation.
This commit is pushed to origin. Its active worktree is
`/Users/jzales/.codex/worktrees/llm-token-estimator/fall-2026-nesl-crest`.

The former proposal-contract branch was a strict descendant of the LLM branch,
not a competing implementation. The LLM branch was fast-forwarded to preserve
all of its shared optimizer work and the merge of main, then the warm-up was
committed on top. `codex/crest-proposal-contract` has been deleted locally and
remotely. No rebase or new integration branch is needed. Continue on this one
LLM branch; do not use the dirty historical CS2 checkout as an implementation base.

Already implemented; reuse rather than rebuild:

- LLM provider transport, prompts, semantic context, recent history, best anchors,
  optional memory, repair, duplicate handling, random fallback, and request ledger.
- Shared optimizer contract, candidate validation, provenance, queue draining,
  budget accounting, optimizer identity and resume protections.
- Reusable CS2 fixes from PR #34, merged into main at `8de2fe9` and carried into
  this branch through `064c865`: STM32 timing/measurement acceptance and evidence,
  typed infrastructure failures, objective eligibility, and failure-streak handling.
- Fixed measured warm-up, committed at `fc3a595`, with focused restart and budget
  regressions.
- Review fixes committed and pushed as `a6aeb0b` and `1012d03`: typed provider
  failures, configurable transport retries, improved deduplication, bounded memory
  repair and persistent suspension, explicit trial-number recovery, YAML-relative
  warm-up paths, shared serialization, and runner/ledger cleanup.
- `optimizer: null` or an omitted optimizer keeps native Optuna (scalar TPE or
  multi-objective NSGA-II). CS3's LLM run must explicitly use `llm_generator`.

Latest offline verification of the prepared changes: **991 tests and
340 subtests passed, 1 skipped**. An additional service-tier resume regression
also passed in the targeted spending suite (17 tests/33 subtests). The existing environment has `pydantic 2.13.4`;
no new environment was needed. Reproduce from the LLM worktree with:

```bash
PYTHONPATH=src MPLCONFIGDIR=/private/tmp/crest-pr-mpl \
  /opt/anaconda3/envs/nesl-tinyodomex/bin/python -m pytest -q \
  test/ analysis_scripts/llm_token_cost/test_estimate.py
```

The reviewer's inability to run tests does not supersede this verification.
These tests do not establish paid-provider compatibility or hardware readiness.

See `optimizer_system_guide.md` and `src/config/README.md` in the LLM worktree.

## CS2 integration: complete in source, deployment validation remains

No further wholesale CS2 source port is needed. The integrated firmware uses
`rtc_wall_v4_dwt_active_stop_evidence_awake_tail`; historical CS2 logs used v3.
Freeze the source and firmware actually deployed for CS3, and report the timing
protocol difference when comparing with the historical baseline. Stop counters
prove entry, not Stop residency duration. Whole-board energy and pre-training
HIL export semantics still apply; this is not exact trained-weight replay.

Before launch, validate the integrated continuous/cadenced acceptance policy for
CS3 audio at **2000 ms** on hardware. Keep the inference-energy score and original
search space; do not adopt CS2's 500 ms cadence objective or clock search. Confirm
infrastructure failures, raw measurement evidence, and eligibility reach the
saved trials through the deployed server/client versions.

CS2 production outputs, E4/fixed-window experiments, micro-workload probes, CS1
utilities, and live progress plotting are not prerequisites for CS3. Record the
actual harness version without importing unrelated historical checkout changes.

## Warm-up: implemented; study inputs still required

`optimizer.llm.warmup_params_file` accepts an ordered nonempty JSON list of
complete raw parameter dictionaries. For CS3, supply **exactly 15** baseline
configurations; the reusable implementation does not hard-code the list length.

```yaml
optimizer:
  type: llm_generator
  llm:
    # Merge with the selected provider/model settings.
    warmup_params_file: experiments/cs3/warmup_params.json
    recent_trial_window: 15
    batch_size: 5
    memory:
      enabled: true
      start_after_trials: 15
      interval_trials: 15
```

The path is illustrative: the baseline JSON and final CS3 LLM YAML have **not**
been prepared. With `load_config`, relative paths resolve from the YAML directory;
use a deployed absolute path when constructing the component directly.

Implemented behavior:

- Validate all entries before evaluation; hash ordered contents in study identity.
  Moving or reformatting the file is allowed; changing its candidates is not.
- Return one fixed candidate per round through the existing evaluation pipeline,
  with source/hash/index provenance. No provider calls occur during warm-up.
- Failed/pruned attempts count toward the 15 slots and existing budgets. Normal
  safety stops and budget limits still apply; warm-up does not bypass them.
- Resume completed/queued work without replay. After confirming all previous
  workers stopped, `optimizer.recover_orphaned_trials: [12]` marks only the
  listed abandoned RUNNING trials FAIL with an audit record. Missing or WAITING
  targets are rejected; terminal targets are harmless on a later invocation.
  Provider configuration and credentials are checked at startup.
- After all entries are terminal, generate ordinary LLM batches. Setting the
  recent window to 15 includes all warm-up outcomes in the first request; this
  window is configured explicitly, not automatically enlarged by warm-up.

Offline tests cover ordered attempts, failures/prunes, first-prompt history,
partial/queued resumes, changed hashes, invalid parameters, orphan RUNNING trials,
and production/smoke budgets using mocked objectives and the fake provider.

Extraction is complete: SQLite, CSV, and recorded hyperparameters agree for all
15 descriptor-valid configurations. Trial 6 is PRUNED with HIL_MASTER_FATAL;
its complete parameters are retained. The canonical ordered-content SHA-256 is
`f1d5025c7e742f64e0d8c8e1b9c2cfb11b50ec04e02eec674d9f0ab4a7a6407e`.
Historical source supports startup 15; exact deployed revision/version is missing.
Freshly evaluate these configurations; do not inject historical scores.

## Remaining implementation and launch gates

### Provider compatibility and bounded spending

- Temperature omission is implemented with `temperature: null`; optional
  `reasoning_effort` and `service_tier` are passed to the provider and recorded in
  optimizer identity. The prepared Luna config uses medium/default. Offline
  transport tests pass; real account/model acceptance still needs a paid smoke.
- Spending/output limits are implemented on the LLM branch: `max_usd`, `max_output_tokens`, `output_token_parameter`,
  `max_input_tokens`, input/output per-million token prices, and `pricing_source`.
  One persistent tracker covers generation, repairs, memory, and every HTTP retry.
  It reserves full token ceilings before each attempt, settles valid usage, and
  retains conservative charges for missing usage or crashes. Cap exhaustion stops
  search without fallback. Warm-up makes no API calls or reservations.
- Before launch, choose verified model/tier prices, a supported output-token field,
  prompt/output ceilings, and a dollar cap. The input guard uses serialized bytes
  plus framing allowance, not model-specific tokenization. The local cap depends
  on correct prices and provider token-limit compliance; it is not an invoice
  guarantee. Preserve `llm_optimizer/spend_state.json` on resume; only the dollar
  cap can change without changing its bound tariff/token settings. Enabling a cap
  over old requests without accounting is rejected. Commit/deploy the new code
  before the paid smoke.
- Reuse `openai_compatible` with `https://api.openai.com/v1`, `OPENAI_API_KEY`, and
  JSON response mode. Verify the actual provider model ID and supported request
  parameters in a bounded real-API smoke; do not assume Codex model labels or
  draft prices establish API availability.
- Phase policy v2 is frozen: use the greater of attempted/300 and feasible/200.
  Specification begins at 70 feasible completions or 105 attempts; finalization
  at 170 feasible completions or 255 attempts, whichever comes first. Warm-up
  counts toward progress. Actual stopping limits and objective are unchanged.
  The version is recorded in optimizer identity; v1 campaigns cannot silently
  resume with changed guidance.

## Second re-review: operational follow-ups

The second re-review found no additional campaign-blocking defect in the reviewed
fixes. That is a code-review disposition, not confirmation that the CS3 inputs,
spending controls, deployment, or hardware preflight are complete.

- **Retry policy is implemented.** Defaults are 5 retries, 5-second exponential
  backoff, and a 60-second per-wait cap. Configure
  `optimizer.llm.max_transport_retries`, `retry_backoff_s`, and
  `retry_backoff_cap_s` for the campaign. `Retry-After` seconds/dates are respected;
  a requested wait above the cap stops the run rather than retrying early. Choose
  and record an acceptable cap before launch. These are transport controls, not
  spending or output-token limits; execution failures never become random search.
- **Memory is on for CS3 (user decision).** Set `start_after_trials: 15` and
  `interval_trials: 15`, alongside `recent_trial_window: 15` and `anchor_count: 5`.
  Summarization begins after the 15 warm-up attempts; at five proposals per batch,
  later updates normally occur every three batches. Thresholds count terminal
  attempts, including FAIL/PRUNED. Pending history bridges summary intervals.
  Three consecutive exhausted summary chunks suspend summaries across restarts.
  A prominent NAS-log warning now reports suspension and persisted suspension
  on resume, alongside memory events and prompt state. Explicit
  reset remains available after correcting the cause with workers stopped.
- **Retain every memory version for review (user decision; already implemented).**
  Under `<models_dir>/<study>/llm_optimizer/memory/`, `snapshots/<request_id>.json`
  preserves every accepted new memory with its version, findings, cited trials,
  coverage hashes, and request ID; `state.json` is only the latest state.
  `requests/` retains summary prompts and raw responses, including rejected updates
  and repairs. `optimizer_events.jsonl` records commits, rejections, retirements,
  and suspension. Review accepted versions using `summary_committed` events and
  their request IDs. Archive the entire memory directory after the campaign, not
  just `state.json`; no separate memory logger is needed.
- **Failure deduplication is conservative.** Known model-construction failures
  carry deterministic candidate evidence. Some downstream compile/conversion or
  evaluation failures already become PRUNED or penalized COMPLETE trials and are
  deduplicated. Other later-stage FAIL paths remain unclassified. During preflight,
  inspect actual failure evidence; add narrow classifications and regressions if
  recurring deterministic failures appear. Do not label every unknown failure
  deterministic or exclude infrastructure retries.

## Engineering sequence and checks

1. **Freeze baseline provenance.** Locate the authoritative database, trial export,
   archived YAML, and historical source evidence. Extract/validate the 15 raw
   configurations and record their hash and the actual startup boundary.
2. **Freeze provider safeguards on the LLM branch.** Review and commit the
   implemented spending/output limits and choose their values. Resolve temperature compatibility
   for the chosen model and validate the implemented error/stop behavior; retain
   offline regression coverage. CS2 integration and warm-up implementation are already complete.
3. **Experiment policy frozen.** Historical TPE only, memory-on/15-attempt cadence,
   and phase policy v2 based on both stopping limits.
4. **Prepare config.** Prefer the baseline's archived exact YAML. The checked-in
   template is `src/config/case_study_configs/nas_config_case3_2_stm32_audio.yaml`;
   the exact archived `_t1` YAML has now been retrieved and used for the
   prepared config in `experiments/cs3/`. Preserve
   scientific settings, enable memory with the agreed cadence, and freeze retry settings;
   review output/optimizer additions and every required
   measurement-policy change explicitly rather than promising a names-only diff.
5. **Offline smoke complete.** `test/test_cs3_offline_campaign.py` runs 15 fixed
   plus 20 generated synthetic trials, memory updates at 15/30, and three restart
   boundaries including a partial batch. It verifies retained findings/snapshots,
   queued work, provenance, and no replay, with HTTP/socket connections blocked.
   Existing targeted tests cover repairs, fallback, budgets, and phase/resume
   behavior. This does not validate actual audio training or hardware.
6. **Offline cost estimate complete.** See `experiments/cs3/cost_estimate.md`
   and its reproducible script/JSON report. Historical records are sizing proxies
   only; input tokens use a characters/4 heuristic. Normal prompts fit the guard;
   the original 100-record suspended backlog exceeded it. The campaign now caps
   pending records and catch-up summary chunks at 15, retaining all saved evidence
   and reporting prompt omissions. The updated proxy fits the guard and a
   persisted-suspension regression passes. Keep the $5 cap; no paid call is authorized.
   Local fixture mode is offline; `--count-api` uses network/credentials even
   though it performs no generation. Verify prices and include reasoning/output,
   repairs, and late-run history. Set an explicit cap before paid calls.
7. **User-run paid smoke and HIL preflight.** Check provider acceptance and usage;
   separately validate the integrated 2000 ms audio path, telemetry, raw evidence,
   and eligibility on hardware. A fake-provider smoke cannot establish these.
8. **User-run launch.** Deploy the frozen integrated source to the normal blinky
   and m202-GPU repos using preserved backups and manual tmux/Python commands.
   Watch ledger growth, fallbacks, measurement failures, budgets, and spending.
   Include scalar final retraining/held-out evaluation in the runtime estimate;
   the 14-17 h historical estimate is not a guarantee.

## After the run

- Copy `optuna.db`, `trials.csv`, `summary.json`, the YAML, and the complete LLM ledger (including all memory snapshots, requests, and events) to
  the replay data folder (read-only), as for the existing runs.
- Analysis script (free): best-so-far curves (hours, trials), the drift check,
  proposal behavior metrics, cost. The replay loader reads any scalar study.
- Write-up: results file entry, one figure (best score vs hours, TPE vs LLM, both
  marking the shared warm-up), and a short "CREST Impact" paragraph.

## Measured API smoke (2026-10-14)

User-run bounded smoke passed: one memory update and five valid new candidates,
with two paid requests and no training/hardware. Accounted total $0.000866375.
Memory used 2,237 input / 35 output tokens (15 reasoning); generation used 2,714
input / 460 output (47 reasoning). Usage and full exact prompts are saved in
`experiments/cs3/api_smoke_usage.json` and `api_smoke_prompts.md` locally on the
preparation computer; those two artifacts are not included in the repository push.

| Projection | 200 attempts | 300 attempts |
|---|---:|---:|
| Repeat smoke sizes | $0.025 | $0.038 |
| Larger proxy inputs + measured outputs | $0.056 | $0.085 |
| Proxy inputs + 4,096 outputs/call + 20% extra calls | $0.18 | $0.27 |

The small synthetic smoke does not establish real campaign memory size, reasoning
usage, or search quality. Keep the $0.10–$0.40 ordinary planning range and $5 cap;
calibrate from saved campaign usage. Next: deployment verification and a bounded
hardware preflight, not the full campaign. Full manual commands and the
config-generation helper are in `experiments/cs3/hardware_preflight.md`.
Remote source must be staged first; existing workers must release the board.

## Costs

- Board and GPU: one campaign, about 14-17 h.
- API: offline proxy estimate about $0.11–$0.38 for 300 attempts before retries;
  full configured token limits imply $1.26 for the normal 76-call schedule.
  These are assumptions, not measured bills. Keep the $5 cap and validate usage
  in a separately authorized smoke.
- Remaining work:
  deployment verification and hardware preflight. Basic provider acceptance passed.
  Safeguards, baseline extraction, and campaign configuration are prepared.

## Limits

- One run against one run: shows whether an LLM can drive the search and how it
  behaves, not that it beats TPE.
- The baseline ran earlier on possibly different code; the drift check measures
  how much that matters but cannot remove it.
- The LLM may have general knowledge of audio CNNs but has no access to the
  baseline's results beyond the shared warm-up.

## Decisions needed from you

1. Fourth case study or part of CS3 in the paper.
2. Whether a Portenta run is wanted later.
