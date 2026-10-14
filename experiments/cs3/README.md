# CS3 Luna preparation

Use branch `codex/llm-generator-mvp`. `nas_config_cs3_luna.yaml` is a campaign
configuration awaiting policy review and API/HIL preflight, not launch approval.
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
rename to `sketches/stm32/crest_stm32_lrun`. The current measurement implementation
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
The updated backlog proxy fits the 100,000 input guard at 48,860 guard units.

Before launch: separately authorize a bounded paid API smoke
and hardware preflight. Account access and actual API acceptance have not been
tested. Do not run this HIL-enabled YAML for an offline smoke.
