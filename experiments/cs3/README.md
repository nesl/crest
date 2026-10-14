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

Before launch: settle historical-only versus fresh TPE comparison and the prompt
phase policy (currently finalization at attempt 255 of 300); run the offline
campaign smoke/cost estimate; then separately authorize a bounded paid API smoke
and hardware preflight. Account access and actual API acceptance have not been
tested. Do not run this HIL-enabled YAML for an offline smoke.
