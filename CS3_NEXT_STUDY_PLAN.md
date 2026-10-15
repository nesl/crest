# CS3 next-study plan: full history, Responses, and explicit exploration

Date: 2026-10-15

Revision: Incorporates both rounds of Claude's October 15 red-team feedback and the subsequent read-only verification. The three reported twin pairs match decoded architecture settings; the baseline repeat estimate is reproduced below.

Status: Implemented on `codex/llm-generator-mvp` after user approval, with offline validation. The current campaign remains unchanged. No paid API calls, hardware tests, deployment, or restart were performed. See `experiments/cs3/next_study_smoke.md` for user-run bounded tests, including the separate fixed-prefix cache diagnostic. The objective is rendered deterministically from configuration; rationales remain free text without additional prediction fields. The sections below retain the agreed design and evaluation policy.

## Objective

Second review correction: the next-study and bounded-smoke input guard is 120,000 conservative serialized-input units; the smoke cap remains $0.05. At the configured $0.125/M input rate, this adds $0.0025 to the maximum per-call input reservation, not to every actual bill. Repair feedback includes at most 500 characters of specific error detail, without rejected response bodies. Unclassified failures are still not coverage evidence, but the same decoded network may be attempted at most twice after unclassified terminal failures (original attempt plus one retry); explicit infrastructure failures do not consume this unknown-failure retry budget. Exhausted identities are exposed through the history's `retry_exhausted_trials`. Existing bounded-history campaigns retain their prior behavior.

Review correction (October 15): full-mode coverage and dedup use explicit outcome classification, not `feasible=False` alone. Penalty-path unknown measurements are not interpreted as hardware readings. Repair/refill feedback is bounded and partial accepted batches survive an additional request's input guard. Error explanations are added to a first-observed dictionary as they appear: rows reference their integer IDs, repeated text is stored once, and a new explanation adds a new entry. The dictionary is reconstructed deterministically from the saved chronological history; it is not generated memory. No unseen error explanations are preloaded.

Give Luna access to exact historical architectures and outcomes, make exploration an explicit part of each batch, reduce duplicate proposals, and reuse prompt prefixes through the Responses API. Keep the implementation simple and reuse the existing optimizer and ranker-replay code.

Implementation belongs on `codex/llm-generator-mvp`, in `/Users/jzales/.codex/worktrees/llm-token-estimator/fall-2026-nesl-crest`. Use explicitly selected non-Astra subagents for implementation and review.

## Proposed changes

| Change | What it addresses |
| --- | --- |
| Every trial in a compact chronological table | Keeps exact parameters and outcomes available instead of compressing them into summaries |
| Generated memory summaries disabled | Removes summary drift and repeated reinforcement of a few observations |
| Progress-dependent allocation with Python-assigned exploration targets | Makes exploration a checkable batch requirement |
| Decoded-network identity and Python-computed coverage | Avoids counting raw parameter aliases as different networks |
| Detailed duplicate feedback and bounded refill | Identifies matching historical trials and requests replacements while retaining accepted candidates |
| Responses API with a simple ordered prompt | Reuses replay API construction while avoiding unnecessary cache-write costs and complex block management |
| Separate implementation checks and descriptive score outcomes | Prevents improved coverage or deduplication from being mistaken for better search quality |

## 1. Reuse the existing search phases

The current phase policy computes progress as the greater of:

- Attempts consumed divided by the attempt limit.
- Feasible completions divided by the feasible-trial target.

Warmup contributes to these counters. The existing boundaries are 35% and 85%; today they change prose instructions only. Extend that policy with explicit candidate allocations rather than introducing another progress calculation.

Proposed allocation for five-candidate batches:

This fixed allocation is a hypothesis chosen for simplicity. The observed plateau does not establish that reducing late exploration is optimal. Do not add adaptive plateau detection to this version; evaluate the stated policy first.

| Progress | Existing phase | Local improvements | Exploration |
| --- | --- | ---: | ---: |
| Below 35% | Exploration | 2 | 3 |
| 35% to below 85% | Specification | 3 | 2 |
| 85% and above | Finalization | 4 | 1 |

Keep the 15-trial warmup. With a 200-feasible target and few failures, these boundaries occur around 70 and 170 feasible completions, not attempts 105 and 255. The LLM's early phase therefore covers approximately trials 15–69. The attempt fraction can move the phase earlier if many attempts fail.

Use this explicit smaller-batch policy (each entry is local/exploration):

| Batch size | Exploration phase | Specification phase | Finalization phase |
| ---: | --- | --- | --- |
| 1 | 0/1 | 1/0 | 1/0 |
| 2 | 1/1 | 1/1 | 1/1 |
| 3 | 1/2 | 2/1 | 2/1 |
| 4 | 2/2 | 2/2 | 3/1 |
| 5 | 2/3 | 3/2 | 4/1 |

Keep batch size five deliberately. Refilling five candidates changes the feedback cadence: t2's reported average of about 2.6 accepted candidates per round yielded more opportunities to incorporate new measurements. A full five-candidate batch shares one evidence snapshot, and refill calls add no measured outcomes. This is another bundled policy change, not just an efficiency fix. Report proposal rounds, accepted batch sizes, observations available per round, and API overhead alongside evaluation time.

If no feasible historical candidate exists to improve, request exploration candidates instead. Finalization means generating novel local variants, not conducting repeat measurements. Require distinct decoded parent networks within a batch whenever enough eligible parents exist, including across refill calls. If too few exist, allow reuse explicitly and log the exception. Ask for parents with strong observed results, not proven superiority; a single high score is noisy evidence.

Each proposed architecture retains its rationale and adds an intent (`local` or `exploration`). A local proposal identifies a feasible historical trial and explains the intended change. Python computes and logs the actual raw-parameter and decoded-network differences from that parent, so incorrect prose claims do not become the authoritative record.

During compulsory coverage, Python assigns each exploration slot a least-covered `num_blocks × resolved strides` cell and asks Luna for its best design within that cell. Schedule names are display aliases, not cell identities. Select targets deterministically with documented tie-breaking; account for pending and already assigned slots to avoid assigning every slot to the same cell. Validate exact membership rather than trusting a self-assigned exploration label. Keep targets fixed across refill attempts. Exhausted or impossible targets produce an explicit unfilled-slot reason, not a silent reclassification.

Stop compulsory cell targeting once every cell has two distinct decoded networks with informative terminal outcomes. Warmup outcomes count. Successful measurements, classified deterministic architecture failures, and confirmed infeasibility count toward this target; infrastructure and unclassified failures do not. Retain separate outcome counts so two rejections cannot be mistaken for two successful measurements. An unknown decoded identity cannot satisfy a distinct-network count. The global attempt cap and bounded refill budget remain authoritative even if some cells never meet the target.

For the current 29 resolved cells, this requires at least 58 informative distinct networks, including warmup contributions. It is a breadth requirement over block counts and strides, not coverage of the full search space or its width, kernel, activation, and other interactions. Record the first terminal completion prefix satisfying the goal, its trial number, and available time axes; separately retain the proposal-round log showing when the optimizer first used the completed-goal policy.

After that coverage goal, retain the phase's exploration allocation but request outcome-informed exploration instead of indefinitely rotating through least-covered cells. Luna may revisit promising regions or test uncertain ones, using full history and coverage. Label these slots explicitly as outcome-informed rather than Python-targeted; their scientific exploratory value is assessed descriptively, without an arbitrary distance gate. Two observations do not justify permanently excluding a cell. Log when the targeting policy changes.

Validate intent, quota, target membership, and parent references. Do not add an arbitrary nearest-neighbor distance threshold. Log distance to the nearest earlier architecture as a diagnostic instead.

## 2. Supply full compact history and disable generated summaries

Reuse existing history extraction and canonical parameter reconstruction internally, but serialize the prompt as one column list plus row arrays, with one row per terminal trial in trial-number order. Do not send repeated `_trial_records` dictionaries or repeat field names in every row.

Include:

- Trial number and all 14 raw architecture parameters where known.
- Decoded per-block widths and a decoded-network identifier.
- State, feasibility, and failure reason/classification.
- Score, macro-F1, energy, latency, and relevant memory measurements.

Measure the actual serialized request at 300 trials, including decoded widths, fixed context, coverage, pending candidates, and repair feedback, against the approved 120,000-byte-based conservative input guard (including its overhead). The review's roughly 37 KB columnar estimate is illustrative, not an acceptance result. Reduce redundant formatting and verbose metadata while preserving every trial's required evidence; do not silently truncate history. This guard increase was approved with the unchanged smoke spending cap and explicit reservation calculation.

Preserve exact parameter values. Missing measurements remain explicitly missing. Failed and pruned trials remain visible, with infrastructure failures distinguished from architecture failures. Missing failure classifications must appear as `unclassified`, never inferred to be architecture failures; trial 200 is a reported example. Partially populated trial parameters should be reconstructed from existing proposal metadata where possible; unknown values must not be invented.

Show running and queued architectures separately as pending work, not measured evidence. Avoid repeating complete records as separate anchors; any best-trial references should point back to the table.

Disable generated memory for the new configuration and remove memory-specific instructions from this prompt mode. Retain the existing memory implementation for older configurations. If full history exceeds the input guard, report the limit explicitly rather than silently dropping old trials.

## 3. Define decoded identity and compute coverage in Python

Raw parameter dictionaries can differ while decoding to the same network because width scaling, rounding, and caps remove distinctions. Reuse the authoritative `audio_dscnn.py` decoding logic instead of writing a second width formula. A decoded fingerprint must include per-block widths and all other graph-affecting settings, including kernels, strides, depthwise multiplier, normalization, activation, pooling, classifier, and training-only graph settings such as dropout. Equal RAM, flash, or latency alone is not proof of graph equivalence.

Maintain both raw fingerprints for provenance and decoded fingerprints for unique-network accounting and duplicate checks. Preserve infrastructure-retry exceptions and explicitly represent candidates that cannot yet be decoded. Never merge distinct training architectures merely because their deployed inference graphs match; record that deployment equivalence separately when applicable.

Start coverage with:

- `num_blocks × resolved per-block strides`: the exploration-target table, with schedule aliases displayed together.
- Decoded per-block width signatures and their unique-network counts.
- `num_blocks × decoded first-block width`: an interpretable width/depth view.

Use actual parameter names: `num_blocks` is block count; `depth_multiplier` is the depthwise channel multiplier. Raw width/growth counts may be supplementary diagnostics but must not stand in for unique decoded-network coverage.

For the current space (`num_blocks` 2–6 and six named schedules), there are 29 resolved cells: at two blocks, `light` and `late` both resolve to `((1, 1), (2, 2))`. Enumerate and collapse aliases from the decoder rather than hardcoding 29. Include zero-count cells. Count distinct decoded networks per cell, separating successful measurements, deterministic rejections, confirmed infeasibility, infrastructure failures, unclassified failures, and pending work. A raw alias or repeat must not increase unique-network coverage. Do not claim exhaustive zero-count coverage over all possible width signatures without enumerating that space. Count partial architectures only where the required parameters are known.

Coverage indicates how much evidence exists, not whether a region is promising. Put changing coverage tables in the dynamic end of the prompt.

## 4. Add Responses with cache-friendly prompt construction

Add a Responses provider path behind the current provider interface. Reuse the existing transport/retry policy, error handling, durable spending reservations, output limits, validation, and provenance. Keep Chat Completions available for existing configurations.

Use the replay's ordering with a deliberately simple layout:

```text
Fixed instructions, MCU/task context, search space, output contract, table columns
Chronological full-history table
Current coverage, phase allocation, pending architectures, request, repair feedback
```

Do not build persisted block-finalization, late-trial prefix rebuilding, or cache-boundary bookkeeping for this iteration. Rebuild the table deterministically from the saved study snapshot on every request. Chronological ordering alone does not guarantee cache hits.

Reuse the replay's explicit no-breakpoint mode as the initial configuration to avoid paying for whole-prompt cache writes that are never read. This intentionally does not cache growing history. A separately authorized bounded test can compare no-cache behavior with one breakpoint after the fixed context; enable that simple fixed-prefix cache only if observed reuse justifies it. Verify current API support and rates at implementation time. Retain cache read/write counters even when zero. Do not budget for assumed savings.

Defer multi-block history caching unless measured costs or a more expensive model justify it. The current Luna run's reported roughly $0.28 settled cost makes search quality and reliability the higher priorities. Hardware delays may outlast cache retention, so replay's 88% hit rate is not a campaign forecast.

Normalize Responses text, status, and usage into the existing response records. Log the exact secret-free wire payload, raw response, cache reads/writes, and returned service tier. Preserve conservative pre-request reservations and unresolved-request charges. Any cache-aware settled cost calculation must use actual returned usage and configured pricing, not assumed cache hits.

## 5. Improve duplicate feedback and refill missing slots

Extend existing canonical deduplication with decoded-network identity. Build a historical lookup once per proposal round so feedback can include the matching trial number, raw parameters, decoded widths, state, and outcome, and distinguish an exact raw repeat from a decoded alias.

When some candidates pass validation:

1. Retain accepted candidates, their rationales, intent, and original request provenance.
2. Request only missing slots, with the remaining local/exploration allocation.
3. Include useful rejection details and the candidates already retained.
4. Check replacements against historical, queued, and retained candidates.

Accept valid candidates up to each requested quota in response order. Log overflow candidates without enqueueing them; they do not fill the other category's missing slots. Wrong labels, invalid references, and wrong exploration-cell membership reject that candidate only when the envelope is otherwise parseable. Never discard the entire valid subset for a quota mismatch.

Propose a separate `max_refill_attempts: 2`, alongside the existing `max_repair_attempts: 1`. Bound each generation round to one initial content request, at most one malformed-response repair, and at most two refill requests: at most four content requests total. These budgets do not reset during refill. Missing/invalid candidates in an otherwise parseable response consume refill opportunities. Transport retries remain separately bounded and every actual API attempt remains subject to the spending cap.

On exhaustion, return valid retained candidates and log unfilled slots and reasons. Preserve the existing explicit random fallback only when nothing passes; identify it separately from LLM local/exploration proposals. Authentication, transport, and spending failures retain their existing handling.

## 6. Offline validation and review artifacts

Before any paid or hardware test, cover:

- Mocked Responses success, malformed/refused/incomplete output, retry behavior, and spending-cap enforcement.
- Usage normalization, settlement, and durable reservations across failures/restarts.
- Deterministic full-history ordering and fixed-context serialization across variable batches, repairs, and resume; no accidental dynamic data in the fixed prefix.
- New Responses requests in no-cache mode carry no write breakpoints; mocked cache accounting covers both hits and misses.
- All terminal trial states, missing metrics, and canonical parameter reconstruction.
- Decoded aliases collapse correctly, graph-changing settings remain distinct, and coverage counts do not inflate with raw aliases or retries.
- Resolved-stride aliases, coverage targets, exact membership checks, informative-outcome counting, and transition after the two-network-per-cell goal.
- Unclassified failures, enforced parent diversity and logged exceptions, computed parent diffs, and nearest-neighbor diagnostics.
- One repeat-aggregation rule applied to intentional and accidental repeats in both studies.
- Actual serialized 300-trial requests, including realistic failure and repair text, fit the unchanged input guard without dropping trials.
- Phase boundaries, reduced batch sizes, and no-feasible-anchor behavior.
- Duplicate identification, retained-candidate provenance, replacement allocation, and bounded exhaustion.
- No generated-memory calls in the new mode.
- A regression fixture rebuilding today's prompt from a saved audit snapshot with the current settings and checking byte-identical prompt text. Record the snapshot and code version; normalize neither scientific content nor prompt order to force a pass.
- Compatibility of existing configurations and tests.

Render complete early-, middle-, and late-search prompts using saved data for offline review. Record an offline cost estimate for explicit no-cache operation and a conservative cold-write bound for any optional cache test. Offline tests can verify request construction and accounting but cannot establish live cache hits.

Version the new prompt/search policy and decoded-fingerprint schema; include behavior-changing settings in the study identity so an existing study cannot silently adopt the new policy. Prepare a separate next-study configuration and commands after review. Do not deploy changes to the current campaign. Wait until its search and any final-training/hardware stages finish before starting the next study: there is one shared board and port 6001.

## 7. Separate implementation checks from search quality

Read-only verification used the decoder in `src/crest/model_families/audio_dscnn.py` and saved SQLite snapshots. Pairs 94/125, 23/44, and 180/181 match decoded per-block widths, resolved strides, and the remaining architecture settings, including training-only settings. `base_channels` enters the graph through pointwise width computation, not a separate stem. Their recorded macro-F1 values are:

| t2 pair | First macro-F1 | Second macro-F1 |
| --- | ---: | ---: |
| 94 / 125 | 0.754464 | 0.621119 |
| 23 / 44 | 0.761422 | 0.708042 |
| 180 / 181 | 0.654522 | 0.717244 |

The saved TPE baseline contains 12 exact-raw-configuration repeat groups with 28 completed macro-F1 measurements. Pooling within-group squared deviations with denominator `sum(n_group - 1)` gives a macro-F1 standard deviation of **0.04662**, reproducing Claude's approximately 0.047 estimate. Claude reported 29 measurements; reconcile that one-measurement discrepancy before publishing the analysis. Sources: `/private/tmp/crest-cs3-baseline-20261013/optuna.db` and `/private/tmp/cs3-oct15-audit/optuna.db`. Archive the exact inputs and reproducible calculation with the final analysis rather than relying on temporary paths.

This is substantial observed within-configuration variability. It is not a standard error for the difference between optimizers, nor proof that every policy effect is undetectable. Small single-run best-score differences are weak evidence; the estimate does not by itself partition training randomness from every other source of variation.

Report exact-raw and decoded-network repeat dispersion separately. Reproduced from the saved snapshots: baseline decoded pooled macro-F1 SD is 0.048604 (15 groups, 36 observations, 21 degrees of freedom); t2 decoded SD is 0.063985 (3 groups, 6 observations, 3 degrees of freedom). The t2 snapshot has no exact-raw repeat estimate. Preserve both definitions rather than silently replacing the original raw estimate. Report best-network identity/observation count and the number of single-observation networks among the top ten; ties use decoded identity for deterministic ordering. Duration sums with missing observations are incomplete observed lower bounds, with separate completeness flags for completed-only and all-attempt time.

Before launching, save the evaluation definitions and baseline values from the completed current study. The following are implementation checks and behavioral diagnostics, not search-quality success criteria:

- Unique decoded training networks evaluated at matched attempt and feasible-completion budgets.
- Duplicate proposal rate: duplicate rejected entries divided by observed candidate decisions (returned LLM candidates plus candidate-level rejections), with raw repeats and decoded aliases reported separately. Exclude malformed calls and random fallbacks and report them separately. This saved-ledger denominator is not necessarily every raw response entry: interrupted rounds or missing streams make it incomplete. Do not infer proposal rejection rates from the Optuna database alone.
- `num_blocks × resolved strides` coverage and its trajectory, counting distinct networks and separating informative outcomes from successful evaluations.
- Valid batch fill rate, refill calls per accepted architecture, and fallback count.
- Distribution of nearest-prior distances. Report raw-parameter Hamming distance separately from decoded structural distance; freeze the decoded comparison fields and variable-length width handling before launch. Use neither as an acceptance threshold.

Unique-network counts are already close to their ceiling in t2, and target-cell coverage is deliberately encouraged by the new algorithm. Lower duplication or higher coverage therefore cannot establish better search. Retain these checks to show whether the policy operates as intended. Also report total calls, spend, proposal rounds, evidence available per round, accepted batch sizes, API/repair/refill time, and infrastructure failures.

Search-quality outcomes are score-based, at matched attempt and feasible-completion budgets and active evaluation time. For every study, group all equivalent decoded training networks, including intentional repeats, accidental twins, and exact raw repeats. Within comparable training/data settings, average the available valid measured scores for each network before ranking network means; report observation counts and dispersion. Failed attempts with no valid score contribute no fabricated score. Use the same identity and aggregation rules for TPE, t2, and the next study.

Every trajectory point means **history available so far**: aggregate only observations available through that completion. Order saved events by completion timestamps (trial number breaks ties), rather than allowing an earlier-started but later-finished trial to leak future evidence into a prefix. Never use later repeats to revise earlier curve points. A repeat such as trial 125 can lower the estimated mean of trial 94's network when it arrives: the best-network-mean and top-network curves may decrease. Do not apply a cumulative maximum to them. The historical best individual observation remains a separate monotonic curve.

Report trajectories of the mean and median of the top 10 network means, the best network mean, and the best individual observed score (the historical maximum, clearly labeled). Use all available networks when fewer than ten exist and show the count. Unequal repeat counts still create unequal uncertainty, and top-result selection remains biased; aggregation does not turn this into an unbiased deployment-quality comparison. Keep single-observation networks identifiable rather than treating their means as equally certain.

Retain these separate time definitions:

- **Completed-trial time:** sum of COMPLETE trial durations, excluding FAIL/PRUNED and between-trial gaps, for continuity with the requested earlier comparison.
- **All-attempt evaluation time:** includes completed, failed, and pruned attempt durations. Stored Optuna start-to-completion durations are wall-duration proxies, not measurements of actual GPU/hardware activity. Label them accordingly; do not call them measured active time. Flag missing durations.
- **API overhead:** separately report recorded provider request latency, including repair/refill and memory calls where applicable. Missing or unresolved request timings make the total incomplete, not zero. Client processing outside those timings remains unmeasured.
- **Elapsed wall time:** first trial start through each completion, retaining gaps and outages. Label the origin; setup/API work before that first start is excluded. Overlapping trial durations may exceed elapsed time and must not be summed with overlapping API intervals as a reconstructed wall clock.

Show both completed-only and all-attempt axes so a policy with more failures cannot appear efficient merely by omitting their costs. Filling batches reduces feedback frequency and may change API overhead; report it rather than assuming a benefit. Saved ledger and database snapshots must have aligned capture boundaries for a strictly matched overhead comparison; otherwise disclose the mismatch.

The warmup-rerun comparison across studies is now optional. Within-study TPE repeats already provide the main variability estimate. A later read-only warmup comparison may examine cross-study consistency after checking training settings, data splits, and code/firmware differences, but is not required before implementation. No new hardware measurements are included in this plan revision.

A single next run is descriptive. Neither a small best-score gap nor improved process metrics establish superior expected search quality. Any causal claim about history, memory, allocation, refill, deduplication, or API choice requires separate ablations or replicated runs.

## Reuse points

LLM implementation paths, relative to its worktree:

- `src/crest/optimizers/llm/phase_policy.py`: existing progress calculation and phase thresholds.
- `src/crest/optimizers/llm/history.py`: terminal history and measured outcome extraction.
- `src/crest/optimizers/llm/prompt_builder.py`: semantic context, output contract, and prompt construction.
- `src/crest/optimizers/llm/provider.py`: provider interface, transport, and normalized responses.
- `src/crest/optimizers/llm/spending.py`: durable budget accounting.
- `src/crest/optimizers/llm/schemas.py` and `enqueue.py`: validation, canonical deduplication, repair, and proposal provenance.
- `src/crest/optimizers/llm/component.py`: configuration, identity, and memory enablement.
- `src/crest/model_families/audio_dscnn.py`: authoritative pointwise widths and resolved stride schedules for decoded identity and coverage aliases.

Replay reference directory:

`/Users/jzales/Documents/Projects/fall-2026-nesl-crest/.claude/worktrees/ranker-replay/analysis_scripts/ranker_replay`

- `llm_ranker.py`: Responses request construction and explicit cache controls.
- `replay.py`, `full_log_segments()`: stable context followed by chronological history and a dynamic suffix.
- `RESULTS.md`: measured replay cache reuse and scientific limitations.

Reuse the replay's layout and API construction rather than replacing the campaign's stronger durable accounting with the replay client wholesale.

## Scientific interpretation and decisions for discussion

Recommend keeping Luna for the next study so the model does not change alongside the search policy. The proposed changes still form a bundled optimizer experiment, not an isolated memory-versus-full-history ablation.

The replay reported 88% cached input reads for Luna full history, but full history did not improve ranking accuracy there. Neither that cache rate nor the ranking result directly predicts performance in a hardware-paced architecture-generation campaign.

Items to settle before implementation:

- Accept the proposed 2/3 → 3/2 → 4/1 allocation and existing phase thresholds.
- Confirm Python-assigned exploration cells, decoded identity, and intent/reference fields.
- Use two distinct informative networks per resolved cell as the finite compulsory-coverage goal; retain outcome-informed exploration afterward.
- Confirm the explicit smaller-batch allocation and separate refill budget.
- Retain batch size five deliberately and document its changed feedback cadence; enforce parent diversity where possible.
- Freeze the metric definitions and baseline before launch; keep cache-block machinery outside initial scope.
- Keep model, seed, warmup, training, scoring, and hardware settings comparable, documenting any intentional changes.

After offline review, a separately authorized bounded API test should verify valid proposals and requested cache behavior before any new hardware campaign. No-cache operation is acceptable; cache hits are not a prerequisite for this study.
