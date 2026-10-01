# CREST proposal contract implementation plan

Keep Optuna as CREST's shared execution engine and study store. Add a contract for choosing the next search round, then move the current LLM orchestration behind that contract. Preserve `study.optimize(objective)` and the existing objective, pruning, logging, sampler completion hooks, reporting, and replay paths.

This replaces the previous proposal to evaluate outside Optuna and import results afterward. There will be no external-result projector, authoritative candidate journal, manual constraint-system metadata, new recovery framework, or extracted second evaluation path. This is one end-to-end refactor; the steps below are implementation order, not separate releases.

The reviewed source is `codex/llm-generator-mvp`, commit `0f870764b6cf83d0024bc98139276a993d141871`, at `/Users/jzales/.codex/worktrees/llm-token-estimator/fall-2026-nesl-crest`. This plan is saved in the main workspace on `codex/cs1-epoch-front-pilot`. Implement against a deliberate descendant of the LLM branch; do not accidentally apply the refactor to the experiment branch. Resolve any changed baseline before implementation.

## Motivation and rules

CREST is a **contract-based experimentation system** for embedded model search that makes deployment choices explicit and preserves shared evaluation semantics as workloads, models, devices, schedules, and proposal mechanisms change. As AI makes it easier to generate new models, optimizers, and experiment scripts, common contracts help ensure that differences in measured performance remain interpretable and results remain comparable across implementations.

**simple is better**: modularize proposal selection without replacing working execution infrastructure. Add only interfaces and helpers needed by the existing implementations and a test-only third proposer.

**code reuse is the name of the game**: retain one objective, one scoring/feasibility implementation, one trial lifecycle, existing search declarations, existing history/memory algorithms, and existing logging. Extract ownership; do not copy behavior into adapters.

This contract makes proposal mechanisms interchangeable within CREST's Optuna-backed runtime. It does not claim independence from Optuna, universal support for every optimizer algorithm, identical stochastic search trajectories, or exactly-once hardware execution.

## Match the existing component pattern

| Existing pattern | Implementation for proposal mechanisms |
| --- | --- |
| ABCs in `interfaces.py` with required methods and default hooks | `OptimizerABC` with name, validate_config, identity_config, initialize, propose_round |
| Passive transport types in `pipeline_types.py` | Small SearchContext, BudgetSnapshot, TrialRecord, and explicit round payloads |
| Explicit class registration through ComponentRegistry | Add optimizer_registry in `registry.py`; reuse duplicate/unknown-name handling |
| Lazy, idempotent built-in registration | Register optuna and llm_generator without credentials, provider calls, or hardware access |
| Component-local configuration and runtime setup | Shared selection resolves the registry class; LLM owns provider defaults/validation |
| Family-owned search and decoding | Keep sample_hparams and decoding inside the current objective and model family |

Use zero-argument constructors and then initialize(context, config) for all proposal components. They receive neither a Study nor a Trial. The runner keeps its existing study and sampler setup. There is no special Optuna constructor binding, adapter-specific persistence function, or per-optimizer evaluation loop.

## Contract and round semantics

```python
class OptimizerABC(ABC):
    @property
    def name(self) -> str: ...  # default: class name

    def validate_config(self, config) -> None: ...  # default: no-op

    def identity_config(self, config) -> dict: ...
        # deterministic JSON-safe, non-secret proposal identity;
        # default {} only for components with no identity-relevant settings

    def initialize(self, context: SearchContext, config) -> None: ...
        # default: no-op; components override when they need setup

    @abstractmethod
    def propose_round(
        self,
        history: tuple[TrialRecord, ...],
        budget: BudgetSnapshot,
    ) -> NativeRound | ExplicitRound: ...
```

History is supplied as a fresh read-only snapshot on every proposal call, including after restart. There is no separate observe method or bulk-history restoration path. A stateful proposal algorithm can compare trial numbers with its own processed frontier; it must not modify the study or run an evaluation. Existing LLM memory remains component-owned derived evidence, not a new result store.

| Payload | Required contents |
| --- | --- |
| SearchContext | Study name and artifact directory; ordered objectives/directions; full serialized score configuration as objective_summary; precomputed semantic_context plain data; optional raw search-space descriptor |
| BudgetSnapshot | Existing attempted count, completed-target count, target, total cap, and permitted round size computed by current rules; immutable snapshot |
| TrialRecord | Stable Optuna trial number, neutral state name, actual sampled params, values, and the attributes needed for eligibility/failure/evidence; ordinary data only |
| NativeRound | Positive n_trials within the permitted round size; no explicit candidates |
| ExplicitRound | Nonempty ordered tuple of CandidateProposal(params, provenance); trial count is derived from its length |

An explicit round cannot separately specify a conflicting n_trials. Reject zero-length, oversized, malformed, or unknown round variants before enqueue or HIL work. Never interpret an empty candidate list as an implicit request for native sampling. Partial LLM acceptance produces a smaller explicit round; exhausted repair/fallback remains a clear error as today.

The native component returns NativeRound(budget.permitted_round_size). It does not call ask, suggest, enqueue, or tell. The configured TPE/NSGA-II sampler continues selecting parameters inside objective(trial).

The LLM component prepares context/history/anchors/memory, calls its existing provider, validates/repairs proposals, and returns ExplicitRound. It does not enqueue or execute. A test-only third component can return parameter dictionaries in the same form without importing Optuna or changing the runner.

During setup, the runner computes objective_summary with the existing json.dumps(config.nas.score, sort_keys=True, default=str), and computes the stable semantic context once using the existing builder. Preserve its dataset configuration/bundle, model-build context, task, target spec, metric contract, score, feasibility, model family, descriptor, device, training and compile-metrics inputs, including enabled/version/hash semantics. Pass only the resulting plain data in SearchContext; never pass the NAS client, dataset bundle, Study, or other live runtime objects to the proposal component. Share the context builder through the neutral context layer while retaining compatibility imports. Per-round budget, history, anchors and memory remain dynamic. Native families without descriptors must not be forced through descriptor-dependent context construction.

The two round variants are the declared contract alternatives, not hidden optimizer-name checks. Algorithms that require unsupported dynamic spaces, streaming evaluation callbacks, or different execution semantics are outside this change; do not pretend this interface covers them.

## One shared execution path

For a fresh round with no waiting work:

1. Compute the current budget and take a history snapshot.
2. Ask the registered component for a round.
3. Validate the round. For explicit candidates, validate exact active raw keys/types/ranges/choices before enqueuing any member.
4. Enqueue explicit candidates in order with proposal provenance in user_attrs. Native rounds enqueue nothing.
5. Call `study.optimize(self.objective, n_trials=round_size)` once.
6. Let existing objective/log_trial and Optuna completion logic update the study and CSV. Recheck the target/cap at the same round boundary as today.

Both paths use the same Trial and objective. For enqueued fields, suggest_* reads supplied values. For native fields, the sampler chooses them. Family sampling, model construction, quantization, and clock resolution keep their existing execution order. Neither FixedTrial substitution nor add_trial is part of this design.

Retain the existing next_batch calculation: scalar rounds use the needed completions bounded by remaining attempts; multi-objective rounds use max(needed completions, population_size), bounded by remaining attempts. LLM requests cap this allowance by its configured batch size, and the accepted candidate count determines its actual round size. Native sampling remains sequential inside optimize, receiving trial feedback between evaluations. Do not check completion targets midway through an ordinary round.

## Pre-training rejection remains required

Leave the current gates in objective(trial). A candidate failing a returned HIL-status, resource, arena, or configured prune gate must stop before training, log its reason, and allow the search to continue under the existing budget rules. Feasibility rejection skips training when train_if_infeasible is false; preserve the explicit true override.

Preserve scalar report(-inf, step=0) followed by TrialPruned and the current multi-objective COMPLETE penalty behavior. COMPLETE is not synonymous with a trained or feasible model. Optuna runs the existing completion hooks; the refactor does not manually recreate intermediate values, constraints, or terminal states.

Keep whole-run aborts distinct: device-not-found and uncaught build/runtime/training exceptions retain their current exception policy. A returned HIL_MASTER_FATAL code follows the existing candidate prune/penalty path. No new optimizer.abort API or failure taxonomy is needed.

Required combined regression: first candidate fails a pre-fit gate, training-call count remains zero, rejection is logged, the next viable candidate is evaluated/trained, and each trial consumes its budget once. Exercise scalar and multi-objective paths.

## Search declarations and configuration

Move SearchParam, SearchSpaceDescriptor, and reusable candidate legality checks into a neutral shared module such as `crest/search_space.py`. Retain the old import path as a compatibility re-export and keep SearchParam.suggest while public sample_hparams callers use it. Do not move sampling out of objective or rewrite family decoding.

The runner composes the existing run-activated quantization and CPU-clock fields exactly as today. New explicit proposers must provide all active raw fields; otherwise Optuna could silently sample a missing field and misattribute that choice to the proposer. Validation must finish before any member is enqueued. LLM repair/duplicate/fallback policy stays in the LLM component; shared legality validation does not prohibit deliberate repeated experiments by another proposer.

The native component must not require trial_search_space. Keep that model-family hook optional, so existing external families and their define-by-run sampling continue working without a legacy execution fork. Build a descriptor only when available; explicit rounds and LLM initialization require one and fail clearly before expensive work if it is unavailable. An invalid declared descriptor is not treated as an absent capability.

Preserve optimizer.type, its optuna default, and the existing optimizer.llm fields. Replace VALID_OPTIMIZER_TYPES with registry lookup after optimizer built-in registration. Keep structural normalization and existing normalized defaults available to load_config callers; delegate LLM-local defaults and validation to one LLM-owned helper reused by the compatibility normalizer and component. Resolve registration/import order explicitly so validation does not require dataset imports or introduce circular dependencies. Registration never reads provider credentials or creates clients; initialization may create the selected client.

Every component receives the normalized optimizer block with its custom fields preserved. Only llm_generator invokes the LLM compatibility normalizer; an unknown-to-core plugin is not treated as an LLM. Document and test custom registration before load_config(), including a third proposer with its own setting and no llm block.

## Study optimizer identity

Follow _validate_or_store_feasibility_signature: persist a versioned optimizer signature in study.user_attrs and validate it immediately after opening the study, before draining waiting trials, proposing, or evaluating. The signature contains the registered optimizer name and normalized identity-relevant settings. Components expose those settings through identity_config(config), so the runner does not inspect LLM-specific keys. Configurable third-party components must override the default; test that their identity setting is honored. Include the effective native sampler class/options supplied by existing runner setup as well.

For LLM runs include provider/model, prompt/schema/semantic-context versions, generation settings, batch size, repair/fallback policy, history/anchor/memory settings and relevant seeds. Use explicit non-secret fields; never persist credentials or hash an entire credentials-bearing configuration. Normalize defaults so equivalent configurations compare equal. Exclude output paths, credentials and completion/attempt budgets so extending an otherwise identical campaign remains possible.

An empty unsigned study receives the signature. A matching signed study resumes. A mismatch fails with an actionable message before any queued trial runs. A nonempty unsigned study rejects by default. Explicit `optimizer.adopt_legacy_study: true` permits adoption only of native Optuna history with no WAITING trials or explicit-proposer provenance, after existing feasibility-policy validation succeeds. The user attests that the active configuration matches the original experiment; CREST does not automatically certify historical provenance. Adoption stamps the optimizer signature and a small study attribute without changing trial evidence. Signed studies always retain full identity validation, including when the adoption flag is present. The flag is excluded from identity so it can be removed after adoption and budgets can be extended. Unsigned LLM and plugin adoption remain unsupported. Intentional hybrid optimizer campaigns are likewise outside scope. This adds a configuration compatibility guard, not new crash recovery behavior.

## History and memory without changing evidence

The current `_trial_records` is a compact prompt projection, not a complete replacement for FrozenTrial. Anchor selection separately uses feasibility, pruned flags, error codes, values, directions and trial-number tie breaks. Introduce one study-to-TrialRecord adapter that preserves all of those fields, then reuse the existing compact projection and selection algorithms over records.

Keep trial numbers unchanged. Preserve the provider-visible compact record keys/values, recent-window ordering, scalar anchors, Pareto/knee selection, terminal-history filtering, and memory coverage/digest semantics. Extra eligibility fields belong in the neutral record, not automatically in provider prompts. Preserve pending-versus-summarized evidence and summary call timing. Saved LLM memory should see the same compact observations for unchanged study history.

Do not reinterpret WAITING/RUNNING trials as completed evidence. Keep state and actual sampled params distinct from a proposal's intended params. No new ordinal, candidate journal, observer replay, or duplicate history authority is introduced. Unit tests should compare old/new compact payloads and memory hashes, not merely counts.

Keep duplicate detection and anchor eligibility separate. Current duplicate detection includes descriptor-complete COMPLETE trials, including penalized multi-objective failures, and excludes PRUNED/FAIL trials. Anchors additionally exclude penalties and failures. Preserve these distinct sets using actual sampled params; test successful COMPLETE, penalized COMPLETE, PRUNED and FAIL records together.

## Provenance, waiting trials, and restart

Store proposal_source, optimizer name, request/batch ID, batch index, and intended proposal_params under a namespaced user_attr. LLM versus random fallback provenance comes from the actual generator result. Keep metadata out of trial.params: `_family_trial_params` strips only known runtime keys, so arbitrary metadata there can leak into decode_trial_hparams. Both families' replay/finalization must remain unchanged.

The shared runner is the sole enqueue owner and writes provenance only into Optuna user_attrs, uniformly for every explicit proposer. It never imports or writes the LLM ledger. The LLM ledger keeps request/response/validation details and records candidates as returned_to_runner, not queued. Rename or explicitly version the old accepted_candidates/batch_enqueued artifact semantics and update their documentation/tests; do not silently retain a misleading event name. Existing historical artifacts remain readable as historical evidence.

Actual queue acceptance is evidenced by the study trial and its provenance attributes. Include those attributes in TrialRecord snapshots so the LLM can reconcile its returned proposals if needed, without a new callback or runner-owned ledger helper. A request/source reference is metadata, not a second trial identity. No commit/observe lifecycle is added to the ABC. Retain the queue's partial-enqueue restart test, but remove the cross-module ledger-write test.

Handle waiting work before asking for a new round. Otherwise optimize may consume an older queued candidate while the new batch is incorrectly credited. This is a small, explicit queue-accounting correction, not preservation of the current restart bug:

- Check the existing completed target first. If already satisfied, leave waiting work untouched and stop as today.
- When WAITING trials exist and the target is unmet, execute already-reserved waiting work before making new proposals. Select the waiting prefix in Optuna trial-number order, not ledger order. Set n_trials to exactly the selected waiting group's length, never allowing it to fall through into new sampler-generated trials.
- For newly tagged queues, use the oldest contiguous batch's waiting remainder as the resumption round. For legacy untagged waiting entries, consume the existing waiting prefix deterministically; do not invent proposal provenance.
- Waiting trials already contribute to len(study.trials). Consuming them does not allocate another attempt and must remain possible when the cap has been reached by reservations. Apply the cap to new proposals after the waiting group finishes. Preserve the existing max(configured cap, existing study size) behavior.
- Do not read waiting params as if they were already sampled: Optuna WAITING trials can have empty trial.params. Own proposal metadata is available for new tagged queues; legacy queues can simply be consumed without reconstructing their intended dictionaries.
- Do not add automatic stale-RUNNING repair, fail-closed campaign halts, writer locks, or new Ctrl-C behavior. Keep current orphan-RUNNING accounting and exception propagation. Concurrent study writers remain outside the existing serial runner assumption.

If enqueueing a batch stops partway, previously queued members stay visible and are consumed before new proposals on restart. No all-or-nothing enqueue or exactly-once physical measurement guarantee is claimed. These rules need focused real-Optuna tests because queued trial counts and executed trial counts differ.

## Files and implementation order

| Area | Work |
| --- | --- |
| interfaces.py, pipeline_types.py, registry.py | ABC, small round/context/history types, optimizer registry |
| builtin_components.py and NAS construction | Lazy optimizer registration and uniform zero-arg construction/initialization |
| crest/search_space.py and existing LLM import path | Relocate/re-export shared declarations and legality checks |
| LLM config/helper modules and model.py | Delegate component-local validation; remove hardcoded type allowlist while retaining config compatibility |
| optimizers/optuna.py | Native-round component; no study handle or replacement sampler |
| optimizers/llm component and enqueue helper | Move current inline orchestration; separate generation from queue mutation; reuse providers/prompts/ledger |
| Shared history adapter plus LLM history/memory | Neutral snapshot conversion and unchanged compact evidence/selection |
| nas_model_client.py run_nas and smoke_test | Shared round dispatch, enqueue provenance and bounded pending-queue handling |
| Tests and source/config guides | Compatibility, third-party registration example, explicit scope and lifecycle docs |

Complete in this order: establish characterization tests; introduce types/registry/config wiring; adapt history and extract LLM proposal preparation; route both built-ins through the shared round executor; finish pending-queue/provenance handling and all consumer checks. Both built-ins, both shipped families, smoke testing and downstream consumers must work before declaring the refactor complete.

Keep objective, log_trial, sampler construction, scoring, constraint callback, HIL implementation, model-family decoding, final training and replay bodies unchanged unless a narrow call-site adaptation is demonstrated necessary. No manual writes to Optuna's constraints system_attrs are permitted by this design.

## Acceptance tests

1. Registry contract: unknown/duplicate names, lazy registration, config defaults/errors, no provider calls for native mode, and a test-only third explicit proposer requiring no runner edits or Optuna import.
2. Round contract: native count bounds; nonempty explicit candidates; size derived from candidates; all-batch legality validation before enqueue; no silent sampling of missing explicit fields; invalid rounds do not reach HIL.
3. Native parity: same sampler options, objective call/sampling order, no enqueue, unchanged round stopping/counting, scalar and NSGA-II near-target/population cases, optional family descriptor remains optional.
4. Explicit execution: actual enqueued values reach the same objective; partial acceptance, repair, random fallback and provenance remain correct; every supplied active field is consumed on successful evaluations; early failures retain the distinction between intended and sampled parameters.
5. Candidate rejection: pre-fit resource/HIL/prune/feasibility gates skip training and permit the next trial; train_if_infeasible override; device-not-found/uncaught exception run aborts; scalar -inf report and multi-objective penalty semantics unchanged.
6. History/memory parity: exact compact records, eligibility filtering, recent windows, anchor ties, pending evidence, summary schedule/digests and restart behavior. Do not double-count duplicate observations.
7. Queue restart: existing waiting trials at the attempt cap; queued prefix smaller than requested n_trials; partially enqueued batches; empty waiting params; no new proposal while waiting work is being consumed; legacy RUNNING behavior remains unchanged.
8. Consumers: log CSV schema/values, constrained best_trial/best_trials, best-parameter family decoding, finalization, Pareto replay, token estimator and smoke_test use the unchanged real execution path. Proposal metadata must never enter family params.
9. Study identity: empty-study signature creation, equivalent normalized defaults, matching resume, native/LLM mismatch, changed model/prompt or third-plugin setting, nonempty unsigned legacy rejection, and budget-only extension. A mismatch with WAITING work must execute zero trials. Secrets never appear in the signature.
10. Context parity and module ownership: exact objective_summary and semantic payload/version/hash match current prompt inputs; stable context built once; no live client/dataset passed to a proposer; runner has no LLM-ledger dependency; returned proposals do not claim queue success.

Use real Optuna with fake provider/model/HIL fixtures for these gates. A fixed-candidate before/after comparison must produce equivalent objective inputs, outputs, attributes and CSV meaning. No paid API call or new broad hardware campaign is needed to test this orchestration-only refactor. If implementation changes actual evaluation/hardware behavior, stop and reassess scope rather than using this plan to justify it silently. Existing physical measurement variability and historical results remain subject to their original limits.

## Evidence and review disposition

The prior plan's three adversarial reviews found real semantic risks, but did not sufficiently challenge the decision to bypass Optuna execution. Claude's review identified that unnecessary scope. This revision removes its cause and retains useful checks: early pruning, round timing, history eligibility, existing extension compatibility and provenance. The earlier red-team review is not presented as approval of this different architecture.

The same three reviewers independently reviewed this revised architecture against the motivation above. Their contract, lifecycle and evaluation-semantics reviews identified plugin normalization/registration timing, successful-enqueue ledger ownership, exact waiting-prefix execution, and the distinction between duplicate and anchor eligibility. A subsequent Claude review confirmed the architecture and prompted three further corrections: persist optimizer identity, keep queue evidence solely in study attributes rather than having the runner write the LLM ledger, and specify precomputed prompt context completely. The runner-owned ledger helper proposed during red-team review has been removed. This is design review, not implementation verification.

Earlier planning evidence on Optuna 4.6.0/Pydantic 2.13.4: 36 focused LLM tests plus 19 subtests, 37 objective tests, and 2 search-budget tests passed against the baseline worktree. The 12 isolated API probes remain in `outputs/optimizer_contract_review/`; their add_trial/recovery experiments document a rejected alternative, not required implementation work. Their constraints finding reinforces keeping sampler completion hooks rather than reproducing private metadata.

Fresh revised-design evidence: `outputs/optimizer_contract_review/enqueue_round_probe.py` ran three passing synthetic tests on Optuna 4.6.0. They verify that WAITING params are empty before evaluation and queued execution does not allocate new trials; executing exactly a queued prefix leaves the next batch WAITING; and scalar pre-fit pruning skips the simulated training call while the following candidate runs and both completion callbacks fire. Output is saved alongside the script in `enqueue_round_probe_output.txt`. These probe real Optuna primitives, not the unimplemented CREST runner or hardware behavior.

Re-run impacted baseline tests after implementation and add the acceptance cases above. Passing tests of today's code or small API probes are not proof that the proposed refactor has already passed. Source code has not been refactored as part of writing this plan.


## Implementation clarifications (2026-10-01)

The implementation preserves native smoke testing's documented meaning: `trials` is exactly the number of additional evaluation attempts, including already reserved waiting trials consumed by the invocation. All registered components use the same round dispatcher and real objective. Partial explicit rounds continue until the requested attempt count is exhausted. The old LLM-only smoke delegation to completion-target NAS has been normalized to this shared attempt contract; ordinary `run_nas` still uses feasible completion targets and the existing round-boundary stopping rules. The smoke target supplied in prompt budgets is fixed at the initial eligible completed count plus the requested attempts, and configuration overrides are restored.

Sampler construction options are extracted into `_sampler_setup`, reused by `_build_sampler` and study identity. This narrow extraction keeps effective TPE (`n_startup_trials=15`, `multivariate=True`) and NSGA-II (`population_size` configured, `seed=42`) options and the persisted-feasibility constraints callback unchanged while avoiding a second declaration of identity settings. Objective, scoring, logging, constraint callback, penalty handling, final training, family decoding and replay bodies remain unchanged.

LLM `base_url` must be a clean endpoint without userinfo, query or fragment. The existing provider appends `/chat/completions` directly, so these shapes are ambiguous and could hide routing identity or credentials. The normalized nonsecret endpoint is signed; credentials belong in `api_key_env` or `extra_headers`, which remain excluded from identity. Users must keep routing semantics consistent across resumes, including when using transport headers.

Runner-owned provenance lives only under `crest_proposal` user attributes. Its generated `round_id` uniformly identifies enqueue groups even when a plugin provides no request ID or reuses one. Source/request/batch metadata from components is retained alongside the enforced optimizer name, batch index and intended proposal parameters. The LLM ledger now records `returned_candidates.jsonl` and `batch_returned_to_runner`; it does not certify queue acceptance. Older artifacts remain historical evidence.

Implementation evidence is recorded separately in `optimizer_contract_validation.md`; the design's older test/probe evidence above is not implementation verification.


## Claude review disposition and revised implementation plan (2026-10-01)

The user needs to extend existing campaigns, superseding the earlier blanket rejection of unsigned legacy studies. Implement native Optuna adoption as an explicit, default-false boolean with no WAITING work and no explicit-proposer provenance. Validate stored ordered study directions, existing feasibility signatures and COMPLETE feasibility evidence before stamping adoption. Preserve trial numbers, states, actual params, values, attributes and orphan-RUNNING budget accounting; add no recovery framework or trial rewrite. This is user-attested configuration compatibility, not automatic provenance verification. Already signed native and LLM campaigns may extend completion/attempt budgets with unchanged optimizer identity. The user confirmed there are no pre-contract LLM campaigns to migrate. Unsigned LLM adoption remains unsupported, with no migration path needed. Reject any trial containing the `fixed_params` system-attribute key (including an empty dictionary) and an existing per-study `llm_optimizer` artifact path before stamping native unsigned adoption. These conservative checks cover older explicit proposals with no user attributes and do not affect matching signed resumes.

Persist sampler identity through stable public names `optuna.samplers.TPESampler` and `optuna.samplers.NSGAIISampler`. Compare existing version-1 signatures after mapping only the two exact private paths previously emitted by the reviewed implementation, retaining strict optimizer, proposal-settings and sampler-options comparison. New signatures use public paths; do not rewrite existing signatures merely to resume them.

Proposal setup advertises minimal capabilities on OptimizerABC: `requires_search_space` defaults true and `requires_semantic_context(config)` defaults true. Native Optuna opts out of both, bypassing descriptor/dependency/semantic construction. LLM keeps the descriptor requirement and derives semantic requirements from normalized `llm.semantic_context`. The runner detects an absent `trial_search_space` by exact inherited ModelFamilyABC method identity; explicit components fail clearly before initialization/provider/HIL, while internal errors from an overridden descriptor hook propagate. Semantic-disabled LLM skips the builder and retains its existing disabled prompt shape in initialization. Defaults preserve setup for registered third-party explicit proposers; no live runtime objects enter SearchContext.

Retain the changed LLM ledger names and endpoint validation with explicit documentation. Leave WAITING duplicate behavior alone: the runner drains waiting reservations before requesting proposals, and duplicate/anchor eligibility remains unchanged. Validate the fixes with real SQLite legacy extension/repeated resume, signed native/LLM extension, zero-evaluation adoption/mismatch rejection, sampler alias strictness and context capability tests using the existing Python environment. Make plain pytest select this worktree through project configuration and verify it without PYTHONPATH. Read historical database snapshots and exercise adoption/extension only on temporary copies with a synthetic objective, preserving hashes of all historical trials and the untouched source files. No real campaign execution, paid providers or hardware are used.
