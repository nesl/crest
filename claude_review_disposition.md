# Claude review disposition (2026-10-01)

The review covers the proposal-contract implementation at `b29cae9` on
`codex/crest-proposal-contract`; its architecture and scientific evaluation
path were accepted. The user needs existing campaigns to extend, superseding
the original blanket rejection of unsigned legacy studies. CREST remains a
contract-based experimentation system: common evaluation semantics preserve
interpretable, comparable measurements as workloads, models, devices,
schedules and proposers change. **Simple is better; code reuse is the name of
the game.**

| Finding | Disposition and concrete change |
| --- | --- |
| Existing unsigned campaigns cannot extend | Fix with explicit default-false `optimizer.adopt_legacy_study` for native Optuna only. The user attests original configuration compatibility. Reject WAITING work and explicit-proposer provenance; validate stored ordered directions and existing feasibility policy/evidence before stamping. Preserve trials and RUNNING accounting. Signed native and LLM campaigns continue to extend budgets normally. |
| Native execution enters proposal descriptor and semantic setup | Fix through component capability declarations. Native opts out of both; LLM requires a descriptor and obeys its semantic-context switch. Existing explicit plugins retain setup by default. Detect the inherited absent family hook, and propagate errors inside declared hooks. |
| Sampler identity stores unstable internal module paths | Fix new signatures to public `optuna.samplers.TPESampler`/`NSGAIISampler`. Accept only exact historical `_tpe.sampler.TPESampler` and `nsgaii._sampler.NSGAIISampler` module paths with unchanged remaining identity fields. Preserve existing stored signatures on compatible resume. |
| LLM ledger names changed | Accept and document. `returned_candidates.jsonl`/`batch_returned_to_runner` describe proposals returned to the runner; study user attributes alone evidence actual queue acceptance. Historical artifacts remain evidence under their original semantics. |
| Provider endpoint rejects login/query/fragment | Accept and document. Keep a clean nonsecret endpoint as signed identity, with credentials in transport credential settings. |
| WAITING duplicate lookup sees empty trial params | No change needed in the shared execution path. Waiting reservations drain before new proposals; sampled `trial.params` remain distinct from intended `proposal_params`. Duplicate and anchor eligibility retain their existing evidence rules. |
| Plain pytest imports a different editable repository | Fix project pytest configuration to select local `src`, and verify without inherited PYTHONPATH. |

Native adoption is a narrow compatibility option, not provenance certification,
a hybrid campaign mechanism, crash recovery, a trial importer or a new journal.
The implementation leaves objective, scoring, pruning, feasibility penalties,
logging, family decoding, final training and replay behavior in the shared path.
Unsigned LLM adoption requires a separate user decision and is outside this fix.

For extension, retain the same database, study name and original experiment
configuration. `training.nas_trials` is the total completed/feasible-completed
target, not the number of additional evaluations; `max_total_trials` is the
total attempt cap, including failed, pruned, infeasible and RUNNING trials.
Increase both as needed, then remove the adoption flag after first successful
adoption. See [configuration instructions](src/config/README.md#optimizer).

Implementation/verification follows the revised appendix in
[optimizer_contract_plan.md](optimizer_contract_plan.md). Evidence belongs in
[optimizer_contract_validation.md](optimizer_contract_validation.md): focused
real-Optuna SQLite resume/rejection tests; native and signed LLM budget extension;
strict sampler alias tests; capability/context tests; plain pytest integration;
and adoption/extension probes on temporary copies of historical snapshots with a
synthetic objective. Source snapshots are read only and their hashes verified;
no campaign execution, provider payment or hardware measurements are performed.
