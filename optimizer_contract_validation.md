# Proposal contract implementation validation

Verified against reviewed baseline `0f870764b6cf83d0024bc98139276a993d141871`, on implementation branch `codex/crest-proposal-contract`, using `/opt/anaconda3/envs/nesl-tinyodomex/bin/python` (Optuna 4.6.0, Pydantic 2.13.4). No paid provider API or physical hardware was used.

## Executed checks

The final complete default suite plus token-estimator tests passed **737 tests and 203 subtests**, with **1 skip**, in 28.90 seconds. The skip is the existing STM32 template ownership check because the STM32CubeN6 checkout is absent under `tools/stm32`. The 88 warnings are existing TensorFlow/Optuna experimental or deprecation warnings and repeated Optuna warnings from the new real-study cases. Hardware integration suites remain opt-in through the existing `test/conftest.py` rule; the three unrelated analysis-script test modules also remain opt-in. The requested token estimator is included explicitly.

Command from the implementation worktree:

```sh
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/crest-proposal-mpl PYTHONPATH=src:analysis_scripts/llm_token_cost /opt/anaconda3/envs/nesl-tinyodomex/bin/python -m pytest test analysis_scripts/llm_token_cost/test_estimate.py -q -p no:cacheprovider
```

Saved output: `outputs/optimizer_contract_validation/complete_tests.log`.

Baseline characterization was run before editing: 24 LLM enqueue/history/anchor/memory tests passed, then 55 objective, NAS budget/orchestration and smoke tests passed. Two invocation-selection errors (a nonexistent test class and token-estimator import without its directory on PYTHONPATH) were corrected; they were not product failures. A clean rerun in the original baseline worktree, with HEAD verified at the reviewed commit, passed **150 tests and 50 subtests**:

```sh
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/crest-proposal-mpl PYTHONPATH=src:analysis_scripts/llm_token_cost /opt/anaconda3/envs/nesl-tinyodomex/bin/python -m pytest test/test_llm_*.py test/test_nas_model_client.py analysis_scripts/llm_token_cost/test_estimate.py -q -p no:cacheprovider
```

Saved output: `outputs/optimizer_contract_validation/baseline_characterization.log`. The baseline command is narrower than the complete implementation suite; these counts are not a before/after count comparison.

## Acceptance coverage

| Plan section | Executed evidence |
| --- | --- |
| 1. Registry contract | `test_optimizer_contract.py`: duplicate/unknown names, idempotent and dependency-safe lazy registration, defaults/config errors, plugin registration before `load_config`, custom-field preservation, mixed-case names, built-in case aliases and exact-key precedence. `test/proposal_plugin.py` implements a third explicit proposer without importing Optuna; `test_optimizer_runner.py` executes it through the runner unchanged. |
| 2. Round contract | Parametrized invalid-round runner tests reject zero/bool/oversized native counts, empty/unknown variants, missing keys, out-of-range values and non-JSON metadata before any enqueue or evaluation. Shared candidate checks preserve all raw fields and permit deliberate repeats. |
| 3. Native parity | Existing retry/cap/constraints-sampler tests retained. Real native tests verify scalar near-target size and NSGA-II population expansion capped by remaining attempts, no enqueue, optional descriptor, invalid descriptor propagation and orphan RUNNING accounting. Sampler settings now have one source shared with identity. |
| 4. Explicit execution | Real third-proposer studies verify exact sampled values, partial rounds, ordered namespaced provenance and immutable snapshots. Generation tests retain repair, partial acceptance and random fallback. Real objective consumes width, quantization and CPU-clock fields; uncaught build failure keeps intended fields separate from sampled params. |
| 5. Candidate rejection | Real Optuna scalar and multi-objective combined tests: first candidate returns HIL_FATAL before fit; its reason/penalty is logged in real CSV; next candidate trains once; exactly two reserved trials are consumed. Scalar intermediate `-inf`/PRUNED and multi-objective COMPLETE penalty semantics verified. Existing objective tests retain resource, arena, prune, feasibility, train-if-infeasible override, device-not-found and uncaught exception policy coverage. |
| 6. History/memory parity | `test_llm_proposal_contract.py` compares exact compact study/snapshot records including FAIL evidence, successful and penalized COMPLETE duplicate sets distinct from anchor eligibility, scalar/Pareto ties, immutable metadata, memory hashes/coverage/pending observations, summary schedule and restart. Existing history/anchor/memory tests retained. |
| 7. Queue restart | Real SQLite-backed tests cover WAITING at reservation cap, empty waiting params, exact oldest tagged group, deterministic legacy prefix, no proposal before draining, partially interrupted enqueue, target-already-satisfied queue preservation and unchanged orphan RUNNING accounting. |
| 8. Consumers | Fixed candidates through prior direct-enqueued `study.optimize(objective)` and new shared dispatcher produce equal HIL inputs, values, sampled params, evaluation user attrs, intermediate/system attrs and full CSV text after fixed timestamps. Odom LLM smoke and Audio third-proposer smoke use real families and the same real objective without HIL. Existing best-trial/finalization/Pareto replay tests and token estimator tests all pass. Provenance is absent from family params and CSV hyperparameter fields. |
| 9. Study identity | Empty signature creation, matching resume, budget-only extension, unsigned nonempty rejection, native/third setting changes, and actual LLM model/prompt/native mismatch with WAITING trials all verified. Mismatches run zero trials and do not initialize providers. Component tests verify normalized-default equivalence and secret exclusion. |
| 10. Context and ownership | Runner context test verifies exact sorted score JSON, semantic builder called once and deeply frozen plain context. Relocated semantic builder functions are AST-identical to baseline, with exact version/hash and disabled representation tests. Runner imports no LLM provider, history, memory or ledger and only it enqueues; ledger tests assert returned-to-runner events and artifacts. |

## Preservation checks

`outputs/optimizer_contract_validation/verify_unchanged_bodies.py` compares the baseline Git object with implementation ASTs. It passed for objective, constraint callback, direction penalties, family-param filtering, final training, checkpoint evaluation, `log_trial`, trial-outcome construction and score/prune/feasibility evaluation. It also confirmed byte-identical Pareto replay modules, both shipped model-family modules and token-estimator implementation, plus all 11 relocated semantic-context function bodies. Output is saved in `unchanged_bodies.log`. These preservation checks supplement real execution tests; they do not replace them.

`git diff --check` passed. Changes are committed locally; nothing was pushed, merged or published.

## Implementation review and correction

The independent contract reviewer ran 45 tests and 7 subtests and found one P2
issue: the registry preserves custom key case, but config normalization and
runner selection lowercased all names. The correction adds one shared resolver
in `crest/component_selection.py`. It strips whitespace, prefers an exact
registered key, and falls back to lowercase only for the historical built-in
names `optuna` and `llm_generator`. This follows the existing case-sensitive
registry behavior, including deliberate exact registrations such as `OPTUNA`.
The resolved registered name is retained in config, campaign identity and
proposal provenance.

Post-correction targeted `test_optimizer_contract.py` and
`test_optimizer_runner.py` checks passed **45 tests and 9 subtests** in 6.16
seconds. Regressions register `MixedCasePlugin`, `OPTUNA` and `LLM_GENERATOR`
before `load_config()`, preserve their custom fields without LLM defaults, and
execute the third proposer through real SQLite-backed runner studies. The
complete suite and preservation checks above were rerun after this correction.
The independent contract reviewer closed P2 after rerunning the original
reproduction, built-in alias and exact-registration precedence probes, and
45 contract/runner tests plus 9 subtests in 6.02 seconds. The SQLite execution,
custom settings, campaign identity and proposal provenance checks all passed.
No actionable contract findings remain.

The independent lifecycle reviewer reported no defects, with 45 tests and 7
subtests passing plus three additional real SQLite probes: mixed COMPLETE and
orphan RUNNING history followed by tagged WAITING groups at the reservation cap;
smoke execution that splits an existing group and drains the remainder before a
new proposal; and KeyboardInterrupt during partial enqueue followed by resume
at the cap. The reviewer probe is preserved as
`outputs/optimizer_contract_validation/lifecycle_review_probe.py`.

The independent scientific-semantics reviewer reported no actionable
regressions. Its independently executed checks passed 57 tests and 14 subtests
for history and the runner; 56 tests and 5 subtests for objective, finalization
and smoke behavior; 34 tests and 5 subtests for context, replay and token
estimation; and, after the selection correction, 45 tests and 9 subtests for
the proposal contract and runner. These overlapping selections are independent
review evidence, not additional tests to add to the complete-suite count. The
reviewer also passed the AST preservation verifier and `git diff --check`.

Six additional real Optuna cases covered scalar and multi-objective fatal
rejection, feasibility rejection and the `train_if_infeasible` override. They
also verified sampler `after_trial` hooks for both candidates, persisted
positive and negative constraints, and selection of the viable successor by
constrained `best_trials`. All passed. Initial probe-fixture mistakes were
corrected and were not product failures. All three independent reviews are complete, including focused closure of the
contract correction, with no remaining actionable findings.

## Deliberate scope and remaining limits

Smoke now follows the same exact-attempt behavior for all components; the old LLM-only completion-target delegation is normalized as documented in the plan appendix. Clean provider endpoints are required to keep routing identity explicit. Study signing rejects nonempty unsigned legacy studies rather than claiming historical provenance. Waiting reservations are drained deterministically, but partial enqueue remains visible and no exactly-once hardware, concurrent-writer or stale-RUNNING recovery guarantee is added. Physical measurement variability and actual paid-provider integration are outside this orchestration refactor's validation.
