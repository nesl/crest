# Proposal contract implementation validation

Verified against reviewed baseline `0f870764b6cf83d0024bc98139276a993d141871`, on implementation branch `codex/crest-proposal-contract`, using `/opt/anaconda3/envs/nesl-tinyodomex/bin/python` (Optuna 4.6.0, Pydantic 2.13.4). No paid provider API or physical hardware was used.


The latest Claude-review follow-up passed **780 tests and 203 subtests**, with
**1 expected skip**, using plain pytest without `PYTHONPATH`. All three
independent follow-up reviewers reported no remaining actionable findings.
Detailed original-refactor and follow-up evidence is retained below.

## Executed checks

The original refactor's final complete default suite plus token-estimator tests passed **737 tests and 203 subtests**, with **1 skip**, in 28.90 seconds. The skip is the existing STM32 template ownership check because the STM32CubeN6 checkout is absent under `tools/stm32`. The 88 warnings are existing TensorFlow/Optuna experimental or deprecation warnings and repeated Optuna warnings from the new real-study cases. Hardware integration suites remain opt-in through the existing `test/conftest.py` rule; the three unrelated analysis-script test modules also remain opt-in. The requested token estimator is included explicitly.

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

Smoke now follows the same exact-attempt behavior for all components; the old LLM-only completion-target delegation is normalized as documented in the plan appendix. Clean provider endpoints are required to keep routing identity explicit. Unsigned studies reject by default; the follow-up permits explicit native-only legacy adoption under the checks described below. Historical provenance remains the user's attestation. Waiting reservations are drained deterministically, but partial enqueue remains visible and no exactly-once hardware, concurrent-writer or stale-RUNNING recovery guarantee is added. Physical measurement variability and actual paid-provider integration are outside this orchestration refactor's validation.


## Claude review follow-up validation

The follow-up retains the shared objective and proposal lifecycle while adding
explicit native-only legacy adoption, public sampler names with compatibility
for the two exact earlier private paths, and context capability gates. Native
Optuna setup now skips descriptor and semantic-context construction entirely.
Disabled LLM semantic context skips the builder and retains the exact disabled
version/hash representation. Genuine errors inside an overridden descriptor
hook propagate rather than being mistaken for an absent hook.

After the final source/test handoffs, the complete default suite plus token
estimator passed **780 tests and 203 subtests**, with **1 skip**, without
`PYTHONPATH`. The existing STM32-template skip and opt-in collection rules are
unchanged. No paid provider or physical hardware ran.

```sh
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/crest-proposal-mpl /opt/anaconda3/envs/nesl-tinyodomex/bin/python -m pytest test analysis_scripts/llm_token_cost/test_estimate.py -q -p no:cacheprovider
```

Saved output: `outputs/optimizer_contract_validation/followup_complete_tests.log`.
The scientific AST/file preservation checker passed again; output is in
`followup_unchanged_bodies.log`. `git diff --check` passed. An earlier full run
collected a resume test before its fixture correction; the only failure was a
live `Study.trials` snapshot assertion, corrected before this final run.

`pyproject.toml` now sets pytest's import paths to `src` and
`analysis_scripts/llm_token_cost`, preserving all existing collection rules.
Before that change, collection without `PYTHONPATH` imported `crest` from the
unrelated editable installation at
`/Users/jzales/Documents/Projects/nesl-tinyodomex/src/crest/builtin_components.py`
and failed. `followup_import_failure_before.log` preserves that reproduction.
A separate pytest subprocess asserts the actual imported `crest.__file__` and
`estimate.__file__` both belong to this checkout, in
`followup_import_provenance.log`; its runnable check is
`pytest_import_provenance.py`. The corrected plain command also passed the
contract and estimator selection: **20 tests and 14 subtests**.

The new `test_optimizer_resume.py` cases verify SQLite-backed legacy adoption,
repeated extension with every old trial unchanged, default rejection, WAITING
and explicit-proposer rejection, feasibility-policy compatibility, ordered
objective-direction compatibility, retained orphan RUNNING accounting, signed
native and LLM budget-only extension, exact old sampler-name aliases with
strict options, public names surviving internal module relocation, and strict
boolean configuration. Adoption does not bypass signed identity mismatches.
The new `test_optimizer_context.py` cases verify native descriptor/semantic
skips, missing-hook errors for explicit proposers, propagation of internal
`NotImplementedError`, disabled LLM semantics, exact enabled context and hashes,
and third-proposer context support.

### Historical campaign copies

`legacy_copy_extension_probe.py` opens each source database using SQLite
`mode=ro`, backs it up into a temporary directory, and mutates only that copy.
This is a synthetic adoption/extension probe; it does not extend an actual
hardware campaign or report new measured results. The original saved score
and sampler configuration are retained: two minimized objectives
`rmse_total`/`energy_mj_per_inference`, NSGA-II population 50, seed 42. Each copy
rejects unsigned resume by default, adopts with explicit native opt-in, adds
one synthetic trial under a one-attempt budget extension, then resumes its
signed study with the opt-in removed and adds one more synthetic trial.
Native descriptor and semantic builders are patched to fail during adoption,
confirming neither is used. The probe passed again after the final ordered
objective-direction guard.

| Historical study | Original trials | Original trial-evidence SHA-256, identical before/after |
| --- | --- | --- |
| `OxIOD_STM32_B2B_case1_5_t1` | 250 COMPLETE | `5f1f4a658c86b0b3ac913514bd5923d2c0d93b30235623dd09204abdc89bd036` |
| `OxIOD_PORTENTA_M7_B2B_case1_3_t1` | 147 COMPLETE, 1 FAIL | `1a8691cacd170f20bcd1143b2f060bb9fab1897401eb85fbf25f5e0c19166329` |

All original trial parameters, values, states, attributes, intermediate values,
distributions and timestamps have identical hashes after both copied-study
extensions. The STM32 source database SHA-256 remained
`786789fb9360521ed7bd8ce42d5e160e6f5d7f2dacb5da14d40fe3c0947841c0`;
the Portenta source remained
`686ef478628a2e09a1e07f3281c4d13b43a74f34a714ae3ee4303f61ce8ff9c3`.
The exact copied-study results and source hashes are saved in
`legacy_copy_extension_probe.log`. The source snapshots are under the main
workspace's `outputs/search_budget_2026-09-24/{stm32_cs1,portenta_m7}/source/`.

The adoption flag is an explicit assertion that the active experiment matches
the original. The tests establish guarded append/resume behavior and preserved
stored evidence; they cannot automatically establish historical provenance or
physical measurement equivalence.


### Independent follow-up reviews

All three follow-up reviews completed with no actionable findings. Their
focused selections overlap the complete suite and each other; the counts
below are independent review evidence and must not be added to the **780**
complete-suite total.

The contract reviewer passed **54 tests and 9 subtests** across resume,
context and contract coverage, then **39 runner/proposal tests**. It verified
local module import provenance without `PYTHONPATH` and found no remaining
contract or plugin-compatibility issues.

The lifecycle reviewer passed **77 checked-in tests**, plus **4 separate
real SQLite cases** preserved in
`outputs/optimizer_contract_validation/followup_lifecycle_review_probe.py`.
The extra cases verify queued work at the reservation cap under an old signed
TPE identity with the adoption flag both false and true, preserving the old
signature and executing exactly the oldest group; a nonempty old signed
NSGA-II study extended once with prior FrozenTrials unchanged; and rejection
of mismatched ordered directions before feasibility/adoption attributes or
any evaluation. No lifecycle defects remained.

The scientific-semantics reviewer passed **87 focused resume/context/runner/
history/semantic tests** and **54 objective/finalization/smoke tests with
5 subtests**, without `PYTHONPATH`. It independently reran both historical
copied-study probes, confirmed unchanged source and trial-evidence hashes,
and passed the scientific AST verifier and `git diff --check`.
An additional independent real-objective/CSV/constrained-TPE adoption probe
preserved the old infeasible and feasible FrozenTrials. A new fatal candidate
was PRUNED before fit with positive feasibility sentinels; its successor fit
once and persisted negative constraints. Both sampler completion callbacks
fired, the following sample saw the pruned feedback, CSV evidence represented
all four evaluations, and constrained best-trial selection retained a feasible
candidate. These checks used test fixtures rather than physical hardware.

The accepted dispositions remain in `claude_review_disposition.md`: guarded
native-only legacy adoption, component context capabilities, stable public
sampler identities with exact historical aliases, corrected pytest imports,
truthful returned-proposal ledger names, clean provider endpoints, and
unchanged actual-versus-intended WAITING evidence semantics. Unsigned LLM or
plugin adoption is unsupported; native adoption remains an explicit user
attestation rather than automatic historical-provenance certification.
Documentation-only finalization required no further suite execution or
production-code changes. The follow-up is committed locally; nothing was
pushed, merged or published.
