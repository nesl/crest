<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->

# Optimizer contracts

`optuna` is the default native optimizer: it requests a `NativeRound`, and Optuna
samples parameters inside the objective. Explicit optimizers return an
`ExplicitRound` of complete raw parameter mappings and provenance. Both run
through the same objective, feasibility checks, trial budget, and callbacks.

[`runner.py`](runner.py) validates an entire explicit batch before reserving any
trials, then evaluates it through Optuna. Candidates must match the active
[`search-space descriptor`](search_space.py) exactly. Reservations are individual
database writes; resume drains existing WAITING trials before requesting new work.
[`history.py`](history.py) supplies immutable proposal/history records.

[`study_state.py`](study_state.py) owns persisted optimizer identity, strict resume
checks and interrupted-trial recovery. New studies receive an optimizer signature
after successful initialization. Existing signed studies resume only when their
signature matches the active optimizer configuration. A nonempty unsigned study
must use a new study name to continue searching; its history remains available
for analysis and offline plotting. Changing optimizer settings on resume must not
silently reinterpret an existing study.

[`context.py`](context.py) assembles bounded, deterministic semantic context for
explicit proposers. Its helpers in [`contexts/`](contexts/) cover
[data/task](contexts/data_task.py), [device](contexts/device.py),
[runtime](contexts/runtime.py), [STM32](contexts/stm32.py), and
[value normalization](contexts/values.py). Architectural hints are priors;
measured results take precedence. Native sampling skips descriptor/context setup.

See [`OptimizerABC`](../interfaces/optimizer.py), the round and budget types in
[`pipeline_types.py`](../pipeline_types.py), the
[model-family contract](../model_families/README.md), and the
[configuration guide](../../config/README.md).
Progress reporting is shared across optimizers; see
[search utilities](../search/README.md).

## Budgets and resume

`training.nas_trials` and `training.max_total_trials` are study totals. To extend
250 completions to 300, keep the same study name, database, artifact directory,
and original experiment configuration, then raise the total target and cap.
Optimizer identity checks do not fingerprint the whole experiment: retain the
original dataset, model, device, training, and score settings yourself. Changing
proposal settings requires a new study; nonempty unsigned studies cannot resume.

All reserved trials, including WAITING and RUNNING, consume the attempt cap.
With feasibility enabled, only feasible COMPLETE trials satisfy the target;
otherwise every COMPLETE trial counts. Failed or pruned attempts still use budget.
The runner checks the target between rounds, drains existing WAITING reservations
before requesting new work, and can drain them even at the attempt cap. A batch
can exceed the remaining completion need; an already satisfied target leaves
WAITING work untouched. `--smoke-test N` instead runs N additional attempts in
a separate smoke-study database.

RUNNING trials are never recovered automatically. After confirming their workers
have stopped, set `optimizer.recover_orphaned_trials: [12]` to the exact orphan
trial numbers and resume. Listed RUNNING trials become FAIL and still consume
attempts; unknown or WAITING numbers are rejected. Remove the list after recovery.
See the [configuration guide](../../config/README.md) for budget settings.

## Responsibilities and lifecycle

An optimizer chooses **raw trial parameters** or requests native sampling. It does
not build models, train, measure devices, calculate the objective, enqueue trials,
or change an Optuna Study or Trial. The shared runner owns those operations.
Built-in optimizers include native `optuna` and the explicit `llm_generator`; the
interface also supports application-defined proposers.

Components must support zero-argument construction. The lifecycle is:

1. Resolve `optimizer.type` from the registry and call `validate_config` with the
   normalized whole `optimizer` block. Validate settings without opening clients
   or requiring credentials.
2. Build `identity_config` and check the persisted campaign identity. Include
   deterministic, JSON-safe settings that change proposals; exclude secrets,
   machine-specific paths, and run budgets.
3. Assemble the requested `SearchContext`, then call `initialize(context, config)`.
   Open any runtime clients here. The context contains detached run metadata,
   objective names/directions, optional semantic context, and the active descriptor.
4. Call `propose_round(history, budget)` with fresh immutable `TrialRecord`
   snapshots. History includes failed and reserved trials; use recorded states
   and actual parameters rather than assuming every record is a success.
5. Validate and execute the requested round through the shared objective.

Every round must be nonempty and fit `budget.permitted_round_size`. An explicit
candidate must contain every active descriptor key and no extra keys, including
active deployment fields such as `quantization_mode` or `cpu_clock_mhz_index`.
Use raw sampled values, not a decoded architecture dictionary. Integer, numeric,
and categorical values must satisfy their declarations; categorical types matter.
Provenance must be JSON-safe. Duplicate candidates are legal at this boundary;
the proposer owns any duplicate avoidance or fallback policy. Raise
`OptimizerBudgetExhausted` when an optimizer-owned resource cannot fund more work.
Do not return an empty round to signal exhaustion.

## Minimal explicit proposer

This deliberately simple component repeats one configured candidate. It shows the
contract without prescribing an optimization algorithm. Save it in an importable
application module such as `fixed_optimizer.py`:

```python
from collections.abc import Mapping

from crest.interfaces import OptimizerABC
from crest.interfaces.optimizer import OptimizerBudgetExhausted
from crest.optimizers.search_space import validate_candidate
from crest.pipeline_types import CandidateProposal, ExplicitRound


class FixedCandidateOptimizer(OptimizerABC):
    def requires_semantic_context(self, config):
        return False

    def validate_config(self, config):
        candidate = config.get("candidate")
        if not isinstance(candidate, Mapping) or not candidate:
            raise ValueError("optimizer.candidate must be a nonempty raw mapping")

    def identity_config(self, config):
        return {"candidate": dict(config["candidate"])}

    def initialize(self, context, config):
        self.validate_config(config)
        if context.search_space is None:
            raise ValueError("FixedCandidateOptimizer requires a descriptor")
        validate_candidate(config["candidate"], context.search_space)
        self.candidate = dict(config["candidate"])

    def propose_round(self, history, budget):
        if budget.permitted_round_size < 1:
            raise OptimizerBudgetExhausted("No trial reservation is available")
        proposal = CandidateProposal(
            params=dict(self.candidate),
            provenance={"strategy": "fixed_candidate"},
        )
        return ExplicitRound((proposal,))
```

Register the class **before loading/normalizing the run configuration** in the
same Python process that launches CREST:

```python
from crest.registry import optimizer_registry
from fixed_optimizer import FixedCandidateOptimizer

optimizer_registry.register("fixed_candidate", FixedCandidateOptimizer)
```

Select it using `optimizer.type`; settings live directly in the optimizer block,
not in a nested `params` block:

```yaml
optimizer:
  type: fixed_candidate
  candidate:
    width: 16
```

This YAML is valid only for a descriptor whose entire raw surface is an integer
`width` parameter allowing 16. For a real family, replace `candidate` with the
complete mapping for that run's descriptor and active deployment options. Merely
putting a Python module on disk does not register it in a separate CLI process.
For a repository-owned optimizer, add lazy, idempotent registration to
[`ensure_optimizer_components_registered`](../builtin_components.py), following
its `optuna` entry, so normal CLI selection can discover it.

For native sampling, the existing [`OptunaOptimizer`](optuna.py) is the minimal
example: it sets `requires_search_space = False`, disables semantic context, and
returns `NativeRound(budget.permitted_round_size)`. Select it with
`optimizer: {type: optuna}`. It leaves parameter sampling inside the objective;
an explicit proposer needs the descriptor even when it does not need semantic
context.

## Adding an optimizer

1. Implement `OptimizerABC` in a module that imports without datasets, runtime
   clients, or credentials. Keep proposal logic separate from evaluation logic.
2. Choose native or explicit rounds. For explicit rounds, confirm each supported
   model family implements `trial_search_space` and that its raw declarations
   match objective-side sampling. Request semantic context only when needed.
3. Implement configuration validation and proposal identity. Delay runtime setup
   until `initialize`; derive adaptive state from the fresh history on each call
   so restart behavior does not depend on a lost in-memory counter.
4. Return bounded rounds and useful JSON-safe provenance. Define failure,
   duplicate, and resource-exhaustion behavior explicitly.
5. Register the class, select its key with `optimizer.type`, and document required
   settings plus one complete candidate for a concrete descriptor.
6. Exercise the component with detached context/history first, then a small shared
   runner search and resume. [`test/proposal_plugin.py`](../../../test/proposal_plugin.py)
   demonstrates a test-only proposer without an Optuna dependency.

## Definition of Done

- The class constructs without arguments, imports without clients or credentials,
  and rejects invalid settings before runtime initialization.
- Its identity contains every proposal-affecting setting in deterministic,
  JSON-safe form, and resuming with changed settings fails clearly.
- Explicit proposals match the complete active raw descriptor, use legal values,
  carry serializable provenance, and never exceed the permitted round size.
- The component works with empty history, failed/reserved records, exhausted
  resources, and fresh history after restart without mutating any snapshot.
- A small integration run uses the shared objective, feasibility policy, budgets,
  and callbacks; an invalid batch reserves no trials, and resume drains existing
  WAITING work before asking for more proposals.
- Registration, configuration selection, and the native path remain usable;
  focused tests cover the new component's behavior and resume identity.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Unknown optimizer type | Register the exact case-sensitive key before configuration normalization, in the launching process. |
| Missing or unknown candidate keys | Compare against the active descriptor, including deployment fields; do not pass decoded model config. |
| Candidate value rejected | Check bounds and scalar types; `True` is not an integer candidate, and categorical choices require exact types. |
| Descriptor unavailable | Implement the family's `trial_search_space`, or use native sampling when no descriptor is needed. |
| Resume identity mismatch | Keep proposal settings consistent with the existing campaign, or start a new study for changed settings. |
| Proposer not called immediately after resume | Existing WAITING reservations are drained first. |
| Empty or oversized round rejected | Return positive work within `permitted_round_size`; use `OptimizerBudgetExhausted` for resource exhaustion. |

For LLM evidence, provider setup, and recovery, see the [LLM guide](llm/README.md).
