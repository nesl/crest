<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->

# Optimizer contracts

`optuna` requests native sampling inside the objective.
Explicit optimizers return complete raw parameters and provenance.
Both use the shared runner for evaluation and budgets.

See [`OptimizerABC`](../interfaces/optimizer.py), [round and budget types](../pipeline_types.py), the [model-family contract](../model_families/README.md), and the [configuration guide](../../config/README.md).

Choose a task below.

- Operate a study using [budgets and resume](#budgets-and-resume), [progress reports](#progress-and-reproducibility), or [troubleshooting](#troubleshooting).
- Build a proposer using the [lifecycle contract](#responsibilities-and-lifecycle), [complete example](#minimal-explicit-proposer), and [extension checklist](#adding-an-optimizer).
- Configure LLM proposals using the [LLM guide](llm/README.md#run-an-experiment).

## Budgets and resume

`training.nas_trials` and `training.max_total_trials` are study totals.
To extend 250 completions to 300, keep the same study name, database, artifact directory, and original experiment configuration, then raise the total target and cap.
Optimizer identity checks do not fingerprint the whole experiment.
Retain the original dataset, model, device, training, and score settings yourself.
Changing proposal settings requires a new study.
Nonempty unsigned studies cannot resume.

All reserved trials, including WAITING and RUNNING, consume the attempt cap.
With feasibility enabled, only feasible COMPLETE trials satisfy the target.
Otherwise every COMPLETE trial counts.
Failed or pruned attempts still use budget.
The runner checks the target between rounds, drains existing WAITING reservations before requesting new work, and can drain them even at the attempt cap.
A batch can exceed the remaining completion need.
An already satisfied target leaves WAITING work untouched.
`--smoke-test N` instead runs N additional attempts in a separate smoke-study database.

RUNNING trials are never recovered automatically.
After confirming their workers have stopped, set `optimizer.recover_orphaned_trials: [12]` to the exact orphan trial numbers and resume.
Listed RUNNING trials become FAIL and still consume attempts.
Unknown or WAITING numbers are rejected.
Remove the list after recovery.
See the [configuration guide](../../config/README.md) for budget settings.

## Responsibilities and lifecycle

An optimizer chooses **raw trial parameters** or requests native sampling.
It does not build models, train, measure devices, calculate the objective, enqueue trials, or change an Optuna Study or Trial.
The shared runner owns those operations.
Built-in optimizers include native `optuna` and the explicit `llm_generator`.
The interface also supports application-defined proposers.

[`runner.py`](runner.py) validates explicit batches before reservation.
[`history.py`](history.py) supplies immutable records, and [`study_state.py`](study_state.py) handles identity checks and recovery.
[`context.py`](context.py) assembles semantic context using [workload](contexts/data_task.py), [device](contexts/device.py), [runtime](contexts/runtime.py), [STM32](contexts/stm32.py), and [normalization](contexts/values.py) helpers.
Architectural hints are priors and measured results take precedence.
Native sampling skips descriptor/context setup.

Components must support zero-argument construction.
The lifecycle is as follows.

1. Resolve `optimizer.type` from the registry and call `validate_config` with the normalized whole `optimizer` block.
   Validate settings without opening clients or requiring credentials.
2. Build `identity_config` and check the persisted campaign identity.
   Include deterministic, JSON-safe settings that change proposals.
   Exclude secrets, machine-specific paths, and run budgets.
3. Assemble the requested `SearchContext`, then call `initialize(context, config)`.
   Open any runtime clients here.
   The context contains detached run metadata, objective names/directions, optional semantic context, and the active descriptor.
4. Call `propose_round(history, budget)` with fresh immutable `TrialRecord` snapshots.
   History includes failed and reserved trials.
   Use recorded states and actual parameters rather than assuming every record is a success.
5. Validate and execute the requested round through the shared objective.

Every round must be nonempty and fit `budget.permitted_round_size`.
An explicit candidate must contain every active descriptor key and no extra keys, including active deployment fields such as `quantization_mode` or `cpu_clock_mhz_index`.
Use raw sampled values, not a decoded architecture dictionary.
Integer, numeric, and categorical values must satisfy their declarations.
Categorical types matter.
Provenance must be JSON-safe.
Duplicate candidates are legal at this boundary.
The proposer owns any duplicate avoidance or fallback policy.
Raise `OptimizerBudgetExhausted` when an optimizer-owned resource cannot fund more work.
Do not return an empty round to signal exhaustion.

## Minimal explicit proposer

This deliberately simple component repeats one configured candidate.
It shows the contract without prescribing an optimization algorithm.
Save it in an importable application module such as `fixed_optimizer.py`.

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

Register the class **before loading/normalizing the run configuration** in the same Python process that launches CREST.

```python
from crest.registry import optimizer_registry
from fixed_optimizer import FixedCandidateOptimizer

optimizer_registry.register("fixed_candidate", FixedCandidateOptimizer)
```

Select it using `optimizer.type`.
Settings live directly in the optimizer block, not in a nested `params` block.

```yaml
optimizer:
  type: fixed_candidate
  candidate:
    width: 16
```

This YAML is valid only for a descriptor whose entire raw surface is an integer `width` parameter allowing 16.
For a real family, replace `candidate` with the complete mapping for that run's descriptor and active deployment options.
Merely putting a Python module on disk does not register it in a separate CLI process.
For a repository-owned optimizer, add lazy, idempotent registration to [`ensure_optimizer_components_registered`](../builtin_components.py), following its `optuna` entry, so normal CLI selection can discover it.

For native sampling, the existing [`OptunaOptimizer`](optuna.py) is the minimal example.
It sets `requires_search_space = False`, disables semantic context, and returns `NativeRound(budget.permitted_round_size)`.
Select it with `optimizer: {type: optuna}`.
It leaves parameter sampling inside the objective.
An explicit proposer needs the descriptor even when it does not need semantic context.

## Adding an optimizer

1. Implement `OptimizerABC` in a module that imports without datasets, runtime clients, or credentials.
   Keep proposal logic separate from evaluation logic.
2. Choose native or explicit rounds.
   For explicit rounds, confirm each supported model family implements `trial_search_space` and that its raw declarations match objective-side sampling.
   Request semantic context only when needed.
3. Implement configuration validation and proposal identity.
   Delay runtime setup until `initialize`.
   Derive adaptive state from the fresh history on each call so restart behavior does not depend on a lost in-memory counter.
4. Return bounded rounds and useful JSON-safe provenance.
   Define failure, duplicate, and resource-exhaustion behavior explicitly.
5. Register the class, select its key with `optimizer.type`, and document required settings plus one complete candidate for a concrete descriptor.
6. Exercise the component with detached context/history first, then a small shared runner search and resume.
   [`test/proposal_plugin.py`](../../../test/proposal_plugin.py) demonstrates a test-only proposer without an Optuna dependency.

## Definition of Done

- The class constructs without arguments, imports without clients or credentials, and rejects invalid settings before runtime initialization.
- Its identity contains every proposal-affecting setting in deterministic, JSON-safe form, and resuming with changed settings fails clearly.
- Explicit proposals match the complete active raw descriptor, use legal values, carry serializable provenance, and never exceed the permitted round size.
- The component works with empty history, failed/reserved records, exhausted resources, and fresh history after restart without mutating any snapshot.
- A small integration run uses the shared objective, feasibility policy, budgets, and callbacks.
  An invalid batch reserves no trials, and resume drains existing WAITING work before asking for more proposals.
- Registration, configuration selection, and the native path remain usable.
  Focused tests cover the new component's behavior and resume identity.

## Progress and reproducibility

Search execution refreshes progress after every completed attempt and when a study resumes, under `<models_dir>/<study>/progress/`.
Failed, pruned, infeasible, and nonfinite attempts retain their attempt positions and time accounting, but cannot improve incumbents or hypervolume.

[`progress.py`](../search/progress.py) writes scalar score-progress CSV and PNG/PDF charts against attempts and elapsed time.
Multiobjective reports use only hypervolume, with `hypervolume_progress.csv`, `hypervolume_history.png`/`.pdf`, `hypervolume_vs_time.png`/`.pdf`, and `hypervolume_reference.json`.
An `error.txt` in the progress directory indicates a reporting failure.
Existing plots may be stale even though search results remain available.

Hypervolume uses a fixed reference point in objective order and original units.
Configure `outputs.hypervolume_reference_point` for a meaningful comparison target.
The selected reference is persisted for resume.
Without a configured reference, the first eligible finite objective vector freezes a monitoring reference with a 10% margin in each objective's worse direction.
This fallback is for monitoring, not a substitute for a domain-chosen benchmark.
It does not move as results change.
Points outside it need not improve hypervolume.
Comparable studies must use the same reference, objective definitions, units, directions, and eligibility rules.
Only feasible, finite COMPLETE vectors contribute.
Pruned attempts cannot contribute partial results.

Generate a progress report from a saved study database.

```python
from pathlib import Path
from crest.search.progress import read_trials, write_progress

trials = read_trials(Path("results/optuna.db"), "my-study")
write_progress(trials, Path("report"), "my-study")
# For multiobjective reports, optionally pass reference_point=[...] in objective order.
```

Use an existing database and exact study name.
This writes to `report/progress/`.
The SQLite reader uses a read-only snapshot.
Feasibility defaults to requiring affirmative evidence.
Set `feasibility_enabled=False` only for a study configured without feasibility rules.
For in-memory multiobjective records, pass objective `directions` to `write_progress`.
SQLite records carry their directions.

[`training_randomness.py`](../search/training_randomness.py) validates `training.random_seed` and `training.deterministic_ops`, then resets Python/NumPy/TensorFlow randomness before evaluation.
The same configured seed is reused across architectures and final fits.
Deterministic operations require a compatible runtime.
A seed alone does not guarantee cross-platform reproducibility.

See the [configuration reference](../../config/README.md) for training settings.
Targeted coverage lives in [`test_progress.py`](../../../test/test_progress.py) and [`test_training_randomness.py`](../../../test/test_training_randomness.py).

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Unknown optimizer type | Register the exact case-sensitive key before configuration normalization, in the launching process. |
| Missing or unknown candidate keys | Compare against the active descriptor, including deployment fields. Do not pass decoded model config. |
| Candidate value rejected | Check bounds and scalar types. `True` is not an integer candidate, and categorical choices require exact types. |
| Descriptor unavailable | Implement the family's `trial_search_space`, or use native sampling when no descriptor is needed. |
| Resume identity mismatch | Keep proposal settings consistent with the existing campaign, or start a new study for changed settings. |
| Proposer not called immediately after resume | Existing WAITING reservations are drained first. |
| Empty or oversized round rejected | Return positive work within `permitted_round_size`. Use `OptimizerBudgetExhausted` for resource exhaustion. |

For LLM evidence, provider setup, and recovery, see the [LLM guide](llm/README.md).
