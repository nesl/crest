# LLM candidate generation

`llm_generator` proposes raw model parameters, validates them against the model
family's search space, and hands them to CREST's shared runner for training and
evaluation. The runner owns trials and budgets; the LLM supplies proposals,
not measurement evidence.

## Run an experiment

1. Start with a complete experiment YAML for your dataset, model family,
   objective, and runtime. Replace its optimizer block with:

   ```yaml
   optimizer:
     type: llm_generator
     llm:
       provider: openrouter
       model: YOUR_PROVIDER_MODEL_ID
       api_key_env: OPENROUTER_API_KEY
       batch_size: 1
       memory:
         enabled: false
   ```

2. Choose a model available to your provider account. From the repository root,
   activate your CREST environment, export your key, and run:

   ```bash
   export OPENROUTER_API_KEY='REPLACE_WITH_YOUR_API_KEY'
   python src/nas_model_client.py --config /path/to/experiment.yaml --study-name llm-live
   unset OPENROUTER_API_KEY
   ```

The key above is a placeholder. Keep real credentials out of configuration files
and source control. Live calls may incur charges, and the experiment can train
models or access hardware according to its configuration.

See the [provider guide](providers/README.md) for other endpoints, credential
entry without shell-history exposure, and [pricing catalogs and spending caps](providers/README.md#configure-pricing-and-a-spending-cap). For a local example
that makes no provider calls and performs no training or hardware evaluation,
use the [generation examples](EXAMPLES.md).

Bounded history is the default. Accumulated memory is enabled by default; the
example disables it. Full-history mode is available for `audio_dscnn` with its
required rationale and policy settings. See the
[configuration guide](../../../config/README.md) for evidence policies, warm-ups,
rationale requirements, budgets, and resume rules. Changing identity-bearing settings,
including the rationale contract, requires a new study.

## Evidence and memory

Bounded history supplies recent trials, best eligible anchors, and optional
memory derived from older terminal evidence. Semantic context adds workload and
deployment guidance; it does not change parameter legality. Failed attempts are
useful evidence, and repeated references to a trial represent one observation.
Memory and rationales are proposal guidance, not measured results. Inspect trial
state, feasibility, and measurements before treating a candidate as successful:
`COMPLETE` alone can include penalty results or skipped training.

Memory updates retain cited findings and bounded pending evidence. Rejected
summaries get one repair; further rejection preserves accepted memory and reports
omitted evidence. After three consecutive exhausted summary chunks, summary calls
remain suspended across restarts while generation continues with bounded pending
records. Inspect the memory ledger's `optimizer_events.jsonl`, correct the cause,
and call `ExperimentalMemory.reset_summary_failures()` on the owning memory
instance to resume summaries without discarding findings or coverage. There is no
CLI reset switch. See the [configuration guide](../../../config/README.md) for
history, semantic-context, and memory settings.

## Validation and failures

Candidates must supply every active raw descriptor field with legal values;
required rationales must use the configured envelope. Bounded generation returns
any accepted subset immediately. If none pass, it uses bounded content repairs,
then tries a legal unique local random fallback. Full-history mode has separate
refill rules below. `random_seed` controls local fallback, not provider output.

Authentication and credit failures stop immediately. Transient transport failures
receive bounded retries, then stop; they do not trigger random fallback. Inspect
`rejected_candidates.jsonl` and request/response pairs for content errors before
changing the experiment contract. Fixed warm-up attempts run before provider
calls, consume trial budget, and resume from saved provenance; keep their file
unchanged. See [budgets and resume](../README.md#budgets-and-resume) for interrupted
trials and the [pricing guide](providers/README.md#configure-pricing-and-a-spending-cap)
for provider caps. Retain `spend_state.json` on resume: uncertain usage keeps its
reservation, and insufficient funds stop generation without fallback. The dollar
cap may increase, but saved tariffs and token bounds must match.

## Full-history generation

For `audio_dscnn`, `history_mode: full` supplies exact columnar evidence and
network coverage instead of bounded summaries. It requires rationales, requires memory
to be disabled, and supports batch sizes 1–5. Accepted entries are retained across up to
`max_refill_attempts` refill calls (default 2); malformed responses use the
separate shared repair allowance (at most 1). Exhaustion returns any retained
subset, or attempts local random fallback when nothing was accepted.

Python assigns local/exploration quotas and coverage targets. Local candidates
name eligible historical parents; exploration has no parent. Raw and decoded
network duplicates are rejected. See the
[configuration and response example](EXAMPLES.md#full-history-audio-ds-cnn) for
parent rules and response semantics.

## Inspect a run

The ledger is stored under `<artifact_dir>/llm_optimizer`:

| Artifact | Contents |
| --- | --- |
| `requests/` | Paired request and response JSON |
| `prompt_contexts.jsonl` | Evidence supplied to generation |
| `rejected_candidates.jsonl` | Validation rejection reasons |
| `returned_candidates.jsonl` | Proposals returned to the runner |
| `accepted_candidates.jsonl` | Runner-confirmed acceptance |
| `optimizer_events.jsonl` | Repair and fallback events |

Returned proposals alone do not establish that trials were queued or evaluated.
Use Optuna trial state and `crest_proposal` attributes for queue acceptance.
Actual sampled values are in `trial.params`; early failures can leave it incomplete.
Keep the study database and artifact directory together. `trials.csv` exports
Optuna parameters and attributes; the configured measurement CSV records outcome
and feasibility evidence. Each ledger directory requires a single writer.

## Extend the optimizer

| Location | Responsibility |
| --- | --- |
| [`component.py`](component.py) | Implements `OptimizerABC`; coordinates lifecycle, warm-ups, and proposal rounds |
| [`providers/`](providers/README.md) | Completion interface, adapters, and provider setup |
| [`prompting/`](prompting/README.md) | Request construction, evidence selection, and phase guidance |
| [`generation.py`](generation.py), [`schemas.py`](schemas.py) | Bounded/full request loops; candidate, rationale, quota, and duplicate validation |
| [`memory.py`](memory.py), [`full_history.py`](full_history.py) | Summary updates; exact evidence, decoded identities, coverage, and parent selection |
| [`config.py`](config.py), [`component.py`](component.py) | Settings validation and deterministic resume identity |
| [`ledger.py`](ledger.py), [`spending.py`](spending.py) | Proposal provenance and provider spending accounting |

1. Choose the responsible module; follow the provider or prompting guide for
   their extension procedures. Model parameters belong in the family's search
   descriptor, not a provider-specific schema.
2. Preserve local candidate validation and resume identity. Update the prompt
   version or optimizer identity when the experiment contract changes.
3. Add focused valid/invalid cases and verify provenance, then run:

   ```sh
   python -m pytest -q test/test_llm_candidate_validation.py test/test_llm_generation.py test/test_llm_provider_prompt_ledger.py test/test_llm_proposal_contract.py
   ```

A change is done when the generation example works, focused tests cover its
behavior, proposal evidence remains auditable, and the shared runner retains
ownership of queueing and evaluation.
