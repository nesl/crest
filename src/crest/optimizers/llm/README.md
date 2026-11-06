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

## How bounded generation works

1. The component proposes configured warm-ups first, then builds a request from
   recent history, best-trial anchors, the search space, objective, and budget.
2. A provider returns candidates. Local validation checks raw parameter names,
   values, duplicates, and any required rationales.
3. Invalid content can trigger bounded repair requests, followed by a random
   fallback if repairs are exhausted. Other provider failures propagate.
4. The component returns validated proposals to the shared runner for queueing
   and evaluation. A valid proposal is not yet a feasible or successful trial.

See the [shared optimizer guide](../README.md) for evaluation and lifecycle
contracts, and the [component interfaces](../../interfaces/README.md) for APIs.

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

## Extend the optimizer

| Location | Responsibility |
| --- | --- |
| [`component.py`](component.py) | Implements `OptimizerABC`; coordinates lifecycle, warm-ups, and proposal rounds |
| [`providers/`](providers/README.md) | Completion interface, adapters, and provider setup |
| [`prompting/`](prompting/README.md) | Request construction, evidence selection, and phase guidance |
| [`generation.py`](generation.py), [`schemas.py`](schemas.py) | Generation, local validation, repair, and fallback |
| [`memory.py`](memory.py), [`full_history.py`](full_history.py) | Accumulated summaries and full-history evidence policies |
| [`config.py`](config.py) | Settings defaults and validation |
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
