# LLM candidate generation

`llm_generator` proposes raw model parameters, validates them against the model family's search space, and hands them to CREST's shared runner for training and evaluation.
The runner owns trials and budgets.
The LLM supplies proposals, not measurement evidence.

- Run a [live experiment](#run-an-experiment), try [offline generation](#try-generation-without-running-a-campaign), or configure a [fixture campaign](#configure-a-campaign).
- Understand [evidence](#evidence-and-memory), [validation](#validation-and-failures), and [full-history behavior](#full-history-generation), then [inspect a run](#inspect-a-run).
- Extend the [optimizer](#extend-the-optimizer), add a [provider adapter](#add-an-adapter), or change the [prompts](#extend-the-prompts).

## Run an experiment

1. Start with a complete experiment YAML for your dataset, model family, objective, and runtime.
   Replace its optimizer block with the following configuration.

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

2. Choose a model available to your provider account.
   From the repository root, activate your CREST environment, export your key, and run the following command.

   ```bash
   export OPENROUTER_API_KEY='REPLACE_WITH_YOUR_API_KEY'
   python src/nas_model_client.py --config /path/to/experiment.yaml --study-name llm-live
   unset OPENROUTER_API_KEY
   ```

The key above is a placeholder.
Keep real credentials out of configuration files and source control.
Live calls may incur charges, and the experiment can train models or access hardware according to its configuration.

See [LLM settings](../../../config/README.md#llm-settings) for endpoint choices and [pricing catalogs and spending caps](../../../config/README.md#configure-pricing-and-a-spending-cap).
For the OpenAI Responses [pricing example](../../../config/README.md#configure-pricing-and-a-spending-cap), enter credentials without shell-history exposure using the following zsh commands.
Use the environment-variable name configured by `api_key_env`.

```zsh
read -rs 'OPENAI_API_KEY?API key: '; printf '\n'
export OPENAI_API_KEY
python src/nas_model_client.py --config /path/to/experiment.yaml --study-name llm-live
unset OPENAI_API_KEY
```

For a local example that makes no provider calls and performs no training or hardware evaluation, use the [offline generation example](#try-generation-without-running-a-campaign).

The example disables accumulated memory.
See [history configuration](../../../config/README.md#responses-transport-and-history-modes) for defaults, supported families, and full-history constraints, and [memory settings](../../../config/README.md#accumulated-experimental-memory) for summary configuration.

## Evidence and memory

Bounded history supplies recent trials, best eligible anchors, and optional memory derived from older terminal evidence.
Semantic context adds workload and deployment guidance.
It does not change parameter legality.
Failed attempts are useful evidence, and repeated references to a trial represent one observation.
Memory and rationales are proposal guidance, not measured results.
Inspect trial state, feasibility, and measurements before treating a candidate as successful because `COMPLETE` alone can include penalty results or skipped training.

Repeated summary failures suspend memory updates while candidate generation continues.
Inspect `optimizer_events.jsonl` in the memory ledger to diagnose the cause.
For programmatic recovery, see [`ExperimentalMemory.reset_summary_failures`](memory.py).
There is no CLI reset switch.
The [configuration guide](../../../config/README.md) covers history and memory settings.

## Validation and failures

Candidates must supply every active raw descriptor field with legal values.
Required rationales must use the configured envelope.
Bounded generation returns any accepted subset immediately.
If none pass, it uses bounded content repairs, then tries a legal unique local random fallback.
Full-history mode has separate refill rules below.
`random_seed` controls local fallback, not provider output.

Authentication and credit failures stop immediately.
Transient transport failures receive bounded retries, then stop.
They do not trigger random fallback.
Inspect `rejected_candidates.jsonl` and request/response pairs for content errors before changing the experiment contract.
Fixed warm-up attempts run before provider calls, consume trial budget, and resume from saved provenance.
Keep their candidate contents unchanged.
See [budgets and resume](../README.md#budgets-and-resume) for interrupted trials.
Insufficient funds stop generation without fallback.
The [pricing guide](../../../config/README.md#configure-pricing-and-a-spending-cap) owns spend-state recovery and compatible cap changes.

## Full-history generation

For `audio_dscnn`, `history_mode: full` supplies exact columnar evidence and network coverage instead of bounded summaries.
Accepted entries are retained across refill calls controlled by `max_refill_attempts`.
Malformed responses use the separate shared repair allowance.
See [history configuration](../../../config/README.md#responses-transport-and-history-modes) for required settings and allowed limits.
Exhaustion returns any retained subset, or attempts local random fallback when nothing was accepted.

Python assigns local/exploration quotas and coverage targets.
Local candidates name eligible historical parents.
Exploration has no parent.
Raw and decoded network duplicates are rejected.
See the [configuration and response example](#full-history-audio-ds-cnn) for parent rules and response semantics.

## Inspect a run

The ledger is stored under `<artifact_dir>/llm_optimizer`.

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
Actual sampled values are in `trial.params`.
Early failures can leave it incomplete.
Keep the study database and artifact directory together.
`trials.csv` exports Optuna parameters and attributes.
The configured measurement CSV records outcome and feasibility evidence.
Each ledger directory requires a single writer.

## Try generation without running a campaign

From the repository root, in an environment with CREST's dependencies installed, run the following example.
It validates two configured candidates and writes a temporary ledger.
It does not start the runner, train a model, or access hardware.

```sh
PYTHONPATH=src python - <<'PY'
import random
from pathlib import Path
from tempfile import TemporaryDirectory

from crest.optimizers.llm import (
    FakeProvider, LLMLedger, PromptContext, SearchParam,
    SearchSpaceDescriptor, generate_llm_batch,
)

space = SearchSpaceDescriptor((
    SearchParam("width", "int", low=2, high=8),
    SearchParam("mode", "categorical", choices=("small", "large")),
))
context = PromptContext(
    study_name="llm-smoke",
    model_family="example",
    descriptor=space,
    objective_summary="maximize measured accuracy",
    attempted_trials=0,
    feasible_completed_trials=0,
    target_feasible_trials=2,
    max_total_attempts=4,
    batch_size=2,
)
provider = FakeProvider([{"candidates": [
    {"width": 4, "mode": "small"},
    {"width": 7, "mode": "large"},
]}])
with TemporaryDirectory() as directory:
    ledger = LLMLedger(Path(directory) / "llm_optimizer")
    proposals = generate_llm_batch(
        (), provider, context, ledger,
        prompt_version="v1", max_repair_attempts=0,
        rng=random.Random(0),
    )
    assert [dict(p.params) for p in proposals] == [
        {"width": 4, "mode": "small"},
        {"width": 7, "mode": "large"},
    ]
    assert all(p.provenance["proposal_source"] == "llm" for p in proposals)
    print([dict(p.params) for p in proposals])
    print(sorted(path.name for path in ledger.root.iterdir()))
PY
```

The expected result is two raw mappings matching the assertions, followed by the ledger filenames.
No trials are created.

To inspect the files afterward, replace the temporary directory with a dedicated persistent output path.

## Configure a campaign

Keep an existing complete campaign configuration for the chosen dataset, model family, objective, and runtime.
Replace its optimizer block with this pattern, substituting legal raw parameters for that family's descriptor.

```yaml
optimizer:
  type: llm_generator
  llm:
    provider: fake
    batch_size: 2
    max_repair_attempts: 1
    random_seed: 0
    recent_trial_window: 10
    anchor_count: 5
    semantic_context: true
    require_rationale: false
    memory:
      enabled: false
    responses:
      - candidates:
          - {width: 4, mode: small}
          - {width: 7, mode: large}
```

This is an optimizer fragment, not a complete campaign configuration.
`width` and `mode` belong only to the smoke example above.
Use the normal campaign entry point after configuring actual parameters and evaluation resources.
Selecting `fake` prevents provider network calls and does **not** disable training, dataset access, deployment, or hardware evaluation in the runner.

Launch the configured campaign from the repository root.

```sh
python3 src/nas_model_client.py --config /path/to/config.yaml --study-name llm_fake_smoke
```

Fixtures are consumed once per provider request, including repair requests, and must cover the intended run.
Exhaustion raises an error and does not automatically trigger random fallback.
Each fixture may be a JSON object or a string containing completion text, so strings such as `"not JSON"` can exercise repair behavior.

For `require_rationale: true`, each candidate must instead have exactly this shape.

```json
{"candidates": [{"params": {"width": 4, "mode": "small"}, "rationale": "Test a smaller architecture first."}]}
```

For fixed candidates before generation, see [warm-up configuration and recovery](../../../config/README.md#fixed-warm-up-and-recovery).

## Full-history audio DS-CNN

Merge this block into a complete `audio_dscnn` experiment.
The existing dataset, score, deployment, training, and budget settings still apply.

```yaml
optimizer:
  type: llm_generator
  llm:
    provider: openrouter
    model: YOUR_PROVIDER_MODEL_ID
    api_key_env: OPENROUTER_API_KEY
    history_mode: full
    require_rationale: true
    batch_size: 1
    max_repair_attempts: 1
    max_refill_attempts: 2
    memory:
      enabled: false
```

A valid response for an initial exploration slot, with no extra runtime search fields active, is shown below.

```json
{
  "candidates": [{
    "params": {
      "base_channels": 8,
      "num_blocks": 2,
      "kernel_time": 3,
      "kernel_freq": 3,
      "stride_schedule": "light",
      "channel_growth": 1.0,
      "max_channels": 64,
      "depth_multiplier": 1,
      "pointwise_scale": 1.0,
      "dropout_rate": 0.0,
      "norm_flag": true,
      "dense_units": 0,
      "global_pool_type": "avg",
      "activation": "relu"
    },
    "rationale": "Measure a compact baseline in the assigned coverage cell.",
    "intent": "exploration",
    "parent_trial": null
  }]
}
```

The initial prompt assigns exploration.
Later prompts can assign a different coverage cell, so this response is not a fixture for every round.
Supply every active descriptor field, including runtime fields when enabled.
For a local slot, set `intent` to `local` and `parent_trial` to an integer from the prompt's eligible parents, and supply a new candidate with complete parameters.

Accepted proposals enter the normal trial queue and evaluation path.
The example response is valid only when it matches the prompt's assigned coverage cell and active descriptor.
See [full-history generation](#full-history-generation) for duplicate, repair, and refill behavior.

## Extend the optimizer

| Location | Responsibility |
| --- | --- |
| [`component.py`](component.py) | Implements `OptimizerABC`. Coordinates lifecycle, warm-ups, and proposal rounds |
| [Completion providers](#completion-providers) | Completion interface, adapters, and provider setup |
| [Prompt construction](#prompt-construction-and-evidence) | Request construction, evidence selection, and phase guidance |
| [`generation.py`](generation.py), [`schemas.py`](schemas.py) | Bounded/full request loops. Candidate, rationale, quota, and duplicate validation |
| [`memory.py`](memory.py), [`full_history.py`](full_history.py) | Summary updates. Exact evidence, decoded identities, coverage, and parent selection |
| [`config.py`](config.py), [`component.py`](component.py) | Settings validation and deterministic resume identity |
| [`ledger.py`](ledger.py), [`spending.py`](spending.py) | Proposal provenance and provider spending accounting |

1. Choose the responsible module.
   Follow the [adapter](#add-an-adapter) or [prompt](#extend-the-prompts) extension procedure below.
   Model parameters belong in the family's search descriptor, not a provider-specific schema.
2. Preserve local candidate validation and resume identity.
   Update the prompt version or optimizer identity when the experiment contract changes.
3. Add focused valid/invalid cases and verify provenance, then run the following checks.

   ```sh
   python -m pytest -q test/test_llm_candidate_validation.py test/test_llm_generation.py test/test_llm_provider_prompt_ledger.py test/test_llm_proposal_contract.py
   ```

A change is done when the generation example works, focused tests cover its behavior, proposal evidence remains auditable, and the shared runner retains ownership of queueing and evaluation.

## Completion providers

Providers turn prompts into normalized completion responses.
See [LLM settings](../../../config/README.md#llm-settings) for endpoints, authentication defaults, JSON mode, and retry settings.

| File | Responsibility |
| --- | --- |
| `providers/provider.py` | Public imports and `build_provider` factory |
| `providers/records.py` | Request/response records, `LLMProvider` protocol, errors, and `FakeProvider` |
| `providers/chat.py`, `providers/responses.py` | Chat Completions and Responses adapters |
| `providers/transport.py` | Authentication, retries, and per-attempt spend accounting |
| `providers/utils.py` | JSON-mode defaults and pricing lookup |
| `providers/pricing.json` | Bundled conservative accounting rates |


### Try the provider interface offline

```python
from crest.optimizers.llm.providers.provider import FakeProvider, LLMRequest

provider = FakeProvider([{"candidates": [{"width": 4}]}])
response = provider.complete_json(LLMRequest(
    system_prompt="Return candidate parameters as JSON.",
    user_prompt="Propose a width.",
    prompt_version="v1",
))
```

The [offline generation example](#try-generation-without-running-a-campaign) also exercises candidate validation and ledger output.

### Add an adapter

1. Implement `LLMProvider.complete_json`, returning a normalized `LLMResponse`.
   Live adapters can reuse `HTTPProvider` in `providers/transport.py` for authentication, retries, and accounting.
2. Add factory selection in `providers/provider.py` and validation in `config.py`.
3. Distinguish transport/authentication failures from repairable response errors.
   Keep credentials out of saved request and response evidence.
4. Test valid and malformed responses, retries, and spending limits with mocked transport.
   From the repository root, run the following checks.

   ```sh
   python -m pytest -q test/test_llm_openai_provider.py test/test_llm_responses_provider.py test/test_llm_provider_defaults.py test/test_llm_pricing.py test/test_llm_spending.py
   ```

An adapter is ready when configuration selects it, the checks pass, and generation still validates candidates before returning them to the runner.

## Prompt construction and evidence

This package turns search snapshots into provider-neutral requests.
It performs no provider calls, study writes, training, or hardware evaluation.
The [optimizer component](component.py) supplies context.
[generation](generation.py) sends requests and validates responses.

| File | Responsibility |
| --- | --- |
| `prompting/prompt_builder.py` | `PromptContext` and `build_candidate_request`. Assemble declarations, evidence, objectives, budgets, and repair feedback. |
| `prompting/history.py` | Select recent trial evidence and best-trial anchors from snapshots. |
| `prompting/objective_renderer.py` | Explain supported scoring expressions using known context. Does not calculate measured scores. |
| `prompting/phase_policy.py` | Derive exploration/refinement guidance from trial budgets. Does not enforce stopping conditions. |

### Use the request builder

The [offline generation example](#try-generation-without-running-a-campaign) has a complete generation example.
Prompt construction can also be exercised directly using that example's `PromptContext`.

```python
from crest.optimizers.llm.prompting.prompt_builder import build_candidate_request

request = build_candidate_request(context, prompt_version="v1")
print(request.user_prompt)
```

### Extend the prompts

1. Change the module responsible for the evidence or text you need.
   Keep provider transport details out of request construction.
2. Preserve raw parameter declarations and distinguish observed trial evidence from suggestions or explanations.
   Do not introduce credentials into context.
3. Update prompt/optimizer identity when the experiment contract changes, so incompatible settings cannot silently resume an existing study.
4. Add focused examples for the new behavior and missing or invalid evidence.

A change is complete when request construction remains deterministic, evidence selection tests pass, and generation still validates all returned candidates.
From the repository root, run the following checks.

```sh
python -m pytest -q test/test_llm_provider_prompt_ledger.py test/test_llm_history.py test/test_llm_anchors.py test/test_llm_objective_renderer.py test/test_llm_semantic_context.py
```
