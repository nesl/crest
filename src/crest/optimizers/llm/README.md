# LLM candidate generation

`llm_generator` proposes raw architecture parameters through CREST's shared
optimizer contract. It does not own a search loop: the shared runner supplies
immutable history and a budget, receives an `ExplicitRound`, and executes the
returned `CandidateProposal` records through the normal evaluation pipeline.
Candidates must match the model family's declared `trial_search_space` exactly.
A legal proposal still needs training and measurement to establish its quality
and feasibility.

The supported provider is `fake`, which consumes configured response fixtures.
History is bounded and `memory.enabled` must be false. Live provider selection,
persistent memory, and full-history execution are not supported by the current
configuration and factory. See the [shared optimizer guide](../README.md),
[configuration guide](../../../config/README.md), and
[component interfaces](../../interfaces/README.md) for the surrounding contracts.

## Try generation without running a campaign

From the repository root, in an environment with CREST's dependencies installed,
run the following example. It creates a temporary ledger and exercises the same
prompt, provider, validation, and provenance path used by the optimizer. It does
not start the runner, train a model, or access hardware.

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

To inspect the files afterward, replace the temporary directory with a dedicated
persistent output path. Each ledger directory must have a single writer.

## Configure a campaign

Keep an existing complete campaign configuration for the chosen dataset, model
family, objective, and runtime. Replace its optimizer block with this pattern,
substituting legal raw parameters for that family's descriptor:

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

This is an optimizer fragment, not a complete campaign configuration. `width`
and `mode` belong only to the smoke example above. Use the normal campaign entry
point after configuring actual parameters and evaluation resources. Selecting
`fake` prevents provider network calls; it does **not** disable training, dataset
access, deployment, or hardware evaluation in the runner.

Launch the configured campaign from the repository root:

```sh
python3 src/nas_model_client.py --config /path/to/config.yaml --study-name llm_fake_smoke
```

Fixtures are consumed once per provider request, including repair requests, and
must cover the intended run. Exhaustion raises an error; it does not automatically
trigger random fallback. Each fixture may be a JSON object or a string containing
completion text, so strings such as `"not JSON"` can exercise repair behavior.

For `require_rationale: true`, each candidate must instead have exactly this shape:

```json
{"candidates": [{"params": {"width": 4, "mode": "small"}, "rationale": "Test a smaller architecture first."}]}
```

Optional `warmup_params_file` points to a nonempty JSON list of raw parameter
mappings. Configuration loading resolves its path relative to the YAML file.
Warm-ups are validated locally and proposed one at a time before provider calls;
resume checks verify their order, parameters, and provenance. Changes to fixed
warm-ups, fixtures, or identity-bearing policy settings can prevent resuming an
existing study. Use a new study for a changed experiment contract.

## Follow one proposal round

1. `component.py` checks fixed warm-up progress, then bounds the requested batch
   by both the runner's permitted round size and `llm.batch_size`.
2. `history.py` selects recent evidence and successful anchors. `prompt_builder.py`
   combines these with the descriptor, objective, semantic context, budget, and
   deterministic guidance from `phase_policy.py`.
3. `generation.py` records a request and calls `LLMProvider.complete_json`.
   `schemas.py` checks the JSON envelope, raw names and distributions, rationales,
   batch limit, and duplicates against terminal, waiting, and same-batch candidates.
4. Invalid content receives repair feedback for up to `max_repair_attempts`
   additional calls. A nonempty accepted subset returns immediately, even if it
   is smaller than requested. If content repairs are exhausted, generation tries
   up to 100 random samples to return one unique legal fallback. Provider failures
   other than repairable response errors propagate.
5. The component returns proposals with request provenance to the runner. Queue
   mutation and evaluation happen there, not in `generation.py`.

The ledger lives under `<artifact_dir>/llm_optimizer`. Inspect `requests/` for
paired request and response JSON, `prompt_contexts.jsonl` for evidence snapshots,
`rejected_candidates.jsonl` for rejection reasons, `returned_candidates.jsonl`
for generated proposals, and `optimizer_events.jsonl` for repairs/fallback events.
`accepted_candidates.jsonl` records runner-confirmed acceptance separately;
returned proposals alone are not proof that trials were queued or evaluated.

## Where to make changes

There are several files because provider I/O, pure evidence transformations,
local validation, and persistent audit writes have different responsibilities.
Most features should change one boundary and its focused tests rather than add a
new optimizer loop.

| Module | Role and intended extension point |
| --- | --- |
| `component.py` | Runner adapter implementing `OptimizerABC`; coordinates configuration, warm-ups, identity, and `ExplicitRound` output. Start here for lifecycle changes. |
| `provider.py` | Provider interface: `LLMRequest`, `LLMResponse`, and `LLMProvider.complete_json`. `FakeProvider` is the current adapter; `build_provider` selects it. |
| `schemas.py` | Response and local-validation contracts: `CandidateBatch`, rejection/result records, canonicalization, and duplicate checks. Change this when candidate acceptance changes. |
| `prompt_builder.py` | `PromptContext` input record and deterministic request builder. Change prompt content here without adding provider calls. |
| `config.py` | Configuration defaults and validation shared by loading and initialization. Keep new settings and incompatible combinations explicit here. |
| `history.py` | Evidence-selection helpers for recent records and best anchors; consumes snapshots without modifying the study. |
| `phase_policy.py` | Pure budget-to-guidance helper; does not enforce runner stopping conditions. |
| `objective_renderer.py` | Pure helper explaining supported scoring expressions using known context; does not compute scores or measured limits. |
| `generation.py` | Proposal orchestration helper for request logging, response parsing, repair, and random fallback. |
| `ledger.py` | Single-writer filesystem persistence helper; distinguishes proposals returned from proposals accepted by the runner. |
| [`tools/rationale_migration.py`](tools/README.md) | Explicit maintenance tool for a stopped SQLite study's rationale contract; outside ordinary generation. Read its preconditions before use. |
| `__init__.py` | Convenience exports for generation callers; the runner adapter remains in `component.py`. |

The main external contracts are the shared `OptimizerABC` lifecycle and proposal
records, the provider protocol, the raw `SearchSpaceDescriptor`, and the candidate
JSON envelope. History selection, phase guidance, objective prose, and ledger
serialization are supporting helpers, not separate optimizer interfaces.

## Extend and verify

1. Identify the boundary above. A new provider implements `complete_json` and
   returns normalized content/provenance; it also needs explicit factory selection
   and configuration validation before it can be selected. A model parameter
   change starts with the family's descriptor, not a provider-specific schema.
2. Keep candidate legality and duplicate detection local. For prompt or acceptance
   changes, update `prompt_version` or schema/optimizer identity where appropriate
   so resume does not mix incompatible experiment contracts.
3. Add a focused case in `test/test_llm_*.py`: valid output, the relevant invalid
   output, and repair/fallback or propagation behavior. Use fake fixtures for
   repeatability and check provenance as well as returned parameter values.
4. Run the relevant checks from the repository root, for example:

   ```sh
   python -m pytest -q test/test_llm_candidate_validation.py test/test_llm_generation.py test/test_llm_provider_prompt_ledger.py test/test_llm_proposal_contract.py
   ```

A change is done when the generation example still returns legal proposals,
focused tests cover the changed contract, requests/rejections/returned proposals
remain auditable, and the runner retains sole ownership of queueing and evaluation.
