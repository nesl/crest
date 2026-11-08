# Prompt construction and evidence

This package turns search snapshots into provider-neutral requests. It performs
no provider calls, study writes, training, or hardware evaluation. The
[optimizer component](../component.py) supplies context; [generation](../generation.py)
sends requests and validates responses.

| File | Responsibility |
| --- | --- |
| `prompt_builder.py` | `PromptContext` and `build_candidate_request`: assemble declarations, evidence, objectives, budgets, and repair feedback. |
| `history.py` | Select recent trial evidence and best-trial anchors from snapshots. |
| `objective_renderer.py` | Explain supported scoring expressions using known context; does not calculate measured scores. |
| `phase_policy.py` | Derive exploration/refinement guidance from trial budgets; does not enforce stopping conditions. |

## Use the request builder

The [examples guide](../EXAMPLES.md#try-generation-without-running-a-campaign) has a
complete generation example. Prompt construction can also be exercised directly
using that example's `PromptContext`:

```python
from crest.optimizers.llm.prompting.prompt_builder import build_candidate_request

request = build_candidate_request(context, prompt_version="v1")
print(request.user_prompt)
```

## Extend the prompts

1. Change the module responsible for the evidence or text you need. Keep provider
   transport details out of request construction.
2. Preserve raw parameter declarations and distinguish observed trial evidence
   from suggestions or explanations. Do not introduce credentials into context.
3. Update prompt/optimizer identity when the experiment contract changes, so
   incompatible settings cannot silently resume an existing study.
4. Add focused examples for the new behavior and missing or invalid evidence.

A change is complete when request construction remains deterministic, evidence
selection tests pass, and generation still validates all returned candidates.
From the repository root, run:

```sh
python -m pytest -q test/test_llm_provider_prompt_ledger.py test/test_llm_history.py test/test_llm_anchors.py test/test_llm_objective_renderer.py test/test_llm_semantic_context.py
```
