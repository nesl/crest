# Generation examples

Start with a local generation check, then adapt the campaign fragments below.
The full-history fragment uses a live provider. For credentials and launch steps,
see the [package guide](README.md#run-an-experiment).

## Try generation without running a campaign

From the repository root, in an environment with CREST's dependencies installed,
run the following example. It validates two configured candidates and writes a
temporary ledger. It does
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

Expected result: two raw mappings matching the assertions, followed by the
ledger filenames. No trials are created.

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
mappings, relative to the YAML file. Warm-ups run before provider calls. See the
[configuration guide](../../../config/README.md) for warm-up and resume settings.

## Full-history audio DS-CNN

Merge this block into a complete `audio_dscnn` experiment. The existing dataset,
score, deployment, training, and budget settings still apply.

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

A valid response for an initial exploration slot, with no extra runtime search
fields active, is:

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

The initial prompt assigns exploration. Later prompts can assign a different
coverage cell, so this response is not a fixture for every round. Supply every
active descriptor field, including runtime fields when enabled. For a local slot,
set `intent` to `local` and `parent_trial` to an integer from the prompt's eligible
parents, and supply a new candidate with complete parameters.

Expected result: accepted proposals enter the normal trial queue and evaluation
path. The example response is valid only when it matches the prompt's assigned
coverage cell and active descriptor. See [full-history generation](README.md#full-history-generation)
for duplicate, repair, and refill behavior.
