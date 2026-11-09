# Generation examples

These examples use configured responses without calling an LLM. For a live
experiment, start with the [package guide](README.md#run-an-experiment).

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
