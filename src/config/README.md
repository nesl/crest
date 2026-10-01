<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->

# Config Reference

This directory documents the runtime configuration surface for CREST.

For the source architecture and extension map, see
[`../README.md`](../README.md).

Case-study reproducibility configs live under
[`case_study_configs/`](case_study_configs/).

## Example Configs

| If You Want To... | Start With | Hardware Requirement |
|-------------------|------------|----------------------|
| Train or search OxIOD models without hardware energy metrics | [`nas_config_flops_rmse.yaml`](nas_config_flops_rmse.yaml) | No hardware required |
| Run STM32 OxIOD HIL experiments | [`nas_config_stm32.yaml`](nas_config_stm32.yaml) | NUCLEO-N657X0-Q board required; HIL harness required for energy measurement |
| Run Arduino OxIOD HIL experiments | [`nas_config_ble.yaml`](nas_config_ble.yaml) or [`nas_config_portenta.yaml`](nas_config_portenta.yaml) | Target board required; HIL harness required for energy measurement |
| Run UrbanSound8K audio HIL experiments | [`nas_config_audio_stm32.yaml`](nas_config_audio_stm32.yaml) or [`nas_config_audio_portenta.yaml`](nas_config_audio_portenta.yaml) | Target board and HIL harness required as shipped; use these as templates with HIL/measured scoring disabled for hardware-free training |

- [`nas_config_stm32.yaml`](nas_config_stm32.yaml)
  NUCLEO-N657X0-Q OxIOD measured-energy NAS example and the most complete
  commented reference for score/prune policy shapes.
- [`nas_config_ble.yaml`](nas_config_ble.yaml)
  Arduino Nano 33 BLE Sense OxIOD measured-energy NAS example.
- [`nas_config_portenta.yaml`](nas_config_portenta.yaml)
  Portenta H7 CM7 OxIOD measured-energy NAS example.
- [`nas_config_audio_stm32.yaml`](nas_config_audio_stm32.yaml)
  NUCLEO-N657X0-Q UrbanSound8K / DS-CNN audio NAS example.
- [`nas_config_audio_portenta.yaml`](nas_config_audio_portenta.yaml)
  Portenta H7 CM7 UrbanSound8K / DS-CNN audio NAS example.
- [`nas_config_flops_rmse.yaml`](nas_config_flops_rmse.yaml)
  Pure desktop OxIOD NAS example that optimizes validation RMSE and FLOPs.
- [`nas_config_memory_proxy.yaml`](nas_config_memory_proxy.yaml)
  Pure desktop OxIOD NAS example that optimizes validation RMSE and static
  memory traffic.

## Supporting Files

- [`stm32_nucleo_mypool.json`](stm32_nucleo_mypool.json)
  ST Edge AI memory-pool description used by STM32 configs when
  `device.stm32.weight_storage_mode: external_flash` is enabled. This is a
  repo-local version modified from the STM32 LRUN example for the
  NUCLEO-N657X0-Q externalized-weights flow.

## Audio Fold Rotation

Audio configs can optionally run final reporting across UrbanSound8K folds:

- `task.params.evaluation.protocol: fixed_split | fold_rotation`
- `task.params.evaluation.fold_rotation.test_folds`, defaulting to all 10
  UrbanSound8K folds.
- `dataset.params.fold_rotation_cache_dir`, required only when fold rotation is
  enabled.

The fixed `dataset.params.cache_dir` remains the source for NAS, HIL runs, and
deployable export. Fold rotation runs after the fixed-split final checkpoint and
does not export per-fold models.
`task.params.evaluation.protocol: fold_rotation` is single-objective only;
multi-objective NAS fails during config validation. The dataset owns where
fold-specific caches live; the task owns whether final reporting rotates
through those folds.

The runtime loader and validator live in
[`../crest/model.py`](../crest/model.py), while the task-aware bootstrap
that resolves components and validates NAS policy against the active task lives
in [`../crest/runtime_bootstrap.py`](../crest/runtime_bootstrap.py).

## Top-Level Blocks

The main top-level blocks are:

- `device`
  Hardware target, HIL runtime behavior, timing, harness options, and
  backend-owned device options.
- `dataset`
  Required dataset selection block. The built-in OxIOD dataset uses
  `dataset.name: oxiod` plus the keys under `dataset.params`.
- `task`
  Required task selection block. Task-owned final reporting policy, such as
  audio fold rotation, lives under `task.params.evaluation`.
- `model`
  Required model-family selection block.
- `training`
  NAS and training limits plus runtime-side training switches such as
  `energy_aware` and `input_mode`.
- `nas`
  Scoring and pruning policy.
- `outputs`
  Output directories and derived artifact naming inputs.
- `network`
  HIL server/client socket settings.
- `logging`
  Runtime log level.
- `optimizer`
  Optional candidate source selection. Omit it, or set `type: optuna`, to use
  the existing Optuna sampler path. Set `type: llm_generator` to validate and
  enqueue exact candidates before the unchanged CREST objective runs.

Those component blocks are resolved by
[`../crest/component_selection.py`](../crest/component_selection.py), and
they are mandatory. Use `dataset`, `task`, and `model`; the old top-level
`data` block is not supported.

## `device`

The `device` block owns target selection and runtime behavior.

Common keys:

- `device.name`
  Target device identifier.
- `device.hil`
  Enables or disables hardware-in-the-loop measurement.
- `device.compile_when_hil_disabled`
  Controls whether `device.hil: false` still calls the HIL server for
  compile-only proxy metrics. Valid values are `auto`, `true`, and `false`;
  YAML booleans are accepted and normalized to `true`/`false`. The default
  `auto` compiles only when the active score/prune policy references
  compile-derived metrics such as `ram_bytes`, `flash_bytes`,
  `external_flash_bytes`, or `arena_bytes`.
- `device.runtime_mode`
  `back_to_back` or `cadenced`.
- `device.latency_budget_ms`
  Optional shared cadence-budget override. When omitted, the runtime derives
  it from the active dataset cadence: first `dataset.params.batch_period_ms`,
  then `dataset.params.stride / dataset.params.sampling_rate_hz * 1000` for
  the built-in `oxiod` dataset.
- `device.serial_port`
  DUT serial port.
- `device.measured_inference_runs`
  Number of repeated inference invokes averaged into one measured pass.
- `device.serial_timeout_s`
- `device.dut_ready_timeout_s`
- `device.cpu_clock_mhz_options`
  Optional per-trial CPU presets for boards that support runtime clock
  selection.

Harness-related keys:

- `device.harness_serial_port`
- `device.harness_fqbn`
- `device.harness_auto_flash`
- `device.harness_arm_pin`
- `device.harness_trigger_pin`
- `device.dut_arm_hold_ms`
- `device.harness_stable_low_ms`
- `device.harness_ready_timeout_s`
- `device.harness_arm_timeout_s`
- `device.harness_active_timeout_s`
- `device.harness_done_timeout_s`

Per-backend nested blocks include:

- `device.portenta.*`
- `device.stm32.*`

STM32 option plumbing is resolved by
[`../crest/microcontrollers/__init__.py`](../crest/microcontrollers/__init__.py).
Examples of STM32-owned keys include:

- `template_root`
- `project_root`
- `gdbserver`
- `gdb`
- `cubeprog_bin`
- `signing_tool`
- `gdb_port`
- `apid`
- `server_ready_timeout_s`
- `wake_margin_us`
- `min_sleep_us`
- `weight_storage_mode`
- `appli_flash_address`
- `weights_flash_address`
- `weights_memory_pool`
- `weights_external_loader`
- `signing_load_offset`
- `signing_header_version`
- `max_external_flash_bytes`

Validation notes:

- `device.runtime_mode` must be `back_to_back` or `cadenced`.
- `device.stm32.runtime_mode` is no longer supported. Use
  `device.runtime_mode` instead.
- `device.stm32.project_layout` is no longer supported. LRUN `dev_boot` is
  implicit for the STM32 backend.
- `device.latency_budget_ms` must be positive when set.
- `device.measured_inference_runs` must be an integer `>= 1`.
- `device.cpu_clock_mhz_options` must be a non-empty integer list when set.
- For `STM32_NUCLEO_N657X0_Q`, `device.cpu_clock_mhz_options` is validated
  against the backend-supported set in code.
- For `PORTENTA_H7` and `ARDUINO_NANO_33_BLE_SENSE`, cadenced mode requires
  `training.input_mode: uniform`.

## `training`

The `training` block owns the main NAS/training runtime switches.

Common keys in the shipped config:

- `training.nas_epochs`
- `training.model_epochs`
- `training.nas_trials`
- `training.nas_multiobjective_population_size`
- `training.max_total_trials`
- `training.quantization`
- `training.latency_proxy_max_flops`
- `training.train`
- `training.energy_aware`
- `training.input_mode`

Runtime behavior:

- `training.energy_aware` defaults to `false` when omitted
- `training.quantization` is required and must use the mapping shape:
  `mode`, `search`, and non-empty `choices`. The main measured-board configs
  search `choices: [float, int8_ptq]` so float32 and int8 PTQ exports can be
  compared on the same backend.
- Supported quantization modes are `float` and `int8_ptq`. Enabling
  `training.quantization.search: true` samples `quantization_mode` from
  `choices`; this expands the effective NAS search space and usually needs a
  larger trial budget. Mixed `float`/`int8_ptq` studies also conflate
  architecture quality with quantization effects, so compare them deliberately.
- `int8_ptq` export uses full-integer TensorFlow Lite conversion with int8
  inputs and outputs. Representative calibration data is taken from the
  explicit calibration split when available, otherwise the training split, and
  conversion samples up to 1000 windows evenly across that split. This avoids
  calibrating on one contiguous prefix of an ordered time-series split.
- HIL metrics are deployment-mode preflight metrics collected before training.
  Per-trial NAS scoring evaluates the trained checkpoint with host-side TFLite
  on the validation split using interpreter-provided input/output
  scale/zero-point values and TFLite signature output order; final fixed-split
  reporting exports/evaluates the trained TFLite on the test split after
  `train_best_trial`.
- Some closeout artifact paths are Keras-derived unless the path explicitly
  requests TFLite evaluation.
- `training.input_mode` defaults to `uniform` when omitted
- `training.input_mode` supports dataset-agnostic `uniform` plus
  dataset-specific analysis modes: `oxiod_representative`, `oxiod_real`,
  `urbansound8k_representative`, and `urbansound8k_real`
- `training.max_total_trials` defaults to `training.nas_trials * 2` when
  omitted
- `training.nas_trials` is the target number of feasible completed trials when
  `nas.feasibility.rules` is enabled. Infeasible, failed, and pruned attempts
  still count against `training.max_total_trials`, so constrained hardware
  runs usually need a larger total-attempt budget than the feasible target.

## `dataset`, `task`, and `model`

The modular component-selection surface is resolved by
[`../crest/component_selection.py`](../crest/component_selection.py).

Keys:

- `dataset.name`
- `dataset.params`
- `task.name`
- `task.params`
- `model.family`
- `model.params`
- `model.search`

For `audio_dscnn`, `model.search: {}` means "use the model-family default
search surface" from `AudioDSCNNFamily.AUDIO_DSCNN_SEARCH_CHOICES`. Add keys
under `model.search` only when you want to narrow that default surface.

Validation notes:

- `dataset`, `task`, and `model` are required top-level blocks
- `dataset.params` is required for the built-in `oxiod` dataset path
- dataset classes are instantiated as zero-argument classes
- model family classes are instantiated as zero-argument classes
- task classes are expected to use the explicit keyword-only constructor
  contract `__init__(*, checkpoint_path, early_stopping_patience)`; the runtime
  does not probe constructor signatures or provide compatibility shims for
  older task classes.

## `optimizer`

Optimizer names are resolved through `optimizer_registry` after lazy built-in
registration. Names strip surrounding whitespace and preserve the registry's
case-sensitive custom keys. An exact registration takes precedence; when none
exists, case variants of the historical built-in names `optuna` and
`llm_generator` resolve to those lowercase keys. The resolved registration name
is stored in campaign identity and proposal provenance. Register custom classes
before calling `load_config()`; their custom optimizer fields are preserved,
and they need no `llm` block. Only
`llm_generator` receives the LLM-specific compatibility defaults and validation.
Registration and config validation do not read credentials or create clients.

Each component receives the whole normalized `optimizer` block. Implement
`OptimizerABC` with a zero-argument constructor and `propose_round`, and override
`validate_config` and `identity_config` for configurable behavior. Identity
settings must be deterministic JSON-safe plain data with no secrets or paths.
An example plugin can use the existing execution path without importing Optuna:

```python
from crest.interfaces import OptimizerABC
from crest.pipeline_types import CandidateProposal, ExplicitRound
from crest.registry import optimizer_registry
from crest.model import load_config

class FixedWidth(OptimizerABC):
    def validate_config(self, config):
        if type(config.get("width")) is not int:
            raise ValueError("optimizer.width must be an integer")

    def identity_config(self, config):
        return {"width": config.width}

    def initialize(self, context, config):
        self.params = {"width": config.width}
        # Supply every raw field declared by context.search_space in a real run.

    def propose_round(self, history, budget):
        return ExplicitRound((CandidateProposal(self.params),))

optimizer_registry.register("fixed_width", FixedWidth)
config = load_config("config.yaml")
```

```yaml
optimizer:
  type: fixed_width
  width: 8
```

Native sampling does not require a model-family descriptor. Explicit proposals
require one and must supply exactly all active raw fields. The runner validates
every candidate before queueing any candidate; it owns proposal provenance in
namespaced trial user attributes and executes the same objective for all modes.

Study resume requires the same registered optimizer, proposal identity, and
native sampler configuration. Completion/attempt budgets and output paths can
change without changing identity. `training.nas_trials` is the **total completed
trial target** (feasible completions when feasibility is enabled), and
`training.max_total_trials` is the **total attempt cap**, including failed,
pruned, infeasible and RUNNING trials. To add evaluations, keep the same study
name, database and original experiment configuration, then increase the total
target and cap as needed. These are not counts of additional trials.

A signed native or LLM study resumes with unchanged optimizer identity when these
budgets increase. Sampler signatures now use stable public TPE/NSGA-II names;
the exact old private paths emitted by the first contract implementation remain
compatible when all other identity fields match.

Unsigned native studies require an explicit one-time opt-in:

```yaml
optimizer:
  type: optuna
  adopt_legacy_study: true
training:
  nas_trials: 300       # Example: raise a completed target of 250 to 300.
  max_total_trials: 350 # Total attempts, with room for failures/pruning.
```

Set the flag only after checking that the active workload, model, device,
schedule, score/feasibility policy and sampler settings match the original
native experiment. This is your attestation of the original configuration;
CREST cannot prove an unsigned history was native or certify its provenance.
The default is false. Adoption rejects WAITING trials, recorded
explicit-proposer provenance, any trial with Optuna's `fixed_params` enqueue
metadata (even an empty dictionary), and an existing per-study `llm_optimizer`
artifact path. Older explicit proposals may have no user-attribute provenance;
these markers conservatively prevent labeling that history as native. Adoption
also validates stored ordered study directions, existing feasibility signatures
and required COMPLETE feasibility evidence before storing the optimizer signature.
These adoption-only guards do not block matching signed native or LLM resumes.
It preserves every existing trial number, state, parameter, value and attribute.
RUNNING trials keep consuming the attempt cap; adoption does not recover them.

Remove `adopt_legacy_study` after successful adoption. It is excluded from study
identity, so later extensions use the normal signed resume check. The flag
never overrides an existing optimizer or feasibility signature mismatch.
Unsigned LLM and arbitrary plugin histories cannot be adopted through this
option; use a new study name. Do not label a historical LLM study as native to
make it resume.

The default remains the existing Optuna path:

```yaml
optimizer:
  type: optuna
```

The LLM generator uses an OpenRouter/OpenAI-compatible endpoint:

```yaml
optimizer:
  type: llm_generator
  llm:
    provider: openrouter
    model: openai/gpt-5-mini
    base_url: https://openrouter.ai/api/v1
    api_key_env: OPENROUTER_API_KEY
    batch_size: 5
    temperature: 0.4
    timeout_s: 60
    json_response_mode: true
    max_repair_attempts: 1
    prompt_version: v1
    random_seed: 0
    recent_trial_window: 10
    anchor_count: 5
    semantic_context: true
    memory:
      enabled: true
      start_after_trials: 10
      interval_trials: 5
      max_findings: 12
      max_chars: 6000
      max_pending_trials: 100
    extra_headers: {}
```

The generator emits raw Optuna parameters such as `dilations_index` and
`cpu_clock_mhz_index`; decoded fields are not accepted. `quantization_mode`
and CPU clock search appear only when their corresponding runtime paths are
active. The required JSON envelope is `{"candidates": [{...}]}`. Local validation
checks exact keys, types, bounds/choices, batch size, and duplicates against
descriptor-complete COMPLETE trials and earlier accepted candidates in the
batch. Penalized COMPLETE trials count for duplicate detection but do not
qualify as anchors. The shared runner drains waiting work before fresh proposals;
WAITING parameters are never interpreted as already sampled evidence.
Any accepted candidates run through the existing CREST objective; a partially
accepted batch is evaluated without filling its rejected slots. A batch with no
accepted candidates gets at most `max_repair_attempts` additional calls (default
1). After all attempts fail, the component logs and returns one random candidate from
the same descriptor. If 100 fallback samples cannot pass the same validation,
generation stops explicitly. `random_seed` controls this local fallback
RNG; it does not seed the remote provider.

Requests/responses live in `models/<study_name>/llm_optimizer/requests/` as
numbered `.request.json` / `.response.json` pairs. The same directory's parent
contains `prompt_contexts.jsonl`, `returned_candidates.jsonl`,
`rejected_candidates.jsonl`, and `optimizer_events.jsonl`. Responses retain raw
provider output, usage when supplied, finish reason, and latency.
`returned_candidates.jsonl` records `returned_to_runner`, which means validated
proposals were returned, not that the queue accepted them. Successful queue
acceptance is evidenced only by study trial attributes. Historical
`accepted_candidates.jsonl` and `batch_enqueued` artifacts retain their original
historical meaning and remain readable. API keys are
read from the configured environment variable and are not included in the
serialized requests or normal responses.
`provider: openrouter` defaults `api_key_env` to `OPENROUTER_API_KEY`.
Generic `provider: openai_compatible` endpoints require `api_key_env` to be
set explicitly so they never inherit an OpenRouter credential name silently.
`base_url` must be a clean endpoint without URL userinfo, query parameters, or
fragments; the transport appends `/chat/completions` to it. Use `api_key_env` or
`extra_headers` for authentication. Extra headers and credentials are excluded
from the study identity.
The MVP transport is `/chat/completions`. OpenRouter defaults
`json_response_mode` to `true`, which sends
`response_format: {type: json_object}`. Generic `openai_compatible` providers
default it to `false` because that option is not universally supported; set it
to `true` only when the selected endpoint and model accept JSON response mode.
The runtime always sends `temperature` and does not expose a provider output-token
cap. `provider: fake` uses a required non-empty `responses` list to exercise the
same validation/ledger path without network access; memory summary responses must
be included when memory is enabled.

Each prompt includes attempted and feasible-trial budgets. Its phase guidance is
`exploration` below 35% of the total-attempt budget, `specification` below 85%,
and `finalization` afterward. These are prompt instructions; CREST still evaluates
candidates with the configured objective and feasibility rules.

`recent_trial_window` bounds compact records read directly from recent Optuna
trials and included in the next prompt. It defaults to 10; set it to 0 to
disable that recent window. `batch_size` controls candidates per generation call;
`recent_trial_window` controls detailed recent records. Both are independent of
`memory.interval_trials`, which controls when accumulated evidence is summarized.
`anchor_count` bounds persistent best-candidate records in every generation prompt.
It defaults to 5 and must be a non-negative integer; set it to 0 to disable anchors.
Anchors are selected directly from the full study, independently of the recent
window and memory. Only COMPLETE trials with finite objective values and no
recorded failure, pruning, or infeasibility are eligible; when feasibility rules
are active, recorded status must be `feasible`. Scalar studies retain the best
trials according to the study direction. Multi-objective studies retain the feasible nondominated front; if it
exceeds the limit, normalize each objective's front values to [0, 1], accounting
for its direction, and keep candidates nearest the knee. For two objectives, the
knee has the greatest positive distance toward the ideal from the line joining
the objective extremes. When no positive bend exists (including a flat front),
an objective has zero span, or there are more than two objectives, the candidate
closest to the normalized ideal is used as a compromise proxy.
The knee and its nearest neighbors are selected using normalized Euclidean
distance. Trial numbers break ties deterministically. Records reuse the recent
history format, including exact raw parameters and available measurements.
Repeated trial numbers across prompt sections are the same evidence.

`semantic_context` defaults to `true`. It adds a deterministic, versioned,
hashed summary of the normalized dataset/input contract, task outputs and
metrics, objective and feasibility policy, model family, known device
capacities and deployment choices, and runtime flags. Parameter descriptions,
units, and typical architectural effects augment the existing search-space
schema; those effects are priors, and measured CREST/HIL observations take
precedence. Set `semantic_context: false` for an ablation that retains legal
parameter names/ranges/choices, recent trial history, best-candidate anchors,
and trial-budget state.
Paths, serial ports, credential values, dataset examples, and unfiltered config
trees are not included.


### Accumulated experimental memory

For `llm_generator`, memory is enabled by default; set `optimizer.llm.memory.enabled:
false` to disable accumulated memory while retaining configured recent trials
and anchors. The same configured provider/model makes separate JSON summarization
calls. This adds API usage once the start threshold is reached. Fake-provider
fixtures must interleave candidate and summary responses in the same order; the
two response schemas are different.

The default first summary is created once ten **terminal trials** (COMPLETE, FAIL,
or PRUNED) exist, before the next candidate-generation call. Later summaries are
created after five additional terminal trials. Thresholds are checked at generation
batch boundaries, not in the middle of evaluating a batch. Partial accepted batches
therefore change the timing. A final summary is not requested when no more candidates
are needed. `start_after_trials` and `interval_trials` are independent positive
integers; changing the recent-trial window does not change either setting.

Every summary update receives the previous structured findings, new trial records,
and the static task/model/deployment context. The LLM returns updates to stable
finding IDs, with observation, conditions, evidence, exceptions, uncertainty, and
audit trial numbers. Findings must carry their evidence in readable prose, not just
references to unavailable trials. Unmentioned findings persist. Replacing a finding
must explicitly retain still-relevant caveats. Retirement requires an ID and reason;
old snapshots remain available. The generator receives a paragraph for each finding,
rendered without another LLM rewrite, plus the recent detailed trials.

Any unsummarized trials outside the recent window appear as `pending_trials`.
Thus an interval of 20 with a recent window of 10 does not silently lose the
intervening observations. Disabling memory intentionally removes this bridge.
`max_findings` and `max_chars` bound accepted memory; `max_chars` is a character
limit on the structured memory and its rendered prose, **not a tokenizer count or
an API output-token cap**. Oversized/malformed summaries are rejected, never
truncated. The model's factual conclusions are not automatically proven correct.

`max_pending_trials` bounds the number of new records in each summary call and the
unsummarized backlog permitted before another generation call. It must be at least
both timing thresholds. On resume, old pending results are summarized in bounded
chunks. On provider/schema failure, the last valid memory stays in use and pending
results remain visible; the next generation boundary tries again. If that backlog
exceeds the limit and cannot be summarized, generation stops with an actionable
error instead of silently discarding evidence or indefinitely expanding the prompt.
A reduced memory size on resume must still fit the saved findings.

Artifacts live under `models/<study>/llm_optimizer/memory/`:

- `state.json`: atomically replaced current memory, version, context hash, and hashes
  of covered trial records. Coverage IDs/hashes remain on disk, not in every prompt.
- `snapshots/`: prior committed memory versions, including subsequently retired findings.
- `requests/`: summary requests/responses, including provider usage and latency.
- `optimizer_events.jsonl`: commits, failures, rejections, and retirement reasons.

Resume verifies the context and previously covered trial records. A changed context
or incompatible study fails explicitly; use a new study/output directory. This is a
single-writer workflow. It does not provide concurrent-worker locking. Summary
quality, loss of nuance during revisions, and live provider behavior require empirical
validation; schema and provenance checks cannot guarantee semantic faithfulness.

The input-token estimator can count these summary request files separately and
combine them with generation estimates; see
[`analysis_scripts/llm_token_cost/README.md`](../../analysis_scripts/llm_token_cost/README.md).

Minimal example:

```yaml
dataset:
  name: oxiod
  params:
    directory: "data/oxiod/"
    sampling_rate_hz: 100
    window_size: 200
    stride: 20
    calibration_windows: 10000

task:
  name: odometry_regression
  params:
    early_stopping_patience: 40

model:
  family: odom_tcn
  params:
    export_variant: approx_trained
  search: {}
```

## `nas`

The `nas` block owns scoring, pruning, and constrained-feasibility policy.

Structure:

- `nas.score.type`
  `scoring-function` or `multi-objective`
- `nas.score.metrics`
  Optional derived metrics
- `nas.score.params`
  Score terms or objectives depending on `score.type`
- `nas.prune.rules`
  Optional post-build/pre-fit hard termination gates
- `nas.feasibility.train_if_infeasible`
  Whether trials that violate feasibility constraints should still train and
  return real objective values. Defaults to `false`.
- `nas.feasibility.rules`
  Optional post-build/pre-fit deployability constraints using the same rule
  shape as `nas.prune.rules`: `rule`, `metric`, `condition`, `reference`, and
  `reason`.

Built-in derived metric types validated in code include:

- `add`
- `energy-budget-from-power`

Scalar term types include:

- `weighted`
- `normalized-weighted`
- `boundary`
- `target`

Practical guidance:

- use `scoring-function` when you want one scalar score
- use `multi-objective` when you want a Pareto front instead of one scalar
- use `nas.prune.rules` with either score type when a pre-fit metric should
  hard-stop a trial before feasibility is evaluated. Rules run after model
  build/compile, FLOP counting,
  and HIL/compile metric collection, but before `task.build_fit_plan`,
  `model.fit`, TFLite validation, or Keras validation. Multi-objective gate
  hits are logged with `pruned=True` but remain Optuna COMPLETE trials with
  direction-aware penalty values.
- use `nas.feasibility.rules` for deployability constraints that should be
  visible to Optuna constrained samplers. Feasibility is evaluated after hard
  failures and `nas.prune.rules`; each rule persists one signed constraint
  where `<= 0` is feasible and `> 0` is infeasible. With
  `train_if_infeasible: false`, infeasible trials skip training and return
  penalties while still consuming `training.max_total_trials`. With
  `train_if_infeasible: true`, trials train normally but remain
  constrained-infeasible for samplers, CSV filtering, and plots.
- move latency/deadline budget gates such as `latency_ms > latency_budget_ms`
  or `cadenced_deadline_miss_count > 0` to `nas.feasibility.rules` when you
  want constrained dominance. Keep the same metric in `nas.prune.rules` only
  when you deliberately want hard early termination for debugging or resource
  protection.
- keep non-HIL configs away from score/prune terms that require measured
  latency or energy. `latency_ms`, energy/power/current/voltage metrics,
  `clock_hz`, `harness_latency_ms`, and `cadenced_*` metrics require
  `device.hil: true`.
- set `device.compile_when_hil_disabled: false` for pure desktop scores such
  as RMSE/FLOPs; leave it as `auto` when non-HIL score/prune terms still need
  compile-derived resource metrics.
- in cadenced multi-objective runs, overload is telemetry unless you add a
  `nas.feasibility.rules` constraint such as
  `cadenced_deadline_miss_count > 0`

The most complete commented examples remain in
[`nas_config_stm32.yaml`](nas_config_stm32.yaml) itself.

## `outputs`

The `outputs` block controls directory roots and naming inputs.

Keys:

- `outputs.models_dir`
- `outputs.candidate_dir`
- `outputs.artifact_stem`
- `outputs.log_file_name`

Runtime notes:

- `load_config(...)` derives read-only runtime fields `model_name` and
  `checkpoint_name` from `outputs.artifact_stem` and `device.name`, then
  populates derived paths such as `tflite_model_path` and `checkpoint_path`
- YAML-authored `outputs.model_name` and `outputs.checkpoint_name` are rejected
  because artifact names now follow the shared `{artifact_stem}_{device.name}`
  rule
- `models_dir` and `candidate_dir` are resolved into absolute paths in memory

So the final in-memory values may differ from the literal YAML text.

## `network`

The `network` block owns HIL socket settings.

Keys:

- `network.host`
- `network.port`
- `network.recv_timeout_sec`
- `network.send_timeout_sec`

These must match the HIL client/server deployment you actually run.

## `logging`

The `logging` block exposes:

- `logging.level`

Valid values are:

- `CRITICAL`
- `ERROR`
- `WARNING`
- `INFO`
- `DEBUG`
- `NOTSET`

## Where To Look Next

Use these files together:

- [`nas_config_stm32.yaml`](nas_config_stm32.yaml) for the STM32 commented example
- [`../crest/model.py`](../crest/model.py) for validation and derived
  runtime behavior
- [`../README.md`](../README.md) for source architecture
- [`../crest/model_families/README.md`](../crest/model_families/README.md)
  for model-family selection and extension
- [`../crest/microcontrollers/README.md`](../crest/microcontrollers/README.md)
  for backend-owned hardware options
