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

The STM32 example is the most complete commented score/prune reference.
[`nas_config_memory_proxy.yaml`](nas_config_memory_proxy.yaml) adds a desktop
RMSE/static-memory-traffic example.

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

`training.random_seed` optionally resets Python, NumPy and TensorFlow to the same
integer seed before each candidate evaluation and final fit. Acceptable values
are integers in `[0, 2**32)`; omission preserves unseeded behavior. The seed is
independent of trial number and `optimizer.llm.random_seed`, which controls local
proposal fallback sampling. `training.deterministic_ops: true` additionally asks
TensorFlow for deterministic operations and requires a training seed. It is
process-wide and may reject unsupported kernels; fixed seeds alone do not
promise bitwise-identical GPU results.

The configured randomness policy is recorded in trial attributes and optimizer
identity, with a `.randomness.json` sidecar for final fits. Changing the policy
requires a fresh study. Other training settings are not covered by this identity
check, so retain the original configuration when extending an experiment.

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
- `training.nas_trials` is the total COMPLETE trial target, or the total feasible
  COMPLETE target when `nas.feasibility.rules` is enabled. Without feasibility,
  penalized multi-objective COMPLETE trials also count toward that target.
- `training.max_total_trials` is a total attempt cap. All stored trials, including
  failed, pruned, infeasible, RUNNING, and reserved WAITING trials, consume it.
  Constrained runs usually need a larger attempt cap than the completion target.
  See [optimizer resume behavior](#optimizer) before extending a study.

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

The [optimizer system guide](../crest/optimizers/README.md) explains the
proposal contract, shared evaluation, evidence, and restart behavior.
Optimizer components reuse the same objective while making candidate choices
explicit. The YAML blocks below are merge snippets for a complete study config.

`optimizer.type` defaults to `optuna`; use `llm_generator` for explicit LLM
candidates. Custom registered optimizers may supply their own configuration
fields and need no `llm` block. See the [extension guide](../crest/optimizers/README.md)
for registration, interfaces, and a complete plugin example.

Native sampling does not require a model-family descriptor. Explicit proposals
require one and must supply exactly all active raw fields. The runner validates
every candidate before queueing any candidate; it owns proposal provenance in
namespaced trial user attributes and executes the same objective for all modes.

Resume requires the same registered optimizer, proposal identity, native sampler
configuration, and feasibility policy. Keep the study name, database, and original
experiment configuration, then increase `training.nas_trials` (the total COMPLETE
target, or feasible COMPLETE target with feasibility enabled) and
`training.max_total_trials` (the total stored-attempt cap). These are total counts,
not additional trials. Budgets, output paths, and transport `timeout_s` may change;
proposal-affecting settings such as `json_response_mode` remain signed.

The optimizer signature does not fingerprint dataset, model, training, device,
runtime schedule, or score settings. Preserve and compare the original config;
budget-only extension is the supported use. Optimizer switching and searching
nonempty unsigned studies are unsupported; use a new study name. Unsigned
histories remain available for analysis and offline plotting.

The runner stops when the completion target is met, leaving WAITING work untouched.
Otherwise it drains queued work before requesting new candidates, including at the
attempt cap. The cap must cover all stored trials, including FAIL, PRUNED,
infeasible, RUNNING, and WAITING trials. A population round may exceed the remaining
completion target. Orphan RUNNING trials consume attempts until explicitly
recovered (see below). `--smoke-test N` runs N additional attempts in a separate
per-study smoke database. See the [optimizer guide](../crest/optimizers/GUIDE.md)
for the complete queue, identity, and compatibility contract.

The default remains the existing Optuna path:

```yaml
optimizer:
  type: optuna
```

The LLM generator supports OpenRouter, OpenAI-compatible Chat Completions,
and Responses endpoints:

```yaml
optimizer:
  type: llm_generator
  llm:
    provider: openrouter
    model: your-provider-model-id
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

Request/response logs and proposal evidence live under
`models/<study_name>/llm_optimizer/`; see the
[artifact guide](../crest/optimizers/GUIDE.md#inspect-results-finalize-and-replay).
API keys are read from the configured environment variable and excluded from
serialized requests and normal responses.
Nonfake providers require the API key environment variable during component
initialization, before new study signatures are stored. The fake provider
requires no credentials. `provider: openrouter` defaults `api_key_env` to
`OPENROUTER_API_KEY`.
Generic `provider: openai_compatible` endpoints require `api_key_env` to be
set explicitly so they never inherit an OpenRouter credential name silently.
`base_url` must be a clean endpoint without URL userinfo, query parameters, or
fragments; Chat Completions transports append `/chat/completions` to it. Use `api_key_env` or
`extra_headers` for authentication. Extra headers and credentials are excluded
from the study identity.
The `openrouter` and `openai_compatible` transports use `/chat/completions`. OpenRouter defaults
`json_response_mode` to `true`, which sends
`response_format: {type: json_object}`. Generic `openai_compatible` providers
default it to `false` because that option is not universally supported; set it
to `true` only when the selected endpoint and model accept JSON response mode.
Set `temperature: null` to omit temperature from the request. The optional output-token limit below
is sent using the selected provider field. `provider: fake` uses a required non-empty `responses` list to exercise the
same validation/ledger path without network access; memory summary responses must
be included when memory is enabled.

Provider authentication/credit errors stop immediately. Transient HTTP 429/5xx,
network failures and incomplete HTTP reads use bounded retries, then stop if
exhausted. YAML settings `optimizer.llm.max_transport_retries` (default 5),
`retry_backoff_s` (5 seconds), and `retry_backoff_cap_s` (60 seconds per wait)
control transport retry policy without changing study identity. Valid
`Retry-After` seconds or HTTP dates set a minimum wait; if the server requests
more than the configured cap, the study stops instead of retrying early.
Only malformed provider responses or candidate content use repair/random fallback;
transport failures never become random search. Proposal-source counts in the run
summary distinguish LLM candidates, fixed warm-up attempts, and random fallback.
Duplicate filtering includes pruned and known deterministic failed candidates;
infrastructure and unknown failures remain retryable. See the
[optimizer guide](../crest/optimizers/GUIDE.md) for failure classification and
proposal provenance.

### Responses transport and history modes

For a Responses endpoint, select `provider: openai_responses` and set `base_url`,
`model`, and `api_key_env` explicitly. The transport appends `/responses` to the
base URL. Temperature defaults to `null`; output limits use `max_output_tokens`.
Optional `reasoning_effort` and `service_tier` must be supported by the configured
endpoint and model. Prices come from the bundled provider catalog or a custom `pricing_file`.

`history_mode` defaults to `bounded`, using the recent window, anchors, and
optional accumulated memory described below. `history_mode: full` is supported
for `model.family: audio_dscnn` and includes exact columnar trial evidence and
decoded architecture coverage. It requires `require_rationale: true`,
`memory.enabled: false`, `batch_size` between 1 and 5, and `max_repair_attempts`
at most 1. In full mode, `max_refill_attempts` accepts 0 through 2 (default 2);
this option is invalid in bounded mode. The allocation policy and decoded
architecture duplicate checks are described in the
[optimizer guide](../crest/optimizers/GUIDE.md#provider-transports-and-full-history).

`require_rationale` defaults to `false`. Enabling it requires candidates shaped as `{"params": {...}, "rationale": "..."}`
with a nonempty rationale of at most 2000 characters. Rationale is retained as provenance;
it does not alter candidate legality, measured evidence, or scoring.

### Output-token and spending limits

Set `max_output_tokens` to limit each completion, including memory and repair
calls. `output_token_parameter` selects `max_completion_tokens` (default) or
`max_tokens`; verify which field the chosen model accepts and that it covers all
billable completion/reasoning tokens. The limit is part of optimizer identity;
changing it requires a new study. Omitting it preserves existing behavior.

For a monetary cap, supply token limits and select a catalog matching your
provider, endpoint, model, and service tier. Omit `pricing_file` to use the bundled
catalog, or provide a custom JSON file relative to this experiment YAML:

```yaml
optimizer:
  type: llm_generator
  llm:
    # Keep the verified provider/model/authentication settings.
    max_output_tokens: 4096
    output_token_parameter: max_completion_tokens
    max_usd: 10.0
    max_input_tokens: 100000
    pricing_file: prices.json  # Omit when the bundled catalog has a matching tariff.
```

See the [pricing catalog guide](../crest/optimizers/llm/providers/README.md#configure-pricing-and-a-spending-cap)
for the JSON schema and supported bundled rates. The selected rates, source, and
verification date are saved with the study; a changed tariff is rejected on
resume. Existing explicit inline prices remain supported.

`llm_optimizer/spend_state.json` tracks generation, repairs, summaries, and HTTP
retries. Each request reserves its full configured input/output ceilings at the
selected prices; valid usage releases the unused portion. Unknown usage,
transport failures, and crashes retain the reservation, so tracked cost may
exceed the invoice. An observed token overrun stops further requests.

Input size is checked conservatively using serialized message bytes plus 1024
framing tokens; this is not a model-specific tokenizer. Choose a ceiling large
enough for saved prompts. The provider receives the output ceiling. The cap is
local accounting under the configured prices and token assumptions, not a
provider-enforced dollar limit; extra fees or provider violations are not bounded.

When the next reservation cannot fit, NAS/smoke stops cleanly before HTTP or new
candidate enqueue, with no random fallback. `max_usd: 0` allows fixed warm-up
without paid calls. Already queued candidates use no new API call and still run
under the normal evaluation budget. To continue a capped study, retain its state
file and increase `max_usd`; model, prices, provenance and token limits must match.
Removing the cap from a tracked study is rejected. Enabling a cap over existing
request logs without spend state is also rejected: use a new study rather than
silently resetting earlier cost. Use one writer per study, as with the runner.

To give the LLM a fixed measured warm-up, add an ordered JSON parameter file:

```yaml
optimizer:
  type: llm_generator
  llm:
    # Keep the provider/model settings from your LLM configuration.
    warmup_params_file: warmup_params.json
    recent_trial_window: 15
    memory:
      enabled: true
      start_after_trials: 15
      interval_trials: 15
```

The file must contain a nonempty JSON list of complete raw parameter dictionaries
for the active search space. The 15-entry window above is an example; choose
a window large enough to show the complete warm-up. When loaded from YAML, relative paths
resolve from the YAML file directory (and `~` expands to the home directory).
Direct component callers should supply an absolute path. All entries are validated before
execution. One fixed candidate runs per round through the same objective before
any provider request; failures and pruned trials count as warm-up attempts.
Warm-up uses the normal attempt/completion budgets, so a smaller budget can stop
partway through it. Set the recent window to at least the warm-up length to expose all warm-up
attempts in the first LLM prompt.

Warm-up resumes without replaying completed entries. Identity hashes the ordered
file contents: moving or reformatting the file is allowed; changing candidates
requires a new study. Omitting warm-up preserves existing LLM behavior and identity.
Provider configuration and credentials are still checked during initialization.

For orphan recovery, first confirm **the workers for the selected trials have
stopped**, then set `optimizer.recover_orphaned_trials: [12]` with exact Optuna
trial numbers. Only listed RUNNING trials are marked FAIL. Terminal numbers are
harmless; unknown or WAITING numbers reject the request before changes. Recovery
defaults to `[]`; legacy `false` disables it and `true` is rejected. It never
infers that a worker is dead. Remove the list when recovery is finished. See the
[recovery guide](../crest/optimizers/GUIDE.md) for recorded evidence and restart behavior.

`recent_trial_window` bounds compact records read directly from recent Optuna
trials and included in the next prompt. It defaults to 10; set it to 0 to
disable that recent window. `batch_size` controls candidates per generation call;
`recent_trial_window` controls detailed recent records. Both are independent of
`memory.interval_trials`, which controls when accumulated evidence is summarized.
`anchor_count` bounds persistent best-candidate records in every generation prompt.
It defaults to 5 and must be a non-negative integer; set it to 0 to disable anchors.
Anchors come from eligible finite COMPLETE trials across the full study,
independently of the recent window and memory. Scalar studies retain the best
trials; multi-objective studies select from the feasible nondominated front.
Recorded failure, pruning, or infeasibility excludes a trial. With feasibility
rules enabled, a recorded `feasible` status is required. See the
[optimizer guide](../crest/optimizers/GUIDE.md) for selection details.
Repeated trial numbers across prompt sections represent the same evidence.

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
calls. This adds API usage once the start threshold is reached. For the fake provider, interleave candidate and summary responses in request order.

The default first summary is created once ten **terminal trials** (COMPLETE, FAIL,
or PRUNED) exist, before the next candidate-generation call. Later summaries are
created after five additional terminal trials. Thresholds are checked at generation
batch boundaries, not in the middle of evaluating a batch. Partial accepted batches
therefore change the timing. A final summary is not requested when no more candidates
are needed. `start_after_trials` and `interval_trials` are independent positive
integers; changing the recent-trial window does not change either setting.

`max_findings` (default 12) and `max_chars` (6000) bound accepted structured
memory and its rendered prose. `max_chars` is a character limit, not a token
limit. `max_pending_trials` (100) bounds summary chunks and pending records shown
to the generator; it must be at least both timing thresholds. Unsummarized
records outside the recent window appear as bounded pending evidence, with
explicit omission counts. Disabling memory removes this additional window.

Rejected summaries receive one repair attempt. Failed repairs retain the last
valid memory and underlying study evidence; three consecutive deferrals suspend
summary calls across restarts. Provider execution errors stop the study. After
fixing the cause with workers stopped, use the explicit reset procedure in the
[optimizer guide](../crest/optimizers/GUIDE.md#recovery-and-provider-failures).
Memory is fallible guidance, not verified conclusions.

Memory state, snapshots, and request logs live under
`models/<study>/llm_optimizer/memory/`. Resume checks the saved context and covered
trial evidence; incompatible changes require a new study/output directory.
Reduced memory bounds must still fit saved findings. Use one writer per study.
See the [optimizer guide](../crest/optimizers/GUIDE.md) for memory updates,
coverage, audit artifacts, and recovery.

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
- `outputs.hypervolume_reference_point` (optional): fixed multiobjective progress
  reference, one finite value per objective in objective order and original units.
  The reference is persisted under the study's `progress/` directory and cannot
  change on resume. Without one, the first eligible trial plus a 10% worse-direction
  margin supplies a monitoring-only reference. See [progress reporting](../crest/search/README.md).

- `outputs.candidate_dir`
- `outputs.artifact_stem`
- `outputs.log_file_name`

Runtime notes:

- `load_config(...)` derives read-only runtime fields `model_name` and
  `checkpoint_name` from `outputs.artifact_stem` and `device.name`, then
  populates derived paths such as `tflite_model_path` and `checkpoint_path`
- YAML-authored `outputs.model_name` and `outputs.checkpoint_name` are rejected
  because artifact names follow the shared `{artifact_stem}_{device.name}`
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
- [Optimizer system guide](../crest/optimizers/README.md) for the full proposal,
  evaluation, budget, and resume contract
- [`../crest/model_families/README.md`](../crest/model_families/README.md)
  for model-family selection and extension
- [`../crest/microcontrollers/README.md`](../crest/microcontrollers/README.md)
  for backend-owned hardware options
