<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->

# Config reference

This directory documents the runtime configuration surface for CREST.

For the source architecture and extension map, see [`../README.md`](../README.md).

Case-study reproducibility configs live under [`case_study_configs/`](case_study_configs/).

## Example configs

| If You Want To... | Start With | Hardware Requirement |
|-------------------|------------|----------------------|
| Train or search OxIOD models without hardware energy metrics | [`nas_config_flops_rmse.yaml`](nas_config_flops_rmse.yaml) | No hardware required |
| Run STM32 OxIOD HIL experiments | [`nas_config_stm32.yaml`](nas_config_stm32.yaml) | NUCLEO-N657X0-Q board required. HIL harness required for energy measurement |
| Run Arduino OxIOD HIL experiments | [`nas_config_ble.yaml`](nas_config_ble.yaml) or [`nas_config_portenta.yaml`](nas_config_portenta.yaml) | Target board required. HIL harness required for energy measurement |
| Run UrbanSound8K audio HIL experiments | [`nas_config_audio_stm32.yaml`](nas_config_audio_stm32.yaml) or [`nas_config_audio_portenta.yaml`](nas_config_audio_portenta.yaml) | Target board and HIL harness required as shipped. Use these as templates with HIL/measured scoring disabled for hardware-free training |

The STM32 example is the most complete commented score/prune reference.
[`nas_config_memory_proxy.yaml`](nas_config_memory_proxy.yaml) adds a desktop RMSE/static-memory-traffic example.

## Supporting files

- [`stm32_nucleo_mypool.json`](stm32_nucleo_mypool.json) ST Edge AI memory-pool description used by STM32 configs when `device.stm32.weight_storage_mode: external_flash` is enabled.
  This is a repo-local version modified from the STM32 LRUN example for the NUCLEO-N657X0-Q externalized-weights flow.

## Audio fold rotation

Audio configs can optionally run final reporting across UrbanSound8K folds.

- `task.params.evaluation.protocol: fixed_split | fold_rotation`
- `task.params.evaluation.fold_rotation.test_folds`, defaulting to all 10 UrbanSound8K folds.
- `dataset.params.fold_rotation_cache_dir`, required only when fold rotation is enabled.

The fixed `dataset.params.cache_dir` remains the source for NAS, HIL runs, and deployable export.
Fold rotation runs after the fixed-split final checkpoint and does not export per-fold models.
`task.params.evaluation.protocol: fold_rotation` is single-objective only.
Multi-objective NAS fails during config validation.
The dataset owns where fold-specific caches live.
The task owns whether final reporting rotates through those folds.

## Top-Level blocks

The main top-level blocks are as follows.

- `device` Hardware target, HIL runtime behavior, timing, harness options, and backend-owned device options.
- `dataset` Required dataset selection block.
  The built-in OxIOD dataset uses `dataset.name: oxiod` plus the keys under `dataset.params`.
- `task` Required task selection block.
  Task-owned final reporting policy, such as audio fold rotation, lives under `task.params.evaluation`.
- `model` Required model-family selection block.
- `training` NAS and training limits plus runtime-side training switches such as `energy_aware` and `input_mode`.
- `nas` Scoring and pruning policy.
- `outputs` Output directories and derived artifact naming inputs.
- `network` HIL server/client socket settings.
- `logging` Runtime log level.
- `optimizer` Optional candidate source selection.
  Omit it, or set `type: optuna`, to use the existing Optuna sampler path.
  Set `type: llm_generator` to propose candidates with an LLM.

`dataset`, `task`, and `model` are required configuration blocks.

## `device`

The `device` block owns target selection and runtime behavior.

Common keys

- `device.name` Target device identifier.
- `device.hil` Enables or disables hardware-in-the-loop measurement.
- `device.compile_when_hil_disabled` Controls whether `device.hil: false` still calls the HIL server for compile-only proxy metrics.
  Valid values are `auto`, `true`, and `false`.
  YAML booleans are accepted and normalized to `true`/`false`.
  The default `auto` compiles only when the active score/prune policy references compile-derived metrics such as `ram_bytes`, `flash_bytes`, `external_flash_bytes`, or `arena_bytes`.
- `device.runtime_mode` `back_to_back` selects continuous execution, while `cadenced` selects periodic execution.
- `device.latency_budget_ms` Optional shared cadence-budget override.
  When omitted, the runtime derives it from the active dataset cadence, first `dataset.params.batch_period_ms`, then `dataset.params.stride / dataset.params.sampling_rate_hz * 1000` for the built-in `oxiod` dataset.
- `device.serial_port` DUT serial port.
- `device.measured_inference_runs` Number of repeated inference invokes averaged into one measured pass.
- `device.serial_timeout_s`
- `device.dut_ready_timeout_s`
- `device.cpu_clock_mhz_options` Optional per-trial CPU presets for boards that support runtime clock selection.

Harness-related keys

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

Per-backend nested blocks include the following.

- `device.portenta.*`
- `device.stm32.*`

See the [STM32 backend guide](../crest/microcontrollers/README.md#shipped-config-shape) for setup and supported options.
STM32 settings include the following.

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

Validation notes

- `device.runtime_mode` must be `back_to_back` or `cadenced`.
- The STM32 backend uses the LRUN `dev_boot` layout.
- `device.latency_budget_ms` must be positive when set.
- `device.measured_inference_runs` must be an integer `>= 1`.
- `device.cpu_clock_mhz_options` must be a non-empty integer list when set.
- For `STM32_NUCLEO_N657X0_Q`, `device.cpu_clock_mhz_options` is validated against the backend-supported set in code.
- For `PORTENTA_H7` and `ARDUINO_NANO_33_BLE_SENSE`, cadenced mode requires `training.input_mode: uniform`.

## `training`

`training.random_seed` optionally resets Python, NumPy and TensorFlow to the same integer seed before each candidate evaluation and final fit.
Acceptable values are integers in `[0, 2**32)`.
Omission preserves unseeded behavior.
The seed is independent of trial number and `optimizer.llm.random_seed`, which controls local proposal fallback sampling.
`training.deterministic_ops: true` additionally asks TensorFlow for deterministic operations and requires a training seed.
It is process-wide and may reject unsupported kernels.
Fixed seeds alone do not promise bitwise-identical GPU results.

The configured randomness policy is recorded in trial attributes and optimizer identity, with a `.randomness.json` sidecar for final fits.
Changing the policy requires a fresh study.
Other training settings are not covered by this identity check, so retain the original configuration when extending an experiment.

The `training` block owns the main NAS/training runtime switches.

Common keys in the shipped config

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

Runtime behavior

- `training.energy_aware` defaults to `false` when omitted
- `training.quantization` is required and must use the mapping shape with `mode`, `search`, and non-empty `choices`.
  The main measured-board configs search `choices: [float, int8_ptq]` so float32 and int8 PTQ exports can be compared on the same backend.
- Supported quantization modes are `float` and `int8_ptq`.
  Enabling `training.quantization.search: true` samples `quantization_mode` from `choices`.
  This expands the effective NAS search space and usually needs a larger trial budget.
  Mixed `float`/`int8_ptq` studies also conflate architecture quality with quantization effects, so compare them deliberately.
- `int8_ptq` export uses full-integer TensorFlow Lite conversion with int8 inputs and outputs.
  Representative calibration data is taken from the explicit calibration split when available, otherwise the training split, and conversion samples up to 1000 windows evenly across that split.
  This avoids calibrating on one contiguous prefix of an ordered time-series split.
- HIL metrics are deployment-mode preflight metrics collected before training.
  Per-trial NAS scoring evaluates the trained checkpoint with host-side TFLite on the validation split.
  Final fixed-split reporting evaluates the selected trained TFLite model on the test split.
- Some closeout artifact paths are Keras-derived unless the path explicitly requests TFLite evaluation.
- `training.input_mode` defaults to `uniform` when omitted
- `training.input_mode` supports dataset-agnostic `uniform` plus dataset-specific analysis modes including `oxiod_representative`, `oxiod_real`, `urbansound8k_representative`, and `urbansound8k_real`
- `training.max_total_trials` defaults to `training.nas_trials * 2` when omitted
- `training.nas_trials` is the total COMPLETE trial target, or the total feasible COMPLETE target when `nas.feasibility.rules` is enabled.
  Without feasibility, penalized multi-objective COMPLETE trials also count toward that target.
- `training.max_total_trials` is a total attempt cap.
  All stored trials, including failed, pruned, infeasible, RUNNING, and reserved WAITING trials, consume it.
  Constrained runs usually need a larger attempt cap than the completion target.
  See [optimizer resume behavior](#optimizer) before extending a study.

## `dataset`, `task`, and `model`

Keys

- `dataset.name`
- `dataset.params`
- `task.name`
- `task.params`
- `model.family`
- `model.params`
- `model.search`

For `audio_dscnn`, `model.search: {}` means "use the model-family default search surface" described in the [audio model-family guide](../crest/model_families/audio_dscnn/README.md).
Add keys under `model.search` only when you want to narrow that default surface.

Validation notes

- `dataset`, `task`, and `model` are required top-level blocks
- `dataset.params` is required for the built-in `oxiod` dataset path
## `optimizer`

`optimizer.type` defaults to `optuna`.
Use `llm_generator` for LLM proposals.
Custom registered optimizers can supply their own fields without an `llm` block.
See the [optimizer interface and extension guide](../crest/optimizers/README.md).
The YAML blocks below are merge snippets for a complete experiment configuration.

```yaml
optimizer:
  type: optuna
```

See [training budgets](#training) for completion targets and attempt caps.
To extend a study, retain its name, database, and original experiment configuration and raise these totals.
Proposal settings, sampler configuration, and feasibility policy must match on resume.
Budgets, output paths, and transport timeouts may change.
The optimizer signature does not fingerprint the whole experiment.
Budget-only extension is the supported use.
See [budgets and resume](../crest/optimizers/README.md#budgets-and-resume) for queue handling, compatibility, smoke tests, and recovery.

### LLM settings

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

| Setting | Default and requirements |
| --- | --- |
| `provider` | `openrouter`. Also supports `openai_compatible`, `openai_responses`, and `fake`. |
| `model` | Required for live providers. Use a model available to your account. |
| `base_url` | OpenRouter defaults to `https://openrouter.ai/api/v1`. Generic Chat Completions and Responses providers require it explicitly. No URL userinfo, query parameters, or fragments. |
| `api_key_env` | OpenRouter defaults to `OPENROUTER_API_KEY`. Generic Chat Completions and Responses providers require an explicit environment-variable name. Live credentials are required at initialization. |
| `json_response_mode` | `true` for OpenRouter and Responses. `false` for generic Chat Completions. Enable only when the endpoint/model supports it. This setting is part of resume identity. |
| `temperature` | Default 0.4 for Chat Completions, `null` for Responses. Set `null` to omit it. |
| `batch_size` | Default 5. Positive integer (at most 5 in full-history mode). |
| `timeout_s` | Default 60 seconds. Transport timeout, excluded from resume identity. |
| `prompt_version` | Default `v1`. Nonempty version label, part of resume identity. |
| `max_repair_attempts` | Default 1. Additional content-repair calls after the initial generation request. |
| `random_seed` | Default 0. Controls local fallback randomness, not the remote provider. |
| `recent_trial_window` | Default 10. Non-negative integer, with 0 disabling recent records. Independent of batch size and memory timing. |
| `anchor_count` | Default 5. Non-negative integer, with 0 disabling best-trial anchors. |
| `semantic_context` | Default `true`. `false` omits explanatory context while retaining the legal search descriptor and configured trial evidence. |
| `responses` | Required nonempty ordered fixture list for `fake`. Include summary fixtures in request order when memory is enabled. |
| `extra_headers` | Optional provider headers. Excluded from study identity, as are credentials. |
| `max_transport_retries` | Default 5. Excluded from study identity. |
| `retry_backoff_s` | Default 5 seconds. Excluded from study identity. |
| `retry_backoff_cap_s` | Default 60 seconds per wait. Excluded from study identity. |

`base_url` is the API root.
Chat Completions adapters append `/chat/completions`, and the Responses adapter appends `/responses`.

Use complete raw descriptor parameters such as `dilations_index` and `cpu_clock_mhz_index`, not decoded architecture fields.
Include deployment fields only when active for the experiment.
See [validation and failures](../crest/optimizers/llm/README.md#validation-and-failures) for response validation and fallback, [run artifacts](../crest/optimizers/llm/README.md#inspect-a-run) for saved evidence, and the [provider interface](../crest/optimizers/llm/README.md#completion-providers) for authentication, transport behavior, and retry handling.

### Responses transport and history modes

For `provider: openai_responses`, set `base_url`, `model`, and `api_key_env` explicitly.
Output limits use `max_output_tokens`.
Optional `reasoning_effort` and `service_tier` must be supported by the configured endpoint and model.

`history_mode` defaults to `bounded`.
`history_mode: full` is supported only for `model.family: audio_dscnn` and requires `require_rationale: true`, `memory.enabled: false`, `batch_size` from 1 through 5, and `max_repair_attempts` at most 1.
Full mode accepts `max_refill_attempts` from 0 through 2 (default 2).
That option is invalid in bounded mode.
See [full-history generation](../crest/optimizers/llm/README.md#full-history-generation) for policy behavior and response examples.

`require_rationale` defaults to `false`.
When enabled, each candidate has the shape `{"params": {...}, "rationale": "..."}` with a nonempty rationale of at most 2000 characters.
Rationale is proposal metadata and does not change scoring.
Changing identity-bearing settings requires a new study.

### Output-token and spending limits

`max_output_tokens` optionally limits each completion, including repair and memory calls.
`output_token_parameter` selects `max_completion_tokens` (default) or `max_tokens` for Chat Completions.
Choose a field supported by the endpoint and model that covers billable completion/reasoning tokens.
These limits are part of resume identity.
Omitting them preserves uncapped completion behavior.

<a id="configure-pricing-and-a-spending-cap"></a>

#### Pricing catalogs and spending caps

Add this optimizer block to a complete experiment configuration.

```yaml
optimizer:
  type: llm_generator
  llm:
    provider: openai_responses
    base_url: https://api.openai.com/v1
    model: gpt-6-luna
    api_key_env: OPENAI_API_KEY
    max_usd: 5
    max_input_tokens: 100000
    max_output_tokens: 2000
    # pricing_file: prices.json  # Optional custom catalog, relative to this YAML.
```

A spending cap requires `max_usd`, `max_input_tokens`, `max_output_tokens`, and matching prices.
Explicit inline prices remain supported.
The bundled [`pricing.json`](../crest/optimizers/llm/providers/pricing.json) covers OpenAI `gpt-6-luna`.
Its conservative rates can exceed the actual bill.
For another model, endpoint, or service tier, copy that file and set `pricing_file`.
A custom catalog replaces the bundled one.
Entries must match provider, base URL, model, and tier (`default` when omitted).

`max_usd: 0` permits fixed warm-ups without paid requests.
Capped studies save their selected rates in `llm_optimizer/spend_state.json`.
Retain that file when resuming and increase `max_usd` to continue.
Model, rates, pricing provenance, and token limits must match.
Removing an existing cap or enabling one over existing request logs without spend state is rejected.
Failed or interrupted requests retain their reservation because they may have been billed.
Recorded charges are conservative estimates, not provider invoices.
Without `max_usd`, spending is not tracked.

### Fixed warm-up and recovery

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

`warmup_params_file` must contain a nonempty ordered JSON list of complete raw parameter dictionaries.
Relative paths resolve from the YAML directory.
`~` expands to the home directory.
Set `recent_trial_window` at least as large as the warm-up to expose all attempts in the first LLM prompt.
Warm-up consumes normal attempt/completion budgets and may stop partway through.
Moving or reformatting the file is allowed.
Changing its candidates requires a new study.
Credentials are still checked at initialization.

`optimizer.recover_orphaned_trials` is a list of trial numbers, defaulting to `[]`.
After confirming the selected workers have stopped, supply exact trial numbers, for example `[12]`.
Only listed RUNNING trials become FAIL.
Terminal numbers are harmless, while unknown or WAITING numbers reject the request.
Remove the list afterward.
See [budgets and resume](../crest/optimizers/README.md#budgets-and-resume).

### Accumulated experimental memory

Memory is enabled by default.
Set `optimizer.llm.memory.enabled: false` to disable summary calls while retaining configured recent trials and anchors.
Memory uses the configured provider/model and incurs separate API usage.

| Memory setting | Default and requirements |
| --- | --- |
| `start_after_trials` | 10 eligible terminal trials (COMPLETE, FAIL, or PRUNED). Positive integer. |
| `interval_trials` | 5 additional eligible terminal trials. Positive integer, independent of the recent window. |
| `lag_trials` | Default 0. Non-negative number of newest terminal trials withheld from summarization. |
| `max_findings` | 12 structured findings. |
| `max_chars` | 6000 characters, not tokens. |
| `max_pending_trials` | 100. Must be at least both timing thresholds. |

Reduced bounds must still fit saved findings.
Resume requires compatible saved context and covered evidence.
Incompatible changes require a new study/output directory.
Use one writer per study.
See [evidence and memory](../crest/optimizers/llm/README.md#evidence-and-memory) for summary timing, rejection/suspension handling, audit artifacts, and recovery.

Minimal dataset/task/model example to accompany an optimizer block.

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

Structure

- `nas.score.type` `scoring-function` or `multi-objective`
- `nas.score.metrics` Optional derived metrics
- `nas.score.params` Score terms or objectives depending on `score.type`
- `nas.prune.rules` Optional post-build/pre-fit hard termination gates
- `nas.feasibility.train_if_infeasible` Whether trials that violate feasibility constraints should still train and return real objective values.
  Defaults to `false`.
- `nas.feasibility.rules` Optional post-build/pre-fit deployability constraints using the same rule shape as `nas.prune.rules`, with fields `rule`, `metric`, `condition`, `reference`, and `reason`.

Built-in derived metric types validated in code include the following.

- `add`
- `energy-budget-from-power`

Scalar term types include the following.

- `weighted`
- `normalized-weighted`
- `boundary`
- `target`

Practical guidance

- use `scoring-function` when you want one scalar score
- use `multi-objective` when you want a Pareto front instead of one scalar
- use `nas.prune.rules` with either score type when a pre-fit metric should hard-stop a trial before feasibility is evaluated.
  Rules run after model build/compile, FLOP counting, and HIL/compile metric collection, but before `task.build_fit_plan`, `model.fit`, TFLite validation, or Keras validation.
  Multi-objective gate hits are logged with `pruned=True` but remain Optuna COMPLETE trials with direction-aware penalty values.
- use `nas.feasibility.rules` for deployability constraints that should be visible to Optuna constrained samplers.
  Feasibility is evaluated after hard failures and `nas.prune.rules`.
  Each rule persists one signed constraint where `<= 0` is feasible and `> 0` is infeasible.
  With `train_if_infeasible: false`, infeasible trials skip training and return penalties while still consuming `training.max_total_trials`.
  With `train_if_infeasible: true`, trials train normally but remain constrained-infeasible for samplers, CSV filtering, and plots.
- move latency/deadline budget gates such as `latency_ms > latency_budget_ms` or `cadenced_deadline_miss_count > 0` to `nas.feasibility.rules` when you want constrained dominance.
  Keep the same metric in `nas.prune.rules` only when you deliberately want hard early termination for debugging or resource protection.
- keep non-HIL configs away from score/prune terms that require measured latency or energy.
  `latency_ms`, energy/power/current/voltage metrics, `clock_hz`, `harness_latency_ms`, and `cadenced_*` metrics require `device.hil: true`.
- set `device.compile_when_hil_disabled: false` for pure desktop scores such as RMSE/FLOPs.
  Leave it as `auto` when non-HIL score/prune terms still need compile-derived resource metrics.
- in cadenced multi-objective runs, overload is telemetry unless you add a `nas.feasibility.rules` constraint such as `cadenced_deadline_miss_count > 0`

The most complete commented examples remain in [`nas_config_stm32.yaml`](nas_config_stm32.yaml) itself.

## `outputs`

The `outputs` block controls directory roots and naming inputs.

Keys

- `outputs.models_dir`
- `outputs.hypervolume_reference_point` (optional).
  Fixed multiobjective progress reference, one finite value per objective in objective order and original units.
  The reference is persisted under the study's `progress/` directory and cannot change on resume.
  Without one, the first eligible trial plus a 10% worse-direction margin supplies a monitoring-only reference.
  See [progress reporting](../crest/optimizers/README.md#progress-and-reproducibility).

- `outputs.candidate_dir`
- `outputs.artifact_stem`
- `outputs.log_file_name`

Artifact names follow `{artifact_stem}_{device.name}`.
Configure `artifact_stem` and the directory roots.
Do not set the derived `model_name` or `checkpoint_name` fields in YAML.

## `network`

The `network` block owns HIL socket settings.

Keys

- `network.host`
- `network.port`
- `network.recv_timeout_sec`
- `network.send_timeout_sec`

These must match the HIL client/server deployment you actually run.

## `logging`

The `logging` block exposes the following.

- `logging.level`

Valid values are listed below.

- `CRITICAL`
- `ERROR`
- `WARNING`
- `INFO`
- `DEBUG`
- `NOTSET`

## Case study configurations

`case_study_configs/` contains portable copies of saved experiment configurations, organized by the current paper's case-study numbering.
Each YAML records its source and original file hash.
Dataset, device, network, output, and firmware paths are adjusted for this checkout, while search objectives, budgets, precision choices, and proposal policies retain their saved values.
Use these for new studies after preparing data and setting device ports using the [getting-started guide](../GETTING_STARTED.md).
The general configs above remain the simpler starting point for a new application.

These configurations do not recreate historical firmware, trained models, or saved measurements.
Use fresh study names and output directories rather than resuming an archived study with changed paths or runtime versions.

### Case study 1 STM32 runtime schedule comparison

All six configurations use OxIOD and the STM32 N657.
Continuous and periodic identify the energy objective, not necessarily the measurement mode.
The 100 and 500 ms searches collect both measurement phases even when scoring continuous energy.

| Release period | Continuous-energy objective | Periodic-energy objective |
| --- | --- | --- |
| 100 ms | [Continuous](case_study_configs/cs1/stm32_continuous_100ms.yaml) | [Periodic](case_study_configs/cs1/stm32_periodic_100ms.yaml) |
| 200 ms | [Continuous](case_study_configs/cs1/stm32_continuous_200ms.yaml) | [Periodic](case_study_configs/cs1/stm32_periodic_200ms.yaml) |
| 500 ms | [Continuous](case_study_configs/cs1/stm32_continuous_500ms.yaml) | [Periodic](case_study_configs/cs1/stm32_periodic_500ms.yaml) |

The 100 and 500 ms campaigns use int8 and a 256-attempt cap.
The older 200 ms configurations search float and int8 and retain their saved 600-attempt cap, although the paper analyzes only the first 256 attempts.
The 200 ms periodic campaign used earlier firmware with per-slot UART output and is excluded from the paper's model and selection analyses.
Current firmware runs cannot reproduce that historical measurement protocol from YAML alone.

### Case Study 2 OxIOD TCN Across Targets

The two proxy searches use mixed float/int8 precision and require no board.
The four measured-energy searches require the corresponding target and measurement harness.

| Experiment | Configuration |
| --- | --- |
| FLOPs proxy | [OxIOD FLOPs](case_study_configs/cs2/oxiod_flops_proxy.yaml) |
| Memory-traffic proxy | [OxIOD memory](case_study_configs/cs2/oxiod_memory_proxy.yaml) |
| Nano 33 BLE Sense measured energy | [BLE](case_study_configs/cs2/oxiod_ble_measured_energy.yaml) |
| Portenta H7 CM4 measured energy | [CM4](case_study_configs/cs2/oxiod_portenta_m4_measured_energy.yaml) |
| Portenta H7 CM7 measured energy | [CM7](case_study_configs/cs2/oxiod_portenta_m7_measured_energy.yaml) |
| STM32 N657 measured energy and clock search | [STM32](case_study_configs/cs2/oxiod_stm32_measured_energy.yaml) |

These are the saved paper campaigns, including their precision choices and budgets.
The generic FLOPs and memory examples elsewhere in this directory use different precision settings.
For cross-target evaluation of selected candidates, use [Pareto HIL replay](../README.md#pareto-hil-replay).

### Case study 3 UrbanSound8K DS-CNN application scoring

All six campaigns score validation macro-F1 and continuous energy using the same application score.
Portenta TPE collects continuous measurements only, while the other campaigns also collect periodic measurements.
Their recorded elapsed times therefore include different measurement work.

| Target | TPE | Memory | History |
| --- | --- | --- | --- |
| Portenta H7 CM7 | [TPE](case_study_configs/cs3/audio_portenta_tpe.yaml) | [Memory](case_study_configs/cs3/audio_portenta_memory.yaml) | [History](case_study_configs/cs3/audio_portenta_history.yaml) |
| STM32 N657 | [TPE](case_study_configs/cs3/audio_stm32_tpe.yaml) | [Memory](case_study_configs/cs3/audio_stm32_memory.yaml) | [History](case_study_configs/cs3/audio_stm32_history.yaml) |

Memory and History use the same [15 warm-up candidates](case_study_configs/cs3/warmup_params.json) before provider calls.
Keep that file beside the YAMLs.
For live credentials and provider access, follow the [LLM guide](../crest/optimizers/llm/README.md#run-an-experiment).
The saved model choice is retained, but provider availability and nondeterministic responses can affect a new run.

## Where to look next

Use these files together.

- [`nas_config_stm32.yaml`](nas_config_stm32.yaml) for the STM32 commented example
- [`../crest/model.py`](../crest/model.py) for validation and derived runtime behavior
- [`../README.md`](../README.md) for source architecture
- [Optimizer system guide](../crest/optimizers/README.md) for the full proposal, evaluation, budget, and resume contract
- [`../crest/model_families/README.md`](../crest/model_families/README.md) for model-family selection and extension
- [`../crest/microcontrollers/README.md`](../crest/microcontrollers/README.md) for backend-owned hardware options
