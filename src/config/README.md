# Config Reference

This directory documents the runtime configuration surface for TinyODOM-EX.

For the source architecture and extension map, see
[`../README.md`](../README.md).

The current shipped config examples are:

- [`nas_config.yaml`](nas_config.yaml)
- [`nas_config_ble.yaml`](nas_config_ble.yaml)
- [`nas_config_portenta.yaml`](nas_config_portenta.yaml)
- [`nas_config_audio_stm32.yaml`](nas_config_audio_stm32.yaml)
- [`nas_config_audio_portenta.yaml`](nas_config_audio_portenta.yaml)

Use [`nas_config.yaml`](nas_config.yaml) as the default starting point for the
repo. It is the main STM32-oriented example config and the most complete
reference for the current score/prune surface. Use the BLE and Portenta files
when you want board-specific starting points for those Arduino-backed targets.
Use [`nas_config_audio_stm32.yaml`](nas_config_audio_stm32.yaml) for the
desktop-first UrbanSound8K / DS-CNN audio path before moving into STM32 HIL
work. Use [`nas_config_audio_portenta.yaml`](nas_config_audio_portenta.yaml)
for the Phase 8 Arduino audio path on Portenta H7 CM7.

Audio analysis runners live under [`../../analysis_scripts`](../../analysis_scripts).
They measure classifier inference over precomputed log-mel feature tensors; they
do not include firmware-side microphone capture or audio feature extraction.

Phase 9 adds optional audio final fold-rotation reporting through:

- `task.params.evaluation.protocol: fixed_split | fold_rotation`
- `task.params.evaluation.fold_rotation.test_folds`, defaulting to all 10
  UrbanSound8K folds.
- `dataset.params.fold_rotation_cache_dir`, required only when fold rotation is
  enabled.

The fixed `dataset.params.cache_dir` remains the source for NAS, HIL smoke, and
deployable export. Fold rotation runs after the fixed-split final checkpoint and
does not export per-fold models.
`task.params.evaluation.protocol: fold_rotation` is single-objective only;
multi-objective NAS fails during config validation. The dataset owns where
fold-specific caches live; the task owns whether final reporting rotates
through those folds.

The runtime loader and validator live in
[`../tinyodom/model.py`](../tinyodom/model.py), while the task-aware bootstrap
that resolves components and validates NAS policy against the active task lives
in [`../tinyodom/runtime_bootstrap.py`](../tinyodom/runtime_bootstrap.py).

## Current Shape

The main top-level blocks in the current config surface are:

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

Those component blocks are resolved by
[`../tinyodom/component_selection.py`](../tinyodom/component_selection.py), and
they are now mandatory. The older top-level `data` block is no longer part of
the supported config contract.

## `device`

The `device` block owns target selection and runtime behavior.

Common keys:

- `device.name`
  Target device identifier.
- `device.hil`
  Enables or disables hardware-in-the-loop measurement.
- `device.runtime_mode`
  `back_to_back` or `cadenced`.
- `device.latency_budget_ms`
  Optional shared cadence-budget override. When omitted, the runtime derives
  it from the active dataset cadence: first `dataset.params.batch_period_ms`
  when present, then legacy
  `dataset.params.stride / dataset.params.sampling_rate_hz * 1000` for the
  built-in `oxiod` dataset.
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

Per-backend nested blocks currently include:

- `device.portenta.*`
- `device.stm32.*`

The current STM32 option plumbing is resolved by
[`../tinyodom/microcontrollers/__init__.py`](../tinyodom/microcontrollers/__init__.py).
Examples of STM32-owned keys currently supported in code include:

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

Important current caveats:

- `device.runtime_mode` must be `back_to_back` or `cadenced`.
- `device.stm32.runtime_mode` is no longer supported. Use
  `device.runtime_mode` instead.
- `device.stm32.project_layout` is no longer supported. LRUN `dev_boot` is
  implicit for the current STM32 backend.
- `device.latency_budget_ms` must be positive when set.
- `device.measured_inference_runs` must be an integer `>= 1`.
- `device.cpu_clock_mhz_options` must be a non-empty integer list when set.
- For `STM32_NUCLEO_N657X0_Q`, `device.cpu_clock_mhz_options` is validated
  against the backend-supported set in code.
- For `PORTENTA_H7` and `ARDUINO_NANO_33_BLE_SENSE`, cadenced mode currently
  requires `training.input_mode: uniform`.

## `training`

The `training` block still owns the main NAS/training runtime switches.

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

Current runtime behavior:

- `training.energy_aware` defaults to `false` when omitted
- `training.quantization` is required and must use the mapping shape:
  `mode`, `search`, and non-empty `choices`. Most shipped configs fix
  `mode: int8_ptq`, `search: false`, and `choices: [int8_ptq]`; the audio
  STM32 config intentionally searches `choices: [float, int8_ptq]`.
- Supported v1 quantization modes are `float` and `int8_ptq`. Enabling
  `training.quantization.search: true` samples `quantization_mode` from
  `choices`; this expands the effective NAS search space and usually needs a
  larger trial budget. Mixed `float`/`int8_ptq` studies also conflate
  architecture quality with quantization effects, so compare them deliberately.
- HIL metrics are deployment-mode preflight metrics collected before training.
  Per-trial NAS scoring evaluates the trained checkpoint with host-side TFLite
  on the validation split; final fixed-split reporting exports/evaluates the
  trained TFLite on the test split after `train_best_trial`.
- Closeout artifacts may still be Keras-derived in this phase unless a specific
  path explicitly requests TFLite evaluation.
- `training.input_mode` defaults to `uniform` when omitted
- `training.input_mode` supports dataset-agnostic `uniform` plus
  dataset-specific analysis modes: `oxiod_representative`, `oxiod_real`,
  `urbansound8k_representative`, and `urbansound8k_real`
- `training.max_total_trials` defaults to `training.nas_trials * 2` when
  omitted

## `dataset`, `task`, and `model`

The modular component-selection surface is resolved by
[`../tinyodom/component_selection.py`](../tinyodom/component_selection.py).

Current keys:

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

Current caveats:

- `dataset`, `task`, and `model` are required top-level blocks
- `dataset.params` is required for the built-in `oxiod` dataset path
- dataset classes are instantiated as zero-argument classes
- model family classes are instantiated as zero-argument classes
- task classes are expected to use the explicit keyword-only constructor
  contract `__init__(*, checkpoint_path, early_stopping_patience)`; the runtime
  no longer probes constructor signatures and does not provide backward-
  compatibility shims for older task classes

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

The `nas` block owns scoring and pruning policy.

Current structure:

- `nas.score.type`
  `scoring-function` or `multi-objective`
- `nas.score.metrics`
  Optional derived metrics
- `nas.score.params`
  Score terms or objectives depending on `score.type`
- `nas.prune.rules`
  Optional pre-training hard-reject rules

Built-in derived metric types currently documented by the shipped config and
validated in code include:

- `add`
- `energy-budget-from-power`

Current scalar term types include:

- `weighted`
- `normalized-weighted`
- `boundary`
- `target`

Current practical guidance:

- use `scoring-function` when you want one scalar score and config-driven
  prune rules
- use `multi-objective` when you want a Pareto front instead of one scalar
- keep non-HIL configs away from score/prune terms that require measured
  latency or energy
- in cadenced multi-objective runs, overload remains telemetry rather than an
  automatic prune

The most readable examples remain in
[`nas_config.yaml`](nas_config.yaml) itself.

## `outputs`

The `outputs` block controls directory roots and naming inputs.

Current shipped keys:

- `outputs.models_dir`
- `outputs.candidate_dir`
- `outputs.artifact_stem`
- `outputs.log_file_name`

Important runtime caveat:

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

Current shipped keys:

- `network.host`
- `network.port`
- `network.recv_timeout_sec`
- `network.send_timeout_sec`

These must match the HIL client/server deployment you actually run.

## `logging`

The `logging` block currently exposes:

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

- [`nas_config.yaml`](nas_config.yaml) for the main commented example
- [`../tinyodom/model.py`](../tinyodom/model.py) for validation and derived
  runtime behavior
- [`../README.md`](../README.md) for source architecture
- [`../tinyodom/model_families/README.md`](../tinyodom/model_families/README.md)
  for model-family selection and extension
- [`../tinyodom/microcontrollers/README.md`](../tinyodom/microcontrollers/README.md)
  for backend-owned hardware options
