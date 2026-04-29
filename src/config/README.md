# Config Reference

This directory documents the runtime configuration surface for TinyODOM-EX.

For the source architecture and extension map, see
[`../README.md`](../README.md).

The current shipped config examples are:

- [`nas_config.yaml`](nas_config.yaml)
- [`nas_config_ble.yaml`](nas_config_ble.yaml)
- [`nas_config_portenta.yaml`](nas_config_portenta.yaml)

Use [`nas_config.yaml`](nas_config.yaml) as the default starting point for the
repo. It is the main STM32-oriented example config and the most complete
reference for the current score/prune surface. Use the BLE and Portenta files
when you want board-specific starting points for those Arduino-backed targets.

The runtime loader and validator live in
[`../tinyodom/model.py`](../tinyodom/model.py), especially `load_config(...)`
and the NAS-policy validation helpers it calls.

## Current Shape

The main top-level blocks in the current config surface are:

- `device`
  Hardware target, HIL runtime behavior, timing, harness options, and
  backend-owned device options.
- `data`
  Dataset sampling/preprocessing settings used by the built-in OxIOD path.
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

The modular component-selection blocks also exist:

- `dataset`
- `task`
- `model`

Those are resolved by
[`../tinyodom/component_selection.py`](../tinyodom/component_selection.py).

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
  it from `data.stride / data.sampling_rate_hz * 1000`.
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
- `project_layout`
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
- `device.latency_budget_ms` must be positive when set.
- `device.measured_inference_runs` must be an integer `>= 1`.
- `device.cpu_clock_mhz_options` must be a non-empty integer list when set.
- For `STM32_NUCLEO_N657X0_Q`, `device.cpu_clock_mhz_options` is validated
  against the backend-supported set in code.
- For `PORTENTA_H7` and `ARDUINO_NANO_33_BLE_SENSE`, cadenced mode currently
  requires `training.input_mode: uniform`.

## `data`

The built-in OxIOD dataset path still uses the top-level `data` block by
default.

Current built-in keys used by [`../tinyodom/datasets/oxiod.py`](../tinyodom/datasets/oxiod.py):

- `data.directory`
- `data.sampling_rate_hz`
- `data.window_size`
- `data.stride`
- `data.calibration_windows`

Current caveats:

- these values are required for the built-in `oxiod` dataset path
- `sampling_rate_hz`, `window_size`, and `stride` must be positive
- `calibration_windows` must be positive when set

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
- `training.input_mode` defaults to `uniform` when omitted
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

Current caveats:

- `dataset.params` overrides the top-level `data` block when present
- the shipped default YAML does not yet include a full modular example
- dataset classes are instantiated as zero-argument classes
- model family classes are instantiated as zero-argument classes
- task classes only receive `checkpoint_path` and
  `early_stopping_patience` when those exact constructor kwargs are present

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
  family: tinyodom_tcn
  params: {}
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
- `outputs.tcn_dir`
- `outputs.model_name`
- `outputs.checkpoint_name`
- `outputs.log_file_name`

Important runtime caveat:

- `load_config(...)` derives `model_name` and `checkpoint_name` from
  `device.name`, then populates derived paths such as `tflite_model_path` and
  `checkpoint_path`
- `models_dir` and `tcn_dir` are resolved into absolute paths in memory

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
