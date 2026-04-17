# Source Overview

This folder contains the Python entry points and core library code for the
TinyODOM-EX training, hardware-in-the-loop, and deployment flow.

## NAS Policy Configuration

NAS evaluation policy is configured from the top-level `nas:` block in the NAS
YAML files. This replaces the old hard-coded `training.nas_multiobjective`
behavior.

### Score modes

- `nas.score.type: multi-objective`
  - Optuna receives one objective value per entry in `nas.score.params.objectives`.
  - Each objective must define:
    - `metric`
    - `direction`
  - `direction` must be either `minimize` or `maximize`.

- `nas.score.type: scoring-function`
  - Optuna receives one scalar score.
  - The scalar score is the sum of all terms in `nas.score.params.terms`.
  - Each term must define:
    - `type`
    - `metric`
  - `weight` is optional and defaults to `1.0`.

### Prune policy

There are two kinds of pruning in the NAS flow:

- Automatic pruning
  - Built-in infrastructure checks can stop a trial before training even when
    `nas.prune.rules` is empty.
  - This includes hard failures such as RAM overflow, flash overflow,
    arena-allocation failures, and fatal HIL/controller errors.
  - In practice, RAM overflow is the most common automatic prune: if the model
    does not fit in device memory, the trial is rejected immediately rather
    than spending time on training.
  - Automatic prunes populate `prune_reason` in the CSV and Optuna user attrs.
    They leave `prune_rule` empty because they did not come from a config rule.

- Config-driven pruning
- `nas.prune.rules`
  - Optional list of pre-training hard reject rules.
  - Rules are evaluated after HIL/proxy/resource metrics are collected and
    before model training begins.
  - V1 pruning is supported only when `nas.score.type: scoring-function`.
  - Pruned trials do not submit final objective values to Optuna.
  - `prune_reason` is the human-readable explanation recorded in the CSV and
    trial user attrs.
  - `prune_rule` is the stable machine-readable identifier recorded in the CSV
    and trial user attrs.
  - Supported rule conditions are `gt`, `gte`, `lt`, and `lte`.
  - Rules reuse the same typed `reference` shape as score terms.

Example:

```yaml
nas:
  score:
    type: scoring-function
    params:
      terms:
        - type: weighted
          metric: rmse_total
          weight: -1.0
  prune:
    rules:
      - rule: latency_budget
        metric: latency_ms
        condition: gt
        reference:
          type: metric
          metric: latency_budget_ms
        reason: "Latency exceeds deployment budget"
```

### Available metric names

Built-in metrics that can be referenced directly from `nas.score.params.objectives`,
`nas.score.params.terms`, `nas.prune.rules`, or derived metrics:

- `rmse_vel_x`
  - Validation RMSE for the predicted X-axis velocity component.
- `rmse_vel_y`
  - Validation RMSE for the predicted Y-axis velocity component.
- `rmse_total`
  - Built-in aggregate metric equal to `rmse_vel_x + rmse_vel_y`.
- `ram_bytes`
  - Estimated or measured peak RAM usage for the deployed model on the target.
- `flash_bytes`
  - Program-storage bytes consumed in the target's main/internal flash.
- `max_ram_bytes`
  - Device RAM capacity used for normalized RAM penalties and prune rules.
- `max_flash_bytes`
  - Device flash capacity used for normalized flash penalties and prune rules.
- `external_flash_bytes`
  - Additional model-storage bytes placed in external flash, if the DUT
    supports external weight storage.
- `flops`
  - Estimated floating-point operation count for one inference of the sampled
    architecture.
- `latency_ms`
  - Measured on-device inference latency in milliseconds for the current trial.
- `energy_mj_per_inference`
  - Measured energy in millijoules consumed by one inference attempt.
- `avg_power_mw`
  - Average DUT power during the measured inference window, in milliwatts.
- `avg_current_ma`
  - Average DUT current during the measured inference window, in milliamps.
- `bus_voltage_v`
  - Average measured DUT supply voltage during the inference window, in volts.
- `cpu_clock_mhz_requested`
  - Requested CPU clock preset for the trial, in MHz, when the DUT supports
    per-trial clock selection.
- `clock_hz`
  - Reported DUT CPU clock frequency for the trial, in Hz, when the backend
    provides it.
- `latency_budget_ms`
  - Configured latency budget passed into HIL/proxy evaluation and reusable in
    score terms or prune rules.
- `arena_bytes`
  - Tensor arena size required or reported by the backend for the deployed
    model.
- `error_code`
  - Numeric HIL/controller status code for the trial outcome.

Notes:

- Metrics with negative sentinel values are treated as unavailable for scoring
  when the metric is expected to be non-negative.
- `energy_mj_per_inference` is usually only available when HIL and energy-aware
  measurement are both enabled.
- `latency_ms` is unavailable in non-HIL proxy runs.
- `cpu_clock_mhz_requested` is optional and backend-dependent.
- `weight_storage_mode` is logged as trial metadata, but it is not part of the
  numeric scoring/pruning metric registry.
- If you disable HIL, avoid score terms and prune rules that reference
  `latency_ms` or `energy_mj_per_inference`.

### Derived metrics

You can define custom derived metrics under `nas.score.metrics`.

The supported derived metric types right now are:

- `add`
- `energy-budget-from-power`

Example:

```yaml
nas:
  score:
    type: multi-objective
    metrics:
      model_size_bytes:
        type: add
        metrics:
          - flash_bytes
          - external_flash_bytes
    params:
      objectives:
        - metric: rmse_total
          direction: minimize
        - metric: model_size_bytes
          direction: minimize
```

`add` behavior:

- It sums the listed metrics.
- If any input metric is unavailable, the derived metric is also unavailable.
- Built-in metrics cannot be redefined inside `nas.score.metrics`.

`energy-budget-from-power` behavior:

- It computes `power_mw * duration_ms / 1000`.
- The result is an energy budget in `mJ`.
- `power_mw` and `duration_ms` both use the same typed reference shape used by scalar
  score terms:
  - metric reference: `{type: metric, metric: latency_budget_ms}`
  - literal reference: `{type: literal, value: 100.0}`
- `duration_ms` must resolve to a value greater than zero.
- `power_mw` must resolve to a non-negative value.

Example:

```yaml
nas:
  score:
    type: scoring-function
    metrics:
      energy_budget_mj:
        type: energy-budget-from-power
        power_mw:
          type: literal
          value: 100.0
        duration_ms:
          type: metric
          metric: latency_budget_ms
    params:
      terms:
        - type: target
          metric: energy_mj_per_inference
          weight: 0.15
          reference:
            type: metric
            metric: energy_budget_mj
```

That example creates an energy budget target from a 100 mW power budget and the
configured latency budget. If `latency_budget_ms` is `20.0`, the derived metric
resolves to `2.0 mJ`.

### Scalar term types

There are four scalar term types.

#### 1. `weighted`

This adds `weight * metric_value` to the total score.

Use it when you want a metric to contribute directly to the scalar score.

Example:

```yaml
nas:
  score:
    type: scoring-function
    params:
      terms:
        - type: weighted
          metric: rmse_total
          weight: -1.0
```

That example rewards smaller RMSE because the weight is negative.

#### 2. `normalized-weighted`

This adds `weight * (metric / reference)` to the total score.

Use it when you want a metric to be scaled by a hardware budget or some other
reference value before it contributes to the score.

This is the term to use for things like:

- RAM usage normalized by available RAM
- flash usage normalized by available flash
- latency normalized by a latency budget

Example:

```yaml
nas:
  score:
    type: scoring-function
    params:
      terms:
        - type: normalized-weighted
          metric: ram_bytes
          weight: 0.01
          reference:
            type: metric
            metric: max_ram_bytes
```

That example contributes `0.01 * (ram_bytes / max_ram_bytes)` to the total
score.

The `reference` value must resolve to something greater than zero.

#### 3. `boundary`

This penalizes a metric only when it exceeds a threshold.

Formula:

```text
-weight * max(0, metric - reference)
```

Use it when a metric is acceptable below a limit and should only be penalized
 once it crosses that limit.

Example with a literal threshold:

```yaml
nas:
  score:
    type: scoring-function
    params:
      terms:
        - type: boundary
          metric: latency_ms
          weight: 1.0
          reference:
            type: literal
            value: 50.0
```

Example with another metric as the threshold:

```yaml
nas:
  score:
    type: scoring-function
    params:
      terms:
        - type: boundary
          metric: latency_ms
          weight: 1.0
          reference:
            type: metric
            metric: latency_budget_ms
```

#### 4. `target`

This penalizes distance from a target value.

Formula:

```text
-weight * abs(metric - reference)
```

Use it when the metric should stay close to some ideal point rather than just
 stay below a limit.

Example:

```yaml
nas:
  score:
    type: scoring-function
    params:
      terms:
        - type: target
          metric: latency_ms
          weight: 0.5
          reference:
            type: literal
            value: 20.0
```

### Full examples

Multi-objective example:

```yaml
nas:
  score:
    type: multi-objective
    params:
      objectives:
        - metric: rmse_total
          direction: minimize
        - metric: latency_ms
          direction: minimize
  prune:
    rules: []
```

Scalar scoring example:

```yaml
nas:
  score:
    type: scoring-function
    metrics:
      total_model_flash:
        type: add
        metrics:
          - flash_bytes
          - external_flash_bytes
      energy_budget_mj:
        type: energy-budget-from-power
        power_mw:
          type: literal
          value: 100.0
        duration_ms:
          type: metric
          metric: latency_budget_ms
    params:
      terms:
        - type: weighted
          metric: rmse_total
          weight: -1.0
        - type: normalized-weighted
          metric: ram_bytes
          weight: 0.01
          reference:
            type: metric
            metric: max_ram_bytes
        - type: boundary
          metric: latency_ms
          weight: 1.0
          reference:
            type: metric
            metric: latency_budget_ms
        - type: target
          metric: energy_mj_per_inference
          weight: 0.15
          reference:
            type: metric
            metric: energy_budget_mj
        - type: weighted
          metric: total_model_flash
          weight: -0.000001
  prune:
    rules:
      - rule: latency_budget
        metric: latency_ms
        condition: gt
        reference:
          type: metric
          metric: latency_budget_ms
        reason: "Latency exceeds deployment budget"
```

### Practical limitations

The current scalar scoring DSL is intentionally small.

It does support:

- direct weighted sums
- normalized weighted ratios against a literal or metric reference
- threshold penalties
- target penalties
- simple additive derived metrics
- simple energy-budget derivation from power and duration

It does not support:

- arbitrary nested division between metrics
- min/max clipping beyond the built-in `boundary` behavior
- nested arithmetic expressions beyond `add`
- piecewise formulas more complex than the provided term types

If you need a scalar score that depends on ratios, caps, or multi-step derived
quantities, the config may only be able to approximate the old hard-coded
formula rather than reproduce it exactly.

## Top-level entry points

- `hil_server.py`
  - ZeroMQ REP server that builds model variants, exports TFLite/C++, flashes
    firmware, and returns HIL metrics.
- `nas_model_client.py`
  - ZeroMQ REQ client that runs NAS/training workflows and queries the HIL
    server for hardware metrics.
- `nas_config.yaml`
  - Default configuration for dataset paths, training, score definition,
    device selection, network settings, and output paths.
- `nas_config_ble.yaml`
  - Alternate BLE-oriented configuration.
- `two_board_hil_notes.txt`
  - Notes on the two-board DUT/harness measurement setup.

## Python package

The `tinyodom/` package holds the reusable logic shared by the scripts above.

- `tinyodom/data.py`
  - OxIOD dataset import and split handling.
- `tinyodom/model.py`
  - Model construction, config loading, metric-collection request building, and
    training utilities.
- `tinyodom/hardware.py`
  - Export, compile, upload, and metric normalization helpers.
- `tinyodom/hil_protocol.py`
  - DUT/harness serial handshake and telemetry collection protocol.
- `tinyodom/devices.py`
  - Device abstraction layer and board-spec plumbing.
- `tinyodom/geometry.py`
  - Geometry and trajectory helper functions.
- `tinyodom/errors.py`
  - Shared error-code definitions and helpers.
- `tinyodom/microcontrollers/`
  - Board-specific Arduino and non-Arduino backends.
  - Arduino boards follow the integration guide in
    `src/tinyodom/microcontrollers/README.md`.
  - The STM32 backend is split into:
    - `stm32_nucleo_n657x0.py` for the concrete `DeviceInterface`
      implementation
    - `stm32_cube_clt.py` for build/load/toolchain helpers
    - `stm32_runtime.py` for direct-serial runtime protocol handling

## Notes

- Generated or cache-like artifacts such as `__pycache__/` and
  `tinyodom.egg-info/` may appear here during local development.
- Analysis utilities that sit outside the core runtime live under
  `analysis_scripts/`, not in this folder.
