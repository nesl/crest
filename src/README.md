# Source Overview

This folder contains the Python entry points and core library code for the
TinyODOM-EX training, hardware-in-the-loop, and deployment flow.

## Score Configuration

Scoring is configured from the top-level `score:` block in the NAS YAML files.
This replaces the old hard-coded `training.nas_multiobjective` behavior.

### Score modes

- `score.type: multi-objective`
  - Optuna receives one objective value per entry in `score.params.objectives`.
  - Each objective must define:
    - `metric`
    - `direction`
  - `direction` must be either `minimize` or `maximize`.

- `score.type: scoring-function`
  - Optuna receives one scalar score.
  - The scalar score is the sum of all terms in `score.params.terms`.
  - Each term must define:
    - `type`
    - `metric`
  - `weight` is optional and defaults to `1.0`.

### Available metric names

Built-in metrics that can be referenced directly from `score.params.objectives`,
`score.params.terms`, or derived metrics:

- `rmse_vel_x`
- `rmse_vel_y`
- `rmse_total`
  - Built-in aggregate metric equal to `rmse_vel_x + rmse_vel_y`.
- `ram_bytes`
- `flash_bytes`
- `max_ram_bytes`
- `max_flash_bytes`
- `external_flash_bytes`
- `flops`
- `latency_ms`
- `energy_mj_per_inference`
- `avg_power_mw`
- `avg_current_ma`
- `bus_voltage_v`
- `cpu_clock_mhz_requested`
- `clock_hz`
- `latency_budget_ms`
- `arena_bytes`
- `error_code`

Notes:

- Metrics with negative sentinel values are treated as unavailable for scoring
  when the metric is expected to be non-negative.
- `energy_mj_per_inference` is usually only available when HIL and energy-aware
  measurement are both enabled.
- `latency_ms` is unavailable in non-HIL proxy runs.
- `cpu_clock_mhz_requested` is optional and backend-dependent.

### Derived metrics

You can define custom derived metrics under `score.metrics`.

The supported derived metric types right now are:

- `add`
- `energy-budget-from-power`

Example:

```yaml
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
- Built-in metrics cannot be redefined inside `score.metrics`.

`energy-budget-from-power` behavior:

- It computes `power_mw * duration_ms / 1000`.
- The result is an energy budget in `mJ`.
- `power` and `duration` both use the same typed reference shape used by scalar
  score terms:
  - metric reference: `{type: metric, metric: latency_budget_ms}`
  - literal reference: `{type: literal, value: 100.0}`
- `duration` must resolve to a value greater than zero.
- `power` must resolve to a non-negative value.

Example:

```yaml
score:
  type: scoring-function
  metrics:
    energy_budget_mj:
      type: energy-budget-from-power
      power:
        type: literal
        value: 100.0
      duration:
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
score:
  type: multi-objective
  params:
    objectives:
      - metric: rmse_total
        direction: minimize
      - metric: latency_ms
        direction: minimize
```

Scalar scoring example:

```yaml
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
      power:
        type: literal
        value: 100.0
      duration:
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
