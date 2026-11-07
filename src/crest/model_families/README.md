<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->

# Model Families

Model families define trainable and exportable model architectures for CREST.

Hardware requirement: no hardware is required to implement model construction,
search-space validation, or export materialization. Use a target board and HIL
harness only when validating an exported model through a hardware backend.

For the broader source map, see [`../../README.md`](../../README.md). If you
are working on board support, toolchains, flashing, or runtime measurement, use
[`../microcontrollers/README.md`](../microcontrollers/README.md). The
[optimizer system guide](../optimizers/README.md) explains how family
declarations feed shared evaluation across native and explicit proposers.

## What This Layer Owns

Model families own:

- architecture search knobs, sampled hyperparameters, and optional raw search
  descriptors for explicit proposers;
- model-family-local config and hyperparameter validation;
- Keras model construction from `ModelBuildContext`;
- optional custom object registration for reloads;
- export-model materialization before HIL/backend staging;
- optional FLOP counting and TFLite support reporting.

Model families do not own dataset loading, task metrics, score policies,
candidate staging, compile/upload flows, or hardware telemetry.

## Shipped Implementations

- [`odom_tcn.py`](odom_tcn.py)
  Registered as `odom_tcn`. Implements the built-in odometry TCN family and
  odometry-specific export/materialization variants.
- [`audio_dscnn/family.py`](audio_dscnn/family.py)
  Registered as `audio_dscnn`. Implements the built-in audio DS-CNN family for
  cached UrbanSound8K log-mel inputs.

## Key Files

- [`../interfaces/model_family.py`](../interfaces/model_family.py)
  Defines `ModelFamilyABC`.
- [`../pipeline_types.py`](../pipeline_types.py)
  Defines `ModelBuildContext`, the normalized build-time input passed into a
  family.
- [`../registry.py`](../registry.py)
  Owns `model_family_registry`.
- [`../builtin_components.py`](../builtin_components.py)
  Registers shipped model families under stable config names.
- [`../component_selection.py`](../component_selection.py)
  Resolves `model.family`, `model.params`, and `model.search` from config.
- [`../../nas_model_client.py`](../../nas_model_client.py)
  Training/NAS-side consumption of sampling, validation, and model
  construction.
- [`../../hil_server.py`](../../hil_server.py)
  Export/HIL-side consumption of `materialize_export_model(...)`.

## Runtime Flow

The model-family path runs after dataset and task selection:

1. Entry points register built-ins through
   [`../builtin_components.py`](../builtin_components.py).
2. [`../component_selection.py`](../component_selection.py) resolves
   `model.family` plus `model.params` and `model.search`.
3. The entry point resolves the class with `model_family_registry.get(...)`
   from [`../registry.py`](../registry.py) and instantiates it.
4. Training/NAS code calls `validate_config(...)`, `sample_hparams(...)`,
   `validate_hparams(...)`, and `build_model(...)`.
5. The selected task compiles and evaluates the model.
6. Export/HIL code calls `materialize_export_model(...)`, validates model
   shape against the active task, and passes the materialized model to the
   selected microcontroller backend.

That boundary matters: model families own architecture and export-oriented
materialization; microcontroller backends own TFLite staging, compile/upload
flow, and device telemetry.

## Add A New Model Family

Use [`odom_tcn.py`](odom_tcn.py) or [`audio_dscnn/family.py`](audio_dscnn/family.py) as the
concrete starting point.

1. Add a new module under [`src/crest/model_families/`](.).
2. Implement a `ModelFamilyABC` subclass from
   [`../interfaces/model_family.py`](../interfaces/model_family.py).
3. Implement the required methods:
   - `sample_hparams(trial, ctx, config)`
   - `build_model(hparams, ctx, config)`
   - For explicit proposers, including LLM generation, also implement
     `trial_search_space(ctx, config)`
     with the exact raw parameters sampled by `sample_hparams(...)`.
4. Keep family-specific logic inside the family:
   - search-space sampling;
   - input-shape interpretation from `ModelBuildContext`;
   - family-local helper composition for builders, decoders, or export
     materialization;
   - family-specific export-model materialization.
5. Register the class in [`../builtin_components.py`](../builtin_components.py)
   with `model_family_registry.register(...)`.
6. Select it through `model.family` in config.
7. Put family-owned knobs under `model.params` and `model.search`.
8. Verify both the training/NAS path and the HIL/export path.

## Contract In Practice

`ModelFamilyABC` exposes these main hooks:

- `name`
  Optional property. Returns the human-readable family identifier; shipped
  families override it with the stable registry key.
- `sample_hparams(trial, ctx, config)`
  Required. Produces normalized model-family hyperparameters for one trial.
- `trial_search_space(ctx, config)`
  Required for explicit proposers such as `optimizer.type: llm_generator`;
  optional for native Optuna sampling. Returns ordered `SearchParam` declarations
  from [`../optimizers/search_space.py`](../optimizers/search_space.py), using
  raw persisted trial names, inclusive numeric bounds, or categorical choices.
  Optional descriptions, units, and typical effects provide prompt semantics.
  The default hook raises `NotImplementedError`. Both shipped families reuse
  these declarations in `sample_hparams(...)` through `SearchParam.suggest(...)`
  so legal ranges and choices stay consistent. Keep decoding inside the family
  (for example, `dilations_index` becomes a dilation schedule); the runner adds
  `quantization_mode` when quantization search and its deployment path are active,
  and `cpu_clock_mhz_index` when compile-derived metrics and configured clock
  options activate that search. Explicit candidates must supply exactly every
  active raw field.
  Import these declarations from `crest.optimizers.search_space`. See the
  [configuration guide](../../config/README.md).
- `build_model(hparams, ctx, config)`
  Required. Builds the uncompiled Keras model from normalized hyperparameters
  and `ModelBuildContext`.
- `validate_config(model_config)` and `validate_hparams(...)`
  Optional but expected for shipped families when config/search spaces have
  constraints.
- `decode_trial_hparams(...)`
  Optional. Reconstructs persisted trial parameters when replay/export needs a
  different build-time shape.
- `default_seed_trial(...)`
  Optional hook for callers that request a seed; the NAS runner does not enqueue
  it automatically.
- `load_model(...)` and `custom_objects()`
  Optional. Used by the default trained-model materialization path.
- `count_flops(...)` and `supports_tflite()`
  Optional. Capability hooks used by NAS scoring and export gating.
- `estimate_static_memory(...)`
  Optional. Provides the static tensor-memory proxy used by memory-oriented
  scoring or replay comparisons.
- `materialize_export_model(...)`
  Optional but important for HIL. Selects the model variant prepared for a
  backend.

The default materialization path supports:

- `untrained`
- any variant whose name starts with `trained`

[`odom_tcn.py`](odom_tcn.py) extends the base behavior to also handle:

- `approx_trained`
- `representative`
- `bn_full_plus_non_bn_bias_perturbed`

Those extra variants are odometry-specific export/materialization modes used
for controlled backend preparation. New families do not need to support those
names unless they intentionally need the same materialization behavior.

## Registration And Config Selection

Model families must be registered explicitly; they are not discovered from files
in this directory. Follow the [extension steps](#add-a-new-model-family) above.

Families receive the `model` configuration subtree, with options under
`model.params` and search controls under `model.search`.

## Minimal Model Family Example

This standalone example builds a single-output regression model. Run it from
`src/` with the project's TensorFlow and Optuna dependencies installed. The
example deliberately accepts one target shape; a production family should
validate every supported task/input contract and expose its own config knobs.

```python
import optuna
import tensorflow as tf

from crest.interfaces import ModelFamilyABC
from crest.optimizers.search_space import SearchParam
from crest.pipeline_types import ModelBuildContext, TargetSpec


class TinyRegressionFamily(ModelFamilyABC):
    @property
    def name(self):
        return "tiny_regression"

    def trial_search_space(self, ctx, config):
        return [SearchParam("units", "int", low=4, high=16)]

    def sample_hparams(self, trial, ctx, config):
        return {p.name: p.suggest(trial)
                for p in self.trial_search_space(ctx, config)}

    def validate_hparams(self, hparams, ctx, config):
        if ctx.target_spec.output_shapes != [(1,)]:
            raise ValueError("tiny_regression requires one scalar output")
        if not 4 <= hparams["units"] <= 16:
            raise ValueError("units must be between 4 and 16")

    def build_model(self, hparams, ctx, config):
        self.validate_hparams(hparams, ctx, config)
        inputs = tf.keras.Input(shape=ctx.input_shape, dtype=ctx.input_dtype)
        hidden = tf.keras.layers.Flatten()(inputs)
        hidden = tf.keras.layers.Dense(hparams["units"], activation="relu")(hidden)
        output = tf.keras.layers.Dense(
            1, name=ctx.target_spec.output_names[0]
        )(hidden)
        return tf.keras.Model(inputs, output)


ctx = ModelBuildContext(
    input_shape=(8, 3),
    input_dtype="float32",
    target_spec=TargetSpec("regression", ["value"], [(1,)]),
)
family = TinyRegressionFamily()
model_config = {"params": {}, "search": {}}
family.validate_config(model_config)
hparams = family.sample_hparams(optuna.trial.FixedTrial({"units": 8}), ctx, model_config)
model = family.build_model(hparams, ctx, model_config)
assert model.output_shape == (None, 1)
export_model = family.materialize_export_model(
    hparams, ctx, model_config, model_variant="untrained"
)
assert export_model.output_shape == model.output_shape
```

The model remains uncompiled: the selected task supplies losses and metrics.
The `untrained` export hook builds a new model with fresh weights; use a trained
variant and a checkpoint for trained-model export. Register the class explicitly
in `builtin_components.py`, then select it with this config fragment:

```yaml
model:
  family: tiny_regression
  params: {}
  search: {}
```

This fragment selects only the family. A runnable config also needs a compatible
dataset, task, and the runner's remaining configuration blocks.

## Contributor Checklist

- Keep architecture/search logic in the model family.
- Keep board/toolchain logic in the microcontroller backend.
- Validate family-local config and hyperparameters before export.
- For explicit proposer support, verify that descriptor names/types/choices match
  persisted `sample_hparams(...)` parameters, including any narrowed `model.search`
  options.
- Keep export/materialization decisions explicit; do not hide backend-specific
  behavior in the family.
- Add or update NumPy-style docstrings for changed functions, classes,
  dataclasses, and tests.
- Add hardware-free tests for model construction and export request generation
  wherever possible before running HIL validation.

## Definition Of Done

A model family is ready when:

1. `sample_hparams(...)` and `build_model(...)` work for the configured search
   space.
2. Config and sampled hyperparameters are validated locally.
3. Built models match the active task target spec.
4. Export/materialization behavior is explicit and tested.
5. The family is registered and selectable with `model.family`.
6. Hardware-free tests cover construction, validation, and export preflight
   behavior.
7. HIL validation is run only after the software path is covered.

## Troubleshooting

If model-family bring-up fails, check these layers in order:

1. Config:
   Does `model.family` match the registered key and do `model.params` /
   `model.search` match the family contract?
2. Target spec:
   Does the model output shape match the selected task?
3. Search:
   Are sampled hyperparameters normalized before `build_model(...)` receives
   them?
4. Export:
   Does `materialize_export_model(...)` return the intended trained,
   untrained, or family-specific variant?
5. Backend handoff:
   Are board-specific failures coming from the microcontroller backend rather
   than hidden model-family assumptions?
