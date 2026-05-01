# Model Families

This README is the contributor guide for adding or changing a TinyODOM model
family.

If you are working on board support, toolchains, flashing, or runtime
measurement, use
[`../microcontrollers/README.md`](../microcontrollers/README.md).

For the broader `src/` architecture map, see
[`../../README.md`](../../README.md).

## Key Files

The current key files for this layer are:

- [`../interfaces.py`](../interfaces.py)
  `ModelFamilyABC` and the default helper behavior.
- [`../pipeline_types.py`](../pipeline_types.py)
  `ModelBuildContext`, which is the normalized build-time input passed into a
  family.
- [`tinyodom_tcn.py`](tinyodom_tcn.py)
  The only built-in concrete family today.
- [`../registry.py`](../registry.py)
  `model_family_registry`, the runtime lookup table.
- [`../builtin_components.py`](../builtin_components.py)
  The built-in registration path.
- [`../component_selection.py`](../component_selection.py)
  Config selection for `model.family`, `model.params`, and `model.search`.
- [`../../nas_model_client.py`](../../nas_model_client.py)
  Training/NAS-side consumption of `sample_hparams(...)`,
  `validate_hparams(...)`, and `build_model(...)`.
- [`../../hil_server.py`](../../hil_server.py)
  Export/HIL-side consumption of `materialize_export_model(...)`.

## Current Runtime Shape

The active flow is:

1. Entry points call `ensure_builtin_components_registered()` from
   [`../builtin_components.py`](../builtin_components.py).
2. `resolve_component_selection(...)` in
   [`../component_selection.py`](../component_selection.py) reads
   `model.family` plus the family config block.
3. The entry point resolves the class with `model_family_registry.get(...)`
   from [`../registry.py`](../registry.py) and instantiates it.
4. Training/NAS code in [`../../nas_model_client.py`](../../nas_model_client.py)
   calls:
   - `validate_config(...)`
   - `sample_hparams(...)`
   - `validate_hparams(...)`
   - `build_model(...)`
5. Export/HIL code in [`../../hil_server.py`](../../hil_server.py) calls
   `materialize_export_model(...)`, then validates input/output shape before
   candidate preparation.
6. After that, the selected microcontroller backend receives the already
   materialized model and owns candidate staging, compile, upload, and runtime
   measurement.

That boundary matters:

- Model families own Keras-model construction and export-oriented model
  selection/materialization.
- Microcontroller backends own TFLite staging, compile/upload flow, and device
  telemetry.

## Add A New Model Family

Use [`tinyodom_tcn.py`](tinyodom_tcn.py) as the concrete example.

1. Add a new module under [`.`](.).
2. Implement a `ModelFamilyABC` subclass from
   [`../interfaces.py`](../interfaces.py).
3. Implement the required methods:
   - `sample_hparams(trial, ctx, config)`
   - `build_model(hparams, ctx, config)`
4. Keep the family-specific logic inside the family:
   - search-space sampling
   - input-shape interpretation from `ModelBuildContext`
   - family-local helper composition for builders, decoders, or export
     materialization
   - family-specific export-model materialization
5. Register the class in [`../builtin_components.py`](../builtin_components.py)
   with `model_family_registry.register(...)`.
6. Select it through `model.family` in config.
7. Put family-owned knobs under `model.params` and `model.search`.
8. Verify both the training/NAS path and the HIL/export path.

## `ModelFamilyABC` In Practice

The abstract contract is in [`../interfaces.py`](../interfaces.py). The hooks
that matter most in the current code are:

- `sample_hparams(...)`
  Used by [`../../nas_model_client.py`](../../nas_model_client.py) to produce
  family-owned hyperparameters for a trial.
- `build_model(...)`
  Builds the uncompiled Keras model from normalized hyperparameters plus
  `ModelBuildContext`.
- `validate_config(...)`
  Optional config validation before use.
- `validate_hparams(...)`
  Optional validation for one sampled hyperparameter set.
- `decode_trial_hparams(...)` and `default_seed_trial(...)`
  Optional family-owned hooks for reconstructing persisted trial params and
  publishing a default seed trial when the family needs one.
- `load_model(...)` and `custom_objects()`
  Used by the default trained-model materialization path.
- `count_flops(...)` and `supports_tflite()`
  Family-owned capability hooks used by NAS scoring and export gating.
- `materialize_export_model(...)`
  The export/HIL hook that decides which model variant should be prepared for a
  backend.

The default `ModelFamilyABC.materialize_export_model(...)` supports:

- `untrained`
- any variant whose name starts with `trained`

[`tinyodom_tcn.py`](tinyodom_tcn.py) extends that to also handle:

- `approx_trained`
- `representative`
- `bn_full_plus_non_bn_bias_perturbed`

Those extra variants are currently implemented by building the model and then
applying deterministic perturbation before backend preparation.

## Registration And Config Selection

Registration is explicit. Model families are not auto-discovered from this
directory.

The current built-in pattern is:

1. Define the class in this package.
2. Import it in [`../builtin_components.py`](../builtin_components.py).
3. Register it under a stable string key with
   `model_family_registry.register(...)`.
4. Select that key through `model.family`.

`resolve_component_selection(...)` in
[`../component_selection.py`](../component_selection.py) currently resolves:

- `model.family`
- `model.params`
- `model.search`

and returns that `model` subtree as a mapping under `model_config`.

## Where Export/Materialization Fits

Model-family work stops before board-specific candidate handling starts.

The current handoff is:

1. [`../../hil_server.py`](../../hil_server.py) calls
   `model_family.materialize_export_model(...)`.
2. The HIL server validates the materialized model against the resolved input
   shape and task outputs.
3. The HIL server packages that model into a backend request.
4. The microcontroller backend consumes that request and performs export
   staging, compile, upload, and measurement according to its own backend
   contract.

For Arduino, STM32, and other device-specific work, use
[`../microcontrollers/README.md`](../microcontrollers/README.md).

## Model Family Work vs Microcontroller Backend Work

Model family work:

- define architecture search knobs
- build the Keras model
- validate family-local config/hyperparameters
- decide how trained or approximate export variants are materialized
- provide `custom_objects()` for model reloads when needed

Microcontroller backend work:

- choose sketches/project templates
- convert/stage exported artifacts
- compile and parse memory diagnostics
- upload or flash the image
- run runtime/power measurement
- interpret backend-specific device options

Do not put board or toolchain logic in a model family. Do not put
architecture/search logic in a microcontroller backend.
