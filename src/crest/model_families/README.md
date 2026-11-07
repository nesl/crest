<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->

# Model families

Model families own architecture search, family-local validation, Keras construction
from `ModelBuildContext`, and export-model materialization. Dataset loading, task
metrics, scoring, and board staging/measurement belong to their respective layers.
Construction and export preflight can be tested without hardware; HIL validation
requires the selected board and backend.

The shipped families are [`odom_tcn.py`](odom_tcn.py) (`odom_tcn`, odometry TCN)
and [`audio_dscnn/family.py`](audio_dscnn/family.py) (`audio_dscnn`, cached log-mel
classification). See the [optimizer guide](../optimizers/README.md) for how their
search declarations feed shared evaluation.

## Hook contract

[`ModelFamilyABC`](../interfaces/model_family.py) defines the hooks, and
[`ModelBuildContext`](../pipeline_types.py) supplies normalized build inputs.

- `sample_hparams(trial, ctx, config)` returns normalized hyperparameters;
  `build_model(hparams, ctx, config)` returns an uncompiled Keras model.
- `trial_search_space(ctx, config)` is required for explicit proposers and optional
  for native Optuna sampling. Return ordered `SearchParam` declarations from
  [`crest.optimizers.search_space`](../optimizers/search_space.py) with the exact
  raw persisted names, bounds, and choices used by sampling. Keep decoding inside
  the family. Optional descriptions, units, and effects supply semantic hints.
  Explicit candidates must include every active field, including runner-added
  quantization or CPU-clock fields when those searches are enabled.
- `validate_config(...)` and `validate_hparams(...)` enforce local constraints;
  `decode_trial_hparams(...)` supports replay from persisted raw parameters.
- `materialize_export_model(...)` chooses the backend-ready model. The default
  handles `untrained` and names beginning with `trained`, using `load_model(...)`
  and `custom_objects()` where needed. Odometry adds its own export variants.
- `count_flops(...)`, `estimate_static_memory(...)`, and `supports_tflite()` expose
  optional capabilities. `default_seed_trial(...)` remains a legacy hook; fresh
  studies do not enqueue it.

Native Optuna skips descriptor and semantic-context setup. For explicit proposers,
descriptor errors propagate before evaluation; descriptions do not alter legal
parameter values.

## Add a family

1. Implement a `ModelFamilyABC` subclass here, using a shipped family as a guide.
   Implement sampling and construction, local validation, and explicit export
   behavior. Add a matching raw descriptor if explicit proposers must support it.
2. Import and register it with `model_family_registry.register(...)` in
   [`builtin_components.py`](../builtin_components.py); registration is explicit,
   not automatic discovery.
3. Select the stable key through `model.family`. Put family options in
   `model.params` and search controls in `model.search`; see the
   [config guide](../../config/README.md).
4. Test descriptor/sampling agreement, normalized hyperparameters, task output
   shape, and export materialization without hardware. Validate HIL afterward when
   deployment support is needed. Keep board/toolchain logic in the
   [microcontroller backend](../microcontrollers/README.md).
