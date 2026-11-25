# Add a model family

A model family converts a trial's architecture parameters and a `ModelBuildContext` into a Keras model. The context brings together the dataset's input shape and dtype and the task's output specification. See [Model family API](../api/models.md).

1. Create a `ModelFamilyABC` subclass under `src/crest/model_families/`. Implement `sample_hparams(trial, ctx, config)` and `build_model(hparams, ctx, config)`. Return an uncompiled model for the task to compile.
2. Validate family parameters and search overrides with `validate_config`. Validate sampled or replayed choices with `validate_hparams`. If persisted trial values need reconstruction into a different build-time representation, implement `decode_trial_hparams`.
3. Check the export path. The base class can build an untrained model or load a trained checkpoint. Override `materialize_export_model` if a family needs its own export transformation and use `custom_objects` if checkpoints need custom Keras types.
4. Register the family with `model_family_registry` in `src/crest/builtin_components.py`. Select it under `model.family`, put build options in `model.params`, and put search overrides in `model.search`. Selection requires a nonempty `model.params.export_variant`.

For example, the Portenta audio configuration builds the DS-CNN family for the [sound classification task](tasks.md) and uses a freshly built model for export.

```yaml
model:
  family: audio_dscnn
  params:
    export_variant: untrained
  search: {}
```

Study `src/crest/model_families/audio_dscnn.py` for categorical search overrides and `src/crest/model_families/odom_tcn.py` for family-specific export materialization. Use a small context to test sampling and output shape, then materialize an export and exercise the intended conversion path. See `test/test_audio_dscnn.py` and `test/test_odom_tcn.py`.
