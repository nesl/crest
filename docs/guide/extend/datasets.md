# Add a dataset

A dataset adapter loads local data and supplies model-ready splits, input shape and dtype, and metadata for the selected task and model family. See [Dataset API](../api/datasets.md).

1. Add a class under `src/crest/datasets/` that implements `DatasetABC.load(dataset_config)`. Return a `DatasetBundle` with a training split. Supply the validation and test splits required by the selected task. Set `input_shape` to the shape of one example without its batch dimension.
2. Implement `validate_config(dataset_config)` to check the parameters the adapter owns. Check the loaded data for shape, dtype, split integrity, and metadata consistency.
3. Populate `bundle.calibration` for representative export samples. The default `make_calibration_data` returns that field. Override it if your adapter needs a deterministic sampling rule or a fallback.
4. Add a guarded call to `dataset_registry.register("your_key", YourDataset)` in `src/crest/builtin_components.py`, following the existing registrations. Select your key under `dataset.name` and place its options under `dataset.params`.

The UrbanSound8K adapter reads precomputed features from `cache_dir` and supplies class names, class count, and label encoding to the [sound classification task](tasks.md). This selection comes from `src/config/nas_config_audio_portenta.yaml`.

```yaml
dataset:
  name: urbansound8k_mel
  params:
    cache_dir: "data/urbansound8k/cache/v2_logmel_16k_2s_64mels_25ms_10ms_folds"
    batch_period_ms: 2000
```

Use `src/crest/datasets/urbansound8k_mel.py` as a template for cached features or `src/crest/datasets/oxiod.py` to wrap an existing splitter. Use a tiny local fixture to check deterministic loading, validation failures, and the actual calibration split. The UrbanSound8K tests are in `test/test_urbansound8k_mel_dataset.py`.
