<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->

# Dataset Preparation

These preparation steps are software-only. They do not require a development
board or HIL harness, but they do require local storage for the downloaded raw
datasets and generated caches.

## UrbanSound8K Audio Cache

UrbanSound8K supplies the cached log-mel inputs for the audio workload. The dataset license is
non-commercial/by-nc; automatic download requires an explicit license
acknowledgement.

Plan for about 5.6 GB of local data for the UrbanSound8K download and generated
cache artifacts.

Generated UrbanSound8K files are local-only and ignored by git:

- Raw/downloaded audio: `data/urbansound8k/raw/`
- Cached features: `data/urbansound8k/cache/v2_logmel_16k_2s_64mels_25ms_10ms_folds/`

To validate an existing local soundata copy and build the deterministic log-mel
cache:

```bash
make prepare-audio-dataset
```

The current audio pipeline uses schema-2 caches. If an older schema-1 cache is
present, regenerate it with the same command. To also build the full
train-8/validate-1/test-1 fold-rotation caches used for cross-fold reporting:

```bash
make prepare-audio-dataset URBANSOUND8K_ARGS="--fold-rotation"
```

To download through soundata first:

```bash
make prepare-audio-dataset URBANSOUND8K_ARGS="--download --accept-license"
```

The audio preparation script writes `metadata.json`, `train.npz`, `val.npz`,
`test.npz`, and `calibration.npz`. Use the generated cache directory as
`dataset.params.cache_dir` with `dataset.name: urbansound8k_mel`. Keep its metadata and arrays together.

The script lives at:

- `data/dataset_download_and_splits/urbansound8k/prepare_urbansound8k.py`

## OxIOD Dataset Preparation

Download the OxIOD "Complete Dataset" zip from `http://deepio.cs.ox.ac.uk/`,
rename it to `OxIOD.zip`, then run the preparation step from the repo root.
The raw archive is not stored in this repository:

```bash
make prepare-dataset
# or:
python data/dataset_download_and_splits/oxiod/prepare_oxiod.py --zip-path /path/to/OxIOD.zip
```

Preparation replaces `data/oxiod` with a fresh extraction, normalizes activity
folder names, and restores the curated train/validation/test split files. Keep
any local data you need outside that directory before rerunning preparation.

Dataset-specific preparation files:

- `data/dataset_download_and_splits/oxiod/prepare_oxiod.py`
- `data/dataset_download_and_splits/oxiod/reference_splits/`

The preparation script uses the per-activity split files under
`data/oxiod/<activity>/` as its default template source.

## Use the prepared datasets

Select `dataset.name: oxiod` with `dataset.params.directory: data/oxiod/`, or
`dataset.name: urbansound8k_mel` with the generated audio cache path. See the
[configuration examples](../../src/config/README.md#example-configs) for complete
experiments and the [dataset guide](../../src/crest/datasets/README.md) to add
another dataset.

## Raw vs Syn Data

Each activity folder contains `raw/` and `syn/` subfolders:

- `raw/`
  Raw, unsynchronized data with high-precision timestamps.
- `syn/`
  Synchronized data where IMU and VI are aligned, with slightly less precise
  timestamps.

The shipped splits use `syn/`, which is the correct default for the built-in
odometry path in this repo.

## Split Summary

The current curated splits cover these activities:

- `handbag`
- `handheld`
- `pocket`
- `running`
- `slow_walking`
- `trolley`

| Subfolder    | Total Files | Train          | Valid         | Test          | Train_Valid    |
|--------------|-------------|----------------|---------------|---------------|----------------|
| handbag      | 8           | 5 (62.5%)      | 2 (25%)       | 1 (12.5%)     | 7 (87.5%)      |
| handheld     | 24          | 18 (75%)       | 4 (16.7%)     | 2 (8.3%)      | 22 (91.7%)     |
| pocket       | 11          | 7 (63.6%)      | 3 (27.3%)     | 1 (9.1%)      | 10 (90.9%)     |
| running      | 7           | 5 (71.4%)      | 1 (14.3%)     | 1 (14.3%)     | 6 (85.7%)      |
| slow walking | 8           | 5 (62.5%)      | 2 (25%)       | 1 (12.5%)     | 7 (87.5%)      |
| trolley      | 13          | 10 (76.9%)     | 2 (15.4%)     | 1 (7.7%)      | 12 (92.3%)     |
| **Total**    | **71**      | **50 (70.4%)** | **14 (19.7%)**| **7 (9.9%)**  | **64 (90.1%)** |
