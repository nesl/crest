# Dataset Preparation

## UrbanSound8K Audio Cache

UrbanSound8K is used for the first audio backend path. The dataset license is
non-commercial/by-nc; automatic download requires an explicit license
acknowledgement.

Plan for about 5.6 GB of local data for the UrbanSound8K download and generated
cache artifacts.

Generated UrbanSound8K files are local-only and ignored by git:

- Raw/downloaded audio: `data/urbansound8k/raw/`
- Cached features: `data/urbansound8k/cache/v1_logmel_16k_2s_64mels_25ms_10ms/`

To validate an existing local soundata copy and build the deterministic log-mel
cache:

```bash
make prepare-audio-dataset
```

To download through soundata first:

```bash
make prepare-audio-dataset URBANSOUND8K_ARGS="--download --accept-license"
```

The audio preparation script writes `metadata.json`, `train.npz`, `val.npz`,
`test.npz`, and `calibration.npz`. The cache schema, feature parameters, crop
policy, normalization, class ordering, and calibration selection are defined in
[`audio_backend_p0_5_contract.md`](../../audio_backend_p0_5_contract.md).

The script lives at:

- `data/dataset_download_and_splits/urbansound8k/prepare_urbansound8k.py`

## OxIOD Dataset Preparation

Download the OxIOD "Complete Dataset" zip from `http://deepio.cs.ox.ac.uk/`,
rename it to `OxIOD.zip`, then run the preparation step from the repo root:

```bash
make prepare-dataset
# or:
python data/dataset_download_and_splits/oxiod/prepare_oxiod.py --zip-path /path/to/OxIOD.zip
```

The preparation script does four things:

1. Captures the curated tracked split files already present in the repo.
2. Replaces `data/oxiod` with a fresh extraction from the OxIOD archive.
3. Normalizes folder names such as `slow walking -> slow_walking`.
4. Writes the curated `Train.txt`, `Valid.txt`, `Test.txt`, and
   `Train_Valid.txt` files back into each activity folder.

The split files in this repo are not placeholders. They are the curated
tracked splits used by the built-in OxIOD path.

Dataset-specific preparation files now live under:

- `data/dataset_download_and_splits/oxiod/prepare_oxiod.py`
- `data/dataset_download_and_splits/oxiod/reference_splits/`

The preparation script still uses the per-activity split files under
`data/oxiod/<activity>/` as its default template source so the loader-facing
dataset layout remains unchanged.

## Current Loader Path

The built-in OxIOD dataset adapter lives in:

- [`src/tinyodom/datasets/oxiod.py`](../../src/tinyodom/datasets/oxiod.py)
- [`src/tinyodom/data.py`](../../src/tinyodom/data.py)

`import_oxiod_dataset(...)` in `src/tinyodom/data.py` reads the split files
from each activity folder and uses:

- `type_flag=1` for `Train_Valid.txt`
- `type_flag=2` for `Train.txt`
- `type_flag=3` for `Valid.txt`
- `type_flag=4` for `Test.txt`

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
