# Set up CREST

Install Git, Conda, and Make. Run the remaining commands from the repository root with the `crest` environment active.

```bash
git clone --recurse-submodules https://github.com/nesl/crest.git
cd crest
conda env create -f environment.yml -n crest
conda activate crest
make install
python src/nas_model_client.py --help
```

In an existing clone, run `git submodule update --init --recursive` to initialize any missing submodules before installation. The help command should list `--config`, `--study-name`, and `--smoke-test`.

## Prepare one dataset

### OxIOD for inertial odometry

Obtain the OxIOD Complete Dataset archive using the [dataset preparation guide](https://github.com/nesl/crest/blob/main/data/dataset_download_and_splits/README.md). Replace the archive path below with the path on your machine. Preparation replaces `data/oxiod/` and restores the curated split files, so keep any local custom data or splits elsewhere.

```bash
make prepare-dataset OXIOD_ZIP=/path/to/OxIOD.zip
```

Check that the activity directories under `data/oxiod/` contain extracted data and their `Train.txt`, `Valid.txt`, and `Test.txt` lists. If your data lives elsewhere, update `dataset.params.directory` in the YAML you run.

### UrbanSound8K for sound classification

With local audio in the expected directory, run `make prepare-audio-dataset`. To download the dataset first, review its terms and pass `URBANSOUND8K_ARGS="--download --accept-license"` to the Make command. Check that the cache specified by `dataset.params.cache_dir` contains `metadata.json` and the train, validation, test, and calibration NPZ files. Cross-fold reporting requires additional caches generated with `URBANSOUND8K_ARGS="--fold-rotation"`. [Dataset preparation guide](https://github.com/nesl/crest/blob/main/data/dataset_download_and_splits/README.md).

## Enable an NVIDIA GPU if needed

The base environment supports CPU execution. On a compatible NVIDIA host, install the CUDA dependencies for the TensorFlow version in the active environment and verify detection.

```bash
tf_version=$(python -c 'from importlib.metadata import version; print(version("tensorflow"))') &&
  python -m pip install "tensorflow[and-cuda]==${tf_version}" &&
  python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Continue to [your first study](first-study.md) or [hardware setup](hardware.md).
