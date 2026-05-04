# TinyODOM-EX

TinyODOM-EX is a hardware-aware tinyML NAS repo. It combines dataset, task, and
model-family selection with board-specific hardware-in-the-loop measurement so
the same training/NAS flow can target odometry and audio-classification models
on Arduino-class boards, Portenta H7, and the current STM32 N6 backend.

## Architecture At A Glance

- A dataset adapter loads and normalizes data.
- A task adapter defines targets, fitting/evaluation behavior, and task-owned
  metrics.
- A model family samples hyperparameters, builds models, and materializes the
  export variant used for HIL.
- A microcontroller backend stages, compiles, uploads, runs, and measures one
  candidate on the selected target.
- Shared scoring, pruning, and trial logging operate on the generic trial
  outcome produced by those layers.

For the source-level architecture and extension points, see
[src/README.md](src/README.md).

## Choose Your Workflow

1. **Training only**
   Use this when you want to run NAS/training without talking to hardware.
   Start from `src/config/nas_config.yaml`, set `device.hil: false`, and read
   [src/config/README.md](src/config/README.md) plus [src/README.md](src/README.md).
   For the UrbanSound8K audio DS-CNN path, start from
   [src/config/nas_config_audio_stm32.yaml](src/config/nas_config_audio_stm32.yaml)
   and use `make audio-desktop-smoke` for a quick hardware-free check.

2. **Arduino HIL**
   Use this for Arduino CLI-backed DUTs and harness-backed measurement flows.
   Start from [src/config/nas_config_ble.yaml](src/config/nas_config_ble.yaml)
   for Nano 33 BLE or
   [src/config/nas_config_portenta.yaml](src/config/nas_config_portenta.yaml)
   for Portenta H7. For the audio DS-CNN smoke path, use
   [src/config/nas_config_audio_portenta.yaml](src/config/nas_config_audio_portenta.yaml)
   with `make audio-portenta-hil-smoke`. Then read
   [src/tinyodom/microcontrollers/README.md](src/tinyodom/microcontrollers/README.md)
   and [sketches/README.md](sketches/README.md).

3. **STM32 HIL**
   Use this for the current STM32 N6 backend.
   Start from [src/config/nas_config.yaml](src/config/nas_config.yaml), then
   use [src/config/nas_config_audio_stm32.yaml](src/config/nas_config_audio_stm32.yaml)
   for the audio DS-CNN smoke path. Then
   read [src/tinyodom/microcontrollers/README.md](src/tinyodom/microcontrollers/README.md)
   and the committed STM32 workspace notes under
   [sketches/stm32/tinyodom_stm32_lrun/README.md](sketches/stm32/tinyodom_stm32_lrun/README.md).

4. **Analysis scripts / one-off experiments**
   Use this for focused measurement or validation runs outside the main NAS
   loop. Start with [analysis_scripts/README.md](analysis_scripts/README.md),
   then open the package-specific README for the script family you need.

## Environment Setup

1. **Clone with submodules.**
   ```bash
   git clone --recurse-submodules <url>
   ```
   If you already cloned without submodules:
   ```bash
   git submodule update --init --recursive
   ```

2. **Create and activate the Conda environment.**
   ```bash
   conda env create -f environment.yml -n tinyodomex
   conda activate tinyodomex
   ```

3. **Install the repo in editable mode.**
   ```bash
   make install
   ```

4. **If you are using GPUs, install the repo-tested TensorFlow CUDA wheel set.**
   ```bash
   pip install --upgrade pip
   pip install tensorflow[and-cuda]==2.20.0
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```

   Re-run this step after recreating or replacing the Conda environment. The
   base `environment.yml` installs CPU-usable TensorFlow so CPU-only machines
   do not download CUDA runtime wheels by default; GPU servers need the
   `tensorflow[and-cuda]` extra installed after environment creation. If
   `nvidia-smi` sees GPUs but TensorFlow prints `[]`, this CUDA wheel step is
   usually missing.

If you are CPU-only, the Conda environment already provides the dependencies
needed by the repo.

## Dataset Preparation

For UrbanSound8K audio experiments, prepare the cached log-mel tensors before
running audio training or HIL smoke commands:

```bash
make prepare-audio-dataset
```

Use `URBANSOUND8K_ARGS="--download --accept-license"` when you want the
preparation script to download the dataset through soundata, or
`URBANSOUND8K_ARGS="--fold-rotation"` when you also need the Phase 9
fold-rotation reporting caches.

1. Download the OxIOD "Complete Dataset" zip from `http://deepio.cs.ox.ac.uk/`.
2. Rename it to `OxIOD.zip` or pass an explicit path.
3. Prepare the dataset from the repo root:
   ```bash
   make prepare-dataset
   # or:
   make prepare-dataset OXIOD_ZIP=/path/to/OxIOD.zip
   ```

This extracts the dataset into `data/oxiod`, normalizes folder names such as
`slow walking -> slow_walking`, and restores the curated tracked split files
for each activity. The dataset-specific details live in
[data/dataset_download_and_splits/README.md](data/dataset_download_and_splits/README.md).

## Arduino Tooling Setup

All Arduino CLI state is kept inside `tools/` so the repo does not need to
write into `$HOME` or system directories.

1. Ensure `tinyodomex` is active.
2. Bootstrap Arduino CLI and repo-local hooks:
   ```bash
   make arduino-setup
   ```
3. Reactivate the environment so the new hooks are loaded:
   ```bash
   conda deactivate
   conda activate tinyodomex
   ```
4. Verify the CLI:
   ```bash
   arduino-cli --config-file tools/arduino-cli.yaml version
   ```
5. Install the board package you need. Example for Nano 33 BLE:
   ```bash
   arduino-cli core install arduino:mbed_nano --config-file tools/arduino-cli.yaml
   ```

If Portenta uploads on Linux fail with `LIBUSB_ERROR_ACCESS`, add the udev
rules documented in
[src/tinyodom/microcontrollers/README.md](src/tinyodom/microcontrollers/README.md).

## STM32 Setup

The STM32 flow keeps `STM32CubeCLT` installed outside the repo while cloning
the STM32CubeN6 firmware package into `tools/stm32/STM32CubeN6`.

Run STM32 bootstrap only on the machine that is physically connected to the
STM32 board and will run `python src/hil_server.py`.

Before running the bootstrap, ensure these tools are on your shell `PATH`:

- `ST-LINK_gdbserver`
- `arm-none-eabi-gdb`
- `STM32_Programmer_CLI`
- `arm-none-eabi-gcc`
- `arm-none-eabi-size`
- `arm-none-eabi-objdump`
- `STM32_SigningTool_CLI` or `STM32TrustedPackageCreator_CLI`

Then run:

```bash
make stm32-setup
```

That script validates the toolchain, clones or repairs
`tools/stm32/STM32CubeN6`, checks out the pinned `v1.3.0` baseline, and
refreshes the repo-local STM32 vendor subsets.

## Config Files

The shipped starting points are:

- [src/config/nas_config.yaml](src/config/nas_config.yaml)
  Default STM32-oriented config for the current `STM32_NUCLEO_N657X0_Q`
  backend. This is the main starting point for STM32 runs and the general
  example config for the repo.
- [src/config/nas_config_ble.yaml](src/config/nas_config_ble.yaml)
  BLE-focused starting point for `ARDUINO_NANO_33_BLE_SENSE`.
- [src/config/nas_config_portenta.yaml](src/config/nas_config_portenta.yaml)
  Portenta H7-focused starting point.
- [src/config/nas_config_audio_stm32.yaml](src/config/nas_config_audio_stm32.yaml)
  UrbanSound8K audio DS-CNN starting point for desktop training and STM32 N657
  smoke/HIL work.
- [src/config/nas_config_audio_portenta.yaml](src/config/nas_config_audio_portenta.yaml)
  UrbanSound8K audio DS-CNN starting point for Arduino-backed Portenta H7 and
  BLE compile/preflight smoke work.

The highest-signal fields for a first pass are:

- `device.*`
  Target selection, HIL enable/disable, serial ports, runtime mode, and
  backend-specific nested options.
- `dataset.*`
  Dataset adapter selection and dataset-local paths/parameters, including
  OxIOD windowing or UrbanSound8K cache locations.
- `training.*`
  NAS epochs/trials, full-training epochs, quantization, and the runtime-side
  `energy_aware` / `input_mode` switches.
- `nas.*`
  Score and prune configuration.
- `dataset`, `task`, `model`
  Modular component selection blocks when you want to override the built-in
  defaults explicitly.

For the full config reference, score/prune schema, and current runtime caveats,
see [src/config/README.md](src/config/README.md).

## Running NAS And HIL

TinyODOM-EX runs a NAS/training client on a training host and talks to the HIL
server running on the board-connected device host.

### 1. Start the HIL server on the device host

```bash
cd /path/to/TinyODOM-EX
conda activate tinyodomex
python src/hil_server.py
```

For STM32, install `STM32CubeCLT`, ensure its tools are on `PATH`, and run
`make stm32-setup` on that same host before starting the server.

### 2. Open a reverse SSH tunnel from the device host to the training host

```bash
ssh -R "6001:127.0.0.1:6001" <gpu_server>
```

The default configs expect the HIL server at `127.0.0.1:6001`.

### 3. Run the NAS client on the training host

```bash
cd /path/to/TinyODOM-EX
conda activate tinyodomex

# Quick smoke pass
python3 src/nas_model_client.py --smoke-test 3 --study-name smoke_run

# Full NAS run
python3 src/nas_model_client.py --study-name tinyodom_run
```

Useful flags:

- `--config /path/to/config.yaml`
- `--smoke-test N`
- `--study-name NAME`

### 4. Outputs

Artifacts are written under the configured `outputs.models_dir` and
`outputs.candidate_dir`. Typical outputs include:

- `models/<study_name>/optuna.db`
- `models/<study_name>/trials.csv`
- `models/<study_name>/train_history.json`
- `models/<study_name>/summary.json`
- generated TFLite and `.keras` artifacts

## Smoke Tests

- Audio desktop training smoke:
  [analysis_scripts/audio_desktop_smoke/README.md](analysis_scripts/audio_desktop_smoke/README.md)
- Audio STM32 HIL smoke:
  [analysis_scripts/audio_stm32_hil_smoke/README.md](analysis_scripts/audio_stm32_hil_smoke/README.md)
- Audio Portenta/BLE HIL smoke:
  [analysis_scripts/audio_portenta_hil_smoke/README.md](analysis_scripts/audio_portenta_hil_smoke/README.md)
- Quick HIL sanity check:
  [analysis_scripts/hil_single_run/README.md](analysis_scripts/hil_single_run/README.md)
- STM32 toy AI smoke test:
  [analysis_scripts/stm32_example_project/README.md](analysis_scripts/stm32_example_project/README.md)
- Additional one-off hardware analysis packages:
  [analysis_scripts/README.md](analysis_scripts/README.md)

## Docs Map

- [src/README.md](src/README.md)
  Source architecture, shared abstractions, trial logging, and extension seams.
- [src/config/README.md](src/config/README.md)
  Full config reference and scoring/pruning semantics.
- [src/tinyodom/microcontrollers/README.md](src/tinyodom/microcontrollers/README.md)
  Backend contracts, bring-up, staging, compile, upload, and runtime flows.
- [src/tinyodom/model_families/README.md](src/tinyodom/model_families/README.md)
  Model-family-specific extension guide.
- [sketches/README.md](sketches/README.md)
  Shared Arduino sketch and STM32 workspace layout.
- [analysis_scripts/README.md](analysis_scripts/README.md)
  One-off analysis and validation utilities.
- [data/dataset_download_and_splits/README.md](data/dataset_download_and_splits/README.md)
  OxIOD preparation plus UrbanSound8K audio cache preparation.

## Troubleshooting

- If training-only runs should not touch hardware, set `device.hil: false`.
- If Arduino uploads fail on Linux with `LIBUSB_ERROR_ACCESS`, apply the udev
  rules documented in the MCU README.
- If STM32 bootstrap fails, confirm the full `STM32CubeCLT` toolchain is on
  `PATH` before rerunning `make stm32-setup`.
- If OxIOD preparation fails, confirm the zip exists and that the repo still
  contains the tracked split templates under `data/oxiod/<activity>/`.
- If audio smoke commands fail while loading data, run `make prepare-audio-dataset`
  and confirm the UrbanSound8K cache path in the selected config exists.
