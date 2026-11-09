<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->

# Getting started

Run commands from the repository root with the `crest` environment active.

## Environment setup

Install Git, Conda, and Make, then clone and install CREST.

```bash
git clone --recurse-submodules https://github.com/nesl/crest.git
cd crest
conda env create -f environment.yml -n crest
conda activate crest
make install
```

For an existing clone without submodules, run `git submodule update --init --recursive`.
The base environment supports CPU execution.
On a compatible NVIDIA GPU host, install the TensorFlow CUDA dependencies and verify GPU detection.

```bash
pip install --upgrade pip
pip install 'tensorflow[and-cuda]==2.20.0'
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Repeat the CUDA installation after recreating the environment.
If `nvidia-smi` sees the GPU but TensorFlow returns `[]`, check that the CUDA extra is installed.

## Dataset preparation

For OxIOD, obtain the Complete Dataset archive using the [dataset guide](../data/dataset_download_and_splits/README.md), then run.

```bash
make prepare-dataset OXIOD_ZIP=/path/to/OxIOD.zip
```

For UrbanSound8K, follow the same guide to obtain the audio and prepare cached log-mel features.

```bash
make prepare-audio-dataset
```

Download options, dataset terms, paths, and fold-rotation caches are documented in the dataset guide.

## Choose a configuration

| Workflow | Starting configuration | Hardware |
| --- | --- | --- |
| OxIOD proxy search | [FLOPs](config/nas_config_flops_rmse.yaml) or [memory proxy](config/nas_config_memory_proxy.yaml) | None |
| OxIOD Arduino HIL | [Nano 33 BLE](config/nas_config_ble.yaml) or [Portenta H7](config/nas_config_portenta.yaml) | Target board; harness for harness measurements |
| OxIOD STM32 HIL | [STM32](config/nas_config_stm32.yaml) | NUCLEO-N657X0-Q; harness for energy measurements |
| Audio HIL | [STM32](config/nas_config_audio_stm32.yaml) or [Portenta H7](config/nas_config_audio_portenta.yaml) | Selected target and measurement harness |

Set dataset paths and training budgets before running.
For HIL, also set serial ports and connection settings.
The [configuration reference](config/README.md) explains these fields.
For hardware-free audio, adapt the audio config with `device.hil: false` and objectives/pruning that do not require hardware measurements.

## Run without hardware

Start with three OxIOD search attempts.

```bash
python src/nas_model_client.py \
  --config src/config/nas_config_flops_rmse.yaml \
  --smoke-test 3 --study-name oxiod_proxy_smoke
```

This requires the prepared dataset and evaluates models on the host.
For a full search, omit `--smoke-test` and choose a fresh study name after reviewing the training and attempt budgets in the config.

## Hardware setup

Install tools on the machine connected to the board.
The [hardware guide](crest/microcontrollers/README.md) covers board packages, vendor tools, measurement requirements, and troubleshooting.
The [firmware guide](../sketches/README.md) covers target and harness firmware.

For Arduino targets, bootstrap the CLI and reactivate the environment.

```bash
make arduino-setup
conda deactivate
conda activate crest
arduino-cli --config-file tools/arduino-cli.yaml version
```

Install the board core specified by the hardware guide.
For Nano 33 BLE.

```bash
arduino-cli core install arduino:mbed_nano --config-file tools/arduino-cli.yaml
```

For STM32, first install the [required host tools](crest/microcontrollers/README.md#required-host-tools), including ST Edge AI for model deployment, then bootstrap the workspace.

```bash
make stm32-setup
```

## Running NAS and HIL

Use matching configuration settings on the training host and board-connected host.
This example uses the STM32 config.
Substitute your board's config.

### 1. Start the HIL server on the device host

```bash
conda activate crest
python src/hil_server.py --config src/config/nas_config_stm32.yaml
```

### 2. Connect the hosts

If training runs on another machine, open a reverse SSH tunnel **from the device host to the training host** and leave it open.

```bash
ssh -R "6001:127.0.0.1:6001" user@gpu-server
```

Replace `user@gpu-server` with your training-host login.
The default connection uses `127.0.0.1:6001`.
Skip the tunnel when the client and server run on the same machine.

### 3. Run the client on the training host

```bash
conda activate crest
python src/nas_model_client.py \
  --config src/config/nas_config_stm32.yaml \
  --smoke-test 3 --study-name smoke_run
```

For a full run, omit `--smoke-test` and use a fresh study name.
Native Optuna is the default proposer.
For LLM generation, follow the [LLM guide](crest/optimizers/llm/README.md) for configuration and credentials.

### 4. Outputs

Results use the configured `outputs.models_dir` and `outputs.candidate_dir`.
Typical study files include `optuna.db`, `trials.csv`, `train_history.json`, `summary.json`, and exported model artifacts.
Smoke runs use `optuna_smoke_test.db` and append the device name to the supplied study name.
LLM runs also write request and proposal records under `llm_optimizer/`.

## Continue an experiment

To extend a study, keep its name, database, and original experiment configuration, then raise `training.nas_trials` and `training.max_total_trials` to the desired **total** completion target and attempt cap.
Use a new study when changing the optimizer or other identity-bearing settings.
See the [resume reference](config/README.md#optimizer) for compatibility requirements.

For further experiments, see the [case-study configurations](config/README.md#case-study-configurations) and [analysis utilities](../analysis_scripts/README.md).
Analysis commands usually consume saved results.
Check each package's guide for its inputs and hardware requirements.
