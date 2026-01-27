# TinyODOM-EX

## Environment Setup

0. **Ensure TensorFlowLite Micro is installed**
   When cloning, ensure to use `git clone --recurse-submodules <url>`.
   If this wasn't done, run `git submodule update --init --recursive` to ensure that TensorFlowLite Micro is installed.

1. **Create the Conda environment.**  
   ```bash
   conda env create -f environment.yml -n tinyodomex
   ```
   This pins all Python dependencies (NumPy, PyTorch, etc.) exactly as expected by the notebooks and scripts in `src/`. Re-run with `--force` if you need to rebuild from scratch after dependency changes.

2. **Activate the environment.**  
   ```bash
   conda activate tinyodomex
   ```
   Stay inside this shell for every training or preprocessing command—our tooling scripts expect `CONDA_PREFIX` and `python` to point into this env.

3. **(If using GPUs) Install TensorFlow with bundled CUDA wheels.**  
   TensorFlow ships CUDA/cuDNN wheels via the `and-cuda` extra, so you do **not** need system-level CUDA installs beyond the NVIDIA driver. The latest stable release as of 14 Nov 2025 is `2.20.0`, so pin it explicitly on shared servers:  
   ```bash
   pip install --upgrade pip
   pip install tensorflow[and-cuda]==2.20.0
   ```  
   This pulls CUDA 12.4+ compatible binaries plus matching NCCL/cuDNN wheels recommended by the TensorFlow team.

> **Tip:** If you are CPU-only you don't need to do this as it the non-cuda version is automaticaly installed by the conda environment.

## Dataset Preparation (OxIOD)

1. **Download OxIOD.** Grab the “Complete Dataset” zip from http://deepio.cs.ox.ac.uk/ and rename it to `OxIOD.zip` once it’s on the server.  
2. **Run the provided splitter.** From the repo root:  
   ```bash
   python data/dataset_download_and_splits/prepare_oxiod.py --zip-path OxIOD.zip
   ```  
   The script extracts into `data/oxiod`, normalizes folder names, and regenerates the curated `Train.txt`, `Valid.txt`, `Test.txt`, and `Train_Valid.txt` files that match the splits documented in `data/dataset_download_and_splits/README.md`.
3. **Verify folder structure.** You should now have `data/oxiod/<device>/<syn|raw>/...` plus the four split text files under each activity folder as described in the dataset README.

## Arduino CLI & Microcontroller Tooling

All firmware builds happen in a sandboxed `tools/` directory inside this repo so we never touch system locations or `$HOME`.

1. **Ensure the Conda env is active** (`conda activate tinyodomex`). The setup script installs Conda activation hooks into `CONDA_PREFIX`.
2. **Run the bootstrapper:**  
   ```bash
   ./setup_arduino.sh
   ```  
   This script:
   - Downloads the Arduino CLI binary into `tools/bin` without requiring root.  
   - Generates `tools/arduino-cli.yaml` and points `directories.data/downloads/user` to `tools/arduino-*` so cores, caches, and libraries stay inside the repo.  
   - Copies the hook templates from `env_setup/` into `$CONDA_PREFIX/etc/conda/{activate.d,deactivate.d}/arduino.sh`, ensuring every future `conda activate tinyodom-ex` automatically sets `PATH`, `ARDUINO_DIRECTORIES_*`, and `ARDUINO_CONFIG_FILE`.
3. **Reactivate the environment** so the new hooks run:  
   ```bash
   conda deactivate
   conda activate tinyodomex
   ```
4. **Verify the CLI:**  
   ```bash
   arduino-cli --config-file tools/arduino-cli.yaml version
   ```
5. **Install the required board packages** (example for Nano 33 BLE):  
   ```bash
   arduino-cli core install arduino:mbed_nano --config-file tools/arduino-cli.yaml
   ```

> **Why this flow?** This setup allows us to run the needed modules without modifying `$HOME` or using `apt`. Keeping binaries + caches in `tools/` allows everything to be contained within the folder, and the Conda hooks ensure our pipelines (training, conversion, firmware compile) always see the same CLI without editing shell rc files.

## Quick Checklist
- [ ] Clone with submodules: `git clone --recurse-submodules <url>`
- [ ] `conda env create -f environment.yml -n tinyodomex`
- [ ] `conda activate tinyodomex`
- [ ] (GPU) `pip install tensorflow[and-cuda]==2.20.0`
- [ ] Download OxIOD and run `python data/dataset_download_and_splits/prepare_oxiod.py --zip-path OxIOD.zip`
- [ ] `./setup_arduino.sh` then `conda deactivate && conda activate tinyodomex`
- [ ] `arduino-cli --config-file tools/arduino-cli.yaml version`
- [ ] Start HIL server on the device host: `python src/hil_server.py`
- [ ] SSH to the GPU box with reverse tunnel: `ssh -R "6001:127.0.0.1:6001" <gpu_server>`
- [ ] On the GPU box, run NAS: `python3 src/nas_model_client.py --study-name <name>`

## NAS configuration (`src/nas_config.yaml`)

Before running long NAS jobs, skim and adjust `src/nas_config.yaml`:

- **Device block (`device.*`)**
   - `serial_port`: set to your board’s serial device (e.g. `/dev/cu.usbmodem*` on macOS, `/dev/ttyACM*` on Linux).
   - `hil`: keep `true` for full hardware-in-the-loop, or set `false` to run latency/energy proxies without talking to a board.
- **Data block (`data.*`)**
   - `directory`: root of the OxIOD dataset if you didn’t use the default `data/oxiod/` location.
   - `calibration_windows`: reduce for faster experiments, increase or set `null` for more representative calibration.
- **Training block (`training.*`)**
   - `nas_trials`, `nas_epochs`, `model_epochs`: trade off search depth vs wall-clock time.
   - `nas_multiobjective`: enable/disable multi-objective NAS (accuracy + latency/energy).
   - `energy_aware`: toggle whether energy (instead of latency) is used as the secondary objective/penalty; leave `false` if you do not have INA228 energy hardware attached.
- **Outputs and network (`outputs.*`, `network.*`)**
   - `models_dir`, `tcn_dir`: where Optuna DBs, metrics, and TFLite/C++ artifacts are written.
   - `host`, `port`: must match the HIL server and SSH tunnel; defaults (`127.0.0.1:6001`) usually work as-is.

## Running NAS and HIL

TinyODOM-EX runs a hardware-in-the-loop NAS loop between a GPU box (training) and a device host (Arduino Nano 33 BLE Sense).

### 1. Start the HIL server on the device host

On the machine physically connected to the board:

```bash
cd /path/to/TinyODOM-EX
conda activate tinyodomex
python src/hil_server.py
```

This starts a ZeroMQ REP server on `tcp://127.0.0.1:6001` (see `src/nas_config.yaml` for `network.host`/`network.port`).

### 2. Open a reverse SSH tunnel to the GPU server

From the device host, create a tunnel so the GPU server can reach the local HIL port:

```bash
ssh -R "6001:127.0.0.1:6001" <gpu_server>
```

After this, processes on `<gpu_server>` can talk to the HIL server via `127.0.0.1:6001` using the default config.

### 3. Run the NAS client on the GPU server

On the GPU box (inside the repo, with the environment created/activated):

```bash
cd /path/to/TinyODOM-EX
conda activate tinyodomex  # or an equivalent env with the same deps

# Quick smoke test (few trials, good for sanity checks)
python3 src/nas_model_client.py --smoke-test 3 --study-name smoke_nano33

# Full NAS + scoring run (uses config.training.nas_trials, HIL enabled)
python3 src/nas_model_client.py --study-name tinyodom_nas_nano33
```

Useful flags:

- `--smoke-test N` – run a short NAS smoke test with `N` trials (no final long retrain).
- `--smoke-test-multiobjective` – make the smoke test multi-objective regardless of the YAML.
- `--study-name` – label used for the Optuna study and artifact directory.

### 4. Where outputs go

Artifacts are organized under `models/`:

- `models/<study_name>/optuna.db` – Optuna study storage.
- `models/<study_name>/trials.csv` – per-trial metrics and hyperparameters.
- `models/<study_name>/train_history.json` and `*_loss.png` – training curves.
- `models/<study_name>/summary.json` – summary bundle with best params and key metrics.
- TFLite and `.keras` checkpoints are written under `tinyodom_tcn/` and `models/` as configured in `src/nas_config.yaml`.
