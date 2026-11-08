<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->

# CREST: Cross-platform Runtime Evaluation and Search Tool

![CREST framework structure](assets/crest_stack_diagram.png)

## Summary

Selecting a neural network for an embedded sensing application requires more than
comparing prediction accuracy or operation counts. Memory limits, the target
microcontroller, and the time spent waiting between inferences all affect which
model can meet an application's latency and energy requirements.

CREST connects neural architecture search with hardware-in-the-loop evaluation.
It trains candidate models, compiles and deploys them to a target MCU, and uses
measured latency and energy alongside task quality to guide model selection.
Experiments can compare continuous inference with periodic execution, including
waiting and sleep, and express application requirements through scalar or
multiple objectives and feasibility constraints.

The framework separates workloads, model families, candidate generators, and
hardware backends around a shared evaluation pipeline. It includes inertial
odometry and audio classification examples, native Optuna search, optional LLM
candidate generation, and tools for replaying selected candidates on another
supported target. The accompanying analyses examine how hardware measurements,
runtime schedules, and search choices affect model selection.

## Code architecture

| Location | Contents |
| --- | --- |
| [`src/`](src/README.md) | Search client, HIL server, and shared evaluation pipeline |
| [`src/crest/`](src/crest/README.md) | Dataset, task, model-family, optimizer, and hardware components |
| [`src/config/`](src/config/README.md) | Experiment configurations and settings reference |
| [`sketches/`](sketches/README.md) | Target-device and power-measurement harness firmware |
| [`data/`](data/dataset_download_and_splits/README.md) | Dataset preparation and split definitions |
| [`analysis_scripts/`](analysis_scripts/README.md) | Analysis and plotting of saved experiment results |
| [`test/`](test/) | Automated tests |

The [source guide](src/README.md) describes the component interfaces and how to
extend CREST to another workload, model family, optimizer, or hardware backend.

## Requirements

- Git, Conda, and Make. The supplied environment installs Python 3.11 and the
  project's Python dependencies.
- A local dataset: OxIOD for odometry or UrbanSound8K for audio classification.
- For hardware evaluation, a supported target board and its build/programming
  tools. Energy measurements also need the CREST harness and INA228 monitor.
- A GPU is optional; the hardware-free example below can run on a CPU.
- Provider credentials are needed only for live LLM candidate generation.

## Getting started

```bash
git clone --recurse-submodules https://github.com/nesl/crest.git
cd crest
conda env create -f environment.yml -n crest
conda activate crest
make install
```

Obtain OxIOD using the [dataset instructions](data/dataset_download_and_splits/README.md),
then prepare it and run three hardware-free search attempts:

```bash
make prepare-dataset OXIOD_ZIP=/path/to/OxIOD.zip

python src/nas_model_client.py \
  --config src/config/nas_config_flops_rmse.yaml \
  --smoke-test 3 \
  --study-name oxiod_proxy_smoke
```

This example evaluates TCN models using validation error and FLOPs. It requires
the prepared dataset but no board or LLM credentials. See the
[getting-started guide](src/GETTING_STARTED.md) for GPU setup, full searches,
hardware connections, and saved outputs.

## Supported workloads and hardware

| Workload | Dataset | Model family | Example configuration |
| --- | --- | --- | --- |
| Inertial odometry | OxIOD | Temporal convolutional network (TCN) | [Proxy search](src/config/nas_config_flops_rmse.yaml) |
| Audio classification | UrbanSound8K log-mel features | Depthwise-separable CNN (DS-CNN) | [STM32 audio search](src/config/nas_config_audio_stm32.yaml) |

Hardware backends support **Arduino Nano 33 BLE Sense**, **Arduino Portenta H7**
(CM7 and CM4), and **STM32 NUCLEO-N657X0-Q**. Energy measurements use a separate
harness MCU and an INA228 power monitor. Follow the
[hardware guide](src/crest/microcontrollers/README.md) for board and toolchain setup.

## Documentation and examples

- [Experiment configuration](src/config/README.md): objectives, runtime schedules,
  constraints, and search settings.
- [Case studies](src/config/case_study_configs/README.md): proxy versus measured
  energy selection and runtime-schedule comparisons.
- [Optimizers](src/crest/optimizers/README.md): native and LLM candidate generation,
  provider setup, and extension instructions.
- [Candidate replay](src/README.md#pareto-hil-replay) and
  [offline analysis](analysis_scripts/README.md): evaluate and inspect saved experiments.

## Contributing

Questions, bug reports, and focused contributions are welcome through
[GitHub issues](https://github.com/nesl/crest/issues) and
[pull requests](https://github.com/nesl/crest/pulls). For a bug report, include the
configuration, reproduction steps, software versions, and target board.

Component guides linked from the [source guide](src/README.md) include extension
steps and completion checklists. Run `make test` in the installed environment
before submitting code changes.

## License

Repo-owned CREST code, documentation, configuration, and tests are licensed
under the BSD 3-Clause License unless otherwise noted. See [LICENSE](LICENSE)
for the full license text.

Important exceptions and separately governed materials:

- STM32/ST/CMSIS material under
  [`sketches/stm32/crest_stm32_lrun/`](sketches/stm32/crest_stm32_lrun/)
  remains subject to the upstream STMicroelectronics, CMSIS, and related
  license terms documented in that directory.
- STM32 vendor-derived and tool-generated build recipe files in the LRUN
  workspace are not relicensed by CREST headers.
- The Arduino TensorFlow Lite Micro dependency tracked as the
  [`tools/arduino-user/libraries/Arduino_TensorFlowLite`](tools/arduino-user/libraries/Arduino_TensorFlowLite)
  submodule remains governed by its upstream Apache-2.0 license and is not
  relicensed by CREST.
- OxIOD split metadata and upstream OxIOD readme snippets are dataset-related
  materials; raw OxIOD data is downloaded separately and is not stored in this
  repository.
- UrbanSound8K downloaded audio and generated feature caches are governed by
  the UrbanSound8K dataset terms and are not stored in this repository.
