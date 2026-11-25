# Run with hardware

The training client proposes and trains candidates. A server on the board-connected host compiles, uploads, and measures them. You can run both processes on one machine or keep training on a separate host.

## Prepare the device host

Complete the [environment and dataset setup](setup.md). Dataset paths and any cached inputs must resolve on the hosts that use them. Connect the development board. For energy measurement, use the CREST harness with an INA228 in the DUT power path and GPIO measurement markers. Follow the board-specific deployment and measurement instructions for the checkout you run. [Microcontroller bring-up guide](https://github.com/nesl/crest/blob/main/src/crest/microcontrollers/README.md).

For Arduino targets or an Arduino measurement harness, install the repo-local Arduino CLI and activate its hooks.

```bash
make arduino-setup
conda deactivate
conda activate crest
arduino-cli --config-file tools/arduino-cli.yaml version
arduino-cli --config-file tools/arduino-cli.yaml core list
```

Check for the core matching the DUT or harness (`arduino:mbed_nano` or `arduino:mbed_portenta`). For an STM32 DUT, put the STM32CubeCLT tools, CubeProgrammer and signing CLI, and `stedgeai` on the device host’s `PATH`. Run `make stm32-setup` to validate the tools and prepare the CubeN6 workspace. [STM32 host requirements](https://github.com/nesl/crest/blob/main/src/crest/microcontrollers/README.md#required-host-tools).

## Set up the experiment

| Workload | Starting YAML |
| --- | --- |
| OxIOD on Arduino | [Nano 33 BLE Sense](https://github.com/nesl/crest/blob/main/src/config/nas_config_ble.yaml) or [Portenta H7](https://github.com/nesl/crest/blob/main/src/config/nas_config_portenta.yaml) |
| OxIOD on STM32 | [NUCLEO-N657X0-Q](https://github.com/nesl/crest/blob/main/src/config/nas_config_stm32.yaml) |
| UrbanSound8K on a board | [Portenta H7](https://github.com/nesl/crest/blob/main/src/config/nas_config_audio_portenta.yaml) or [STM32](https://github.com/nesl/crest/blob/main/src/config/nas_config_audio_stm32.yaml) |

Copy the selected YAML to `hil-study.yaml` on both hosts. Set `device.serial_port` and, for harness measurement, `device.harness_serial_port` to the actual device-host paths. On Portenta, verify the selected core. Keep the board, workload, measurement schedule, and scoring settings aligned across hosts.

## Start the server and run a smoke test

On the device host, from the repository root with the environment active:

```bash
python src/hil_server.py --config hil-study.yaml
```

For separate hosts with the shipped loopback endpoint, open a reverse SSH tunnel **from the device host to the training host** in another terminal. Replace `user@gpu-server` with your SSH login and leave the tunnel running. For a single-host setup, skip this step.

```bash
ssh -N -o ExitOnForwardFailure=yes -R 6001:127.0.0.1:6001 user@gpu-server
```

On the training host or in a second terminal on the same machine:

```bash
python src/nas_model_client.py --config hil-study.yaml \
  --smoke-test 3 --study-name board_smoke
```

Check the server log for completed measurement requests and the client trial log for hardware status, finite objective inputs, and any feasibility or prune decisions. For energy objectives, verify that the expected measurement phase returned energy. If the client times out, check server startup and the tunnel before rerunning. Once the smoke run has yielded a usable measurement, launch with the reviewed search budget, a fresh study name, and no `--smoke-test` flag.
