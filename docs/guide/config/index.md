# Choose a configuration

Start with the closest shipped YAML. Its scoring policy determines which measurements the run needs. For a first run on a workstation, the FLOPs/RMSE example disables both board measurement and compile-only requests.

| Study | Configuration | What it needs |
| --- | --- | --- |
| OxIOD validation error and FLOPs | [Desktop FLOPs](https://github.com/nesl/crest/blob/main/src/config/nas_config_flops_rmse.yaml) | Prepared OxIOD data |
| OxIOD validation error and memory traffic | [Desktop memory proxy](https://github.com/nesl/crest/blob/main/src/config/nas_config_memory_proxy.yaml) | Prepared OxIOD data |
| OxIOD with board measurements | [STM32](https://github.com/nesl/crest/blob/main/src/config/nas_config_stm32.yaml) · [Nano 33 BLE Sense](https://github.com/nesl/crest/blob/main/src/config/nas_config_ble.yaml) · [Portenta H7](https://github.com/nesl/crest/blob/main/src/config/nas_config_portenta.yaml) | Configured board host and measurement harness |
| UrbanSound8K audio classification | [STM32](https://github.com/nesl/crest/blob/main/src/config/nas_config_audio_stm32.yaml) · [Portenta H7](https://github.com/nesl/crest/blob/main/src/config/nas_config_audio_portenta.yaml) | Prepared feature cache and board setup as shipped |

## Find a setting

- [Workload and training](workload.md) covers component names, quantization and training budgets.
- [Target and schedule](target.md) covers HIL, compile-only requests and board-specific options.
- [Objectives and feasibility](search.md) explains how CREST ranks candidates and handles deadline misses.
- [Outputs and connections](outputs.md) covers artifact names, HIL endpoints and logging.

[Run your first study](../run/first-study.md)
