# Source Overview

This folder contains the Python entry points and core library code for the
TinyODOM-EX training, hardware-in-the-loop, and deployment flow.

## Top-level entry points

- `hil_server.py`
  - ZeroMQ REP server that builds model variants, exports TFLite/C++, flashes
    firmware, and returns HIL metrics.
- `nas_model_client.py`
  - ZeroMQ REQ client that runs NAS/training workflows and queries the HIL
    server for hardware metrics.
- `nas_config.yaml`
  - Default configuration for dataset paths, training, device selection,
    network settings, and output paths.
- `nas_config_ble.yaml`
  - Alternate BLE-oriented configuration.
- `two_board_hil_notes.txt`
  - Notes on the two-board DUT/harness measurement setup.

## Python package

The `tinyodom/` package holds the reusable logic shared by the scripts above.

- `tinyodom/data.py`
  - OxIOD dataset import and split handling.
- `tinyodom/model.py`
  - Model construction, config loading, metric-collection request building, and
    training utilities.
- `tinyodom/hardware.py`
  - Export, compile, upload, and metric normalization helpers.
- `tinyodom/hil_protocol.py`
  - DUT/harness serial handshake and telemetry collection protocol.
- `tinyodom/devices.py`
  - Device abstraction layer and board-spec plumbing.
- `tinyodom/geometry.py`
  - Geometry and trajectory helper functions.
- `tinyodom/errors.py`
  - Shared error-code definitions and helpers.
- `tinyodom/microcontrollers/`
  - Board-specific Arduino and non-Arduino backends.
  - Arduino boards follow the integration guide in
    `src/tinyodom/microcontrollers/README.md`.
  - The STM32 backend is split into:
    - `stm32_nucleo_n657x0.py` for the concrete `DeviceInterface`
      implementation
    - `stm32_cube_clt.py` for build/load/toolchain helpers
    - `stm32_runtime.py` for direct-serial runtime protocol handling

## Notes

- Generated or cache-like artifacts such as `__pycache__/` and
  `tinyodom.egg-info/` may appear here during local development.
- Analysis utilities that sit outside the core runtime live under
  `analysis_scripts/`, not in this folder.
