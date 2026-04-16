# TinyODOM STM32 FSBL Template

This directory contains the canonical STM32 FSBL template used by the
TinyODOM STM32 backend for `STM32_NUCLEO_N657X0_Q`.

The Python side of that backend is split as follows:

- `src/tinyodom/microcontrollers/stm32_nucleo_n657x0.py`
  The concrete TinyODOM `DeviceInterface` backend for this board.
- `src/tinyodom/microcontrollers/stm32_cube_clt.py`
  Shared STM32 build/debug-load/toolchain helpers.
- `src/tinyodom/microcontrollers/stm32_runtime.py`
  Shared STM32 direct-serial runtime protocol and telemetry parsing helpers.

`setup_stm32.sh` is the source of truth for rebuilding the template. It
refreshes the vendor-owned files from the pinned `tools/stm32/STM32CubeN6`
checkout into this self-contained template and preserves the repo-owned
TinyODOM overlay files.

## What Lives Here

- `FSBL/`
  The canonical firmware template staged by the backend before ST Edge AI
  outputs are copied in.
- `fsbl_ownership_manifest.tsv`
  Ownership table used by `setup_stm32.sh` to distinguish vendor-copy,
  vendor-derived, TinyODOM-owned, generated, and CLI build-recipe files.

## Build Model

- The backend stages this template into a per-candidate workspace under
  `tinyodom_tcn/stm32/...`.
- ST Edge AI generated `network*.c/.h` files are copied into that staged
  workspace during candidate preparation.
- The staged firmware also receives a generated
  `FSBL/Inc/tcn_dut_phase_config.h` header that fixes the CPU clock preset and
  keeps the current production flow in `back_to_back` mode.
- The committed `FSBL/Debug` makefiles are retained so the canonical template
  remains CLI-buildable without regenerating CubeIDE metadata.
- After `./setup_stm32.sh` runs, the template no longer depends on live include
  or source paths under `tools/stm32/STM32CubeN6`; those files are copied into
  `FSBL/Drivers/...` and travel with staged backend copies.

## Common Commands

From the repo root:

```bash
# Refresh the canonical template and the pinned STM32CubeN6 checkout.
./setup_stm32.sh

# Clean-build the canonical template when generated AI files are present.
make -C sketches/stm32/tinyodom_tcn_stm32/FSBL/Debug clean all -j1
```

The analysis scripts under `analysis_scripts/stm32_example_project/` can still
be used as optional smoke-test helpers, but this directory is the production
STM template that the backend targets.
