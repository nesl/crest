# STM32 Blink Example Project

This package contains a working STM32N6 blink example for the `NUCLEO-N657X0-Q`
board inside `tinyodom-ex`.

It started as a fresh STM32CubeIDE project and was then adapted from the
STM32CubeN6 Template FSBL project. The committed result is a repo-local,
self-contained CubeIDE project that builds through the generated `FSBL/Debug`
makefiles and debug-loads through a Python wrapper.

## Current Status

- The project builds successfully from the committed `FSBL/Debug` makefiles.
- The board can be debug-loaded and run in development boot mode.
- The Python wrapper rebuilds the project, starts `ST-LINK_gdbserver`, loads the
  ELF with `arm-none-eabi-gdb`, and resumes execution.
- The STM32 firmware headers are expected under `tools/stm32/STM32CubeN6`,
  pinned to `v1.3.0`.

## Setup

The STM32 flow is split on purpose:

- `STM32CubeCLT` stays installed outside the repo
- STM32CubeN6 firmware is cloned into `tools/stm32/STM32CubeN6`

Before the first STM32 bootstrap run, ensure these `STM32CubeCLT` tools are on
your shell `PATH`:

- `ST-LINK_gdbserver`
- `arm-none-eabi-gdb`
- `STM32_Programmer_CLI`

Then bootstrap the repo-local STM32 dependencies:

```bash
make stm32-setup
```

What `make stm32-setup` does:

- validates `ST-LINK_gdbserver`, `arm-none-eabi-gdb`, and `STM32_Programmer_CLI`
  on `PATH`
- clones or repairs `tools/stm32/STM32CubeN6`
- checks out the pinned firmware baseline `v1.3.0`

## Running The Wrapper

Default command from the repo root:

```bash
python analysis_scripts/stm32_blink_example_project/build_and_upload_stm32_blink.py
```

Useful flags:

```bash
python analysis_scripts/stm32_blink_example_project/build_and_upload_stm32_blink.py --clean
python analysis_scripts/stm32_blink_example_project/build_and_upload_stm32_blink.py --no-build
python analysis_scripts/stm32_blink_example_project/build_and_upload_stm32_blink.py --no-run
python analysis_scripts/stm32_blink_example_project/build_and_upload_stm32_blink.py --verbose
```

Explicit path overrides still work:

```bash
python analysis_scripts/stm32_blink_example_project/build_and_upload_stm32_blink.py \
  --gdbserver /path/to/ST-LINK_gdbserver \
  --gdb /path/to/arm-none-eabi-gdb \
  --cubeprog-bin /path/to/STM32CubeProgrammer/bin
```

Important: this is a development-memory debug load, not persistent flashing. It
does not sign binaries, program external flash, or change device security
state.

## Project Layout

- `build_and_upload_stm32_blink.py`
  - CLI wrapper for build + debug-load
- `stm32_blink_example_project/FSBL/`
  - the real working STM32CubeIDE FSBL subproject
- `stm32_blink_example_project/FSBL/Debug/`
  - committed generated make metadata required by the wrapper

The FSBL project now expects the STM32CubeN6 firmware headers at:

```text
tools/stm32/STM32CubeN6
```

## First Run Notes

- Use development boot mode:
  - `BOOT1 = 2-3`
  - `BOOT0` does not matter for this debug flow
- Start with the debug-load flow, not the external-flash boot flow

## Related Notes

The tracked setup, wrapper usage, and first-run notes for this STM32 blink
example are documented in this README.
