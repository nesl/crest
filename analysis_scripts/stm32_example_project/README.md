# STM32 Example Project

This package contains the current STM32N6 example assets for the
`NUCLEO-N657X0-Q` board inside `tinyodom-ex`.

It started as a fresh STM32CubeIDE project and was then adapted from the
STM32CubeN6 Template FSBL project. The committed result is a repo-local,
self-contained CubeIDE project that builds through the generated `FSBL/Debug`
makefiles and debug-loads through a Python wrapper. It also now includes the
host-only ST Edge AI Phase 0 probe used to validate the CPU-first TinyODOM
path without touching the board.

The outer package has been renamed to `stm32_example_project` to reflect that
it now carries more than the original blink-only bring-up. The nested
STM32CubeIDE project name remains `stm32_blink_example_project` for now so the
existing generated makefiles, launch metadata, and debug workflow stay stable.

## Current Status

- The project builds successfully from the committed `FSBL/Debug` makefiles.
- The board can be debug-loaded and run in development boot mode.
- The Python wrapper rebuilds the project, starts `ST-LINK_gdbserver`, loads the
  ELF with `arm-none-eabi-gdb`, and resumes execution.
- The ST Edge AI probe script can reproduce the current host-only Phase 0
  findings for STM32N6 CPU generation.
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
python analysis_scripts/stm32_example_project/build_and_upload_stm32_blink.py
```

Useful flags:

```bash
python analysis_scripts/stm32_example_project/build_and_upload_stm32_blink.py --clean
python analysis_scripts/stm32_example_project/build_and_upload_stm32_blink.py --no-build
python analysis_scripts/stm32_example_project/build_and_upload_stm32_blink.py --no-run
python analysis_scripts/stm32_example_project/build_and_upload_stm32_blink.py --verbose
```

Explicit path overrides still work:

```bash
python analysis_scripts/stm32_example_project/build_and_upload_stm32_blink.py \
  --gdbserver /path/to/ST-LINK_gdbserver \
  --gdb /path/to/arm-none-eabi-gdb \
  --cubeprog-bin /path/to/STM32CubeProgrammer/bin
```

Important: this is a development-memory debug load, not persistent flashing. It
does not sign binaries, program external flash, or change device security
state.

## Running The Phase 0 Smoke Test

The smoke test builds, loads, and runs the toy AI project end-to-end and
confirms the expected UART telemetry in one command:

```bash
python analysis_scripts/stm32_example_project/smoke_test_stm32_toy_ai.py --clean
```

What the smoke test does:

- Opens `/dev/ttyACM0` at 115200 baud **before** triggering the load, so no
  early output is missed
- Optionally cleans and rebuilds the toy AI project
- Loads the toy AI ELF through the existing `ST-LINK_gdbserver` +
  `arm-none-eabi-gdb` RAM/debug-load flow
- Prints every received serial line to the terminal in real time
- Waits up to 30 seconds for all three Phase 0 tokens:
  - `STM32_AI_INIT=OK`
  - `STM32_AI_RUN=OK`
  - `STM32_AI_LATENCY_CYCLES=<value>`
- Exits 0 (PASS) if all tokens arrive, 1 (FAIL) otherwise

Useful flags:

```bash
# Skip rebuild if the ELF is already current:
python analysis_scripts/stm32_example_project/smoke_test_stm32_toy_ai.py --no-build

# Use a different serial port or baud rate:
python analysis_scripts/stm32_example_project/smoke_test_stm32_toy_ai.py --port /dev/ttyACM1 --baud 115200

# Extend the token-wait timeout (default 30 s):
python analysis_scripts/stm32_example_project/smoke_test_stm32_toy_ai.py --serial-timeout 60
```

Phase 0 is considered complete when this script exits with status 0.

## Running The Phase 0 Probe

The package also contains a host-only ST Edge AI probe:

```bash
python analysis_scripts/stm32_example_project/run_stedgeai_phase0_probe.py --clean
```

What the probe does:

- runs `stedgeai analyze` on the representative TinyODOM `.tflite`
- runs a CM55-style binary `generate` pass with `--c-api legacy`
- runs a non-binary legacy `generate` pass
- captures logs and generated artifacts under `/tmp` by default

This probe is intentionally board-free. It does not use `ST-LINK_gdbserver`,
does not load firmware, and does not program flash.

## Project Layout

- `build_and_upload_stm32_blink.py`
  - CLI wrapper for build + debug-load
- `run_stedgeai_phase0_probe.py`
  - host-only ST Edge AI probe for the STM32N6 CPU-first Phase 0 work
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

