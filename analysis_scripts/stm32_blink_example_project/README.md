# STM32 Blink Example Project

This package contains a working STM32N6 blink example for the
`NUCLEO-N657X0-Q` board inside `tinyodom-ex`.

It started as a fresh STM32CubeIDE project created from scratch and then was
adapted from the STM32CubeN6 firmware package Template FSBL project. The result
is a repo-local, self-contained STM32CubeIDE project that builds and loads in
development boot mode.

## What This Is For

- Learning the STM32CubeIDE project flow on STM32N6.
- Preserving a known-good board bring-up target inside the repo.
- Establishing the first STM32-oriented project shape for future TinyODOM
  backend work.
- Documenting the gap between a fresh "empty" STM32CubeIDE project and the
  full vendor template build wiring.

## Current Status

- The project builds successfully in STM32CubeIDE.
- The board can be debug-loaded and run from the GUI in development mode.
- The package now includes a Python wrapper for the current safe CLI
  build-and-debug-load workflow.
- The Python wrapper has been confirmed to rebuild the project and load/run it
  on the board from the command line.

## What A New User Needs

This repo copy is enough to open the project and inspect the source, but a new
user needs more than just this folder to build and load it.

Required:

- [STM32CubeCLT](https://www.st.com/en/development-tools/stm32cubeclt.html)
- the STM32Cube firmware package for STM32N6 from
  [STM32CubeN6](https://github.com/STMicroelectronics/STM32CubeN6), currently
  expected at:
  - `/home/<user>/STM32Cube_FW_N6_V1.3.0`
- a connected `NUCLEO-N657X0-Q` board with ST-LINK access
- Linux permissions that allow access to the ST-LINK USB device

Important nuance:

- the committed project is self-contained in terms of local `Src/` files, but
  the build still pulls headers from the STM32Cube firmware package through the
  include paths in `.cproject`
- the Python wrapper has default tool paths for local CLT install, but
  those can be overridden with flags or environment variables on another
  machine
- CubeIDE is useful for creating or inspecting projects, but it is not required
  to run the current committed project once the project files already exist

For the current wrapper, the minimum practical requirement is:

- STM32CubeCLT installed
- the STM32CubeN6 firmware package available
- Python available in the `tinyodom-ex` environment

CubeIDE is optional for a new user unless they want to regenerate or edit the
project through the GUI.

## Package Layout

- `IDE_SETUP_CHECKLIST.md`
  - The original step-by-step CubeIDE creation notes.
- `stm32_blink_example_project/`
  - The generated STM32CubeIDE container project.
- `stm32_blink_example_project/FSBL/`
  - The real working FSBL subproject.

Important files:

- [`stm32_blink_example_project/FSBL/.cproject`](/home/joe/Projects/tinyodom-ex/analysis_scripts/stm32_blink_example_project/stm32_blink_example_project/FSBL/.cproject)
- [`stm32_blink_example_project/FSBL/.project`](/home/joe/Projects/tinyodom-ex/analysis_scripts/stm32_blink_example_project/stm32_blink_example_project/FSBL/.project)
- [`stm32_blink_example_project/FSBL/stm32_blink_example_project_FSBL.launch`](/home/joe/Projects/tinyodom-ex/analysis_scripts/stm32_blink_example_project/stm32_blink_example_project/FSBL/stm32_blink_example_project_FSBL.launch)
- [`stm32_blink_example_project/FSBL/STM32N657X0HXQ_AXISRAM2_fsbl.ld`](/home/joe/Projects/tinyodom-ex/analysis_scripts/stm32_blink_example_project/stm32_blink_example_project/FSBL/STM32N657X0HXQ_AXISRAM2_fsbl.ld)

## Provenance

The project was adapted from the STM32CubeN6 firmware package Template project
for `NUCLEO-N657X0-Q`.

Vendor application-layer source/header files were taken from:

- [`FSBL/Src/main.c`](/home/joe/STM32Cube_FW_N6_V1.3.0/Projects/NUCLEO-N657X0-Q/Templates/Template/FSBL/Src/main.c)
- [`FSBL/Src/stm32n6xx_hal_msp.c`](/home/joe/STM32Cube_FW_N6_V1.3.0/Projects/NUCLEO-N657X0-Q/Templates/Template/FSBL/Src/stm32n6xx_hal_msp.c)
- [`FSBL/Src/stm32n6xx_it.c`](/home/joe/STM32Cube_FW_N6_V1.3.0/Projects/NUCLEO-N657X0-Q/Templates/Template/FSBL/Src/stm32n6xx_it.c)
- [`FSBL/Src/system_stm32n6xx_fsbl.c`](/home/joe/STM32Cube_FW_N6_V1.3.0/Projects/NUCLEO-N657X0-Q/Templates/Template/FSBL/Src/system_stm32n6xx_fsbl.c)
- [`FSBL/Inc/main.h`](/home/joe/STM32Cube_FW_N6_V1.3.0/Projects/NUCLEO-N657X0-Q/Templates/Template/FSBL/Inc/main.h)
- [`FSBL/Inc/stm32n6xx_hal_conf.h`](/home/joe/STM32Cube_FW_N6_V1.3.0/Projects/NUCLEO-N657X0-Q/Templates/Template/FSBL/Inc/stm32n6xx_hal_conf.h)
- [`FSBL/Inc/stm32n6xx_it.h`](/home/joe/STM32Cube_FW_N6_V1.3.0/Projects/NUCLEO-N657X0-Q/Templates/Template/FSBL/Inc/stm32n6xx_it.h)
- [`FSBL/Inc/stm32n6xx_nucleo_conf.h`](/home/joe/STM32Cube_FW_N6_V1.3.0/Projects/NUCLEO-N657X0-Q/Templates/Template/FSBL/Inc/stm32n6xx_nucleo_conf.h)

The project was then made self-contained by copying in the HAL/BSP/runtime
source files that the original ST template normally pulls in through linked
resources.

## What Was Needed To Make It Work

The initial fresh `STM32CubeIDE Empty Project` was too bare to build the vendor
FSBL application files directly.

The final working project needed:

- corrected HAL/CMSIS/BSP include paths in `.cproject`
- STM32 HAL defines such as `USE_HAL_DRIVER` and `STM32N657xx`
- local HAL/BSP source files in `FSBL/Src`
- `syscalls.c` and `sysmem.c` for clean newlib-nano builds

This means the current project is no longer just a thin transplant. It is a
portable, repo-owned STM32Cube-style example.

## Python CLI Wrapper

This package includes a repo-local Python helper:

- [`build_and_upload_stm32_blink.py`](/home/joe/Projects/tinyodom-ex/analysis_scripts/stm32_blink_example_project/build_and_upload_stm32_blink.py)

Default behavior:

- builds with `make -C Debug`
- starts `ST-LINK_gdbserver`
- connects with `arm-none-eabi-gdb`
- loads `Debug/stm32_blink_example_project_FSBL.elf`
- continues execution by default
- tears down the server before exiting

Default command from the repo root:

```bash
python analysis_scripts/stm32_blink_example_project/build_and_upload_stm32_blink.py
```

Recommended debug command while iterating on the wrapper or tool paths:

```bash
python analysis_scripts/stm32_blink_example_project/build_and_upload_stm32_blink.py --verbose
```

Useful flags:

```bash
python analysis_scripts/stm32_blink_example_project/build_and_upload_stm32_blink.py --clean
python analysis_scripts/stm32_blink_example_project/build_and_upload_stm32_blink.py --no-build
python analysis_scripts/stm32_blink_example_project/build_and_upload_stm32_blink.py --no-run
python analysis_scripts/stm32_blink_example_project/build_and_upload_stm32_blink.py --verbose
```

Tool-path override examples:

```bash
python analysis_scripts/stm32_blink_example_project/build_and_upload_stm32_blink.py \
  --gdbserver /path/to/ST-LINK_gdbserver \
  --gdb /path/to/arm-none-eabi-gdb \
  --cubeprog-bin /path/to/STM32CubeProgrammer/bin
```

Equivalent environment-variable overrides:

```bash
STM32_GDBSERVER=/path/to/ST-LINK_gdbserver \
STM32_GDB=/path/to/arm-none-eabi-gdb \
STM32_CUBEPROG_BIN=/path/to/STM32CubeProgrammer/bin \
python analysis_scripts/stm32_blink_example_project/build_and_upload_stm32_blink.py
```

On another machine, it is normal to need path overrides if the local CLT
installation is not under `/home/joe/st/...`.

Important: this is a development-memory debug load, not persistent flashing.
It does not sign binaries, program external flash, or modify device security
state.

### Wrapper Notes

The working wrapper behavior now mirrors the GUI launch more closely:

- it starts `ST-LINK_gdbserver` with SWD, connect-under-reset, and access port
  `1`
- it uses `arm-none-eabi-gdb` to `load` the ELF and resume execution
- it detects server readiness from the GDB server log rather than by probing
  the TCP port directly

That last point matters. A raw TCP readiness probe caused the non-persistent
GDB server to believe a debugger had briefly connected and disconnected, which
made it shut down before the real `gdb` session attached.

## First Run Notes

- Use development boot mode first:
  - `BOOT1 = 2-3`
  - `BOOT0` does not matter for this debug flow
- Build and launch with `Run > Debug` in STM32CubeIDE.
- Do not start with the external-flash boot flow for this example.

Reference:

- [`Template README.md`](/home/joe/STM32Cube_FW_N6_V1.3.0/Projects/NUCLEO-N657X0-Q/Templates/Template/README.md#L55)

## Related Notes

The longer-form bring-up history, what failed, and the current CLI understanding
are documented in:

- [`memory_stm.md`](/home/joe/Projects/tinyodom-ex/memory_stm.md)
