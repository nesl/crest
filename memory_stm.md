# STM32 N6 Memory

## Purpose

This file records what was learned while bringing up an STM32CubeIDE-based
project for the `NUCLEO-N657X0-Q` board inside `tinyodom-ex`.

The immediate goal was to create a fresh STM32CubeIDE blink project under
`analysis_scripts/`, get it building and debugging in the GUI, and understand
what will be needed later for a TinyODOM STM32 backend.

The resulting working project is:

- [`analysis_scripts/stm32_blink_example_project/stm32_blink_example_project/FSBL`](/home/joe/Projects/tinyodom-ex/analysis_scripts/stm32_blink_example_project/stm32_blink_example_project/FSBL)

This is a development-boot, debug-loaded STM32N6 FSBL project, not an external
flash boot image.

## Current Status

- The project builds cleanly in STM32CubeIDE.
- The board can be loaded and run from the GUI in development mode.
- The blink timing was modified and successfully re-uploaded through the GUI.
- The repo now includes a Python wrapper for the current safe CLI build +
  debug-load flow.
- The Python wrapper was then debugged and confirmed to rebuild the project and
  load/run it successfully from the command line.
- The repo now also includes a setup/bootstrap flow that keeps STM32CubeN6
  firmware under `tools/stm32/STM32CubeN6` and uses Conda hooks to expose the
  required STM32CubeCLT paths.
- The committed CubeIDE project and generated `Debug` makefiles were normalized
  to use that repo-local firmware path instead of a user-home STM32 firmware
  install.

## Board And Boot Mode

For this STM32N6 template-style FSBL flow:

- `BOOT1 = 2-3`
- `BOOT0` does not matter for development debug mode

This is required because the project is meant to be loaded into target memory
for debug, not immediately booted from external flash.

Reference:

- [`Template README.md`](/home/joe/STM32Cube_FW_N6_V1.3.0/Projects/NUCLEO-N657X0-Q/Templates/Template/README.md#L55)

## How The New Project Was Created

The project was created as a fresh STM32CubeIDE project inside the repo, not by
importing ST's existing Eclipse project metadata.

Working path in this CubeIDE version:

1. Open `Create/Import STM32 Project`
2. Open the `Create New STM32 Project` folder
3. Choose `STM32CubeIDE Empty Project`
4. Select board `NUCLEO-N657X0-Q`
5. Create the new project under:
   - `/home/joe/Projects/tinyodom-ex/analysis_scripts/stm32_blink_example_project/stm32_blink_example_project`

Important nuance:

- In this installed CubeIDE version, the UI path was not the older
  `File > New > STM32 Project` path.
- The menu flow was more indirect than older docs suggest.

## What We Intended To Reuse

The plan was to reuse the vendor template's application-layer FSBL files while
keeping CubeIDE-generated metadata fresh.

The source/header files reused from the STM32CubeN6 vendor template were:

- [`FSBL/Src/main.c`](/home/joe/STM32Cube_FW_N6_V1.3.0/Projects/NUCLEO-N657X0-Q/Templates/Template/FSBL/Src/main.c)
- [`FSBL/Src/stm32n6xx_hal_msp.c`](/home/joe/STM32Cube_FW_N6_V1.3.0/Projects/NUCLEO-N657X0-Q/Templates/Template/FSBL/Src/stm32n6xx_hal_msp.c)
- [`FSBL/Src/stm32n6xx_it.c`](/home/joe/STM32Cube_FW_N6_V1.3.0/Projects/NUCLEO-N657X0-Q/Templates/Template/FSBL/Src/stm32n6xx_it.c)
- [`FSBL/Src/system_stm32n6xx_fsbl.c`](/home/joe/STM32Cube_FW_N6_V1.3.0/Projects/NUCLEO-N657X0-Q/Templates/Template/FSBL/Src/system_stm32n6xx_fsbl.c)
- [`FSBL/Inc/main.h`](/home/joe/STM32Cube_FW_N6_V1.3.0/Projects/NUCLEO-N657X0-Q/Templates/Template/FSBL/Inc/main.h)
- [`FSBL/Inc/stm32n6xx_hal_conf.h`](/home/joe/STM32Cube_FW_N6_V1.3.0/Projects/NUCLEO-N657X0-Q/Templates/Template/FSBL/Inc/stm32n6xx_hal_conf.h)
- [`FSBL/Inc/stm32n6xx_it.h`](/home/joe/STM32Cube_FW_N6_V1.3.0/Projects/NUCLEO-N657X0-Q/Templates/Template/FSBL/Inc/stm32n6xx_it.h)
- [`FSBL/Inc/stm32n6xx_nucleo_conf.h`](/home/joe/STM32Cube_FW_N6_V1.3.0/Projects/NUCLEO-N657X0-Q/Templates/Template/FSBL/Inc/stm32n6xx_nucleo_conf.h)

We kept the new project's generated:

- `.project`
- `.cproject`
- `.settings`
- startup assembly
- linker script
- launch config

## What Went Wrong

### 1. The Fresh Project Was Too Empty

The new project built as an `STM32CubeIDE Empty Project`, which produced only a
minimal FSBL shell.

After copying in the vendor template application files, the first build failed
because the project did not know where to find the STM32 HAL/CMSIS/BSP headers:

- `stm32n6xx_hal.h`
- `stm32n6xx.h`

The empty project only had `../Inc` on its include path and did not carry over
the template's HAL driver wiring.

### 2. We Initially Underestimated How Metadata-Heavy STM32Cube Projects Are

The wrong assumption was:

- "If the `.c` and `.h` files are copied, the project should build."

That is not true for this STM32Cube template. The original ST template also
depends on:

- HAL include paths
- CMSIS include paths
- BSP include paths
- HAL/BSP source files
- syscall stubs

### 3. Relative Include Paths Were Wrong The First Time

When the missing include paths were added manually, they were first written
relative to the FSBL project root.

That was still wrong because the generated `make` files compile from:

- `FSBL/Debug`

So those include paths had to be corrected to be relative to the build
directory, not just the project root.

### 4. The Build Produced An ELF But CubeIDE Still Reported Failure

After the HAL/BSP include-path issue was fixed, the project linked and produced
an ELF, but CubeIDE still reported a failed build because the no-syscall
newlib-nano runtime diagnostics were still present.

That was resolved by adding:

- `syscalls.c`
- `sysmem.c`

## What We Changed To Make It Work

### A. Replaced The Generated App-Layer Source/Header Files

The vendor template application-layer files were pasted into the new project's:

- [`FSBL/Src`](/home/joe/Projects/tinyodom-ex/analysis_scripts/stm32_blink_example_project/stm32_blink_example_project/FSBL/Src)
- [`FSBL/Inc`](/home/joe/Projects/tinyodom-ex/analysis_scripts/stm32_blink_example_project/stm32_blink_example_project/FSBL/Inc)

### B. Made The Empty Project Self-Contained

To make this repo-owned project build without depending on Eclipse linked
resources, the HAL and BSP sources that the original ST template uses were
copied into the local `FSBL/Src` folder:

- `stm32n6xx_hal.c`
- `stm32n6xx_hal_cortex.c`
- `stm32n6xx_hal_dma.c`
- `stm32n6xx_hal_dma_ex.c`
- `stm32n6xx_hal_exti.c`
- `stm32n6xx_hal_gpio.c`
- `stm32n6xx_hal_pwr.c`
- `stm32n6xx_hal_pwr_ex.c`
- `stm32n6xx_hal_rcc.c`
- `stm32n6xx_hal_rcc_ex.c`
- `stm32n6xx_nucleo.c`

This is why the local `Src` folder now looks larger than the vendor `FSBL/Src`
folder.

### C. Patched `.cproject`

The project's [`FSBL/.cproject`](/home/joe/Projects/tinyodom-ex/analysis_scripts/stm32_blink_example_project/stm32_blink_example_project/FSBL/.cproject)
was updated to add:

- HAL/CMSIS/BSP include paths
- `USE_HAL_DRIVER`
- `STM32N657xx`

The include paths had to be correct relative to the generated `Debug/`
directory.

### D. Added Runtime Stub Files

The following files were added under local `FSBL/Src`:

- [`syscalls.c`](/home/joe/Projects/tinyodom-ex/analysis_scripts/stm32_blink_example_project/stm32_blink_example_project/FSBL/Src/syscalls.c)
- [`sysmem.c`](/home/joe/Projects/tinyodom-ex/analysis_scripts/stm32_blink_example_project/stm32_blink_example_project/FSBL/Src/sysmem.c)

These are standard STM32Cube-style newlib support files.

## Why The Original Template Looked Smaller

The original ST template did not actually avoid those files. It organized them
through Eclipse linked resources and multiple logical folders.

The original template build includes:

- application files under `Application/User`
- startup under `Application/Startup`
- HAL drivers under `Drivers/STM32N6xx_HAL_Driver`
- BSP under `Drivers/BSP/STM32N6xx_Nucleo`
- syscall stubs under `Application/User`

Evidence:

- [`Template FSBL .project`](/home/joe/STM32Cube_FW_N6_V1.3.0/Projects/NUCLEO-N657X0-Q/Templates/Template/STM32CubeIDE/FSBL/.project)
- [`Template objects.list`](/home/joe/STM32Cube_FW_N6_V1.3.0/Projects/NUCLEO-N657X0-Q/Templates/Template/STM32CubeIDE/FSBL/Debug/objects.list)
- [`Template syscalls.c`](/home/joe/STM32Cube_FW_N6_V1.3.0/Projects/NUCLEO-N657X0-Q/Templates/Template/STM32CubeIDE/FSBL/Application/User/syscalls.c)
- [`Template sysmem.c`](/home/joe/STM32Cube_FW_N6_V1.3.0/Projects/NUCLEO-N657X0-Q/Templates/Template/STM32CubeIDE/FSBL/Application/User/sysmem.c)

So the difference is:

- original template: logically split, linked resources
- repo copy: self-contained, local copies in one source tree

## What Is Actually Needed

### Definitely Needed For The Working Blink Project

- startup assembly
- linker script
- `main.c`
- `stm32n6xx_hal_msp.c`
- `stm32n6xx_it.c`
- `system_stm32n6xx_fsbl.c`
- `main.h`
- `stm32n6xx_hal_conf.h`
- `stm32n6xx_it.h`
- `stm32n6xx_nucleo_conf.h`
- HAL core/cortex/gpio/pwr/pwr_ex/rcc/rcc_ex
- `stm32n6xx_nucleo.c`

These are actively reflected in the final ELF through symbols like:

- `BSP_LED_Init`
- `BSP_LED_Toggle`
- `HAL_GPIO_*`
- `HAL_PWREx_*`
- `HAL_RCC_*`
- `HAL_Delay`

### Likely Optional For A Future Minimalized Blink

These are currently compiled because they came from the vendor baseline, but do
not appear to contribute meaningfully to the final blink behavior:

- `stm32n6xx_hal_dma.c`
- `stm32n6xx_hal_dma_ex.c`
- `stm32n6xx_hal_exti.c`

### Useful For Clean Builds But Not Blink Logic

- `syscalls.c`
- `sysmem.c`

These are not LED logic, but they are useful to keep the build clean in this
toolchain setup.

## Current Blink Behavior

The current project is configured to blink faster than the original template.
At one point, `main.c` was changed from the template's `200 ms` delays to
`50 ms` delays.

If behavior looks different from the vendor template, this timing change is one
reason why.

## How The Project Builds

This project is a generated-make STM32CubeIDE project.

Meaning:

- CubeIDE stores build settings in `.cproject`
- generated makefiles live in `FSBL/Debug`
- the actual compile is performed by `make`

Important generated files:

- [`FSBL/Debug/makefile`](/home/joe/Projects/tinyodom-ex/analysis_scripts/stm32_blink_example_project/stm32_blink_example_project/FSBL/Debug/makefile)
- [`FSBL/Debug/objects.list`](/home/joe/Projects/tinyodom-ex/analysis_scripts/stm32_blink_example_project/stm32_blink_example_project/FSBL/Debug/objects.list)

## How The Project Runs In The GUI

The GUI debug flow:

1. builds the ELF
2. starts `ST-LINK_gdbserver`
3. connects with `arm-none-eabi-gdb`
4. downloads `Debug/stm32_blink_example_project_FSBL.elf`
5. runs to `main`

Reference:

- [`stm32_blink_example_project_FSBL.launch`](/home/joe/Projects/tinyodom-ex/analysis_scripts/stm32_blink_example_project/stm32_blink_example_project/FSBL/stm32_blink_example_project_FSBL.launch#L18)
- [`stm32_blink_example_project_FSBL.launch`](/home/joe/Projects/tinyodom-ex/analysis_scripts/stm32_blink_example_project/stm32_blink_example_project/FSBL/stm32_blink_example_project_FSBL.launch#L29)
- [`stm32_blink_example_project_FSBL.launch`](/home/joe/Projects/tinyodom-ex/analysis_scripts/stm32_blink_example_project/stm32_blink_example_project/FSBL/stm32_blink_example_project_FSBL.launch#L45)
- [`stm32_blink_example_project_FSBL.launch`](/home/joe/Projects/tinyodom-ex/analysis_scripts/stm32_blink_example_project/stm32_blink_example_project/FSBL/stm32_blink_example_project_FSBL.launch#L63)
- [`stm32_blink_example_project_FSBL.launch`](/home/joe/Projects/tinyodom-ex/analysis_scripts/stm32_blink_example_project/stm32_blink_example_project/FSBL/stm32_blink_example_project_FSBL.launch#L77)

## Current CLI Understanding

The low-level CLI flow was researched first and then wrapped in a repo-local
Python script.

### Build

Fastest path with generated makefiles:

```bash
cd /home/joe/Projects/tinyodom-ex/analysis_scripts/stm32_blink_example_project/stm32_blink_example_project/FSBL
make -C Debug clean
make -C Debug -j"$(nproc)" all
```

Alternative headless build via CubeIDE:

```bash
/home/joe/st/stm32cubeide_2.1.1/headless-build.sh \
  -data /tmp/stm32cubeide-cli-ws \
  -import /home/joe/Projects/tinyodom-ex/analysis_scripts/stm32_blink_example_project/stm32_blink_example_project/FSBL \
  -cleanBuild stm32_blink_example_project_FSBL/Debug
```

### Load / Run In Development Mode

Safe CLI equivalent of the GUI debug flow:

Terminal 1:

```bash
/home/joe/st/stm32cubeclt_1.21.0/STLink-gdb-server/bin/ST-LINK_gdbserver \
  -d \
  -cp /home/joe/st/stm32cubeclt_1.21.0/STM32CubeProgrammer/bin
```

Terminal 2:

```bash
cd /home/joe/Projects/tinyodom-ex/analysis_scripts/stm32_blink_example_project/stm32_blink_example_project/FSBL
/home/joe/st/stm32cubeclt_1.21.0/GNU-tools-for-STM32/bin/arm-none-eabi-gdb Debug/stm32_blink_example_project_FSBL.elf
```

Then inside GDB:

```gdb
target remote localhost:61234
load
tbreak main
continue
```

This is the safer STM32N6 FSBL development-mode path because it mirrors the GUI
debug launch and does not attempt to create a persistent external-flash boot
image.

## Python Wrapper

The repo-local automation entry point is:

- [`analysis_scripts/stm32_blink_example_project/build_and_upload_stm32_blink.py`](/home/joe/Projects/tinyodom-ex/analysis_scripts/stm32_blink_example_project/build_and_upload_stm32_blink.py)

Default decisions chosen for the wrapper:

- build with generated `make` files, not CubeIDE headless build
- use the current local CLT paths by default, with CLI/env overrides
- load the ELF via `ST-LINK_gdbserver` + `arm-none-eabi-gdb`
- continue execution after load by default
- select access port `1` when starting `ST-LINK_gdbserver`, matching the
  working CubeIDE launch configuration

Useful behaviors:

- `--clean`
  - run `make -C Debug clean` before build
- `--no-build`
  - skip the build and reuse an existing ELF
- `--no-run`
  - load the ELF but do not resume execution after download
- `--verbose`
  - print the underlying subprocess commands

The wrapper intentionally uses the GDB server path instead of
`STM32_Programmer_CLI` because this project is currently being used in the
same development-memory debug flow as the GUI. That is the safer and more
direct match for this STM32N6 FSBL example.

Confirmed working invocation from the repo root:

```bash
python analysis_scripts/stm32_blink_example_project/build_and_upload_stm32_blink.py --verbose
```

### Wrapper Failure That Had To Be Fixed

The first version of the wrapper built correctly but failed during server
startup.

What it looked like:

- `make` completed successfully
- `ST-LINK_gdbserver` started
- the script waited for port `61234` by opening a raw TCP connection
- the server log showed:
  - `Waiting for debugger connection...`
  - `Debugger connected`
  - `Waiting for debugger connection...`
  - `Shutting down...`

The important lesson was that the readiness probe itself counted as a debugger
connection for this non-persistent GDB server. That caused the server to think
a debugger had attached and then disconnected before the real `gdb` process was
started.

The fix was:

- stop probing the TCP port directly
- wait for the server log to report `Waiting for debugger connection...`
- then start the real `arm-none-eabi-gdb` batch session

So the CLI failure was not a board issue and not a build issue. It was a
wrapper bug caused by using the wrong readiness-detection method for this GDB
server mode.

## What CLT Means In Practice

`STM32CubeCLT` is not one single command.

It is a bundle that provides:

- GNU build tools
- `arm-none-eabi-gdb`
- `ST-LINK_gdbserver`
- `STM32_Programmer_CLI`

So yes, this workflow can be done with CLT, but in multiple steps:

1. build
2. start gdbserver
3. start GDB
4. `load`
5. `continue`

## Safety Notes

For this project and board, the safe path so far is:

- development boot mode
- debug-load ELF into target memory

Avoid for now:

- OTP programming
- lifecycle/security changes
- option-byte experimentation
- external-flash boot signing/programming for this simple blink test

Those are separate flows and not needed for basic TinyODOM-oriented bring-up.

## Why This Matters For TinyODOM

This project is the first proof that a CubeIDE/STM32-style backend is feasible
inside `tinyodom-ex`.

The important outcome is not just "the board blinks." The important outcome is:

- a fresh STM32CubeIDE project can live inside the repo
- vendor FSBL application files can be adapted into that project
- the build can be made self-contained
- the debug flow can be reproduced conceptually from the CLI

This is the right first step before trying to turn TinyODOM's current
Arduino-centric pipeline into a second backend family for STM32.

## Bootstrap Update

After the first CLI wrapper was proven out, the next major issue was
portability.

The initial working STM32 flow still depended on two local-machine assumptions:

- STM32CubeCLT tool binaries under `/home/<user>/st/...`
- STM32CubeN6 firmware headers under `/home/<user>/STM32Cube_FW_N6_V1.3.0`

That was good enough for local experimentation, but not good enough for a clean
repo workflow where another user could clone the project and reproduce the STM32
path without hand-editing project files.

The repo was then updated to make the STM32 flow match the existing
Arduino-style tooling philosophy more closely:

- keep external vendor tooling outside the repo when it must be installed
  manually
- keep repo-owned firmware/package state under `tools/`
- expose repo-specific paths through setup scripts

The resulting STM32 bootstrap shape is now:

- `make stm32-setup`
- validate externally installed `STM32CubeCLT` tools on `PATH`:
  - `ST-LINK_gdbserver`
  - `arm-none-eabi-gdb`
  - `STM32_Programmer_CLI`
- clone or repair `STM32CubeN6` under:
  - `tools/stm32/STM32CubeN6`
- pin the firmware checkout to:
  - `v1.3.0`

This changed the project in three important ways:

1. The STM32 setup is now reproducible without relying on `/home/<user>/...`
   firmware paths.
2. The Python wrapper no longer has `joe`-specific CLT defaults baked in and
   now resolves the CLT tools from `PATH`.
3. The committed STM32 project metadata now points at a stable repo-local
   firmware location under `tools/`.

One additional nuance surfaced during this cleanup:

- the Python wrapper depends on the generated `FSBL/Debug` make metadata
- that metadata was originally ignored
- the repo now keeps the portable generated makefiles (`makefile`,
  `sources.mk`, `objects.mk`, `objects.list`, and `subdir.mk` files) while
  still ignoring rebuildable outputs like `.o`, `.elf`, `.map`, and `.list`

That means the STM32 example is now much closer to the Arduino flow in spirit:

- setup once
- keep repo-owned support files inside the repo tree
- then run the wrapper without hand-patching local paths

## Next Steps For TinyODOM

Recommended order:

1. Keep this blink project stable as a known-good board bring-up target.
2. Keep the STM32 bootstrap + wrapper flow stable as the supported CLI path.
3. Replace the blink loop with a minimal TinyODOM-style serial protocol.
4. Move generated model artifacts into the STM32 project.
5. Add a real STM32 device backend in Python.

### 1. Preserve A Known-Good Baseline

Before turning this into something more complex, keep one working revision that:

- builds in the GUI
- can be loaded onto the board
- blinks in development mode

This gives a clean recovery point if later TinyODOM integration breaks clocks,
GPIO, serial, or memory layout.

### 2. Preserve The Existing CLI Flow

That wrapper/bootstrap step is now in place, so the next requirement is to keep
it stable while the firmware evolves.

The STM32 path now depends on:

- `make -C Debug ...`
- `ST-LINK_gdbserver`
- `arm-none-eabi-gdb`
- `STM32_Programmer_CLI`
- repo-local firmware includes under `tools/stm32/STM32CubeN6`

Before model integration, treat that path as infrastructure:

- do not casually break the committed `Debug` make metadata
- do not reintroduce user-home firmware paths
- keep `make stm32-setup` + the Python wrapper as the normal non-GUI entry
  point

### 3. Replace Blink With A Minimal Host Protocol

Before trying to run a model, change the firmware from "blink forever" into
"bring up clocks/GPIO/UART, then speak a simple text protocol."

Suggested first protocol:

- print `READY`
- wait for a host trigger
- toggle one LED or run one dummy code path
- print timing/status text

That will prove that TinyODOM can talk to the board as a host-controlled
device, not just flash arbitrary firmware.

### 4. Introduce Model Artifacts Carefully

After the serial protocol works, integrate:

- generated `model.cc`
- generated `model.h`

Only after that should arena sizing and inference timing be layered in.

This keeps the debugging surface small:

- first board bring-up
- then host protocol
- then model linkage
- then inference/runtime behavior

### 5. Add A Non-Arduino Device Backend

The correct architecture for `tinyodom-ex` is not "treat STM32 like another
Arduino board." It is "add a second backend family."

That likely means:

- implement `compile()`, `upload()`, and `measure()` in a real STM32 device
  class
- stop assuming `.ino` as the firmware artifact
- point the STM32 backend at a CubeIDE/CubeCLT project instead

### 6. Avoid Premature External-Flash Boot Work

For TinyODOM experimentation, the best first target is still:

- development boot mode
- debug-load ELF into target memory

Do not make the early backend depend on:

- signing
- external flash programming
- persistent boot images

Those can come later if standalone rebootable deployment becomes necessary.

### 7. Likely First Backend Milestone

A good first real TinyODOM STM32 milestone would be:

- Python generates or copies config/model inputs
- Python runs CLI build
- Python loads the ELF through the debug path
- board prints `READY`
- host triggers one execution path
- firmware returns timing text over serial

That is enough to validate the backend shape without requiring full TFLM
integration on day one.
