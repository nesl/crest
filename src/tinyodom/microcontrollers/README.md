# Microcontroller Backends

This package contains TinyODOM hardware backends.

A backend is the code that turns one trained/model-candidate export into
something TinyODOM can compile, upload, run on a board, and measure. The
backend boundary is important because Arduino boards, STM32 Cube projects, and
future non-Arduino vendors do not all use the same build and runtime flow.

This README is the bring-up guide for:

1. Adding a new Arduino board.
2. Adding another STM32 board that is similar to the current STM32 flow.
3. Adding a new non-Arduino vendor backend such as NXP or Nordic.

## Start Here

Choose the path that matches your board.

### Use the Arduino path when:

1. The board has an Arduino FQBN.
2. The board builds with `arduino-cli compile`.
3. The board uploads with `arduino-cli upload`.

### Use the STM32 path when:

1. The board uses the STM32 Cube / ST Edge AI / ST-LINK workflow already used
   in this repo.
2. The board can be staged as a project directory, built with the generated
   `Debug/makefile`, loaded with ST-LINK tooling, and measured through the
   current runtime protocol.

### Use the new-vendor path when:

1. The board does not use Arduino CLI.
2. The board does not fit the current STM32 Cube flow.
3. You need a new toolchain, flash/load method, or runtime telemetry contract.

## MCU Backend Contract

All microcontroller backends must satisfy the `DeviceInterface` contract in
[`src/tinyodom/devices.py`](/home/joe/Projects/tinyodom-ex/src/tinyodom/devices.py:177).

In plain English, a backend owns these responsibilities:

1. `spec`
   Returns the device limits and arena search space exposed to the rest of the
   system through `DeviceSpec`.
2. `prepare_candidate(...)`
   Creates the board-specific build directory or project for one model
   candidate.
3. `compile(...)`
   Builds that candidate and returns normalized flash/RAM diagnostics through
   `CompileResult`.
4. `upload(...)`
   Flashes or loads the built image onto the device and returns `UploadResult`.
5. `measure(...)`
   Captures runtime telemetry and returns normalized latency / power / error
   data through `MeasureResult`.
6. `evaluate(...)`
   Orchestrates compile, upload, and runtime measurement, then translates
   backend-specific behavior into TinyODOM `DeviceMetrics`.

Backends also declare behavior through capability flags:

1. `requires_candidate_model()`
   Whether the backend consumes generated model artifacts.
2. `requires_training_data()`
   Whether candidate preparation needs calibration/training data.
3. `requires_arena_validation()`
   Whether arena search should treat runtime arena validation as required.
4. `supports_energy_measurement()`
   Whether real power/energy telemetry is supported.
5. `supports_runtime_measurement()`
   Whether upload/runtime HIL passes are supported.
6. `runtime_measure_mode()`
   Whether runtime is measured via direct DUT serial or via a harness-led flow.

If you are adding a new backend family, start by understanding this contract.

## File Map

These files are the main integration points in this package.

1. [`devices.py`](/home/joe/Projects/tinyodom-ex/src/tinyodom/devices.py:177)
   Defines the shared dataclasses and the `DeviceInterface` contract. Also
   contains the shared Arduino bridge class `ArduinoDevice`.
2. [`__init__.py`](/home/joe/Projects/tinyodom-ex/src/tinyodom/microcontrollers/__init__.py:7)
   Registry, factory, and config-option resolution for known devices.
3. [`arduino_base.py`](/home/joe/Projects/tinyodom-ex/src/tinyodom/microcontrollers/arduino_base.py:557)
   Shared Arduino CLI compile/upload/measurement helpers.
4. `arduino_<board>.py`
   One concrete Arduino-backed board wrapper per board family.
5. [`stm32_nucleo_n657x0.py`](/home/joe/Projects/tinyodom-ex/src/tinyodom/microcontrollers/stm32_nucleo_n657x0.py:1386)
   The concrete STM32 backend currently shipped in this repo.
6. [`stm32_cube_clt.py`](/home/joe/Projects/tinyodom-ex/src/tinyodom/microcontrollers/stm32_cube_clt.py:121)
   STM32 Cube / build / ELF / ST-LINK / CubeProgrammer helper layer.
7. [`stm32_runtime.py`](/home/joe/Projects/tinyodom-ex/src/tinyodom/microcontrollers/stm32_runtime.py:439)
   STM32 runtime serial protocol parser and runtime error classifier.

## Backend Types In This Repo

### Arduino CLI Backends

Arduino boards are the most standardized path in this repo.

Shared behavior comes from
[`ArduinoDevice`](/home/joe/Projects/tinyodom-ex/src/tinyodom/devices.py:644)
and
[`arduino_base.py`](/home/joe/Projects/tinyodom-ex/src/tinyodom/microcontrollers/arduino_base.py:557).

That shared path already handles:

1. TFLite export and `model.cc` / `model.h` generation during candidate prep.
2. Sketch synchronization into the active output directory.
3. Arduino CLI compile command construction.
4. Arduino CLI upload command construction.
5. Flash/RAM parsing from Arduino summary lines.
6. Fallback flash/RAM parsing from the board package `platform.txt` size regexes.
7. Direct-serial runtime measurement and harness-assisted measurement.

Current examples:

1. [`arduino_ble33.py`](/home/joe/Projects/tinyodom-ex/src/tinyodom/microcontrollers/arduino_ble33.py:1)
2. [`arduino_portenta_h7.py`](/home/joe/Projects/tinyodom-ex/src/tinyodom/microcontrollers/arduino_portenta_h7.py:1)

### STM32 Cube Backend

The STM32 implementation is intentionally split across three modules because
they own different responsibilities and failure domains.

1. [`stm32_nucleo_n657x0.py`](/home/joe/Projects/tinyodom-ex/src/tinyodom/microcontrollers/stm32_nucleo_n657x0.py:1386)
   The concrete TinyODOM board backend. It implements `DeviceInterface`,
   resolves STM options, stages candidate-specific projects, invokes ST Edge AI
   generation, and translates compile/upload/runtime results into TinyODOM
   dataclasses and error codes.
2. [`stm32_cube_clt.py`](/home/joe/Projects/tinyodom-ex/src/tinyodom/microcontrollers/stm32_cube_clt.py:121)
   The STM32 toolchain/helper layer. It owns build, ELF discovery, size
   parsing, ST-LINK debug-load workflow, and external flash programming
   helpers. It does not implement `DeviceInterface`.
3. [`stm32_runtime.py`](/home/joe/Projects/tinyodom-ex/src/tinyodom/microcontrollers/stm32_runtime.py:439)
   The STM32 runtime protocol layer. It owns serial monitoring, READY/START
   handshake handling, telemetry parsing, and runtime protocol error typing. It
   does not implement `DeviceInterface`.

Important current limitation:

1. This is a concrete STM32 implementation, not yet a generic `stm_base.py`.
2. It assumes a specific project template shape, ST Edge AI generation flow,
   ST-LINK load path, and serial telemetry contract.
3. A second STM32 board can likely reuse parts of this stack, but should not
   assume every STM32 board is a small metadata-only addition.

### Other-Vendor Backends

For NXP, Nordic, RP2040 SDK, Zephyr, or any other non-Arduino/non-current-STM
target, you should treat the work as a new backend implementation.

That usually means:

1. Implement `DeviceInterface` directly.
2. Add a registry entry in
   [`__init__.py`](/home/joe/Projects/tinyodom-ex/src/tinyodom/microcontrollers/__init__.py:7).
3. Add config-option resolution in
   [`resolve_device_options(...)`](/home/joe/Projects/tinyodom-ex/src/tinyodom/microcontrollers/__init__.py:133)
   when needed.
4. Provide your own build, flash/load, and runtime telemetry path.

## Registry And Config Plumbing

All backends, regardless of family, need to integrate with the shared factory
and config plumbing.

### Registry

Update
[`src/tinyodom/microcontrollers/__init__.py`](/home/joe/Projects/tinyodom-ex/src/tinyodom/microcontrollers/__init__.py:7).

Required work:

1. Add an entry to `_registry_entries()`.
2. Add lazy export support in `__getattr__` if the backend class should be
   importable from the package root.
3. Add the class name to `__all__`.

Why this matters:

1. `get_device(...)` instantiates backend wrappers.
2. `list_device_specs()` projects default specs for compatibility paths.
3. `devices.DEVICE_SPECS` is rebuilt from this registry plus legacy entries.

### Device Options

If the board has no custom options:

1. Set `device.name` in config.
2. Set `device.serial_port`.

If the board has custom options:

1. Add a nested config block such as `device.portenta.*` or `device.stm32.*`.
2. Parse it in
   [`resolve_device_options(...)`](/home/joe/Projects/tinyodom-ex/src/tinyodom/microcontrollers/__init__.py:133).
3. Ensure those options flow through
   [`build_collect_metrics_request(...)`](/home/joe/Projects/tinyodom-ex/src/tinyodom/model.py:488)
   and the HIL server path in
   [`src/hil_server.py`](/home/joe/Projects/tinyodom-ex/src/hil_server.py:279).

Do not bypass `collect_metrics(...)` or `HIL_controller(...)`. Device options
must flow through the shared controller path.

## Runtime Sketch And Candidate Staging

TinyODOM has two major candidate-prep families today.

### Arduino Candidate Staging

Arduino backends use the shared candidate path in
[`ArduinoDevice.prepare_candidate(...)`](/home/joe/Projects/tinyodom-ex/src/tinyodom/devices.py:716).

That path:

1. Converts the Keras model to TFLite.
2. Converts the TFLite file to `model.cc` / `model.h`.
3. Copies the selected sketch variant into the active output directory.

Runtime sketch selection is handled by
[`_sync_arduino_sketch_variant_for_config(...)`](/home/joe/Projects/tinyodom-ex/src/tinyodom/devices.py:584)
and uses this layout:

1. Shared uniform variants:
   - `sketches/tinyodom_tcn_energy.ino`
   - `sketches/tinyodom_tcn_no_energy.ino`
2. Shared analysis variants:
   - `sketches/analysis_sketches/tinyodom_tcn_energy_representative.ino`
   - `sketches/analysis_sketches/tinyodom_tcn_energy_real_data.ino`
   - `sketches/analysis_sketches/tinyodom_tcn_input_data.h`
3. Shared headers copied from `sketches/common/`:
   - `tinyodom_hil_config.h`
   - `tinyodom_power.h`
   - `tinyodom_clock_telemetry.h`

Board/core behavior should be selected in Python wrappers and compile-time
defines, not by duplicating the shared uniform sketch sources.

### STM32 Candidate Staging

STM32 does not use Arduino sketch staging.

The current backend in
[`stm32_nucleo_n657x0.py`](/home/joe/Projects/tinyodom-ex/src/tinyodom/microcontrollers/stm32_nucleo_n657x0.py:1386)
instead:

1. Copies a canonical FSBL template into a candidate-specific staging root.
2. Exports the Keras model to TFLite.
3. Runs ST Edge AI analyze/generate steps.
4. Stages generated outputs into the project.
5. Writes manifest/config artifacts needed for later compile/upload phases.

Important boundary:

1. STM32 does not consume Arduino `model.cc` / `model.h` artifacts.
2. STM32 does not use the Arduino sketch variant sync path.
3. Arduino and STM32 candidate prep already follow separate backend-owned paths
   under the shared HIL server flow.

## Arduino Bring-Up Guide

Use this path only for boards built and flashed through `arduino-cli`.

### Files You Will Usually Touch

1. `src/tinyodom/microcontrollers/arduino_<board>.py`
2. `src/tinyodom/microcontrollers/__init__.py`
3. `setup_arduino.sh` when a new Arduino core package is required
4. `src/nas_config.yaml` and/or board-specific configs
5. Tests in `test/test_hardware.py` and `test/test_model.py`
6. `src/tinyodom/model.py` only when you are changing shared request plumbing,
   which should be unusual for a normal Arduino board bring-up

### Board Module Contract

Create `src/tinyodom/microcontrollers/arduino_<board>.py` with this shape:

1. Define board identity constants:
   - `BOARD_NAME`
   - `BOARD_FQBN`
2. Define resolver/spec symbols:
   - `resolve_<board>_options(...)`
   - `build_<board>_spec(...)`
   - `BOARD_DEFAULT_SPEC`
3. Define a default `DeviceSpec` with:
   - explicit `arena_sizes_kb`
   - explicit `max_ram_bytes`
   - explicit `max_flash_bytes`
   - `toolchain="arduino-cli"`
4. If the board has options, define:
   - an options dataclass if useful
   - a resolver/validator
   - a spec builder using resolved options
5. Implement a wrapper class that extends
   [`ArduinoDevice`](/home/joe/Projects/tinyodom-ex/src/tinyodom/devices.py:644)

Current in-repo examples:

1. BLE33:
   [`arduino_ble33.py`](/home/joe/Projects/tinyodom-ex/src/tinyodom/microcontrollers/arduino_ble33.py:1)
2. Portenta H7:
   [`arduino_portenta_h7.py`](/home/joe/Projects/tinyodom-ex/src/tinyodom/microcontrollers/arduino_portenta_h7.py:1)

Example minimal static board pattern:

```python
from __future__ import annotations

from typing import Mapping, Optional

from ..devices import ArduinoDevice, DeviceSpec

BOARD_NAME = "ARDUINO_GIGA_R1"
BOARD_FQBN = "arduino:mbed_giga:giga"


def resolve_giga_r1_options(
    device_options: Optional[Mapping[str, object]],
) -> None:
    del device_options
    return None


def build_giga_r1_spec(options: None = None) -> DeviceSpec:
    del options
    return DeviceSpec(
        name=BOARD_NAME,
        arena_sizes_kb=[32, 64, 96, 128],
        max_ram_bytes=512_000,
        max_flash_bytes=2_000_000,
        fqbn=BOARD_FQBN,
        toolchain="arduino-cli",
    )


BOARD_DEFAULT_SPEC = build_giga_r1_spec()


class ArduinoGigaR1Device(ArduinoDevice):
    def __init__(
        self,
        *,
        serial_port: Optional[str] = None,
        device_options: Optional[dict[str, str]] = None,
    ) -> None:
        resolved_options = resolve_giga_r1_options(device_options)
        super().__init__(
            BOARD_NAME,
            serial_port=serial_port,
            device_options=resolved_options,
            spec_override=build_giga_r1_spec(resolved_options),
        )
```

### Arduino Core Setup

If the board needs a new Arduino core package:

1. Update `setup_arduino.sh`.
2. Add the correct `arduino-cli core install ...` line.
3. Keep existing core installs intact.

### Memory Parsing Notes

[`arduino_base.compile_sketch(...)`](/home/joe/Projects/tinyodom-ex/src/tinyodom/microcontrollers/arduino_base.py:557)
parses memory in two stages:

1. Primary: Arduino CLI `Sketch uses ...` output.
2. Fallback: board-defined `platform.txt` regexes
   (`recipe.size.regex` and `recipe.size.regex.data`) over
   `arm-none-eabi-size -A` output.

Why the fallback exists:

1. Some boards may not print summary lines.
2. `platform.txt` is the board package's own source of truth for Arduino-style
   memory accounting.
3. It avoids counting linker-reserved sections such as `.heap`, `.stack`, or
   board-reserved regions as "used RAM".

If a board does not print `Sketch uses ...`:

1. Inspect resolved properties with:
   `arduino-cli compile --show-properties --fqbn <fqbn> <sketch>`
2. Confirm these properties exist:
   - `recipe.size.pattern`
   - `recipe.size.regex`
   - `recipe.size.regex.data`
3. Confirm board limits resolve:
   - `upload.maximum_size`
   - `upload.maximum_data_size`
4. If needed, inject `upload.maximum_size` via `--build-property`.
5. Re-run compile and verify either:
   - Arduino CLI prints summary lines, or
   - the fallback returns correct flash/RAM values.

### Upload Error Notes

[`arduino_base.upload_sketch(...)`](/home/joe/Projects/tinyodom-ex/src/tinyodom/microcontrollers/arduino_base.py:678)
appends Linux DFU permission guidance for known `LIBUSB_ERROR_ACCESS` patterns.

If your board uses DFU on Linux:

1. Add or update root `README.md` udev instructions.
2. Keep Linux and macOS notes explicit.

## STM32 Bring-Up Guide

Use this section as the canonical reference for the current production
`STM32_NUCLEO_N657X0_Q` backend and as the orientation guide for adding a
second STM32 backend.

### Current Production Config Shape

Example config block:

```yaml
device:
  name: STM32_NUCLEO_N657X0_Q
  serial_port: /dev/ttyACM0
  stm32:
    template_root: sketches/stm32/tinyodom_tcn_stm32/FSBL
    cpu_clock_mhz: 600
    weight_storage_mode: external_flash
    weights_flash_address: 0x71000000
    weights_memory_pool: analysis_scripts/stm32_example_project/nucleo_mypool.json
```

Supported STM options today:

1. `template_root` or legacy `project_root` for the canonical FSBL template.
2. `cpu_clock_mhz` fixed presets: `200`, `300`, `400`, `600`, `800`.
3. `weight_storage_mode`: `embedded` or `external_flash`.
4. `weights_flash_address`, `weights_memory_pool`, and optional
   `weights_external_loader` for external flash mode.
5. Optional tool overrides:
   - `gdbserver`
   - `gdb`
   - `cubeprog_bin`
   - `gdb_port`
   - `apid`
   - `server_ready_timeout_s`
   - `max_external_flash_bytes`

### Required Host Tools

1. `ST-LINK_gdbserver`
2. `arm-none-eabi-gdb`
3. `STM32_Programmer_CLI`
4. ST Edge AI CLI/install reachable from the host build environment

`STM32_Programmer_CLI` is required both for external-flash programming and for
the current ST-LINK debug-load helper flow, because the GDB server is launched
with the STM32CubeProgrammer `bin` directory.

### What The Current STM Backend Assumes

This matters if you want to add a second STM board.

The current backend assumes:

1. A template-root project that can be copied into a candidate-specific staging
   directory.
2. A `Debug` directory with a generated `makefile` build path.
3. ST Edge AI generated outputs with the expected filenames.
4. An ST-LINK based load flow via GDB server + GDB.
5. A serial runtime protocol compatible with
   [`stm32_runtime.py`](/home/joe/Projects/tinyodom-ex/src/tinyodom/microcontrollers/stm32_runtime.py:439).
6. Optional harness-assisted energy measurement when `harness_serial_port` is
   configured during HIL runs.

If your new STM board does not meet those assumptions, you are doing more than
board bring-up. You are defining a new STM backend shape.

### Recommended Path For A Second STM32 Board

Today, the practical path is:

1. Treat
   [`stm32_nucleo_n657x0.py`](/home/joe/Projects/tinyodom-ex/src/tinyodom/microcontrollers/stm32_nucleo_n657x0.py:1386)
   as the reference implementation.
2. Reuse
   [`stm32_cube_clt.py`](/home/joe/Projects/tinyodom-ex/src/tinyodom/microcontrollers/stm32_cube_clt.py:121)
   if your board still uses the same Cube/ST-LINK workflow.
3. Reuse
   [`stm32_runtime.py`](/home/joe/Projects/tinyodom-ex/src/tinyodom/microcontrollers/stm32_runtime.py:439)
   only if your firmware emits the same runtime tokens.
4. Add a new concrete STM backend module when the board-specific differences are
   too large to express as metadata only.

There is no generic `stm_base.py` yet.

## New-Vendor Bring-Up Guide

Use this path for NXP, Nordic, Zephyr-based targets, or any board that does not
fit the Arduino path or the current STM32 path.

### Mindset

You are building a new backend, not adding a board definition.

### Minimum Deliverables

1. A new module that implements `DeviceInterface`.
2. A registry entry in
   [`src/tinyodom/microcontrollers/__init__.py`](/home/joe/Projects/tinyodom-ex/src/tinyodom/microcontrollers/__init__.py:7).
3. An option resolver in
   [`resolve_device_options(...)`](/home/joe/Projects/tinyodom-ex/src/tinyodom/microcontrollers/__init__.py:133)
   when the backend needs config-owned options.
4. A candidate preparation path.
5. A compile/build path and size parser.
6. A flash/load path.
7. A runtime telemetry parser and error-classification path.
8. Tests and at least one smoke workflow.

### Questions To Answer Before You Start

1. What tool builds the firmware?
2. What tool flashes or loads the device?
3. How will flash and RAM usage be parsed?
4. What serial or harness protocol defines a successful runtime run?
5. How are compile overflow, upload failure, runtime timeout, and runtime
   protocol errors classified?

If those answers are not yet clear, the backend boundary is not ready.

## Runtime Measurement Notes

### Direct Serial

This is the default mode. The DUT is uploaded, opened over serial, and TinyODOM
waits for the backend-specific runtime contract to produce latency and optional
power data.

### Harness-Only

Some targets cannot provide reliable host-visible DUT serial during runtime.
When that happens, `runtime_measure_mode()` may return `harness_only`, which
requires `device.harness_serial_port` and the harness flow in the shared
controller path.

### Portenta CM4 Limitation

For `PORTENTA_H7` with `target_core=cm4`, TinyODOM uses `harness_only`
measurement.

Important notes:

1. Harness metrics remain the source of truth for energy and harness latency.
2. DUT-side clock telemetry may still be merged in when available.
3. TinyODOM uploads a CM7 boot-helper sketch before the CM4 DUT upload:
   `sketches/boot_m4_helper/boot_m4_helper.ino`
4. `serial_port` is still required because it is the upload path for both the
   helper and the CM4 DUT sketch.

Current limitation:

1. The Arduino `mbed_portenta` CM4 path does not expose host USB CDC `Serial`
   on the primary ACM port used by TinyODOM uploads.
2. Harness telemetry is therefore authoritative for runtime latency and energy.
3. DUT-side arena/runtime error lines are not reliably visible from the host.

Practical implication:

1. A CM4 runtime arena allocation failure can surface as a generic
   `HIL_ERROR_LATENCY` rather than `HIL_ERROR_UNDER_SIZED`.
2. Deterministic CM4 runtime-fault classification would require a separate
   fault channel and is out of scope for the current backend.

## Required Tests

Add or update tests in `test/test_hardware.py` and `test/test_model.py`.

For a new Arduino board, cover:

1. Board option validation if applicable.
2. Command construction including expected `--fqbn` and `--board-options`.
3. Memory parsing returning usable values.
4. Config -> `device_options` -> `collect_metrics` -> `HIL_controller`
   plumbing.
5. Compatibility behavior for `DEVICE_SPECS` helpers.

For a new STM or non-Arduino backend, cover:

1. Option validation and config resolution.
2. Candidate staging behavior.
3. Build/size parsing behavior.
4. Upload failure classification.
5. Runtime protocol success and error parsing.
6. End-to-end `evaluate(...)` normalization into TinyODOM metrics and error
   codes.

## Quick Smoke Checklist

Before opening a PR:

1. Compile-only smoke (`run_hil=False`) on the new board/backend config.
2. Upload smoke on connected hardware.
3. Runtime smoke on connected hardware when HIL is supported.
4. Confirm serial port defaults in config are correct.
5. Confirm harness pins/settings when energy-aware mode or harness-only mode is
   enabled.
6. Confirm generated output filenames and staging layout are what the backend
   expects.

## Definition Of Done

A backend bring-up is done when:

1. The device can be constructed through `get_device(...)`.
2. `prepare_candidate(...)` produces the correct build input directory.
3. `compile(...)` returns real flash/RAM diagnostics or correctly typed
   compile-overflow failures.
4. `upload(...)` works on connected hardware or returns a clear actionable
   failure.
5. `measure(...)` returns real telemetry or correctly typed runtime failures.
6. `evaluate(...)` returns normalized `DeviceMetrics` that the shared HIL
   controller can consume.
7. Tests cover the backend-specific control path.
8. This README and the root README document any required setup or platform
   caveats.

## Troubleshooting

If bring-up stalls, check these layers in order:

1. Registry:
   Is the backend reachable through `get_device(...)`?
2. Config:
   Are `device.name`, `device.serial_port`, and any backend options resolving
   correctly?
3. Candidate prep:
   Is the backend producing the expected build directory or project layout?
4. Compile:
   Are flash/RAM diagnostics being parsed, not silently returned as sentinels?
5. Upload:
   Is the correct tool and transport being used for the target?
6. Runtime:
   Does the firmware emit the tokens the backend parser expects?

If you are unsure whether your target is "just another board" or "a new
backend", the answer is usually:

1. Arduino FQBN + Arduino CLI compile/upload means board bring-up.
2. Same STM32 Cube/ST Edge AI/ST-LINK/runtime contract may be close to board
   bring-up.
3. New toolchain or new runtime contract means new backend.
