<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->

# Microcontroller backends

Microcontroller backends turn one trained/model-candidate export into something CREST can compile, upload, run on a board, and measure.

For the broader source map, see [`../../README.md`](../../README.md).
If you are working on dataset loading, task semantics, or model architecture, use the neighboring component-layer READMEs instead.

## Use a supported board

CREST supports Arduino Nano 33 BLE Sense, Portenta H7 (CM7 and CM4), and STM32 NUCLEO-N657X0-Q.
Choose a matching [example configuration](../../config/README.md#example-configs), prepare the dataset, and set the device and harness serial ports.
Follow [Getting started](../../GETTING_STARTED.md#running-nas-and-hil) to launch the HIL server and training client.

Compile-only checks need the target toolchain but no connected board.
Uploading and runtime measurement need the development board.
Energy measurement also needs the CREST harness and INA228 monitor.
See the [firmware guide](../../../firmware/README.md) for target and harness firmware.

### Arduino tools

Run `make arduino-setup` from the repository root and reactivate the environment as described in [hardware setup](../../GETTING_STARTED.md#hardware-setup).
Install `arduino:mbed_nano` for Nano 33 BLE Sense or `arduino:mbed_portenta` for Portenta H7 using `arduino-cli core install`.
Pass `--config-file tools/arduino-cli.yaml` to use the repository's CLI configuration.
For Portenta, select `device.portenta.target_core` as `cm7` or `cm4`.

### STM32 setup

Install the tools below, then run `make stm32-setup` from the repository root.
Use the [STM32 example](../../config/nas_config_stm32.yaml) for board and storage settings.
See [STM32 configuration](#shipped-config-shape) for available overrides.

### Required host tools

1. `ST-LINK_gdbserver`
2. `arm-none-eabi-gdb`
3. `STM32_Programmer_CLI`
4. `arm-none-eabi-gcc`
5. `arm-none-eabi-size`
6. `arm-none-eabi-objdump`
7. `STM32_SigningTool_CLI` or `STM32TrustedPackageCreator_CLI`
8. ST Edge AI CLI (`stedgeai`) reachable from the host build environment for normal STM32 NAS/HIL candidate generation

`STM32_Programmer_CLI` is required both for external-flash programming and for the current ST-LINK debug-load helper flow, because the GDB server is launched with the STM32CubeProgrammer `bin` directory.

### Shipped config shape

Example config block

```yaml
device:
  name: STM32_NUCLEO_N657X0_Q
  serial_port: /dev/ttyACM0
  measured_inference_runs: 10
  runtime_mode: cadenced
  cpu_clock_mhz_options: [200, 300, 400, 600, 800]
  stm32:
    project_root: firmware/stm32/crest_stm32_lrun
    appli_flash_address: 0x70100000
    weight_storage_mode: external_flash
    weights_memory_pool: src/config/stm32_nucleo_mypool.json
    wake_margin_us: 5000
    min_sleep_us: 5000
```

Current default behavior

1. If `device.stm32.project_root` is omitted, the backend defaults to `firmware/stm32/crest_stm32_lrun`.
2. LRUN `dev_boot` is the only supported layout shipped for this board and is implied automatically.

Supported STM options today

1. Shared runtime knob.
   `device.measured_inference_runs` controls how many on-device inference invokes are averaged into one measured HIL attempt.
   This is not an STM-only board option.
2. Shared cadence/runtime knobs.
   - `device.runtime_mode`.
     `back_to_back` or `cadenced`
   - `device.latency_budget_ms`.
     Optional cadence-budget override
3. Shared CPU clock sweep knob (if allowed by the DUT).
   - `device.cpu_clock_mhz_options`.
     Optional per-trial presets `200`, `300`, `400`, `600`, `800`
4. STM32 workspace/layout options.
   - `template_root`
   - `project_root`
5. STM32 runtime/storage options.
   - `weight_storage_mode`.
     `embedded` or `external_flash`
   - `appli_flash_address`
   - `weights_flash_address`
   - `weights_memory_pool`
   - `weights_external_loader`
   - `wake_margin_us`
   - `min_sleep_us`
   - `max_external_flash_bytes`
6. Optional tool/signing overrides.
   - `gdbserver`
   - `gdb`
   - `cubeprog_bin`
   - `signing_tool`
   - `gdb_port`
   - `apid`
   - `server_ready_timeout_s`
   - `signing_load_offset`
   - `signing_header_version`

## Runtime measurement notes

### Direct serial

This is the default mode.
The DUT is uploaded, opened over serial, and CREST waits for the backend-specific runtime contract to produce latency and optional power data.

### Harness-only

Some targets cannot provide reliable host-visible DUT serial during runtime.
When that happens, `runtime_measure_mode()` may return `harness_only`, which requires `device.harness_serial_port` and the harness flow in the shared controller path.

### Portenta CM4 limitation

For `PORTENTA_H7` with `target_core=cm4`, CREST uses `harness_only` measurement.

CREST uploads `firmware/boot_m4_helper/boot_m4_helper.ino` on CM7 before uploading the CM4 DUT sketch.
`serial_port` is required for both uploads.

The Arduino `mbed_portenta` CM4 path does not expose host USB CDC `Serial` on the primary ACM upload port.
Harness telemetry is therefore authoritative for runtime latency and energy.
DUT-side clock telemetry may be merged when available.
Arena/runtime error lines are not reliably visible, so a CM4 arena allocation failure can surface as `HIL_ERROR_LATENCY` rather than `HIL_ERROR_UNDER_SIZED`.
Deterministic runtime-fault classification requires a separate fault channel.

## Troubleshooting

- If the board cannot be opened or programmed, check its serial port, USB connection, host permissions, and selected board/core.
- If compilation fails, check that the vendor tools and board packages are installed and that the configured workspace exists.
  Read the build log for memory overflow or missing tool errors.
- If runtime measurement times out, check that target and harness firmware match the selected configuration and that harness connections and serial ports are correct.
- For Portenta CM4, use harness measurements rather than expecting USB serial telemetry from the DUT.

For a new backend, also verify registration, resolved options, candidate staging, and runtime parsing using the [software test plan](#software-tests).

## Extend a hardware backend

Backends own board metadata, staging, build and upload tools, telemetry, and normalized metrics.
Keep model architecture and task scoring in their respective components.

### Backend map

| Module | Responsibility |
| --- | --- |
| [`../devices.py`](../devices.py) | `DeviceInterface`, shared records, and the `ArduinoDevice` implementation |
| [`__init__.py`](__init__.py) | Board registry and option resolution |
| [`arduino_base.py`](arduino_base.py) | Arduino CLI compilation, upload, and measurement helpers |
| [`arduino_ble33.py`](arduino_ble33.py), [`arduino_portenta_h7.py`](arduino_portenta_h7.py) | Board-specific Arduino wrappers |
| [`stm32_nucleo_n657x0.py`](stm32_nucleo_n657x0.py) | STM32 candidate staging and concrete backend |
| [`stm32_cube_clt.py`](stm32_cube_clt.py), [`stm32_runtime.py`](stm32_runtime.py) | STM32 build/programming tools and runtime protocol |

Arduino backends export TFLite as `model.cc` and `model.h` and stage the selected [Arduino sketch](../../../firmware/README.md).
STM32 backends instead generate ST Edge AI outputs into a staged STM32 project.
Keep these toolchain differences behind the shared device interface.

### Add a new hardware backend

Choose the path that matches your toolchain and runtime contract.

<a id="use-the-arduino-path-when"></a>
<a id="use-the-stm32-path-when"></a>
<a id="use-the-new-vendor-path-when"></a>

- Arduino FQBN with `arduino-cli compile` and `arduino-cli upload`.
  [Arduino bring-up](#arduino-bring-up-guide).
- STM32 Cube / ST Edge AI / ST-LINK with a generated `Debug/makefile` and the shipped serial protocol.
  [STM32 bring-up](#stm32-bring-up-guide).
- A different build, flash/load, or runtime telemetry flow.
  [new-vendor bring-up](#new-vendor-bring-up-guide).

### Contract in practice

All microcontroller backends must satisfy the `DeviceInterface` contract in [`../devices.py`](../devices.py).

In plain English, a backend owns these responsibilities.

1. `spec` Returns the device limits and arena search space exposed to the rest of the system through `DeviceSpec`.
2. `prepare_candidate(...)` Creates the board-specific build directory or project for one model candidate.
3. `compile(...)` Builds that candidate and returns normalized flash/RAM diagnostics through `CompileResult`.
4. `upload(...)` Flashes or loads the built image onto the device and returns `UploadResult`.
5. `measure(...)` Captures runtime telemetry and returns normalized latency / power / error data through `MeasureResult`.
6. `evaluate(...)` Orchestrates compile, upload, and runtime measurement, then translates backend-specific behavior into CREST `DeviceMetrics`.

Backends also declare behavior through capability flags.

1. `requires_candidate_model()` Whether the backend consumes generated model artifacts.
2. `requires_training_data()` Whether candidate preparation needs calibration/training data.
3. `requires_arena_validation()` Whether arena search should treat runtime arena validation as required.
4. `supports_energy_measurement()` Whether real power/energy telemetry is supported.
5. `supports_runtime_measurement()` Whether upload/runtime HIL passes are supported.
6. `runtime_measure_mode()` Whether runtime is measured via direct DUT serial or via a harness-led flow.

If you are adding a new backend family, start by understanding this contract.

### Registration and config selection

All backends, regardless of family, need to integrate with the shared factory and config plumbing.

#### Registry

Update [`__init__.py`](__init__.py).

Required work

1. Add an entry to `_registry_entries()`.
2. Add lazy export support in `__getattr__` if the backend class should be importable from the package root.
3. Add the class name to `__all__`.

#### Device options

If the board has no custom options, follow these steps.

1. Set `device.name` in config.
2. Set `device.serial_port`.

If the board has custom options, follow these steps.

1. Add a nested config block such as `device.portenta.*` or `device.stm32.*`.
2. Parse it in [`resolve_device_options(...)`](__init__.py).
3. Ensure those options flow through [`build_collect_metrics_request(...)`](../hil/runtime.py) and the HIL server path in [`../../hil_server.py`](../../hil_server.py).

Do not bypass `collect_metrics(...)` or `HIL_controller(...)`.
Device options must flow through the shared controller path.

### Arduino bring-up guide

Use this path only for boards built and flashed through `arduino-cli`.

#### Files you will usually touch

1. `src/crest/microcontrollers/arduino_<board>.py`
2. `src/crest/microcontrollers/__init__.py`
3. `setup_arduino.sh` when a new Arduino core package is required
4. `src/config/` example configs and any board-specific config variants
5. Tests in `test/test_hardware.py` and `test/test_model.py`
6. `src/crest/model.py` only when you are changing shared request plumbing, which should be unusual for a normal Arduino board bring-up

#### Board module contract

Create `src/crest/microcontrollers/arduino_<board>.py` with this shape.

1. Define board identity constants.
   - `BOARD_NAME`
   - `BOARD_FQBN`
2. Define resolver/spec symbols.
   - `resolve_<board>_options(...)`
   - `build_<board>_spec(...)`
   - `BOARD_DEFAULT_SPEC`
3. Define a default `DeviceSpec` with the following settings.
   - explicit `arena_sizes_kb`
   - explicit `max_ram_bytes`
   - explicit `max_flash_bytes`
   - `toolchain="arduino-cli"`
4. If the board has options, define the following symbols.
   - an options dataclass if useful
   - a resolver/validator
   - a spec builder using resolved options
5. Implement a wrapper class that extends [`ArduinoDevice`](../devices.py)

Current in-repo examples

1. BLE33.
   [`arduino_ble33.py`](arduino_ble33.py)
2. Portenta H7.
   [`arduino_portenta_h7.py`](arduino_portenta_h7.py)

Example minimal static board pattern

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

#### Arduino core setup

If the board needs a new Arduino core package, follow these steps.

1. Update `setup_arduino.sh`.
2. Add the correct `arduino-cli core install ...` line.
3. Keep existing core installs intact.

#### Memory parsing notes

[`arduino_base.compile_sketch(...)`](arduino_base.py) parses memory in two stages.

1. Primary.
   Arduino CLI `Sketch uses ...` output.
2. Fallback.
   Board-defined `platform.txt` regexes (`recipe.size.regex` and `recipe.size.regex.data`) over `arm-none-eabi-size -A` output.

Why the fallback exists

1. Some boards may not print summary lines.
2. `platform.txt` is the board package's own source of truth for Arduino-style memory accounting.
3. It avoids counting linker-reserved sections such as `.heap`, `.stack`, or board-reserved regions as "used RAM".

If a board does not print `Sketch uses ...`, follow these steps.

1. Inspect resolved properties with the following command.
   `arduino-cli compile --show-properties --fqbn <fqbn> <sketch>`
2. Confirm these properties exist.
   - `recipe.size.pattern`
   - `recipe.size.regex`
   - `recipe.size.regex.data`
3. Confirm the following board limits resolve.
   - `upload.maximum_size`
   - `upload.maximum_data_size`
4. If needed, inject `upload.maximum_size` via `--build-property`.
5. Re-run compile and verify either of the following results.
   - Arduino CLI prints summary lines, or
   - the fallback returns correct flash/RAM values.

#### Upload error notes

[`arduino_base.upload_sketch(...)`](arduino_base.py) appends Linux DFU permission guidance for known `LIBUSB_ERROR_ACCESS` patterns.

If your board uses DFU on Linux, follow these steps.

1. Document required udev rules in this guide’s user setup instructions.
2. Keep Linux and macOS notes explicit.

### STM32 bring-up guide

This section documents the shipped `STM32_NUCLEO_N657X0_Q` backend and the starting point for adding a second STM32 backend.

#### What the current STM backend assumes

This matters if you want to add a second STM board.

The current backend assumes the following.

1. A template-root project that can be copied into a candidate-specific staging directory.
2. A `Debug` directory with a generated `makefile` build path.
3. ST Edge AI generated outputs with the expected filenames.
4. An ST-LINK based load flow via GDB server + GDB.
5. A serial runtime protocol compatible with [`stm32_runtime.py`](stm32_runtime.py).
6. Optional harness-assisted energy measurement when `harness_serial_port` is configured during HIL runs.

#### Recommended path for a second STM32 board

1. Treat [`stm32_nucleo_n657x0.py`](stm32_nucleo_n657x0.py) as the reference implementation.
2. Reuse [`stm32_cube_clt.py`](stm32_cube_clt.py) if your board still uses the same Cube/ST-LINK workflow.
3. Reuse [`stm32_runtime.py`](stm32_runtime.py) only if your firmware emits the same runtime tokens.
4. Add a new concrete STM backend module when the board-specific differences are too large to express as metadata only.

### New-vendor bring-up guide

For a new toolchain or runtime contract, implement a backend with the following deliverables.

#### Minimum deliverables

1. A new module that implements `DeviceInterface`.
2. A registry entry in [`__init__.py`](__init__.py).
3. An option resolver in [`resolve_device_options(...)`](__init__.py) when the backend needs config-owned options.
4. A candidate preparation path.
5. A compile/build path and size parser.
6. A flash/load path.
7. A runtime telemetry parser and error-classification path.
8. Tests and at least one smoke workflow.

Before implementing a backend, identify the build and programming tools, memory-size output, and runtime success/error protocol.
Use those contracts to define staging, telemetry parsing, and failure tests.

### Test plan

Most backend work should have a hardware-free preflight path first.
Use connected hardware only after config resolution, request construction, staging, and parser behavior are covered locally.

#### Software tests

Add or update tests in the backend-specific and orchestration test surfaces.

Important current test files include the following.

1. `test/test_hardware.py`
2. `test/test_model.py`
3. `test/test_hil_server.py`
4. `test/test_nas_model_client.py`
5. `test/test_odom_tcn.py`
6. `test/test_stm32_backend.py`
7. `test/test_stm32_runtime.py`
8. `test/integration/test_hardware_integration.py`

For a new Arduino board, cover the following cases.

1. Board option validation if applicable.
2. Command construction including expected `--fqbn` and `--board-options`.
3. Memory parsing returning usable values.
4. Config -> `device_options` -> `collect_metrics` -> `HIL_controller` plumbing.
5. Compatibility behavior for `DEVICE_SPECS` helpers.

For a new STM or non-Arduino backend, cover the following cases.

1. Option validation and config resolution.
2. Candidate staging behavior.
3. Build/size parsing behavior.
4. Upload failure classification.
5. Runtime protocol success and error parsing.
6. End-to-end `evaluate(...)` normalization into CREST metrics and error codes.
7. Compile-only smoke (`run_hil=False`) on the new board/backend config when the required toolchain is available.

### Definition of done

Complete the [software test plan](#software-tests) before connected-hardware validation.
Keep board/toolchain code in the backend, architecture/search logic in model families, and task metrics in task adapters.

- A config constructs the backend through `get_device(...)` with the expected serial port defaults and resolved options.
- Candidate staging produces the expected filenames and build directory layout.
- Compile-only smoke with `run_hil=False` returns real flash/RAM diagnostics or correctly typed compile-overflow failures.
- Upload smoke succeeds on connected hardware or returns a clear actionable failure.
- Runtime smoke, when HIL is supported, returns real telemetry or correctly typed runtime failures.
- Energy-aware and harness-only checks confirm the configured harness pins and settings.
- `evaluate(...)` returns normalized `DeviceMetrics` accepted by the shared HIL controller.
- This README and the root README describe required setup and platform caveats, and NumPy-style docstrings match changed functions, classes, dataclasses, and tests.

## Freenove ESP32-S3 backend

Select `device.name: FREENOVE_ESP32_S3_WROOM`. The backend builds the tracked
ESP-IDF project in `firmware/esp32/crest_esp32_tflm`, preserves its incremental
build cache, and programs only the application partition with `app-flash`.
The verified profile is ESP-IDF 5.5.1, ESP-TFLM 1.3.3~1, a 240 MHz CPU, 8 MiB
DIO/40 MHz flash, 8 MiB octal/40 MHz PSRAM, debug optimization with ESP-NN,
and a 4 MiB factory application partition. Unsupported target, clock, runtime,
optimization, or hardware-capacity overrides are rejected.

The backend consumes prepared representative tensors without performing FER
image preprocessing. Configure both `device.esp32.fixture_npz_path` and
`device.esp32.fixture_manifest_path`. The NPZ contains exactly float32
`inputs` with shape `(N,H,W)` and an ordered Unicode `ids` vector. The manifest
binds `npz_sha256`, `inputs_shape`, `inputs_dtype`, and `ids` to those bytes.
Each artifact's actual int8 input scale and zero point are used for
quantization. `device.esp32.warmup_runs` and the shared
`device.measured_inference_runs` are counts per image; the total timed invokes
are `N * measured_inference_runs`. `device.esp32.latency_aggregation` is
`median` by default and may be set to `mean`.

Only `MicroInterpreter::Invoke()` is timed. Input bytes are recopied before
every warmup and measured invoke, and the 20 ms scheduler yield occurs after
timing. Runtime telemetry is accepted only when the model and fixture hashes,
settings, ordered samples, output vector sizes, arena accounting, sample count,
and recomputed aggregate all agree. A completed run with incomplete evidence is
reported as `HIL_ERROR_INTEGRITY`. Build evidence also checks that the generated
C array and built application contain the exact staged TFLite bytes.

The saved harness contract uses GPIO 21 as a DUT output to harness D2 trigger
(input pulldown; rising begins activity and falling ends it) and GPIO 47 as a
DUT output to harness D3 arm (input pullup; active low). This mapping comes from
the repository harness firmware and saved deployment sources; it does not prove
what is currently flashed. The GPIO interval is activity signaling only.
Classifier latency comes exclusively from invoke samples, and this backend does
not currently report energy or cadenced measurements.

Runtime measurement requires explicit serial and network timeouts large enough
for every warmup and measured invoke. The HCI profile uses a 1200 second DUT
serial timeout, 30 second DUT-ready timeout, 2400 second network receive timeout,
900 second build timeout, and 120 second upload timeout. Compile-only operation
requires the IDF toolchain but no connected board.
