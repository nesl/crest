# Adding A New Arduino Board

This guide is only for adding boards that are built and flashed through `arduino-cli`.

For non-Arduino targets (for example STM32 Cube-based flows), do not follow this file as-is. Those boards should implement `DeviceInterface` with a separate backend path.

## Scope

Use this guide when the board:

1. Has an Arduino FQBN.
2. Can compile through `arduino-cli compile`.
3. Can upload through `arduino-cli upload`.

## Files You Will Touch

1. `src/tinyodom/microcontrollers/arduino_<board>.py`
2. `src/tinyodom/microcontrollers/__init__.py`
3. `setup_arduino.sh` (if a new Arduino core package is needed)
4. `src/tinyodom/model.py` (only if board-specific config options are required)
5. `src/nas_config.yaml` and/or board-specific config files
6. Tests in `test/test_hardware.py` and `test/test_model.py`

## Board Module Contract

Create `src/tinyodom/microcontrollers/arduino_<board>.py` with this shape.

1. Define board identity constants:
   - `BOARD_NAME`
   - `BOARD_FQBN`
2. Define resolver/spec symbols:
   - `resolve_<board>_options(...)`
   - `build_<board>_spec(...)`
   - `BOARD_DEFAULT_SPEC`
3. Define a default `DeviceSpec`:
   - include explicit `arena_sizes_kb`
   - include explicit `max_ram_bytes`
   - include explicit `max_flash_bytes`
   - set `toolchain="arduino-cli"`
4. If board has options, define:
   - options dataclass
   - resolver/validator
   - limits/spec builder using resolved options
5. Implement a device wrapper class that extends `ArduinoDevice`.

Current in-repo examples:

1. BLE33: `BOARD_NAME`, `BOARD_FQBN`, `BOARD_DEFAULT_SPEC`, `resolve_ble33_options`, `build_ble33_spec`, `ArduinoBLE33Device`.
2. Portenta H7: `BOARD_NAME`, `BOARD_FQBN`, `BOARD_DEFAULT_SPEC`, `PortentaH7BoardOptions`, `resolve_portenta_h7_options`, `build_portenta_h7_spec`, `ArduinoPortentaH7Device`.

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

## Registry Integration

Update `src/tinyodom/microcontrollers/__init__.py`.

1. Add an entry in `_registry_entries()`:
   - key = device name
   - tuple = `(module_path, class_name, default_spec_symbol)`
2. Add lazy export support in `__getattr__`.
3. Add class name to `__all__`.

Why this matters:

1. `get_device(...)` uses this registry to instantiate board wrappers.
2. `list_device_specs()` projects default specs for compatibility paths.
3. `devices.DEVICE_SPECS` is rebuilt from this registry plus legacy entries.

## Config Integration

If the board has no custom options:

1. Set `device.name` in config.
2. Set `device.serial_port`.

If the board has custom options:

1. Add a nested config block (for example `device.giga.*`).
2. Parse that block in `build_collect_metrics_request(...)` in `src/tinyodom/model.py`.
3. Populate `CollectMetricsRequest.device_options`.
4. Do not bypass `collect_metrics(...)` or `HIL_controller(...)`; options must flow through those paths.

## Arduino Core Setup

If the board needs a new Arduino core package:

1. Update `setup_arduino.sh`.
2. Add the correct `arduino-cli core install ...` line.
3. Keep existing core installs intact.

## Memory Parsing Notes

`arduino_base.compile_sketch(...)` currently parses memory in two ways:

1. Primary: Arduino CLI `Sketch uses ...` output.
2. Fallback: ELF section accounting via `arm-none-eabi-size -A`.

For new boards, validate at least one compile run and confirm RAM/flash values are non-sentinel.

## Upload Error Notes

`arduino_base.upload_sketch(...)` appends Linux DFU permission guidance for known `LIBUSB_ERROR_ACCESS` patterns.

If your board uses DFU on Linux:

1. Add or update `README.md` udev instructions as needed.
2. Keep macOS/Linux notes explicit.

## Required Tests For New Arduino Boards

Add tests in `test/test_hardware.py` and `test/test_model.py` for:

1. Board option validation (if applicable).
2. Command construction includes expected `--fqbn` and `--board-options`.
3. Memory parse path still returns usable values.
4. Config -> `device_options` -> `collect_metrics` -> `HIL_controller` plumbing.
5. Compatibility behavior for `DEVICE_SPECS` helpers.

## Quick Smoke Checklist

Before opening a PR:

1. Compile-only smoke (`run_hil=False`) on the new board config.
2. Upload smoke on connected hardware.
3. Confirm serial port defaults in config are correct.
4. Confirm harness pins if energy-aware mode is enabled.
5. Confirm generated output filenames reflect the board name in config.
