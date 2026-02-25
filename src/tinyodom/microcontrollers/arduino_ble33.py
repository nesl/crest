from __future__ import annotations

from typing import Mapping, Optional

from ..devices import ArduinoDevice, DeviceSpec

BOARD_NAME = "ARDUINO_NANO_33_BLE_SENSE"
BOARD_FQBN = "arduino:mbed_nano:nano33ble"


def resolve_ble33_options(
    device_options: Optional[Mapping[str, object]],
) -> None:
    """Resolve BLE33 board options.

    Parameters
    ----------
    device_options : Mapping[str, object] | None
        Optional options passed by callers. BLE33 does not support board
        options in this backend and ignores the value.

    Returns
    -------
    None
        BLE33 has no board options.
    """
    del device_options
    return None


def build_ble33_spec(options: None = None) -> DeviceSpec:
    """Build the default BLE33 ``DeviceSpec``.

    Parameters
    ----------
    options : None, optional
        Unused. Included for contract symmetry with optionized boards.

    Returns
    -------
    DeviceSpec
        Fully-populated static BLE33 device specification.
    """
    del options
    return DeviceSpec(
        name=BOARD_NAME,
        arena_sizes_kb=[10, 25, 40, 70, 95, 120, 140, 160, 180, 200, 210],
        max_ram_bytes=215_000,
        max_flash_bytes=800_000,
        fqbn=BOARD_FQBN,
        toolchain="arduino-cli",
    )


BOARD_DEFAULT_SPEC = build_ble33_spec()


class ArduinoBLE33Device(ArduinoDevice):
    """Concrete Arduino Nano 33 BLE Sense device wrapper.

    Parameters
    ----------
    serial_port : str | None, optional
        Target serial port for upload and measurement.
    device_options : dict[str, str] | None, optional
        Optional board options. Ignored for BLE33 but accepted for interface
        compatibility with optionized boards.
    """

    def __init__(
        self,
        *,
        serial_port: Optional[str] = None,
        device_options: Optional[dict[str, str]] = None,
    ) -> None:
        """Initialize BLE33 wrapper with fixed board specification.

        Parameters
        ----------
        serial_port : str | None, optional
            Target serial port for upload and measurement.
        device_options : dict[str, str] | None, optional
            Optional board options accepted for API compatibility.
        """
        resolved_options = resolve_ble33_options(device_options)
        super().__init__(
            BOARD_NAME,
            serial_port=serial_port,
            device_options=resolved_options,
            spec_override=build_ble33_spec(resolved_options),
        )
