from __future__ import annotations

from typing import Any, Optional


def _registry_entries() -> dict[str, tuple[str, str, str]]:
    """Return lazy registry metadata for supported microcontroller wrappers.

    Returns
    -------
    dict[str, tuple[str, str, str]]
        Mapping of ``device_name -> (module_path, class_name, default_spec_symbol)``.
    """
    return {
        "ARDUINO_NANO_33_BLE_SENSE": (
            "tinyodom.microcontrollers.arduino_ble33",
            "ArduinoBLE33Device",
            "BOARD_DEFAULT_SPEC",
        ),
        "PORTENTA_H7": (
            "tinyodom.microcontrollers.arduino_portenta_h7",
            "ArduinoPortentaH7Device",
            "BOARD_DEFAULT_SPEC",
        ),
    }


def _load_symbol(module_path: str, symbol_name: str) -> Any:
    """Import and return a symbol from a module path.

    Parameters
    ----------
    module_path : str
        Fully-qualified module path.
    symbol_name : str
        Symbol defined in ``module_path``.

    Returns
    -------
    Any
        Imported symbol object.
    """
    module = __import__(module_path, fromlist=[symbol_name])
    return getattr(module, symbol_name)


def list_device_specs() -> dict[str, dict[str, Any]]:
    """Return registry-backed default specs for supported board classes.

    Returns
    -------
    dict[str, dict[str, Any]]
        Compatibility-style metadata with keys:
        ``arena_sizes``, ``max_ram``, ``max_flash``, and ``fqbn``.
    """
    specs: dict[str, dict[str, Any]] = {}
    entries = _registry_entries()
    for device_name, (module_path, _class_name, spec_symbol) in entries.items():
        spec = _load_symbol(module_path, spec_symbol)
        specs[device_name] = {
            "arena_sizes": list(spec.arena_sizes_kb),
            "max_ram": int(spec.max_ram_bytes),
            "max_flash": int(spec.max_flash_bytes),
            "fqbn": spec.fqbn,
        }
    return specs


def get_device(
    name: str,
    *,
    device_options: Optional[dict[str, Any]] = None,
    serial_port: Optional[str] = None,
):
    """Construct a board-specific device wrapper when available.

    Parameters
    ----------
    name : str
        Device identifier.
    device_options : dict[str, Any] | None, optional
        Optional board-specific options (for example Portenta core split).
    serial_port : str | None, optional
        Serial port used for upload/measure operations.

    Returns
    -------
    Any
        Device wrapper implementing the ``DeviceInterface`` contract.
    """
    entries = _registry_entries()
    if name in entries:
        module_path, class_name, _spec_symbol = entries[name]
        device_cls = _load_symbol(module_path, class_name)
        return device_cls(
            serial_port=serial_port,
            device_options=device_options,
        )

    # Fallback keeps compatibility with legacy dictionary-only board entries.
    from ..devices import ArduinoDevice

    try:
        return ArduinoDevice(
            name,
            serial_port=serial_port,
            device_options=device_options,
        )
    except ValueError as exc:
        # Legacy catalog entries without FQBN are not Arduino-backed and should
        # instruct callers to register a dedicated backend implementation.
        if "has no Arduino FQBN" in str(exc):
            raise ValueError(
                f"Device '{name}' has no registered backend in tinyodom.microcontrollers "
                "and no Arduino FQBN fallback. Register a DeviceInterface-backed "
                "microcontroller class for this board."
            ) from exc
        raise


def __getattr__(name: str) -> Any:
    """Lazily expose board classes to avoid import cycles.

    Parameters
    ----------
    name : str
        Requested module attribute.

    Returns
    -------
    Any
        Loaded class object for known names.

    Raises
    ------
    AttributeError
        If ``name`` is not a supported lazy export.
    """
    if name == "ArduinoBLE33Device":
        return _load_symbol(
            "tinyodom.microcontrollers.arduino_ble33",
            "ArduinoBLE33Device",
        )
    if name == "ArduinoPortentaH7Device":
        return _load_symbol(
            "tinyodom.microcontrollers.arduino_portenta_h7",
            "ArduinoPortentaH7Device",
        )
    raise AttributeError(name)


__all__ = [
    "ArduinoBLE33Device",
    "ArduinoPortentaH7Device",
    "get_device",
    "list_device_specs",
]
