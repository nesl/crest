from __future__ import annotations

from dataclasses import asdict
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
        "STM32_NUCLEO_N657X0_Q": (
            "tinyodom.microcontrollers.stm32_nucleo_n657x0",
            "STM32NucleoN657X0QDevice",
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
    normalized_name = name.strip().upper()
    entries = _registry_entries()
    if normalized_name in entries:
        module_path, class_name, _spec_symbol = entries[normalized_name]
        device_cls = _load_symbol(module_path, class_name)
        return device_cls(
            serial_port=serial_port,
            device_options=device_options,
        )

    # Fallback keeps compatibility with legacy dictionary-only board entries.
    from ..devices import ArduinoDevice

    try:
        return ArduinoDevice(
            normalized_name,
            serial_port=serial_port,
            device_options=device_options,
        )
    except ValueError as exc:
        # Legacy catalog entries without FQBN are not Arduino-backed and should
        # instruct callers to register a dedicated backend implementation.
        if "has no Arduino FQBN" in str(exc):
            raise ValueError(
                f"Device '{normalized_name}' has no registered backend in tinyodom.microcontrollers "
                "and no Arduino FQBN fallback. Register a DeviceInterface-backed "
                "microcontroller class for this board."
            ) from exc
        raise


def resolve_device_options(
    device_name: str,
    device_config: Any,
) -> dict[str, Any] | None:
    """Resolve backend-owned device options from a config subtree.

    Parameters
    ----------
    device_name : str
        Target device identifier.
    device_config : Any
        Device config subtree or namespace-like object.

    Returns
    -------
    dict[str, Any] | None
        Normalized backend options, or ``None`` when the backend does not use
        explicit option fields.
    """

    def _cfg_get(container: Any, key: str, default: Any = None) -> Any:
        getter = getattr(container, "get", None)
        if callable(getter):
            return getter(key, default)
        return getattr(container, key, default)

    normalized_name = device_name.strip().upper()
    if normalized_name == "PORTENTA_H7":
        from .arduino_portenta_h7 import resolve_portenta_h7_options

        portenta_cfg = _cfg_get(device_config, "portenta", None)
        resolved = resolve_portenta_h7_options(
            {
                "target_core": _cfg_get(portenta_cfg, "target_core", None) if portenta_cfg is not None else None,
                "split": _cfg_get(portenta_cfg, "split", None) if portenta_cfg is not None else None,
                "security": _cfg_get(portenta_cfg, "security", None) if portenta_cfg is not None else None,
            }
        )
        return resolved.to_board_options()

    if normalized_name == "STM32_NUCLEO_N657X0_Q":
        from .stm32_nucleo_n657x0 import resolve_stm32_nucleo_n657x0_q_options

        stm32_cfg = _cfg_get(device_config, "stm32", None)
        raw_options = {
            "template_root": _cfg_get(stm32_cfg, "template_root", None) if stm32_cfg is not None else None,
            "project_root": _cfg_get(stm32_cfg, "project_root", None) if stm32_cfg is not None else None,
            "gdbserver": _cfg_get(stm32_cfg, "gdbserver", None) if stm32_cfg is not None else None,
            "gdb": _cfg_get(stm32_cfg, "gdb", None) if stm32_cfg is not None else None,
            "cubeprog_bin": _cfg_get(stm32_cfg, "cubeprog_bin", None) if stm32_cfg is not None else None,
            "gdb_port": _cfg_get(stm32_cfg, "gdb_port", None) if stm32_cfg is not None else None,
            "apid": _cfg_get(stm32_cfg, "apid", None) if stm32_cfg is not None else None,
            "server_ready_timeout_s": _cfg_get(stm32_cfg, "server_ready_timeout_s", None)
            if stm32_cfg is not None
            else None,
            "cpu_clock_mhz": _cfg_get(stm32_cfg, "cpu_clock_mhz", None) if stm32_cfg is not None else None,
        }
        resolved = resolve_stm32_nucleo_n657x0_q_options(
            {
                key: value
                for key, value in raw_options.items()
                if value not in (None, "")
            }
        )
        return asdict(resolved)

    return None


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
    if name == "STM32NucleoN657X0QDevice":
        return _load_symbol(
            "tinyodom.microcontrollers.stm32_nucleo_n657x0",
            "STM32NucleoN657X0QDevice",
        )
    raise AttributeError(name)


__all__ = [
    "ArduinoBLE33Device",
    "ArduinoPortentaH7Device",
    "STM32NucleoN657X0QDevice",
    "get_device",
    "list_device_specs",
    "resolve_device_options",
]
