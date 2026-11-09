# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Registered device capacities and active deployment choices."""
from __future__ import annotations

from typing import Any
from crest.devices import get_device_spec
from ..search_space import SearchSpaceDescriptor
from .values import _cfg_get, _json_scalar

def _device_capacities(device_name: str, device_config: Any) -> dict[str, int]:
    """Read registered capacities without constructing a device backend.

    Parameters
    ----------
    device_name : str
        Registered device name.
    device_config : Any
        Device configuration, including Portenta core and split options.

    Returns
    -------
    dict of str to int
        Registered RAM and flash capacities; empty if resolution is unavailable.
    """
    try:
        if device_name == "PORTENTA_H7":
            from crest.microcontrollers.arduino_portenta_h7 import (
                build_portenta_h7_spec,
                resolve_portenta_h7_options,
            )

            portenta = _cfg_get(device_config, "portenta", {})
            options = resolve_portenta_h7_options(
                {
                    key: _cfg_get(portenta, key)
                    for key in ("target_core", "split", "security")
                    if _cfg_get(portenta, key) not in (None, "")
                }
            )
            spec = build_portenta_h7_spec(options=options)
        else:
            spec = get_device_spec(device_name)
    except (ImportError, TypeError, ValueError):
        return {}
    return {
        "ram_capacity_bytes": int(spec.max_ram_bytes),
        "flash_capacity_bytes": int(spec.max_flash_bytes),
    }


def _device_context(
    device_config: Any,
    training_config: Any,
    descriptor: SearchSpaceDescriptor,
) -> dict[str, Any]:
    """Build device resource and deployment-choice semantics.

    Parameters
    ----------
    device_config : Any
        Device identity and resource options.
    training_config : Any
        Configured quantization mode.
    descriptor : SearchSpaceDescriptor
        Active quantization and clock search declarations.

    Returns
    -------
    dict
        Device capacities, configured clocks, and deployment choices.
    """
    device_name = str(_cfg_get(device_config, "name", "")).strip().upper()
    context: dict[str, Any] = {"name": device_name}
    context.update(_device_capacities(device_name, device_config))

    clocks = _cfg_get(device_config, "cpu_clock_mhz_options")
    if isinstance(clocks, (list, tuple)) and clocks:
        normalized_clocks = [_json_scalar(clock) for clock in clocks]
        if all(clock is not None for clock in normalized_clocks):
            context["cpu_clock_mhz_choices"] = normalized_clocks

    quantization = _cfg_get(training_config, "quantization", {})
    configured_mode = _json_scalar(_cfg_get(quantization, "mode"))
    if configured_mode not in (None, ""):
        context["configured_quantization_mode"] = configured_mode
    if "quantization_mode" in descriptor:
        context["quantization_choices"] = list(
            descriptor["quantization_mode"].choices or ()
        )

    stm32 = _cfg_get(device_config, "stm32", {})
    weight_storage_mode = _json_scalar(_cfg_get(stm32, "weight_storage_mode"))
    if weight_storage_mode not in (None, ""):
        context["weight_storage_mode"] = weight_storage_mode
    return context


