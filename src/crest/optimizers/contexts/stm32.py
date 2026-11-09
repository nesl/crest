# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""STM32 board and firmware measurement facts, resolved without a controller."""
from __future__ import annotations

from typing import Any
from .values import _cfg_get

def _stm32_context(device_config: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Describe resolved STM32 options without initializing a controller.

    Parameters
    ----------
    device_config : Any
        Mapping-style or attribute-style device configuration.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        Whitelisted device and runtime facts without host paths or tool options.
        Unknown board profiles return empty mappings.

    Notes
    -----
    Option resolution and specification construction are read-only. The legacy
    C API and linker describe the checked-in template, not silicon capability.
    """
    if str(_cfg_get(device_config, "name", "")).strip().upper() != "STM32_NUCLEO_N657X0_Q":
        return {}, {}
    from crest.microcontrollers import resolve_device_options
    from crest.microcontrollers.stm32_nucleo_n657x0 import (
        DEFAULT_MAX_EXTERNAL_FLASH_BYTES,
        DEFAULT_TEMPLATE_ROOT,
        DEFAULT_WEIGHTS_FLASH_ADDRESS,
        build_stm32_nucleo_n657x0_q_spec,
        resolve_stm32_nucleo_n657x0_q_options,
    )

    # Follow the execution path's shared runtime overrides and backend defaults.
    options = resolve_stm32_nucleo_n657x0_q_options(
        resolve_device_options("STM32_NUCLEO_N657X0_Q", device_config)
    )
    spec = build_stm32_nucleo_n657x0_q_spec(options)
    checked_in_template = options.project_root == DEFAULT_TEMPLATE_ROOT.resolve()
    device = {
        "cpu_architecture": "Arm Cortex-M55",
        "configured_cpu_clock_mhz": options.cpu_clock_mhz,
        "weight_storage_mode": options.weight_storage_mode,
        "memory_semantics": {
            "registered_ram_bytes": spec.max_ram_bytes,
            "registered_ram_meaning": "Capacity metadata, not free arena; default LRUN app links code/data/heap/stack into 2047 KiB RAM.",
            "code_image_region_bytes": spec.max_flash_bytes,
            "code_image_region_meaning": "16 MiB Boot/application region within board external flash; not internal flash or additional capacity.",
            "board_external_flash_capacity_bytes": DEFAULT_MAX_EXTERNAL_FLASH_BYTES,
            "configured_external_blob_limit_bytes": spec.max_external_flash_bytes,
            "usable_weights_budget_bytes": None,
            "weights_budget_meaning": "Blob-size guard is not a proven usable weights budget; placement, pool, reserved images, and generated artifacts still constrain fit.",
        },
        "execution_backend": (
            "ST Edge AI --target stm32n6 --c-api legacy; ai_network_run; Cortex-M55 runtime library"
            if checked_in_template else "Custom template: execution path unverified"
        ),
        "acceleration_semantics": "NPU use unverified; stm32n6 target and silicon accelerator capability do not establish its use.",
        "profile_sources": [
            "src/crest/microcontrollers/stm32_nucleo_n657x0.py",
            "firmware/stm32/crest_stm32_lrun/STM32CubeIDE/AppS/Debug/stedgeai.mk",
            "firmware/stm32/crest_stm32_lrun/STM32CubeIDE/AppS/STM32N657XX_LRUN.ld",
        ],
    }
    if options.weight_storage_mode == "external_flash":
        device["memory_semantics"]["weights_flash_address"] = options.weights_flash_address
        if options.weights_flash_address == DEFAULT_WEIGHTS_FLASH_ADDRESS:
            # Physical tail arithmetic only, not an effective deployment budget.
            device["memory_semantics"]["default_address_physical_tail_bytes"] = 48 * 1024 * 1024
    runtime = {
        "runtime_mode": options.runtime_mode,
        "latency_budget_ms": options.latency_budget_ms,
        "wake_margin_us": options.wake_margin_us,
        "min_sleep_us": options.min_sleep_us,
        "measurement_semantics": (
            "Active invocation latency differs from the release period. Cadenced mode uses latency_budget_ms "
            "as release period and active-latency limit. Trial-level energy_mj_per_inference remains "
            "back-to-back energy per inference, including when runtime_mode is cadenced; this is the energy "
            "metric used by the CS3 score. Cadenced energy is separate: cadenced_energy_mj_per_window "
            "is energy per release including wait/Stop/wake overhead, and cadenced_energy_mj_per_trial "
            "is aggregate energy across measured releases. Back-to-back mode repeats invocations without cadence sleep."
        ),
        "measurement_scope": (
            "Default firmware invokes the network on zero-filled input; sensor capture, feature extraction, "
            "application logging, and communications workloads are excluded. Host task accuracy is evaluated separately."
            if checked_in_template else "Custom firmware measurement scope unverified"
        ),
        "measurement_source": "src/crest/microcontrollers/stm32_measurement.py; firmware/stm32/crest_stm32_lrun/Appli/Src/crest_dut_runner.c",
    }
    return device, runtime


