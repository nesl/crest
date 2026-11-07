# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Evaluation policy flags shared by candidate proposers."""
from __future__ import annotations

from typing import Any
from ..search_space import SearchSpaceDescriptor
from .values import _cfg_get, _json_scalar

def _runtime_context(
    device_config: Any,
    training_config: Any,
    descriptor: SearchSpaceDescriptor,
    *,
    collect_compile_metrics: bool,
) -> dict[str, Any]:
    """Build compact flags governing candidate evaluation.

    Parameters
    ----------
    device_config : Any
        HIL and compile-policy settings.
    training_config : Any
        Training-enabled flag.
    descriptor : SearchSpaceDescriptor
        Active quantization and CPU-clock search parameters.
    collect_compile_metrics : bool
        Resolved runner decision to collect compile metrics.

    Returns
    -------
    dict
        Compact evaluation flags without host paths or runtime payloads.
    """
    context = {
        "hil_enabled": bool(_cfg_get(device_config, "hil", False)),
        "training_enabled": bool(_cfg_get(training_config, "train", False)),
        "compile_metrics_enabled": bool(collect_compile_metrics),
        "quantization_search_enabled": "quantization_mode" in descriptor,
        "cpu_clock_search_enabled": "cpu_clock_mhz_index" in descriptor,
    }
    compile_policy = _json_scalar(_cfg_get(device_config, "compile_when_hil_disabled"))
    if compile_policy is not None:
        context["compile_when_hil_disabled"] = compile_policy
    return context


