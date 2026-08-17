# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Bounded compact history extracted directly from Optuna trials."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


MEASUREMENT_FIELDS = (
    "latency_ms",
    "energy_mj_per_inference",
    "ram_bytes",
    "flash_bytes",
    "external_flash_bytes",
    "arena_bytes",
    "cpu_clock_mhz_requested",
    "clock_hz",
)


def _json_safe(value: Any) -> Any:
    """Convert common Optuna/NumPy/path values into JSON-safe data."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _available_measurement(value: Any) -> bool:
    """Return whether a metric is more informative than CREST's sentinels."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value)) and float(value) >= 0.0


def build_recent_trial_history(study: Any, *, window_size: int) -> tuple[dict[str, Any], ...]:
    """Build compact records for only the newest configured Optuna trials."""
    if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 0:
        raise ValueError("window_size must be a non-negative integer.")
    if window_size == 0:
        return ()

    trials = list(study.trials)[-window_size:]
    directions = [
        str(getattr(direction, "name", direction)).strip().lower()
        for direction in getattr(study, "directions", ())
    ]
    records: list[dict[str, Any]] = []
    for trial in trials:
        state = getattr(trial, "state", None)
        state_name = str(getattr(state, "name", state)).strip().lower()
        attrs = dict(getattr(trial, "user_attrs", {}) or {})
        values = getattr(trial, "values", None)
        if values is None:
            single_value = getattr(trial, "value", None)
            values = None if single_value is None else [single_value]

        record: dict[str, Any] = {
            "number": int(getattr(trial, "number", len(records))),
            "state": state_name,
            "params": _json_safe(dict(getattr(trial, "params", {}) or {})),
        }
        if values is not None:
            safe_values = _json_safe(list(values))
            if len(safe_values) == 1:
                record["value"] = safe_values[0]
            else:
                record["values"] = safe_values
            if directions:
                record["directions"] = directions[: len(safe_values)]

        for name in ("feasibility_status", "prune_reason", "prune_rule", "error_code_label"):
            value = attrs.get(name)
            if value not in (None, ""):
                record[name] = _json_safe(value)
        if state_name == "pruned" or bool(attrs.get("pruned", False)):
            record["pruned"] = True
        if state_name == "fail":
            failure_reason = attrs.get("failure_reason") or attrs.get("prune_reason")
            system_attrs = dict(getattr(trial, "system_attrs", {}) or {})
            failure_reason = failure_reason or system_attrs.get("fail_reason")
            if failure_reason:
                record["failure_reason"] = _json_safe(failure_reason)

        for name in MEASUREMENT_FIELDS:
            value = attrs.get(name)
            if _available_measurement(value):
                record[name] = _json_safe(value)
        quantization_mode = attrs.get("quantization_mode")
        if quantization_mode not in (None, ""):
            record["quantization_mode"] = str(quantization_mode)
        task_metrics = attrs.get("task_metrics")
        if isinstance(task_metrics, Mapping):
            record["task_metrics"] = _json_safe(task_metrics)

        records.append(record)
    return tuple(records)
