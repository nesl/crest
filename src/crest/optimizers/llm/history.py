# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Compact recent history and best candidates extracted from Optuna trials."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ...errors import HIL_MASTER_SUCCESS


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


def _history_view(history: Any, directions: tuple[str, ...]) -> Any:
    """Keep existing evidence algorithms usable with snapshots and legacy studies."""
    if isinstance(history, (tuple, list)):
        return SimpleNamespace(trials=history, directions=directions)
    return history


def _state_name(trial: Any) -> str:
    state = getattr(trial, "state", None)
    return str(getattr(state, "name", state)).strip()


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


def build_recent_trial_history(study: Any, *, window_size: int, directions: tuple[str, ...] = ()) -> tuple[dict[str, Any], ...]:
    """Build compact records for only the newest configured Optuna trials."""
    study = _history_view(study, directions)
    if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 0:
        raise ValueError("window_size must be a non-negative integer.")
    if window_size == 0:
        return ()

    return _trial_records(study, list(study.trials)[-window_size:])


def build_terminal_trial_history(study: Any, *, directions: tuple[str, ...] = ()) -> tuple[dict[str, Any], ...]:
    """Return completed, failed and pruned trials; unfinished trials are not evidence."""
    study = _history_view(study, directions)
    trials = sorted((t for t in study.trials
                     if _state_name(t).upper()
                     in {"COMPLETE", "FAIL", "PRUNED"}), key=lambda t: t.number)
    return _trial_records(study, trials)


def build_best_trial_anchors(
    study: Any,
    *,
    anchor_count: int,
    feasibility_enabled: bool = False,
    directions: tuple[str, ...] = (),
) -> tuple[dict[str, Any], ...]:
    """Keep top scalar trials or a bounded, representative feasible Pareto front.

    Oversized fronts retain the region nearest the normalized two-objective knee.
    Flat/degenerate fronts and higher-dimensional fronts use the closest point
    to the normalized ideal as a compromise. Trial numbers break ties.
    """
    study = _history_view(study, directions)
    if isinstance(anchor_count, bool) or not isinstance(anchor_count, int) or anchor_count < 0:
        raise ValueError("anchor_count must be a non-negative integer.")
    if anchor_count == 0:
        return ()
    directions = [str(getattr(d, "name", d)).strip().lower() for d in study.directions]
    candidates: list[tuple[Any, tuple[float, ...]]] = []
    for trial in study.trials:
        if _state_name(trial).upper() != "COMPLETE":
            continue
        attrs = trial.user_attrs
        status = str(attrs.get("feasibility_status", "")).strip().lower()
        if (attrs.get("pruned") or attrs.get("feasible") is False
                or status == "infeasible"
                or (feasibility_enabled and status != "feasible")):
            continue
        # Multi-objective execution failures can be COMPLETE with finite penalties.
        if any(attrs.get(key) not in (None, HIL_MASTER_SUCCESS) for key in ("error_code", "hil_error_code")):
            continue
        if attrs.get("error_code_label") not in (None, "", "HIL_MASTER_SUCCESS"):
            continue
        values = trial.values
        if values is None or len(values) != len(directions) or not all(
            value is not None and math.isfinite(value) for value in values
        ):
            continue
        costs = tuple(value if direction == "minimize" else -value
                      for value, direction in zip(values, directions))
        candidates.append((trial, costs))
    candidates.sort(key=lambda item: item[0].number)
    if len(directions) == 1:
        candidates.sort(key=lambda item: (item[1], item[0].number))
        return _trial_records(study, [trial for trial, _ in candidates[:anchor_count]])

    front = [item for item in candidates if not any(
        all(left <= right for left, right in zip(other[1], item[1]))
        and any(left < right for left, right in zip(other[1], item[1]))
        for other in candidates
    )]
    if len(front) <= anchor_count:
        return _trial_records(study, [trial for trial, _ in front])

    low = [min(item[1][j] for item in front) for j in range(len(directions))]
    spans = [max(item[1][j] for item in front) - low[j] for j in range(len(directions))]
    normalized = [tuple((cost - lo) / span if span else 0.0
                        for cost, lo, span in zip(item[1], low, spans)) for item in front]
    knee = min(range(len(front)), key=lambda i: (
        sum(cost ** 2 for cost in normalized[i]), front[i][0].number,
    ))
    if len(directions) == 2 and all(spans):
        # The normalized extreme chord joins (0, 1) to (1, 0). Its signed
        # distance toward the ideal is proportional to 1 - x - y.
        bend = [1.0 - sum(point) for point in normalized]
        if max(bend) > 1e-12:
            knee = max(range(len(front)), key=lambda i: (bend[i], -front[i][0].number))
    nearest = sorted(range(len(front)), key=lambda i: (
        sum((a - b) ** 2 for a, b in zip(normalized[i], normalized[knee])),
        front[i][0].number,
    ))
    return _trial_records(study, [front[i][0] for i in nearest[:anchor_count]])


def _trial_records(study: Any, trials: list[Any]) -> tuple[dict[str, Any], ...]:
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
