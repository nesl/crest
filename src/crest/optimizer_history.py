# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Copy study evidence into immutable, ordinary proposal records."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .pipeline_types import TrialRecord


def freeze_plain_data(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): freeze_plain_data(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_plain_data(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(value.item):
        return freeze_plain_data(value.item())
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def build_trial_snapshot(study: Any) -> tuple[TrialRecord, ...]:
    """Preserve actual params, eligibility attrs and failure evidence, by number."""
    records = []
    for trial in study.trials:
        values = getattr(trial, "values", None)
        if values is None:
            value = getattr(trial, "value", None)
            values = None if value is None else (value,)
        state = getattr(trial, "state", None)
        records.append(TrialRecord(
            number=int(trial.number),
            state=str(getattr(state, "name", state)).strip().upper(),
            params=freeze_plain_data(dict(getattr(trial, "params", {}) or {})),
            values=None if values is None else tuple(freeze_plain_data(v) for v in values),
            user_attrs=freeze_plain_data(dict(getattr(trial, "user_attrs", {}) or {})),
            system_attrs=freeze_plain_data(dict(getattr(trial, "system_attrs", {}) or {})),
        ))
    return tuple(records)
