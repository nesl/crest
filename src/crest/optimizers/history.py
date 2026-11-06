# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Copy study evidence into immutable, ordinary proposal records."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ..pipeline_types import TrialRecord


def to_plain(value: Any, *, freeze: bool = False, nonfinite_to_none: bool = False) -> Any:
    """Normalize nested evidence with explicit mutability and metric policies.

    Parameters
    ----------
    value : Any
        Evidence containing mappings, sequences, dataclasses, paths or scalars.
    freeze : bool, optional
        Return immutable mappings and tuples for optimizer snapshots.
    nonfinite_to_none : bool, optional
        Replace nonfinite metrics with null for prompts and JSON ledgers. Leave
        false for identity data so strict JSON encoding rejects invalid values.

    Returns
    -------
    Any
        Detached plain data, optionally frozen.
    """
    def convert(item):
        """Apply the selected normalization policies recursively.

        Parameters
        ----------
        item : Any
            Nested evidence value.

        Returns
        -------
        Any
            Normalized value.
        """
        return to_plain(item, freeze=freeze, nonfinite_to_none=nonfinite_to_none)

    if is_dataclass(value) and not isinstance(value, type):
        value = {field.name: getattr(value, field.name) for field in fields(value)}
    if isinstance(value, Mapping):
        result = {str(key): convert(item) for key, item in value.items()}
        return MappingProxyType(result) if freeze else result
    if isinstance(value, (list, tuple)):
        result = [convert(item) for item in value]
        return tuple(result) if freeze else result
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(value.item):
        return convert(value.item())
    if nonfinite_to_none and isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def freeze_plain_data(value: Any) -> Any:
    """Preserve immutable evidence without sanitizing nonfinite numbers.

    Parameters
    ----------
    value : Any
        Study evidence to detach.

    Returns
    -------
    Any
        Immutable normalized evidence.
    """
    return to_plain(value, freeze=True)


def build_trial_snapshot(study: Any) -> tuple[TrialRecord, ...]:
    """Copy study trials into detached immutable evidence records.

    Parameters
    ----------
    study : Any
        Study exposing ordered trials and their Optuna-style attributes.

    Returns
    -------
    tuple of TrialRecord
        Trial snapshots in study iteration order, with normalized state names.
    """
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
