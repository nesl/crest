# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Normalize configuration and scalar values without importing ML runtimes."""
from __future__ import annotations

from typing import Any

def _cfg_get(container: Any, key: str, default: Any = None) -> Any:
    """Read a field from a mapping-style or attribute-style configuration.

    Parameters
    ----------
    container : Any
        Configuration object or ``None``.
    key : str
        Field to read.
    default : Any, optional
        Fallback for absent fields or containers.

    Returns
    -------
    Any
        Configured value or the fallback.
    """
    if container is None:
        return default
    getter = getattr(container, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(container, key, default)


def _json_scalar(value: Any) -> Any | None:
    """Normalize a finite JSON scalar, including NumPy-like scalar values.

    Parameters
    ----------
    value : Any
        Scalar or NumPy-like value exposing ``item``.

    Returns
    -------
    Any or None
        Finite JSON scalar, or ``None`` for unsupported or nonfinite input.
    """
    if hasattr(value, "item") and callable(value.item):
        try:
            value = value.item()
        except (TypeError, ValueError):
            return None
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else None
    return None


def _json_shape(value: Any) -> list[Any] | None:
    """Normalize a shape sequence without retaining dataset payloads.

    Parameters
    ----------
    value : Any
        List or tuple of scalar shape dimensions.

    Returns
    -------
    list or None
        Normalized dimensions, or ``None`` when the input cannot be represented.
    """
    if not isinstance(value, (list, tuple)):
        return None
    normalized = [_json_scalar(item) for item in value]
    if any(item is None and original is not None for item, original in zip(normalized, value)):
        return None
    return normalized


