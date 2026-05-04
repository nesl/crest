"""Shared cadence helpers for dataset-agnostic runtime budgeting."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

_MISSING = object()


def _field_present(container: Any, key: str) -> bool:
    """Return whether a mapping- or namespace-like object explicitly has a key.

    Parameters
    ----------
    container : Any
        Mapping-like or namespace-like object to inspect.
    key : str
        Field name to check.

    Returns
    -------
    bool
        ``True`` when ``key`` is explicitly present, otherwise ``False``.
    """

    if container is None:
        return False
    if isinstance(container, Mapping):
        return key in container
    return hasattr(container, key)


def _field_value(container: Any, key: str) -> Any:
    """Return one explicit field value or the private missing sentinel.

    Parameters
    ----------
    container : Any
        Mapping-like or namespace-like object to inspect.
    key : str
        Field name to resolve.

    Returns
    -------
    Any
        Explicit field value, or the private missing sentinel when absent.
    """

    if not _field_present(container, key):
        return _MISSING
    if isinstance(container, Mapping):
        return container[key]
    return getattr(container, key)


def _parse_positive_float(raw_value: Any, *, field_name: str) -> float:
    """Parse one positive finite cadence value.

    Parameters
    ----------
    raw_value : Any
        Raw value to parse.
    field_name : str
        Human-readable field name included in errors.

    Returns
    -------
    float
        Parsed positive finite value.

    Raises
    ------
    ValueError
        If ``raw_value`` is null, empty, boolean, nonnumeric, non-finite, or
        non-positive.
    """

    if raw_value in (None, "") or isinstance(raw_value, bool):
        raise ValueError(f"Cadence field '{field_name}' must be a positive finite number.")
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Cadence field '{field_name}' must be numeric.") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"Cadence field '{field_name}' must be a positive finite number.")
    return value


def _resolve_metadata_or_params_value(
    key: str,
    *,
    dataset_params: Any,
    dataset_metadata: Mapping[str, Any] | None,
) -> float:
    """Resolve one legacy cadence field from metadata first, then params.

    Parameters
    ----------
    key : str
        Field name to resolve.
    dataset_params : Any
        ``config.dataset.params`` subtree.
    dataset_metadata : Mapping[str, Any] | None
        Optional loaded ``DatasetBundle.metadata`` mapping.

    Returns
    -------
    float
        Positive finite numeric value.

    Raises
    ------
    ValueError
        If the field is absent from both sources or present but invalid.
    """

    metadata_value = _field_value(dataset_metadata, key)
    if metadata_value is not _MISSING:
        return _parse_positive_float(metadata_value, field_name=f"dataset.metadata.{key}")

    params_value = _field_value(dataset_params, key)
    if params_value is not _MISSING:
        return _parse_positive_float(params_value, field_name=f"dataset.params.{key}")

    raise ValueError(
        f"Unable to resolve cadence field '{key}' from dataset metadata or dataset.params."
    )


def resolve_batch_period_ms(
    dataset_params: Any,
    dataset_metadata: Mapping[str, Any] | None = None,
    device_config: Any | None = None,
) -> float:
    """Resolve the logical per-batch runtime cadence in milliseconds.

    Parameters
    ----------
    dataset_params : Any
        ``config.dataset.params`` subtree. This is the same object returned as
        ``selection["dataset_config"]`` by component selection.
    dataset_metadata : Mapping[str, Any] | None, optional
        Loaded ``DatasetBundle.metadata``. When present, its
        ``batch_period_ms`` field has precedence over config-owned dataset
        cadence fields.
    device_config : Any | None, optional
        Device configuration subtree. Explicit ``device.latency_budget_ms``
        overrides all dataset-derived cadence values.

    Returns
    -------
    float
        Positive finite logical cadence period in milliseconds.

    Raises
    ------
    ValueError
        If no valid cadence contract can be resolved.
    """

    device_budget = _field_value(device_config, "latency_budget_ms")
    if device_budget is not _MISSING and device_budget is not None:
        return _parse_positive_float(device_budget, field_name="device.latency_budget_ms")

    metadata_batch = _field_value(dataset_metadata, "batch_period_ms")
    if metadata_batch is not _MISSING:
        return _parse_positive_float(
            metadata_batch,
            field_name="dataset.metadata.batch_period_ms",
        )

    params_batch = _field_value(dataset_params, "batch_period_ms")
    if params_batch is not _MISSING:
        return _parse_positive_float(params_batch, field_name="dataset.params.batch_period_ms")

    # Legacy odometry cadence uses two independent fields. Preserve the current
    # metadata-first lookup for each field so mixed metadata/config sources
    # continue to work.
    stride = _resolve_metadata_or_params_value(
        "stride",
        dataset_params=dataset_params,
        dataset_metadata=dataset_metadata,
    )
    sampling_rate_hz = _resolve_metadata_or_params_value(
        "sampling_rate_hz",
        dataset_params=dataset_params,
        dataset_metadata=dataset_metadata,
    )
    return (stride / sampling_rate_hz) * 1000.0
