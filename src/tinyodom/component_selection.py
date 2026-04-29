"""Helpers for resolving component choices from legacy and modular config.

This module bridges the older global TinyOdom config layout with the newer
explicit dataset, task, and model-family selection blocks. The helpers accept
mapping-like and namespace-like config containers so callers can reuse the same
selection logic across runtime config representations.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from addict import Dict


def cfg_get(container: Any, key: str, default: Any = None) -> Any:
    """Read one field from a mapping-like or namespace-like container.

    Parameters
    ----------
    container : Any
        Object exposing either ``get`` or attribute access. When both are
        available, a callable ``get`` is preferred.
    key : str
        Field name to resolve.
    default : Any, optional
        Fallback value returned when ``key`` is unavailable.

    Returns
    -------
    Any
        Resolved configuration value or ``default``.
    """

    getter = getattr(container, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(container, key, default)


def resolve_component_selection(config: Any) -> dict[str, Any]:
    """Resolve active dataset, task, and model-family selections.

    Parameters
    ----------
    config : Any
        Loaded HIL/NAS configuration object.

    Returns
    -------
    dict[str, Any]
        Resolved component names and local configuration subtrees. The returned
        mapping always includes ``dataset_name``, ``dataset_config``,
        ``task_name``, ``task_config``, ``model_family_name``, and
        ``model_config``. When no explicit component blocks are provided, the
        helper falls back to the legacy defaults ``"oxiod"``,
        ``"odometry_regression"``, and ``"tinyodom_tcn"``. Dataset
        configuration prefers ``config.dataset.params`` when present and not
        ``None``; otherwise it falls back to legacy ``config.data``.
        ``task_config`` defaults to an empty ``Dict()``, and ``model_config``
        is returned as a ``SimpleNamespace`` with ``params`` and ``search``
        attributes.
    """

    dataset_block = cfg_get(config, "dataset", None)
    task_block = cfg_get(config, "task", None)
    model_block = cfg_get(config, "model", None)
    dataset_config = cfg_get(config, "data", None)
    if dataset_block is not None:
        dataset_params = cfg_get(dataset_block, "params", None)
        if dataset_params is not None:
            dataset_config = dataset_params
    return {
        "dataset_name": str(cfg_get(dataset_block, "name", "oxiod")),
        "dataset_config": dataset_config,
        "task_name": str(cfg_get(task_block, "name", "odometry_regression")),
        "task_config": cfg_get(task_block, "params", Dict()) if task_block is not None else Dict(),
        "model_family_name": str(cfg_get(model_block, "family", "tinyodom_tcn")),
        "model_config": SimpleNamespace(
            params=cfg_get(model_block, "params", Dict()) if model_block is not None else Dict(),
            search=cfg_get(model_block, "search", Dict()) if model_block is not None else Dict(),
        ),
    }
