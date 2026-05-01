"""Helpers for resolving explicit component choices from runtime config."""

from __future__ import annotations

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


def _require_block(config: Any, key: str) -> Any:
    """Return one required top-level config block.

    Parameters
    ----------
    config : Any
        Loaded runtime configuration object.
    key : str
        Required top-level block name.

    Returns
    -------
    Any
        Resolved config block.

    Raises
    ------
    KeyError
        If the block is missing.
    """

    block = cfg_get(config, key, None)
    if block is None:
        raise KeyError(
            f"Missing required '{key}' config block. "
            "TinyODOM-EX now requires explicit dataset/task/model component selection."
        )
    return block


def _require_string(block: Any, key: str, *, block_name: str) -> str:
    """Return one required non-empty string field from a config block.

    Parameters
    ----------
    block : Any
        Config subtree that owns ``key``.
    key : str
        Required field name inside ``block``.
    block_name : str
        Human-readable block label used in errors.

    Returns
    -------
    str
        Normalized string value.

    Raises
    ------
    KeyError
        If the field is missing or empty.
    """

    raw_value = cfg_get(block, key, None)
    if raw_value is None or str(raw_value).strip() == "":
        raise KeyError(f"Missing required '{block_name}.{key}' field.")
    return str(raw_value)


def _normalize_mapping_like(value: Any, *, field_name: str) -> Dict:
    """Normalize a mapping-like or namespace-like config subtree to ``Dict``.

    Parameters
    ----------
    value : Any
        Value to normalize.
    field_name : str
        Human-readable field label used in errors.

    Returns
    -------
    addict.Dict
        Normalized mapping payload.

    Raises
    ------
    TypeError
        If ``value`` cannot be interpreted as mapping-like data.
    """

    if isinstance(value, Dict):
        return Dict(value)
    if isinstance(value, dict):
        return Dict(value)
    if hasattr(value, "__dict__"):
        return Dict(vars(value))
    raise TypeError(
        f"Expected '{field_name}' to be mapping-like, got {type(value).__name__}."
    )


def resolve_component_selection(config: Any) -> dict[str, Any]:
    """Resolve explicit dataset, task, and model-family selections.

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
        ``model_config``. TinyODOM-EX now requires explicit ``dataset``,
        ``task``, and ``model`` blocks and does not fall back to legacy
        top-level config fields.

    Raises
    ------
    KeyError
        If any required component block or required field is missing.
    TypeError
        If a config subtree that must be mapping-like cannot be normalized.
    """

    dataset_block = _require_block(config, "dataset")
    task_block = _require_block(config, "task")
    model_block = _require_block(config, "model")

    dataset_params = cfg_get(dataset_block, "params", None)
    if dataset_params is None:
        raise KeyError("Missing required 'dataset.params' field.")

    # `task.params`, `model.params`, and `model.search` stay optional, but once
    # the component blocks exist they are treated as native modular config
    # payloads instead of compatibility wrappers.
    task_params = _normalize_mapping_like(
        cfg_get(task_block, "params", Dict()),
        field_name="task.params",
    )
    model_params = _normalize_mapping_like(
        cfg_get(model_block, "params", Dict()),
        field_name="model.params",
    )
    model_search = _normalize_mapping_like(
        cfg_get(model_block, "search", Dict()),
        field_name="model.search",
    )
    return {
        "dataset_name": _require_string(dataset_block, "name", block_name="dataset"),
        "dataset_config": _normalize_mapping_like(dataset_params, field_name="dataset.params"),
        "task_name": _require_string(task_block, "name", block_name="task"),
        "task_config": task_params,
        "model_family_name": _require_string(model_block, "family", block_name="model"),
        "model_config": Dict(
            family=_require_string(model_block, "family", block_name="model"),
            params=model_params,
            search=model_search,
        ),
    }
