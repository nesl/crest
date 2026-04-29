"""Top-level TinyODOM package exports.

This package lazily exposes the major TinyODOM submodules so callers can
import package-level names without eagerly importing heavy runtime
dependencies.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "hardware",
    "model",
    "data",
    "geometry",
    "devices",
    "errors",
    "interfaces",
    "pipeline_types",
    "registry",
]


def __getattr__(name: str) -> Any:
    """Lazily import exported submodules on first attribute access.

    Parameters
    ----------
    name : str
        Package attribute being requested.

    Returns
    -------
    Any
        Imported submodule when ``name`` is listed in :data:`__all__`.

    Raises
    ------
    AttributeError
        If ``name`` is not one of the exported lazy submodules.
    """
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
