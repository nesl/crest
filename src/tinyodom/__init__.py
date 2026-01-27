from __future__ import annotations

import importlib
from typing import Any

__all__ = ["hardware", "model", "data", "geometry", "devices"]


def __getattr__(name: str) -> Any:
    """Lazily import submodules to avoid heavy deps at import time."""
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
