# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Lazy exports for concrete model-family implementations."""

from importlib import import_module
from typing import Any

__all__ = ["AudioDSCNNFamily", "OdomTCNFamily"]


def __getattr__(name: str) -> Any:
    """Load a public export on demand.

    Parameters
    ----------
    name : str
        Exported package attribute to load.

    Returns
    -------
    Any
        Requested family class, metadata, or decoding helper.

    Raises
    ------
    AttributeError
        If the requested name is not a package export.
    """
    modules = {"AudioDSCNNFamily": "audio_dscnn", "OdomTCNFamily": "odom_tcn"}
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{modules[name]}"), name)
    globals()[name] = value
    return value
