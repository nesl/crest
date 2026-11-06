# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Audio DS-CNN family exports with runtime-independent decoding helpers."""

from importlib import import_module
from typing import Any

_DECODE_EXPORTS = (
    "AUDIO_DSCNN_SEARCH_CHOICES",
    "BASE_CHANNELS_CHOICES",
    "NUM_BLOCKS_CHOICES",
    "KERNEL_TIME_CHOICES",
    "KERNEL_FREQ_CHOICES",
    "STRIDE_SCHEDULE_CHOICES",
    "CHANNEL_GROWTH_CHOICES",
    "MAX_CHANNELS_CHOICES",
    "DEPTH_MULTIPLIER_CHOICES",
    "POINTWISE_SCALE_CHOICES",
    "DROPOUT_RATE_CHOICES",
    "NORM_FLAG_CHOICES",
    "DENSE_UNITS_CHOICES",
    "GLOBAL_POOL_TYPE_CHOICES",
    "ACTIVATION_CHOICES",
    "resolve_stride_schedule",
    "_pointwise_filters",
)
_FAMILY_EXPORTS = (
    "AudioDSCNNFamily",
    "AUDIO_DSCNN_SEARCH_SEMANTICS",
    "DEFAULT_AUDIO_DSCNN_SEED",
    "INTEGER_CATEGORICAL_FIELDS",
    "BOOLEAN_FIELDS",
    "FLOAT_CATEGORICAL_FIELDS",
    "STRING_CATEGORICAL_FIELDS",
)

__all__ = list(_FAMILY_EXPORTS) + list(_DECODE_EXPORTS)


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
    if name in _DECODE_EXPORTS:
        module = import_module(f"{__name__}.decode")
    elif name in _FAMILY_EXPORTS:
        module = import_module(f"{__name__}.family")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(module, name)
    globals()[name] = value
    return value
