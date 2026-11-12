# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Pure FER Mini-Xception search decoding without ML-runtime imports."""

from __future__ import annotations

from typing import Any, Mapping

STEM_CHANNELS_CHOICES = (4, 8, 12, 16)
NUM_BLOCKS_CHOICES = (3, 4)
FIRST_BLOCK_CHANNELS_CHOICES = (8, 16, 24, 32)
CHANNEL_GROWTH_CHOICES = (1.0, 1.5, 2.0)
CHANNEL_CAP_CHOICES = (64, 96, 128)
KERNEL_SIZE_CHOICES = (3, 5)
ATTENTION_PLACEMENT_CHOICES = ("none", "last", "last_two")
DOWNSAMPLING_CHOICES = ("every_block", "preserve_first")

FER_MINI_XCEPTION_SEARCH_CHOICES: dict[str, tuple[Any, ...]] = {
    "stem_channels": STEM_CHANNELS_CHOICES,
    "num_blocks": NUM_BLOCKS_CHOICES,
    "first_block_channels": FIRST_BLOCK_CHANNELS_CHOICES,
    "channel_growth": CHANNEL_GROWTH_CHOICES,
    "channel_cap": CHANNEL_CAP_CHOICES,
    "kernel_size": KERNEL_SIZE_CHOICES,
    "attention_placement": ATTENTION_PLACEMENT_CHOICES,
    "downsampling": DOWNSAMPLING_CHOICES,
}

DEFAULT_FER_MINI_XCEPTION_SEED: dict[str, Any] = {
    "stem_channels": 8,
    "num_blocks": 4,
    "first_block_channels": 16,
    "channel_growth": 2.0,
    "channel_cap": 128,
    "kernel_size": 3,
    "attention_placement": "last_two",
    "downsampling": "every_block",
}


def _normalize_growth(value: Any) -> float:
    """Normalize one exact channel-growth category."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("channel_growth must be an allowed numeric category.")
    normalized = float(value)
    if normalized not in CHANNEL_GROWTH_CHOICES:
        raise ValueError(f"channel_growth must be one of {CHANNEL_GROWTH_CHOICES}.")
    return normalized


def validate_raw_hparams(hparams: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one complete raw FER search payload."""
    expected = set(FER_MINI_XCEPTION_SEARCH_CHOICES)
    missing = sorted(expected - set(hparams))
    extra = sorted(set(hparams) - expected)
    if missing:
        raise ValueError(f"FER Mini-Xception hyperparameters missing keys: {', '.join(missing)}.")
    if extra:
        raise ValueError(f"FER Mini-Xception hyperparameters contain extra keys: {', '.join(extra)}.")
    normalized = dict(hparams)
    normalized["channel_growth"] = _normalize_growth(normalized["channel_growth"])
    integer_names = {
        "stem_channels",
        "num_blocks",
        "first_block_channels",
        "channel_cap",
        "kernel_size",
    }
    for name, choices in FER_MINI_XCEPTION_SEARCH_CHOICES.items():
        value = normalized[name]
        if name in integer_names and (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError(f"{name} must be one of {choices}.")
        if name in {"attention_placement", "downsampling"} and not isinstance(value, str):
            raise ValueError(f"{name} must be one of {choices}.")
        if value not in choices:
            raise ValueError(f"{name} must be one of {choices}; got {value!r}.")
    return normalized


def decode_block_widths(hparams: Mapping[str, Any]) -> tuple[int, ...]:
    """Decode capped widths using Python's deterministic ties-to-even round."""
    values = validate_raw_hparams(hparams)
    return tuple(
        min(
            values["channel_cap"],
            max(1, round(values["first_block_channels"] * values["channel_growth"] ** index)),
        )
        for index in range(values["num_blocks"])
    )


def decode_block_strides(hparams: Mapping[str, Any]) -> tuple[int, ...]:
    """Decode the shared pooling/projection stride for every residual block."""
    values = validate_raw_hparams(hparams)
    if values["downsampling"] == "every_block":
        return (2,) * values["num_blocks"]
    return (1,) + (2,) * (values["num_blocks"] - 1)


def decode_attention_indices(hparams: Mapping[str, Any]) -> tuple[int, ...]:
    """Decode zero-based residual-block indices that receive SE attention."""
    values = validate_raw_hparams(hparams)
    if values["attention_placement"] == "none":
        return ()
    if values["attention_placement"] == "last":
        return (values["num_blocks"] - 1,)
    return (values["num_blocks"] - 2, values["num_blocks"] - 1)


def decoded_architecture_identity(
    hparams: Mapping[str, Any],
    *,
    stem_convolutions: int = 2,
    separable_convolutions_per_block: int = 2,
    se_reduction_ratio: int = 8,
    l2: float = 0.001,
) -> dict[str, Any]:
    """Return a canonical identity derived from the actual training graph.

    Redundant raw growth/cap combinations collapse when they produce the same
    widths. CREST's current optimizer duplicate policies do not consume this
    family helper, so redundant raw combinations may still be sampled.
    """
    values = validate_raw_hparams(hparams)
    return {
        "schema_version": "fer-mini-xception-training-v1",
        "logical_input_adapter": "reshape_h_w_to_h_w_1",
        "stem_channels": values["stem_channels"],
        "stem_convolutions": int(stem_convolutions),
        "block_widths": list(decode_block_widths(values)),
        "block_strides": list(decode_block_strides(values)),
        "kernel_size": values["kernel_size"],
        "attention_block_indices": list(decode_attention_indices(values)),
        "separable_convolutions_per_block": int(separable_convolutions_per_block),
        "se_reduction_ratio": int(se_reduction_ratio),
        "l2": float(l2),
        "head": "conv3_same_average_pool_flatten_softmax5",
    }
