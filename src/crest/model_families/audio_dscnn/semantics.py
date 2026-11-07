# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Architectural priors for audio search parameters, independent of TensorFlow.

These hints describe typical effects; measured task and hardware results remain
the authority. Legal choices and decoding live in ``decode.py``.
"""
from typing import Any

AUDIO_DSCNN_SEARCH_SEMANTICS: dict[str, dict[str, Any]] = {
    "base_channels": {
        "description": "Controls the initial channel width of the DS-CNN feature extractor.",
        "units": "channels",
        "typical_effects": (
            "A wider base typically increases feature capacity, parameters, compute, and tensor memory.",
        ),
    },
    "num_blocks": {
        "description": "Controls the number of depthwise-plus-pointwise convolution blocks.",
        "units": "blocks",
        "typical_effects": (
            "Additional blocks typically increase depth, receptive-field processing, and compute.",
        ),
    },
    "kernel_time": {
        "description": "Controls convolution-kernel extent along the log-mel time axis.",
        "units": "frames",
        "typical_effects": (
            "Larger temporal kernels typically combine information across more adjacent frames.",
        ),
    },
    "kernel_freq": {
        "description": "Controls convolution-kernel extent along the mel-frequency axis.",
        "units": "mel bins",
        "typical_effects": (
            "Larger frequency kernels typically combine information across more adjacent mel bins.",
        ),
    },
    "stride_schedule": {
        "description": "Controls where time and frequency downsampling occurs across blocks.",
        "typical_effects": (
            "The schedule typically changes intermediate feature-map shapes, retained resolution, and compute.",
        ),
    },
    "channel_growth": {
        "description": "Controls nominal channel-width growth across successive blocks.",
        "units": "ratio",
        "typical_effects": (
            "Higher growth typically shifts more channel capacity and compute into later blocks.",
        ),
    },
    "max_channels": {
        "description": "Caps the pointwise output width of every DS-CNN block.",
        "units": "channels",
        "typical_effects": (
            "The cap typically limits late-stage parameters, compute, and activation width.",
        ),
    },
    "depth_multiplier": {
        "description": "Controls the channel multiplier used by depthwise convolutions.",
        "units": "multiplier",
        "typical_effects": (
            "A larger multiplier typically increases depthwise features, intermediate tensors, and compute.",
        ),
    },
    "pointwise_scale": {
        "description": "Scales the nominal pointwise-convolution output widths.",
        "units": "ratio",
        "typical_effects": (
            "A larger scale typically increases pointwise parameters, compute, and activation width.",
        ),
    },
    "dropout_rate": {
        "description": "Controls the training-time dropout probability before classification.",
        "units": "probability",
        "typical_effects": (
            "Dropout typically changes regularization during training rather than input feature dimensions.",
        ),
    },
    "norm_flag": {
        "description": "Controls whether batch normalization is used after convolutions.",
        "typical_effects": (
            "Normalization typically affects training dynamics and adds normalization operations and state.",
        ),
    },
    "dense_units": {
        "description": "Controls the optional hidden classifier-head width; zero disables that hidden layer.",
        "units": "units",
        "typical_effects": (
            "A larger hidden head typically increases classifier parameters and compute.",
        ),
    },
    "global_pool_type": {
        "description": "Controls how spatial time-frequency features are aggregated before classification.",
        "typical_effects": (
            "Average and maximum pooling typically emphasize different summaries without changing class count.",
        ),
    },
    "activation": {
        "description": "Controls the hidden rectified activation variant used in convolution blocks.",
        "typical_effects": (
            "The activation typically affects feature range and deployment arithmetic characteristics.",
        ),
    },
}

