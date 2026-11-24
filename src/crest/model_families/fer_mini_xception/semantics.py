# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Pure descriptive semantics for FER Mini-Xception search parameters."""

FER_MINI_XCEPTION_SEARCH_SEMANTICS = {
    "stem_channels": {"description": "Channels in both valid-padding stem convolutions.", "units": "channels"},
    "num_blocks": {"description": "Number of residual separable-convolution blocks.", "units": "blocks"},
    "first_block_channels": {"description": "Nominal output width of the first residual block.", "units": "channels"},
    "channel_growth": {"description": "Geometric nominal width growth across blocks.", "units": "ratio"},
    "channel_cap": {"description": "Maximum decoded residual-block width.", "units": "channels"},
    "kernel_size": {"description": "Shared square kernel size for separable convolutions.", "units": "pixels"},
    "attention_placement": {"description": "Placement of squeeze-and-excitation channel attention."},
    "downsampling": {"description": "Shared pooling/projection stride schedule across residual blocks."},
}
