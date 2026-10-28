"""Pure authoritative audio DS-CNN graph decoding; no ML runtime imports."""

from __future__ import annotations

from typing import Any

BASE_CHANNELS_CHOICES = (4, 8, 12, 16, 20, 24, 32)
NUM_BLOCKS_CHOICES = (2, 3, 4, 5, 6)
KERNEL_TIME_CHOICES = (3, 5)
KERNEL_FREQ_CHOICES = (3, 5)
STRIDE_SCHEDULE_CHOICES = ("light", "balanced", "aggressive", "early_time", "early_freq", "late")
CHANNEL_GROWTH_CHOICES = (1.0, 1.5, 2.0)
MAX_CHANNELS_CHOICES = (64, 96, 128)
DEPTH_MULTIPLIER_CHOICES = (1, 2)
POINTWISE_SCALE_CHOICES = (0.75, 1.0, 1.25, 1.5)
DROPOUT_RATE_CHOICES = (0.0, 0.1, 0.2, 0.3)
NORM_FLAG_CHOICES = (True, False)
DENSE_UNITS_CHOICES = (0, 16, 32, 64)
GLOBAL_POOL_TYPE_CHOICES = ("avg", "max")
ACTIVATION_CHOICES = ("relu", "relu6")

AUDIO_DSCNN_SEARCH_CHOICES: dict[str, tuple[Any, ...]] = {
    "base_channels": BASE_CHANNELS_CHOICES,
    "num_blocks": NUM_BLOCKS_CHOICES,
    "kernel_time": KERNEL_TIME_CHOICES,
    "kernel_freq": KERNEL_FREQ_CHOICES,
    "stride_schedule": STRIDE_SCHEDULE_CHOICES,
    "channel_growth": CHANNEL_GROWTH_CHOICES,
    "max_channels": MAX_CHANNELS_CHOICES,
    "depth_multiplier": DEPTH_MULTIPLIER_CHOICES,
    "pointwise_scale": POINTWISE_SCALE_CHOICES,
    "dropout_rate": DROPOUT_RATE_CHOICES,
    "norm_flag": NORM_FLAG_CHOICES,
    "dense_units": DENSE_UNITS_CHOICES,
    "global_pool_type": GLOBAL_POOL_TYPE_CHOICES,
    "activation": ACTIVATION_CHOICES,
}



def resolve_stride_schedule(name: str, num_blocks: int) -> tuple[tuple[int, int], ...]:
    """Resolve a named stride schedule for a variable number of DS-CNN blocks.

    Parameters
    ----------
    name : str
        Schedule name from ``STRIDE_SCHEDULE_CHOICES``.
    num_blocks : int
        Number of repeated DS-CNN blocks in the model.

    Returns
    -------
    tuple[tuple[int, int], ...]
        Per-block ``(time_stride, freq_stride)`` pairs.

    Raises
    ------
    ValueError
        If the schedule name or block count is invalid.
    """
    if name not in STRIDE_SCHEDULE_CHOICES:
        raise ValueError(f"audio_dscnn stride_schedule must be one of {STRIDE_SCHEDULE_CHOICES}.")
    if isinstance(num_blocks, bool) or not isinstance(num_blocks, int) or num_blocks <= 0:
        raise ValueError("audio_dscnn num_blocks must be a positive integer.")

    strides = [(1, 1) for _ in range(num_blocks)]

    def set_when_present(index: int, stride: tuple[int, int]) -> None:
        """Assign one stride when its zero-based block index exists.

        Parameters
        ----------
        index : int
            Block index used to derive convolution naming and placement.
        stride : tuple[int, int]
            Temporal stride applied by the convolution block.
        """
        if 0 <= index < num_blocks:
            strides[index] = stride

    if name == "light":
        set_when_present(1, (2, 2))
    elif name == "balanced":
        set_when_present(0, (2, 2))
        set_when_present(2, (2, 2))
    elif name == "aggressive":
        set_when_present(0, (2, 2))
        set_when_present(1, (2, 2))
        set_when_present(2, (2, 2))
    elif name == "early_time":
        set_when_present(0, (2, 1))
        set_when_present(1, (2, 1))
    elif name == "early_freq":
        set_when_present(0, (1, 2))
        set_when_present(1, (1, 2))
    elif name == "late":
        set_when_present(num_blocks - 1, (2, 2))
        if num_blocks >= 4:
            set_when_present(num_blocks - 3, (2, 2))

    return tuple(strides)

def _pointwise_filters(hparams: dict[str, Any], block_index: int) -> int:
    """Compute the pointwise output width for one DS-CNN block.

    Parameters
    ----------
    hparams : dict[str, Any]
        Validated audio DS-CNN hyperparameters.
    block_index : int
        Zero-based block index.

    Returns
    -------
    int
        Clamped pointwise ``Conv2D`` filter count.
    """
    nominal = round(hparams["base_channels"] * (hparams["channel_growth"] ** block_index))
    filters = round(nominal * hparams["pointwise_scale"])
    return min(hparams["max_channels"], max(1, filters))
