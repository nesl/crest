#!/usr/bin/env python3
"""
Shared fixed architecture spec for HIL noise-scan experiments.
"""

from __future__ import annotations

from addict import Dict

from tinyodom.model import build_tinyodom_model, count_flops


def build_noise_scan_hyperparams(window_size: int, input_dim: int) -> Dict:
    """
    Return the fixed hyperparameter set used by HIL noise scan scripts.

    Parameters
    ----------
    window_size : int
        Number of timesteps in each input window.
    input_dim : int
        Number of channels per timestep.

    Returns
    -------
    addict.Dict
        Hyperparameters including computed FLOPs.
    """
    hyperparams = Dict(
        nb_filters=10,
        kernel_size=12,
        dilations=[1, 4, 8, 64],
        dropout_rate=0.0,
        use_skip_connections=False,
        norm_flag=True,
        batch_size=256,
        timesteps=int(window_size),
        input_dim=int(input_dim),
    )
    model = build_tinyodom_model(hyperparams)
    hyperparams.flops = count_flops(model, (hyperparams.timesteps, hyperparams.input_dim))
    return hyperparams

