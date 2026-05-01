#!/usr/bin/env python3
"""
Shared fixed architecture spec for HIL noise-scan experiments.
"""

from __future__ import annotations

from addict import Dict

from tinyodom.analysis_support import build_fixed_tinyodom_hyperparams


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
    return build_fixed_tinyodom_hyperparams(
        window_size=int(window_size),
        input_dim=int(input_dim),
    )
