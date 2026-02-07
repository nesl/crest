#!/usr/bin/env python3
"""
Run a single HIL controller pass and print the resulting metrics.

This is a lightweight sanity check to confirm the repository layout,
Arduino toolchain, and board communication are all working together.

Examples
--------
python analysis_scripts/hil_single_run/run_single_hil.py
python analysis_scripts/hil_single_run/run_single_hil.py --input-mode standard
python analysis_scripts/hil_single_run/run_single_hil.py --output analysis_scripts/hil_single_run/last_run.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from addict import Dict

sys.path.insert(0, os.path.abspath("src"))

from hil_server import HILServer
from tinyodom.model import DEFAULT_CONFIG_PATH, build_tinyodom_model, count_flops


def _build_hyperparams(server: HILServer) -> Dict:
    """
    Construct a fixed hyperparameter set and annotate it with model FLOPs.

    Parameters
    ----------
    server : HILServer
        Active HIL server instance used to resolve window size and input dims.

    Returns
    -------
    addict.Dict
        Hyperparameter dictionary with an added ``flops`` attribute.
    """
    hyperparams = Dict(
        nb_filters=10,
        kernel_size=12,
        dilations=[1, 4, 8, 64],
        dropout_rate=0.0,
        use_skip_connections=False,
        norm_flag=True,
        batch_size=256,
        timesteps=server.config.data.window_size,
        input_dim=server.training_data.inputs.shape[2],
    )
    model = build_tinyodom_model(hyperparams)
    hyperparams.flops = count_flops(model, (hyperparams.timesteps, hyperparams.input_dim))
    return hyperparams


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one HIL controller pass and print metrics.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config YAML.")
    parser.add_argument(
        "--input-mode",
        default=None,
        help="Override input_mode for this run (standard/uniform/representative/real).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path to write the metrics.",
    )
    args = parser.parse_args()

    server = HILServer(config_path=Path(args.config))
    if args.input_mode:
        server.set_input_mode(args.input_mode)

    hyperparams = _build_hyperparams(server)
    metrics = server.determine_metrics(hyperparams)

    print("Single HIL metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2))
        print(f"\nWrote metrics JSON: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
