#!/usr/bin/env python3
"""
Run a single HIL controller pass and print the resulting metrics.

This is a lightweight sanity check to confirm the repository layout,
Arduino toolchain, and board communication are all working together.

Examples
--------
python analysis_scripts/hil_single_run/run_single_hil.py
python analysis_scripts/hil_single_run/run_single_hil.py --input-mode uniform
python analysis_scripts/hil_single_run/run_single_hil.py --output analysis_scripts/hil_single_run/last_run.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from addict import Dict

sys.path.insert(0, os.path.abspath("src"))

from hil_server import HILServer
from tinyodom.analysis_support import (
    build_fixed_tinyodom_hyperparams,
    split_hil_request_hyperparams,
)
from tinyodom.model import DEFAULT_CONFIG_PATH


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
    window_size, input_dim = server.get_runtime_dimensions()
    return build_fixed_tinyodom_hyperparams(
        window_size=window_size,
        input_dim=input_dim,
    )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )
    parser = argparse.ArgumentParser(description="Run one HIL controller pass and print metrics.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config YAML.")
    parser.add_argument(
        "--input-mode",
        type=str.lower,
        choices=["uniform", "representative", "real"],
        default=None,
        help="Override input_mode for this run (uniform/representative/real).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path to write the metrics.",
    )
    parser.add_argument(
        "--harness-arm-pin",
        type=int,
        default=None,
        help="Override device.harness_arm_pin for this run.",
    )
    parser.add_argument(
        "--harness-trigger-pin",
        type=int,
        default=None,
        help="Override device.harness_trigger_pin for this run.",
    )
    parser.add_argument(
        "--dut-arm-hold-ms",
        type=int,
        default=None,
        help="Override device.dut_arm_hold_ms for this run.",
    )
    parser.add_argument(
        "--harness-stable-low-ms",
        type=int,
        default=None,
        help="Override device.harness_stable_low_ms for this run.",
    )
    args = parser.parse_args()

    server = HILServer(config_path=Path(args.config))
    if args.harness_arm_pin is not None:
        server.config.device.harness_arm_pin = args.harness_arm_pin
    if args.harness_trigger_pin is not None:
        server.config.device.harness_trigger_pin = args.harness_trigger_pin
    if args.dut_arm_hold_ms is not None:
        server.config.device.dut_arm_hold_ms = args.dut_arm_hold_ms
    if args.harness_stable_low_ms is not None:
        server.config.device.harness_stable_low_ms = args.harness_stable_low_ms
    if args.input_mode:
        server.set_input_mode(args.input_mode)

    print("Effective harness wiring/timing:")
    print(f"  harness_arm_pin: {server.config.device.harness_arm_pin}")
    print(f"  harness_trigger_pin: {server.config.device.harness_trigger_pin}")
    print(f"  dut_arm_hold_ms: {server.config.device.dut_arm_hold_ms}")
    print(f"  harness_stable_low_ms: {server.config.device.harness_stable_low_ms}")

    hyperparams = _build_hyperparams(server)
    family_hparams, runtime_metadata = split_hil_request_hyperparams(hyperparams)
    metrics = server.determine_metrics(
        family_hparams,
        runtime_metadata,
        model_variant="approx_trained",
    )

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
