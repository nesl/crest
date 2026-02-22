#!/usr/bin/env python3
"""
Run exactly one perturbed-variant HIL pass and print the resulting metrics.

This script is opinionated for the perturbation experiment:
- Forces energy-aware mode on.
- Forces uniform input mode.
- Forces model_variant='approx_trained'.
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

from hil_server import HILServer, PERTURBED_VARIANT_NAME
from tinyodom.model import DEFAULT_CONFIG_PATH, build_tinyodom_model, count_flops


def _build_hyperparams(server: HILServer) -> Dict:
    """
    Construct the fixed TinyOdom hyperparameter set and annotate FLOPs.
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Run one HIL pass with the BN+bias perturbed model variant."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config YAML.")
    parser.add_argument(
        "--output",
        default="analysis_scripts/hil_single_run/last_run_perturbed.json",
        help="JSON output path for the single-run metrics.",
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

    # Force this experiment's intended runtime path.
    server.config.training.energy_aware = True
    server.set_input_mode("uniform")

    print("Effective run settings:")
    print(f"  model_variant: {PERTURBED_VARIANT_NAME}")
    print(f"  input_mode: {server.config.training.input_mode}")
    print(f"  energy_aware: {bool(server.config.training.energy_aware)}")
    print(f"  harness_arm_pin: {server.config.device.harness_arm_pin}")
    print(f"  harness_trigger_pin: {server.config.device.harness_trigger_pin}")
    print(f"  dut_arm_hold_ms: {server.config.device.dut_arm_hold_ms}")
    print(f"  harness_stable_low_ms: {server.config.device.harness_stable_low_ms}")

    hyperparams = _build_hyperparams(server)
    metrics = server.determine_metrics(hyperparams, model_variant=PERTURBED_VARIANT_NAME)

    print("Single perturbed HIL metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nWrote metrics JSON: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
