#!/usr/bin/env python3
"""
Run a multi-mode HIL energy/latency noise scan and save results to CSV.

This script cycles through one or more input modes (uniform, representative,
real), re-syncs the Arduino sketch variant for each mode, and records the
per-run metrics returned by the HIL controller.

Examples
--------
python analysis_scripts/hil_noise_analysis/hil_energy_noise_scan.py
python analysis_scripts/hil_noise_analysis/hil_energy_noise_scan.py --runs 5 --cooldown 10
python analysis_scripts/hil_noise_analysis/hil_energy_noise_scan.py --input-modes uniform,real
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

from addict import Dict
from tqdm import tqdm

sys.path.insert(0, os.path.abspath("src"))

from hil_server import HILServer, logger
from nas_utils_ex import DEFAULT_CONFIG_PATH, build_tinyodom_model, count_flops


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
    parser = argparse.ArgumentParser(description="Run HIL energy/latency noise scans.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config YAML.")
    parser.add_argument("--runs", type=int, default=20, help="Number of runs per input mode.")
    parser.add_argument("--cooldown", type=float, default=20.0, help="Cooldown seconds between runs.")
    parser.add_argument(
        "--csv-path",
        default="hil_energy_noise_scan.csv",
        help="Output CSV file path.",
    )
    parser.add_argument(
        "--input-modes",
        default="uniform,representative,real",
        help="Comma-separated list of input modes to scan.",
    )
    parser.add_argument(
        "--energy-aware",
        action="store_true",
        help="Force energy-aware sketch selection even if config disables it.",
    )
    args = parser.parse_args()

    input_modes = [mode.strip().lower() for mode in args.input_modes.split(",") if mode.strip()]
    if not input_modes:
        raise ValueError("No input modes provided. Use --input-modes with at least one mode.")

    server = HILServer(config_path=Path(args.config))
    if args.energy_aware:
        server.config.training.energy_aware = True
        server.set_input_mode(server.config.training.get("input_mode", "uniform"))

    hyperparams = _build_hyperparams(server)

    csv_path = Path(args.csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting noise scan: %s runs per mode, cooldown %.1fs, modes=%s",
        args.runs,
        args.cooldown,
        ", ".join(input_modes),
    )

    with csv_path.open("w", newline="") as csvfile:
        writer = None
        for mode in input_modes:
            server.set_input_mode(mode)
            for run_idx in tqdm(range(1, args.runs + 1), desc=f"HIL noise scan ({mode})"):
                logger.info("Noise scan mode=%s run %d/%d", mode, run_idx, args.runs)
                metrics = server.determine_metrics(hyperparams)
                if writer is None:
                    fieldnames = ["input_mode", "run_index", *metrics.keys()]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                row = {"input_mode": mode, "run_index": run_idx, **metrics}
                writer.writerow(row)
                csvfile.flush()
                if run_idx < args.runs:
                    logger.info("Cooling down for %.1f seconds", args.cooldown)
                    for _ in tqdm(range(int(args.cooldown)), desc="Cooldown", leave=False):
                        time.sleep(1)

    logger.info("Noise scan complete. Metrics saved to %s", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
