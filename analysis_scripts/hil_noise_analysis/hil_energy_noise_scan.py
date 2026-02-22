#!/usr/bin/env python3
"""
Run a multi-mode HIL energy/latency noise scan and save results to CSV.

This script cycles through one or more model variants (e.g., trained_50ep,
untrained) and input modes (uniform, representative, real), re-syncs the
Arduino sketch variant for each mode, and records the per-run metrics returned
by the HIL controller.

Examples
--------
python analysis_scripts/hil_noise_analysis/hil_energy_noise_scan.py
python analysis_scripts/hil_noise_analysis/hil_energy_noise_scan.py --runs 5 --cooldown 10
python analysis_scripts/hil_noise_analysis/hil_energy_noise_scan.py --input-modes uniform,real
python analysis_scripts/hil_noise_analysis/hil_energy_noise_scan.py --model-variants trained_50ep,untrained --trained-checkpoint artifacts/noise_scan_50ep.keras
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, os.path.abspath("src"))

from hil_server import HILServer, logger
from tinyodom.model import DEFAULT_CONFIG_PATH

from noise_scan_model_spec import build_noise_scan_hyperparams

SUPPORTED_UNTRAINED_VARIANTS = {
    "untrained",
    "approx_trained",
    "representative",  # legacy alias
    "bn_full_plus_non_bn_bias_perturbed",  # legacy alias
}


def _parse_csv_list(raw: str, field_name: str) -> list[str]:
    values = [value.strip().lower() for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError(f"No {field_name} provided.")
    return values


def _is_trained_variant(model_variant: str) -> bool:
    return str(model_variant).strip().lower().startswith("trained")


def _is_supported_variant(model_variant: str) -> bool:
    variant = str(model_variant).strip().lower()
    return variant in SUPPORTED_UNTRAINED_VARIANTS or _is_trained_variant(variant)


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
    parser.add_argument(
        "--model-variants",
        default="untrained",
        help=(
            "Comma-separated model variants to scan (e.g. trained_50ep,untrained,"
            "approx_trained)."
        ),
    )
    parser.add_argument(
        "--trained-checkpoint",
        default=None,
        help="Path to trained .keras checkpoint used for trained model variants.",
    )
    parser.add_argument(
        "--trained-meta",
        default=None,
        help="Optional JSON metadata produced by train_noise_scan_model.py.",
    )
    args = parser.parse_args()

    input_modes = _parse_csv_list(args.input_modes, "input modes")
    model_variants = _parse_csv_list(args.model_variants, "model variants")
    invalid_variants = [variant for variant in model_variants if not _is_supported_variant(variant)]
    if invalid_variants:
        parser.error(
            "Unsupported --model-variants entries: "
            f"{', '.join(invalid_variants)}. Use one of "
            f"{', '.join(sorted(SUPPORTED_UNTRAINED_VARIANTS))}, "
            "or names that start with 'trained'."
        )

    server = HILServer(config_path=Path(args.config))
    if args.energy_aware:
        server.config.training.energy_aware = True
        server.set_input_mode(server.config.training.get("input_mode", "uniform"))

    needs_trained_checkpoint = any(_is_trained_variant(variant) for variant in model_variants)
    trained_checkpoint = Path(args.trained_checkpoint).resolve() if args.trained_checkpoint else None
    if needs_trained_checkpoint and trained_checkpoint is None:
        parser.error("--trained-checkpoint is required when --model-variants includes a trained variant.")
    if trained_checkpoint is not None and not trained_checkpoint.exists():
        raise FileNotFoundError(f"Trained checkpoint not found: {trained_checkpoint}")

    trained_metadata = {}
    trained_metadata_path = None
    if args.trained_meta:
        trained_metadata_path = Path(args.trained_meta).resolve()
        if not trained_metadata_path.exists():
            raise FileNotFoundError(f"Trained metadata file not found: {trained_metadata_path}")
        with trained_metadata_path.open("r", encoding="utf-8") as meta_file:
            trained_metadata = json.load(meta_file)

    trained_metadata_id = ""
    if trained_metadata_path is not None:
        trained_metadata_id = str(
            trained_metadata.get("artifact_id")
            or trained_metadata.get("artifact_prefix")
            or trained_metadata.get("checkpoint_path")
            or trained_metadata_path.stem
        )

    hyperparams = build_noise_scan_hyperparams(
        window_size=server.config.data.window_size,
        input_dim=server.training_data.inputs.shape[2],
    )
    if trained_metadata:
        meta_window_size = trained_metadata.get("window_size")
        meta_input_dim = trained_metadata.get("input_dim")
        if meta_window_size is not None and int(meta_window_size) != int(hyperparams.timesteps):
            raise ValueError(
                "Trained metadata window_size mismatch: "
                f"metadata={meta_window_size}, expected={hyperparams.timesteps}."
            )
        if meta_input_dim is not None and int(meta_input_dim) != int(hyperparams.input_dim):
            raise ValueError(
                "Trained metadata input_dim mismatch: "
                f"metadata={meta_input_dim}, expected={hyperparams.input_dim}."
            )
        meta_ckpt_path = trained_metadata.get("checkpoint_path")
        if meta_ckpt_path and trained_checkpoint is not None:
            if Path(meta_ckpt_path).name != trained_checkpoint.name:
                raise ValueError(
                    "Trained metadata checkpoint_path basename does not match --trained-checkpoint: "
                    f"metadata={Path(meta_ckpt_path).name}, cli={trained_checkpoint.name}."
                )

    csv_path = Path(args.csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting noise scan: variants=%s, runs=%s per mode, cooldown %.1fs, modes=%s",
        ", ".join(model_variants),
        args.runs,
        args.cooldown,
        ", ".join(input_modes),
    )

    with csv_path.open("w", newline="") as csvfile:
        writer = None
        for model_variant in model_variants:
            checkpoint_path = trained_checkpoint if _is_trained_variant(model_variant) else None
            for mode in input_modes:
                server.set_input_mode(mode)
                desc = f"HIL noise scan ({model_variant}|{mode})"
                for run_idx in tqdm(range(1, args.runs + 1), desc=desc):
                    logger.info(
                        "Noise scan variant=%s mode=%s run %d/%d",
                        model_variant,
                        mode,
                        run_idx,
                        args.runs,
                    )
                    metrics = server.determine_metrics(
                        hyperparams,
                        checkpoint_path=checkpoint_path,
                        model_variant=model_variant,
                    )

                    row = {
                        "model_variant": model_variant,
                        "input_mode": mode,
                        "run_index": run_idx,
                        "trained_checkpoint_path": str(checkpoint_path) if checkpoint_path else "",
                        "trained_metadata_id": trained_metadata_id if checkpoint_path else "",
                        **metrics,
                    }

                    if writer is None:
                        writer = csv.DictWriter(csvfile, fieldnames=list(row.keys()))
                        writer.writeheader()
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
