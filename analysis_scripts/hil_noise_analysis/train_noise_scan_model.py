#!/usr/bin/env python3
"""
Train the fixed HIL noise-scan model and export handoff artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tcn import TCN

sys.path.insert(0, os.path.abspath("src"))

from tinyodom.data import import_oxiod_dataset
from tinyodom.hardware import convert_to_tflite_model
from tinyodom.model import DEFAULT_CONFIG_PATH, build_tinyodom_model, load_config

from noise_scan_model_spec import build_noise_scan_hyperparams

SUB_FOLDERS = ["handbag/", "handheld/", "pocket/", "running/", "slow_walking/", "trolley/"]


def _load_split(config, type_flag: int):
    return import_oxiod_dataset(
        type_flag=type_flag,
        useMagnetometer=True,
        useStepCounter=True,
        AugmentationCopies=0,
        dataset_folder=config.data.directory,
        sub_folders=SUB_FOLDERS,
        sampling_rate=config.data.sampling_rate_hz,
        window_size=config.data.window_size,
        stride=config.data.stride,
        verbose=False,
    )


def _git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train the fixed noise-scan model and export .keras + metadata artifacts."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config YAML.")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum training epochs.")
    parser.add_argument(
        "--out-dir",
        default="analysis_scripts/hil_noise_analysis/artifacts",
        help="Directory for checkpoint/metadata artifacts.",
    )
    parser.add_argument(
        "--artifact-prefix",
        default="noise_scan_50ep",
        help="Prefix used for output artifact files.",
    )
    parser.add_argument(
        "--export-tflite",
        action="store_true",
        help="Also export a TFLite copy from the best checkpoint.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    config = load_config(cfg_path)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = out_dir / f"{args.artifact_prefix}.keras"
    metadata_path = out_dir / f"{args.artifact_prefix}.json"
    tflite_path = out_dir / f"{args.artifact_prefix}.tflite"

    print("Loading train/validation splits...")
    training_data = _load_split(config, type_flag=2)
    validation_data = _load_split(config, type_flag=3)
    print("Loaded training/validation data.")

    hyperparams = build_noise_scan_hyperparams(
        window_size=config.data.window_size,
        input_dim=training_data.inputs.shape[2],
    )
    model = build_tinyodom_model(hyperparams)
    model.compile(loss={"velx": "mse", "vely": "mse"}, optimizer=optimizers.Adam())

    checkpoint_cb = ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor="val_loss",
        mode="min",
        verbose=1,
        save_best_only=True,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        patience=40,
        mode="min",
        verbose=1,
        restore_best_weights=True,
    )

    print(f"Training fixed noise-scan model for up to {args.epochs} epochs...")
    history = model.fit(
        x=training_data.inputs,
        y=[training_data.x_vel, training_data.y_vel],
        epochs=args.epochs,
        shuffle=True,
        callbacks=[checkpoint_cb, early_stop_cb],
        batch_size=hyperparams.batch_size,
        validation_data=(validation_data.inputs, [validation_data.x_vel, validation_data.y_vel]),
    )
    epochs_ran = len(history.history.get("loss", []))

    tflite_written = None
    if args.export_tflite:
        best_model = load_model(str(checkpoint_path), custom_objects={"TCN": TCN})
        convert_to_tflite_model(
            model=best_model,
            training_data=training_data.inputs,
            quantization=config.training.quantization,
            output_name=tflite_path,
        )
        tflite_written = str(tflite_path)
        print(f"Exported TFLite artifact: {tflite_path}")

    config_sha256 = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
    timestamp_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    repo_root = Path(__file__).resolve().parents[2]
    metadata = {
        "artifact_id": f"{args.artifact_prefix}_{timestamp_utc}",
        "artifact_prefix": args.artifact_prefix,
        "timestamp_utc": timestamp_utc,
        "git_commit": _git_commit(repo_root),
        "config_path": str(cfg_path),
        "config_sha256": config_sha256,
        "checkpoint_path": str(checkpoint_path),
        "tflite_path": tflite_written,
        "quantization": bool(config.training.quantization),
        "window_size": int(config.data.window_size),
        "input_dim": int(training_data.inputs.shape[2]),
        "hyperparams": {
            "nb_filters": int(hyperparams.nb_filters),
            "kernel_size": int(hyperparams.kernel_size),
            "dilations": [int(v) for v in hyperparams.dilations],
            "dropout_rate": float(hyperparams.dropout_rate),
            "use_skip_connections": bool(hyperparams.use_skip_connections),
            "norm_flag": bool(hyperparams.norm_flag),
            "batch_size": int(hyperparams.batch_size),
            "timesteps": int(hyperparams.timesteps),
            "input_dim": int(hyperparams.input_dim),
            "flops": int(hyperparams.flops),
        },
        "epochs_requested": int(args.epochs),
        "epochs_ran": int(epochs_ran),
        "early_stopped": bool(epochs_ran < args.epochs),
        "train_windows": int(training_data.inputs.shape[0]),
        "valid_windows": int(validation_data.inputs.shape[0]),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

