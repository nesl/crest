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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tcn import TCN

sys.path.insert(0, os.path.abspath("src"))

from tinyodom.hardware import convert_to_tflite_model
from tinyodom.model import DEFAULT_CONFIG_PATH, build_tinyodom_model, load_config

from noise_scan_model_spec import build_noise_scan_hyperparams

SUB_FOLDERS = ["handbag/", "handheld/", "pocket/", "running/", "slow_walking/", "trolley/"]
IMU_COL_GYRO = (4, 5, 6)
IMU_COL_GRAV = (7, 8, 9)
IMU_COL_LINACC = (10, 11, 12)
IMU_COL_MAG = (13, 14, 15)
GT_COL_POSE_XY = (2, 3)


@dataclass
class SplitData:
    inputs: np.ndarray
    x_vel: np.ndarray
    y_vel: np.ndarray


def _window_2d(values: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """Create sliding windows from a 2D array without pandas/gtda dependencies."""
    if values.ndim != 2:
        raise ValueError(f"Expected 2D input for windowing, got shape {values.shape}")
    n_rows = values.shape[0]
    if n_rows < window_size:
        return np.empty((0, window_size, values.shape[1]), dtype=np.float32)
    starts = np.arange(0, n_rows - window_size + 1, stride, dtype=np.int64)
    idx = starts[:, None] + np.arange(window_size, dtype=np.int64)[None, :]
    return values[idx]


def _load_split_numpy(config, type_flag: int) -> SplitData:
    """
    Numpy-only OxIOD loader used when tinyodom.data/pandas is unavailable.

    Notes
    -----
    - Mirrors the feature layout expected by TinyODOM: acc(3), gyro(3), mag(3), step(1).
    - Step-counter channel is emitted as zeros in this fallback path so model input
      dimensionality stays compatible with HIL (10 channels).
    """
    split_name_map = {2: "Train.txt", 3: "Valid.txt"}
    if type_flag not in split_name_map:
        raise ValueError(f"Unsupported type_flag for fallback loader: {type_flag}")
    split_file_name = split_name_map[type_flag]

    dataset_root = Path(config.data.directory)
    inputs_list: list[np.ndarray] = []
    vx_list: list[np.ndarray] = []
    vy_list: list[np.ndarray] = []

    for folder in SUB_FOLDERS:
        split_file = dataset_root / folder / split_file_name
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        rel_paths = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]

        for rel_path in rel_paths:
            imu_path = dataset_root / folder / rel_path
            gt_path = Path(str(imu_path).replace("imu", "vi"))
            if not imu_path.exists():
                raise FileNotFoundError(f"IMU file not found: {imu_path}")
            if not gt_path.exists():
                raise FileNotFoundError(f"GT file not found: {gt_path}")

            imu_raw = np.loadtxt(imu_path, delimiter=",", dtype=np.float32)
            gt_raw = np.loadtxt(gt_path, delimiter=",", dtype=np.float32)
            if imu_raw.ndim == 1:
                imu_raw = imu_raw[np.newaxis, :]
            if gt_raw.ndim == 1:
                gt_raw = gt_raw[np.newaxis, :]

            acc = imu_raw[:, IMU_COL_LINACC] + imu_raw[:, IMU_COL_GRAV]
            gyro = imu_raw[:, IMU_COL_GYRO]
            mag = imu_raw[:, IMU_COL_MAG]
            features = np.concatenate((acc, gyro, mag), axis=1)

            # Keep channel count compatible with default useStepCounter=True pipeline.
            step_channel = np.zeros((features.shape[0], 1), dtype=np.float32)
            features = np.concatenate((features, step_channel), axis=1)

            gt_xy = gt_raw[:, GT_COL_POSE_XY]
            feat_windows = _window_2d(features, config.data.window_size, config.data.stride)
            gt_windows = _window_2d(gt_xy, config.data.window_size, config.data.stride)
            if feat_windows.shape[0] == 0 or gt_windows.shape[0] == 0:
                continue

            vx = gt_windows[:, -1, 0] - gt_windows[:, 0, 0]
            vy = gt_windows[:, -1, 1] - gt_windows[:, 0, 1]

            inputs_list.append(feat_windows.astype(np.float32, copy=False))
            vx_list.append(vx.astype(np.float32, copy=False))
            vy_list.append(vy.astype(np.float32, copy=False))

    if not inputs_list:
        raise RuntimeError("No windows were loaded by numpy fallback data loader.")

    return SplitData(
        inputs=np.concatenate(inputs_list, axis=0),
        x_vel=np.concatenate(vx_list, axis=0),
        y_vel=np.concatenate(vy_list, axis=0),
    )


def _load_split(config, type_flag: int) -> tuple[SplitData, str]:
    """
    Load a split, preferring tinyodom.data and falling back to numpy-only parsing.
    """
    try:
        from tinyodom.data import import_oxiod_dataset  # local import to avoid hard pandas dependency

        split = import_oxiod_dataset(
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
        return SplitData(inputs=split.inputs, x_vel=split.x_vel, y_vel=split.y_vel), "tinyodom.data"
    except (ImportError, ModuleNotFoundError, OSError) as exc:
        # Expected dependency- or environment-related failures: fall back to numpy-only loader.
        print(
            "[WARN] tinyodom.data.import_oxiod_dataset unavailable or dependency-related error; "
            f"falling back to numpy loader (no pandas required): {exc}"
        )
        return _load_split_numpy(config=config, type_flag=type_flag), "numpy_fallback_no_pandas"
    except Exception:
        # Re-raise unexpected exceptions to avoid masking logic or schema bugs.
        raise


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
    training_data, train_loader_name = _load_split(config, type_flag=2)
    validation_data, valid_loader_name = _load_split(config, type_flag=3)
    print(f"Loaded training data via: {train_loader_name}")
    print(f"Loaded validation data via: {valid_loader_name}")

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
        "data_loader_train": train_loader_name,
        "data_loader_valid": valid_loader_name,
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
