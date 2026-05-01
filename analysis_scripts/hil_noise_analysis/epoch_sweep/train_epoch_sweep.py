#!/usr/bin/env python3
"""
Train a fixed noise-scan model in staged epochs and export per-stage artifacts.

This script trains continuously with a global early-stopping policy, then saves
stage-end checkpoints every N epochs (default 50). For each saved checkpoint it
exports a quantized TFLite model, extracts graph stats, writes a loss plot, and
appends a row to a training stats CSV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import Callback
from tensorflow.lite.python import schema_py_generated as schema_fb

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "analysis_scripts" / "hil_noise_analysis"))

from analysis_scripts.hil_noise_analysis.noise_scan_model_spec import build_noise_scan_hyperparams
from analysis_scripts.hil_noise_analysis.train_noise_scan_model import _git_commit, _load_split
from tinyodom.analysis_support import (
    build_model_context,
    resolve_model_family_contract,
    resolve_task_contract,
)
from tinyodom.hardware import convert_to_tflite_model
from tinyodom.model import DEFAULT_CONFIG_PATH, load_config
from tinyodom.pipeline_types import DataSplit, DatasetBundle


def _to_utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _builtin_operator_name_map() -> dict[int, str]:
    return {
        value: key
        for key, value in schema_fb.BuiltinOperator.__dict__.items()
        if isinstance(value, int)
    }


def _extract_tflite_graph_stats(tflite_path: Path) -> tuple[int, int, dict[str, int]]:
    """Return (op_count, add_count, op_histogram) from a TFLite flatbuffer."""
    model_bytes = tflite_path.read_bytes()
    model = schema_fb.Model.GetRootAsModel(model_bytes, 0)
    if model.SubgraphsLength() == 0:
        raise ValueError(f"TFLite model has no subgraphs: {tflite_path}")

    subgraph = model.Subgraphs(0)
    op_codes = {idx: model.OperatorCodes(idx).BuiltinCode() for idx in range(model.OperatorCodesLength())}
    op_name_map = _builtin_operator_name_map()
    op_histogram: dict[str, int] = {}

    for op_idx in range(subgraph.OperatorsLength()):
        operator = subgraph.Operators(op_idx)
        builtin_code = op_codes[operator.OpcodeIndex()]
        op_name = op_name_map.get(builtin_code, f"UNKNOWN_{builtin_code}")
        op_histogram[op_name] = op_histogram.get(op_name, 0) + 1

    op_count = int(subgraph.OperatorsLength())
    add_count = int(op_histogram.get("ADD", 0))
    return op_count, add_count, dict(sorted(op_histogram.items()))


def _save_loss_plot(train_loss: list[float], val_loss: list[float], plot_path: Path, title: str) -> None:
    """Persist a train/validation loss curve PNG."""
    epochs = list(range(1, len(train_loss) + 1))
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_loss, label="loss")
    if val_loss:
        plt.plot(epochs[: len(val_loss)], val_loss, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()


class EpochSweepCallback(Callback):
    """Save stage artifacts while applying global early stopping."""

    def __init__(
        self,
        *,
        stage_size: int,
        patience: int,
        min_delta: float,
        artifact_prefix: str,
        out_dir: Path,
        plots_dir: Path,
        csv_path: Path,
        training_inputs: np.ndarray,
        quantization_enabled: bool,
        static_metadata: dict[str, str | int | float | bool],
    ) -> None:
        super().__init__()
        self.stage_size = stage_size
        self.patience = patience
        self.min_delta = min_delta
        self.artifact_prefix = artifact_prefix
        self.out_dir = out_dir
        self.plots_dir = plots_dir
        self.csv_path = csv_path
        self.training_inputs = training_inputs
        self.quantization_enabled = quantization_enabled
        self.static_metadata = static_metadata

        self.best_val_loss_so_far = float("inf")
        self.best_epoch_so_far = 0
        self.global_wait_counter = 0
        self.early_stopped = False

        self._train_loss_history: list[float] = []
        self._val_loss_history: list[float] = []
        self._csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
        self._writer = None
        self._saved_artifacts: list[dict[str, object]] = []
        self.manifest_path = self.out_dir / f"{self.artifact_prefix}_training_manifest.json"

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = logs or {}
        current_epoch = epoch + 1

        train_loss = float(logs.get("loss", math.nan))
        self._train_loss_history.append(train_loss)

        val_loss_raw = logs.get("val_loss")
        val_loss = float(val_loss_raw) if val_loss_raw is not None else math.nan
        self._val_loss_history.append(val_loss)

        is_improved = False
        if math.isfinite(val_loss):
            if val_loss < (self.best_val_loss_so_far - self.min_delta):
                self.best_val_loss_so_far = val_loss
                self.best_epoch_so_far = current_epoch
                self.global_wait_counter = 0
                is_improved = True
            else:
                self.global_wait_counter += 1
        else:
            self.global_wait_counter += 1

        is_milestone = (current_epoch % self.stage_size) == 0
        should_stop = (self.patience >= 0) and (self.global_wait_counter >= self.patience) and (not is_improved)

        if is_milestone:
            self._save_stage_artifact(stage_type="milestone", epoch=current_epoch, early_stopped=should_stop)

        if should_stop:
            self.early_stopped = True
            if not is_milestone:
                self._save_stage_artifact(stage_type="final_early_stop", epoch=current_epoch, early_stopped=True)
            self.model.stop_training = True

    def on_train_end(self, logs: dict | None = None) -> None:
        if self._csv_file and not self._csv_file.closed:
            self._csv_file.close()
        manifest_payload = {
            "artifact_prefix": self.artifact_prefix,
            "timestamp_utc": _to_utc_timestamp(),
            "best_val_loss_so_far": self.best_val_loss_so_far,
            "best_epoch_so_far": self.best_epoch_so_far,
            "early_stopped": self.early_stopped,
            "artifacts": self._saved_artifacts,
        }
        self.manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

    def _save_stage_artifact(self, *, stage_type: str, epoch: int, early_stopped: bool) -> None:
        assert self.model is not None

        if stage_type == "final_early_stop":
            checkpoint_path = self.out_dir / f"{self.artifact_prefix}_epoch_{epoch}_final.keras"
            plot_path = self.plots_dir / f"{self.artifact_prefix}_loss_epoch_{epoch}_final.png"
            tflite_path = self.out_dir / f"{self.artifact_prefix}_epoch_{epoch}_final.tflite"
            metadata_json_path = self.out_dir / f"{self.artifact_prefix}_epoch_{epoch}_final.json"
        else:
            checkpoint_path = self.out_dir / f"{self.artifact_prefix}_epoch_{epoch}.keras"
            plot_path = self.plots_dir / f"{self.artifact_prefix}_loss_epoch_{epoch}.png"
            tflite_path = self.out_dir / f"{self.artifact_prefix}_epoch_{epoch}.tflite"
            metadata_json_path = self.out_dir / f"{self.artifact_prefix}_epoch_{epoch}.json"

        self.model.save(checkpoint_path)
        _save_loss_plot(
            self._train_loss_history,
            self._val_loss_history,
            plot_path,
            title=f"Epoch Sweep Loss Through Epoch {epoch}",
        )
        convert_to_tflite_model(
            model=self.model,
            training_data=self.training_inputs,
            quantization=self.quantization_enabled,
            output_name=tflite_path,
        )
        tflite_quant_op_count, tflite_quant_add_count, tflite_quant_op_hist = _extract_tflite_graph_stats(
            tflite_path
        )

        row = {
            "stage_type": stage_type,
            "epoch": epoch,
            "checkpoint_path": str(checkpoint_path),
            "metadata_json_path": str(metadata_json_path),
            "plot_path": str(plot_path),
            "best_val_loss_so_far": self.best_val_loss_so_far,
            "best_epoch_so_far": self.best_epoch_so_far,
            "early_stopped": bool(early_stopped),
            "global_wait_counter": self.global_wait_counter,
            "quantization_enabled": bool(self.quantization_enabled),
            "tflite_quant_path": str(tflite_path),
            "tflite_quant_bytes": int(tflite_path.stat().st_size),
            "tflite_quant_op_count": int(tflite_quant_op_count),
            "tflite_quant_add_count": int(tflite_quant_add_count),
            "tflite_quant_op_hist_json": json.dumps(tflite_quant_op_hist, sort_keys=True),
            "timestamp_utc": _to_utc_timestamp(),
            **self.static_metadata,
        }
        metadata_payload = {
            "artifact_id": f"{self.artifact_prefix}_epoch_{epoch}_{row['timestamp_utc']}",
            "artifact_prefix": self.artifact_prefix,
            "stage_type": stage_type,
            "epoch": epoch,
            "timestamp_utc": row["timestamp_utc"],
            "checkpoint_path": str(checkpoint_path),
            "tflite_path": str(tflite_path),
            "plot_path": str(plot_path),
            "quantization": bool(self.quantization_enabled),
            "best_val_loss_so_far": self.best_val_loss_so_far,
            "best_epoch_so_far": self.best_epoch_so_far,
            "early_stopped": bool(early_stopped),
            "global_wait_counter": self.global_wait_counter,
            "tflite_graph": {
                "bytes": int(tflite_path.stat().st_size),
                "op_count": int(tflite_quant_op_count),
                "add_count": int(tflite_quant_add_count),
                "op_histogram": tflite_quant_op_hist,
            },
            "hyperparams": {
                "window_size": int(self.static_metadata["window_size"]),
                "input_dim": int(self.static_metadata["input_dim"]),
                "nb_filters": int(self.static_metadata["nb_filters"]),
                "kernel_size": int(self.static_metadata["kernel_size"]),
                "dilations": json.loads(str(self.static_metadata["dilations_json"])),
                "dropout_rate": float(self.static_metadata["dropout_rate"]),
                "use_skip_connections": bool(self.static_metadata["use_skip_connections"]),
                "norm_flag": bool(self.static_metadata["norm_flag"]),
            },
            "config_path": str(self.static_metadata["config_path"]),
            "config_sha256": str(self.static_metadata["config_sha256"]),
            "git_commit": str(self.static_metadata["git_commit"]),
        }
        metadata_json_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
        self._saved_artifacts.append(
            {
                "epoch": epoch,
                "stage_type": stage_type,
                "checkpoint_path": str(checkpoint_path),
                "metadata_json_path": str(metadata_json_path),
                "tflite_path": str(tflite_path),
                "plot_path": str(plot_path),
            }
        )

        if self._writer is None:
            self._writer = csv.DictWriter(self._csv_file, fieldnames=list(row.keys()))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._csv_file.flush()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train fixed hyperparameters in staged epochs and export per-stage artifacts."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config YAML.")
    parser.add_argument(
        "--out-dir",
        default="analysis_scripts/hil_noise_analysis/epoch_sweep/artifacts",
        help="Output directory for checkpoints/TFLite artifacts.",
    )
    parser.add_argument(
        "--artifact-prefix",
        default="noise_scan_epoch_sweep",
        help="Prefix for emitted artifact files.",
    )
    parser.add_argument("--max-epochs", type=int, default=500, help="Maximum epochs to train.")
    parser.add_argument("--stage-size", type=int, default=50, help="Epoch interval for stage artifacts.")
    parser.add_argument("--patience", type=int, default=40, help="Global early-stopping patience.")
    parser.add_argument("--min-delta", type=float, default=0.0, help="Minimum val_loss improvement threshold.")
    parser.add_argument(
        "--csv-path",
        default=None,
        help="Optional output CSV path. Defaults to <out-dir>/epoch_sweep_training_stats.csv",
    )
    parser.add_argument(
        "--plots-dir",
        default=None,
        help="Optional output plot directory. Defaults to <out-dir>/plots",
    )
    parser.add_argument(
        "--verbose-fit",
        action="store_true",
        help="Enable verbose per-epoch Keras fit logs.",
    )
    parser.add_argument(
        "--calibration-windows-override",
        type=int,
        default=None,
        help="Optional cap on train/valid windows for smoke testing.",
    )
    args = parser.parse_args()

    if args.max_epochs <= 0:
        raise ValueError("--max-epochs must be > 0")
    if args.stage_size <= 0:
        raise ValueError("--stage-size must be > 0")
    if args.patience < 0:
        raise ValueError("--patience must be >= 0")
    if args.min_delta < 0:
        raise ValueError("--min-delta must be >= 0")

    cfg_path = Path(args.config).resolve()
    config = load_config(cfg_path)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = Path(args.plots_dir).resolve() if args.plots_dir else out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv_path).resolve() if args.csv_path else out_dir / "epoch_sweep_training_stats.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading train/validation splits...")
    training_data, train_loader_name = _load_split(config, type_flag=2)
    validation_data, valid_loader_name = _load_split(config, type_flag=3)
    if args.calibration_windows_override is not None:
        limit = int(args.calibration_windows_override)
        if limit <= 0:
            raise ValueError("--calibration-windows-override must be > 0")
        training_data.inputs = training_data.inputs[:limit]
        training_data.x_vel = training_data.x_vel[:limit]
        training_data.y_vel = training_data.y_vel[:limit]
        validation_data.inputs = validation_data.inputs[:limit]
        validation_data.x_vel = validation_data.x_vel[:limit]
        validation_data.y_vel = validation_data.y_vel[:limit]
        print(f"Applied calibration window override: {limit}")

    print(f"Loaded training data via: {train_loader_name}")
    print(f"Loaded validation data via: {valid_loader_name}")

    hyperparams = build_noise_scan_hyperparams(
        window_size=config.dataset.params.window_size,
        input_dim=training_data.inputs.shape[2],
    )
    bundle = DatasetBundle(
        train=DataSplit(
            inputs=training_data.inputs,
            targets={"velx": training_data.x_vel, "vely": training_data.y_vel},
            metadata={},
        ),
        val=DataSplit(
            inputs=validation_data.inputs,
            targets={"velx": validation_data.x_vel, "vely": validation_data.y_vel},
            metadata={},
        ),
        test=None,
        input_shape=(int(hyperparams.timesteps), int(hyperparams.input_dim)),
        input_dtype="float32",
        metadata=dict(config.dataset.params),
    )
    checkpoint_path = out_dir / f"{args.artifact_prefix}.keras"
    task, task_config, target_spec = resolve_task_contract(
        config,
        bundle,
        checkpoint_path=checkpoint_path,
        early_stopping_patience=int(args.patience),
    )
    model_family, model_config = resolve_model_family_contract(config)
    model = model_family.build_model(
        dict(hyperparams),
        build_model_context(bundle, target_spec),
        model_config,
    )
    task.compile_model(model, task_config, target_spec)

    config_sha256 = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
    static_metadata = {
        "config_path": str(cfg_path),
        "config_sha256": config_sha256,
        "git_commit": _git_commit(REPO_ROOT) or "",
        "window_size": int(config.dataset.params.window_size),
        "input_dim": int(training_data.inputs.shape[2]),
        "nb_filters": int(hyperparams.nb_filters),
        "kernel_size": int(hyperparams.kernel_size),
        "dilations_json": json.dumps([int(v) for v in hyperparams.dilations]),
        "dropout_rate": float(hyperparams.dropout_rate),
        "use_skip_connections": bool(hyperparams.use_skip_connections),
        "norm_flag": bool(hyperparams.norm_flag),
    }

    callback = EpochSweepCallback(
        stage_size=int(args.stage_size),
        patience=int(args.patience),
        min_delta=float(args.min_delta),
        artifact_prefix=str(args.artifact_prefix),
        out_dir=out_dir,
        plots_dir=plots_dir,
        csv_path=csv_path,
        training_inputs=training_data.inputs,
        quantization_enabled=bool(config.training.quantization),
        static_metadata=static_metadata,
    )

    print(
        f"Starting epoch sweep: max_epochs={args.max_epochs} "
        f"stage_size={args.stage_size} patience={args.patience} min_delta={args.min_delta:.4f}"
    )
    model.fit(
        x=training_data.inputs,
        y=[training_data.x_vel, training_data.y_vel],
        epochs=int(args.max_epochs),
        batch_size=int(hyperparams.batch_size),
        shuffle=True,
        validation_data=(validation_data.inputs, [validation_data.x_vel, validation_data.y_vel]),
        callbacks=[callback],
        verbose=1 if args.verbose_fit else 0,
    )

    print(f"Epoch sweep complete. Training stats CSV: {csv_path}")
    print(f"Epoch sweep manifest JSON: {callback.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
