#!/usr/bin/env python3
"""
Audit a fresh untrained model's quantized TFLite op histogram.

This script builds the fixed noise-scan architecture from current config,
exports a quantized TFLite model via the same conversion path used in HIL,
and prints:
- tflite_quant_op_hist_json
- tflite_quant_add_count
- tflite_quant_op_count
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

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


def _extract_tflite_graph_stats(tflite_path: Path) -> tuple[int, int, dict[str, int]]:
    model_bytes = tflite_path.read_bytes()
    model = schema_fb.Model.GetRootAsModel(model_bytes, 0)
    if model.SubgraphsLength() == 0:
        raise ValueError(f"TFLite model has no subgraphs: {tflite_path}")

    subgraph = model.Subgraphs(0)
    op_codes = {idx: model.OperatorCodes(idx).BuiltinCode() for idx in range(model.OperatorCodesLength())}
    op_name_map = {
        value: key for key, value in schema_fb.BuiltinOperator.__dict__.items() if isinstance(value, int)
    }
    op_histogram: dict[str, int] = {}
    for op_idx in range(subgraph.OperatorsLength()):
        operator = subgraph.Operators(op_idx)
        code = op_codes[operator.OpcodeIndex()]
        name = op_name_map.get(code, f"UNKNOWN_{code}")
        op_histogram[name] = op_histogram.get(name, 0) + 1

    op_count = int(subgraph.OperatorsLength())
    add_count = int(op_histogram.get("ADD", 0))
    return op_count, add_count, dict(sorted(op_histogram.items()))


def _append_audit_row(csv_path: Path, audit_row: dict[str, object]) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"--append-csv target does not exist: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
    if not fieldnames:
        raise ValueError(f"--append-csv has no header row: {csv_path}")

    full_row = {column: "" for column in fieldnames}
    for key, value in audit_row.items():
        if key in full_row:
            full_row[key] = value

    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writerow(full_row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit fresh untrained model TFLite op histogram.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config YAML.")
    parser.add_argument(
        "--output-tflite",
        default="analysis_scripts/hil_noise_analysis/epoch_sweep/artifacts/fresh_untrained_audit.tflite",
        help="Output path for the generated TFLite model.",
    )
    parser.add_argument(
        "--calibration-windows-override",
        type=int,
        default=None,
        help="Optional cap on train windows used for quantization representative samples.",
    )
    parser.add_argument(
        "--append-csv",
        default="analysis_scripts/hil_noise_analysis/epoch_sweep/artifacts/epoch_sweep_training_stats.csv",
        help="CSV path to append a fresh-untrained audit row to.",
    )
    parser.add_argument(
        "--skip-csv-append",
        action="store_true",
        help="If set, do not append a row to --append-csv.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    config = load_config(cfg_path)
    config_sha256 = hashlib.sha256(cfg_path.read_bytes()).hexdigest()

    training_data, loader_name = _load_split(config, type_flag=2)
    if args.calibration_windows_override is not None:
        limit = int(args.calibration_windows_override)
        if limit <= 0:
            raise ValueError("--calibration-windows-override must be > 0")
        training_data.inputs = training_data.inputs[:limit]

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
        val=None,
        test=None,
        input_shape=(int(hyperparams.timesteps), int(hyperparams.input_dim)),
        input_dtype="float32",
        metadata=dict(config.dataset.params),
    )
    task, task_config, target_spec = resolve_task_contract(config, bundle)
    model_family, model_config = resolve_model_family_contract(config)
    model = model_family.build_model(
        dict(hyperparams),
        build_model_context(bundle, target_spec),
        model_config,
    )
    task.compile_model(model, task_config, target_spec)

    output_tflite = Path(args.output_tflite).resolve()
    output_tflite.parent.mkdir(parents=True, exist_ok=True)
    convert_to_tflite_model(
        model=model,
        training_data=training_data.inputs,
        quantization=bool(config.training.quantization),
        output_name=output_tflite,
    )

    op_count, add_count, op_hist = _extract_tflite_graph_stats(output_tflite)
    timestamp_utc = _to_utc_timestamp()

    print(f"loader={loader_name}")
    print(f"quantization={bool(config.training.quantization)}")
    print(f"tflite_path={output_tflite}")
    print(f"tflite_quant_op_count={op_count}")
    print(f"tflite_quant_add_count={add_count}")
    print(f"tflite_quant_op_hist_json={json.dumps(op_hist, sort_keys=True)}")
    if not args.skip_csv_append:
        append_csv_path = Path(args.append_csv).resolve()
        audit_row = {
            "stage_type": "fresh_untrained_audit",
            "epoch": -1,
            "checkpoint_path": "",
            "metadata_json_path": "",
            "plot_path": "",
            "best_val_loss_so_far": "",
            "best_epoch_so_far": "",
            "early_stopped": False,
            "global_wait_counter": 0,
            "quantization_enabled": bool(config.training.quantization),
            "tflite_quant_path": str(output_tflite),
            "tflite_quant_bytes": int(output_tflite.stat().st_size),
            "tflite_quant_op_count": op_count,
            "tflite_quant_add_count": add_count,
            "tflite_quant_op_hist_json": json.dumps(op_hist, sort_keys=True),
            "timestamp_utc": timestamp_utc,
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
        _append_audit_row(append_csv_path, audit_row)
        print(f"appended_row_to_csv={append_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
