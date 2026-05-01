#!/usr/bin/env python3
"""
Probe when extra TFLite ops appear across untrained/perturbed model variants.

The script builds the fixed noise-scan architecture and exports both float and
int8 TFLite variants, then compares op histograms against a fresh-untrained
baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema_fb

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from analysis_scripts.hil_noise_analysis.noise_scan_model_spec import build_noise_scan_hyperparams
from analysis_scripts.hil_noise_analysis.train_noise_scan_model import _load_split
from tinyodom.analysis_support import (
    build_model_context,
    resolve_model_family_contract,
    resolve_task_contract,
)
from tinyodom.hardware import convert_to_tflite_model
from tinyodom.model import DEFAULT_CONFIG_PATH, load_config
from tinyodom.pipeline_types import DataSplit, DatasetBundle


ALLOWED_VARIANTS = (
    "fresh_untrained",
    "bn_gamma_beta_perturbed",
    "bn_moving_stats_perturbed",
    "bn_full_perturbed",
    "bn_calibrated_no_train",
    "non_bn_bias_perturbed",
    "bn_full_plus_non_bn_bias_perturbed",
    "trained_checkpoint",
)
ALLOWED_QUANT_MODES = ("float", "int8")


def _to_utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_csv_list(raw: str, field_name: str) -> list[str]:
    values = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError(f"No {field_name} values provided.")
    return values


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


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


def _iter_layers(model: tf.keras.Model) -> list[tf.keras.layers.Layer]:
    """Return all reachable Keras layers from a model graph."""

    seen: set[int] = set()
    queue = [model]
    layers: list[tf.keras.layers.Layer] = []
    while queue:
        node = queue.pop(0)
        node_id = id(node)
        if node_id in seen:
            continue
        seen.add(node_id)
        if isinstance(node, tf.keras.layers.Layer):
            layers.append(node)
        queue.extend(getattr(node, "layers", []) or [])
    return layers


def collect_bn_layers(model: tf.keras.Model) -> list[tf.keras.layers.BatchNormalization]:
    """Collect reachable batch-normalization layers from ``model``."""

    return [
        layer
        for layer in _iter_layers(model)
        if isinstance(layer, tf.keras.layers.BatchNormalization)
    ]


def collect_non_bn_bias_layers(model: tf.keras.Model) -> list[tf.keras.layers.Layer]:
    """Collect reachable non-BN layers that expose a mutable bias tensor."""

    bias_layers: list[tf.keras.layers.Layer] = []
    for layer in _iter_layers(model):
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            continue
        if getattr(layer, "bias", None) is None:
            continue
        bias_layers.append(layer)
    return bias_layers


def _perturb_bn_gamma_beta(model: tf.keras.Model, rng: np.random.Generator) -> int:
    touched = 0
    for layer in collect_bn_layers(model):
        if layer.gamma is not None:
            values = rng.uniform(0.7, 1.3, size=layer.gamma.shape).astype(np.float32)
            layer.gamma.assign(values)
        if layer.beta is not None:
            values = rng.uniform(-0.3, 0.3, size=layer.beta.shape).astype(np.float32)
            layer.beta.assign(values)
        touched += 1
    return touched


def _perturb_bn_moving_stats(model: tf.keras.Model, rng: np.random.Generator) -> int:
    touched = 0
    for layer in collect_bn_layers(model):
        if layer.moving_mean is not None:
            values = rng.uniform(-0.5, 0.5, size=layer.moving_mean.shape).astype(np.float32)
            layer.moving_mean.assign(values)
        if layer.moving_variance is not None:
            values = rng.uniform(0.3, 2.0, size=layer.moving_variance.shape).astype(np.float32)
            layer.moving_variance.assign(values)
        touched += 1
    return touched


def _perturb_non_bn_biases(model: tf.keras.Model, rng: np.random.Generator) -> int:
    touched = 0
    for layer in collect_non_bn_bias_layers(model):
        bias = getattr(layer, "bias", None)
        if bias is None:
            continue
        values = rng.uniform(-0.3, 0.3, size=bias.shape).astype(np.float32)
        bias.assign(values)
        touched += 1
    return touched


def _calibrate_bn_without_training(
    model: tf.keras.Model,
    calibration_inputs: np.ndarray,
    calibration_windows: int,
    calibration_passes: int,
    batch_size: int,
) -> int:
    if calibration_windows <= 0:
        raise ValueError("--bn-calibration-windows must be > 0")
    if calibration_passes <= 0:
        raise ValueError("--bn-calibration-passes must be > 0")
    if batch_size <= 0:
        raise ValueError("--bn-calibration-batch-size must be > 0")

    capped = calibration_inputs[:calibration_windows]
    if capped.size == 0:
        raise ValueError("No calibration inputs available for BN calibration.")

    for _ in range(calibration_passes):
        for start in range(0, len(capped), batch_size):
            batch = capped[start : start + batch_size]
            _ = model(batch, training=True)

    return len(collect_bn_layers(model))


def _hist_delta(base_hist: dict[str, int], cur_hist: dict[str, int]) -> dict[str, int]:
    delta: dict[str, int] = {}
    for key in sorted(set(base_hist) | set(cur_hist)):
        value = int(cur_hist.get(key, 0)) - int(base_hist.get(key, 0))
        if value != 0:
            delta[key] = value
    return delta


def _build_fresh_model(
    hyperparams,
    *,
    model_family: Any,
    model_context: Any,
    model_config: Any,
) -> tf.keras.Model:
    return model_family.build_model(dict(hyperparams), model_context, model_config)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe TFLite op transitions across untrained/BN-perturbed variants."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config YAML.")
    parser.add_argument(
        "--out-dir",
        default="analysis_scripts/hil_noise_analysis/op_transition_probe_output",
        help="Output directory for probe artifacts and CSV reports.",
    )
    parser.add_argument(
        "--variants",
        default=(
            "fresh_untrained,bn_gamma_beta_perturbed,bn_moving_stats_perturbed,"
            "bn_full_perturbed,bn_calibrated_no_train,non_bn_bias_perturbed,"
            "bn_full_plus_non_bn_bias_perturbed"
        ),
        help=(
            "Comma-separated variants to run. Allowed: "
            + ", ".join(ALLOWED_VARIANTS)
        ),
    )
    parser.add_argument(
        "--quant-modes",
        default="float,int8",
        help="Comma-separated quantization export modes. Allowed: float,int8",
    )
    parser.add_argument(
        "--trained-checkpoint",
        default=None,
        help="Optional .keras checkpoint path used by variant 'trained_checkpoint'.",
    )
    parser.add_argument(
        "--calibration-windows-override",
        type=int,
        default=None,
        help="Optional cap on loaded train windows used for conversion/calibration.",
    )
    parser.add_argument(
        "--bn-calibration-windows",
        type=int,
        default=2048,
        help="Number of windows used for bn_calibrated_no_train forward passes.",
    )
    parser.add_argument(
        "--bn-calibration-passes",
        type=int,
        default=1,
        help="How many full passes to run during bn_calibrated_no_train.",
    )
    parser.add_argument(
        "--bn-calibration-batch-size",
        type=int,
        default=256,
        help="Batch size used for bn_calibrated_no_train passes.",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Global random seed.")
    args = parser.parse_args()

    variants = _parse_csv_list(args.variants, "variants")
    invalid_variants = [variant for variant in variants if variant not in ALLOWED_VARIANTS]
    if invalid_variants:
        raise ValueError(
            f"Unsupported variants: {', '.join(invalid_variants)}. "
            f"Allowed: {', '.join(ALLOWED_VARIANTS)}"
        )

    quant_modes = _parse_csv_list(args.quant_modes, "quant modes")
    invalid_quant_modes = [mode for mode in quant_modes if mode not in ALLOWED_QUANT_MODES]
    if invalid_quant_modes:
        raise ValueError(
            f"Unsupported quant modes: {', '.join(invalid_quant_modes)}. "
            f"Allowed: {', '.join(ALLOWED_QUANT_MODES)}"
        )

    if "trained_checkpoint" in variants and not args.trained_checkpoint:
        raise ValueError("--trained-checkpoint is required when 'trained_checkpoint' variant is requested.")

    _set_global_seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))

    cfg_path = Path(args.config).resolve()
    config = load_config(cfg_path)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

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
    del task, task_config
    model_family, model_config = resolve_model_family_contract(config)
    model_context = build_model_context(bundle, target_spec)

    mutators: dict[str, Callable[[tf.keras.Model], tuple[int, str]]] = {
        "fresh_untrained": lambda model: (0, "no mutation"),
        "bn_gamma_beta_perturbed": lambda model: (
            _perturb_bn_gamma_beta(model, rng),
            "perturbed batch norm gamma/beta",
        ),
        "bn_moving_stats_perturbed": lambda model: (
            _perturb_bn_moving_stats(model, rng),
            "perturbed batch norm moving mean/variance",
        ),
        "bn_full_perturbed": lambda model: (
            _perturb_bn_gamma_beta(model, rng) + _perturb_bn_moving_stats(model, rng),
            "perturbed batch norm gamma/beta + moving stats",
        ),
        "bn_calibrated_no_train": lambda model: (
            _calibrate_bn_without_training(
                model=model,
                calibration_inputs=training_data.inputs,
                calibration_windows=int(args.bn_calibration_windows),
                calibration_passes=int(args.bn_calibration_passes),
                batch_size=int(args.bn_calibration_batch_size),
            ),
            "ran forward passes with training=True to update BN stats",
        ),
        "non_bn_bias_perturbed": lambda model: (
            _perturb_non_bn_biases(model, rng),
            "perturbed non-BN layer biases (Conv/Dense/etc.)",
        ),
        "bn_full_plus_non_bn_bias_perturbed": lambda model: (
            _perturb_bn_gamma_beta(model, rng)
            + _perturb_bn_moving_stats(model, rng)
            + _perturb_non_bn_biases(model, rng),
            "perturbed BN params + non-BN layer biases",
        ),
    }

    rows: list[dict[str, object]] = []
    for variant in variants:
        if variant == "trained_checkpoint":
            checkpoint_path = Path(str(args.trained_checkpoint)).resolve()
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            model = model_family.load_model(checkpoint_path, model_context, model_config)
            notes = f"loaded checkpoint {checkpoint_path.name}"
            mutated_layers_touched = 0
        else:
            model = _build_fresh_model(
                hyperparams,
                model_family=model_family,
                model_context=model_context,
                model_config=model_config,
            )
            mutated_layers_touched, notes = mutators[variant](model)

        bn_layers_total = len(collect_bn_layers(model))
        non_bn_bias_layers_total = len(collect_non_bn_bias_layers(model))

        for quant_mode in quant_modes:
            quantization = quant_mode == "int8"
            tflite_path = out_dir / f"{variant}__{quant_mode}.tflite"
            convert_to_tflite_model(
                model=model,
                training_data=training_data.inputs,
                quantization=quantization,
                output_name=tflite_path,
            )
            op_count, add_count, op_hist = _extract_tflite_graph_stats(tflite_path)

            rows.append(
                {
                    "variant": variant,
                    "quant_mode": quant_mode,
                    "timestamp_utc": _to_utc_timestamp(),
                    "loader_name": loader_name,
                    "seed": int(args.seed),
                    "bn_layers_total": int(bn_layers_total),
                    "mutated_layers_touched": int(mutated_layers_touched),
                    "non_bn_bias_layers_total": int(non_bn_bias_layers_total),
                    "notes": notes,
                    "tflite_path": str(tflite_path),
                    "tflite_bytes": int(tflite_path.stat().st_size),
                    "op_count": int(op_count),
                    "add_count": int(add_count),
                    "op_hist": op_hist,
                }
            )

    baseline_by_quant = {
        str(row["quant_mode"]): row for row in rows if str(row["variant"]) == "fresh_untrained"
    }
    for row in rows:
        baseline = baseline_by_quant.get(str(row["quant_mode"]))
        if baseline is None:
            row["delta_tflite_bytes_vs_fresh"] = ""
            row["delta_op_count_vs_fresh"] = ""
            row["delta_add_count_vs_fresh"] = ""
            row["op_hist_delta_vs_fresh"] = {}
            continue

        row["delta_tflite_bytes_vs_fresh"] = int(row["tflite_bytes"]) - int(baseline["tflite_bytes"])
        row["delta_op_count_vs_fresh"] = int(row["op_count"]) - int(baseline["op_count"])
        row["delta_add_count_vs_fresh"] = int(row["add_count"]) - int(baseline["add_count"])
        row["op_hist_delta_vs_fresh"] = _hist_delta(
            baseline["op_hist"],  # type: ignore[arg-type]
            row["op_hist"],  # type: ignore[arg-type]
        )

    csv_path = out_dir / "op_transition_probe_results.csv"
    fieldnames = [
        "variant",
        "quant_mode",
        "timestamp_utc",
        "loader_name",
        "seed",
        "bn_layers_total",
        "mutated_layers_touched",
        "non_bn_bias_layers_total",
        "notes",
        "tflite_path",
        "tflite_bytes",
        "op_count",
        "add_count",
        "delta_tflite_bytes_vs_fresh",
        "delta_op_count_vs_fresh",
        "delta_add_count_vs_fresh",
        "op_hist_json",
        "op_hist_delta_vs_fresh_json",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{key: row.get(key, "") for key in fieldnames},
                    "op_hist_json": json.dumps(row["op_hist"], sort_keys=True),
                    "op_hist_delta_vs_fresh_json": json.dumps(row["op_hist_delta_vs_fresh"], sort_keys=True),
                }
            )

    summary_lines = [
        "op_transition_probe summary",
        f"timestamp_utc: {_to_utc_timestamp()}",
        f"config: {cfg_path}",
        f"out_dir: {out_dir}",
        f"variants: {', '.join(variants)}",
        f"quant_modes: {', '.join(quant_modes)}",
        "",
    ]
    for quant_mode in quant_modes:
        summary_lines.append(f"[{quant_mode}]")
        bucket = [row for row in rows if str(row["quant_mode"]) == quant_mode]
        bucket.sort(key=lambda item: str(item["variant"]))
        for row in bucket:
            summary_lines.append(
                "  "
                + f"{row['variant']}: bytes={row['tflite_bytes']} "
                + f"ops={row['op_count']} add={row['add_count']} "
                + f"delta_ops={row['delta_op_count_vs_fresh']} "
                + f"delta_add={row['delta_add_count_vs_fresh']}"
            )
        summary_lines.append("")

    summary_path = out_dir / "op_transition_probe_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Wrote probe CSV: {csv_path}")
    print(f"Wrote probe summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
