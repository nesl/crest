"""Compare odometry Keras and exported TFLite accuracy on the same split."""

from __future__ import annotations

import argparse
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import mean_squared_error


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the comparison diagnostic.

    Returns
    -------
    argparse.Namespace
        Parsed command-line options.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("models/OxIOD_FLOPS_PROXY_case1_1/nas_config_flops_rmse.yaml"),
        help="Run configuration to load dataset/model paths from.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="val",
        help="Odometry dataset split to evaluate.",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=4096,
        help="Maximum number of windows to evaluate. Use 0 for the full split.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="TFLite invocation batch size for the diagnostic.",
    )
    parser.add_argument(
        "--keep-tflite-dir",
        type=Path,
        default=None,
        help="Optional directory where exported diagnostic TFLite files are kept.",
    )
    return parser.parse_args()


def _require_odometry_targets(targets: Any) -> Mapping[str, Any]:
    """Validate that a split uses the odometry target contract.

    Parameters
    ----------
    targets : Any
        Split target payload from the bootstrapped dataset bundle.

    Returns
    -------
    collections.abc.Mapping[str, Any]
        Mapping containing ``velx`` and ``vely`` target arrays.

    Raises
    ------
    ValueError
        If the selected split does not expose odometry velocity targets.
    """

    if not isinstance(targets, Mapping) or "velx" not in targets or "vely" not in targets:
        raise ValueError(
            "compare_keras_tflite_accuracy.py currently supports odometry "
            "splits with mapping targets named 'velx' and 'vely'."
        )
    return targets


def _normalize_keras_prediction_outputs(predictions: Any) -> list[np.ndarray]:
    """Normalize Keras odometry predictions to a two-output list.

    Parameters
    ----------
    predictions : Any
        Raw prediction payload returned by ``model.predict(...)``.

    Returns
    -------
    list[numpy.ndarray]
        Two prediction arrays in ``[velx, vely]`` order.

    Raises
    ------
    ValueError
        If the model does not produce exactly two output heads.
    """

    if not isinstance(predictions, (list, tuple)) or len(predictions) != 2:
        raise ValueError(
            "compare_keras_tflite_accuracy.py currently supports odometry "
            "models with exactly two Keras outputs: [velx, vely]."
        )
    return [np.asarray(output) for output in predictions]


def _rmse_pair(predictions: list[np.ndarray], targets: Mapping[str, Any]) -> tuple[float, float, float]:
    """Compute odometry RMSE for one prediction pair.

    Parameters
    ----------
    predictions : list[numpy.ndarray]
        Prediction arrays in ``[velx, vely]`` order.
    targets : collections.abc.Mapping[str, Any]
        Split targets containing ``velx`` and ``vely`` arrays.

    Returns
    -------
    tuple[float, float, float]
        ``(rmse_vel_x, rmse_vel_y, rmse_total)``.
    """

    rmse_x = mean_squared_error(targets["velx"], predictions[0], squared=False)
    rmse_y = mean_squared_error(targets["vely"], predictions[1], squared=False)
    return float(rmse_x), float(rmse_y), float(rmse_x + rmse_y)


def _array_stats(values: np.ndarray) -> dict[str, float | int]:
    """Return compact range and saturation diagnostics for one output.

    Parameters
    ----------
    values : numpy.ndarray
        Prediction array to summarize.

    Returns
    -------
    dict[str, float | int]
        Range, mean/std, unique count, and endpoint saturation fractions.
    """

    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    min_value = float(np.min(flat))
    max_value = float(np.max(flat))
    return {
        "min": min_value,
        "max": max_value,
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "unique": int(np.unique(flat).size),
        "frac_min": float(np.mean(flat == min_value)),
        "frac_max": float(np.mean(flat == max_value)),
    }


def _batched_tflite_predict(
    tflite_path: Path,
    inputs: np.ndarray,
    batch_size: int,
) -> list[np.ndarray]:
    """Run TFLite inference in batches while preserving signature output order.

    Parameters
    ----------
    tflite_path : pathlib.Path
        TFLite flatbuffer to evaluate.
    inputs : numpy.ndarray
        Float32 split inputs with a leading sample dimension.
    batch_size : int
        Number of windows per interpreter invocation.

    Returns
    -------
    list[numpy.ndarray]
        Dequantized prediction arrays in Keras output order.
    """

    import tensorflow as tf

    from tinyodom.hardware import (
        _dequantize_tflite_tensor,
        _ordered_tflite_output_details,
        _quantize_tflite_tensor,
    )

    data = np.asarray(inputs, dtype=np.float32)
    interpreter = tf.lite.Interpreter(
        model_path=str(tflite_path),
        experimental_delegates=[],
        num_threads=1,
    )
    input_detail = interpreter.get_input_details()[0]
    outputs: list[list[np.ndarray]] | None = None
    for start in range(0, len(data), batch_size):
        batch = data[start : start + batch_size]
        interpreter.resize_tensor_input(input_detail["index"], [len(batch), *data.shape[1:]])
        interpreter.allocate_tensors()
        input_detail = interpreter.get_input_details()[0]
        output_details = _ordered_tflite_output_details(interpreter)
        if outputs is None:
            outputs = [[] for _ in output_details]
        interpreter.set_tensor(input_detail["index"], _quantize_tflite_tensor(batch, input_detail))
        interpreter.invoke()
        for output_index, output_detail in enumerate(output_details):
            raw_output = interpreter.get_tensor(output_detail["index"])
            outputs[output_index].append(_dequantize_tflite_tensor(raw_output, output_detail))
    if outputs is None:
        raise ValueError("No inputs were provided for TFLite prediction.")
    return [np.concatenate(parts, axis=0) for parts in outputs]


def _print_model_report(
    label: str,
    predictions: list[np.ndarray],
    targets: Mapping[str, Any],
    keras_predictions: list[np.ndarray],
) -> None:
    """Print metrics and ranges for one backend.

    Parameters
    ----------
    label : str
        Backend label.
    predictions : list[numpy.ndarray]
        Backend prediction arrays.
    targets : collections.abc.Mapping[str, Any]
        Ground-truth split targets.
    keras_predictions : list[numpy.ndarray]
        Keras prediction arrays used for backend-vs-Keras deltas.
    """

    rmse = _rmse_pair(predictions, targets)
    print(f"{label}_rmse_vel_x={rmse[0]:.6f}")
    print(f"{label}_rmse_vel_y={rmse[1]:.6f}")
    print(f"{label}_rmse_total={rmse[2]:.6f}")
    for axis, prediction, keras_prediction in zip(("vel_x", "vel_y"), predictions, keras_predictions):
        delta = np.asarray(prediction, dtype=np.float32) - np.asarray(keras_prediction, dtype=np.float32)
        print(f"{label}_{axis}_stats={_array_stats(prediction)}")
        print(f"{label}_{axis}_vs_keras_mae={float(np.mean(np.abs(delta))):.6f}")
        print(f"{label}_{axis}_vs_keras_max_abs={float(np.max(np.abs(delta))):.6f}")


def main() -> None:
    """Run the Keras/TFLite comparison diagnostic.

    Returns
    -------
    None
    """

    args = _parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

    import tensorflow as tf

    from tinyodom.builtin_components import ensure_builtin_components_registered
    from tinyodom.hardware import convert_to_tflite_model
    from tinyodom.model import load_config
    from tinyodom.runtime_bootstrap import bootstrap_pipeline

    ensure_builtin_components_registered()
    config = load_config(args.config)
    pipeline = bootstrap_pipeline(config)
    split = getattr(pipeline.bundle, args.split)
    split_targets = _require_odometry_targets(split.targets)
    max_windows = None if args.max_windows == 0 else int(args.max_windows)
    inputs = np.asarray(split.inputs[:max_windows], dtype=np.float32)
    targets = {
        "velx": np.asarray(split_targets["velx"][:max_windows]),
        "vely": np.asarray(split_targets["vely"][:max_windows]),
    }
    calibration = pipeline.bundle.calibration or pipeline.bundle.train
    model_path = Path(config.outputs.checkpoint_path)
    model = pipeline.model_family.load_model(
        model_path,
        pipeline.model_build_context,
        pipeline.selection["model_config"],
    )
    keras_predictions = _normalize_keras_prediction_outputs(model.predict(inputs, verbose=0))

    print(f"config={args.config}")
    print(f"model_path={model_path}")
    print(f"split={args.split}")
    print(f"windows={len(inputs)}")
    print(f"input_shape={inputs.shape}")
    print(f"calibration_shape={np.asarray(calibration.inputs).shape}")
    _print_model_report("keras", keras_predictions, targets, keras_predictions)

    if args.keep_tflite_dir is None:
        temp_context = tempfile.TemporaryDirectory(prefix="tinyodom_compare_")
        export_dir = Path(temp_context.name)
    else:
        temp_context = None
        export_dir = args.keep_tflite_dir
        export_dir.mkdir(parents=True, exist_ok=True)

    try:
        float_path = export_dir / "diagnostic_float.tflite"
        int8_path = export_dir / "diagnostic_int8.tflite"
        convert_to_tflite_model(
            model,
            training_data=None,
            quantization_mode="float",
            output_name=float_path,
        )
        convert_to_tflite_model(
            model,
            training_data=calibration.inputs,
            quantization_mode="int8_ptq",
            output_name=int8_path,
        )
        for label, path in (("float_tflite", float_path), ("int8_tflite", int8_path)):
            interpreter = tf.lite.Interpreter(model_path=str(path), experimental_delegates=[], num_threads=1)
            print(f"{label}_input_details={interpreter.get_input_details()}")
            print(f"{label}_output_details={interpreter.get_output_details()}")
            predictions = _batched_tflite_predict(path, inputs, args.batch_size)
            _print_model_report(label, predictions, targets, keras_predictions)
        print(f"tflite_dir={export_dir}")
    finally:
        if temp_context is not None:
            temp_context.cleanup()


if __name__ == "__main__":
    main()
