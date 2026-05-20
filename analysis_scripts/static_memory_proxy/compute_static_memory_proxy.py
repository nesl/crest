"""Compute static memory-traffic proxy metrics for logged OdomTCN trials."""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402
import yaml  # noqa: E402

from tinyodom.model_families.odom_tcn import OdomTCNFamily  # noqa: E402
from tinyodom.pipeline_types import ModelBuildContext, TargetSpec  # noqa: E402


DTYPE_BYTES = {
    "float": 4,
    "float32": 4,
    "int8": 1,
    "int8_ptq": 1,
}
PROXY_COLUMNS = [
    "proxy_weight_bytes",
    "proxy_activation_bytes",
    "proxy_memory_traffic_bytes",
    "proxy_dtype_bytes",
    "proxy_warning_count",
    "proxy_quantization_mode",
    "proxy_quantization_mode_source",
]


@dataclass
class LayerEstimate:
    """Memory proxy estimate for one accounted layer operation.

    Parameters
    ----------
    layer_name : str
        Human-readable layer path.
    layer_type : str
        Keras layer class name.
    input_activation_bytes : int
        Estimated bytes read from the input activation tensor.
    weight_bytes : int
        Estimated bytes read from layer weights.
    output_activation_bytes : int
        Estimated bytes written to the output activation tensor.
    traffic_bytes : int
        Sum of input activation, weight, and output activation bytes.
    warning : str
        Empty string when shapes were directly available, otherwise a short
        explanation of the approximation used.
    shape_source : str
        Source used for activation shape estimates.
    """

    layer_name: str
    layer_type: str
    input_activation_bytes: int
    weight_bytes: int
    output_activation_bytes: int
    traffic_bytes: int
    warning: str
    shape_source: str


@dataclass
class ProxyEstimate:
    """Aggregate proxy metrics for one trial row.

    Parameters
    ----------
    weight_bytes : int
        Total unique model weight bytes accounted in the proxy.
    activation_bytes : int
        Sum of per-layer input and output activation bytes.
    memory_traffic_bytes : int
        Sum of per-layer input activation, weight, and output activation bytes.
    dtype_bytes : int
        Deployment dtype width in bytes.
    warning_count : int
        Number of layer operations whose activation shape was inferred or
        unavailable.
    layer_details : list[LayerEstimate]
        Per-layer estimate details.
    """

    weight_bytes: int
    activation_bytes: int
    memory_traffic_bytes: int
    dtype_bytes: int
    warning_count: int
    layer_details: list[LayerEstimate]


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed CLI options.
    """

    parser = argparse.ArgumentParser(
        description="Augment OdomTCN NAS trial CSVs with static memory-traffic proxy metrics."
    )
    parser.add_argument("--config", required=True, type=Path, help="NAS config YAML path.")
    parser.add_argument(
        "--trials-csv",
        required=True,
        type=Path,
        nargs="+",
        help="One or more NAS trial CSVs to augment.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Output CSV path. Only valid when one --trials-csv is provided.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for augmented CSVs when processing one or more inputs.",
    )
    parser.add_argument(
        "--input-shape",
        help="Override input shape as TIMESTEPS,INPUT_DIM. Defaults to row hparams, then config.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        help="Process only the first N rows. Useful for quick validation.",
    )
    parser.add_argument(
        "--include-layer-details",
        action="store_true",
        help="Include proxy_layer_details_json in the augmented CSV.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Write scatter plots and print optional correlations.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        help="Directory for plot PNGs. Defaults to each output CSV directory.",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file as a plain dictionary.

    Parameters
    ----------
    path : pathlib.Path
        YAML file path.

    Returns
    -------
    dict[str, Any]
        Parsed YAML payload.
    """

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping at config root: {path}")
    return payload


def _is_missing(value: Any) -> bool:
    """Return whether a CSV cell should be treated as missing.

    Parameters
    ----------
    value : Any
        Candidate cell value.

    Returns
    -------
    bool
        True when the value is absent or NaN-like.
    """

    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _parse_cell(value: Any) -> Any:
    """Parse one CSV cell into a Python value when possible.

    Parameters
    ----------
    value : Any
        Raw CSV cell value.

    Returns
    -------
    Any
        Parsed scalar/list/dict value, or the original value.
    """

    if _is_missing(value):
        return None
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text == "":
        return None
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    if text[:1] in {"[", "{", "("}:
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(text)
            except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
                continue
    try:
        if any(marker in text for marker in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _first_present(row: pd.Series, names: Iterable[str]) -> tuple[Any, str | None]:
    """Return the first present value from a row.

    Parameters
    ----------
    row : pandas.Series
        Input row.
    names : Iterable[str]
        Candidate column names in priority order.

    Returns
    -------
    tuple[Any, str | None]
        Parsed value and the column that supplied it.
    """

    for name in names:
        if name in row.index and not _is_missing(row[name]):
            parsed = _parse_cell(row[name])
            if parsed is not None:
                return parsed, name
    return None, None


def _row_value(row: pd.Series, logical_name: str) -> tuple[Any, str | None]:
    """Resolve a logical trial field across supported CSV schemas.

    Parameters
    ----------
    row : pandas.Series
        Input row.
    logical_name : str
        Unprefixed hparam or metric name.

    Returns
    -------
    tuple[Any, str | None]
        Parsed value and source column.
    """

    return _first_present(
        row,
        (
            logical_name,
            f"hparam__{logical_name}",
            f"user_attrs_hparam__{logical_name}",
            f"params_{logical_name}",
            f"user_attrs_{logical_name}",
        ),
    )


def _config_input_shape(config: dict[str, Any]) -> tuple[int, int] | None:
    """Infer a model input shape from the run config without loading data.

    Parameters
    ----------
    config : dict[str, Any]
        Parsed NAS config.

    Returns
    -------
    tuple[int, int] | None
        ``(timesteps, input_dim)`` when inferable.
    """

    dataset = config.get("dataset", {}) or {}
    params = dataset.get("params", {}) or {}
    timesteps = params.get("window_size")
    input_dim = params.get("input_dim")
    if input_dim is None and str(dataset.get("name", "")).strip().lower() == "oxiod":
        input_dim = 10
    if timesteps is None or input_dim is None:
        return None
    return int(timesteps), int(input_dim)


def _parse_input_shape_override(raw_value: str | None) -> tuple[int, int] | None:
    """Parse the optional CLI input-shape override.

    Parameters
    ----------
    raw_value : str | None
        Raw CLI value.

    Returns
    -------
    tuple[int, int] | None
        Parsed shape or None when no override was provided.
    """

    if raw_value is None:
        return None
    parts = [part.strip() for part in raw_value.split(",")]
    if len(parts) != 2:
        raise ValueError("--input-shape must use TIMESTEPS,INPUT_DIM format.")
    timesteps, input_dim = (int(parts[0]), int(parts[1]))
    if timesteps <= 0 or input_dim <= 0:
        raise ValueError("--input-shape dimensions must be positive.")
    return timesteps, input_dim


def _resolve_input_shape(
    row: pd.Series,
    config_shape: tuple[int, int] | None,
    override_shape: tuple[int, int] | None,
) -> tuple[int, int]:
    """Resolve the OdomTCN logical input shape for one row.

    Parameters
    ----------
    row : pandas.Series
        Input trial row.
    config_shape : tuple[int, int] | None
        Shape inferred from config.
    override_shape : tuple[int, int] | None
        Explicit CLI override.

    Returns
    -------
    tuple[int, int]
        ``(timesteps, input_dim)``.
    """

    if override_shape is not None:
        return override_shape
    row_timesteps, _ = _row_value(row, "timesteps")
    row_input_dim, _ = _row_value(row, "input_dim")
    if row_timesteps is not None and row_input_dim is not None:
        return int(row_timesteps), int(row_input_dim)
    if config_shape is not None:
        return config_shape
    raise ValueError(
        "Could not resolve input shape. Provide hparam__timesteps/hparam__input_dim "
        "columns or pass --input-shape TIMESTEPS,INPUT_DIM."
    )


def _normalize_quantization_mode(row: pd.Series) -> tuple[str, int, str]:
    """Resolve quantization mode and dtype width for one row.

    Parameters
    ----------
    row : pandas.Series
        Input trial row.

    Returns
    -------
    tuple[str, int, str]
        Normalized mode, dtype bytes, and source description.
    """

    value, source = _first_present(
        row,
        (
            "quantization_mode",
            "params_quantization_mode",
            "user_attrs_quantization_mode",
            "hparam__quantization_mode",
            "user_attrs_hparam__quantization_mode",
        ),
    )
    if value is None:
        return "int8_ptq", DTYPE_BYTES["int8_ptq"], "defaulted_missing"
    mode = str(value).strip().lower()
    if mode not in DTYPE_BYTES:
        raise ValueError(f"Unsupported quantization mode '{value}'. Expected float or int8_ptq.")
    if mode in {"float32"}:
        mode = "float"
    if mode in {"int8"}:
        mode = "int8_ptq"
    return mode, DTYPE_BYTES[mode], str(source)


def _trial_hparams(row: pd.Series, input_shape: tuple[int, int]) -> dict[str, Any]:
    """Decode OdomTCN hyperparameters from one trial row.

    Parameters
    ----------
    row : pandas.Series
        Input trial row.
    input_shape : tuple[int, int]
        Logical model input shape.

    Returns
    -------
    dict[str, Any]
        Build-time OdomTCN hyperparameters.
    """

    raw: dict[str, Any] = {}
    for name in (
        "nb_filters",
        "kernel_size",
        "dropout_rate",
        "use_skip_connections",
        "norm_flag",
        "dilations",
        "dilations_index",
    ):
        value, _ = _row_value(row, name)
        if value is not None:
            raw[name] = value
    missing = [
        name
        for name in ("nb_filters", "kernel_size", "dropout_rate", "use_skip_connections", "norm_flag")
        if name not in raw
    ]
    if "dilations" not in raw and "dilations_index" not in raw:
        missing.append("dilations or dilations_index")
    if missing:
        raise ValueError(f"Missing OdomTCN hparam columns: {', '.join(missing)}")

    ctx = _build_context(input_shape)
    decoded = OdomTCNFamily().decode_trial_hparams(raw, ctx, {})
    decoded["nb_filters"] = int(decoded["nb_filters"])
    decoded["kernel_size"] = int(decoded["kernel_size"])
    decoded["dropout_rate"] = float(decoded["dropout_rate"])
    decoded["use_skip_connections"] = bool(decoded["use_skip_connections"])
    decoded["norm_flag"] = bool(decoded["norm_flag"])
    decoded["dilations"] = [int(value) for value in decoded["dilations"]]
    return decoded


def _build_context(input_shape: tuple[int, int]) -> ModelBuildContext:
    """Build the minimal OdomTCN model context used by this analysis.

    Parameters
    ----------
    input_shape : tuple[int, int]
        Logical model input shape.

    Returns
    -------
    tinyodom.pipeline_types.ModelBuildContext
        Build context with the odometry two-head target spec.
    """

    return ModelBuildContext(
        input_shape=input_shape,
        input_dtype="float32",
        target_spec=TargetSpec(
            task_type="regression",
            output_names=["velx", "vely"],
            output_shapes=[(1,), (1,)],
        ),
    )


def _shape_elements(shape_like: Any) -> int | None:
    """Count elements in one tensor shape with batch size fixed to one.

    Parameters
    ----------
    shape_like : Any
        Tensor, TensorShape, tuple, or nested collection of those.

    Returns
    -------
    int | None
        Element count, or None when any non-batch dimension is unknown.
    """

    if shape_like is None:
        return None
    if isinstance(shape_like, (list, tuple)) and shape_like and not all(
        isinstance(dim, (int, type(None))) for dim in shape_like
    ):
        total = 0
        for item in shape_like:
            item_elements = _shape_elements(item)
            if item_elements is None:
                return None
            total += item_elements
        return total
    shape = getattr(shape_like, "shape", shape_like)
    try:
        dims = list(shape.as_list())
    except AttributeError:
        try:
            dims = list(shape)
        except TypeError:
            return None
    if not dims:
        return 1
    if dims[0] is None:
        dims[0] = 1
    elements = 1
    for dim in dims:
        if dim is None:
            return None
        elements *= int(dim)
    return int(elements)


def _layer_tensor_elements(layer: tf.keras.layers.Layer, attr_name: str) -> int | None:
    """Read layer input/output tensor element counts.

    Parameters
    ----------
    layer : tensorflow.keras.layers.Layer
        Layer to inspect.
    attr_name : str
        Either ``"input"`` or ``"output"``.

    Returns
    -------
    int | None
        Element count when Keras exposes a concrete symbolic tensor shape.
    """

    try:
        return _shape_elements(getattr(layer, attr_name))
    except (AttributeError, RuntimeError, ValueError):
        return None


def _weight_bytes(layer: tf.keras.layers.Layer, dtype_bytes: int, seen_weights: set[int]) -> int:
    """Count unique weight bytes for one layer.

    Parameters
    ----------
    layer : tensorflow.keras.layers.Layer
        Layer whose weights should be counted.
    dtype_bytes : int
        Deployment dtype width.
    seen_weights : set[int]
        Mutable set of already-counted Keras variable identities.

    Returns
    -------
    int
        Bytes for weights not previously seen.
    """

    total = 0
    for weight in getattr(layer, "weights", []) or []:
        key = id(weight)
        if key in seen_weights:
            continue
        seen_weights.add(key)
        elements = _shape_elements(weight)
        if elements is not None:
            total += elements * dtype_bytes
    return int(total)


def _make_estimate(
    *,
    layer_name: str,
    layer_type: str,
    input_elements: int | None,
    weight_bytes: int,
    output_elements: int | None,
    dtype_bytes: int,
    warning: str,
    shape_source: str,
) -> LayerEstimate:
    """Create a layer estimate from element counts.

    Parameters
    ----------
    layer_name : str
        Layer path.
    layer_type : str
        Layer class name.
    input_elements : int | None
        Input activation elements.
    weight_bytes : int
        Weight bytes.
    output_elements : int | None
        Output activation elements.
    dtype_bytes : int
        Deployment dtype width.
    warning : str
        Warning text, if any.
    shape_source : str
        Shape source label.

    Returns
    -------
    LayerEstimate
        Completed per-layer estimate.
    """

    input_bytes = 0 if input_elements is None else int(input_elements) * dtype_bytes
    output_bytes = 0 if output_elements is None else int(output_elements) * dtype_bytes
    return LayerEstimate(
        layer_name=layer_name,
        layer_type=layer_type,
        input_activation_bytes=input_bytes,
        weight_bytes=int(weight_bytes),
        output_activation_bytes=output_bytes,
        traffic_bytes=input_bytes + int(weight_bytes) + output_bytes,
        warning=warning,
        shape_source=shape_source,
    )


def _generic_layer_estimate(
    layer: tf.keras.layers.Layer,
    dtype_bytes: int,
    seen_weights: set[int],
) -> LayerEstimate:
    """Estimate memory traffic for a standard Keras layer.

    Parameters
    ----------
    layer : tensorflow.keras.layers.Layer
        Layer to inspect.
    dtype_bytes : int
        Deployment dtype width.
    seen_weights : set[int]
        Mutable set of already-counted Keras weights.

    Returns
    -------
    LayerEstimate
        Estimate based on Keras symbolic input/output tensor shapes.
    """

    input_elements = _layer_tensor_elements(layer, "input")
    output_elements = _layer_tensor_elements(layer, "output")
    warning = ""
    shape_source = "keras_tensor"
    if input_elements is None or output_elements is None:
        warning = "missing_keras_activation_shape"
        shape_source = "unavailable"
    return _make_estimate(
        layer_name=layer.name,
        layer_type=type(layer).__name__,
        input_elements=input_elements,
        weight_bytes=_weight_bytes(layer, dtype_bytes, seen_weights),
        output_elements=output_elements,
        dtype_bytes=dtype_bytes,
        warning=warning,
        shape_source=shape_source,
    )


def _conv_output_channels(layer: tf.keras.layers.Layer) -> int | None:
    """Infer output channel count for a Conv1D-like layer.

    Parameters
    ----------
    layer : tensorflow.keras.layers.Layer
        Candidate Conv1D layer.

    Returns
    -------
    int | None
        Output channel count when available.
    """

    filters = getattr(layer, "filters", None)
    if filters is not None:
        return int(filters)
    weights = getattr(layer, "weights", []) or []
    if weights:
        shape = list(weights[0].shape)
        if shape:
            return int(shape[-1])
    return None


def _estimate_tcn_layer(
    layer: tf.keras.layers.Layer,
    dtype_bytes: int,
    seen_weights: set[int],
) -> list[LayerEstimate]:
    """Estimate a keras-tcn TCN layer, including exposed residual blocks.

    Parameters
    ----------
    layer : tensorflow.keras.layers.Layer
        TCN layer to inspect.
    dtype_bytes : int
        Deployment dtype width.
    seen_weights : set[int]
        Mutable set of already-counted Keras weights.

    Returns
    -------
    list[LayerEstimate]
        Per-operation estimates for the TCN internals and final output slice.
    """

    input_shape = getattr(getattr(layer, "input", None), "shape", None)
    output_elements = _layer_tensor_elements(layer, "output")
    residual_blocks = list(getattr(layer, "residual_blocks", []) or [])
    if input_shape is None or len(input_shape) < 3 or not residual_blocks:
        estimate = _generic_layer_estimate(layer, dtype_bytes, seen_weights)
        estimate.warning = "tcn_internal_layers_unavailable"
        estimate.shape_source = "keras_tensor_black_box"
        return [estimate]

    timesteps = int(input_shape[1])
    current_channels = int(input_shape[2])
    estimates: list[LayerEstimate] = []
    for block_index, block in enumerate(residual_blocks):
        residual_channels = current_channels
        block_layers = list(getattr(block, "_layers", []) or getattr(block, "layers", []) or [])
        for child in block_layers:
            layer_type = type(child).__name__
            child_name = f"{layer.name}/residual_block_{block_index}/{child.name}"
            if layer_type == "Conv1D":
                output_channels = _conv_output_channels(child)
                if output_channels is None:
                    input_elements = None
                    output_child_elements = None
                else:
                    input_channels = (
                        residual_channels
                        if str(child.name).startswith("matching_")
                        else current_channels
                    )
                    input_elements = timesteps * input_channels
                    output_child_elements = timesteps * output_channels
                    if not str(child.name).startswith("matching_"):
                        current_channels = output_channels
            elif layer_type == "Lambda" and str(child.name).startswith("matching_"):
                input_elements = timesteps * residual_channels
                output_child_elements = timesteps * residual_channels
            else:
                input_elements = timesteps * current_channels
                output_child_elements = timesteps * current_channels
            estimates.append(
                _make_estimate(
                    layer_name=child_name,
                    layer_type=layer_type,
                    input_elements=input_elements,
                    weight_bytes=_weight_bytes(child, dtype_bytes, seen_weights),
                    output_elements=output_child_elements,
                    dtype_bytes=dtype_bytes,
                    warning="inferred_tcn_internal_activation_shape",
                    shape_source="inferred_tcn_residual_block",
                )
            )
        current_channels = _conv_output_channels(block_layers[0]) or current_channels

    estimates.append(
        _make_estimate(
            layer_name=f"{layer.name}/output_slice",
            layer_type=type(layer).__name__,
            input_elements=timesteps * current_channels,
            weight_bytes=0,
            output_elements=output_elements,
            dtype_bytes=dtype_bytes,
            warning="inferred_tcn_final_sequence_to_vector_shape",
            shape_source="inferred_tcn_output_slice",
        )
    )
    return estimates


def _estimate_model(model: tf.keras.Model, dtype_bytes: int) -> ProxyEstimate:
    """Estimate static memory traffic for one built Keras model.

    Parameters
    ----------
    model : tensorflow.keras.Model
        Built OdomTCN model.
    dtype_bytes : int
        Deployment dtype width.

    Returns
    -------
    ProxyEstimate
        Aggregate and layer-level proxy metrics.
    """

    details: list[LayerEstimate] = []
    seen_weights: set[int] = set()
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        if type(layer).__name__ == "TCN":
            details.extend(_estimate_tcn_layer(layer, dtype_bytes, seen_weights))
        else:
            details.append(_generic_layer_estimate(layer, dtype_bytes, seen_weights))

    unassigned_weight_bytes = 0
    for weight in model.weights:
        key = id(weight)
        if key in seen_weights:
            continue
        seen_weights.add(key)
        elements = _shape_elements(weight)
        if elements is not None:
            unassigned_weight_bytes += elements * dtype_bytes
    if unassigned_weight_bytes:
        details.append(
            _make_estimate(
                layer_name="unassigned_model_weights",
                layer_type="Weights",
                input_elements=0,
                weight_bytes=unassigned_weight_bytes,
                output_elements=0,
                dtype_bytes=dtype_bytes,
                warning="weights_not_attributed_to_layer",
                shape_source="model_weights",
            )
        )

    weight_bytes = sum(item.weight_bytes for item in details)
    activation_bytes = sum(
        item.input_activation_bytes + item.output_activation_bytes for item in details
    )
    memory_traffic_bytes = sum(item.traffic_bytes for item in details)
    warning_count = sum(1 for item in details if item.warning)
    return ProxyEstimate(
        weight_bytes=int(weight_bytes),
        activation_bytes=int(activation_bytes),
        memory_traffic_bytes=int(memory_traffic_bytes),
        dtype_bytes=dtype_bytes,
        warning_count=int(warning_count),
        layer_details=details,
    )


def _estimate_row(
    row: pd.Series,
    config_shape: tuple[int, int] | None,
    override_shape: tuple[int, int] | None,
) -> dict[str, Any]:
    """Compute proxy output columns for one CSV row.

    Parameters
    ----------
    row : pandas.Series
        Trial row.
    config_shape : tuple[int, int] | None
        Shape inferred from config.
    override_shape : tuple[int, int] | None
        Explicit CLI override.

    Returns
    -------
    dict[str, Any]
        Proxy output columns and layer details.
    """

    quant_mode, dtype_bytes, quant_source = _normalize_quantization_mode(row)
    input_shape = _resolve_input_shape(row, config_shape, override_shape)
    hparams = _trial_hparams(row, input_shape)
    family = OdomTCNFamily()
    ctx = _build_context(input_shape)
    model = family.build_model(hparams, ctx, {})
    estimate = _estimate_model(model, dtype_bytes)
    tf.keras.backend.clear_session()
    return {
        "proxy_weight_bytes": estimate.weight_bytes,
        "proxy_activation_bytes": estimate.activation_bytes,
        "proxy_memory_traffic_bytes": estimate.memory_traffic_bytes,
        "proxy_dtype_bytes": estimate.dtype_bytes,
        "proxy_warning_count": estimate.warning_count,
        "proxy_quantization_mode": quant_mode,
        "proxy_quantization_mode_source": quant_source,
        "proxy_layer_details_json": json.dumps(
            [asdict(item) for item in estimate.layer_details],
            sort_keys=True,
        ),
    }


def _default_output_path(input_csv: Path, output_dir: Path | None) -> Path:
    """Return the default augmented CSV path for one input.

    Parameters
    ----------
    input_csv : pathlib.Path
        Source CSV path.
    output_dir : pathlib.Path | None
        Optional output directory override.

    Returns
    -------
    pathlib.Path
        Destination CSV path.
    """

    parent = input_csv.parent if output_dir is None else output_dir
    return parent / f"{input_csv.stem}_with_memory_proxy{input_csv.suffix}"


def _augment_csv(
    *,
    input_csv: Path,
    output_csv: Path,
    config_shape: tuple[int, int] | None,
    override_shape: tuple[int, int] | None,
    max_rows: int | None,
    include_layer_details: bool,
) -> pd.DataFrame:
    """Read, augment, and write one trial CSV.

    Parameters
    ----------
    input_csv : pathlib.Path
        Source trial CSV.
    output_csv : pathlib.Path
        Destination augmented CSV.
    config_shape : tuple[int, int] | None
        Shape inferred from config.
    override_shape : tuple[int, int] | None
        Explicit CLI shape override.
    max_rows : int | None
        Optional row cap.
    include_layer_details : bool
        Whether to retain the large layer-detail JSON column.

    Returns
    -------
    pandas.DataFrame
        Augmented dataframe.
    """

    frame = pd.read_csv(input_csv)
    if max_rows is not None:
        frame = frame.head(max_rows).copy()
    proxy_rows: list[dict[str, Any]] = []
    for index, row in frame.iterrows():
        try:
            proxy_rows.append(_estimate_row(row, config_shape, override_shape))
        except Exception as exc:  # pragma: no cover - exercised by CLI diagnostics.
            raise RuntimeError(f"Failed to estimate row {index} from {input_csv}: {exc}") from exc

    proxy_frame = pd.DataFrame(proxy_rows)
    if not include_layer_details and "proxy_layer_details_json" in proxy_frame:
        proxy_frame = proxy_frame.drop(columns=["proxy_layer_details_json"])
    output = pd.concat([frame.reset_index(drop=True), proxy_frame], axis=1)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False)
    return output


def _numeric_series(
    frame: pd.DataFrame,
    candidates: Iterable[str],
) -> tuple[pd.Series | None, str | None]:
    """Return the first usable numeric series from candidate columns.

    Parameters
    ----------
    frame : pandas.DataFrame
        Dataframe to inspect.
    candidates : Iterable[str]
        Candidate column names.

    Returns
    -------
    tuple[pandas.Series | None, str | None]
        Numeric series and source column name.
    """

    for column in candidates:
        if column not in frame.columns:
            continue
        series = pd.to_numeric(frame[column], errors="coerce")
        if series.notna().any():
            return series, column
    return None, None


def _plot_outputs(frame: pd.DataFrame, output_csv: Path, plot_dir: Path | None) -> None:
    """Write optional scatter plots and print rank correlations.

    Parameters
    ----------
    frame : pandas.DataFrame
        Augmented dataframe.
    output_csv : pathlib.Path
        CSV path used for plot naming.
    plot_dir : pathlib.Path | None
        Optional destination directory.

    Returns
    -------
    None
        Writes PNGs when matplotlib is available.
    """

    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        print("Plotting skipped: matplotlib is not available.")
        return

    target_dir = output_csv.parent if plot_dir is None else plot_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    x = pd.to_numeric(frame["proxy_memory_traffic_bytes"], errors="coerce")
    plot_specs = [
        (
            ("flops", "user_attrs_flops"),
            "flops",
        ),
        (
            ("metric__rmse_total", "rmse_total", "values_rmse_total", "user_attrs_metric__rmse_total"),
            "rmse_total",
        ),
        (
            (
                "energy_mj_per_inference",
                "values_energy_mj_per_inference",
                "user_attrs_energy_mj_per_inference",
            ),
            "energy_mj_per_inference",
        ),
    ]
    for candidates, label in plot_specs:
        y, source = _numeric_series(frame, candidates)
        if y is None or source is None:
            continue
        valid = x.notna() & y.notna() & (x >= 0) & (y >= 0) & (y < 1.0e11)
        if valid.sum() < 2:
            continue
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.scatter(x[valid], y[valid], s=18, alpha=0.75)
        ax.set_xlabel("static memory traffic bytes")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs static memory traffic")
        fig.tight_layout()
        path = target_dir / f"{output_csv.stem}_{label}_vs_memory_traffic.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        print(f"Wrote plot: {path}")
        _print_correlations(x[valid], y[valid], label)


def _print_correlations(x: pd.Series, y: pd.Series, label: str) -> None:
    """Print optional Spearman and Kendall correlations.

    Parameters
    ----------
    x : pandas.Series
        Proxy metric values.
    y : pandas.Series
        Target metric values.
    label : str
        Target metric label.

    Returns
    -------
    None
        Prints correlations to stdout.
    """

    try:
        from scipy import stats
    except ImportError:
        spearman = x.corr(y, method="spearman")
        kendall = x.corr(y, method="kendall")
        print(
            f"{label}: scipy unavailable; pandas Spearman={spearman:.4g}, "
            f"Kendall={kendall:.4g}"
        )
        return
    spearman = stats.spearmanr(x, y, nan_policy="omit")
    kendall = stats.kendalltau(x, y, nan_policy="omit")
    if not math.isnan(float(spearman.statistic)):
        print(
            f"{label}: Spearman={spearman.statistic:.4g} (p={spearman.pvalue:.4g}), "
            f"Kendall={kendall.statistic:.4g} (p={kendall.pvalue:.4g})"
        )


def main() -> None:
    """Run the static memory proxy CLI.

    Returns
    -------
    None
        Writes augmented CSVs and optional plots.
    """

    args = _parse_args()
    if args.output_csv is not None and len(args.trials_csv) != 1:
        raise SystemExit("--output-csv can only be used with exactly one --trials-csv input.")
    if args.output_csv is not None and args.output_dir is not None:
        raise SystemExit("Use only one of --output-csv or --output-dir.")
    if args.max_rows is not None and args.max_rows <= 0:
        raise SystemExit("--max-rows must be positive.")

    config = _load_yaml(args.config)
    config_shape = _config_input_shape(config)
    override_shape = _parse_input_shape_override(args.input_shape)

    for input_csv in args.trials_csv:
        output_csv = args.output_csv or _default_output_path(input_csv, args.output_dir)
        output = _augment_csv(
            input_csv=input_csv,
            output_csv=output_csv,
            config_shape=config_shape,
            override_shape=override_shape,
            max_rows=args.max_rows,
            include_layer_details=args.include_layer_details,
        )
        print(f"Wrote augmented CSV: {output_csv} ({len(output)} rows)")
        if args.plot:
            _plot_outputs(output, output_csv, args.plot_dir)


if __name__ == "__main__":
    main()
