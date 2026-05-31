"""Static model resource metric estimators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


@dataclass(frozen=True)
class StaticMemoryEstimate:
    """Static tensor-memory proxy estimate for one built model.

    Parameters
    ----------
    weight_bytes : int
        Unique model weight bytes under the selected deployment dtype.
    activation_bytes : int
        Sum of per-layer input and output activation bytes for batch size 1.
    memory_traffic_bytes : int
        Sum of per-layer input activation, weight, and output activation bytes.
    dtype_bytes : int
        Deployment dtype width in bytes.
    warning_count : int
        Count of layer estimates that required an inferred or incomplete shape.

    Attributes
    ----------
    weight_bytes : int
        Unique model weight bytes under the selected deployment dtype.
    activation_bytes : int
        Sum of per-layer input and output activation bytes for batch size 1.
    memory_traffic_bytes : int
        Sum of per-layer input activation, weight, and output activation bytes.
    dtype_bytes : int
        Deployment dtype width in bytes.
    warning_count : int
        Count of layer estimates that required an inferred or incomplete shape.
    """

    weight_bytes: int
    activation_bytes: int
    memory_traffic_bytes: int
    dtype_bytes: int
    warning_count: int = 0


def dtype_bytes_for_quantization(quantization_mode: str) -> int:
    """Return deployment dtype width for a supported quantization mode.

    Parameters
    ----------
    quantization_mode : str
        Deployment quantization mode, such as ``"float"`` or ``"int8_ptq"``.

    Returns
    -------
    int
        Number of bytes per scalar value in the static proxy.

    Raises
    ------
    ValueError
        If the mode is not supported by the static proxy.
    """
    normalized = str(quantization_mode).strip().lower()
    if normalized in {"float", "float32"}:
        return 4
    if normalized in {"int8_ptq", "int8"}:
        return 1
    raise ValueError(
        f"Unsupported quantization mode for static memory estimate: {quantization_mode!r}."
    )


def tensor_shape_elements(shape_like: Any) -> int | None:
    """Count elements for a tensor shape with batch size fixed to one.

    Parameters
    ----------
    shape_like : Any
        Tensor, TensorShape, tuple/list, variable, or nested output collection.

    Returns
    -------
    int | None
        Element count, or ``None`` when a non-batch dimension is unknown.
    """
    if shape_like is None:
        return None
    if isinstance(shape_like, (list, tuple)) and shape_like and not all(
        isinstance(dim, (int, type(None))) for dim in shape_like
    ):
        total = 0
        for item in shape_like:
            item_elements = tensor_shape_elements(item)
            if item_elements is None:
                return None
            total += item_elements
        return int(total)

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


def layer_tensor_elements(layer: tf.keras.layers.Layer, attr_name: str) -> int | None:
    """Return element count for a layer input or output tensor.

    Parameters
    ----------
    layer : tensorflow.keras.layers.Layer
        Layer to inspect.
    attr_name : str
        Tensor attribute name, usually ``"input"`` or ``"output"``.

    Returns
    -------
    int | None
        Element count when Keras exposes a concrete symbolic shape.
    """
    try:
        return tensor_shape_elements(getattr(layer, attr_name))
    except (AttributeError, RuntimeError, ValueError):
        return None


def unique_weight_bytes(
    layer: tf.keras.layers.Layer,
    *,
    dtype_bytes: int,
    seen_weights: set[int],
) -> int:
    """Count bytes for weights that have not already been counted.

    Parameters
    ----------
    layer : tensorflow.keras.layers.Layer
        Layer whose weights should be counted.
    dtype_bytes : int
        Deployment dtype width.
    seen_weights : set[int]
        Mutable set of Keras variable identities already counted.

    Returns
    -------
    int
        Unique weight bytes for this layer.
    """
    total = 0
    for weight in getattr(layer, "weights", []) or []:
        key = id(weight)
        if key in seen_weights:
            continue
        seen_weights.add(key)
        elements = tensor_shape_elements(weight)
        if elements is not None:
            total += elements * dtype_bytes
    return int(total)


def estimate_static_memory_keras(
    model: tf.keras.Model,
    *,
    quantization_mode: str,
) -> StaticMemoryEstimate:
    """Estimate static memory traffic for a generic Keras model.

    Parameters
    ----------
    model : tensorflow.keras.Model
        Built model to inspect.
    quantization_mode : str
        Deployment quantization mode used to choose scalar byte width.

    Returns
    -------
    StaticMemoryEstimate
        Static tensor traffic proxy for batch size 1.
    """
    dtype_bytes = dtype_bytes_for_quantization(quantization_mode)
    seen_weights: set[int] = set()
    weight_bytes = 0
    activation_bytes = 0
    warning_count = 0

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        input_elements = layer_tensor_elements(layer, "input")
        output_elements = layer_tensor_elements(layer, "output")
        if input_elements is None or output_elements is None:
            warning_count += 1
        input_bytes = 0 if input_elements is None else input_elements * dtype_bytes
        output_bytes = 0 if output_elements is None else output_elements * dtype_bytes
        activation_bytes += input_bytes + output_bytes
        weight_bytes += unique_weight_bytes(
            layer,
            dtype_bytes=dtype_bytes,
            seen_weights=seen_weights,
        )

    for weight in model.weights:
        key = id(weight)
        if key in seen_weights:
            continue
        seen_weights.add(key)
        elements = tensor_shape_elements(weight)
        if elements is None:
            warning_count += 1
            continue
        weight_bytes += elements * dtype_bytes

    return StaticMemoryEstimate(
        weight_bytes=int(weight_bytes),
        activation_bytes=int(activation_bytes),
        memory_traffic_bytes=int(weight_bytes + activation_bytes),
        dtype_bytes=int(dtype_bytes),
        warning_count=int(warning_count),
    )


def count_flops_keras(model: tf.keras.Model, input_shape: tuple[int, ...]) -> int:
    """Estimate Keras model FLOPs by profiling a frozen forward graph.

    Parameters
    ----------
    model : tensorflow.keras.Model
        Built Keras model to profile.
    input_shape : tuple[int, ...]
        Logical input shape excluding the batch dimension.

    Returns
    -------
    int
        TensorFlow profiler FLOP count for a single forward pass with batch size 1.

    Notes
    -----
    This is a static graph proxy. It is useful for relative NAS comparisons,
    but it is not predicted latency or measured energy.
    """
    concrete = tf.function(model).get_concrete_function(
        tf.TensorSpec([1, *input_shape], tf.float32)
    )
    frozen = convert_variables_to_constants_v2(concrete)
    graph_def = frozen.graph.as_graph_def()
    with tf.Graph().as_default() as graph:
        tf.compat.v1.import_graph_def(graph_def, name="")
        options = (
            tf.compat.v1.profiler.ProfileOptionBuilder(
                tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            )
            .with_empty_output()
            .build()
        )
        flops = tf.compat.v1.profiler.profile(graph, options=options)
    return int(flops.total_float_ops)
