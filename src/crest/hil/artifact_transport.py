# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Wire validation helpers for exact TFLite artifact HIL requests."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import tensorflow as tf

ARTIFACT_PROTOCOL_VERSION = 1
ARTIFACT_FORMAT = "tflite"
DEFAULT_MAX_ARTIFACT_BYTES = 16 * 1024 * 1024
DEFAULT_HIL_REQUEST_RETRIES = 1


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase SHA-256 digest of the supplied bytes."""
    return hashlib.sha256(data).hexdigest()


def _tensor_description(detail: Mapping[str, Any]) -> dict[str, Any]:
    parameters = detail.get("quantization_parameters", {})
    scales = np.asarray(parameters.get("scales", []))
    zero_points = np.asarray(parameters.get("zero_points", []))
    if not all(math.isfinite(float(value)) for value in scales.tolist()):
        raise ValueError("TFLite tensor quantization scales must be finite.")
    return {
        "shape": [int(value) for value in np.asarray(detail["shape"]).tolist()],
        "dtype": np.dtype(detail["dtype"]).name,
        "quantization": {
            "scales": [float(value) for value in scales.tolist()],
            "zero_points": [int(value) for value in zero_points.tolist()],
            "quantized_dimension": int(parameters.get("quantized_dimension", 0)),
        },
    }


def describe_tflite_bytes(data: bytes) -> dict[str, list[dict[str, Any]]]:
    """Read ordered input/output tensor metadata from valid TFLite bytes."""
    if len(data) < 8 or data[4:8] != b"TFL3":
        raise ValueError("Artifact does not contain a TFLite flatbuffer header.")
    try:
        interpreter = tf.lite.Interpreter(
            model_content=data,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        )
        interpreter.allocate_tensors()
    except (ValueError, RuntimeError) as exc:
        raise ValueError("Artifact is not a valid executable TFLite model.") from exc
    inputs = [_tensor_description(item) for item in interpreter.get_input_details()]
    outputs = [_tensor_description(item) for item in interpreter.get_output_details()]
    if not inputs or not outputs:
        raise ValueError("TFLite artifact must expose at least one input and output tensor.")
    return {"inputs": inputs, "outputs": outputs}


def read_tflite_artifact(path: Path, *, max_bytes: int) -> tuple[bytes, dict[str, Any]]:
    """Read and describe a regular TFLite file within the configured limit."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"TFLite artifact is not a regular file: {path}")
    if path.stat().st_size > max_bytes:
        raise ValueError(f"TFLite artifact exceeds maximum of {max_bytes} bytes.")
    with path.open("rb") as handle:
        data = handle.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"TFLite artifact is {len(data)} bytes; maximum is {max_bytes} bytes.")
    return data, describe_tflite_bytes(data)


def _validate_declared_tensors(value: object, *, field: str) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"artifact.{field} must be a non-empty list.")
    for tensor in value:
        if not isinstance(tensor, Mapping):
            raise ValueError(f"artifact.{field} entries must be JSON objects.")
        shape = tensor.get("shape")
        if not isinstance(shape, list) or not shape or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in shape):
            raise ValueError(f"artifact.{field}.shape must contain non-negative integers.")
        dtype = tensor.get("dtype")
        if not isinstance(dtype, str):
            raise ValueError(f"artifact.{field}.dtype must be a string.")
        quantization = tensor.get("quantization")
        if not isinstance(quantization, Mapping) or set(quantization) != {"scales", "zero_points", "quantized_dimension"}:
            raise ValueError(f"artifact.{field}.quantization has invalid fields.")
        scales = quantization["scales"]
        zero_points = quantization["zero_points"]
        dimension = quantization["quantized_dimension"]
        if not isinstance(scales, list) or any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)) or float(item) <= 0 for item in scales):
            raise ValueError(f"artifact.{field}.quantization.scales must be finite positive numbers.")
        if not isinstance(zero_points, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in zero_points):
            raise ValueError(f"artifact.{field}.quantization.zero_points must contain integers.")
        if len(scales) != len(zero_points):
            raise ValueError(f"artifact.{field} quantization arrays must have equal lengths.")
        if dtype in {"int8", "uint8"} and not scales:
            raise ValueError(f"artifact.{field} integer tensors require quantization parameters.")
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise ValueError(f"artifact.{field}.quantized_dimension must be an integer.")


def validate_artifact_metadata(
    metadata: Mapping[str, Any],
    data: bytes,
    *,
    max_bytes: int,
) -> dict[str, Any]:
    """Validate declared identity and tensor metadata against raw bytes."""
    if metadata.get("format") != ARTIFACT_FORMAT:
        raise ValueError("artifact.format must be 'tflite'.")
    declared_length = metadata.get("byte_length")
    if isinstance(declared_length, bool) or not isinstance(declared_length, int):
        raise ValueError("artifact.byte_length must be an integer.")
    if declared_length != len(data):
        raise ValueError("artifact.byte_length does not match the received bytes.")
    if len(data) > max_bytes:
        raise ValueError(f"Artifact exceeds maximum payload size of {max_bytes} bytes.")
    digest = sha256_bytes(data)
    if metadata.get("sha256") != digest:
        raise ValueError("artifact.sha256 does not match the received bytes.")
    _validate_declared_tensors(metadata.get("inputs"), field="inputs")
    _validate_declared_tensors(metadata.get("outputs"), field="outputs")
    actual = describe_tflite_bytes(data)
    if metadata.get("inputs") != actual["inputs"] or metadata.get("outputs") != actual["outputs"]:
        raise ValueError("Declared tensor metadata does not match the TFLite artifact.")
    return {
        "format": ARTIFACT_FORMAT,
        "byte_length": len(data),
        "sha256": digest,
        **actual,
    }
