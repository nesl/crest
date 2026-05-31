"""Subprocess entry point for isolated host-side TFLite prediction."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .hardware import predict_tflite_model


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the TFLite prediction worker.

    Returns
    -------
    argparse.Namespace
        Parsed worker arguments containing model, input, and output paths.
    """
    parser = argparse.ArgumentParser(description="Run CREST TFLite prediction in a child process.")
    parser.add_argument("--model", required=True, type=Path, help="Path to the TFLite model flatbuffer.")
    parser.add_argument("--inputs", required=True, type=Path, help="Path to an NPZ file containing key 'inputs'.")
    parser.add_argument("--outputs", required=True, type=Path, help="Destination NPZ prediction file.")
    return parser.parse_args()


def _ordered_outputs(predictions: np.ndarray | list[np.ndarray]) -> list[np.ndarray]:
    """Normalize direct prediction results to an ordered output list.

    Parameters
    ----------
    predictions : numpy.ndarray | list[numpy.ndarray]
        Result returned by ``predict_tflite_model``.

    Returns
    -------
    list[numpy.ndarray]
        Ordered prediction arrays for serialization.
    """
    if isinstance(predictions, list):
        return [np.asarray(output) for output in predictions]
    return [np.asarray(predictions)]


def main() -> None:
    """Run direct TFLite prediction and write the subprocess output contract.

    Returns
    -------
    None
    """
    args = _parse_args()
    with np.load(args.inputs, allow_pickle=False) as input_archive:
        inputs = np.asarray(input_archive["inputs"], dtype=np.float32)
    predictions = _ordered_outputs(predict_tflite_model(args.model, inputs))
    output_payload = {"num_outputs": np.asarray(len(predictions), dtype=np.int64)}
    for output_index, output in enumerate(predictions):
        output_payload[f"output_{output_index}"] = output
    np.savez(args.outputs, **output_payload)


if __name__ == "__main__":
    main()
