# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Focused tests for exact TFLite artifact wire validation."""

from pathlib import Path
import sys
import tempfile
import unittest

import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crest.hil.artifact_transport import (
    describe_tflite_bytes, read_tflite_artifact, sha256_bytes,
    validate_artifact_metadata,
)


class ArtifactTransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        @tf.function(input_signature=[tf.TensorSpec([1, 4], tf.float32)])
        def tiny_model(value):
            return value[:, :3]

        converter = tf.lite.TFLiteConverter.from_concrete_functions(
            [tiny_model.get_concrete_function()]
        )
        cls.model_bytes = converter.convert()

    def metadata(self) -> dict:
        tensors = describe_tflite_bytes(self.model_bytes)
        return {
            "format": "tflite",
            "byte_length": len(self.model_bytes),
            "sha256": sha256_bytes(self.model_bytes),
            **tensors,
        }

    def test_tiny_valid_tflite_round_trip(self) -> None:
        metadata = self.metadata()
        self.assertEqual(
            validate_artifact_metadata(metadata, self.model_bytes, max_bytes=4096),
            metadata,
        )

    def test_hash_and_strict_tensor_types_are_rejected(self) -> None:
        bad_hash = self.metadata()
        bad_hash["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "sha256"):
            validate_artifact_metadata(bad_hash, self.model_bytes, max_bytes=4096)
        bad_shape = self.metadata()
        bad_shape["inputs"][0]["shape"][0] = True
        with self.assertRaisesRegex(ValueError, "shape"):
            validate_artifact_metadata(bad_shape, self.model_bytes, max_bytes=4096)

    def test_file_size_is_bounded_before_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "large.tflite"
            path.write_bytes(b"x" * 9)
            with self.assertRaisesRegex(ValueError, "maximum"):
                read_tflite_artifact(path, max_bytes=8)



if __name__ == "__main__":
    unittest.main()
