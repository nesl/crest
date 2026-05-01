"""Integration tests for end-to-end model metric collection flows."""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

from test.test_support import ROOT_DIR, _cli_exists

from tinyodom.hardware import convert_to_cpp_model, convert_to_tflite_model
from tinyodom.model import CollectMetricsRequest, collect_metrics


@unittest.skipUnless(_cli_exists(), "Arduino CLI not installed")
class CollectMetricsIntegrationTests(unittest.TestCase):
    """Run collect_metrics against the real controller (proxy mode)."""

    def test_proxy_flow_runs_end_to_end(self) -> None:
        # The proxy-only training flow should still run end to end with the integration fixture.
        sketch_src = ROOT_DIR / "test" / "tinyodom_tcn"
        with tempfile.TemporaryDirectory() as tmpdir:
            sketch_dir = Path(tmpdir) / "tinyodom_tcn"
            shutil.copytree(sketch_src, sketch_dir)

            # Build a tiny model so Arduino CLI compiles deterministically without overflow.
            inputs = tf.keras.Input(shape=(64, 3))
            x = tf.keras.layers.Conv1D(4, kernel_size=3, activation="relu")(inputs)
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
            outputs = tf.keras.layers.Dense(2, activation="linear")(x)
            model = tf.keras.Model(inputs, outputs)

            dummy_data = np.random.rand(8, 64, 3).astype(np.float32)
            tflite_path = sketch_dir / "model.tflite"
            convert_to_tflite_model(model, dummy_data, output_name=tflite_path)
            convert_to_cpp_model(
                tflite_path,
                sketch_dir,
                array_name="g_model",
                source_name="model.cc",
                header_name="model.h",
            )

            metrics = collect_metrics(
                CollectMetricsRequest(
                    hil_enabled=False,
                    energy_aware=False,
                    flops=5_000_000,
                    device_name="ARDUINO_NANO_33_BLE_SENSE",
                    window_size=200,
                    input_dim=3,
                    dirpath=sketch_dir,
                    latency_proxy_max_flops=30_000_000,
                    serial_port=None,
                )
            )

        self.assertGreaterEqual(metrics["flash_bytes"], 0)
        self.assertGreaterEqual(metrics["ram_bytes"], -1)
        self.assertGreaterEqual(metrics["arena_bytes"], 0)
        self.assertEqual(metrics["latency_ms"], -1)
        self.assertEqual(metrics["latency_budget_ms"], -1)
