import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from addict import Dict

# Ensure the repository root (`src` directory) is importable whenever this test
# module is executed via `python -m unittest`.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.hil_server import HILServer  # noqa: E402


class HILServerTestCase(unittest.TestCase):
    """Common test scaffolding for all HILServer unit tests.

    The real HIL server pulls configuration and datasets on construction, which
    would slow tests to a crawl. These helpers replace those heavy operations
    with small, deterministic doubles that still behave like the production
    objects.
    """

    def setUp(self) -> None:
        # Create a lightweight mock config object to avoid loading YAML files.
        self.config = SimpleNamespace(
            network=SimpleNamespace(host="127.0.0.1", port=6000, recv_timeout_sec=1, send_timeout_sec=1),
            data=SimpleNamespace(
                directory="data",
                sampling_rate_hz=100,
                window_size=32,
                stride=4,
                sub_folders=["handheld/"],
                calibration_windows=2048,
            ),
            training=SimpleNamespace(
                quantization="float",
                latency_proxy_max_flops=5_000_000,
                energy_aware=False,  # default sketch variant for unit tests
            ),
            device=SimpleNamespace(hil=True, name="TEST_DEVICE", serial_port="ttyACM0"),
            outputs=SimpleNamespace(tflite_model_path=Path("model.tflite"), tcn_dir=Path("tinyodom_tcn")),
        )

        # Create a dummy dataset with minimal data to simulate OxIODSplitData.
        self.dataset = SimpleNamespace(inputs=np.zeros((1, 32, 6), dtype=np.float32))

        # Patch expensive initialization hooks so we can observe the calls without
        # touching disk or hardware.
        self.load_settings_patcher = patch("src.hil_server.load_config", return_value=self.config)
        self.dataset_patcher = patch("src.hil_server.import_oxiod_dataset", return_value=self.dataset)
        self.context = MagicMock()
        self.socket = MagicMock()
        self.context.socket.return_value = self.socket
        self.zmq_patcher = patch("src.hil_server.zmq.Context.instance", return_value=self.context)

        self.load_settings_mock = self.load_settings_patcher.start()
        self.dataset_mock = self.dataset_patcher.start()
        self.zmq_mock = self.zmq_patcher.start()

    def tearDown(self) -> None:
        patch.stopall()

    def build_server(self) -> HILServer:
        """Return a configured HILServer using the mocked dependencies."""
        return HILServer()


class DetermineMetricsTests(HILServerTestCase):
    """Tests for the conversion + metrics pipeline in `determine_metrics`."""

    def test_conversion_pipeline_invoked_in_order(self) -> None:
        """Building and converting the model should feed into collect_metrics."""
        server = self.build_server()
        fake_model = MagicMock()
        fake_metrics = {"ram_bytes": 1024}

        # Mock the entire pipeline to verify call order and arguments.
        with patch("src.hil_server.build_tinyodom_model", return_value=fake_model) as build_mock, patch(
            "src.hil_server.convert_to_tflite_model"
        ) as to_tflite_mock, patch("src.hil_server.convert_to_cpp_model") as to_cpp_mock, patch(
            "src.hil_server.collect_metrics", return_value=fake_metrics
        ) as collect_mock:
            hyperparams = Dict(flops=123, input_dim=6)
            result = server.determine_metrics(hyperparams)

        # Assert each step in the pipeline was called exactly once with correct args.
        build_mock.assert_called_once_with(hyperparams)
        to_tflite_mock.assert_called_once_with(
            model=fake_model,
            training_data=self.dataset.inputs,
            quantization=self.config.training.quantization,
            output_name=str(self.config.outputs.tflite_model_path),
        )
        to_cpp_mock.assert_called_once_with(
            tflite_path=self.config.outputs.tflite_model_path, output_dir=self.config.outputs.tcn_dir
        )
        collect_mock.assert_called_once()
        self.assertEqual(result, fake_metrics)

    def test_collect_metrics_receives_expected_fields(self) -> None:
        """Key hyperparameters should flow through untouched to the controller."""
        server = self.build_server()
        with patch("src.hil_server.build_tinyodom_model"), patch("src.hil_server.convert_to_tflite_model"), patch(
            "src.hil_server.convert_to_cpp_model"
        ), patch("src.hil_server.collect_metrics", return_value={"ok": True}) as collect_mock:
            hyperparams = Dict(flops=999, input_dim=3)
            server.determine_metrics(hyperparams)

        # Verify that collect_metrics gets the right kwargs from hyperparams and config.
        kwargs = collect_mock.call_args.kwargs
        self.assertEqual(kwargs["flops"], 999)
        self.assertEqual(kwargs["input_dim"], 3)
        self.assertEqual(kwargs["device_name"], self.config.device.name)
        self.assertEqual(kwargs["dirpath"], self.config.outputs.tcn_dir)
        self.assertAlmostEqual(
            kwargs["latency_budget_ms"],
            (self.config.data.stride / self.config.data.sampling_rate_hz) * 1000,
        )


class StartLoopTests(HILServerTestCase):
    """Validate the ZeroMQ REP loop implemented in `start`."""

    def test_start_binds_and_serves_single_request(self) -> None:
        """The server should bind, process one payload, and send a reply."""
        server = self.build_server()
        hyperparams = {"flops": 1, "input_dim": 2}
        metrics = {"flash_bytes": 2048}

        # Mock determine_metrics to return fake metrics, and simulate one request then interrupt.
        server.determine_metrics = MagicMock(return_value=metrics)
        self.socket.recv_json.side_effect = [hyperparams, KeyboardInterrupt()]

        server.start()

        # Verify socket binding, message processing, and cleanup.
        endpoint = f"tcp://{self.config.network.host}:{self.config.network.port}"
        self.socket.bind.assert_called_once_with(endpoint)
        server.determine_metrics.assert_called_once_with(Dict(hyperparams))
        self.socket.send_json.assert_called_once_with(metrics)
        self.socket.close.assert_called_once_with(linger=0)
        self.context.term.assert_called_once()

    def test_start_interrupt_cleans_up_resources(self) -> None:
        """If recv_json immediately raises, we should still close the socket."""
        server = self.build_server()
        self.socket.recv_json.side_effect = KeyboardInterrupt()

        server.start()

        # Ensure no reply sent, but cleanup still happens.
        self.socket.send_json.assert_not_called()
        self.socket.close.assert_called_once_with(linger=0)
        self.context.term.assert_called_once()


class InitializationTests(HILServerTestCase):
    """Ensure constructor wiring calls the data loader with correct inputs."""

    def test_import_dataset_called_with_expected_args(self) -> None:
        """The dataset loader should reflect the OxIOD training split."""
        self.build_server()
        
        # Check that import_oxiod_dataset was called with the right parameters for training data.
        self.dataset_mock.assert_called_once()
        kwargs = self.dataset_mock.call_args.kwargs
        self.assertEqual(kwargs["type_flag"], 2)  # Training split
        self.assertEqual(kwargs["dataset_folder"], self.config.data.directory)
        self.assertEqual(
            kwargs["sub_folders"],
            ['handbag/', 'handheld/', 'pocket/', 'running/', 'slow_walking/', 'trolley/'],  # All subfolders
        )
        self.assertEqual(kwargs["sampling_rate"], self.config.data.sampling_rate_hz)
        self.assertEqual(kwargs["window_size"], self.config.data.window_size)
        self.assertEqual(kwargs["stride"], self.config.data.stride)
        self.assertEqual(kwargs["max_windows"], self.config.data.calibration_windows)


if __name__ == "__main__":
    unittest.main()
