import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from addict import Dict

# Ensure `src` is importable whenever this test module is executed via
# `python -m unittest`.
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hil_server import HILServer  # noqa: E402


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
        self.load_settings_patcher = patch("hil_server.load_config", return_value=self.config)
        self.dataset_patcher = patch("hil_server.import_oxiod_dataset", return_value=self.dataset)
        self.context = MagicMock()
        self.socket = MagicMock()
        self.context.socket.return_value = self.socket
        self.zmq_patcher = patch("hil_server.zmq.Context.instance", return_value=self.context)

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
        with patch("hil_server.build_tinyodom_model", return_value=fake_model) as build_mock, patch(
            "hil_server.convert_to_tflite_model"
        ) as to_tflite_mock, patch("hil_server.convert_to_cpp_model") as to_cpp_mock, patch(
            "hil_server.collect_metrics", return_value=fake_metrics
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
        with patch("hil_server.build_tinyodom_model"), patch("hil_server.convert_to_tflite_model"), patch(
            "hil_server.convert_to_cpp_model"
        ), patch("hil_server.collect_metrics", return_value={"ok": True}) as collect_mock:
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


class SketchVariantTests(unittest.TestCase):
    """Validate sketch variant selection and input-mode behaviors."""

    def _build_server(self, sketches_dir: Path, tcn_dir: Path, energy_aware: bool, input_mode: str) -> HILServer:
        server = HILServer.__new__(HILServer)
        server.config = SimpleNamespace(
            training=SimpleNamespace(energy_aware=energy_aware, input_mode=input_mode),
            outputs=SimpleNamespace(tcn_dir=tcn_dir),
        )
        server.sketch_variants_dir = sketches_dir
        server.active_sketch_path = None
        return server

    def _write_sketch(self, path: Path, label: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"// {label}\n")

    def test_selects_standard_energy_sketch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sketches = Path(tmpdir) / "sketches"
            tcn_dir = Path(tmpdir) / "tinyodom_tcn"
            self._write_sketch(sketches / "tinyodom_tcn_energy.ino", "standard")
            server = self._build_server(sketches, tcn_dir, energy_aware=True, input_mode="standard")

            out_path = server._sync_sketch_variant()

            self.assertTrue(out_path.exists())
            self.assertIn("standard", out_path.read_text())

    def test_selects_uniform_variant(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sketches = Path(tmpdir) / "sketches"
            tcn_dir = Path(tmpdir) / "tinyodom_tcn"
            self._write_sketch(sketches / "analysis_sketches/tinyodom_tcn_energy_uniform.ino", "uniform")
            server = self._build_server(sketches, tcn_dir, energy_aware=True, input_mode="uniform")

            out_path = server._sync_sketch_variant()

            self.assertTrue(out_path.exists())
            self.assertIn("uniform", out_path.read_text())

    def test_selects_representative_variant_and_copies_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sketches = Path(tmpdir) / "sketches"
            tcn_dir = Path(tmpdir) / "tinyodom_tcn"
            self._write_sketch(
                sketches / "analysis_sketches/tinyodom_tcn_energy_representative.ino",
                "representative",
            )
            header = sketches / "analysis_sketches/tinyodom_tcn_input_data.h"
            header.write_text("// header\n")
            server = self._build_server(sketches, tcn_dir, energy_aware=True, input_mode="representative")

            out_path = server._sync_sketch_variant()

            self.assertTrue(out_path.exists())
            self.assertIn("representative", out_path.read_text())
            self.assertTrue((tcn_dir / header.name).exists())

    def test_selects_real_variant_and_copies_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sketches = Path(tmpdir) / "sketches"
            tcn_dir = Path(tmpdir) / "tinyodom_tcn"
            self._write_sketch(
                sketches / "analysis_sketches/tinyodom_tcn_energy_real_data.ino",
                "real",
            )
            header = sketches / "analysis_sketches/tinyodom_tcn_input_data.h"
            header.write_text("// header\n")
            server = self._build_server(sketches, tcn_dir, energy_aware=True, input_mode="real")

            out_path = server._sync_sketch_variant()

            self.assertTrue(out_path.exists())
            self.assertIn("real", out_path.read_text())
            self.assertTrue((tcn_dir / header.name).exists())

    def test_missing_header_raises_for_representative(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sketches = Path(tmpdir) / "sketches"
            tcn_dir = Path(tmpdir) / "tinyodom_tcn"
            self._write_sketch(
                sketches / "analysis_sketches/tinyodom_tcn_energy_representative.ino",
                "representative",
            )
            server = self._build_server(sketches, tcn_dir, energy_aware=True, input_mode="representative")

            with self.assertRaises(FileNotFoundError):
                server._sync_sketch_variant()

    def test_invalid_input_mode_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sketches = Path(tmpdir) / "sketches"
            tcn_dir = Path(tmpdir) / "tinyodom_tcn"
            server = self._build_server(sketches, tcn_dir, energy_aware=True, input_mode="bad_mode")

            with self.assertRaises(ValueError):
                server._sync_sketch_variant()

    def test_energy_aware_false_uses_no_energy_sketch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sketches = Path(tmpdir) / "sketches"
            tcn_dir = Path(tmpdir) / "tinyodom_tcn"
            self._write_sketch(sketches / "tinyodom_tcn_no_energy.ino", "no_energy")
            server = self._build_server(sketches, tcn_dir, energy_aware=False, input_mode="standard")

            out_path = server._sync_sketch_variant()

            self.assertTrue(out_path.exists())
            self.assertIn("no_energy", out_path.read_text())

    def test_set_input_mode_updates_config_and_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sketches = Path(tmpdir) / "sketches"
            tcn_dir = Path(tmpdir) / "tinyodom_tcn"
            self._write_sketch(sketches / "analysis_sketches/tinyodom_tcn_energy_uniform.ino", "uniform")
            server = self._build_server(sketches, tcn_dir, energy_aware=True, input_mode="standard")

            out_path = server.set_input_mode("uniform")

            self.assertEqual(server.config.training.input_mode, "uniform")
            self.assertTrue(out_path.exists())


if __name__ == "__main__":
    unittest.main()
