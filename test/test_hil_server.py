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

import hil_server as hil_server_module  # noqa: E402
from hil_server import HILServer  # noqa: E402
from tinyodom.errors import HIL_ERROR_OK  # noqa: E402
from tinyodom.model import CollectMetricsRequest  # noqa: E402


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
        """Return a configured ``HILServer`` using the mocked dependencies.

        Returns
        -------
        HILServer
            Server instance whose heavy filesystem and dataset dependencies have
            already been replaced with lightweight doubles.
        """
        server = HILServer()
        server._sync_sketch_variant = MagicMock(return_value=Path("tinyodom_tcn/tinyodom_tcn.ino"))
        return server


class DetermineMetricsTests(HILServerTestCase):
    """Tests for the conversion + metrics pipeline in `determine_metrics`."""

    def test_conversion_pipeline_invoked_in_order(self) -> None:
        """Backend preparation should feed the request builder and metric collection."""
        server = self.build_server()
        fake_model = MagicMock()
        fake_device = MagicMock()
        fake_device.requires_candidate_model.return_value = True
        fake_device.requires_training_data.return_value = True
        fake_device.supports_runtime_measurement.return_value = True
        fake_device.prepare_candidate.return_value = self.config.outputs.tcn_dir
        fake_metrics = {"ram_bytes": 1024}

        with patch("hil_server.resolve_device_options", return_value=None), patch(
            "hil_server.get_microcontroller_device",
            return_value=fake_device,
        ), patch("hil_server.build_tinyodom_model", return_value=fake_model) as build_mock, patch(
            "hil_server.collect_metrics", return_value=fake_metrics
        ) as collect_mock:
            hyperparams = Dict(flops=123, input_dim=6)
            result = server.determine_metrics(hyperparams)

        build_mock.assert_called_once_with(hyperparams)
        fake_device.prepare_candidate.assert_called_once()
        prepare_kwargs = fake_device.prepare_candidate.call_args.kwargs
        self.assertIs(prepare_kwargs["model"], fake_model)
        self.assertIs(prepare_kwargs["training_data"], self.dataset)
        collect_mock.assert_called_once()
        self.assertEqual(result, fake_metrics)

    def test_collect_metrics_receives_expected_fields(self) -> None:
        """Key hyperparameters should flow through untouched to the controller."""
        server = self.build_server()
        fake_device = MagicMock()
        fake_device.requires_candidate_model.return_value = True
        fake_device.requires_training_data.return_value = True
        fake_device.supports_runtime_measurement.return_value = True
        fake_device.prepare_candidate.return_value = self.config.outputs.tcn_dir
        with patch("hil_server.resolve_device_options", return_value=None), patch(
            "hil_server.get_microcontroller_device",
            return_value=fake_device,
        ), patch("hil_server.build_tinyodom_model"), patch(
            "hil_server.collect_metrics", return_value={"ok": True}
        ) as collect_mock:
            hyperparams = Dict(flops=999, input_dim=3)
            server.determine_metrics(hyperparams)

        # Verify that collect_metrics gets a normalized request object.
        request = collect_mock.call_args.args[0]
        self.assertEqual(request.flops, 999)
        self.assertEqual(request.input_dim, 3)
        self.assertEqual(request.device_name, self.config.device.name)
        self.assertEqual(request.dirpath, self.config.outputs.tcn_dir.resolve())
        self.assertAlmostEqual(
            request.latency_budget_ms,
            (self.config.data.stride / self.config.data.sampling_rate_hz) * 1000,
        )

    def test_collect_metrics_uses_device_latency_budget_override(self) -> None:
        server = self.build_server()
        self.config.device.latency_budget_ms = 75.0
        fake_device = MagicMock()
        fake_device.requires_candidate_model.return_value = True
        fake_device.requires_training_data.return_value = True
        fake_device.supports_runtime_measurement.return_value = True
        fake_device.prepare_candidate.return_value = self.config.outputs.tcn_dir

        with patch("hil_server.resolve_device_options", return_value=None), patch(
            "hil_server.get_microcontroller_device",
            return_value=fake_device,
        ), patch("hil_server.build_tinyodom_model"), patch(
            "hil_server.collect_metrics", return_value={"ok": True}
        ) as collect_mock:
            server.determine_metrics(Dict(flops=999, input_dim=3))

        request = collect_mock.call_args.args[0]
        self.assertEqual(request.latency_budget_ms, 75.0)

    def test_determine_metrics_runs_arduino_cadenced_second_pass_after_successful_base_run(self) -> None:
        server = self.build_server()
        self.config.device.name = "ARDUINO_NANO_33_BLE_SENSE"
        self.config.device.runtime_mode = "cadenced"
        self.config.training.energy_aware = True
        self.config.training.input_mode = "uniform"
        fake_device = MagicMock()
        fake_device.requires_candidate_model.return_value = True
        fake_device.requires_training_data.return_value = True
        fake_device.supports_runtime_measurement.return_value = True
        fake_device.supports_energy_measurement.return_value = True
        fake_device.prepare_candidate.return_value = self.config.outputs.tcn_dir
        fake_device.evaluate.return_value = SimpleNamespace(
            error_code=HIL_ERROR_OK,
            power_metrics={"energy_mj_per_inference": 1.5},
        )

        with patch("hil_server.resolve_device_options", return_value=None), patch(
            "hil_server.get_microcontroller_device",
            return_value=fake_device,
        ), patch("hil_server.build_tinyodom_model"), patch(
            "hil_server.collect_metrics",
            return_value={
                "error_code": HIL_ERROR_OK,
                "arena_bytes": 4096,
                "runtime_mode": "back_to_back",
                "cadenced_error_code": -1,
                "cadenced_error_label": None,
                "cadenced_active_inference_latency_ms": -1.0,
                "cadenced_energy_mj_per_inference": -1.0,
                "cadenced_energy_mj_per_window": -1.0,
            },
        ):
            metrics = server.determine_metrics(Dict(flops=123, input_dim=6))

        fake_device.set_input_mode.assert_called_once()
        self.assertEqual(fake_device.set_input_mode.call_args.kwargs["runtime_phase"], "cadenced")
        fake_device.evaluate.assert_called_once()
        self.assertEqual(metrics["runtime_mode"], "cadenced")
        self.assertEqual(metrics["cadenced_error_code"], HIL_ERROR_OK)
        self.assertAlmostEqual(metrics["cadenced_energy_mj_per_inference"], 1.5)
        self.assertAlmostEqual(metrics["cadenced_energy_mj_per_window"], 15.0)

    def test_determine_metrics_discards_arduino_cadenced_second_pass_latency(self) -> None:
        server = self.build_server()
        self.config.device.name = "ARDUINO_NANO_33_BLE_SENSE"
        self.config.device.runtime_mode = "cadenced"
        self.config.training.energy_aware = True
        self.config.training.input_mode = "uniform"
        fake_device = MagicMock()
        fake_device.requires_candidate_model.return_value = True
        fake_device.requires_training_data.return_value = True
        fake_device.supports_runtime_measurement.return_value = True
        fake_device.supports_energy_measurement.return_value = True
        fake_device.prepare_candidate.return_value = self.config.outputs.tcn_dir
        fake_device.evaluate.return_value = SimpleNamespace(
            error_code=HIL_ERROR_OK,
            latency_s=0.2,
            power_metrics={"energy_mj_per_inference": 2.0},
        )

        with patch("hil_server.resolve_device_options", return_value=None), patch(
            "hil_server.get_microcontroller_device",
            return_value=fake_device,
        ), patch("hil_server.build_tinyodom_model"), patch(
            "hil_server.collect_metrics",
            return_value={
                "error_code": HIL_ERROR_OK,
                "arena_bytes": 4096,
                "runtime_mode": "back_to_back",
                "cadenced_error_code": -1,
                "cadenced_error_label": None,
                "cadenced_active_inference_latency_ms": -1.0,
                "cadenced_energy_mj_per_inference": -1.0,
                "cadenced_energy_mj_per_window": -1.0,
            },
        ):
            metrics = server.determine_metrics(Dict(flops=123, input_dim=6))

        self.assertEqual(metrics["cadenced_active_inference_latency_ms"], -1.0)

    def test_determine_metrics_uses_override_clock_for_runtime_options_only(self) -> None:
        server = self.build_server()
        fake_device = MagicMock()
        fake_device.requires_candidate_model.return_value = True
        fake_device.requires_training_data.return_value = True
        fake_device.supports_runtime_measurement.return_value = True
        fake_device.prepare_candidate.return_value = self.config.outputs.tcn_dir
        fake_metrics = {"ram_bytes": 1024, "clock_hz": 400000000.0}

        with patch("hil_server.resolve_device_options", return_value={"cpu_clock_mhz": 600}), patch(
            "hil_server.get_microcontroller_device",
            return_value=fake_device,
        ) as device_mock, patch("hil_server.build_tinyodom_model"), patch(
            "hil_server.collect_metrics", return_value=fake_metrics
        ) as collect_mock:
            hyperparams = Dict(flops=123, input_dim=6)
            result = server.determine_metrics(
                hyperparams,
                device_options_overrides={"cpu_clock_mhz": 400},
            )

        self.assertEqual(result["cpu_clock_mhz_requested"], 400)
        self.assertEqual(device_mock.call_args.kwargs["device_options"]["cpu_clock_mhz"], 400)
        request = collect_mock.call_args.args[0]
        self.assertEqual(request.device_options["cpu_clock_mhz"], 400)

    def test_determine_metrics_unsampled_stm_clock_logs_minus_one(self) -> None:
        server = self.build_server()
        self.config.device.name = "STM32_NUCLEO_N657X0_Q"
        fake_device = MagicMock()
        fake_device.requires_candidate_model.return_value = True
        fake_device.requires_training_data.return_value = True
        fake_device.supports_runtime_measurement.return_value = True
        fake_device.prepare_candidate.return_value = self.config.outputs.tcn_dir

        with patch("hil_server.resolve_device_options", return_value={"cpu_clock_mhz": 600}), patch(
            "hil_server.get_microcontroller_device",
            return_value=fake_device,
        ), patch("hil_server.build_tinyodom_model"), patch(
            "hil_server.collect_metrics", return_value={"ram_bytes": 1024, "clock_hz": 600000000.0}
        ):
            metrics = server.determine_metrics(Dict(flops=1, input_dim=6))

        self.assertEqual(metrics["cpu_clock_mhz_requested"], -1)

    def test_determine_metrics_invalid_clock_override_returns_request_error(self) -> None:
        server = self.build_server()

        with patch("hil_server.resolve_device_options", return_value={"cpu_clock_mhz": 600}), patch(
            "hil_server.get_microcontroller_device"
        ) as device_mock:
            metrics = server.determine_metrics(
                Dict(flops=123, input_dim=6),
                device_options_overrides={"cpu_clock_mhz": float("nan")},
            )

        device_mock.assert_not_called()
        self.assertEqual(metrics["error_code"], hil_server_module.HIL_MASTER_FATAL)
        self.assertEqual(metrics["backend_error_kind"], "request")
        self.assertIn("device_options_overrides.cpu_clock_mhz", metrics["backend_error_detail"])


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
        server.determine_metrics.assert_called_once_with(Dict(hyperparams), device_options_overrides=None)
        self.socket.send_json.assert_called_once_with(metrics)
        self.socket.close.assert_called_once_with(linger=0)
        self.context.term.assert_called_once()

    def test_start_normalizes_structured_payload(self) -> None:
        server = self.build_server()
        payload = {
            "hyperparams": {"flops": 1, "input_dim": 2},
            "device_options_overrides": {"cpu_clock_mhz": 400},
        }
        metrics = {"flash_bytes": 2048}
        server.determine_metrics = MagicMock(return_value=metrics)
        self.socket.recv_json.side_effect = [payload, KeyboardInterrupt()]

        server.start()

        server.determine_metrics.assert_called_once_with(
            Dict(payload["hyperparams"]),
            device_options_overrides={"cpu_clock_mhz": 400},
        )

    def test_start_returns_request_error_for_malformed_payload(self) -> None:
        server = self.build_server()
        payload = {
            "hyperparams": None,
            "device_options_overrides": {"cpu_clock_mhz": 400},
        }
        self.socket.recv_json.side_effect = [payload, KeyboardInterrupt()]

        server.start()

        self.socket.send_json.assert_called_once()
        metrics = self.socket.send_json.call_args.args[0]
        self.assertEqual(metrics["error_code"], hil_server_module.HIL_MASTER_FATAL)
        self.assertEqual(metrics["backend_error_kind"], "request")
        self.assertIn("hyperparams", metrics["backend_error_detail"])

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
    """Ensure constructor wiring defers dataset loading until needed."""

    def test_import_dataset_is_lazy_until_training_data_is_needed(self) -> None:
        """Ensure the constructor does not eagerly load calibration data.

        Returns
        -------
        None
        """
        server = self.build_server()
        fake_device = MagicMock()
        fake_device.requires_candidate_model.return_value = True
        fake_device.requires_training_data.return_value = True
        fake_device.supports_runtime_measurement.return_value = True
        fake_device.prepare_candidate.return_value = self.config.outputs.tcn_dir

        self.dataset_mock.assert_not_called()

        with patch("hil_server.resolve_device_options", return_value=None), patch(
            "hil_server.get_microcontroller_device",
            return_value=fake_device,
        ), patch("hil_server.build_tinyodom_model"), patch(
            "hil_server.collect_metrics", return_value={"ok": True}
        ):
            server.determine_metrics(Dict(flops=1, input_dim=6))

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

    def test_stm_runtime_backend_keeps_hil_enabled_when_supported(self) -> None:
        """Ensure STM runtime-capable backends keep HIL enabled.

        Returns
        -------
        None
        """
        self.config.device.name = "STM32_NUCLEO_N657X0_Q"
        self.config.device.stm32 = SimpleNamespace(project_root=Path("/tmp/stm_fsbl"))
        server = self.build_server()
        fake_device = MagicMock()
        fake_device.requires_candidate_model.return_value = True
        fake_device.requires_training_data.return_value = True
        fake_device.supports_runtime_measurement.return_value = True
        fake_device.supports_energy_measurement.return_value = True
        fake_device.prepare_candidate.return_value = Path("/tmp/stm_fsbl")

        with patch("hil_server.resolve_device_options", return_value={"project_root": Path("/tmp/stm_fsbl")}), patch(
            "hil_server.get_microcontroller_device",
            return_value=fake_device,
        ), patch("hil_server.build_collect_metrics_request") as request_mock, patch(
            "hil_server.collect_metrics",
            return_value={"ok": True},
        ) as collect_mock, patch("hil_server.build_tinyodom_model") as build_mock:
            request_mock.return_value = CollectMetricsRequest(
                hil_enabled=True,
                energy_aware=False,
                flops=1,
                device_name="STM32_NUCLEO_N657X0_Q",
                window_size=32,
                input_dim=6,
                dirpath=Path("/tmp/stm_fsbl"),
                latency_proxy_max_flops=5_000_000,
                serial_port="ttyACM0",
                latency_budget_ms=40.0,
                dut_ready_timeout_s=5.0,
                serial_timeout_s=12.0,
                harness=None,
                device_options={},
            )
            result = server.determine_metrics(Dict(flops=1, input_dim=6))

        self.assertEqual(result, {"ok": True, "cpu_clock_mhz_requested": -1, "latency_budget_ms": 40.0})
        self.dataset_mock.assert_called_once()
        build_mock.assert_called_once()
        fake_device.prepare_candidate.assert_called_once()
        fake_device.cleanup_prepared_candidate.assert_called_once_with(Path("/tmp/stm_fsbl"))
        collect_mock.assert_called_once()
        self.assertTrue(request_mock.call_args.kwargs["hil_enabled"])
        self.assertIsNone(server.active_sketch_path)

    def test_stm_backend_keeps_energy_aware_requests_when_supported(self) -> None:
        """Ensure STM energy-aware requests remain enabled once the backend supports them.

        Returns
        -------
        None
        """
        self.config.training.energy_aware = True
        self.config.device.name = "STM32_NUCLEO_N657X0_Q"
        self.config.device.stm32 = SimpleNamespace(project_root=Path("/tmp/stm_fsbl"))
        server = self.build_server()
        fake_device = MagicMock()
        fake_device.requires_candidate_model.return_value = True
        fake_device.requires_training_data.return_value = True
        fake_device.supports_runtime_measurement.return_value = True
        fake_device.supports_energy_measurement.return_value = True
        fake_device.prepare_candidate.return_value = Path("/tmp/stm_fsbl")

        with patch(
            "hil_server.resolve_device_options",
            return_value={"project_root": Path("/tmp/stm_fsbl")},
        ), patch(
            "hil_server.get_microcontroller_device",
            return_value=fake_device,
        ), patch(
            "hil_server.collect_metrics",
            return_value={"ok": True},
        ), patch(
            "hil_server.build_collect_metrics_request"
        ) as request_mock, patch(
            "hil_server.build_tinyodom_model"
        ):
            request_mock.return_value = CollectMetricsRequest(
                hil_enabled=True,
                energy_aware=True,
                flops=1,
                device_name="STM32_NUCLEO_N657X0_Q",
                window_size=32,
                input_dim=6,
                dirpath=Path("/tmp/stm_fsbl"),
                latency_proxy_max_flops=5_000_000,
                serial_port="ttyACM0",
                latency_budget_ms=40.0,
                dut_ready_timeout_s=5.0,
                serial_timeout_s=12.0,
                harness=None,
                device_options={},
            )
            server.determine_metrics(Dict(flops=1, input_dim=6))

        self.assertTrue(request_mock.call_args.kwargs["energy_aware"])

    def test_arduino_and_stm_candidate_staging_diverge_at_active_sketch_boundary(self) -> None:
        """Ensure Arduino candidate prep activates a sketch while STM keeps project staging.

        Returns
        -------
        None
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            arduino_server = self.build_server()
            arduino_server.active_sketch_path = None
            arduino_prepared_dir = tmp_path / "arduino"
            arduino_prepared_dir.mkdir()
            expected_sketch = arduino_prepared_dir / "tinyodom_tcn.ino"
            expected_sketch.write_text("// staged sketch\n", encoding="utf-8")

            fake_arduino_device = MagicMock()
            fake_arduino_device.requires_candidate_model.return_value = True
            fake_arduino_device.requires_training_data.return_value = False
            fake_arduino_device.supports_runtime_measurement.return_value = False
            fake_arduino_device.supports_energy_measurement.return_value = False
            fake_arduino_device.prepare_candidate.return_value = arduino_prepared_dir

            with patch("hil_server.resolve_device_options", return_value=None), patch(
                "hil_server.get_microcontroller_device",
                return_value=fake_arduino_device,
            ), patch("hil_server.build_tinyodom_model"), patch(
                "hil_server.build_collect_metrics_request"
            ) as arduino_request_mock, patch(
                "hil_server.collect_metrics",
                return_value={"ok": True},
            ):
                arduino_request_mock.return_value = CollectMetricsRequest(
                    hil_enabled=False,
                    energy_aware=False,
                    flops=1,
                    device_name="ARDUINO_NANO_33_BLE_SENSE",
                    window_size=32,
                    input_dim=6,
                    dirpath=arduino_prepared_dir,
                    latency_proxy_max_flops=5_000_000,
                    serial_port="ttyACM0",
                    device_options=None,
                )
                arduino_server.determine_metrics(Dict(flops=1, input_dim=6))

            self.assertEqual(arduino_server.active_sketch_path, expected_sketch)

            self.config.device.name = "STM32_NUCLEO_N657X0_Q"
            self.config.device.stm32 = SimpleNamespace(project_root=tmp_path / "stm32" / "FSBL")
            stm_server = self.build_server()
            stm_server.active_sketch_path = None
            stm_prepared_dir = tmp_path / "stm32" / "FSBL"
            stm_prepared_dir.mkdir(parents=True)

            fake_stm_device = MagicMock()
            fake_stm_device.requires_candidate_model.return_value = True
            fake_stm_device.requires_training_data.return_value = False
            fake_stm_device.supports_runtime_measurement.return_value = False
            fake_stm_device.supports_energy_measurement.return_value = False
            fake_stm_device.prepare_candidate.return_value = stm_prepared_dir

            with patch(
                "hil_server.resolve_device_options",
                return_value={"project_root": stm_prepared_dir},
            ), patch(
                "hil_server.get_microcontroller_device",
                return_value=fake_stm_device,
            ), patch("hil_server.build_tinyodom_model"), patch(
                "hil_server.build_collect_metrics_request"
            ) as stm_request_mock, patch(
                "hil_server.collect_metrics",
                return_value={"ok": True},
            ):
                stm_request_mock.return_value = CollectMetricsRequest(
                    hil_enabled=False,
                    energy_aware=False,
                    flops=1,
                    device_name="STM32_NUCLEO_N657X0_Q",
                    window_size=32,
                    input_dim=6,
                    dirpath=stm_prepared_dir,
                    latency_proxy_max_flops=5_000_000,
                    serial_port="ttyACM0",
                    device_options={"project_root": stm_prepared_dir},
                )
                stm_server.determine_metrics(Dict(flops=1, input_dim=6))

            self.assertIsNone(stm_server.active_sketch_path)

    def test_stm_prepare_candidate_failure_returns_structured_backend_metrics(self) -> None:
        """STM staging failures should become metrics-shaped backend errors."""
        self.config.device.name = "STM32_NUCLEO_N657X0_Q"
        self.config.device.stm32 = SimpleNamespace(project_root=Path("/tmp/stm_fsbl"))
        server = self.build_server()
        fake_device = MagicMock()
        fake_device.requires_candidate_model.return_value = True
        fake_device.requires_training_data.return_value = True
        fake_device.supports_runtime_measurement.return_value = True
        fake_device.supports_energy_measurement.return_value = True
        fake_device.prepare_candidate.side_effect = hil_server_module.stm32_cube_clt.WorkflowError(
            "ST Edge AI generation failed: unsupported operator"
        )

        with patch("hil_server.resolve_device_options", return_value={"project_root": Path("/tmp/stm_fsbl")}), patch(
            "hil_server.get_microcontroller_device",
            return_value=fake_device,
        ), patch("hil_server.build_tinyodom_model"):
            metrics = server.determine_metrics(Dict(flops=1, input_dim=6))

        self.assertEqual(metrics["backend_error_kind"], "unsupported_model")
        self.assertIn("unsupported operator", metrics["backend_error_detail"])

    def test_request_build_failure_returns_structured_backend_metrics(self) -> None:
        """Request-building config failures should not crash the REP loop."""
        self.config.training.energy_aware = True
        server = self.build_server()
        fake_device = MagicMock()
        fake_device.requires_candidate_model.return_value = True
        fake_device.requires_training_data.return_value = True
        fake_device.supports_runtime_measurement.return_value = True
        fake_device.supports_energy_measurement.return_value = True
        fake_device.prepare_candidate.return_value = self.config.outputs.tcn_dir

        with patch("hil_server.resolve_device_options", return_value=None), patch(
            "hil_server.get_microcontroller_device",
            return_value=fake_device,
        ), patch("hil_server.build_tinyodom_model"), patch(
            "hil_server.build_collect_metrics_request",
            side_effect=RuntimeError(
                "Set device.harness_serial_port when runtime measurement requires the harness."
            ),
        ):
            metrics = server.determine_metrics(Dict(flops=1, input_dim=6))

        self.assertEqual(metrics["error_code"], hil_server_module.HIL_MASTER_FATAL)
        self.assertEqual(metrics["backend_error_kind"], "config")
        self.assertIn("device.harness_serial_port", metrics["backend_error_detail"])
        fake_device.cleanup_prepared_candidate.assert_called_once_with(self.config.outputs.tcn_dir)

    def test_set_input_mode_delegates_to_backend_for_stm_phase1(self) -> None:
        """Ensure STM servers delegate input-mode changes to the backend.

        Returns
        -------
        None
        """
        self.config.device.name = "STM32_NUCLEO_N657X0_Q"
        self.config.device.stm32 = SimpleNamespace(project_root=Path("/tmp/stm_fsbl"))
        server = self.build_server()
        fake_device = MagicMock()
        fake_device.set_input_mode.return_value = Path("/tmp/stm_fsbl")

        with patch("hil_server.resolve_device_options", return_value={"project_root": Path("/tmp/stm_fsbl")}), patch(
            "hil_server.get_microcontroller_device",
            return_value=fake_device,
        ):
            out_path = server.set_input_mode("uniform")

        self.assertEqual(out_path, Path("/tmp/stm_fsbl"))


class SketchVariantTests(unittest.TestCase):
    """Validate sketch variant selection and input-mode behaviors."""

    def _build_server(
        self,
        sketches_dir: Path,
        tcn_dir: Path,
        energy_aware: bool,
        input_mode: str,
        *,
        device_name: str = "ARDUINO_NANO_33_BLE_SENSE",
        target_core: str | None = None,
    ) -> HILServer:
        """Build a lightweight ``HILServer`` double for sketch-selection tests.

        Parameters
        ----------
        sketches_dir : Path
            Root sketches directory used by the selector under test.
        tcn_dir : Path
            Active output sketch directory.
        energy_aware : bool
            Whether energy-aware sketch variants should be selected.
        input_mode : str
            Requested input mode (uniform/representative/real).
        device_name : str, optional
            Device profile used to route uniform sketch variants.
        target_core : str | None, optional
            Optional Portenta target core (``cm7``/``cm4``).

        Returns
        -------
        HILServer
            Server instance with just enough config for variant sync tests.
        """
        server = HILServer.__new__(HILServer)
        portenta_cfg = (
            SimpleNamespace(target_core=target_core) if target_core is not None else SimpleNamespace()
        )
        server.config = SimpleNamespace(
            training=SimpleNamespace(energy_aware=energy_aware, input_mode=input_mode),
            outputs=SimpleNamespace(tcn_dir=tcn_dir),
            device=SimpleNamespace(name=device_name, portenta=portenta_cfg),
        )
        server.sketch_variants_dir = sketches_dir
        server.active_sketch_path = None
        return server

    def _write_sketch(self, path: Path, label: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"// {label}\n")

    def test_selects_uniform_energy_sketch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sketches = Path(tmpdir) / "sketches"
            tcn_dir = Path(tmpdir) / "tinyodom_tcn"
            self._write_sketch(sketches / "tinyodom_tcn_energy.ino", "uniform_shared")
            server = self._build_server(sketches, tcn_dir, energy_aware=True, input_mode="uniform")

            out_path = server._sync_sketch_variant()

            self.assertTrue(out_path.exists())
            self.assertIn("uniform_shared", out_path.read_text())

    def test_selects_cadenced_uniform_sketch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sketches = Path(tmpdir) / "sketches"
            tcn_dir = Path(tmpdir) / "tinyodom_tcn"
            self._write_sketch(sketches / "tinyodom_tcn_energy_cadenced.ino", "cadenced_shared")
            server = self._build_server(sketches, tcn_dir, energy_aware=True, input_mode="uniform")

            out_path = server.set_input_mode("uniform", runtime_phase="cadenced")

            self.assertTrue(out_path.exists())
            self.assertIn("cadenced_shared", out_path.read_text())

    def test_selects_uniform_energy_sketch_for_portenta_cm7(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sketches = Path(tmpdir) / "sketches"
            tcn_dir = Path(tmpdir) / "tinyodom_tcn"
            self._write_sketch(sketches / "tinyodom_tcn_energy.ino", "uniform_shared_cm7")
            server = self._build_server(
                sketches,
                tcn_dir,
                energy_aware=True,
                input_mode="uniform",
                device_name="PORTENTA_H7",
                target_core="cm7",
            )

            out_path = server._sync_sketch_variant()

            self.assertTrue(out_path.exists())
            self.assertIn("uniform_shared_cm7", out_path.read_text())

    def test_selects_uniform_no_energy_sketch_for_portenta_cm4(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sketches = Path(tmpdir) / "sketches"
            tcn_dir = Path(tmpdir) / "tinyodom_tcn"
            self._write_sketch(sketches / "tinyodom_tcn_no_energy.ino", "no_energy_shared_cm4")
            server = self._build_server(
                sketches,
                tcn_dir,
                energy_aware=False,
                input_mode="uniform",
                device_name="PORTENTA_H7",
                target_core="cm4",
            )

            out_path = server._sync_sketch_variant()

            self.assertTrue(out_path.exists())
            self.assertIn("no_energy_shared_cm4", out_path.read_text())

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

    def test_cadenced_runtime_requires_uniform_input_mode(self) -> None:
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

            with self.assertRaises(ValueError):
                server.set_input_mode("representative", runtime_phase="cadenced")

    def test_portenta_uniform_requires_target_core(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sketches = Path(tmpdir) / "sketches"
            tcn_dir = Path(tmpdir) / "tinyodom_tcn"
            server = self._build_server(
                sketches,
                tcn_dir,
                energy_aware=True,
                input_mode="uniform",
                device_name="PORTENTA_H7",
                target_core=None,
            )

            with self.assertRaises(ValueError):
                server._sync_sketch_variant()

    def test_energy_aware_false_uses_no_energy_sketch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sketches = Path(tmpdir) / "sketches"
            tcn_dir = Path(tmpdir) / "tinyodom_tcn"
            self._write_sketch(sketches / "tinyodom_tcn_no_energy.ino", "no_energy_shared")
            server = self._build_server(sketches, tcn_dir, energy_aware=False, input_mode="uniform")

            out_path = server._sync_sketch_variant()

            self.assertTrue(out_path.exists())
            self.assertIn("no_energy_shared", out_path.read_text())

    def test_set_input_mode_updates_config_and_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sketches = Path(tmpdir) / "sketches"
            tcn_dir = Path(tmpdir) / "tinyodom_tcn"
            self._write_sketch(sketches / "tinyodom_tcn_energy.ino", "uniform_shared")
            server = self._build_server(sketches, tcn_dir, energy_aware=True, input_mode="uniform")

            out_path = server.set_input_mode("uniform")

            self.assertEqual(server.config.training.input_mode, "uniform")
            self.assertTrue(out_path.exists())


if __name__ == "__main__":
    unittest.main()
