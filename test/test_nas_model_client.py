import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import optuna
import zmq
from optuna.trial import TrialState

# Ensure `src` is importable when the suite is launched via `python -m unittest`.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# The NASModelClient is the unit under test for this module.
from src.nas_model_client import NASModelClient  # noqa: E402
from src.hardware_utils_ex import (
    HIL_MASTER_DEVICE_NOT_FOUND,
    HIL_MASTER_FLASH_OVERFLOW,
    HIL_MASTER_RAM_OVERFLOW,
    HIL_MASTER_FATAL,
    HIL_MASTER_SUCCESS,
)  # noqa: E402


def _build_test_client(base_dir: Path | None = None) -> NASModelClient:
    """Construct a NASModelClient using lightweight stand-ins for config/data.

    The real class performs heavy dataset loading inside __init__. To keep the
    tests fast and deterministic, we bypass __init__ and fill in the handful of
    attributes the logic depends on.
    """
    client = NASModelClient.__new__(NASModelClient)
    base_dir = Path(tempfile.mkdtemp()) if base_dir is None else Path(base_dir)

    window_size = 16
    input_dim = 3
    # Mimic the (batch, timesteps, dim) tensor structure Optuna expects.
    dataset = SimpleNamespace(
        inputs=np.zeros((2, window_size, input_dim), dtype=np.float32),
        x_vel=np.zeros((2, 1), dtype=np.float32),
        y_vel=np.zeros((2, 1), dtype=np.float32),
        size_of_each=[2],
        x0=[0.0],
        y0=[0.0],
    )

    # Build a SimpleNamespace tree that mirrors the configuration object used in
    # production so attribute lookups behave the same.
    client.config = SimpleNamespace(
        network=SimpleNamespace(host="localhost", port=5555, recv_timeout_sec=5, send_timeout_sec=5),
        data=SimpleNamespace(
            directory="data",
            sampling_rate_hz=100,
            window_size=window_size,
            stride=1,
        ),
        training=SimpleNamespace(
            drop_rate_choices=[0.1, 0.2],
            train=True,
            nas_epochs=10,
            quantization="float",
            latency_proxy_max_flops=1_000_000,
            nas_trials=2,
            max_total_trials=4,
            nas_multiobjective=False,
            energy_aware=False,
            nas_multiobjective_population_size=8,
        ),
        device=SimpleNamespace(name="TEST_DEVICE", hil=True, serial_port="ttyACM0"),
        outputs=SimpleNamespace(
            log_file_name="test_log.csv",
            tflite_model_path=base_dir / "model.tflite",
            tcn_dir=base_dir / "tinyodom_tcn",
            models_dir=base_dir / "models",
            checkpoint_path=base_dir / "model.keras",
        ),
    )
    client.config.outputs.models_dir.mkdir(parents=True, exist_ok=True)
    client.config_path = base_dir / "config.yaml"
    # Reuse the placeholder dataset wherever the client expects training/val/test.
    client.training_data = dataset
    client.validation_data = dataset
    client.test_data = dataset
    # Mirror the production default study name so log_trial calls succeed.
    client.study_name = "default_study"
    # Stub the ZMQ context/socket to avoid opening real network resources.
    client.socket = MagicMock()
    client.context = MagicMock()
    return client


class DummyTrial:
    """Trivial Optuna Trial substitute that records suggestions and reports."""

    def __init__(self) -> None:
        self.report_calls = []
        self.params = {}
        self.user_attrs = {}

    def suggest_int(self, name, low, high):
        value = low
        self.params[name] = value
        return value

    def suggest_categorical(self, name, choices):
        value = choices[0]
        self.params[name] = value
        return value

    def report(self, value, step):
        self.report_calls.append((value, step))

    def set_user_attr(self, key, value):
        self.user_attrs[key] = value


class HILRequestTests(unittest.TestCase):
    """Validate the ZeroMQ request/response helper."""

    def test_hil_request_success(self) -> None:
        """A successful round-trip should return the parsed metrics dict.

        Instead of booting a real server we just drive the mock socket and
        assert that the client sends/receives JSON exactly once.
        """
        client = _build_test_client()
        metrics = {"ram_bytes": 1024}
        client.socket.recv_json.return_value = metrics

        hyperparams = {"nb_filters": 4}
        result = client._hil_request(hyperparams)

        client.socket.send_json.assert_called_once_with(hyperparams)
        client.socket.recv_json.assert_called_once()
        self.assertEqual(result, metrics)

    def test_hil_request_timeout_raises(self) -> None:
        """Timeouts surfaced by pyzmq should be wrapped in a RuntimeError."""
        client = _build_test_client()
        client.socket.recv_json.side_effect = zmq.error.Again()

        with self.assertRaises(RuntimeError):
            client._hil_request({"nb_filters": 8})

        client.socket.send_json.assert_called_once()


class ObjectiveTests(unittest.TestCase):
    """Exercise Optuna objective branches with lightweight stubs."""

    def setUp(self) -> None:
        self.client = _build_test_client()

        # Patch the heavy TensorFlow / hardware helpers to keep tests fast.
        self.build_patcher = patch("src.nas_model_client.build_tinyodom_model", return_value=MagicMock(compile=MagicMock()))
        self.count_patcher = patch("src.nas_model_client.count_flops", return_value=1234)
        self.train_patcher = patch("src.nas_model_client.train_and_score", return_value=(0.1, 0.2, 0.3, 0.4))
        self.hw_specs_patcher = patch("src.nas_model_client.return_hardware_specs", return_value=(2048, 4096))
        self.log_patcher = patch("src.nas_model_client.log_trial")

        self.mock_build = self.build_patcher.start()
        self.mock_count = self.count_patcher.start()
        self.mock_train = self.train_patcher.start()
        self.mock_hw_specs = self.hw_specs_patcher.start()
        self.mock_log = self.log_patcher.start()

    def tearDown(self) -> None:
        patch.stopall()

    def test_objective_prunes_on_flash_overflow(self) -> None:
        """Flash overflow errors should prune the Optuna trial."""
        self.client._hil_request = MagicMock(return_value={"error_code": HIL_MASTER_FLASH_OVERFLOW})
        trial = DummyTrial()

        with self.assertRaises(optuna.TrialPruned):
            self.client.objective(trial)

        self.mock_log.assert_called_once()
        self.assertEqual(trial.report_calls, [(-float("inf"), 0)])
        self.mock_train.assert_not_called()

    def test_objective_prunes_on_ram_overflow(self) -> None:
        """RAM overflow errors should prune the trial to skip training."""
        self.client._hil_request = MagicMock(return_value={"error_code": HIL_MASTER_RAM_OVERFLOW})
        trial = DummyTrial()

        with self.assertRaises(optuna.TrialPruned):
            self.client.objective(trial)

        self.mock_log.assert_called_once()
        self.assertEqual(trial.report_calls, [(-float("inf"), 0)])
        self.mock_train.assert_not_called()

    def test_objective_raises_on_device_not_found(self) -> None:
        """Device-not-found errors should abort the NAS run instead of pruning every trial."""
        self.client._hil_request = MagicMock(return_value={"error_code": HIL_MASTER_DEVICE_NOT_FOUND})
        trial = DummyTrial()

        with self.assertRaises(RuntimeError):
            self.client.objective(trial)

        self.mock_log.assert_not_called()
        self.mock_train.assert_not_called()

    def test_objective_handles_resource_failure(self) -> None:
        """Exceeding estimated resources should skip training and log a fatal code."""
        metrics = {
            "error_code": HIL_MASTER_SUCCESS,
            # Force RAM usage above the limit returned by return_hardware_specs.
            "ram_bytes": 8192,
            "flash_bytes": 1024,
            "arena_bytes": 2048,
        }
        self.client._hil_request = MagicMock(return_value=metrics)
        trial = DummyTrial()

        with self.assertRaises(optuna.TrialPruned):
            self.client.objective(trial)

        self.mock_train.assert_not_called()
        self.assertEqual(trial.report_calls, [(-float("inf"), 0)])


    def test_objective_happy_path_runs_training(self) -> None:
        """Valid metrics should flow into training and return the reported score."""
        metrics = {
            "error_code": HIL_MASTER_SUCCESS,
            "ram_bytes": 512,
            "flash_bytes": 512,
            "arena_bytes": 1024,
            "latency_ms": 10.0,
        }
        self.client._hil_request = MagicMock(return_value=metrics)
        trial = DummyTrial()

        result = self.client.objective(trial)

        self.mock_count.assert_called_once()
        self.mock_train.assert_called_once()
        self.assertEqual(result, 0.3)
        self.mock_log.assert_called_once()


class SmokeTestTests(unittest.TestCase):
    """Ensure the convenience smoke_test helper toggles config safely."""

    def test_smoke_test_restores_config_flags(self) -> None:
        """Config flags should be restored even if the study raises."""
        client = _build_test_client()
        client.objective = MagicMock(return_value=0.1)

        class DummyStudy:
            """Simple Optuna study substitute that tracks optimize invocations."""

            def __init__(self):
                self.best_trial = SimpleNamespace(value=1.0, params={}, user_attrs={})
                self.optimize_calls = []
                self.trials = []

            def optimize(self, func, n_trials):
                self.optimize_calls.append((func, n_trials))

        fake_study = DummyStudy()

        with patch("src.nas_model_client.optuna.create_study", return_value=fake_study):
            client.smoke_test(train=False, hil=False, trials=2, epochs=1)

        self.assertEqual(fake_study.optimize_calls[0][1], 2)
        self.assertEqual(client.config.device.hil, True)
        self.assertEqual(client.config.training.train, True)
        self.assertEqual(client.config.training.nas_epochs, 10)
        self.assertEqual(client.config.training.nas_multiobjective, False)

    def test_smoke_test_handles_multiobjective(self) -> None:
        """Smoke test should configure a multi-objective study when requested."""
        client = _build_test_client()
        client.objective = MagicMock(return_value=(0.1, 1.0))

        class DummyStudy:
            """Simple Optuna study substitute for multi-objective runs."""

            def __init__(self):
                trial = SimpleNamespace(
                    state=TrialState.COMPLETE,
                    number=0,
                    values=(0.5, 1.5),
                    params={"foo": 1},
                    user_attrs={"latency_ms": 1.0},
                )
                self.best_trials = [trial]
                self.trials = [trial]
                self.optimize_calls = []

            def optimize(self, func, n_trials):
                self.optimize_calls.append((func, n_trials))

        fake_study = DummyStudy()

        with patch("src.nas_model_client.optuna.create_study", return_value=fake_study) as mock_create:
            client.smoke_test(train=False, hil=False, trials=1, epochs=1, multiobjective=True)

        self.assertEqual(fake_study.optimize_calls[0][1], 1)
        kwargs = mock_create.call_args.kwargs
        self.assertEqual(kwargs["directions"], ["maximize", "minimize"])
        # Ensure we restore the original config flag after the run.
        self.assertFalse(client.config.training.nas_multiobjective)


class RunNASTests(unittest.TestCase):
    """run_nas should continue until completed trials meet the target."""

    class DummyStudy:
        def __init__(self, states):
            self.states_queue = list(states)
            self.trials = []
            self.optimize_calls = []
            self.best_trial = SimpleNamespace(value=None, params={})
            self.best_value = None
            self.enqueue_calls = []

        def optimize(self, func, n_trials):
            self.optimize_calls.append(n_trials)
            for _ in range(n_trials):
                state = self.states_queue.pop(0) if self.states_queue else TrialState.FAIL
                trial = SimpleNamespace(state=state)
                self.trials.append(trial)
            complete = [t for t in self.trials if t.state == TrialState.COMPLETE]
            if complete:
                self.best_trial = SimpleNamespace(value=1.0, params={})
                self.best_value = self.best_trial.value

        def enqueue_trial(self, params):
            self.enqueue_calls.append(params)

    def test_run_nas_retries_until_completed_target(self) -> None:
        client = _build_test_client()
        client.config.training.nas_trials = 2
        client.config.training.max_total_trials = 5
        states = [TrialState.PRUNED, TrialState.COMPLETE, TrialState.COMPLETE]
        dummy = self.DummyStudy(states)
        client.objective = MagicMock()

        with patch("src.nas_model_client.optuna.create_study", return_value=dummy):
            study = client.run_nas(study_name="demo", storage="sqlite:///dummy.db")

        self.assertIs(study, dummy)
        self.assertEqual(sum(t.state == TrialState.COMPLETE for t in dummy.trials), 2)
        self.assertEqual(len(dummy.trials), 3)
        self.assertEqual(dummy.optimize_calls, [2, 1])
        self.assertEqual(
            dummy.enqueue_calls,
            [
                {
                    "nb_filters": 10,
                    "kernel_size": 12,
                    "dropout_rate": 0.0,
                    "use_skip_connections": False,
                    "norm_flag": True,
                    "dilations_index": 107,
                }
            ],
        )

    def test_run_nas_honors_max_total_trials_cap(self) -> None:
        client = _build_test_client()
        client.config.training.nas_trials = 2
        client.config.training.max_total_trials = 3
        states = [TrialState.PRUNED, TrialState.FAIL, TrialState.PRUNED]
        dummy = self.DummyStudy(states)
        client.objective = MagicMock()

        with patch("src.nas_model_client.optuna.create_study", return_value=dummy):
            study = client.run_nas(study_name="demo", storage="sqlite:///dummy.db")

        self.assertIs(study, dummy)
        self.assertEqual(len(dummy.trials), 3)
        self.assertEqual(dummy.optimize_calls, [2, 1])
        self.assertEqual(sum(t.state == TrialState.COMPLETE for t in dummy.trials), 0)
        self.assertEqual(
            dummy.enqueue_calls,
            [
                {
                    "nb_filters": 10,
                    "kernel_size": 12,
                    "dropout_rate": 0.0,
                    "use_skip_connections": False,
                    "norm_flag": True,
                    "dilations_index": 107,
                }
            ],
        )


class PlotTrainingHistoryTests(unittest.TestCase):
    """Plotting helpers should emit PNGs without requiring a display."""

    def test_plot_training_history_writes_pngs(self) -> None:
        import matplotlib.pyplot as plt

        plt.switch_backend("Agg")
        with tempfile.TemporaryDirectory() as tmpdir:
            client = _build_test_client(base_dir=Path(tmpdir))
            history = {
                "loss": [1.0, 0.5],
                "val_loss": [1.1, 0.6],
                "velx_loss": [0.9, 0.4],
                "val_velx_loss": [1.0, 0.5],
            }

            result = client.plot_training_history(history=history, output_dir=Path(tmpdir), study_name="demo")

            loss_path = Path(result["loss_plot"])
            self.assertTrue(loss_path.is_file())
            self.assertGreater(loss_path.stat().st_size, 0)
            self.assertIn("loss_components_plot", result)
            components_path = Path(result["loss_components_plot"])
            self.assertTrue(components_path.is_file())
            self.assertGreater(components_path.stat().st_size, 0)


class EvaluateCheckpointTests(unittest.TestCase):
    """Checkpoint evaluation should write metrics and (optionally) export TFLite."""

    def test_evaluate_checkpoint_writes_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            client = _build_test_client(base_dir=base)
            length = 4
            gt_vx = np.ones((length, 1), dtype=np.float32)
            gt_vy = np.ones((length, 1), dtype=np.float32)
            client.test_data = SimpleNamespace(
                inputs=np.zeros((length, 1, 1), dtype=np.float32),
                x_vel=gt_vx,
                y_vel=gt_vy,
            )

            class FakeModel:
                def predict(self, _inputs):
                    return [np.ones_like(gt_vx), np.ones_like(gt_vy)]

            metrics_path = base / "metrics.json"
            with patch("src.nas_model_client.load_model", return_value=FakeModel()):
                metrics = client.evaluate_checkpoint(
                    checkpoint_path=base / "ckpt.keras",
                    metrics_path=metrics_path,
                    export_tflite=False,
                )

            self.assertTrue(metrics_path.is_file())
            self.assertTrue(metrics_path.with_suffix(".csv").is_file())
            self.assertAlmostEqual(metrics["rmse_vel_x"], 0.0)
            self.assertAlmostEqual(metrics["rmse_vel_y"], 0.0)
            self.assertEqual(metrics["checkpoint_path"], str(base / "ckpt.keras"))

    def test_evaluate_checkpoint_exports_tflite_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            client = _build_test_client(base_dir=base)
            gt_vx = np.zeros((2, 1), dtype=np.float32)
            gt_vy = np.zeros((2, 1), dtype=np.float32)
            client.test_data = SimpleNamespace(
                inputs=np.zeros((2, 1, 1), dtype=np.float32),
                x_vel=gt_vx,
                y_vel=gt_vy,
            )

            class FakeModel:
                def predict(self, _inputs):
                    return [gt_vx, gt_vy]

            tflite_path = base / "model.tflite"
            with patch("src.nas_model_client.load_model", return_value=FakeModel()), patch(
                "src.nas_model_client.convert_to_tflite_model"
            ) as mock_convert:
                client.evaluate_checkpoint(
                    checkpoint_path=base / "ckpt.keras",
                    metrics_path=base / "metrics.json",
                    export_tflite=True,
                    tflite_path=tflite_path,
                )

            self.assertTrue(mock_convert.called)
            self.assertEqual(mock_convert.call_args.kwargs["output_name"], tflite_path)


class TrajectoryMetricsTests(unittest.TestCase):
    """Trajectory metrics/plots should be generated with stubbed models."""

    def test_trajectory_metrics_and_plots_zero_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            client = _build_test_client(base_dir=base)
            length = 4
            vx = np.full((length, 1), 0.5, dtype=np.float32)
            vy = np.full((length, 1), 0.5, dtype=np.float32)
            client.test_data = SimpleNamespace(
                inputs=np.zeros((length, 1, 1), dtype=np.float32),
                x_vel=vx,
                y_vel=vy,
                size_of_each=[length],
                x0=[0.0],
                y0=[0.0],
            )

            class FakeModel:
                def predict(self, _inputs):
                    return [vx, vy]

            with patch("src.nas_model_client.load_model", return_value=FakeModel()):
                metrics = client.trajectory_metrics_and_plots(
                    checkpoint_path=base / "ckpt.keras",
                    plot_dir=base,
                    stride=1,
                    window_size=2,
                    study_name="demo",
                )

            metrics_path = base / "demo_trajectory_metrics.json"
            self.assertTrue(metrics_path.is_file())
            self.assertAlmostEqual(metrics["ate_mean"], 0.0)
            self.assertEqual(len(metrics["plots"]), 1)
            self.assertTrue(Path(metrics["plots"][0]).is_file())
            # RTE uses a 60s window; with tiny synthetic data it should be NaN.
            self.assertTrue(np.isnan(metrics["rte_median"]))


class SummaryBundleTests(unittest.TestCase):
    """Summary bundle aggregation should fuse existing artifacts."""

    def test_write_summary_bundle_persists_expected_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            client = _build_test_client(base_dir=base)
            history_path = base / "history.json"
            history_path.write_text("{}")
            loss_plots = {"loss_plot": str(base / "loss.png")}
            test_metrics = {"checkpoint_path": str(base / "ckpt.keras"), "tflite_path": None}
            traj_metrics = {"ate_mean": 0.1}

            class DummyStudy:
                def __init__(self):
                    self.best_trial = SimpleNamespace(params={"nb_filters": 8})

            with patch("src.nas_model_client.optuna.load_study", return_value=DummyStudy()):
                summary_path = client.write_summary_bundle(
                    study_storage="sqlite:///dummy.db",
                    study_name="demo",
                    history_path=history_path,
                    loss_plots=loss_plots,
                    test_metrics=test_metrics,
                    traj_metrics=traj_metrics,
                    summary_path=base / "summary.json",
                )

            self.assertTrue(summary_path.is_file())
            content = json.loads(summary_path.read_text())
            self.assertEqual(content["best_params"], {"nb_filters": 8})
            self.assertEqual(content["loss_plots"], loss_plots)
            self.assertEqual(content["test_metrics"], test_metrics)
            self.assertEqual(content["trajectory_metrics"], traj_metrics)


class CloseTests(unittest.TestCase):
    """Verify resources are released when NASModelClient.close is called."""

    def test_close_shuts_socket_and_context(self) -> None:
        """Closing the client should close the socket and terminate the context.

        This protects long test runs from leaking file descriptors, so the unit
        test just confirms we call the underlying pyzmq cleanup hooks.
        """
        client = _build_test_client()
        client.close()
        client.socket.close.assert_called_once_with(linger=0)
        client.context.term.assert_called_once()


if __name__ == "__main__":
    unittest.main()
