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
from addict import Dict
from optuna.trial import TrialState

# Ensure `src` is importable when the suite is launched via `python -m unittest`.
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# The NASModelClient is the unit under test for this module.
from nas_model_client import NASModelClient  # noqa: E402
from tinyodom.hardware import (
    HIL_MASTER_DEVICE_NOT_FOUND,
    HIL_MASTER_FLASH_OVERFLOW,
    HIL_MASTER_RAM_OVERFLOW,
    HIL_MASTER_FATAL,
    HIL_MASTER_SUCCESS,
)  # noqa: E402
from tinyodom.model import ScoringResult, train_and_score  # noqa: E402


def _make_scoring_result(
    *,
    score: float | None = 0.3,
    objective_names: list[str] | None = None,
    objective_values: list[float] | None = None,
    objective_directions: list[str] | None = None,
    rmse_vel_x: float = 0.1,
    rmse_vel_y: float = 0.2,
) -> ScoringResult:
    if objective_names is None:
        objective_names = ["score"]
    if objective_values is None:
        objective_values = [score if score is not None else 0.0]
    if objective_directions is None:
        objective_directions = ["maximize"]
    return ScoringResult(
        rmse_vel_x=rmse_vel_x,
        rmse_vel_y=rmse_vel_y,
        score=score,
        objective_names=objective_names,
        objective_values=objective_values,
        objective_directions=objective_directions,
    )


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
            energy_aware=False,
            nas_multiobjective_population_size=8,
        ),
        device=SimpleNamespace(name="TEST_DEVICE", hil=True, serial_port="ttyACM0"),
        nas=SimpleNamespace(
            score=Dict(
                type="scoring-function",
                metrics=Dict(),
                params=Dict(
                    terms=[
                        Dict(type="weighted", metric="rmse_total", weight=-1.0),
                    ]
                ),
            ),
            prune=Dict(rules=[]),
        ),
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

        payload = {"hyperparams": {"nb_filters": 4}}
        result = client._hil_request(payload)

        client.socket.send_json.assert_called_once_with(payload)
        client.socket.recv_json.assert_called_once()
        self.assertEqual(result, metrics)

    def test_hil_request_timeout_raises(self) -> None:
        """Timeouts surfaced by pyzmq should be wrapped in a RuntimeError."""
        client = _build_test_client()
        client.socket.recv_json.side_effect = zmq.error.Again()

        with self.assertRaises(RuntimeError):
            client._hil_request({"hyperparams": {"nb_filters": 8}})

        client.socket.send_json.assert_called_once()


class ObjectiveTests(unittest.TestCase):
    """Exercise Optuna objective branches with lightweight stubs."""

    def setUp(self) -> None:
        self.client = _build_test_client()

        # Patch the heavy TensorFlow / hardware helpers to keep tests fast.
        self.build_patcher = patch("nas_model_client.build_tinyodom_model", return_value=MagicMock(compile=MagicMock()))
        self.count_patcher = patch("nas_model_client.count_flops", return_value=1234)
        self.train_patcher = patch("nas_model_client.train_and_score", return_value=_make_scoring_result())
        self.hw_specs_patcher = patch("nas_model_client.return_hardware_specs", return_value=(2048, 4096))
        self.log_patcher = patch("nas_model_client.log_trial")

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

    def test_objective_prunes_before_training_on_config_rule(self) -> None:
        metrics = {
            "error_code": HIL_MASTER_SUCCESS,
            "ram_bytes": 512,
            "flash_bytes": 512,
            "arena_bytes": 1024,
            "latency_ms": 25.0,
            "latency_budget_ms": 20.0,
            "energy_mj_per_inference": -1.0,
            "avg_power_mw": -1.0,
            "avg_current_ma": -1.0,
            "bus_voltage_v": -1.0,
        }
        self.client.config.nas.prune = Dict(
            rules=[
                Dict(
                    rule="latency_budget",
                    metric="latency_ms",
                    condition="gt",
                    reference=Dict(type="metric", metric="latency_budget_ms"),
                    reason="Latency exceeds deployment budget",
                )
            ]
        )
        self.client._hil_request = MagicMock(return_value=metrics)
        trial = DummyTrial()

        with self.assertRaises(optuna.TrialPruned):
            self.client.objective(trial)

        self.mock_train.assert_not_called()
        self.mock_log.assert_called_once()
        self.assertEqual(self.mock_log.call_args.kwargs["prune_rule"], "latency_budget")
        self.assertEqual(
            self.mock_log.call_args.kwargs["prune_reason"],
            "Latency exceeds deployment budget",
        )

    def test_objective_prunes_when_rule_metric_is_unavailable(self) -> None:
        metrics = {
            "error_code": HIL_MASTER_SUCCESS,
            "ram_bytes": 512,
            "flash_bytes": 512,
            "arena_bytes": 1024,
            "latency_ms": -1.0,
            "latency_budget_ms": 20.0,
            "energy_mj_per_inference": -1.0,
            "avg_power_mw": -1.0,
            "avg_current_ma": -1.0,
            "bus_voltage_v": -1.0,
        }
        self.client.config.nas.prune = Dict(
            rules=[
                Dict(
                    metric="latency_ms",
                    condition="gt",
                    reference=Dict(type="metric", metric="latency_budget_ms"),
                    reason="Latency exceeds deployment budget",
                    rule="rule_0",
                )
            ]
        )
        self.client._hil_request = MagicMock(return_value=metrics)
        trial = DummyTrial()

        with self.assertRaises(optuna.TrialPruned):
            self.client.objective(trial)

        self.mock_train.assert_not_called()
        self.assertEqual(self.mock_log.call_args.kwargs["prune_rule"], "rule_0")
        self.assertIn("Configured prune metric unavailable", self.mock_log.call_args.kwargs["prune_reason"])

    def test_objective_samples_cpu_clock_into_device_overrides(self) -> None:
        metrics = {
            "error_code": HIL_MASTER_SUCCESS,
            "ram_bytes": 512,
            "flash_bytes": 512,
            "arena_bytes": 1024,
            "latency_ms": 10.0,
            "cpu_clock_mhz_requested": 400,
        }
        self.client.config.device.cpu_clock_mhz_options = [200, 400, 600]
        self.client._hil_request = MagicMock(return_value=metrics)
        trial = DummyTrial()

        self.client.objective(trial)

        sent_payload = self.client._hil_request.call_args.args[0]
        self.assertIn("hyperparams", sent_payload)
        self.assertEqual(sent_payload["device_options_overrides"], {"cpu_clock_mhz": 200})
        self.assertNotIn("cpu_clock_mhz", sent_payload["hyperparams"])
        self.assertEqual(trial.params["cpu_clock_mhz_index"], 0)

    def test_objective_omits_device_overrides_when_clock_options_are_null(self) -> None:
        metrics = {
            "error_code": HIL_MASTER_SUCCESS,
            "ram_bytes": 512,
            "flash_bytes": 512,
            "arena_bytes": 1024,
            "latency_ms": 10.0,
            "cpu_clock_mhz_requested": -1,
        }
        self.client.config.device.cpu_clock_mhz_options = None
        self.client._hil_request = MagicMock(return_value=metrics)
        trial = DummyTrial()

        self.client.objective(trial)

        sent_payload = self.client._hil_request.call_args.args[0]
        self.assertEqual(set(sent_payload.keys()), {"hyperparams"})

    def test_objective_stm_phase1_allows_arena_sentinel(self) -> None:
        """Ensure STM Phase 1 does not get pruned solely for ``arena_bytes=-1``.

        Returns
        -------
        None
        """
        metrics = {
            "error_code": HIL_MASTER_SUCCESS,
            "ram_bytes": 512,
            "flash_bytes": 512,
            "arena_bytes": -1,
            "latency_ms": 10.0,
        }
        self.client.config.device.name = "STM32_NUCLEO_N657X0_Q"
        self.client.config.device.stm32 = SimpleNamespace(project_root="/tmp/stm_fsbl")
        self.client._hil_request = MagicMock(return_value=metrics)
        trial = DummyTrial()

        result = self.client.objective(trial)

        self.mock_train.assert_called_once()
        self.assertEqual(result, 0.3)
        self.mock_log.assert_called_once()

    def test_objective_returns_configured_multiobjective_tuple(self) -> None:
        """Configured multi-objective runs should return all objective values.

        Returns
        -------
        None
        """
        metrics = {
            "error_code": HIL_MASTER_SUCCESS,
            "ram_bytes": 512,
            "flash_bytes": 512,
            "arena_bytes": -1,
            "latency_ms": 10.0,
            "energy_mj_per_inference": -1.0,
            "energy_aware": False,
            "hil_enabled": False,
        }
        self.client.config.device.name = "STM32_NUCLEO_N657X0_Q"
        self.client.config.device.stm32 = SimpleNamespace(project_root="/tmp/stm_fsbl")
        self.client.config.training.energy_aware = True
        self.client.config.nas.score = Dict(
            type="multi-objective",
            metrics=Dict(),
            params=Dict(
                objectives=[
                    Dict(metric="latency_ms", direction="minimize"),
                    Dict(metric="flash_bytes", direction="minimize"),
                ]
            ),
        )
        self.client.config.training.train = False
        self.client._hil_request = MagicMock(return_value=metrics)
        trial = DummyTrial()

        with patch("nas_model_client.train_and_score", wraps=train_and_score):
            result = self.client.objective(trial)

        self.assertEqual(result, (10.0, 512.0))
        self.mock_log.assert_called_once()

    def test_objective_portenta_forwards_device_options_to_hardware_specs(self) -> None:
        metrics = {
            "error_code": HIL_MASTER_SUCCESS,
            "ram_bytes": 512,
            "flash_bytes": 512,
            "arena_bytes": 1024,
            "latency_ms": 10.0,
        }
        self.client.config.device.name = "PORTENTA_H7"
        self.client.config.device.portenta = SimpleNamespace(
            target_core="cm4",
            split="50_50",
            security="none",
        )
        self.client._hil_request = MagicMock(return_value=metrics)
        trial = DummyTrial()

        self.client.objective(trial)

        self.mock_hw_specs.assert_called_once()
        _, kwargs = self.mock_hw_specs.call_args
        self.assertEqual(kwargs["device_options"]["target_core"], "cm4")
        self.assertEqual(kwargs["device_options"]["split"], "50_50")
        self.assertEqual(kwargs["device_options"]["security"], "none")

    def test_objective_portenta_requires_target_core_for_hardware_limits(self) -> None:
        metrics = {
            "error_code": HIL_MASTER_SUCCESS,
            "ram_bytes": 512,
            "flash_bytes": 512,
            "arena_bytes": 1024,
            "latency_ms": 10.0,
        }
        self.client.config.device.name = "PORTENTA_H7"
        self.client.config.device.portenta = SimpleNamespace()
        self.client._hil_request = MagicMock(return_value=metrics)
        trial = DummyTrial()

        with self.assertRaises(RuntimeError):
            self.client.objective(trial)


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

        with patch("nas_model_client.optuna.create_study", return_value=fake_study):
            with self.assertRaises(ValueError):
                client.smoke_test(train=False, hil=False, trials=2, epochs=1)

        self.assertEqual(client.config.device.hil, True)
        self.assertEqual(client.config.training.train, True)
        self.assertEqual(client.config.training.nas_epochs, 10)
        self.assertEqual(client.config.nas.score.type, "scoring-function")

    def test_smoke_test_uses_loaded_multiobjective_config(self) -> None:
        """Smoke test should honor multi-objective mode from the loaded config."""
        client = _build_test_client()
        client.config.nas.score = Dict(
            type="multi-objective",
            metrics=Dict(),
            params=Dict(
                objectives=[
                    Dict(metric="rmse_total", direction="minimize"),
                    Dict(metric="latency_ms", direction="minimize"),
                ]
            ),
        )
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

        with patch("nas_model_client.optuna.create_study", return_value=fake_study) as mock_create:
            client.smoke_test(train=True, hil=False, trials=1, epochs=1)

        self.assertEqual(fake_study.optimize_calls[0][1], 1)
        kwargs = mock_create.call_args.kwargs
        self.assertEqual(kwargs["directions"], ["minimize", "minimize"])
        self.assertFalse(kwargs["load_if_exists"])

    def test_smoke_test_uses_loaded_scalar_config(self) -> None:
        client = _build_test_client()
        client.objective = MagicMock(return_value=0.1)

        class DummyStudy:
            def __init__(self):
                self.best_trial = SimpleNamespace(value=1.0, params={}, user_attrs={})
                self.optimize_calls = []
                self.trials = []

            def optimize(self, func, n_trials):
                self.optimize_calls.append((func, n_trials))

        fake_study = DummyStudy()
        with patch("nas_model_client.optuna.create_study", return_value=fake_study) as mock_create:
            client.smoke_test(train=True, hil=False, trials=1, epochs=1)

        self.assertEqual(mock_create.call_args.kwargs["direction"], "maximize")

    def test_smoke_test_defaults_to_loaded_hil_setting(self) -> None:
        client = _build_test_client()
        observed_hil_values = []

        def _objective(_trial):
            observed_hil_values.append(client.config.device.hil)
            return 0.1

        client.objective = MagicMock(side_effect=_objective)

        class DummyStudy:
            def __init__(self):
                self.best_trial = SimpleNamespace(value=1.0, params={}, user_attrs={})
                self.optimize_calls = []
                self.trials = []

            def optimize(self, func, n_trials):
                self.optimize_calls.append((func, n_trials))
                func(SimpleNamespace())

        fake_study = DummyStudy()
        with patch("nas_model_client.optuna.create_study", return_value=fake_study):
            client.smoke_test(train=True, trials=1, epochs=1)

        self.assertEqual(observed_hil_values, [True])

    def test_smoke_test_rejects_derived_rmse_usage_when_training_disabled(self) -> None:
        client = _build_test_client()
        client.config.nas.score = Dict(
            type="scoring-function",
            metrics=Dict(
                combined_error=Dict(
                    type="add",
                    metrics=["rmse_total", "flops"],
                )
            ),
            params=Dict(
                terms=[
                    Dict(type="weighted", metric="combined_error", weight=-1.0),
                ]
            ),
        )

        with patch("nas_model_client.optuna.create_study") as mock_create:
            with self.assertRaises(ValueError):
                client.smoke_test(train=False, trials=1, epochs=1)

        self.assertFalse(mock_create.called)

    def test_smoke_test_removes_stale_db_and_log_before_starting(self) -> None:
        client = _build_test_client()
        client.objective = MagicMock(return_value=0.1)
        study_name = "stale_smoke"
        client.study_name = study_name

        class DummyStudy:
            def __init__(self):
                self.best_trial = SimpleNamespace(value=1.0, params={}, user_attrs={})
                self.optimize_calls = []
                self.trials = []

            def optimize(self, func, n_trials):
                self.optimize_calls.append((func, n_trials))

        fake_study = DummyStudy()
        artifacts_dir = client._artifacts_dir()
        db_path = artifacts_dir / "optuna_smoke_test.db"
        db_path.write_text("stale-db", encoding="utf-8")
        log_path = artifacts_dir / client.config.outputs.log_file_name
        log_path.write_text("stale-log", encoding="utf-8")

        with patch("nas_model_client.optuna.create_study", return_value=fake_study) as mock_create:
            with self.assertRaises(ValueError):
                client.smoke_test(train=False, hil=False, trials=1, epochs=1, study_name=study_name)

        self.assertFalse(db_path.exists())
        self.assertFalse(log_path.exists())
        self.assertFalse(mock_create.called)

    def test_copy_run_config_skips_same_file(self) -> None:
        client = _build_test_client()
        artifacts_dir = client._artifacts_dir()
        cfg_path = artifacts_dir / "config.yaml"
        cfg_path.write_text("device:\n  name: TEST_DEVICE\n", encoding="utf-8")
        client.config_path = cfg_path

        copied_path = client._copy_run_config(artifacts_dir)

        self.assertEqual(copied_path, cfg_path.resolve())
        self.assertEqual(cfg_path.read_text(encoding="utf-8"), "device:\n  name: TEST_DEVICE\n")


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

        with patch("nas_model_client.optuna.create_study", return_value=dummy):
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

        with patch("nas_model_client.optuna.create_study", return_value=dummy):
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
            with patch("nas_model_client.load_model", return_value=FakeModel()):
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
            with patch("nas_model_client.load_model", return_value=FakeModel()), patch(
                "nas_model_client.convert_to_tflite_model"
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

            with patch("nas_model_client.load_model", return_value=FakeModel()):
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

            with patch("nas_model_client.optuna.load_study", return_value=DummyStudy()):
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
