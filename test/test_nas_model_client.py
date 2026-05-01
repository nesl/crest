"""Regression tests for the legacy NASModelClient orchestration surface.

This module covers bootstrap helpers, objective/pruning branches, smoke-test
behavior, final-training utilities, evaluation and trajectory reporting, and
resource-cleanup paths while replacing heavy production dependencies with test
doubles.
"""

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
from tinyodom.model import ScoreConfigEvaluationError, TrialOutcome  # noqa: E402
from tinyodom.pipeline_types import (
    DataSplit,
    DatasetBundle,
    EvaluationResult,
    FitPlan,
    ModelBuildContext,
    TargetSpec,
    TaskMetricContract,
)  # noqa: E402


def _build_test_client(base_dir: Path | None = None) -> NASModelClient:
    """Construct a NASModelClient using lightweight stand-ins for config/data.

    The real class performs heavy dataset loading inside __init__. To keep the
    tests fast and deterministic, we bypass __init__ and fill in the handful of
    attributes the logic depends on.

    Notes
    -----
    This fixture preserves the production contracts around dataset bundles,
    target specs, metric contracts, model-family hooks, and output-path shape.
    It intentionally skips real dataset loading, real ZMQ sockets, and real
    filesystem/network side effects beyond temporary artifact directories.
    """
    client = NASModelClient.__new__(NASModelClient)
    base_dir = Path(tempfile.mkdtemp()) if base_dir is None else Path(base_dir)

    window_size = 16
    input_dim = 3
    # Minimal dataset/task/model contract: enough structure for build, fit,
    # evaluate, and trajectory helpers to behave like production code.
    legacy_dataset = SimpleNamespace(
        inputs=np.zeros((2, window_size, input_dim), dtype=np.float32),
        x_vel=np.zeros((2, 1), dtype=np.float32),
        y_vel=np.zeros((2, 1), dtype=np.float32),
        size_of_each=[2],
        x0=[0.0],
        y0=[0.0],
    )
    bundle = DatasetBundle(
        train=DataSplit(
            inputs=legacy_dataset.inputs,
            targets={"velx": legacy_dataset.x_vel, "vely": legacy_dataset.y_vel},
            metadata={"size_of_each": legacy_dataset.size_of_each, "x0": legacy_dataset.x0, "y0": legacy_dataset.y0},
        ),
        val=DataSplit(
            inputs=legacy_dataset.inputs,
            targets={"velx": legacy_dataset.x_vel, "vely": legacy_dataset.y_vel},
            metadata={"size_of_each": legacy_dataset.size_of_each, "x0": legacy_dataset.x0, "y0": legacy_dataset.y0},
        ),
        test=DataSplit(
            inputs=legacy_dataset.inputs,
            targets={"velx": legacy_dataset.x_vel, "vely": legacy_dataset.y_vel},
            metadata={"size_of_each": legacy_dataset.size_of_each, "x0": legacy_dataset.x0, "y0": legacy_dataset.y0},
        ),
        input_shape=(window_size, input_dim),
        input_dtype="float32",
        metadata={"input_dim": input_dim},
    )
    target_spec = TargetSpec(
        task_type="regression",
        output_names=["velx", "vely"],
        output_shapes=[(1,), (1,)],
        metadata={},
    )
    metric_contract = TaskMetricContract(
        available_metric_names={"rmse_vel_x", "rmse_vel_y", "rmse_total"},
        training_only_metric_names={"rmse_vel_x", "rmse_vel_y", "rmse_total"},
        nonnegative_metric_names={"rmse_vel_x", "rmse_vel_y", "rmse_total"},
        primary_metric_names={"rmse_total"},
    )
    fake_built_model = MagicMock()
    fake_loaded_model = MagicMock()
    task = SimpleNamespace(
        validate_config=MagicMock(),
        validate_model_outputs=MagicMock(),
        compile_model=MagicMock(),
        make_fit_plan=MagicMock(
            return_value=FitPlan(
                fit_kwargs={
                    "x": bundle.train.inputs,
                    "y": [bundle.train.targets["velx"], bundle.train.targets["vely"]],
                    "validation_data": (
                        bundle.val.inputs,
                        [bundle.val.targets["velx"], bundle.val.targets["vely"]],
                    ),
                    "shuffle": True,
                },
                callbacks=["callback"],
                monitor_metric="val_loss",
            )
        ),
        evaluate=MagicMock(
            return_value=EvaluationResult(
                metrics={"rmse_vel_x": 0.1, "rmse_vel_y": 0.2, "rmse_total": 0.3},
                predictions=[legacy_dataset.x_vel, legacy_dataset.y_vel],
            )
        ),
    )
    model_family = SimpleNamespace(
        validate_config=MagicMock(),
        sample_hparams=MagicMock(
            return_value={
                "nb_filters": 2,
                "kernel_size": 2,
                "dropout_rate": 0.1,
                "use_skip_connections": True,
                "norm_flag": True,
                "dilations": [1, 2],
            }
        ),
        validate_hparams=MagicMock(),
        build_model=MagicMock(return_value=fake_built_model),
        load_model=MagicMock(return_value=fake_loaded_model),
    )

    # Artifact, socket, and config defaults: mirror the production config shape
    # without paying the cost of real I/O-heavy initialization.
    client.config = SimpleNamespace(
        network=SimpleNamespace(host="localhost", port=5555, recv_timeout_sec=5, send_timeout_sec=5),
        training=SimpleNamespace(
            drop_rate_choices=[0.1, 0.2],
            train=True,
            nas_epochs=10,
            model_epochs=20,
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
        dataset=SimpleNamespace(
            name="oxiod",
            params=Dict(
                directory="data",
                sampling_rate_hz=100,
                window_size=window_size,
                stride=1,
            ),
        ),
        task=SimpleNamespace(name="odometry_regression", params=Dict()),
        model=SimpleNamespace(family="tinyodom_tcn", params=Dict(), search=Dict()),
    )
    client.config.outputs.models_dir.mkdir(parents=True, exist_ok=True)
    client.config_path = base_dir / "config.yaml"
    # Reuse the placeholder dataset wherever the client expects training/val/test.
    client.training_data = legacy_dataset
    client.validation_data = legacy_dataset
    client.test_data = legacy_dataset
    client.dataset = SimpleNamespace(validate_config=MagicMock())
    client.dataset_name = "oxiod"
    client.task = task
    client.task_name = "odometry_regression"
    client.model_family = model_family
    client.model_family_name = "tinyodom_tcn"
    client.dataset_bundle = bundle
    client.target_spec = target_spec
    client.metric_contract = metric_contract
    client.model_build_context = ModelBuildContext(
        input_shape=bundle.input_shape,
        input_dtype=bundle.input_dtype,
        target_spec=target_spec,
        dataset_metadata=dict(bundle.metadata),
        task_metadata=dict(target_spec.metadata),
    )
    client.dataset_config = client.config.dataset.params
    client.task_config = Dict()
    client.model_config = Dict(family="tinyodom_tcn", params=Dict(), search=Dict())
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
        """Record one Optuna-style intermediate report call."""
        self.report_calls.append((value, step))

    def set_user_attr(self, key, value):
        """Record one Optuna-style user attribute update."""
        self.user_attrs[key] = value


class HILRequestTests(unittest.TestCase):
    """Validate the ZeroMQ request/response helper."""

    def test_hil_request_success(self) -> None:
        """A successful round-trip should return the parsed metrics dict.

        Instead of booting a real server we just drive the mock socket and
        assert that the client sends/receives JSON exactly once.
        """
        # A successful HIL round trip should deserialize cleanly into the NAS client's metrics payload.
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
        # Socket timeouts should surface as errors so stalled HIL servers do not look like valid trial failures.
        client = _build_test_client()
        client.socket.recv_json.side_effect = zmq.error.Again()

        with self.assertRaises(RuntimeError):
            client._hil_request({"hyperparams": {"nb_filters": 8}})

        client.socket.send_json.assert_called_once()


class InitializationTests(unittest.TestCase):
    """Validate the Phase 4 bootstrap and component-selection helpers."""

    def test_resolve_component_selection_requires_explicit_component_blocks(self) -> None:
        # The breaking contract rejects configs that omit dataset/task/model blocks instead of inventing defaults.
        client = _build_test_client()
        delattr(client.config, "dataset")

        with self.assertRaisesRegex(KeyError, "dataset"):
            client._resolve_component_selection(client.config)

    def test_resolve_component_selection_requires_dataset_params(self) -> None:
        # The breaking contract requires dataset params instead of falling back to the legacy top-level data block.
        client = _build_test_client()
        client.config.dataset = SimpleNamespace(name="custom_dataset")

        with self.assertRaisesRegex(KeyError, "dataset.params"):
            client._resolve_component_selection(client.config)

    def test_resolve_component_selection_honors_explicit_component_blocks(self) -> None:
        # Explicit component blocks should be the only supported path and should come back as native mappings.
        client = _build_test_client()
        client.config.dataset = SimpleNamespace(name="custom_dataset", params=Dict(root="custom"))
        client.config.task = SimpleNamespace(name="custom_task", params=Dict(alpha=3))
        client.config.model = SimpleNamespace(
            family="custom_family",
            params=Dict(width=8),
            search=Dict(depth=[2, 3]),
        )

        selection = client._resolve_component_selection(client.config)

        self.assertEqual(selection["dataset_name"], "custom_dataset")
        self.assertEqual(selection["dataset_config"].root, "custom")
        self.assertEqual(selection["task_name"], "custom_task")
        self.assertEqual(selection["task_config"].alpha, 3)
        self.assertEqual(selection["model_family_name"], "custom_family")
        self.assertEqual(selection["model_config"]["family"], "custom_family")
        self.assertEqual(selection["model_config"]["params"].width, 8)
        self.assertEqual(selection["model_config"]["search"].depth, [2, 3])

    def test_init_reuses_preliminary_bundle_when_dataset_selection_matches(self) -> None:
        # Initialization should reuse a matching preliminary bundle so repeated setup does not reload the same dataset.
        base = Path(tempfile.mkdtemp())
        config = SimpleNamespace(
            network=SimpleNamespace(host="localhost", port=5555, recv_timeout_sec=5, send_timeout_sec=5),
            device=SimpleNamespace(hil=True, name="TEST_DEVICE"),
            nas=SimpleNamespace(score=Dict(type="scoring-function", metrics=Dict(), params=Dict(terms=[]))),
            outputs=SimpleNamespace(
                models_dir=base / "models",
                checkpoint_path=base / "model.keras",
                log_file_name="log.csv",
            ),
            dataset=SimpleNamespace(
                name="oxiod",
                params=Dict(directory="data", sampling_rate_hz=100, window_size=16, stride=1),
            ),
            task=SimpleNamespace(name="odometry_regression", params=Dict()),
            model=SimpleNamespace(family="tinyodom_tcn", params=Dict(), search=Dict()),
        )
        fake_bundle = DatasetBundle(
            train=DataSplit(inputs=np.zeros((1, 16, 3)), targets={"velx": np.zeros((1, 1)), "vely": np.zeros((1, 1))}),
            val=DataSplit(inputs=np.zeros((1, 16, 3)), targets={"velx": np.zeros((1, 1)), "vely": np.zeros((1, 1))}),
            test=DataSplit(inputs=np.zeros((1, 16, 3)), targets={"velx": np.zeros((1, 1)), "vely": np.zeros((1, 1))}),
            input_shape=(16, 3),
            input_dtype="float32",
        )
        target_spec = TargetSpec(
            task_type="regression",
            output_names=["velx", "vely"],
            output_shapes=[(1,), (1,)],
            metadata={},
        )
        metric_contract = TaskMetricContract(
            available_metric_names={"rmse_vel_x", "rmse_vel_y", "rmse_total"},
            training_only_metric_names={"rmse_vel_x", "rmse_vel_y", "rmse_total"},
            nonnegative_metric_names={"rmse_vel_x", "rmse_vel_y", "rmse_total"},
            primary_metric_names={"rmse_total"},
        )
        fake_task = SimpleNamespace(
            build_target_spec=MagicMock(return_value=target_spec),
            metric_contract=MagicMock(return_value=metric_contract),
        )
        fake_socket = MagicMock()
        fake_context = MagicMock()
        fake_context.socket.return_value = fake_socket

        with patch("nas_model_client.ensure_builtin_components_registered"), patch(
            "nas_model_client.load_config",
            side_effect=[config, config],
        ) as mock_load_config, patch.object(
            NASModelClient,
            "_load_dataset_bundle",
            return_value=fake_bundle,
        ) as mock_load_bundle, patch.object(
            NASModelClient,
            "_instantiate_task",
            return_value=fake_task,
        ), patch.object(
            NASModelClient,
            "_initialize_component_state",
        ), patch(
            "nas_model_client.zmq.Context.instance",
            return_value=fake_context,
        ):
            client = NASModelClient(base / "config.yaml")

        self.assertIsInstance(client, NASModelClient)
        self.assertEqual(mock_load_bundle.call_count, 1)
        self.assertEqual(mock_load_config.call_count, 2)
        self.assertEqual(
            mock_load_config.call_args_list[1].kwargs["task_metric_names"],
            metric_contract.available_metric_names,
        )
        self.assertEqual(
            mock_load_config.call_args_list[1].kwargs["training_only_task_metric_names"],
            metric_contract.training_only_metric_names,
        )
        fake_socket.connect.assert_called_once()

    def test_refresh_legacy_split_aliases_tolerates_non_odometry_targets(self) -> None:
        # Legacy split aliases should refresh without assuming odometry-specific target names.
        client = _build_test_client()
        bundle = DatasetBundle(
            train=DataSplit(
                inputs=np.zeros((2, 16, 3), dtype=np.float32),
                targets={"class_id": np.array([0, 1], dtype=np.int32)},
                metadata={},
            ),
            val=DataSplit(
                inputs=np.zeros((1, 16, 3), dtype=np.float32),
                targets={"class_id": np.array([1], dtype=np.int32)},
                metadata={},
            ),
            test=DataSplit(
                inputs=np.zeros((1, 16, 3), dtype=np.float32),
                targets={"class_id": np.array([0], dtype=np.int32)},
                metadata={},
            ),
            input_shape=(16, 3),
            input_dtype="float32",
            metadata={},
        )

        client._refresh_legacy_split_aliases(bundle)

        self.assertIsNone(client.training_data.x_vel)
        self.assertIsNone(client.training_data.y_vel)
        self.assertIsNone(client.validation_data.x_vel)
        self.assertIsNone(client.test_data.y_vel)


class ObjectiveTests(unittest.TestCase):
    """Exercise Optuna objective branches with lightweight stubs."""

    def setUp(self) -> None:
        self.client = _build_test_client()

        self.count_patcher = patch("nas_model_client.count_flops", return_value=1234)
        self.hw_specs_patcher = patch("nas_model_client.return_hardware_specs", return_value=(2048, 4096))
        self.log_patcher = patch("nas_model_client.log_trial")

        self.mock_count = self.count_patcher.start()
        self.mock_hw_specs = self.hw_specs_patcher.start()
        self.mock_log = self.log_patcher.start()

    def tearDown(self) -> None:
        patch.stopall()

    def test_objective_prunes_on_flash_overflow(self) -> None:
        """Flash overflow errors should prune the Optuna trial."""
        # Flash-overflow trials should prune immediately so obviously too-large candidates do not enter training.
        self.client._hil_request = MagicMock(return_value={"error_code": HIL_MASTER_FLASH_OVERFLOW})
        trial = DummyTrial()

        with self.assertRaises(optuna.TrialPruned):
            self.client.objective(trial)

        self.mock_log.assert_called_once()
        self.assertEqual(trial.report_calls, [(-float("inf"), 0)])
        self.client.task.make_fit_plan.assert_not_called()

    def test_objective_prune_logging_populates_cadenced_sentinels(self) -> None:
        """Early-pruned trials should still log stable cadenced metric defaults."""
        # Pruned trials should still log cadenced sentinel fields so CSV rows keep the same schema as successful runs.
        self.client._hil_request = MagicMock(return_value={"error_code": HIL_MASTER_FLASH_OVERFLOW})
        trial = DummyTrial()

        with self.assertRaises(optuna.TrialPruned):
            self.client.objective(trial)

        logged_metrics = self.mock_log.call_args.kwargs["metrics"]
        self.assertEqual(logged_metrics["runtime_mode"], "back_to_back")
        self.assertEqual(logged_metrics["cadenced_energy_mj_per_trial"], -1.0)
        self.assertEqual(logged_metrics["cadenced_error_code"], -1)
        self.assertIsNone(logged_metrics["cadenced_error_label"])

    def test_objective_prunes_on_ram_overflow(self) -> None:
        """RAM overflow errors should prune the trial to skip training."""
        # RAM-overflow trials should prune immediately so impossible memory footprints stop before fit().
        self.client._hil_request = MagicMock(return_value={"error_code": HIL_MASTER_RAM_OVERFLOW})
        trial = DummyTrial()

        with self.assertRaises(optuna.TrialPruned):
            self.client.objective(trial)

        self.mock_log.assert_called_once()
        self.assertEqual(trial.report_calls, [(-float("inf"), 0)])
        self.client.task.make_fit_plan.assert_not_called()

    def test_objective_raises_on_device_not_found(self) -> None:
        """Device-not-found errors should abort the NAS run instead of pruning every trial."""
        # Unknown devices should raise early so hardware catalog mistakes do not look like training failures.
        self.client._hil_request = MagicMock(return_value={"error_code": HIL_MASTER_DEVICE_NOT_FOUND})
        trial = DummyTrial()

        with self.assertRaises(RuntimeError):
            self.client.objective(trial)

    def test_objective_rejects_invalid_model_build_context_input_shapes(self) -> None:
        # Invalid model-build input shapes should fail before the NAS client tries to train or evaluate a broken graph.
        self.client._hil_request = MagicMock()

        for invalid_shape in (None, (None, 3), ("abc", 3), (0, 3), (16, False)):
            with self.subTest(input_shape=invalid_shape):
                self.client.model_build_context.input_shape = invalid_shape
                with self.assertRaisesRegex(ValueError, "2D logical input shape"):
                    self.client.objective(DummyTrial())

        self.client._hil_request.assert_not_called()

        self.mock_log.assert_not_called()
        self.client.task.make_fit_plan.assert_not_called()

    def test_objective_handles_resource_failure(self) -> None:
        """Exceeding estimated resources should skip training and log a fatal code."""
        # Low-level resource failures should prune the trial with a stable penalty instead of leaking raw backend exceptions.
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

        self.client.task.make_fit_plan.assert_not_called()
        self.assertEqual(trial.report_calls, [(-float("inf"), 0)])


    def test_objective_happy_path_runs_training(self) -> None:
        """Valid metrics should flow into training and return the reported score."""
        # A healthy trial should still execute the full fit and task-evaluation path before scoring.
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
        self.client.task.make_fit_plan.assert_called_once()
        self.client.task.evaluate.assert_called_once()
        self.assertEqual(result, -0.3)
        self.mock_log.assert_called_once()

    def test_objective_prunes_before_training_on_config_rule(self) -> None:
        # Config-level prune rules should short-circuit the trial before fit() so obviously bad candidates do not waste training time.
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

        self.client.task.make_fit_plan.assert_not_called()
        self.mock_log.assert_called_once()
        self.assertEqual(self.mock_log.call_args.kwargs["prune_rule"], "latency_budget")
        self.assertEqual(
            self.mock_log.call_args.kwargs["prune_reason"],
            "Latency exceeds deployment budget",
        )

    def test_objective_prunes_when_rule_metric_is_unavailable(self) -> None:
        # Unavailable prune metrics should fail closed and prune the trial instead of letting half-populated hardware results continue.
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

        self.client.task.make_fit_plan.assert_not_called()
        self.assertEqual(self.mock_log.call_args.kwargs["prune_rule"], "rule_0")
        self.assertIn("Configured prune metric unavailable", self.mock_log.call_args.kwargs["prune_reason"])

    def test_objective_uses_negative_one_rmse_sentinels_for_failed_trials(self) -> None:
        # Failed trials should log stable RMSE sentinels so CSV summaries can distinguish a failure from missing training output.
        metrics = {
            "error_code": HIL_MASTER_RAM_OVERFLOW,
            "ram_bytes": -1,
            "flash_bytes": -1,
            "arena_bytes": -1,
            "latency_ms": -1.0,
            "latency_budget_ms": -1.0,
            "energy_mj_per_inference": -1.0,
            "avg_power_mw": -1.0,
            "avg_current_ma": -1.0,
            "bus_voltage_v": -1.0,
        }
        self.client._hil_request = MagicMock(return_value=metrics)
        trial = DummyTrial()

        with self.assertRaises(optuna.TrialPruned):
            self.client.objective(trial)

        trial_outcome = self.mock_log.call_args.kwargs["trial_outcome"]
        self.assertIsInstance(trial_outcome, TrialOutcome)
        self.assertEqual(trial_outcome.task_metrics, {})

    def test_objective_does_not_swallow_generic_training_value_errors(self) -> None:
        # Unexpected training ValueErrors should still escape so genuine code bugs are not hidden behind prune logic.
        metrics = {
            "error_code": HIL_MASTER_SUCCESS,
            "ram_bytes": 512,
            "flash_bytes": 512,
            "arena_bytes": 1024,
            "latency_ms": 10.0,
            "latency_budget_ms": 20.0,
            "energy_mj_per_inference": -1.0,
            "avg_power_mw": -1.0,
            "avg_current_ma": -1.0,
            "bus_voltage_v": -1.0,
        }
        self.client._hil_request = MagicMock(return_value=metrics)
        trial = DummyTrial()

        with patch.object(self.client.task, "make_fit_plan", side_effect=ValueError("bad shape")):
            with self.assertRaisesRegex(ValueError, "bad shape"):
                self.client.objective(trial)

        self.mock_log.assert_not_called()

    def test_objective_converts_score_config_errors_into_prune_penalties(self) -> None:
        # Score-config evaluation errors should convert into prune penalties so search continues without masking the misconfiguration.
        metrics = {
            "error_code": HIL_MASTER_SUCCESS,
            "ram_bytes": 512,
            "flash_bytes": 512,
            "arena_bytes": 1024,
            "latency_ms": 10.0,
            "latency_budget_ms": 20.0,
            "energy_mj_per_inference": -1.0,
            "avg_power_mw": -1.0,
            "avg_current_ma": -1.0,
            "bus_voltage_v": -1.0,
        }
        self.client._hil_request = MagicMock(return_value=metrics)
        trial = DummyTrial()

        with patch(
            "nas_model_client.evaluate_score_config",
            side_effect=ScoreConfigEvaluationError("Metric 'latency_ms' is unavailable for scoring."),
        ):
            with self.assertRaises(optuna.TrialPruned):
                self.client.objective(trial)

        self.mock_log.assert_called_once()
        self.assertEqual(
            self.mock_log.call_args.kwargs["prune_reason"],
            "Training failed to produce valid metrics",
        )

    def test_objective_samples_cpu_clock_into_device_overrides(self) -> None:
        # Sampled CPU-clock choices should flow into device overrides so each trial evaluates the exact hardware point it sampled.
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
        # Null clock-option lists should avoid creating spurious overrides so default board clocks stay in control.
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
        # STM32 phase-one trials should tolerate arena sentinel values because that stage may not know the final arena size yet.
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

        self.client.task.make_fit_plan.assert_called_once()
        self.assertEqual(result, -0.3)
        self.mock_log.assert_called_once()
        logged_outcome = self.mock_log.call_args.kwargs["trial_outcome"]
        self.assertEqual(logged_outcome.task_metrics["rmse_total"], 0.3)

    def test_objective_returns_configured_multiobjective_tuple(self) -> None:
        """Configured multi-objective runs should return all objective values.

        Returns
        -------
        None
        """
        # Configured multi-objective trials should return all objective values in the order Optuna expects.
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

        result = self.client.objective(trial)

        self.assertEqual(result, (10.0, 512.0))
        self.mock_log.assert_called_once()

    def test_objective_train_false_uses_generic_metric_sentinels(self) -> None:
        # Train-disabled trials should still emit generic metric sentinels so downstream logs keep a stable shape.
        metrics = {
            "error_code": HIL_MASTER_SUCCESS,
            "ram_bytes": 512,
            "flash_bytes": 512,
            "arena_bytes": 1024,
            "latency_ms": 10.0,
        }
        self.client.metric_contract = TaskMetricContract(
            available_metric_names={"signed_bias", "signed_offset"},
            training_only_metric_names={"signed_bias", "signed_offset"},
            nonnegative_metric_names=set(),
            primary_metric_names={"signed_bias"},
        )
        self.client.config.training.train = False
        self.client.config.nas.score = Dict(
            type="multi-objective",
            metrics=Dict(),
            params=Dict(
                objectives=[
                    Dict(metric="signed_bias", direction="maximize"),
                    Dict(metric="signed_offset", direction="maximize"),
                ]
            ),
        )
        self.client._hil_request = MagicMock(return_value=metrics)
        trial = DummyTrial()

        result = self.client.objective(trial)

        self.assertEqual(result, (-1.0, -1.0))
        logged_outcome = self.mock_log.call_args.kwargs["trial_outcome"]
        self.assertEqual(
            logged_outcome.task_metrics,
            {"signed_bias": -1.0, "signed_offset": -1.0},
        )

    def test_objective_portenta_forwards_device_options_to_hardware_specs(self) -> None:
        # Portenta trials should forward resolved board options when requesting hardware limits.
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
        # Portenta hardware-limit resolution should require an explicit target core instead of guessing one.
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
        # Smoke tests should restore mutated config flags even when the validation run raises.
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
        # Smoke tests should honor a loaded multi-objective config instead of forcing a scalar study shape.
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
        self.assertTrue(kwargs["load_if_exists"])

    def test_smoke_test_uses_loaded_scalar_config(self) -> None:
        # Scalar smoke tests should preserve the loaded scalar study direction.
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
        self.assertTrue(mock_create.call_args.kwargs["load_if_exists"])

    def test_smoke_test_defaults_to_loaded_hil_setting(self) -> None:
        # Smoke tests should inherit the loaded HIL flag so validation exercises the same execution mode the real run will use.
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
        # Derived RMSE metrics should be rejected when training is disabled because no training pass can produce them.
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

    def test_smoke_test_preserves_existing_db_and_log_before_validation(self) -> None:
        # Smoke-test validation should not clobber an existing DB or log before the run is known to be valid.
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

        self.assertTrue(db_path.exists())
        self.assertTrue(log_path.exists())
        self.assertEqual(db_path.read_text(encoding="utf-8"), "stale-db")
        self.assertEqual(log_path.read_text(encoding="utf-8"), "stale-log")
        self.assertFalse(mock_create.called)

    def test_copy_run_config_skips_same_file(self) -> None:
        # Copying the run config should no-op when the source file already lives in the artifacts directory.
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
        # The orchestration loop should keep retrying until it reaches the requested number of completed trials, not just total attempts.
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
        # The orchestration loop should still stop at the global trial cap even if completed-trial target has not been met.
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

class TrainBestTrialTests(unittest.TestCase):
    """Best-trial retraining should honor the task abstraction."""

    def test_train_best_trial_uses_task_fit_plan_with_override_task_settings(self) -> None:
        # Best-trial retraining should build its fit plan from the override task settings instead of reusing stale defaults.
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            client = _build_test_client(base_dir=base)
            built_model = client.model_family.build_model.return_value
            built_model.fit.return_value = SimpleNamespace(history={"loss": [1.0], "val_loss": [0.5]})
            fit_task = MagicMock()
            fit_task.make_fit_plan.return_value = FitPlan(
                fit_kwargs={
                    "x": client.dataset_bundle.train.inputs,
                    "y": [client.dataset_bundle.train.targets["velx"], client.dataset_bundle.train.targets["vely"]],
                    "validation_data": (
                        client.dataset_bundle.val.inputs,
                        [client.dataset_bundle.val.targets["velx"], client.dataset_bundle.val.targets["vely"]],
                    ),
                    "shuffle": True,
                },
                callbacks=["task-callback"],
                monitor_metric="val_loss",
            )
            best_trial = SimpleNamespace(
                params={
                    "nb_filters": 2,
                    "kernel_size": 2,
                    "dropout_rate": 0.1,
                    "use_skip_connections": True,
                    "norm_flag": True,
                    "dilations_index": 0,
                }
            )

            with patch.object(client, "_instantiate_task", return_value=fit_task) as instantiate_mock, patch(
                "nas_model_client.optuna.load_study",
                return_value=SimpleNamespace(best_trial=best_trial),
            ):
                history = client.train_best_trial(
                    study_storage="sqlite:///optuna.db",
                    study_name="demo",
                    patience=7,
                    checkpoint_path=base / "best.keras",
                    history_path=base / "history.json",
                )

            instantiate_mock.assert_called_once_with(
                client.task_name,
                client.config,
                client.task_config,
                checkpoint_path=base / "best.keras",
                early_stopping_patience=7,
            )
            fit_task.make_fit_plan.assert_called_once_with(
                client.dataset_bundle,
                client.task_config,
                client.target_spec,
            )
            built_model.fit.assert_called_once()
            fit_call = built_model.fit.call_args.kwargs
            self.assertEqual(fit_call["callbacks"], ["task-callback"])
            self.assertEqual(fit_call["epochs"], client.config.training.model_epochs)
            self.assertEqual(fit_call["batch_size"], 256)
            self.assertEqual(history["loss"], [1.0])


class PlotTrainingHistoryTests(unittest.TestCase):
    """Plotting helpers should emit PNGs without requiring a display."""

    def test_plot_training_history_writes_pngs(self) -> None:
        # Training-history plots should materialize PNG artifacts so the run bundle is self-contained.
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
        # Checkpoint evaluation should persist its metrics file so post-train analysis can run offline.
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
            client.dataset_bundle.test = DataSplit(
                inputs=client.test_data.inputs,
                targets={"velx": gt_vx, "vely": gt_vy},
                metadata={},
            )
            client.task.evaluate.return_value = EvaluationResult(
                metrics={"rmse_vel_x": 0.0, "rmse_vel_y": 0.0, "rmse_total": 0.0},
                predictions=[gt_vx, gt_vy],
            )

            metrics_path = base / "metrics.json"
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

    def test_evaluate_checkpoint_preserves_task_defined_metric_names(self) -> None:
        # Checkpoint evaluation should preserve task-defined metric names instead of flattening them into generic labels.
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            client = _build_test_client(base_dir=base)
            client.task.evaluate.return_value = EvaluationResult(
                metrics={"f1_macro": np.float32(0.8), "loss": np.float64(0.25)},
                predictions=None,
            )

            metrics_path = base / "metrics.json"
            metrics = client.evaluate_checkpoint(
                checkpoint_path=base / "ckpt.keras",
                metrics_path=metrics_path,
                export_tflite=False,
            )

            self.assertAlmostEqual(metrics["f1_macro"], 0.8)
            self.assertAlmostEqual(metrics["loss"], 0.25)
            self.assertNotIn("rmse_vel_x", metrics)
            with metrics_path.open() as handle:
                persisted = json.load(handle)
            self.assertAlmostEqual(persisted["f1_macro"], 0.8)
            self.assertAlmostEqual(persisted["loss"], 0.25)

    def test_evaluate_checkpoint_exports_tflite_when_requested(self) -> None:
        # Checkpoint evaluation should export TFLite when requested so downstream deployment steps do not need a second conversion pass.
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            client = _build_test_client(base_dir=base)
            calibration_inputs = np.full((3, 16, 3), 7.0, dtype=np.float32)
            client.dataset_bundle.calibration = DataSplit(
                inputs=calibration_inputs,
                targets={"velx": np.zeros((3, 1), dtype=np.float32), "vely": np.zeros((3, 1), dtype=np.float32)},
                metadata={},
            )
            gt_vx = np.zeros((2, 1), dtype=np.float32)
            gt_vy = np.zeros((2, 1), dtype=np.float32)
            client.dataset_bundle.test = DataSplit(
                inputs=np.zeros((2, 1, 1), dtype=np.float32),
                targets={"velx": gt_vx, "vely": gt_vy},
                metadata={},
            )
            client.training_data = SimpleNamespace(inputs=np.full((1, 1, 1), 99.0, dtype=np.float32))
            client.task.evaluate.return_value = EvaluationResult(
                metrics={"rmse_vel_x": 0.0, "rmse_vel_y": 0.0, "rmse_total": 0.0},
                predictions=[gt_vx, gt_vy],
            )

            tflite_path = base / "model.tflite"
            with patch("nas_model_client.convert_to_tflite_model") as mock_convert:
                client.evaluate_checkpoint(
                    checkpoint_path=base / "ckpt.keras",
                    metrics_path=base / "metrics.json",
                    export_tflite=True,
                    tflite_path=tflite_path,
                )

            self.assertTrue(mock_convert.called)
            self.assertEqual(mock_convert.call_args.kwargs["output_name"], tflite_path)
            self.assertTrue(
                np.array_equal(mock_convert.call_args.kwargs["training_data"], calibration_inputs)
            )


class TrajectoryMetricsTests(unittest.TestCase):
    """Trajectory metrics/plots should be generated with stubbed models."""

    def test_trajectory_metrics_and_plots_zero_error(self) -> None:
        # Perfect trajectories should evaluate to zero error and still emit the expected plots.
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            client = _build_test_client(base_dir=base)
            length = 4
            vx = np.full((length, 1), 0.5, dtype=np.float32)
            vy = np.full((length, 1), 0.5, dtype=np.float32)
            client.dataset_bundle.metadata.update(
                {"sampling_rate_hz": 100, "window_size": 2, "stride": 1}
            )
            client.dataset_bundle.test = DataSplit(
                inputs=np.zeros((length, 1, 1), dtype=np.float32),
                targets={"velx": vx, "vely": vy},
                metadata={"size_of_each": [length], "x0": [0.0], "y0": [0.0]},
            )

            class FakeModel:
                def predict(self, _inputs):
                    return [vx, vy]

            client.model_family.load_model.return_value = FakeModel()
            with patch.object(client.model_family, "load_model", return_value=FakeModel()):
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

    def test_trajectory_metrics_and_plots_requires_odometry_metadata(self) -> None:
        # Trajectory analysis should fail fast when the evaluation metadata is missing the odometry fields it depends on.
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            client = _build_test_client(base_dir=base)
            vx = np.full((2, 1), 0.5, dtype=np.float32)
            vy = np.full((2, 1), 0.5, dtype=np.float32)
            client.dataset_bundle.metadata.update(
                {"sampling_rate_hz": 100, "window_size": 2, "stride": 1}
            )
            client.dataset_bundle.test = DataSplit(
                inputs=np.zeros((2, 1, 1), dtype=np.float32),
                targets={"velx": vx, "vely": vy},
                metadata={},
            )

            with self.assertRaisesRegex(ValueError, "odometry-specific"):
                client.trajectory_metrics_and_plots(
                    checkpoint_path=base / "ckpt.keras",
                    plot_dir=base,
                    study_name="demo",
                )

    def test_trajectory_split_view_requires_velocity_targets(self) -> None:
        # Trajectory split views should require velocity targets so the plotted channels remain meaningful.
        cases = (
            {"class_id": np.array([0, 1], dtype=np.int32)},
            {"velx": None, "vely": np.zeros((2, 1), dtype=np.float32)},
            {"velx": np.zeros((2, 1), dtype=np.float32), "vely": None},
            ["not", "a", "mapping"],
        )
        for targets in cases:
            with self.subTest(targets=targets):
                client = _build_test_client()
                client.dataset_bundle.test = DataSplit(
                    inputs=np.zeros((2, 1, 1), dtype=np.float32),
                    targets=targets,
                    metadata={"size_of_each": [2], "x0": [0.0], "y0": [0.0]},
                )
                with self.assertRaisesRegex(ValueError, "velocity targets named 'velx' and 'vely'"):
                    client._trajectory_split_view()


class SummaryBundleTests(unittest.TestCase):
    """Summary bundle aggregation should fuse existing artifacts."""

    def test_write_summary_bundle_persists_expected_fields(self) -> None:
        # Summary bundles should persist the expected metadata fields so later reporting code can read one stable artifact format.
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
        # Closing the NAS client should shut down both the socket and the ZeroMQ context cleanly.
        client = _build_test_client()
        client.close()
        client.socket.close.assert_called_once_with(linger=0)
        client.context.term.assert_called_once()


if __name__ == "__main__":
    unittest.main()
