"""Top-level NAS orchestration client for TinyODOM experiments.

This module bridges the modular dataset/task/model-family registry system to
the current Optuna, HIL RPC, final-training, and artifact-reporting workflow.
"""

import argparse
import copy
from collections.abc import Mapping
import csv
from dataclasses import dataclass
import json
import logging
import shutil
import socket
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import zmq
from addict import Dict

import absl.logging
import matplotlib.pyplot as plt
import numpy as np
import optuna
import tensorflow as tf
from optuna.trial import TrialState
from tinyodom.hardware import (
    HIL_MASTER_ARENA_EXHAUSTED,
    HIL_MASTER_DEVICE_NOT_FOUND,
    HIL_MASTER_FATAL,
    HIL_MASTER_FLASH_OVERFLOW,
    HIL_MASTER_RAM_OVERFLOW,
    HIL_MASTER_SUCCESS,
    TFLiteSubprocessError,
    convert_to_tflite_model,
    predict_tflite_model_subprocess,
    return_hardware_specs,
)
from tinyodom.microcontrollers import (
    get_device as get_microcontroller_device,
    resolve_device_options,
)
from tinyodom.builtin_components import ensure_builtin_components_registered
from tinyodom.component_selection import cfg_get, resolve_component_selection
from tinyodom.cadence import resolve_batch_period_ms
from tinyodom.model import (
    NONNEGATIVE_METRICS,
    ScoreConfigEvaluationError,
    configured_quantization_mode,
    build_trial_outcome,
    apply_cadenced_metric_defaults,
    evaluate_prune_rules,
    evaluate_score_config,
    get_score_config_directions,
    is_multiobjective_score_config,
    log_trial,
    load_config,
    quantization_requires_calibration,
    require_logical_input_shape,
    TrialOutcome,
    DEFAULT_CONFIG_PATH,
    score_config_uses_training_metrics,
    set_error_code,
)
from tinyodom.pipeline_types import DataSplit, DatasetBundle, ModelBuildContext
from tinyodom.registry import dataset_registry, model_family_registry
from tinyodom.runtime_bootstrap import bootstrap_pipeline

tf.get_logger().setLevel(logging.ERROR)
absl.logging.set_verbosity(absl.logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

logger = logging.getLogger(__name__)

RUNNER_OWNED_TRIAL_PARAM_KEYS = frozenset({"cpu_clock_mhz_index", "quantization_mode"})
COMPILE_DERIVED_METRICS = frozenset(
    {"ram_bytes", "flash_bytes", "external_flash_bytes", "arena_bytes"}
)
RUNTIME_ONLY_METRICS = frozenset(
    {
        "latency_ms",
        "energy_mj_per_inference",
        "avg_power_mw",
        "avg_current_ma",
        "bus_voltage_v",
        "clock_hz",
        "harness_latency_ms",
    }
)


@dataclass(frozen=True)
class NASMetricDependencies:
    """Classified metric dependencies for one NAS score/prune policy.

    Parameters
    ----------
    metrics : frozenset[str]
        All active score/prune metrics and their recursively referenced
        derived-metric dependencies.
    compile_derived : frozenset[str]
        Metrics that require the compile-only HIL backend path.
    runtime_only : frozenset[str]
        Metrics that require real on-device runtime measurement.
    """

    metrics: frozenset[str]
    compile_derived: frozenset[str]
    runtime_only: frozenset[str]


def _family_trial_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Return Optuna parameters that belong to model-family decoding.

    Parameters
    ----------
    params : Mapping[str, Any]
        Raw Optuna trial parameters, including both model-family search values
        and runner-owned deployment/runtime choices.

    Returns
    -------
    dict[str, Any]
        Parameters safe to pass to ``ModelFamilyABC.decode_trial_hparams``.
    """

    return {key: value for key, value in params.items() if key not in RUNNER_OWNED_TRIAL_PARAM_KEYS}


class NASModelClient:
    """Client that orchestrates HIL-assisted NAS, training, and evaluation.

    This class wires together configuration loading, hardware-in-the-loop (HIL)
    measurements, Optuna-based Neural Architecture Search (NAS), model
    training, evaluation, trajectory analysis, and artifact export for the
    TinyODOM-EX workflow.

    It manages a ZeroMQ REQ socket to a HIL server that compiles and flashes
    candidate models to the target board, returning resource usage (RAM/flash,
    arena), latency or energy, and error codes used to prune infeasible trials.
    When a candidate passes resource checks, the corresponding Keras/TCN model
    is trained on the OXIOD dataset splits and scored.

    Parameters
    ----------
    config_path : Path | str, optional
        Path to the NAS configuration YAML. Defaults to
        ``src/config/nas_config.yaml`` via ``DEFAULT_CONFIG_PATH``. The
        configuration
        controls data paths, device settings, NAS options (single vs
        multi-objective), training schedules, output directories, and network
        timeouts.

    Attributes
    ----------
    config_path : Path
        Resolved path to the configuration YAML used by this instance.
    config : addict.Dict
        Parsed configuration with derived fields (e.g., model/checkpoint paths,
        dropout choices). Accessed via dot-notation.
    context : zmq.Context
        Shared ZMQ context for the HIL communication.
    socket : zmq.Socket
        REQ socket used to send hyperparameters and receive HIL metrics. Send
        and receive timeouts are configured from the YAML.
    study_name : str
        Name used for Optuna study registration and artifact prefixes.

    Notes
    -----
    - Multi-objective NAS is enabled via ``config.nas.score.type``. When true,
      NSGA-II is used with the configured objective directions. Otherwise, a
      single-objective TPE sampler is used.
    - Device resource caps (RAM/flash) are checked against board specs returned
      by ``return_hardware_specs``. Trials exceeding limits are pruned.
    - Artifacts (trials CSV, training history, plots, metrics, optional TFLite)
      are written under ``config.outputs.models_dir``.

    Examples
    --------
    Basic smoke test (no HIL, quick validation):

    >>> client = NASModelClient()
    >>> client.smoke_test(trials=3, epochs=3, study_name="smoke")

    Full scoring workflow with local SQLite storage:

    >>> client = NASModelClient()
    >>> client.run_scoring_nas(study_name="tinyodom_nas_study")
    """
    def __init__(self, config_path: Path=DEFAULT_CONFIG_PATH):
        """Initialize NAS state, datasets, and the HIL client socket.

        Parameters
        ----------
        config_path : Path, optional
            Configuration file used to load dataset paths, NAS settings, and
            network endpoints.

        Returns
        -------
        None

        Notes
        -----
        Initialization now performs a single task-aware bootstrap. The config
        is parsed once, the active dataset/task/model-family pipeline is
        bootstrapped, and only then is the NAS policy validated against the
        concrete task metric contract.
        """
        self.config_path = Path(config_path)
        ensure_builtin_components_registered()
        self.config = load_config(self.config_path)
        self._attach_bootstrapped_pipeline(bootstrap_pipeline(self.config))

        if self.config.device.hil is False:
            logger.warning("HIL is disabled in the configuration.")
        if is_multiobjective_score_config(self.config.nas.score):
            logger.info("Using multi-objective NAS.")
        else:
            logger.info("Using single-objective NAS.")

        self.context = None
        self.socket = None

        self.study_name = "default_study"

    def _attach_bootstrapped_pipeline(self, pipeline: Any) -> None:
        """Attach one bootstrapped modular pipeline to this client.

        Parameters
        ----------
        pipeline : Any
            Bootstrap result containing selection metadata plus instantiated
            dataset, task, model-family, and task metric contract objects.

        Returns
        -------
        None
            The client state is populated from the supplied bootstrap result.
        """

        self.dataset = pipeline.dataset
        self.dataset_name = pipeline.selection["dataset_name"]
        self.task = pipeline.task
        self.task_name = pipeline.selection["task_name"]
        self.model_family = pipeline.model_family
        self.model_family_name = pipeline.selection["model_family_name"]
        self.dataset_bundle = pipeline.bundle
        self.target_spec = pipeline.target_spec
        self.metric_contract = pipeline.metric_contract
        self.model_build_context = pipeline.model_build_context
        self.dataset_config = pipeline.selection["dataset_config"]
        self.task_config = pipeline.selection["task_config"]
        self.model_config = pipeline.selection["model_config"]

    @staticmethod
    def _normalize_config_value(value: Any) -> Any:
        """Normalize config-like values into plain comparison-friendly shapes.

        Parameters
        ----------
        value : Any
            Config-like value that may contain mappings, namespaces, or paths.

        Returns
        -------
        Any
            Plain recursively normalized value.
        """
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): NASModelClient._normalize_config_value(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [NASModelClient._normalize_config_value(item) for item in value]
        if hasattr(value, "__dict__") and not isinstance(value, type):
            return {
                str(key): NASModelClient._normalize_config_value(val)
                for key, val in vars(value).items()
            }
        return value

    def _resolve_component_selection(self, config: Any) -> dict[str, Any]:
        """Resolve component names and local config subtrees for one config.

        Parameters
        ----------
        config : Any
            Fully loaded global configuration.

        Returns
        -------
        dict[str, Any]
            Component names plus local configuration payloads.

        Notes
        -----
        The resolved selection is compared across the preliminary and final
        config loads so the client can decide whether the dataset bundle can be
        reused or must be reloaded under a different dataset config.
        """
        return resolve_component_selection(config)

    def _same_component_selection(
        self,
        left: dict[str, Any],
        right: dict[str, Any],
    ) -> bool:
        """Return whether two resolved component selections are equivalent.

        Parameters
        ----------
        left : dict[str, Any]
            Preliminary component selection.
        right : dict[str, Any]
            Authoritative component selection.

        Returns
        -------
        bool
            ``True`` when both selections target the same dataset config.
        """
        return (
            left["dataset_name"] == right["dataset_name"]
            and self._normalize_config_value(left["dataset_config"])
            == self._normalize_config_value(right["dataset_config"])
        )

    def _instantiate_task(
        self,
        task_name: str,
        config: Any,
        task_config: Any,
        *,
        checkpoint_path: Path | None = None,
        early_stopping_patience: int | None = None,
    ) -> Any:
        """Instantiate one task component using the explicit runtime contract.

        Parameters
        ----------
        task_name : str
            Registered task name.
        config : Any
            Global configuration carrying output paths.
        task_config : Any
            Task-local configuration subtree.

        Returns
        -------
        Any
            Instantiated task component.
        """
        from tinyodom.runtime_bootstrap import instantiate_task_component

        return instantiate_task_component(
            task_name,
            config,
            task_config,
            checkpoint_path=checkpoint_path,
            early_stopping_patience=early_stopping_patience,
        )

    def _load_dataset_bundle(
        self,
        dataset_name: str,
        dataset_config: Any,
    ) -> tuple[Any | None, DatasetBundle]:
        """Instantiate one dataset and load its normalized bundle.

        Parameters
        ----------
        dataset_name : str
            Registered dataset name.
        dataset_config : Any
            Dataset-local configuration subtree.

        Returns
        -------
        tuple[Any | None, DatasetBundle]
            Loaded dataset instance plus bundle for the active run.
        """
        dataset_cls = dataset_registry.get(dataset_name)
        dataset = dataset_cls()
        dataset.validate_config(dataset_config)
        bundle = dataset.load(dataset_config)
        print("Imported Training Data")
        print("Imported Validation Data")
        print("Imported Test Data")
        return dataset, bundle

    def _coerce_loaded_dataset_bundle(
        self,
        dataset_name: str,
        loaded: Any,
    ) -> tuple[Any | None, DatasetBundle]:
        """Normalize dataset-loader returns for backward-compatible callers.

        Parameters
        ----------
        dataset_name : str
            Registered dataset name associated with ``loaded``.
        loaded : Any
            Result returned by ``_load_dataset_bundle(...)``.

        Returns
        -------
        tuple[Any | None, DatasetBundle]
            Dataset instance plus normalized dataset bundle.

        Notes
        -----
        ``_load_dataset_bundle(...)`` now returns ``(dataset, bundle)``, but
        some tests and overrides may still return only ``DatasetBundle``.
        Accept both shapes so those callers do not need to change in lockstep.
        """
        if isinstance(loaded, tuple) and len(loaded) == 2:
            return loaded
        return None, loaded

    def _build_model_context(
        self,
        bundle: DatasetBundle,
        target_spec: Any,
    ) -> ModelBuildContext:
        """Build the normalized model-family context for one dataset/task pair.

        Parameters
        ----------
        bundle : DatasetBundle
            Active dataset bundle.
        target_spec : Any
            Task-owned target specification.

        Returns
        -------
        ModelBuildContext
            Normalized model-build context.
        """
        return ModelBuildContext(
            input_shape=bundle.input_shape,
            input_dtype=bundle.input_dtype,
            target_spec=target_spec,
            dataset_metadata=dict(bundle.metadata),
            task_metadata=dict(target_spec.metadata),
        )

    def _initialize_component_state(
        self,
        selection: dict[str, Any],
        dataset: Any,
        bundle: DatasetBundle,
    ) -> None:
        """Instantiate and validate the active dataset/task/model components.

        Parameters
        ----------
        selection : dict[str, Any]
            Resolved component names and local config payloads.
        dataset : Any
            Dataset instance that loaded ``bundle``. When ``None``, the active
            dataset component is instantiated from the registry.
        bundle : DatasetBundle
            Dataset bundle to attach to the client.

        Returns
        -------
        None

        Notes
        -----
        The caller may pass a previously loaded dataset bundle from the
        preliminary config pass. When the final component selection matches,
        that bundle is reused instead of reloading the dataset.
        """
        dataset_cls = dataset_registry.get(selection["dataset_name"])
        model_family_cls = model_family_registry.get(selection["model_family_name"])
        if dataset is None:
            dataset = dataset_cls()
        task = self._instantiate_task(selection["task_name"], self.config, selection["task_config"])
        model_family = model_family_cls()

        task.validate_config(selection["task_config"])
        model_family.validate_config(selection["model_config"])

        target_spec = task.build_target_spec(bundle, selection["task_config"])
        metric_contract = task.metric_contract(target_spec, selection["task_config"])
        model_build_context = self._build_model_context(bundle, target_spec)

        self.dataset = dataset
        self.dataset_name = selection["dataset_name"]
        self.task = task
        self.task_name = selection["task_name"]
        self.model_family = model_family
        self.model_family_name = selection["model_family_name"]
        self.dataset_bundle = bundle
        self.target_spec = target_spec
        self.metric_contract = metric_contract
        self.model_build_context = model_build_context
        self.dataset_config = selection["dataset_config"]
        self.task_config = selection["task_config"]
        self.model_config = selection["model_config"]

    @staticmethod
    def _cfg_get(container, key: str, default=None):
        """Read a config field from dict-like or namespace-like containers.

        Parameters
        ----------
        container : object
            Configuration container supporting either ``get(key, default)`` or
            attribute access.
        key : str
            Field name to retrieve.
        default : object, optional
            Value returned when ``key`` is not present.

        Returns
        -------
        object
            Retrieved field value or ``default`` when unavailable.
        """
        return cfg_get(container, key, default)

    def _score_is_multiobjective(self) -> bool:
        """Return whether the active config uses multi-objective scoring.

        Returns
        -------
        bool
            ``True`` when the active ``score`` block is multi-objective.
        """
        return is_multiobjective_score_config(self.config.nas.score)

    def _study_directions(self) -> list[str]:
        """Return Optuna directions for the active score config.

        Returns
        -------
        list[str]
            Direction list compatible with ``optuna.create_study``.
        """
        return get_score_config_directions(self.config.nas.score)

    def _study_metric_names(self) -> list[str]:
        """Return Optuna metric names for the active score config.

        Returns
        -------
        list[str]
            Metric names aligned with the objective return order.
        """
        if self._score_is_multiobjective():
            return [str(obj.metric) for obj in self.config.nas.score.params.objectives]
        return ["score"]

    def _hardware_limit_device_options(self) -> dict[str, str] | None:
        """Build board options required to resolve dynamic hardware limits.

        Parameters
        ----------
        None
            Reads device settings from ``self.config``.

        Returns
        -------
        dict[str, str] | None
            Portenta board options for dynamic limit resolution, or ``None``
            for boards with static limits.

        Raises
        ------
        RuntimeError
            If ``device.name`` is ``PORTENTA_H7`` and
            ``device.portenta.target_core`` is missing.
        """
        try:
            return resolve_device_options(
                str(self._cfg_get(self.config.device, "name", "")),
                self.config.device,
            )
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc

    def _probe_hil_endpoint(self, timeout_s: float = 5.0) -> None:
        """Fail fast if the HIL REP socket is unreachable.

        Parameters
        ----------
        timeout_s : float, optional
            Socket-connection timeout used for the preflight probe.

        Returns
        -------
        None

        Raises
        ------
        ConnectionError
            If the HIL server cannot be reached within ``timeout_s``.

        Notes
        -----
        This is only a lightweight reachability probe. The main request/response
        contract still flows through :meth:`_hil_request`.
        """
        host = self.config.network.host
        port = self.config.network.port
        try:
            with socket.create_connection((host, port), timeout=timeout_s):
                return
        except OSError as exc:
            raise ConnectionError(
                f"HIL server at {host}:{port} is unreachable. "
                "Is the board connected and is hil_server.py running?"
            ) from exc

    def _ensure_hil_socket(self) -> None:
        """Open the ZeroMQ REQ socket on first real HIL request.

        Returns
        -------
        None

        Notes
        -----
        Desktop/proxy runs may load network settings for completeness but never
        need HIL transport. Keeping socket setup lazy avoids noisy background
        connection retries when ``device.hil`` and compile proxy collection are
        disabled.
        """

        if self.socket is not None:
            return
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.RCVTIMEO = self.config.network.recv_timeout_sec * 1000
        self.socket.SNDTIMEO = self.config.network.send_timeout_sec * 1000
        endpoint = f"tcp://{self.config.network.host}:{self.config.network.port}"
        self.socket.connect(endpoint)
        print(f"[REQ] Connected to HIL server at {endpoint}")

    def _hil_request(self, payload):
        """Send a HIL request payload to the HIL server and receive metrics.

        Parameters
        ----------
        payload : dict
            Structured request payload containing model hyperparameters and any
            device-option overrides.

        Returns
        -------
        dict
            Dictionary containing metrics like ram_bytes, flash_bytes, latency_ms,
            etc.

        Raises
        ------
        RuntimeError
            If the HIL server times out or cannot be reached.
        """
        print(f"[REQ] Sending payload to {self.config.network.host}:{self.config.network.port}: {payload}")

        try:
            self._ensure_hil_socket()
            self.socket.send_json(payload)
            metrics = self.socket.recv_json()
            print(f"[REQ] Received metrics: {metrics}")
            return metrics
        except zmq.error.Again as ex:
            print(f"[REQ] Timed out waiting for metrics after {self.config.network.recv_timeout_sec} seconds")
            # TODO: This is probably bad and should be handled more gracefully.
            raise RuntimeError("Timeout waiting for metrics from HIL server") from ex
        except zmq.error.ZMQError as ex:
            raise RuntimeError(
                f"Failed to reach HIL server at {self.config.network.host}:{self.config.network.port}"
            ) from ex
        finally:
            # Give stdout time to flush before tearing the socket down.
            time.sleep(0.1)

    def _build_runtime_metadata(
        self,
        flops: int,
        batch_size: int,
    ) -> Dict:
        """Build runtime-owned request metadata for HIL and scoring paths.

        Parameters
        ----------
        flops : int
            FLOP count for the built model.
        batch_size : int
            Runner-owned batch size.

        Returns
        -------
        Dict
            Runtime-owned metadata used by HIL transport, pruning, scoring,
            and CSV logging.
        """
        timesteps, input_dim = require_logical_input_shape(
            None if self.model_build_context is None else self.model_build_context.input_shape
        )
        return Dict(
            {
                "batch_size": int(batch_size),
                "timesteps": timesteps,
                "input_dim": input_dim,
                "flops": int(flops),
            }
        )

    def _resolve_trial_quantization_mode(
        self,
        trial: optuna.Trial,
        *,
        allow_search: bool = True,
    ) -> str:
        """Resolve the deployment quantization mode for one NAS trial.

        Parameters
        ----------
        trial : optuna.Trial
            Trial object used only when quantization search is enabled.
        allow_search : bool, optional
            Whether this trial has a deployment backend or TFLite validation
            path where quantization can affect behavior.

        Returns
        -------
        str
            Selected deployment quantization mode.
        """

        quantization = self.config.training.quantization
        if allow_search and bool(getattr(quantization, "search", False)):
            choices = list(getattr(quantization, "choices", []))
            return str(trial.suggest_categorical("quantization_mode", choices))
        return configured_quantization_mode(self.config)

    @staticmethod
    def _metric_is_runtime_only(metric_name: str) -> bool:
        """Return whether a metric requires real on-device runtime data.

        Parameters
        ----------
        metric_name : str
            Metric name from the NAS score or prune policy.

        Returns
        -------
        bool
            ``True`` when the metric cannot be produced by desktop or
            compile-only execution.
        """

        return metric_name in RUNTIME_ONLY_METRICS or metric_name.startswith("cadenced_")

    def _classify_nas_metric_dependencies(self) -> NASMetricDependencies:
        """Classify active score/prune metric dependencies for HIL decisions.

        Returns
        -------
        NASMetricDependencies
            Active dependencies grouped by local, compile-derived, and
            runtime-only availability.
        """

        score_config = self.config.nas.score
        score_metrics = getattr(score_config, "metrics", Dict())
        visited: set[str] = set()

        def _add_reference(reference: Any, stack: tuple[str, ...] = ()) -> None:
            """Add a typed metric reference to the active dependency set."""
            if reference is not None and getattr(reference, "type", None) == "metric":
                _add_metric(str(reference.metric), stack)

        def _add_metric(metric_name: str, stack: tuple[str, ...] = ()) -> None:
            """Add one metric and recursively add derived dependencies."""
            if metric_name in stack:
                return
            visited.add(metric_name)
            if metric_name not in score_metrics:
                return
            metric_cfg = score_metrics[metric_name]
            metric_type = str(getattr(metric_cfg, "type", "")).strip().lower()
            if metric_type == "add":
                for child_metric in getattr(metric_cfg, "metrics", []):
                    _add_metric(str(child_metric), stack + (metric_name,))
            elif metric_type == "energy-budget-from-power":
                _add_reference(getattr(metric_cfg, "power_mw", None), stack + (metric_name,))
                _add_reference(getattr(metric_cfg, "duration_ms", None), stack + (metric_name,))

        if is_multiobjective_score_config(score_config):
            for objective in getattr(score_config.params, "objectives", []):
                _add_metric(str(objective.metric))
        else:
            for term in getattr(score_config.params, "terms", []):
                _add_metric(str(term.metric))
                _add_reference(getattr(term, "reference", None))

        for rule_cfg in getattr(self.config.nas.prune, "rules", []):
            _add_metric(str(rule_cfg.metric))
            _add_reference(getattr(rule_cfg, "reference", None))

        compile_derived = {metric for metric in visited if metric in COMPILE_DERIVED_METRICS}
        runtime_only = {metric for metric in visited if self._metric_is_runtime_only(metric)}
        return NASMetricDependencies(
            metrics=frozenset(visited),
            compile_derived=frozenset(compile_derived),
            runtime_only=frozenset(runtime_only),
        )

    def _should_collect_compile_metrics(
        self,
        dependencies: NASMetricDependencies,
    ) -> bool:
        """Return whether this trial should request HIL/compile metrics.

        Parameters
        ----------
        dependencies : NASMetricDependencies
            Classified dependencies for the active NAS policy.

        Returns
        -------
        bool
            ``True`` when ``objective`` should call ``_hil_request``.

        Raises
        ------
        ValueError
            If a non-HIL policy references runtime-only metrics, or explicitly
            disables compile while depending on compile-derived metrics.
        """

        if bool(self.config.device.hil):
            return True

        if dependencies.runtime_only:
            metric_list = ", ".join(sorted(dependencies.runtime_only))
            raise ValueError(
                "device.hil=false cannot score or prune on runtime-only metric(s): "
                f"{metric_list}. Enable device.hil or remove those metrics."
            )

        compile_policy = str(
            self._cfg_get(self.config.device, "compile_when_hil_disabled", "auto")
        ).strip().lower()
        if compile_policy == "true":
            return True
        if compile_policy == "false":
            if dependencies.compile_derived:
                metric_list = ", ".join(sorted(dependencies.compile_derived))
                raise ValueError(
                    "device.compile_when_hil_disabled=false cannot satisfy "
                    f"compile-derived metric(s): {metric_list}."
                )
            return False
        if compile_policy != "auto":
            raise ValueError("device.compile_when_hil_disabled must be one of: auto, true, false.")
        return bool(dependencies.compile_derived)

    def _latency_budget_metric_value(self) -> float:
        """Resolve the local latency-budget metric for desktop trials.

        Returns
        -------
        float
            Positive latency budget in milliseconds, or ``-1.0`` when no
            dataset/device cadence contract is available.
        """

        try:
            return float(
                resolve_batch_period_ms(
                    self.dataset_config,
                    getattr(self.dataset_bundle, "metadata", None),
                    self.config.device,
                )
            )
        except ValueError:
            return -1.0

    def _synthesize_desktop_success_metrics(self) -> dict[str, Any]:
        """Build a complete successful metrics payload for pure desktop trials.

        Returns
        -------
        dict[str, Any]
            Metrics dictionary with logging-required keys populated using
            desktop-safe values and unavailable sentinels.
        """

        metrics: dict[str, Any] = {
            "ram_bytes": -1,
            "flash_bytes": -1,
            "external_flash_bytes": -1,
            "arena_bytes": -1,
            "latency_ms": -1.0,
            "latency_budget_ms": self._latency_budget_metric_value(),
            "energy_mj_per_inference": -1.0,
            "avg_power_mw": -1.0,
            "avg_current_ma": -1.0,
            "bus_voltage_v": -1.0,
            "cpu_clock_mhz_requested": -1,
            "clock_hz": -1.0,
            "hil_enabled": False,
            "energy_aware": False,
            "weight_storage_mode": "embedded",
        }
        set_error_code(metrics, HIL_MASTER_SUCCESS)
        apply_cadenced_metric_defaults(metrics, metrics)
        return metrics

    def _export_tflite_for_evaluation(
        self,
        *,
        model: Any,
        split_name: str,
        quantization_mode: str,
    ) -> Path:
        """Export a temporary TFLite model for host-side evaluation.

        Parameters
        ----------
        model : Any
            Keras model to export.
        split_name : str
            Evaluation split label used in the temporary filename.
        quantization_mode : str
            Deployment quantization mode for export.

        Returns
        -------
        pathlib.Path
            Path to the written TFLite flatbuffer.
        """

        output_path = self._artifacts_dir() / f"{self.study_name}_{split_name}_eval.tflite"
        representative_split = None
        if quantization_requires_calibration(quantization_mode):
            representative_split = self.dataset_bundle.calibration or self.dataset_bundle.train
        convert_to_tflite_model(
            model=model,
            training_data=None if representative_split is None else representative_split.inputs,
            quantization_mode=quantization_mode,
            output_name=output_path,
        )
        return output_path

    def _evaluate_model_with_backend(
        self,
        *,
        model: Any,
        split: DataSplit,
        split_name: str,
        quantization_mode: str,
        evaluation_backend: str = "keras",
    ):
        """Evaluate a model through the selected host backend.

        Parameters
        ----------
        model : Any
            Keras model to evaluate or export.
        split : DataSplit
            Dataset split to score.
        split_name : str
            Label used for temporary artifacts.
        quantization_mode : str
            Deployment quantization mode used by TFLite evaluation.
        evaluation_backend : {"keras", "tflite"}, optional
            Host evaluation backend.

        Returns
        -------
        EvaluationResult
            Task-owned evaluation result.
        """

        if evaluation_backend == "keras":
            return self.task.evaluate(model, split, self.task_config, self.target_spec)
        if evaluation_backend != "tflite":
            raise ValueError("evaluation_backend must be 'keras' or 'tflite'.")
        tflite_path = self._export_tflite_for_evaluation(
            model=model,
            split_name=split_name,
            quantization_mode=quantization_mode,
        )
        predictions = predict_tflite_model_subprocess(tflite_path, split.inputs)
        return self.task.evaluate_predictions(
            predictions,
            split,
            self.task_config,
            self.target_spec,
        )

    @staticmethod
    def _apply_non_hil_success_sentinels(metrics: dict[str, Any]) -> None:
        """Apply legacy non-HIL sentinels before scoring or logging.

        Parameters
        ----------
        metrics : dict[str, Any]
            Runtime metrics dictionary to mutate in place.

        Returns
        -------
        None
        """
        if (not metrics["hil_enabled"]) and metrics.get("error_code", 0) == 0:
            metrics["latency_ms"] = -1
            set_error_code(metrics, 1)

    @staticmethod
    def _sync_task_metrics(metrics: dict[str, Any], task_metrics: dict[str, Any]) -> None:
        """Write task-owned evaluation metrics back into the shared metrics dict.

        Parameters
        ----------
        metrics : dict[str, Any]
            Shared runtime metrics dictionary.
        task_metrics : dict[str, Any]
            Task-owned evaluation metrics.

        Returns
        -------
        None
        """
        for metric_name, raw_value in task_metrics.items():
            if isinstance(raw_value, np.generic):
                metrics[metric_name] = raw_value.item()
            else:
                metrics[metric_name] = raw_value

    def _resolve_dataset_numeric_setting(
        self,
        key: str,
        *,
        split: DataSplit | None = None,
    ) -> float:
        """Resolve one numeric dataset setting from metadata or config.

        Parameters
        ----------
        key : str
            Numeric dataset field name to resolve.
        split : DataSplit | None, optional
            Optional split whose metadata should take precedence.

        Returns
        -------
        float
            Resolved numeric value.

        Raises
        ------
        ValueError
            If the value cannot be resolved or is not positive and finite.
        """

        raw_value = None
        if split is not None:
            raw_value = split.metadata.get(key)
        if raw_value in (None, ""):
            raw_value = self.dataset_bundle.metadata.get(key)
        if raw_value in (None, ""):
            raw_value = self._cfg_get(self.dataset_config, key, None)
        if raw_value in (None, "") or isinstance(raw_value, bool):
            raise ValueError(f"Unable to resolve dataset setting '{key}' for the active configuration.")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Dataset setting '{key}' must be numeric for the active configuration."
            ) from exc
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"Dataset setting '{key}' must be a positive finite value for the active configuration."
            )
        return value

    def _require_trajectory_split(self) -> DataSplit:
        """Return the active odometry test split for trajectory reporting.

        Returns
        -------
        DataSplit
            Test split with the odometry-specific targets and metadata needed
            by the trajectory reporting helpers.

        Raises
        ------
        ValueError
            If the active test split does not expose the odometry metadata
            required by the trajectory helper.
        """

        split = self.dataset_bundle.test
        if split is None:
            raise ValueError("Trajectory reporting requires a held-out test split.")
        if not isinstance(split.targets, Mapping):
            raise ValueError(
                "Trajectory reporting remains odometry-specific and requires "
                "velocity targets named 'velx' and 'vely'."
            )
        if split.targets.get("velx") is None or split.targets.get("vely") is None:
            raise ValueError(
                "Trajectory reporting remains odometry-specific and requires "
                "velocity targets named 'velx' and 'vely'."
            )
        required_fields = ("size_of_each", "x0", "y0")
        missing = [
            field_name
            for field_name in required_fields
            if split.metadata.get(field_name) in (None, "")
        ]
        if missing:
            raise ValueError(
                "Trajectory reporting remains odometry-specific and requires test-split metadata "
                f"for: {', '.join(missing)}."
            )
        return split

    def objective(self, trial: optuna.Trial) -> float | tuple:
        """Optimize TinyODOM architecture and training hyperparameters.

        This objective samples model hyperparameters (e.g., filters, kernel size,
        dilations) via Optuna, builds the corresponding TCN model to estimate
        FLOPs, queries a hardware-in-the-loop (HIL) server for resource/latency
        metrics, and—when the candidate passes resource and configured
        feasibility gates—trains and scores the model on the OXIOD dataset.
        Trials are pruned on HIL errors or resource violations. The returned
        objective is either a single score or a multi-objective tuple depending
        on configuration.

        Parameters
        ----------
        trial : optuna.Trial
            The Optuna trial object used to sample hyperparameters and report
            intermediate results. Hyperparameters include:
            - ``nb_filters`` (int): number of convolution filters.
            - ``kernel_size`` (int): convolution kernel size.
            - ``dropout_rate`` (float): dropout probability.
            - ``use_skip_connections`` (bool): enable residual/skip connections.
            - ``norm_flag`` (bool): enable normalization layers.
            - ``dilations_index`` (int): index into predefined dilation patterns.

        Returns
        -------
        float or tuple
            Returns either a single scalar score or a tuple of configured
            objective values, depending on ``config.nas.score.type``.

        Raises
        ------
        optuna.TrialPruned
            Raised to prune the trial when the HIL server reports fatal errors
            (e.g., flash/RAM overflow, arena exhaustion) or when resource
            checks fail.
        RuntimeError
            If the target device is not found or the HIL server times out/
            cannot be reached.

        Notes
        -----
        The trial phases are:
        1. sample/build the model family hyperparameters
        2. request HIL metrics for the candidate
        3. apply hardware limit and arena checks
        4. evaluate post-build/pre-fit prune rules
        5. either train/evaluate the task or synthesize task-metric sentinels
        6. validate objective values and log the trial

        Single-objective runs prune by raising ``optuna.TrialPruned``.
        Multi-objective HIL errors, resource failures, and feasibility gates
        instead log ``pruned=True`` and return direction-aware penalty tuples
        so Optuna records a complete trial with the configured objective shape.
        Sentinel conventions such as ``-1`` and ``10000.0`` are used to
        preserve legacy logging/scoring expectations when hardware or training
        metrics are unavailable.
        """
        artifacts_dir = self._artifacts_dir()
        log_path = artifacts_dir / self.config.outputs.log_file_name
        batch_size = 256
        metric_dependencies = self._classify_nas_metric_dependencies()
        collect_compile_metrics = self._should_collect_compile_metrics(metric_dependencies)
        family_hparams = self.model_family.sample_hparams(
            trial,
            self.model_build_context,
            self.model_config,
        )
        self.model_family.validate_hparams(
            family_hparams,
            self.model_build_context,
            self.model_config,
        )
        model = self.model_family.build_model(
            family_hparams,
            self.model_build_context,
            self.model_config,
        )
        self.task.validate_model_outputs(model, self.target_spec)
        self.task.compile_model(model, self.task_config, self.target_spec)
        flops = self.model_family.count_flops(
            model,
            self.model_build_context,
            self.model_config,
        )
        uses_quantized_deployment_path = collect_compile_metrics or bool(self.config.training.train)
        quantization_mode = self._resolve_trial_quantization_mode(
            trial,
            allow_search=uses_quantized_deployment_path,
        )
        runtime_metadata = self._build_runtime_metadata(
            flops,
            batch_size,
        )
        hyperparams = Dict({**family_hparams, **runtime_metadata})

        # Treat CPU clock as a trial variable, but keep it separate from model
        # hyperparameters because it changes board runtime conditions rather
        # than the network topology itself.
        cpu_clock_mhz_options = self._cfg_get(self.config.device, "cpu_clock_mhz_options", None)
        device_options_overrides = None
        if collect_compile_metrics and cpu_clock_mhz_options is not None:
            cpu_clock_mhz_index = trial.suggest_int(
                "cpu_clock_mhz_index",
                0,
                len(cpu_clock_mhz_options) - 1,
            )
            cpu_clock_mhz = int(cpu_clock_mhz_options[cpu_clock_mhz_index])
            device_options_overrides = {"cpu_clock_mhz": cpu_clock_mhz}
            logger.debug(
                "Sampled CPU clock override for trial: index=%s, value_mhz=%s, options=%s",
                cpu_clock_mhz_index,
                cpu_clock_mhz,
                cpu_clock_mhz_options,
            )

        # Ask the HIL server to evaluate the candidate for resource usage and
        # latency, or synthesize a desktop-only payload when the active policy
        # does not need compile-derived metrics.
        request_payload = {
            "family_hparams": family_hparams,
            "runtime_metadata": runtime_metadata,
            "quantization_mode": quantization_mode,
        }
        if device_options_overrides is not None:
            request_payload["device_options_overrides"] = device_options_overrides
        if collect_compile_metrics:
            metrics = self._hil_request(request_payload)
            metrics.setdefault("hil_enabled", bool(self.config.device.hil))
            metrics.setdefault("energy_aware", bool(self.config.training.energy_aware))
        else:
            metrics = self._synthesize_desktop_success_metrics()

        needs_hardware_limits = (
            collect_compile_metrics
            or bool({"max_ram_bytes", "max_flash_bytes"} & set(metric_dependencies.metrics))
        )
        device_options = None
        runtime_device = None
        max_ram = -1.0
        max_flash = -1.0
        if needs_hardware_limits:
            # Gets the hardware *estimated* specifications for the target device
            device_options = self._hardware_limit_device_options()
            max_ram, max_flash = return_hardware_specs(
                self.config.device.name,
                device_options=device_options,
            )
        metrics["max_ram_bytes"] = float(max_ram)
        metrics["max_flash_bytes"] = float(max_flash)
        if collect_compile_metrics:
            try:
                runtime_device = get_microcontroller_device(
                    str(self.config.device.name),
                    serial_port=self._cfg_get(self.config.device, "serial_port", None),
                    device_options=device_options,
                )
            except ValueError:
                runtime_device = None

        penalty_acc = -100.0
        task_nonnegative_metric_names = set(self.metric_contract.nonnegative_metric_names)
        effective_nonnegative_metric_names = set(NONNEGATIVE_METRICS) | task_nonnegative_metric_names
        # If no error code present (or timeout), treat as fatal error
        error_code = metrics.get("error_code", HIL_MASTER_FATAL)

        if error_code == HIL_MASTER_DEVICE_NOT_FOUND:
            serial_hint = self.config.device.serial_port or "<unset port>"
            raise RuntimeError(
                f"Upload failed: target device not found on {serial_hint}. "
                "Stopping NAS run so the board/serial port can be fixed."
            )

        def _report_if_supported(value: float) -> None:
            """Optuna trial.report is unsupported for multi-objective studies."""
            if not self._score_is_multiobjective():
                trial.report(value, step=0)

        def _fail_with_penalty(
            prune_reason: str,
            prune_rule: str = "",
        ):
            """Helper to prune with a penalty score and log the failure."""
            # Normalize missing metrics, build a loggable TrialOutcome, and
            # then diverge between scalar Optuna pruning and multi-objective
            # penalty returns.
            metrics.setdefault("latency_ms", 10000.0)
            metrics.setdefault("energy_mj_per_inference", 10000.0)
            metrics.setdefault("avg_power_mw", -1.0)
            metrics.setdefault("avg_current_ma", -1.0)
            metrics.setdefault("bus_voltage_v", -1.0)
            metrics.setdefault("latency_budget_ms", -1.0)
            metrics.setdefault("arena_bytes", -1)
            apply_cadenced_metric_defaults(metrics, metrics)
            directions = self._study_directions()
            if self._score_is_multiobjective():
                objective_names = [str(obj.metric) for obj in self.config.nas.score.params.objectives]
                objective_values = [
                    -1e12 if direction == "maximize" else 1e12
                    for direction in directions
                ]
                trial_outcome = TrialOutcome(
                    score=None,
                    objective_names=objective_names,
                    objective_values=objective_values,
                    objective_directions=directions,
                    task_metrics={},
                    hyperparams=dict(hyperparams),
                    artifact_summary=None,
                    quantization_mode=quantization_mode,
                )
            else:
                trial_outcome = TrialOutcome(
                    score=penalty_acc,
                    objective_names=["score"],
                    objective_values=[penalty_acc],
                    objective_directions=["maximize"],
                    task_metrics={},
                    hyperparams=dict(hyperparams),
                    artifact_summary=None,
                    quantization_mode=quantization_mode,
                )

            log_trial(
                trial_outcome=trial_outcome,
                metrics=metrics,
                trial=trial,
                log_file_name=str(log_path),
                study_name=self.study_name,
                pruned=True,
                prune_reason=prune_reason,
                prune_rule=prune_rule,
            )

            if self._score_is_multiobjective():
                return tuple(trial_outcome.objective_values)

            _report_if_supported(-float("inf"))
            raise optuna.TrialPruned(prune_reason)

        # Prune trials if they hit known fatal HIL error codes
        if error_code == HIL_MASTER_FLASH_OVERFLOW:
            # Flash overflow is terminal for this trial, so prune immediately.
            return _fail_with_penalty("Model exceeds board flash limit")

        if error_code in (HIL_MASTER_ARENA_EXHAUSTED, HIL_MASTER_FATAL, HIL_MASTER_RAM_OVERFLOW):
            # Convert the specific HIL failure into a descriptive pruning message.
            reason = {
                HIL_MASTER_ARENA_EXHAUSTED: "HIL arena exhausted",
                HIL_MASTER_FATAL: "HIL fatal error",
                HIL_MASTER_RAM_OVERFLOW: "HIL RAM overflow",
            }.get(error_code, "HIL error")
            return _fail_with_penalty(reason)

        if error_code != HIL_MASTER_SUCCESS:
            # Any other non-success code also prunes the trial with diagnostics.
            return _fail_with_penalty(f"HIL error code {error_code}")

        if collect_compile_metrics:
            flash_failure = metrics["flash_bytes"] == -1
            resources_ok = (
                np.isfinite(metrics["ram_bytes"])
                and metrics["ram_bytes"] < max_ram
                and metrics["flash_bytes"] < max_flash
            )
            # STM Phase 1 compile-only runs do not participate in tensor-arena
            # sizing, so `arena_bytes=-1` is an expected sentinel there. Keep the
            # historical arena validity gate for Arduino/TFLM backends.
            arena_ok = (
                metrics["arena_bytes"] != -1
                or (
                    runtime_device is not None
                    and not runtime_device.requires_arena_validation()
                )
            )

            # Distinguish between missing flash metrics, numeric resource overflow,
            # and backends that intentionally skip arena validation.
            if flash_failure or not resources_ok or not arena_ok:
                # Treat missing/invalid resource numbers as fatal so Optuna can move on.
                if not flash_failure and not resources_ok:
                    set_error_code(metrics, HIL_MASTER_FATAL)
                elif not flash_failure and not arena_ok:
                    set_error_code(metrics, HIL_MASTER_ARENA_EXHAUSTED)
                return _fail_with_penalty("Resource or arena check failed")

        prune_hit = evaluate_prune_rules(
            metrics=metrics,
            hyperparams=Dict(hyperparams),
            score_config=self.config.nas.score,
            prune_config=self.config.nas.prune,
            task_nonnegative_metric_names=task_nonnegative_metric_names,
        )
        if prune_hit is not None:
            prune_rule, prune_reason = prune_hit
            return _fail_with_penalty(prune_reason, prune_rule=prune_rule)

        try:
            if not self.config.training.train:
                # The no-training path still needs task-owned metric names in
                # the metrics dict so score evaluation can run uniformly.
                task_metrics = {
                    metric_name: -1.0
                    for metric_name in sorted(self.metric_contract.available_metric_names)
                }
                self._sync_task_metrics(metrics, task_metrics)
                self._apply_non_hil_success_sentinels(metrics)
                score_result = evaluate_score_config(
                    metrics=metrics,
                    hyperparams=hyperparams,
                    score_config=self.config.nas.score,
                    task_nonnegative_metric_names=task_nonnegative_metric_names,
                )
                trial_outcome = build_trial_outcome(
                    score_result=score_result,
                    task_metrics=task_metrics,
                    hyperparams=dict(hyperparams),
                    quantization_mode=quantization_mode,
                )
            else:
                fit_plan = self.task.build_fit_plan(
                    self.dataset_bundle,
                    self.task_config,
                    self.target_spec,
                    mode="search",
                    combine_train_val=False,
                )
                model.fit(
                    **fit_plan.fit_kwargs,
                    callbacks=fit_plan.callbacks,
                    epochs=self.config.training.nas_epochs,
                    batch_size=batch_size,
                )
                model = self.model_family.load_model(
                    self.config.outputs.checkpoint_path,
                    self.model_build_context,
                    self.model_config,
                )
                try:
                    evaluation_result = self._evaluate_model_with_backend(
                        model=model,
                        split=self.dataset_bundle.val,
                        split_name="validation",
                        quantization_mode=quantization_mode,
                        evaluation_backend="tflite",
                    )
                except TFLiteSubprocessError as exc:
                    prune_reason = "TFLite evaluation failed"
                    if not self._score_is_multiobjective():
                        prune_reason = f"{prune_reason}: {exc}"
                    return _fail_with_penalty(prune_reason)
                task_metrics = dict(evaluation_result.metrics)
                keras_evaluation_result = self._evaluate_model_with_backend(
                    model=model,
                    split=self.dataset_bundle.val,
                    split_name="validation_keras",
                    quantization_mode=quantization_mode,
                    evaluation_backend="keras",
                )
                task_metrics.update(
                    {
                        f"keras_{metric_name}": metric_value
                        for metric_name, metric_value in keras_evaluation_result.metrics.items()
                    }
                )
                self._sync_task_metrics(metrics, task_metrics)
                self._apply_non_hil_success_sentinels(metrics)
                score_result = evaluate_score_config(
                    metrics=metrics,
                    hyperparams=hyperparams,
                    score_config=self.config.nas.score,
                    task_nonnegative_metric_names=task_nonnegative_metric_names,
                )
                trial_outcome = build_trial_outcome(
                    score_result=score_result,
                    task_metrics=task_metrics,
                    hyperparams=dict(hyperparams),
                    artifact_summary=evaluation_result.artifacts,
                    quantization_mode=quantization_mode,
                )
        except ScoreConfigEvaluationError:
            return _fail_with_penalty(
                "Training failed to produce valid metrics",
            )

        if any(
            (not np.isfinite(value))
            or (
                direction == "minimize"
                and objective_name in effective_nonnegative_metric_names
                and value < 0.0
            )
            for objective_name, value, direction in zip(
                trial_outcome.objective_names,
                trial_outcome.objective_values,
                trial_outcome.objective_directions,
            )
        ):
            return _fail_with_penalty(
                "Training failed to produce valid metrics",
            )

        log_trial(
            trial_outcome=trial_outcome,
            metrics=metrics,
            trial=trial,
            log_file_name=str(log_path),
            study_name=self.study_name,
        )
        if self._score_is_multiobjective():
            return tuple(trial_outcome.objective_values)
        return float(trial_outcome.score)

    def smoke_test(
        self,
        train: bool=True,
        hil: bool | None = None,
        trials: int=5,
        epochs: int=5,
        study_name: str="smoke_test_study",
    ) -> None:
        """Run a quick Optuna smoke test with configurable training and HIL settings.

        Parameters
        ----------
        train : bool, optional
            Whether to enable model training during the test (default is True).
        hil : bool | None, optional
            Temporary HIL override for the smoke test. When ``None``, the
            loaded YAML setting is used as-is.
        trials : int, optional
            Number of trials to run in the smoke test (default is 5).
        epochs : int, optional 
            Number of epochs to train during the smoke test (default is 5).
        study_name : str, optional
            Name of the Optuna study (default is "smoke_test_study").

        Returns
        -------
        None
            Prints the best trial value, parameters, and runtime metrics.

        Notes
        -----
        Smoke tests reuse the same per-study SQLite storage and trial log when
        the caller passes the same ``study_name``. Re-running the smoke test
        therefore appends more trials instead of deleting prior results.
        """
        self.study_name = study_name
        artifacts_dir = self._artifacts_dir()
        self._copy_run_config(artifacts_dir)
        smoke_db_path = artifacts_dir / "optuna_smoke_test.db"
        storage_uri = f"sqlite:///{smoke_db_path}"
        _previous_hil = self.config.device.hil
        _previous_train = self.config.training.train
        _previous_epochs = self.config.training.nas_epochs
        try:
            if hil is not None:
                self.config.device.hil = hil
            self.config.training.train = train
            self.config.training.nas_epochs = epochs  # Speed up smoke test
            if (not train) and score_config_uses_training_metrics(
                self.config.nas.score,
                self.metric_contract.training_only_metric_names,
            ):
                raise ValueError(
                    "train=False is incompatible with score configs that require training-only metrics."
                )
            if self._score_is_multiobjective():
                sampler = optuna.samplers.NSGAIISampler(
                    population_size=self.config.training.nas_multiobjective_population_size,
                    seed=42,
                )
                single_trial_study = optuna.create_study(
                    directions=self._study_directions(),
                    storage=storage_uri,
                    study_name=study_name,
                    sampler=sampler,
                    load_if_exists=True,
                )
            else:
                sampler = optuna.samplers.TPESampler(
                    n_startup_trials=15,
                    multivariate=True,
                )
                single_trial_study = optuna.create_study(
                    direction="maximize",
                    storage=storage_uri,
                    study_name=study_name,
                    sampler=sampler,
                    load_if_exists=True,
                )
            single_trial_study.set_metric_names(self._study_metric_names())
            try:
                single_trial_study.optimize(self.objective, n_trials=trials)
            except Exception as exc:
                completed = sum(1 for t in single_trial_study.trials if t.state == TrialState.COMPLETE)
                pruned = sum(1 for t in single_trial_study.trials if t.state == TrialState.PRUNED)
                failed = sum(1 for t in single_trial_study.trials if t.state == TrialState.FAIL)
                total = len(single_trial_study.trials)
                print(
                    f"[SMOKE] Aborting after {total} trials: "
                    f"{completed} completed, {pruned} pruned, {failed} failed. "
                    f"Error: {exc}"
                )
                raise
        finally:
            self.config.device.hil = _previous_hil
            self.config.training.train = _previous_train
            self.config.training.nas_epochs = _previous_epochs

        complete_trials = [t for t in single_trial_study.trials if t.state == TrialState.COMPLETE]
        if not complete_trials:
            print("[SMOKE] No completed trials to report.")
            return

        if self._score_is_multiobjective():
            pareto = single_trial_study.best_trials
            print(f"Pareto front ({len(pareto)} trial(s)):")
            for trial in pareto:
                print(f"  Trial {trial.number} values: {trial.values}")
                print("  Params:")
                for name, value in trial.params.items():
                    print(f"    {name}: {value}")
                print("  Runtime metrics (user attrs):")
                for key in ("ram_bytes", "flash_bytes", "latency_ms", "hil_error_code", "arena_bytes", "task_metrics"):
                    print(f"    {key}: {trial.user_attrs.get(key)}")
        else:
            best_trial = single_trial_study.best_trial
            print(f"Single-trial value: {best_trial.value}")
            print("Best params:")
            for name, value in best_trial.params.items():
                print(f"  {name}: {value}")
            print("Runtime metrics (user attrs):")
            for key in ("ram_bytes", "flash_bytes", "latency_ms", "hil_error_code", "arena_bytes", "task_metrics"):
                print(f"  {key}: {best_trial.user_attrs.get(key)}")

    def run_nas(
        self,
        study_name: str,
        storage: str = "sqlite:///optuna.db",
    ) -> optuna.Study:
        """
        Run NAS with production settings, honoring configuration flags.

        Parameters
        ----------
        study_name : str
            Name to register the Optuna study under.
        storage : str, optional
            Optuna storage URI (defaults to a local SQLite DB).
        
        Notes
        -----
        The pipeline targets `config.training.nas_trials` completed trials and will
        retry pruned/failed attempts until that target is met or
        `config.training.max_total_trials` is reached.

        Failed and pruned trials still consume the total-attempt budget.

        Returns
        -------
        optuna.Study
            Completed study for downstream inspection/evaluation.
        """
        self.study_name = study_name
        target_completions = self.config.training.nas_trials
        max_total_trials = self.config.training.max_total_trials

        if self._score_is_multiobjective():
            sampler = optuna.samplers.NSGAIISampler(
                        population_size=self.config.training.nas_multiobjective_population_size,
                        seed=42)
            study = optuna.create_study(
                    directions=self._study_directions(),
                    storage=storage,
                    study_name=study_name,
                    sampler=sampler,
                    load_if_exists=True,
                )
        else:
            # Set up the Optuna study with TPE sampler and persistent storage.
            sampler = optuna.samplers.TPESampler(
                n_startup_trials=15,  # slightly more exploration than default before narrowing in
                multivariate=True,
            )
            study = optuna.create_study(
                direction="maximize",
                storage=storage,
                study_name=study_name,
                sampler=sampler,
                load_if_exists=True,  # resume if the study already exists
            )
        study.set_metric_names(self._study_metric_names())
        # Make sure we never shrink the total budget when resuming an existing study.
        max_total_trials = max(max_total_trials, len(study.trials))

        def _trial_counts():
            """Count completed, pruned, and failed Optuna trials.

            Returns
            -------
            tuple[int, int, int]
                Counts for complete, pruned, and failed trials in that order.
            """
            completed = sum(1 for t in study.trials if t.state == TrialState.COMPLETE)
            pruned = sum(1 for t in study.trials if t.state == TrialState.PRUNED)
            failed = sum(1 for t in study.trials if t.state == TrialState.FAIL)
            return completed, pruned, failed

        round_idx = 0
        try:
            while True:
                completed, pruned, failed = _trial_counts()
                total = len(study.trials)
                print(
                    f"[NAS] Progress: {completed} completed, {pruned} pruned, "
                    f"{failed} failed ({total} attempted)."
                )

                if completed >= target_completions:
                    print(f"[NAS] Reached target of {target_completions} completed trials.")
                    break

                remaining_needed = target_completions - completed
                remaining_budget = max_total_trials - total
                if remaining_budget <= 0:
                    print(
                        f"[NAS] Stopping with {completed}/{target_completions} completed trials "
                        f"after hitting max_total_trials={max_total_trials}."
                    )
                    break

                round_idx += 1
                next_batch = min(remaining_needed, remaining_budget)
                print(f"[NAS] Launching round {round_idx} for {next_batch} additional trial(s).")
                study.optimize(self.objective, n_trials=next_batch)
        except Exception as exc:
            completed, pruned, failed = _trial_counts()
            print(
                f"[NAS] Aborting after {len(study.trials)} trials "
                f"({completed} completed, {pruned} pruned, {failed} failed) because of an error: {exc}"
            )
            raise
        complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if not complete_trials:
            print("[NAS] No completed trials recorded; skipping best-trial reporting.")
            return study
        return study

    def run_scoring_nas(self, study_name: str, storage_uri: str = "sqlite:///optuna.db") -> None:
        """Run NAS, persist artifacts, and optionally finalize model.

        This orchestrates the scoring pipeline end-to-end:
        1) Executes Neural Architecture Search (NAS) with the current
           configuration via `run_nas`.
        2) Saves the full Optuna trials dataframe to CSV for analysis.
        3) If multi-objective NAS is enabled, extracts and saves the Pareto
           front and returns (no final training).
        4) If single-objective NAS, retrains the best architecture with a
           longer schedule and early stopping, plots losses, evaluates on the
           held-out test set (optionally exporting TFLite), computes simple
           trajectory metrics, and writes a summary bundle.

        Parameters
        ----------
        study_name : str
            Name of the Optuna study to create or resume. Used as a prefix for
            persisted artifacts (e.g., CSVs, plots, metrics).
        storage_uri : str, optional
            Optuna storage backend URI (e.g., ``sqlite:///optuna.db``). The
            study and trial history are persisted here. Default is a local
            SQLite file.

        Returns
        -------
        None
            Writes artifacts to disk. In multi-objective mode, also writes a
            Pareto-front CSV and returns without final model training.
        """
        print("[run_scoring_nas] Starting full NAS workflow")

        # Ensure all downstream paths use the caller-provided study_name
        self.study_name = study_name
        artifacts_dir = self._artifacts_dir()
        self._copy_run_config(artifacts_dir)
        storage_uri = f"sqlite:///{artifacts_dir / 'optuna.db'}"

        # 1) Run NAS with configured HIL/train settings.
        study = self.run_nas(study_name=study_name, storage=storage_uri)
        print(f"[run_scoring_nas] Completed NAS study with {len(study.trials)} total trials")
        
        # Persist raw Optuna trial history so convergence plots can be rebuilt later.
        trials_df = study.trials_dataframe()
        trials_csv = artifacts_dir / "trials.csv"
        trials_df.to_csv(trials_csv, index=False)
        print(f"[run_scoring_nas] Saved trials dataframe to {trials_csv}")

        if self._score_is_multiobjective():
            # Multi-objective: keep this as a “scoring + analysis” run.
            pareto_trials = study.best_trials
            pareto_ids = [t.number for t in pareto_trials]
            pareto_df = trials_df[trials_df["number"].isin(pareto_ids)]
            pareto_csv = Path(self.config.outputs.models_dir) / f"{study_name}_pareto.csv"
            pareto_df.to_csv(pareto_csv, index=False)
            print(f"[run_scoring_nas] Saved Pareto front to {pareto_csv}")
            print(f"[run_scoring_nas] Pareto front size: {len(pareto_trials)}")
            return
        
        print(f"[run_scoring_nas] Best value: {study.best_value}")
        # 2) Retrain the best architecture for the long schedule with early stopping.
        history_path = artifacts_dir / "train_history.json"
        history = self.train_best_trial(
            study_storage=storage_uri,
            study_name=study_name,
            patience=40,
            combine_train_val=False,
            history_path=history_path,
        )

        # 3) Plot training/validation losses for the write-up.
        loss_plots = self.plot_training_history(
            history=history,
            study_name=study_name,
            output_dir=artifacts_dir,
        )

        # 4) Evaluate on the held-out test split and optionally export TFLite.
        test_metrics = self.evaluate_checkpoint(
            study_storage=storage_uri,
            study_name=study_name,
            export_tflite=True,
            evaluation_backend="tflite",
        )

        # 5) Let the task decide whether it owns any extra closeout artifacts.
        closeout_model = self.model_family.load_model(
            Path(self.config.outputs.checkpoint_path),
            self.model_build_context,
            self.model_config,
        )
        closeout_artifacts = self.task.generate_closeout_artifacts(
            closeout_model,
            self.dataset_bundle,
            self.task_config,
            self.target_spec,
            output_dir=artifacts_dir / "task_closeout",
        )

        # 6) Optional Phase 9 reporting runs only after the fixed-split
        # deployable checkpoint/export path has completed.
        fold_rotation_artifacts = None
        if self._fold_rotation_enabled():
            fold_rotation_artifacts = self.run_fold_rotation_final_evaluation(
                study_storage=storage_uri,
                study_name=study_name,
                output_dir=artifacts_dir / "fold_rotation",
                patience=40,
            )

        # 7) Collect a summary bundle for reporting.
        self.write_summary_bundle(
            study_storage=storage_uri,
            study_name=study_name,
            history_path=history_path,
            loss_plots=loss_plots,
            test_metrics=test_metrics,
            closeout_artifacts=closeout_artifacts,
            fold_rotation_artifacts=fold_rotation_artifacts,
            summary_path=artifacts_dir / "summary.json",
        ) 

    def _artifacts_dir(self) -> Path:
        """Return the per-study artifacts directory under ``models_dir``.

        Returns
        -------
        Path
            Directory for artifacts associated with ``self.study_name``.

        Notes
        -----
        This helper creates the directory as a side effect and therefore
        depends on ``self.study_name`` already being finalized for the run.
        """
        d = Path(self.config.outputs.models_dir) / self.study_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _copy_run_config(self, artifacts_dir: Path | None = None) -> Path | None:
        """Copy the active NAS config into the study artifacts directory.

        Parameters
        ----------
        artifacts_dir : Path | None, optional
            Explicit destination directory. Defaults to
            :meth:`_artifacts_dir`.

        Returns
        -------
        Path | None
            Copied config path, the existing destination when source and
            destination already match, or ``None`` when the source config file
            is missing.
        """
        cfg_path = Path(self.config_path)
        if not cfg_path.exists():
            logger.warning("Skipping config copy because the config file is missing: %s", cfg_path)
            return None

        target_dir = artifacts_dir if artifacts_dir is not None else self._artifacts_dir()
        source_cfg = cfg_path.resolve()
        dest_cfg = (target_dir / cfg_path.name).resolve()
        if dest_cfg == source_cfg:
            print(f"[CONFIG] Run config already in artifacts dir: {dest_cfg}")
            return dest_cfg
        shutil.copy2(source_cfg, dest_cfg)
        print(f"[CONFIG] Copied run config to {dest_cfg}")
        return dest_cfg

    def _best_trial_params(self, study_storage: str, study_name: str) -> dict[str, Any]:
        """Load best-trial parameters without runtime-only hardware choices.

        Parameters
        ----------
        study_storage : str
            Optuna storage URI containing the study.
        study_name : str
            Optuna study name.

        Returns
        -------
        dict[str, Any]
            Best-trial parameters suitable for model-family decoding.
        """

        study = optuna.load_study(study_name=study_name, storage=study_storage)
        return _family_trial_params(study.best_trial.params)

    def _train_with_decoded_hparams(
        self,
        family_hparams: Any,
        *,
        bundle: DatasetBundle,
        target_spec: Any,
        model_build_context: ModelBuildContext,
        model_config: Any,
        task_config: Any,
        checkpoint_path: Path,
        history_path: Path,
        patience: int,
        combine_train_val: bool,
    ) -> dict:
        """Train one final model using explicit data and artifact paths.

        Parameters
        ----------
        family_hparams : Any
            Model-family hyperparameters already decoded from an Optuna trial.
        bundle : DatasetBundle
            Dataset bundle to train against.
        target_spec : Any
            Task target specification for ``bundle``.
        model_build_context : tinyodom.pipeline_types.ModelBuildContext
            Model construction context for ``bundle``.
        model_config : Any
            Model-family configuration subtree.
        task_config : Any
            Task configuration subtree.
        checkpoint_path : pathlib.Path
            Destination checkpoint path.
        history_path : pathlib.Path
            Destination history JSON path.
        patience : int
            Final-training early stopping patience.
        combine_train_val : bool
            Whether the task should combine train and validation splits.

        Returns
        -------
        dict
            JSON-friendly training history.
        """

        history_path = Path(history_path)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        ckpt_path = Path(checkpoint_path)
        batch_size = 256
        fit_task = self._instantiate_task(
            self.task_name,
            self.config,
            task_config,
            checkpoint_path=ckpt_path,
            early_stopping_patience=patience,
        )
        model = self.model_family.build_model(
            family_hparams,
            model_build_context,
            model_config,
        )
        fit_task.validate_model_outputs(model, target_spec)
        fit_task.compile_model(model, task_config, target_spec)

        fit_plan = fit_task.build_fit_plan(
            bundle,
            task_config,
            target_spec,
            mode="final",
            combine_train_val=combine_train_val,
        )
        fit_kwargs = dict(fit_plan.fit_kwargs)
        fit_kwargs["callbacks"] = list(fit_plan.callbacks)
        fit_kwargs["epochs"] = self.config.training.model_epochs
        fit_kwargs["batch_size"] = batch_size
        history = model.fit(**fit_kwargs)

        history_dict = {k: [float(v) for v in values] for k, values in history.history.items()}
        with open(history_path, "w") as f:
            json.dump(history_dict, f, indent=2)
        print(f"[FINAL TRAIN] Saved history to {history_path}")
        print(f"[FINAL TRAIN] Best checkpoint stored at {ckpt_path}")
        return history_dict

    def _evaluate_checkpoint_with_context(
        self,
        *,
        checkpoint_path: Path,
        metrics_path: Path,
        bundle: DatasetBundle,
        task: Any,
        target_spec: Any,
        model_build_context: ModelBuildContext,
        model_config: Any,
        task_config: Any,
        study_storage: str | None,
        study_name: str | None,
        export_tflite: bool,
        evaluation_backend: str = "keras",
        quantization_mode: str | None = None,
        tflite_path: Path | None = None,
    ) -> dict:
        """Evaluate one checkpoint using explicit data and build context.

        Parameters
        ----------
        checkpoint_path : pathlib.Path
            Checkpoint to evaluate.
        metrics_path : pathlib.Path
            Destination JSON/CSV metrics path.
        bundle : DatasetBundle
            Dataset bundle supplying test and calibration splits.
        task : Any
            Task component used to evaluate the model.
        target_spec : Any
            Task target specification for ``bundle``.
        model_build_context : tinyodom.pipeline_types.ModelBuildContext
            Model loading context for ``bundle``.
        model_config : Any
            Model-family configuration subtree.
        task_config : Any
            Task configuration subtree.
        study_storage : str | None
            Optuna storage URI for best-hyperparameter reporting.
        study_name : str | None
            Optuna study name for best-hyperparameter reporting.
        export_tflite : bool
            Whether to export a TFLite artifact.
        evaluation_backend : {"keras", "tflite"}, optional
            Host backend used for checkpoint scoring.
        quantization_mode : str | None, optional
            Deployment quantization mode used for TFLite export/evaluation.
        tflite_path : pathlib.Path | None, optional
            Destination TFLite path when exporting.

        Returns
        -------
        dict
            JSON-friendly metrics dictionary.
        """

        ckpt_path = Path(checkpoint_path)
        metrics_path = Path(metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        model = self.model_family.load_model(
            ckpt_path,
            model_build_context,
            model_config,
        )
        best_params = None
        best_quantization_mode = None
        if study_storage and study_name:
            study = optuna.load_study(study_name=study_name, storage=study_storage)
            raw_best_params = dict(study.best_trial.params)
            best_params = _family_trial_params(raw_best_params)
            best_quantization_mode = raw_best_params.get("quantization_mode")
        resolved_quantization_mode = (
            quantization_mode
            or best_quantization_mode
            or configured_quantization_mode(self.config)
        )
        if evaluation_backend == "keras":
            evaluation_result = task.evaluate(
                model,
                bundle.test,
                task_config,
                target_spec,
            )
            keras_evaluation_result = evaluation_result
        elif evaluation_backend == "tflite":
            representative_split = None
            if quantization_requires_calibration(resolved_quantization_mode):
                representative_split = bundle.calibration or bundle.train
            eval_tflite_path = metrics_path.with_suffix(".eval.tflite")
            convert_to_tflite_model(
                model=model,
                training_data=None if representative_split is None else representative_split.inputs,
                quantization_mode=resolved_quantization_mode,
                output_name=eval_tflite_path,
            )
            predictions = predict_tflite_model_subprocess(eval_tflite_path, bundle.test.inputs)
            evaluation_result = task.evaluate_predictions(
                predictions,
                bundle.test,
                task_config,
                target_spec,
            )
            keras_evaluation_result = task.evaluate(
                model,
                bundle.test,
                task_config,
                target_spec,
            )
        else:
            raise ValueError("evaluation_backend must be 'keras' or 'tflite'.")
        task_metrics = {}
        for metric_name, raw_value in evaluation_result.metrics.items():
            if isinstance(raw_value, np.generic):
                task_metrics[metric_name] = raw_value.item()
            else:
                task_metrics[metric_name] = raw_value
        if evaluation_backend == "tflite":
            for metric_name, raw_value in keras_evaluation_result.metrics.items():
                prefixed_name = f"keras_{metric_name}"
                if isinstance(raw_value, np.generic):
                    task_metrics[prefixed_name] = raw_value.item()
                else:
                    task_metrics[prefixed_name] = raw_value

        tflite_written = None
        if export_tflite:
            if not self.model_family.supports_tflite():
                raise ValueError(
                    f"Model family '{self.model_family_name}' does not support TFLite export."
            )
            tflite_path = Path(tflite_path) if tflite_path else Path(self.config.outputs.tflite_model_path)
            representative_split = None
            if quantization_requires_calibration(resolved_quantization_mode):
                representative_split = bundle.calibration or bundle.train
            convert_to_tflite_model(
                model=model,
                training_data=representative_split.inputs if representative_split is not None else None,
                quantization_mode=resolved_quantization_mode,
                output_name=tflite_path,
            )
            tflite_written = str(tflite_path)

        metrics = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "checkpoint_path": str(ckpt_path),
            "study_name": study_name,
            "study_storage": study_storage,
            "hyperparameters": best_params,
            "quantization_mode": resolved_quantization_mode,
            "evaluation_backend": evaluation_backend,
            "tflite_path": tflite_written,
            **task_metrics,
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        csv_path = metrics_path.with_suffix(".csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(list(metrics.keys()))
            writer.writerow(list(metrics.values()))
        print(f"[EVAL] Saved test metrics to {metrics_path} and {csv_path}")
        return metrics

    def train_best_trial(
        self,
        study_storage: str,
        study_name: str,
        patience: int = 40,
        combine_train_val: bool = False,
        checkpoint_path: Path | None = None,
        history_path: Path | None = None,
    ) -> dict:
        """
        Rebuild, retrain, and checkpoint the best Optuna trial with a longer schedule.

        Parameters
        ----------
        study_storage : str
            Optuna storage URI (e.g., ``sqlite:///optuna_smoke_test.db``) that contains the completed study.
        study_name : str
            Name of the Optuna study to load (matches what was passed to ``create_study``).
        patience : int, optional
            Early-stopping patience measured in epochs. Default is 40.
        combine_train_val : bool, optional
            If True, concatenate train and validation splits for maximum data;
            the monitor switches to training loss because no validation set remains.
            Default is False (keep validation for monitoring).
        checkpoint_path : Path | None, optional
            Override path for the `.keras` checkpoint. Defaults to ``config.outputs.checkpoint_path``.
        history_path : Path | None, optional
            Override path for writing the JSON training history. Defaults to a file under ``models_dir``.

        Returns
        -------
        dict
            History dictionary captured from Keras ``model.fit`` (converted to JSON-friendly lists).

        Notes
        -----
        - This method intentionally rebuilds the model from scratch using the best hyperparameters,
          then runs a long fit with early stopping and checkpoints the best weights.
        - History is persisted so loss plots can be regenerated later without rerunning training.
        """
        # Resolve output locations up front so they are obvious in logs.
        ckpt_path = Path(checkpoint_path) if checkpoint_path else Path(self.config.outputs.checkpoint_path)
        if history_path is None:
            history_path = Path(self.config.outputs.models_dir) / f"{study_name}_train_history.json"
        else:
            history_path = Path(history_path)
        history_path.parent.mkdir(parents=True, exist_ok=True)

        best_params = self._best_trial_params(study_storage, study_name)
        family_hparams = self.model_family.decode_trial_hparams(
            dict(best_params),
            self.model_build_context,
            self.model_config,
        )
        return self._train_with_decoded_hparams(
            family_hparams,
            bundle=self.dataset_bundle,
            target_spec=self.target_spec,
            model_build_context=self.model_build_context,
            model_config=self.model_config,
            task_config=self.task_config,
            checkpoint_path=ckpt_path,
            history_path=history_path,
            patience=patience,
            combine_train_val=combine_train_val,
        )

    def plot_training_history(
        self,
        history: dict | None = None,
        history_path: Path | None = None,
        output_dir: Path | None = None,
        study_name: str | None = None,
    ) -> dict:
        """
        Plot loss/validation loss curves (and per-output losses if present) to PNGs.

        Parameters
        ----------
        history : dict | None, optional
            Keras History.history-like mapping. If None, `history_path` must be provided.
        history_path : Path | None, optional
            JSON file containing the history dictionary. Used when `history` is None.
        output_dir : Path | None, optional
            Directory to store plot images. Defaults to `config.outputs.models_dir`.
        study_name : str | None, optional
            Name used to derive default filenames when not provided.

        Returns
        -------
        dict
            Mapping containing ``loss_plot`` and, when per-output histories are
            available, ``loss_components_plot``.
        """
        if history is None:
            if history_path is None:
                raise ValueError("Provide either `history` or `history_path` to plot training curves.")
            with open(history_path) as f:
                history = json.load(f)
        output_dir = Path(output_dir) if output_dir else Path(self.config.outputs.models_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Core loss curves (overall loss + val_loss if available).
        loss_png = output_dir / f"{study_name or 'history'}_loss.png"
        fig, ax = plt.subplots()
        if "loss" in history:
            ax.plot(history["loss"], label="loss")
        if "val_loss" in history:
            ax.plot(history["val_loss"], label="val_loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (MSE)")
        ax.set_title("Training/Validation Loss")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        fig.tight_layout()
        fig.savefig(loss_png, dpi=150)
        plt.close(fig)

        # Optional per-output losses if Keras exposed them.
        extra_png = None
        component_keys = self.task.history_component_keys(self.target_spec)
        if any(k in history for pair in component_keys for k in pair):
            extra_png = output_dir / f"{study_name or 'history'}_loss_components.png"
            fig, ax = plt.subplots()
            for train_key, val_key in component_keys:
                if train_key in history:
                    ax.plot(history[train_key], label=train_key)
                if val_key in history:
                    ax.plot(history[val_key], label=val_key)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss (MSE)")
            ax.set_title("Per-output Losses")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend()
            fig.tight_layout()
            fig.savefig(extra_png, dpi=150)
            plt.close(fig)

        written = {"loss_plot": str(loss_png)}
        if extra_png:
            written["loss_components_plot"] = str(extra_png)
        print(f"[PLOTS] Saved loss plots: {written}")
        return written

    def evaluate_checkpoint(
        self,
        checkpoint_path: Path | None = None,
        metrics_path: Path | None = None,
        study_storage: str | None = None,
        study_name: str | None = None,
        export_tflite: bool = False,
        evaluation_backend: str = "keras",
        tflite_path: Path | None = None,
    ) -> dict:
        """
        Evaluate the saved checkpoint on the held-out test split and log metrics.

        Parameters
        ----------
        checkpoint_path : Path | None, optional
            Path to the `.keras` checkpoint. Defaults to `config.outputs.checkpoint_path`.
        metrics_path : Path | None, optional
            Destination for metrics JSON/CSV. Defaults to `models_dir/{study_name}_test_metrics.json`.
        study_storage : str | None, optional
            Optuna storage URI to recover best hyperparameters for logging (optional).
        study_name : str | None, optional
            Optuna study name for metadata/logging (optional).
        export_tflite : bool, optional
            Whether to export a TFLite flatbuffer from the loaded checkpoint.
        evaluation_backend : {"keras", "tflite"}, optional
            Host backend used for final scoring.
        tflite_path : Path | None, optional
            Destination for the TFLite file. Defaults to `config.outputs.tflite_model_path`.

        Returns
        -------
        dict
            Metrics dictionary containing RMSEs, paths, and metadata.
        """
        ckpt_path = Path(checkpoint_path) if checkpoint_path else Path(self.config.outputs.checkpoint_path)
        if metrics_path is None:
            stem = study_name or "study"
            metrics_path = Path(self.config.outputs.models_dir) / f"{stem}_test_metrics.json"
        else:
            metrics_path = Path(metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

        return self._evaluate_checkpoint_with_context(
            checkpoint_path=ckpt_path,
            metrics_path=metrics_path,
            bundle=self.dataset_bundle,
            task=self.task,
            target_spec=self.target_spec,
            model_build_context=self.model_build_context,
            model_config=self.model_config,
            task_config=self.task_config,
            study_storage=study_storage,
            study_name=study_name,
            export_tflite=export_tflite,
            evaluation_backend=evaluation_backend,
            tflite_path=tflite_path,
        )

    def _fold_rotation_enabled(self) -> bool:
        """Return whether task-owned final fold-rotation reporting is enabled.

        Returns
        -------
        bool
            True when ``task.params.evaluation.protocol`` is ``fold_rotation``.
        """

        evaluation = getattr(self.task_config, "evaluation", Dict())
        return str(getattr(evaluation, "protocol", "fixed_split")) == "fold_rotation"

    def _fold_rotation_test_folds(self) -> list[int]:
        """Return normalized fold-rotation test folds from config.

        Returns
        -------
        list[int]
            Requested UrbanSound8K test folds.
        """

        evaluation = getattr(self.task_config, "evaluation", Dict())
        fold_rotation = getattr(evaluation, "fold_rotation", Dict())
        return [int(fold) for fold in fold_rotation.get("test_folds", list(range(1, 11)))]

    def _fold_rotation_cache_dir(self) -> Path:
        """Return the configured root for per-fold audio caches.

        Returns
        -------
        pathlib.Path
            Directory containing ``fold_XX`` cache directories.
        """

        return Path(self.config.dataset.params.fold_rotation_cache_dir).expanduser().resolve()

    def _bootstrap_fold_pipeline(self, fold_cache_dir: Path) -> Any:
        """Bootstrap the active pipeline against one rotated cache directory.

        Parameters
        ----------
        fold_cache_dir : pathlib.Path
            Cache directory containing one fold's split files.

        Returns
        -------
        object
            Bootstrapped pipeline for the fold-specific dataset bundle.
        """

        fold_config = copy.deepcopy(self.config)
        fold_config.dataset.params.cache_dir = str(fold_cache_dir)
        return bootstrap_pipeline(fold_config)

    @staticmethod
    def _require_finite_fold_metrics(metrics: dict[str, Any], fold: int) -> dict[str, float]:
        """Extract required finite reporting metrics for one fold.

        Parameters
        ----------
        metrics : dict[str, Any]
            Evaluation metrics emitted for a fold.
        fold : int
            Fold number used for error messages.

        Returns
        -------
        dict[str, float]
            Required metric values as floats.
        """

        required = {}
        for metric_name in ("accuracy", "macro_f1", "loss"):
            if metric_name not in metrics:
                raise ValueError(f"Fold {fold} metrics missing required '{metric_name}'.")
            value = float(metrics[metric_name])
            if not np.isfinite(value):
                raise ValueError(f"Fold {fold} metric '{metric_name}' is not finite.")
            required[metric_name] = value
        return required

    @staticmethod
    def _aggregate_fold_metrics(per_fold: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
        """Aggregate per-fold metrics for Phase 9 reporting.

        Parameters
        ----------
        per_fold : list[dict[str, Any]]
            Per-fold summary rows containing finite metric values.

        Returns
        -------
        dict[str, dict[str, float | None]]
            Mean and sample standard deviation by metric.
        """

        aggregate: dict[str, dict[str, float | None]] = {}
        for metric_name in ("accuracy", "macro_f1", "loss"):
            values = np.asarray([float(row[metric_name]) for row in per_fold], dtype=np.float64)
            aggregate[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if values.size >= 2 else None,
            }
        return aggregate

    def _write_fold_rotation_summary(
        self,
        *,
        output_dir: Path,
        study_name: str,
        requested_folds: list[int],
        completed_folds: list[int],
        per_fold: list[dict[str, Any]],
        status: str,
        error: str | None = None,
    ) -> Path:
        """Write a Phase 9 fold-rotation summary or partial failure manifest.

        Parameters
        ----------
        output_dir : pathlib.Path
            Fold-rotation artifact directory.
        study_name : str
            Optuna study name.
        requested_folds : list[int]
            Fold list requested by config.
        completed_folds : list[int]
            Folds completed before success or failure.
        per_fold : list[dict[str, Any]]
            Per-fold metric rows.
        status : str
            ``success`` or ``failed``.
        error : str | None, optional
            Failure reason for partial manifests.

        Returns
        -------
        pathlib.Path
            Written summary path.
        """

        output_dir.mkdir(parents=True, exist_ok=True)
        full_run = requested_folds == list(range(1, 11)) and status == "success"
        summary = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "study_name": study_name,
            "status": status,
            "partial": not full_run,
            "requested_test_folds": requested_folds,
            "completed_test_folds": completed_folds,
            "folds": per_fold,
            "aggregates": self._aggregate_fold_metrics(per_fold) if status == "success" else None,
            "error": error,
        }
        summary_path = output_dir / (
            "fold_rotation_summary.json" if status == "success" else "fold_rotation_summary.partial.json"
        )
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        return summary_path

    def run_fold_rotation_final_evaluation(
        self,
        *,
        study_storage: str,
        study_name: str,
        output_dir: Path,
        patience: int = 40,
    ) -> dict[str, Any]:
        """Run Phase 9 fold-rotation final train/evaluate reporting.

        Parameters
        ----------
        study_storage : str
            Optuna storage URI containing the already-completed NAS study.
        study_name : str
            Optuna study name.
        output_dir : pathlib.Path
            Directory where fold-rotation artifacts are written.
        patience : int, optional
            Final-training early stopping patience.

        Returns
        -------
        dict[str, Any]
            Summary containing the fold report path and per-fold rows.
        """

        if not self._fold_rotation_enabled():
            return {}
        requested_folds = self._fold_rotation_test_folds()
        best_params = self._best_trial_params(study_storage, study_name)
        default_quantization_mode = configured_quantization_mode(self.config)
        output_dir = Path(output_dir)
        per_fold: list[dict[str, Any]] = []
        completed_folds: list[int] = []
        try:
            for fold in requested_folds:
                fold_dir = output_dir / f"fold_{fold:02d}"
                fold_dir.mkdir(parents=True, exist_ok=True)
                pipeline = self._bootstrap_fold_pipeline(
                    self._fold_rotation_cache_dir() / f"fold_{fold:02d}"
                )
                family_hparams = self.model_family.decode_trial_hparams(
                    dict(best_params),
                    pipeline.model_build_context,
                    pipeline.selection["model_config"],
                )
                checkpoint_path = fold_dir / "checkpoint.keras"
                history_path = fold_dir / "train_history.json"
                history = self._train_with_decoded_hparams(
                    family_hparams,
                    bundle=pipeline.bundle,
                    target_spec=pipeline.target_spec,
                    model_build_context=pipeline.model_build_context,
                    model_config=pipeline.selection["model_config"],
                    task_config=pipeline.selection["task_config"],
                    checkpoint_path=checkpoint_path,
                    history_path=history_path,
                    patience=patience,
                    combine_train_val=False,
                )
                metrics_path = fold_dir / "test_metrics.json"
                metrics = self._evaluate_checkpoint_with_context(
                    checkpoint_path=checkpoint_path,
                    metrics_path=metrics_path,
                    bundle=pipeline.bundle,
                    task=pipeline.task,
                    target_spec=pipeline.target_spec,
                    model_build_context=pipeline.model_build_context,
                    model_config=pipeline.selection["model_config"],
                    task_config=pipeline.selection["task_config"],
                    study_storage=study_storage,
                    study_name=study_name,
                    export_tflite=False,
                )
                closeout_model = self.model_family.load_model(
                    checkpoint_path,
                    pipeline.model_build_context,
                    pipeline.selection["model_config"],
                )
                closeout_artifacts = pipeline.task.generate_closeout_artifacts(
                    closeout_model,
                    pipeline.bundle,
                    pipeline.selection["task_config"],
                    pipeline.target_spec,
                    output_dir=fold_dir / "task_closeout",
                )
                required_metrics = self._require_finite_fold_metrics(metrics, fold)
                row = {
                    "fold": fold,
                    "quantization_mode": metrics.get("quantization_mode", default_quantization_mode),
                    **required_metrics,
                    "history_path": str(history_path),
                    "metrics_path": str(metrics_path),
                    "checkpoint_path": str(checkpoint_path),
                    "task_closeout_artifacts": closeout_artifacts,
                    "epochs": len(next(iter(history.values()), [])),
                }
                per_fold.append(row)
                completed_folds.append(fold)
        except Exception as exc:
            partial_path = self._write_fold_rotation_summary(
                output_dir=output_dir,
                study_name=study_name,
                requested_folds=requested_folds,
                completed_folds=completed_folds,
                per_fold=per_fold,
                status="failed",
                error=str(exc),
            )
            print(f"[FOLD ROTATION] Saved partial failure summary to {partial_path}")
            raise

        csv_path = output_dir / "fold_metrics.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "fold",
                    "quantization_mode",
                    "accuracy",
                    "macro_f1",
                    "loss",
                    "history_path",
                    "metrics_path",
                    "checkpoint_path",
                ],
            )
            writer.writeheader()
            for row in per_fold:
                writer.writerow({key: row.get(key) for key in writer.fieldnames})
        summary_path = self._write_fold_rotation_summary(
            output_dir=output_dir,
            study_name=study_name,
            requested_folds=requested_folds,
            completed_folds=completed_folds,
            per_fold=per_fold,
            status="success",
        )
        print(f"[FOLD ROTATION] Saved fold metrics to {csv_path}")
        print(f"[FOLD ROTATION] Saved summary to {summary_path}")
        return {
            "summary_path": str(summary_path),
            "fold_metrics_csv": str(csv_path),
            "requested_test_folds": requested_folds,
            "completed_test_folds": completed_folds,
        }

    def trajectory_metrics_and_plots(
        self,
        checkpoint_path: Path | None = None,
        plot_dir: Path | None = None,
        stride: int | None = None,
        window_size: int | None = None,
        study_name: str | None = None,
    ) -> dict:
        """
        Compute simple trajectory metrics (ATE/RTE-style) and save example plots.

        Parameters
        ----------
        checkpoint_path : Path | None, optional
            Path to the `.keras` checkpoint. Defaults to `config.outputs.checkpoint_path`.
        plot_dir : Path | None, optional
            Directory to save trajectory plots. Defaults to `models_dir/trajectories`.
        stride : int | None, optional
            Sliding window stride used during preprocessing. Defaults to the
            active dataset metadata/config when omitted.
        window_size : int | None, optional
            Sliding window length used during preprocessing. Defaults to the
            active dataset metadata/config when omitted.
        study_name : str | None, optional
            Study name to prefix plot filenames.

        Returns
        -------
        dict
            Metrics including per-trajectory ATE/RTE and plot file paths.

        Notes
        -----
        - ATE is computed as mean Euclidean error between integrated GT and predicted tracks.
        - RTE is approximated on 60-second segments (sliding) using the same integration step.
        - Integration follows the notebook heuristic: delta_pos = vel / samples_per_window.
        """
        ckpt_path = Path(checkpoint_path) if checkpoint_path else Path(self.config.outputs.checkpoint_path)
        plot_dir = Path(plot_dir) if plot_dir else Path(self.config.outputs.models_dir) / "trajectories"
        plot_dir.mkdir(parents=True, exist_ok=True)
        test_split = self.dataset_bundle.test
        trajectory_split = self._require_trajectory_split()
        stride = (
            float(stride)
            if stride is not None
            else self._resolve_dataset_numeric_setting("stride", split=test_split)
        )
        window_size = (
            float(window_size)
            if window_size is not None
            else self._resolve_dataset_numeric_setting("window_size", split=test_split)
        )
        sampling_rate_hz = self._resolve_dataset_numeric_setting(
            "sampling_rate_hz",
            split=test_split,
        )

        model = self.model_family.load_model(
            ckpt_path,
            self.model_build_context,
            self.model_config,
        )
        preds = model.predict(trajectory_split.inputs)

        # Use the notebook-style integration heuristic: one windowed velocity
        # prediction covers roughly `samples_per_window` raw samples, and the
        # same cadence estimate drives the approximate RTE segment sizing.
        samples_per_window = max((window_size - stride) / stride, 1)

        def integrate_track(vx, vy, start_x, start_y):
            """Integrate velocity samples into absolute XY positions.

            Parameters
            ----------
            vx : array-like of float
                X velocity samples for one trajectory.
            vy : array-like of float
                Y velocity samples for one trajectory.
            start_x : float
                Initial x position.
            start_y : float
                Initial y position.

            Returns
            -------
            tuple[np.ndarray, np.ndarray]
                Integrated x and y coordinates for the trajectory.
            """
            xs = []
            ys = []
            x = start_x
            y = start_y
            for dx, dy in zip(vx, vy):
                x += dx / samples_per_window
                y += dy / samples_per_window
                xs.append(x)
                ys.append(y)
            return np.array(xs), np.array(ys)

        ate_per_traj = []
        rte_per_traj = []
        plot_paths = []

        idx_start = 0
        size_of_each = trajectory_split.metadata["size_of_each"]
        start_x_values = trajectory_split.metadata["x0"]
        start_y_values = trajectory_split.metadata["y0"]
        vel_x_targets = trajectory_split.targets["velx"]
        vel_y_targets = trajectory_split.targets["vely"]

        for i, length in enumerate(size_of_each):
            idx_end = idx_start + length
            # Flatten in case datasets store (n, 1) vectors.
            gt_vx = vel_x_targets[idx_start:idx_end].ravel()
            gt_vy = vel_y_targets[idx_start:idx_end].ravel()
            pred_vx = preds[0][idx_start:idx_end].ravel()
            pred_vy = preds[1][idx_start:idx_end].ravel()
            start_x = start_x_values[i]
            start_y = start_y_values[i]

            gt_x, gt_y = integrate_track(gt_vx, gt_vy, start_x, start_y)
            pd_x, pd_y = integrate_track(pred_vx, pred_vy, start_x, start_y)

            # Absolute Trajectory Error (mean distance).
            errs = np.sqrt((gt_x - pd_x) ** 2 + (gt_y - pd_y) ** 2)
            ate = float(np.mean(errs))
            ate_per_traj.append(ate)

            # Relative Trajectory Error over ~60s segments (heuristic).
            window_seconds = 60
            # Windows per second: sampling_rate_hz / stride (stride is in samples).
            samples_per_sec = max(sampling_rate_hz / stride, 1)
            segment = max(int(window_seconds * samples_per_sec), 1)
            rte_segments = []
            for j in range(0, len(gt_x) - segment, segment):
                dx_gt = gt_x[j + segment - 1] - gt_x[j]
                dy_gt = gt_y[j + segment - 1] - gt_y[j]
                dx_pd = pd_x[j + segment - 1] - pd_x[j]
                dy_pd = pd_y[j + segment - 1] - pd_y[j]
                rte_segments.append(np.sqrt((dx_gt - dx_pd) ** 2 + (dy_gt - dy_pd) ** 2))
            rte = float(np.median(rte_segments)) if rte_segments else float("nan")
            rte_per_traj.append(rte)

            # Plot sample trajectory overlay.
            fig, ax = plt.subplots()
            ax.plot(gt_x, gt_y, label="ground truth", linewidth=2)
            ax.plot(pd_x, pd_y, label="predicted", linewidth=2, linestyle="--")
            ax.set_xlabel("X position")
            ax.set_ylabel("Y position")
            ax.set_title(f"Trajectory {i} (ATE={ate:.3f}, RTE={rte:.3f})")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.5)
            fig.tight_layout()
            plot_path = plot_dir / f"{study_name or 'study'}_traj_{i}.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            plot_paths.append(str(plot_path))

            idx_start = idx_end

        finite_rte_values = [float(value) for value in rte_per_traj if np.isfinite(value)]
        rte_median = float(np.median(finite_rte_values)) if finite_rte_values else float("nan")

        metrics = {
            "ate_mean": float(np.mean(ate_per_traj)),
            "ate_median": float(np.median(ate_per_traj)),
            "ate_per_traj": ate_per_traj,
            "rte_median": rte_median,
            "rte_per_traj": rte_per_traj,
            "plots": plot_paths,
            "checkpoint_path": str(ckpt_path),
        }
        metrics_path = plot_dir / f"{study_name or 'study'}_trajectory_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        # Preserve the NAS configuration alongside the trajectory artifacts for reproducibility.
        if self.config_path.exists():
            dest_cfg = plot_dir / self.config_path.name
            shutil.copy2(self.config_path, dest_cfg)
        print(f"[TRAJ] Saved trajectory metrics to {metrics_path}")
        return metrics

    def write_summary_bundle(
        self,
        study_storage: str,
        study_name: str,
        history_path: Path,
        loss_plots: dict,
        test_metrics: dict,
        closeout_artifacts: dict | None = None,
        fold_rotation_artifacts: dict | None = None,
        summary_path: Path | None = None,
    ) -> Path:
        """
        Collect training/eval artifacts into a single summary JSON for write-ups.

        Parameters
        ----------
        study_storage : str
            Optuna storage URI.
        study_name : str
            Optuna study name.
        history_path : Path
            Path to the saved training history JSON.
        loss_plots : dict
            Mapping from plot labels to saved PNG paths.
        test_metrics : dict
            Output from `evaluate_checkpoint`.
        closeout_artifacts : dict | None, optional
            Task-owned closeout artifact summary when the active task emits
            extra report outputs.
        fold_rotation_artifacts : dict | None, optional
            Phase 9 fold-rotation report artifact summary.
        summary_path : Path | None, optional
            Destination for the summary JSON. Defaults to `models_dir/{study_name}_summary.json`.

        Returns
        -------
        Path
            Path to the written summary JSON.
        """
        if summary_path is None:
            summary_path = Path(self.config.outputs.models_dir) / f"{study_name}_summary.json"
        else:
            summary_path = Path(summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        study = optuna.load_study(study_name=study_name, storage=study_storage)
        raw_best_params = dict(study.best_trial.params)
        best_params = _family_trial_params(raw_best_params)
        quantization_mode = raw_best_params.get(
            "quantization_mode",
            test_metrics.get("quantization_mode", configured_quantization_mode(self.config)),
        )
        summary = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "study_name": study_name,
            "study_storage": study_storage,
            "best_params": best_params,
            "quantization_mode": quantization_mode,
            "history_path": str(history_path),
            "loss_plots": loss_plots,
            "test_metrics": test_metrics,
            "task_closeout_artifacts": closeout_artifacts,
            "fold_rotation_artifacts": fold_rotation_artifacts,
            "checkpoint_path": test_metrics.get("checkpoint_path"),
            "tflite_path": test_metrics.get("tflite_path"),
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[SUMMARY] Saved summary bundle to {summary_path}")
        return summary_path

    def close(self):
        """Close the ZeroMQ socket and terminate its context.

        Returns
        -------
        None
        """
        if self.socket is not None:
            self.socket.close(linger=0)
            self.socket = None
        if self.context is not None:
            self.context.term()
            self.context = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TinyODOM NAS workflow runner.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the NAS/HIL YAML configuration file.",
    )
    parser.add_argument(
        "--smoke-test",
        type=int,
        nargs="?",
        const=3,
        default=0,
        help=(
            "Run a short smoke test with the given number of trials (e.g., 3). "
            "Use `--smoke-test` without a value to run 3 trials; pass 0 to disable (default)."
        ),
    )
    parser.add_argument(
        "--study-name",
        type=str,
        help="Name of the Optuna study to use for the NAS pipeline.",
        default="tinyodom_nas_study",
    )
    args = parser.parse_args()

    # End-to-end NAS + final training + evaluation pipeline.
    storage_uri = "sqlite:///optuna.db"
    client: NASModelClient | None = None
    try:
        client = NASModelClient(args.config)
        if args.smoke_test > 0:
            print(f"[MAIN] Starting smoke test with {args.smoke_test} trials...")
            study_name = f"{args.study_name}_{client.config.device.name}"
            client.smoke_test(
                trials=args.smoke_test,
                epochs=3,
                study_name=study_name,
            )
            print("[MAIN] Smoke test complete.")
        else:
            print(f"[MAIN] Starting full NAS workflow with study name '{args.study_name}'...")
            client.run_scoring_nas(study_name=args.study_name, storage_uri=storage_uri)
            print("[MAIN] Full NAS workflow complete.")
    finally:
        if client is not None:
            client.close()
