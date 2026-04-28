import argparse
from collections.abc import Mapping
import csv
import inspect
import json
import logging
import shutil
import socket
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import zmq
from addict import Dict

import absl.logging
import matplotlib.pyplot as plt
import numpy as np
import optuna
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from optuna.trial import TrialState
from tinyodom.hardware import (
    HIL_MASTER_ARENA_EXHAUSTED,
    HIL_MASTER_DEVICE_NOT_FOUND,
    HIL_MASTER_FATAL,
    HIL_MASTER_FLASH_OVERFLOW,
    HIL_MASTER_RAM_OVERFLOW,
    HIL_MASTER_SUCCESS,
    convert_to_tflite_model,
    return_hardware_specs,
)
from tinyodom.microcontrollers import (
    get_device as get_microcontroller_device,
    resolve_device_options,
)
from tinyodom.builtin_components import ensure_builtin_components_registered
from tinyodom.component_selection import cfg_get, resolve_component_selection
from tinyodom.model import (
    NONNEGATIVE_METRICS,
    ScoreConfigEvaluationError,
    build_trial_outcome,
    DILATION_CANDIDATES,
    apply_cadenced_metric_defaults,
    count_flops,
    evaluate_prune_rules,
    evaluate_score_config,
    get_score_config_directions,
    is_multiobjective_score_config,
    log_trial,
    load_config,
    require_logical_input_shape,
    TrialOutcome,
    DEFAULT_CONFIG_PATH,
    score_config_uses_training_metrics,
    set_error_code,
)
from tinyodom.pipeline_types import DataSplit, DatasetBundle, ModelBuildContext
from tinyodom.registry import dataset_registry, model_family_registry, task_registry

tf.get_logger().setLevel(logging.ERROR)
absl.logging.set_verbosity(absl.logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

logger = logging.getLogger(__name__)


class _LegacySplitView(SimpleNamespace):
    """Legacy split view backed by the modular ``DataSplit`` contract."""


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
        ``src/nas_config.yaml`` via ``DEFAULT_CONFIG_PATH``. The configuration
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
    training_data, validation_data, test_data : object
        OXIOD dataset splits as loaded by ``import_oxiod_dataset``; each split
        exposes tensors like ``inputs``, ``x_vel``, ``y_vel`` and sequence
        bookkeeping metadata.
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
        """
        self.config_path = Path(config_path)
        ensure_builtin_components_registered()
        preliminary_config = load_config(self.config_path)
        preliminary_selection = self._resolve_component_selection(preliminary_config)
        preliminary_dataset, preliminary_bundle = self._coerce_loaded_dataset_bundle(
            preliminary_selection["dataset_name"],
            self._load_dataset_bundle(
                preliminary_selection["dataset_name"],
                preliminary_selection["dataset_config"],
            ),
        )
        preliminary_task = self._instantiate_task(
            preliminary_selection["task_name"],
            preliminary_config,
            preliminary_selection["task_config"],
        )
        preliminary_target_spec = preliminary_task.build_target_spec(
            preliminary_bundle,
            preliminary_selection["task_config"],
        )
        preliminary_metric_contract = preliminary_task.metric_contract(
            preliminary_target_spec,
            preliminary_selection["task_config"],
        )
        self.config = load_config(
            self.config_path,
            task_metric_names=preliminary_metric_contract.available_metric_names,
            training_only_task_metric_names=preliminary_metric_contract.training_only_metric_names,
        )
        final_selection = self._resolve_component_selection(self.config)
        dataset = preliminary_dataset
        bundle = preliminary_bundle
        if not self._same_component_selection(preliminary_selection, final_selection):
            dataset, bundle = self._coerce_loaded_dataset_bundle(
                final_selection["dataset_name"],
                self._load_dataset_bundle(
                    final_selection["dataset_name"],
                    final_selection["dataset_config"],
                ),
            )
        self._initialize_component_state(final_selection, dataset, bundle)

        if self.config.device.hil is False:
            logger.warning("HIL is disabled in the configuration.")
        if is_multiobjective_score_config(self.config.nas.score):
            logger.info("Using multi-objective NAS.")
        else:
            logger.info("Using single-objective NAS.")

        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.RCVTIMEO = self.config.network.recv_timeout_sec * 1000
        self.socket.SNDTIMEO = self.config.network.send_timeout_sec * 1000  # Avoid hanging forever during tunnel hiccups

        endpoint = f"tcp://{self.config.network.host}:{self.config.network.port}"
        self.socket.connect(endpoint)
        print(f"[REQ] Connected to HIL server at {endpoint}")

        self.study_name = "default_study"

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
        """Instantiate one task component with compatibility constructor args.

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
        task_cls = task_registry.get(task_name)
        signature = inspect.signature(task_cls)
        kwargs: dict[str, Any] = {}
        if "checkpoint_path" in signature.parameters:
            kwargs["checkpoint_path"] = (
                Path(config.outputs.checkpoint_path)
                if checkpoint_path is None
                else Path(checkpoint_path)
            )
        if "early_stopping_patience" in signature.parameters:
            kwargs["early_stopping_patience"] = int(
                self._cfg_get(task_config, "early_stopping_patience", 40)
                if early_stopping_patience is None
                else early_stopping_patience
            )
        return task_cls(**kwargs)

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

    @staticmethod
    def _make_legacy_split_view(split: DataSplit | None) -> _LegacySplitView | None:
        """Convert a modular split into the legacy view expected downstream.

        Parameters
        ----------
        split : DataSplit | None
            Modular split to adapt.

        Returns
        -------
        _LegacySplitView | None
            Namespace exposing the legacy TinyODOM split fields.
        """
        if split is None:
            return None
        metadata = dict(split.metadata)
        targets = split.targets if isinstance(split.targets, Mapping) else {}
        return _LegacySplitView(
            inputs=split.inputs,
            x_vel=targets.get("velx"),
            y_vel=targets.get("vely"),
            disp=metadata.get("disp"),
            heading=metadata.get("heading"),
            position=metadata.get("position"),
            x0=metadata.get("x0"),
            y0=metadata.get("y0"),
            size_of_each=metadata.get("size_of_each"),
            head_s=metadata.get("head_s"),
            head_c=metadata.get("head_c"),
            inputs_orig=metadata.get("inputs_orig"),
        )

    def _refresh_legacy_split_aliases(self, bundle: DatasetBundle) -> None:
        """Refresh legacy split aliases from one modular dataset bundle.

        Parameters
        ----------
        bundle : DatasetBundle
            Modular dataset bundle backing this client.

        Returns
        -------
        None
        """
        self.training_data = self._make_legacy_split_view(bundle.train)
        self.validation_data = self._make_legacy_split_view(bundle.val)
        self.test_data = self._make_legacy_split_view(bundle.test)

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
        self._refresh_legacy_split_aliases(bundle)

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
        """Fail fast if the HIL REP socket is unreachable."""
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

    def _assemble_runtime_hyperparams(
        self,
        family_hparams: dict[str, Any],
        flops: int,
        batch_size: int,
    ) -> Dict:
        """Assemble the legacy hyperparameter payload shared across runtime uses.

        Parameters
        ----------
        family_hparams : dict[str, Any]
            Model-family sampled or reconstructed hyperparameters.
        flops : int
            FLOP count for the built model.
        batch_size : int
            Runner-owned batch size.

        Returns
        -------
        Dict
            Legacy hyperparameter payload used by HIL, pruning, scoring, and
            CSV logging.
        """
        timesteps, input_dim = require_logical_input_shape(
            None if self.model_build_context is None else self.model_build_context.input_shape
        )
        return Dict(
            {
                **family_hparams,
                "batch_size": int(batch_size),
                "timesteps": timesteps,
                "input_dim": input_dim,
                "flops": int(flops),
            }
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

    @staticmethod
    def _expand_best_params_to_family_hparams(best_params: dict[str, Any]) -> dict[str, Any]:
        """Convert Optuna best-trial params into model-family build hparams.

        Parameters
        ----------
        best_params : dict[str, Any]
            Raw Optuna trial parameters.

        Returns
        -------
        dict[str, Any]
            Model-family hyperparameters accepted by ``build_model(...)``.
        """
        family_hparams = dict(best_params)
        if "dilations_index" in family_hparams and "dilations" not in family_hparams:
            family_hparams["dilations"] = DILATION_CANDIDATES[int(family_hparams["dilations_index"])]
        family_hparams.pop("dilations_index", None)
        return family_hparams

    @staticmethod
    def _concatenate_split_payload(
        train_value: Any,
        val_value: Any,
        *,
        context_name: str,
    ) -> Any:
        """Concatenate one train/validation payload for combine-train-val mode.

        Parameters
        ----------
        train_value : Any
            Training split payload.
        val_value : Any
            Validation split payload.
        context_name : str
            Human-readable field name used in errors.

        Returns
        -------
        Any
            Concatenated payload.

        Raises
        ------
        ValueError
            If the payload shape is unsupported for concatenation.
        """

        if isinstance(train_value, np.ndarray) and isinstance(val_value, np.ndarray):
            return np.concatenate([train_value, val_value], axis=0)
        if isinstance(train_value, Mapping) and isinstance(val_value, Mapping):
            if set(train_value) != set(val_value):
                raise ValueError(
                    f"combine_train_val=True requires matching mapping keys for {context_name}."
                )
            return {
                key: NASModelClient._concatenate_split_payload(
                    train_value[key],
                    val_value[key],
                    context_name=f"{context_name}.{key}",
                )
                for key in train_value
            }
        if train_value is None and val_value is None:
            return None
        raise ValueError(
            "combine_train_val=True only supports NumPy arrays or dicts of NumPy arrays; "
            f"unsupported payload encountered for {context_name}."
        )

    def _merge_train_and_val_splits(self) -> DataSplit:
        """Merge train and validation splits for compatibility final training.

        Returns
        -------
        DataSplit
            Concatenated training split.

        Raises
        ------
        ValueError
            If no validation split is available or if the payloads are not
            concatenable under the compatibility rules.
        """

        train_split = self.dataset_bundle.train
        val_split = self.dataset_bundle.val
        if val_split is None:
            raise ValueError("combine_train_val=True requires a validation split.")
        return DataSplit(
            inputs=self._concatenate_split_payload(
                train_split.inputs,
                val_split.inputs,
                context_name="inputs",
            ),
            targets=self._concatenate_split_payload(
                train_split.targets,
                val_split.targets,
                context_name="targets",
            ),
            sample_weights=self._concatenate_split_payload(
                train_split.sample_weights,
                val_split.sample_weights,
                context_name="sample_weights",
            ),
            metadata=dict(train_split.metadata),
        )

    def _fit_targets_for_split(self, split: DataSplit) -> Any:
        """Return task targets in model-output order for manual fit paths.

        Parameters
        ----------
        split : DataSplit
            Dataset split to adapt.

        Returns
        -------
        Any
            Model.fit-compatible target payload.

        Raises
        ------
        ValueError
            If the split targets are missing declared output names.
        """

        if isinstance(split.targets, Mapping):
            missing = [
                output_name
                for output_name in self.target_spec.output_names
                if output_name not in split.targets
            ]
            if missing:
                raise ValueError(
                    "Split targets do not satisfy the active task output contract; "
                    f"missing targets: {', '.join(missing)}."
                )
            return [split.targets[output_name] for output_name in self.target_spec.output_names]
        return split.targets

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

    def _trajectory_split_view(self) -> _LegacySplitView:
        """Return an odometry-oriented view of the test split for reporting.

        Returns
        -------
        _LegacySplitView
            Legacy-compatible test-split view for trajectory analysis.

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
        legacy_view = self._make_legacy_split_view(split)
        if legacy_view is None:
            raise ValueError("Trajectory reporting requires a held-out test split.")
        required_fields = ("size_of_each", "x0", "y0")
        missing = [
            field_name
            for field_name in required_fields
            if getattr(legacy_view, field_name, None) in (None, "")
        ]
        if missing:
            raise ValueError(
                "Trajectory reporting remains odometry-specific and requires test-split metadata "
                f"for: {', '.join(missing)}."
            )
        return legacy_view

    def objective(self, trial: optuna.Trial) -> float | tuple:
        """Optimize TinyODOM architecture and training hyperparameters.

        This objective samples model hyperparameters (e.g., filters, kernel size,
        dilations) via Optuna, builds the corresponding TCN model to estimate
        FLOPs, queries a hardware-in-the-loop (HIL) server for resource/latency
        metrics, and—when the candidate passes resource checks—trains and scores
        the model on the OXIOD dataset. Trials are pruned on HIL errors or
        resource violations. The returned objective is either a single score or
        a multi-objective tuple depending on configuration.

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
        - FLOPs are estimated from the built Keras model to inform constraints.
        - HIL metrics include RAM, flash, arena usage, and latency; these are
            compared against device specs to gate training.
        - Training uses the imported OXIOD dataset splits (train/valid/test)
            and reports RMSE for velocity components.
        """
        artifacts_dir = self._artifacts_dir()
        log_path = artifacts_dir / self.config.outputs.log_file_name
        batch_size = 256
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
        logical_input_shape = require_logical_input_shape(
            None if self.model_build_context is None else self.model_build_context.input_shape
        )
        flops = count_flops(model, logical_input_shape)
        hyperparams = self._assemble_runtime_hyperparams(
            family_hparams,
            flops,
            batch_size,
        )

        # if the board supports runtime CPU clock selection, choose it here
        cpu_clock_mhz_options = self._cfg_get(self.config.device, "cpu_clock_mhz_options", None)
        device_options_overrides = None
        if cpu_clock_mhz_options is not None:
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

        # Ask the HIL server to evaluate the candidate for resource usage and latency.
        request_payload = {"hyperparams": hyperparams}
        if device_options_overrides is not None:
            request_payload["device_options_overrides"] = device_options_overrides
        metrics = self._hil_request(request_payload)
        metrics.setdefault("hil_enabled", bool(self.config.device.hil))
        metrics.setdefault("energy_aware", bool(self.config.training.energy_aware))

        # Gets the hardware *estimated* specifications for the target device
        device_options = self._hardware_limit_device_options()
        max_ram, max_flash = return_hardware_specs(
            self.config.device.name,
            device_options=device_options,
        )
        metrics["max_ram_bytes"] = float(max_ram)
        metrics["max_flash_bytes"] = float(max_flash)
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

        # Shouldn't get to here, still included for completeness
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
        )
        if prune_hit is not None:
            prune_rule, prune_reason = prune_hit
            return _fail_with_penalty(prune_reason, prune_rule=prune_rule)

        try:
            if not self.config.training.train:
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
                )
            else:
                fit_plan = self.task.make_fit_plan(
                    self.dataset_bundle,
                    self.task_config,
                    self.target_spec,
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
                evaluation_result = self.task.evaluate(
                    model,
                    self.dataset_bundle.val,
                    self.task_config,
                    self.target_spec,
                )
                task_metrics = dict(evaluation_result.metrics)
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
        # Make sure we never shrink the total budget when resuming an existing study.
        max_total_trials = max(max_total_trials, len(study.trials))

        # Enqueue the best-known config from the non-energy NAS run as a baseline trial.
        # Only enqueue if the study is new to avoid duplicates.
        if len(study.trials) == 0:
            study.enqueue_trial(
                {
                    "nb_filters": 10,
                    "kernel_size": 12,
                    "dropout_rate": 0.0,
                    "use_skip_connections": False,
                    "norm_flag": True,
                    "dilations_index": 107,
                }
            )

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
        )

        # 5) Optional: compute trajectory metrics and plots (ATE/RTE-style).
        traj_metrics = self.trajectory_metrics_and_plots(
            study_name=study_name,
            plot_dir=artifacts_dir / "trajectories",
        )

        # 6) Collect a summary bundle for reporting.
        self.write_summary_bundle(
            study_storage=storage_uri,
            study_name=study_name,
            history_path=history_path,
            loss_plots=loss_plots,
            test_metrics=test_metrics,
            traj_metrics=traj_metrics,
            summary_path=artifacts_dir / "summary.json",
        ) 

    def _artifacts_dir(self) -> Path:
        """Per-study artifacts directory under models/."""
        d = Path(self.config.outputs.models_dir) / self.study_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _copy_run_config(self, artifacts_dir: Path | None = None) -> Path | None:
        """Copy the active NAS config into the study artifacts directory."""
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

        # Load the completed Optuna study to retrieve the top-scoring trial.
        study = optuna.load_study(study_name=study_name, storage=study_storage)
        best_trial = study.best_trial
        best_params = best_trial.params

        batch_size = 256  # Use the same fixed batch size as in NAS search.
        family_hparams = self._expand_best_params_to_family_hparams(dict(best_params))
        fit_task = self._instantiate_task(
            self.task_name,
            self.config,
            self.task_config,
            checkpoint_path=ckpt_path,
            early_stopping_patience=patience,
        )
        model = self.model_family.build_model(
            family_hparams,
            self.model_build_context,
            self.model_config,
        )
        fit_task.validate_model_outputs(model, self.target_spec)
        fit_task.compile_model(model, self.task_config, self.target_spec)

        if combine_train_val:
            merged_train_split = self._merge_train_and_val_splits()
            checkpoint_cb = ModelCheckpoint(
                filepath=str(ckpt_path),
                monitor="loss",
                mode="min",
                verbose=1,
                save_best_only=True,
            )
            early_stop_cb = EarlyStopping(
                monitor="loss",
                patience=patience,
                mode="min",
                verbose=1,
                restore_best_weights=True,
            )
            fit_kwargs = {
                "x": merged_train_split.inputs,
                "y": self._fit_targets_for_split(merged_train_split),
                "shuffle": True,
                "callbacks": [checkpoint_cb, early_stop_cb],
            }
        else:
            fit_plan = fit_task.make_fit_plan(
                self.dataset_bundle,
                self.task_config,
                self.target_spec,
            )
            fit_kwargs = dict(fit_plan.fit_kwargs)
            fit_kwargs["callbacks"] = list(fit_plan.callbacks)
        fit_kwargs["epochs"] = self.config.training.model_epochs
        fit_kwargs["batch_size"] = batch_size
        history = model.fit(**fit_kwargs)

        # Persist the training history so future plotting/reporting does not require rerunning training.
        history_dict = {k: [float(v) for v in values] for k, values in history.history.items()}
        with open(history_path, "w") as f:
            json.dump(history_dict, f, indent=2)
        print(f"[FINAL TRAIN] Saved history to {history_path}")
        print(f"[FINAL TRAIN] Best checkpoint stored at {ckpt_path}")

        return history_dict

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
            Mapping of plot labels to file paths that were written.
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
        component_keys = [
            ("velx_loss", "val_velx_loss"),
            ("vely_loss", "val_vely_loss"),
        ]
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

        model = self.model_family.load_model(
            ckpt_path,
            self.model_build_context,
            self.model_config,
        )
        evaluation_result = self.task.evaluate(
            model,
            self.dataset_bundle.test,
            self.task_config,
            self.target_spec,
        )
        task_metrics = {}
        for metric_name, raw_value in evaluation_result.metrics.items():
            if isinstance(raw_value, np.generic):
                task_metrics[metric_name] = raw_value.item()
            else:
                task_metrics[metric_name] = raw_value

        # Gather hyperparameters for record-keeping if the study is available.
        best_params = None
        if study_storage and study_name:
            study = optuna.load_study(study_name=study_name, storage=study_storage)
            best_params = study.best_trial.params

        # Optionally emit a TFLite artifact for downstream deployment.
        tflite_written = None
        if export_tflite:
            tflite_path = Path(tflite_path) if tflite_path else Path(self.config.outputs.tflite_model_path)
            representative_split = self.dataset_bundle.calibration or self.dataset_bundle.train
            convert_to_tflite_model(
                model=model,
                training_data=representative_split.inputs,
                quantization=self.config.training.quantization,
                output_name=tflite_path,
            )
            tflite_written = str(tflite_path)

        metrics = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "checkpoint_path": str(ckpt_path),
            "study_name": study_name,
            "study_storage": study_storage,
            "hyperparameters": best_params,
            "tflite_path": tflite_written,
            **task_metrics,
        }

        # Persist JSON for rich write-ups and a simple CSV for quick scanning.
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        csv_path = metrics_path.with_suffix(".csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(list(metrics.keys()))
            writer.writerow(list(metrics.values()))
        print(f"[EVAL] Saved test metrics to {metrics_path} and {csv_path}")

        return metrics

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
        trajectory_split = self._trajectory_split_view()
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

        # Helper to integrate velocities into XY tracks.
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
        for i, length in enumerate(trajectory_split.size_of_each):
            idx_end = idx_start + length
            # Flatten in case datasets store (n, 1) vectors.
            gt_vx = trajectory_split.x_vel[idx_start:idx_end].ravel()
            gt_vy = trajectory_split.y_vel[idx_start:idx_end].ravel()
            pred_vx = preds[0][idx_start:idx_end].ravel()
            pred_vy = preds[1][idx_start:idx_end].ravel()
            start_x = trajectory_split.x0[i]
            start_y = trajectory_split.y0[i]

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
        traj_metrics: dict | None = None,
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
        traj_metrics : dict | None, optional
            Output from `trajectory_metrics_and_plots` if computed.
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
        best_params = study.best_trial.params
        summary = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "study_name": study_name,
            "study_storage": study_storage,
            "best_params": best_params,
            "history_path": str(history_path),
            "loss_plots": loss_plots,
            "test_metrics": test_metrics,
            "trajectory_metrics": traj_metrics,
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
        self.socket.close(linger=0)
        self.context.term()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TinyODOM NAS workflow runner.")
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
        client = NASModelClient()
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
