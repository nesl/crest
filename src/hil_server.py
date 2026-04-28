import argparse
from collections.abc import Mapping
import inspect
import logging
import math
import shutil
from pathlib import Path
from typing import Any

import absl.logging
# import optuna
import serial
import tensorflow as tf
# import tensorflow_model_optimization as tfmot
import zmq
from addict import Dict

from tinyodom.builtin_components import ensure_builtin_components_registered
from tinyodom.component_selection import cfg_get, resolve_component_selection
from tinyodom.devices import CandidatePrepareRequest, _sync_arduino_sketch_variant_for_config
from tinyodom.hardware import (
    HIL_MASTER_DEVICE_NOT_FOUND,
    describe_error_code,
)
from tinyodom.errors import HIL_ERROR_OK, HIL_MASTER_FATAL
from tinyodom.microcontrollers import (
    get_device as get_microcontroller_device,
    resolve_device_options,
)
from tinyodom.microcontrollers.stm32_nucleo_n657x0 import (
    BOARD_NAME as STM32_BOARD_NAME,
    classify_stm32_backend_error,
)
from tinyodom.microcontrollers import stm32_cube_clt, stm32_runtime
from tinyodom.model import (
    DEFAULT_CONFIG_PATH,
    build_collect_metrics_request,
    collect_metrics,
    load_config,
    require_logical_input_shape,
    set_error_code,
    validate_model_input_shape,
)
from tinyodom.pipeline_types import DataSplit, DatasetBundle, ModelBuildContext
from tinyodom.registry import dataset_registry, model_family_registry, task_registry

tf.get_logger().setLevel(logging.ERROR)
absl.logging.set_verbosity(absl.logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

logger = logging.getLogger(__name__)
APPROX_TRAINED_VARIANT_NAME = "approx_trained"
REPRESENTATIVE_VARIANT_LEGACY_NAME = "representative"
PERTURBED_VARIANT_LEGACY_NAME = "bn_full_plus_non_bn_bias_perturbed"
# Backward-compatible alias used by existing scripts/imports.
PERTURBED_VARIANT_NAME = APPROX_TRAINED_VARIANT_NAME


def _configure_logging(level_name: str) -> None:
    """
    Configure root logging for the HIL server process.

    Parameters
    ----------
    level_name : str
        Logging level name (e.g., INFO, DEBUG).

    Returns
    -------
    None
    """
    level_value = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level_value,
        format="%(levelname)s:%(name)s:%(message)s",
    )


def _build_backend_failure_metrics(
    *,
    error_kind: str,
    error_detail: str,
    latency_budget_ms: float,
    hil_enabled: bool,
    energy_aware: bool,
) -> dict:
    """Return a structured metrics dict for backend/request boundary failures.

    Parameters
    ----------
    error_kind : str
        Stable backend error kind.
    error_detail : str
        Human-readable backend failure detail.
    latency_budget_ms : float
        Current trial latency budget in milliseconds.
    hil_enabled : bool
        Whether the request intended to run HIL.
    energy_aware : bool
        Whether the request intended energy-aware execution.

    Returns
    -------
    dict
        Metrics-shaped failure payload compatible with the HIL server contract.
    """
    metrics = {
        "ram_bytes": -1,
        "flash_bytes": -1,
        "external_flash_bytes": -1,
        "latency_ms": -1,
        "latency_budget_ms": latency_budget_ms if hil_enabled else -1,
        "arena_bytes": -1,
        "hil_enabled": hil_enabled,
        "energy_aware": energy_aware,
        "inference_seq": -1,
        "energy_mj_per_inference": -1.0,
        "avg_power_mw": -1.0,
        "avg_current_ma": -1.0,
        "bus_voltage_v": -1.0,
        "idle_power_mw": -1.0,
        "clock_hz": -1.0,
        "harness_latency_ms": -1.0,
        "weight_storage_mode": "embedded",
        "backend_error_kind": str(error_kind),
        "backend_error_detail": str(error_detail),
    }
    set_error_code(metrics, HIL_MASTER_FATAL)
    return metrics

class HILServer:
    """Serve HIL metric requests over a ZeroMQ REP socket.

    Attributes
    ----------
    config : Dict
        Loaded NAS and device configuration.
    repo_root : Path
        Repository root used to locate sketch variants and other artifacts.
    sketch_variants_dir : Path
        Directory containing Arduino sketch templates selected per request.
    active_sketch_path : Path | None
        Most recently staged sketch path, when one has been selected.
    calibration_split : DataSplit | None
        Lazily resolved calibration split for model export workflows.
    context : zmq.Context
        Shared ZeroMQ context backing the server socket.
    socket : zmq.Socket
        REP socket bound by :meth:`start`.
    """

    def __init__(
        self,
        config_path: Path = DEFAULT_CONFIG_PATH,
        config: Dict | None = None,
    ) -> None:
        """
        Initialize the HIL server state and modular pipeline components.

        Parameters
        ----------
        config_path : Path, optional
            Path to the NAS/HIL YAML configuration file. Used when ``config`` is
            not provided.
        config : Dict | None, optional
            Pre-loaded configuration dictionary. When provided, this takes
            precedence over ``config_path``.

        Returns
        -------
        None
        """
        ensure_builtin_components_registered()
        self.config = config if config is not None else load_config(config_path)
        selection = self._resolve_component_selection(self.config)
        self.dataset_name = selection["dataset_name"]
        self.task_name = selection["task_name"]
        self.model_family_name = selection["model_family_name"]
        self.dataset_config = selection["dataset_config"]
        self.task_config = selection["task_config"]
        self.model_config = selection["model_config"]
        self.dataset = None
        self.task = None
        self.model_family = None
        self.dataset_bundle = None
        self.target_spec = None
        self.model_build_context = None
        self._pipeline_bootstrapped = False

        # Resolve repository root once so sketch variants can be copied before each compile.
        self.repo_root = Path(__file__).resolve().parent.parent
        self.sketch_variants_dir = self.repo_root / "sketches"
        self.active_sketch_path: Path | None = None
        self.calibration_split: DataSplit | None = None
        self._calibration_split_resolved = False

        if self.config.device.hil is False:
            logger.warning("HIL is disabled in the configuration.")

        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.REP)

    @staticmethod
    def _cfg_get(container: Any, key: str, default: Any = None) -> Any:
        """Read one field from a mapping-like or namespace-like container.

        Parameters
        ----------
        container : Any
            Object exposing either ``get`` or attribute access.
        key : str
            Field name to resolve.
        default : Any, optional
            Fallback value returned when ``key`` is unavailable.

        Returns
        -------
        Any
            Resolved configuration value or ``default``.
        """

        return cfg_get(container, key, default)

    def _ensure_pipeline_bootstrapped(self) -> None:
        """Load and cache the active dataset/task/model pipeline on first use.

        Returns
        -------
        None
            Subsequent calls are no-ops after the first successful bootstrap.
        """

        if self._pipeline_bootstrapped:
            return
        if self.dataset_bundle is not None:
            dataset_cls = dataset_registry.get(self.dataset_name)
            dataset = self.dataset if self.dataset is not None else dataset_cls()
            dataset.validate_config(self.dataset_config)
            bundle = self.dataset_bundle
        else:
            dataset, bundle = self._load_dataset_bundle(
                self.dataset_name,
                self.dataset_config,
            )
        selection = {
            "dataset_name": self.dataset_name,
            "task_name": self.task_name,
            "model_family_name": self.model_family_name,
            "dataset_config": self.dataset_config,
            "task_config": self.task_config,
            "model_config": self.model_config,
        }
        self._initialize_component_state(selection, dataset, bundle)

    def _resolve_component_selection(self, config: Any) -> dict[str, Any]:
        """Resolve active dataset, task, and model-family selections.

        Parameters
        ----------
        config : Any
            Loaded HIL/NAS configuration object.

        Returns
        -------
        dict[str, Any]
            Resolved component names and local configuration subtrees.
        """

        return resolve_component_selection(config)

    def _instantiate_task(
        self,
        task_name: str,
        task_config: Any,
    ) -> Any:
        """Instantiate the configured task with compatibility constructor args.

        Parameters
        ----------
        task_name : str
            Registered task name.
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
            kwargs["checkpoint_path"] = Path(self.config.outputs.checkpoint_path)
        if "early_stopping_patience" in signature.parameters:
            kwargs["early_stopping_patience"] = int(
                self._cfg_get(task_config, "early_stopping_patience", 40)
            )
        return task_cls(**kwargs)

    def _load_dataset_bundle(
        self,
        dataset_name: str,
        dataset_config: Any,
    ) -> tuple[Any, DatasetBundle]:
        """Instantiate the configured dataset and load its normalized bundle.

        Parameters
        ----------
        dataset_name : str
            Registered dataset name.
        dataset_config : Any
            Dataset-local configuration subtree.

        Returns
        -------
        tuple[Any, DatasetBundle]
            Instantiated dataset and its loaded normalized bundle.
        """

        dataset_cls = dataset_registry.get(dataset_name)
        dataset = dataset_cls()
        dataset.validate_config(dataset_config)
        bundle = dataset.load(dataset_config)
        return dataset, bundle

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
            Normalized build context for model-family operations.
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
        """Instantiate and validate the active modular pipeline components.

        Parameters
        ----------
        selection : dict[str, Any]
            Resolved component names and local configuration subtrees.
        dataset : Any
            Instantiated dataset component that loaded ``bundle``.
        bundle : DatasetBundle
            Normalized dataset bundle backing this server.

        Returns
        -------
        None
        """

        model_family_cls = model_family_registry.get(selection["model_family_name"])
        task = self._instantiate_task(selection["task_name"], selection["task_config"])
        model_family = model_family_cls()

        task.validate_config(selection["task_config"])
        model_family.validate_config(selection["model_config"])

        target_spec = task.build_target_spec(bundle, selection["task_config"])
        model_build_context = self._build_model_context(bundle, target_spec)

        self.dataset_name = selection["dataset_name"]
        self.task_name = selection["task_name"]
        self.model_family_name = selection["model_family_name"]
        self.dataset = dataset
        self.task = task
        self.model_family = model_family
        self.dataset_bundle = bundle
        self.target_spec = target_spec
        self.model_build_context = model_build_context
        self.dataset_config = selection["dataset_config"]
        self.task_config = selection["task_config"]
        self.model_config = selection["model_config"]
        self._pipeline_bootstrapped = True

    def _split_request_hparams(self, hyperparams: Mapping[str, Any]) -> tuple[dict[str, Any], Dict]:
        """Split inbound request fields into family and runtime-owned values.

        Parameters
        ----------
        hyperparams : Mapping[str, Any]
            Inbound HIL request payload.

        Returns
        -------
        tuple[dict[str, Any], Dict]
            Model-family hyperparameters plus runner-owned runtime metadata.
        """

        runtime_keys = {"flops", "timesteps", "input_dim", "batch_size"}
        family_hparams = {key: value for key, value in hyperparams.items() if key not in runtime_keys}
        runtime_metadata = Dict(
            {
                key: value
                for key, value in hyperparams.items()
                if key in runtime_keys
            }
        )
        return family_hparams, runtime_metadata

    @staticmethod
    def _parse_required_runtime_int(
        runtime_metadata: Mapping[str, Any],
        field_name: str,
    ) -> int:
        """Parse one required integer field from runtime metadata.

        Parameters
        ----------
        runtime_metadata : Mapping[str, Any]
            Runtime-owned request metadata.
        field_name : str
            Required field name.

        Returns
        -------
        int
            Parsed integer value.

        Raises
        ------
        ValueError
            If the field is missing or not a valid integer.
        """

        if field_name not in runtime_metadata:
            raise ValueError(f"HIL request hyperparams must include '{field_name}'.")
        raw_value = runtime_metadata[field_name]
        if isinstance(raw_value, bool):
            raise ValueError(f"HIL request hyperparams field '{field_name}' must be an integer.")
        try:
            return int(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"HIL request hyperparams field '{field_name}' must be an integer."
            ) from exc

    @classmethod
    def _normalize_runtime_metadata(cls, runtime_metadata: Mapping[str, Any]) -> Dict:
        """Normalize required runtime metadata into typed integer values.

        Parameters
        ----------
        runtime_metadata : Mapping[str, Any]
            Raw runtime-owned request metadata.

        Returns
        -------
        Dict
            Normalized runtime metadata with validated integer values.
        """

        normalized = Dict(dict(runtime_metadata))
        normalized.flops = cls._parse_required_runtime_int(runtime_metadata, "flops")
        normalized.timesteps = cls._parse_required_runtime_int(runtime_metadata, "timesteps")
        normalized.input_dim = cls._parse_required_runtime_int(runtime_metadata, "input_dim")
        if "batch_size" in runtime_metadata:
            normalized.batch_size = cls._parse_required_runtime_int(runtime_metadata, "batch_size")
        return normalized

    def _validate_request_dimensions(
        self,
        *,
        requested_timesteps: int,
        requested_input_dim: int,
    ) -> None:
        """Reject HIL requests whose model dimensions diverge from the server.

        Parameters
        ----------
        requested_timesteps : int
            Request-owned logical timestep count.
        requested_input_dim : int
            Request-owned logical input dimension.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the bootstrapped model context is unavailable or mismatched.
        """

        self._ensure_pipeline_bootstrapped()
        input_shape = None if self.model_build_context is None else self.model_build_context.input_shape
        expected_timesteps, expected_input_dim = require_logical_input_shape(input_shape)
        if requested_timesteps != expected_timesteps or requested_input_dim != expected_input_dim:
            raise ValueError(
                "HIL request model dimensions do not match the active server configuration: "
                f"expected timesteps={expected_timesteps}, input_dim={expected_input_dim}; "
                f"got timesteps={requested_timesteps}, input_dim={requested_input_dim}."
            )

    def _resolve_dataset_numeric_setting(self, key: str) -> float:
        """Resolve one numeric dataset setting from metadata or config.

        Parameters
        ----------
        key : str
            Numeric dataset field name to resolve.

        Returns
        -------
        float
            Resolved numeric value.

        Raises
        ------
        ValueError
            If the field cannot be resolved to a finite positive number.
        """

        self._ensure_pipeline_bootstrapped()
        metadata = {} if self.dataset_bundle is None else dict(self.dataset_bundle.metadata)
        raw_value = metadata.get(key, self._cfg_get(self.dataset_config, key, None))
        if raw_value in (None, "") or isinstance(raw_value, bool):
            raise ValueError(f"Unable to resolve dataset setting '{key}' for the active configuration.")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Dataset setting '{key}' must be numeric for the active configuration."
            ) from exc
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"Dataset setting '{key}' must be a positive finite value for the active configuration."
            )
        return value

    def _ensure_calibration_split(self) -> DataSplit | None:
        """Resolve and cache calibration data only when a backend needs it.

        Returns
        -------
        DataSplit | None
            Cached calibration split, or ``None`` when the dataset does not
            expose calibration data.
        """

        self._ensure_pipeline_bootstrapped()
        if not self._calibration_split_resolved:
            self.calibration_split = self.dataset.make_calibration_data(
                self.dataset_bundle,
                self.dataset_config,
            )
            self._calibration_split_resolved = True
        return self.calibration_split

    def _require_calibration_split(self) -> DataSplit:
        """Return calibration data required by export-capable backends.

        Returns
        -------
        DataSplit
            Resolved calibration split.

        Raises
        ------
        ValueError
            If the active dataset does not provide calibration data.
        """

        calibration_split = self._ensure_calibration_split()
        if calibration_split is None:
            raise ValueError(
                "The active dataset does not provide calibration data required for export preparation."
            )
        return calibration_split

    def _resolve_runtime_dimensions(self) -> tuple[int, int]:
        """Resolve runtime window size and input dimension from standardized context.

        Returns
        -------
        tuple[int, int]
            Resolved ``(window_size, input_dim)`` pair.

        Raises
        ------
        ValueError
            If the required dimensions cannot be derived from dataset metadata
            or model build context.
        """

        self._ensure_pipeline_bootstrapped()
        metadata = {} if self.dataset_bundle is None else dict(self.dataset_bundle.metadata)
        input_shape = None if self.model_build_context is None else self.model_build_context.input_shape
        resolved_input_shape = None
        raw_window_size = metadata.get("window_size")
        raw_input_dim = metadata.get("input_dim")
        if raw_window_size is None or raw_input_dim is None:
            resolved_input_shape = require_logical_input_shape(input_shape)
        if raw_window_size is None:
            raw_window_size = resolved_input_shape[0]
        if raw_input_dim is None:
            raw_input_dim = resolved_input_shape[1]
        if raw_window_size is None or raw_input_dim is None:
            raise ValueError(
                "Unable to resolve runtime window_size/input_dim from dataset metadata or model context."
            )
        return int(raw_window_size), int(raw_input_dim)

    def _materialize_candidate_model(
        self,
        family_hparams: dict[str, Any],
        *,
        model_variant: str,
        checkpoint_path: Path | str | None,
    ) -> Any:
        """Materialize and validate the model used for export preparation.

        Parameters
        ----------
        family_hparams : dict[str, Any]
            Model-family-owned hyperparameters.
        model_variant : str
            Requested export variant name.
        checkpoint_path : Path | str | None
            Optional checkpoint path for trained variants.

        Returns
        -------
        Any
            Validated model instance ready for export preparation.
        """

        self._ensure_pipeline_bootstrapped()
        model = self.model_family.materialize_export_model(
            family_hparams,
            self.model_build_context,
            self.model_config,
            model_variant=model_variant,
            checkpoint_path=checkpoint_path,
        )
        validate_model_input_shape(model, self.model_build_context.input_shape)
        self.task.validate_model_outputs(model, self.target_spec)
        return model

    def _build_candidate_prepare_request(
        self,
        *,
        model: Any,
        model_variant: str,
        checkpoint_path: Path | str | None,
        calibration_split: DataSplit | None,
    ) -> CandidatePrepareRequest:
        """Assemble the typed backend request for candidate preparation.

        Parameters
        ----------
        model : Any
            Model instance to export and stage.
        model_variant : str
            Human-readable export variant label.
        checkpoint_path : Path | str | None
            Optional checkpoint path associated with the candidate.
        calibration_split : DataSplit | None
            Calibration split supplied to backends that require representative
            export data.

        Returns
        -------
        CandidatePrepareRequest
            Typed backend preparation request.
        """

        return CandidatePrepareRequest(
            config=self.config,
            model=model,
            model_variant=model_variant,
            artifact_root=Path(self.config.outputs.tcn_dir),
            tflite_model_path=Path(self.config.outputs.tflite_model_path),
            calibration_split=calibration_split,
            input_shape=self.model_build_context.input_shape,
            checkpoint_path=checkpoint_path,
        )

    def _normalized_device_name(self) -> str:
        """Return the normalized configured device name.

        Returns
        -------
        str
            Upper-cased device name from the loaded configuration.
        """
        return str(getattr(self.config.device, "name", "")).strip().upper()

    def _effective_latency_budget_ms(self) -> float:
        """Return the resolved latency budget for the active configuration."""
        configured_budget = getattr(self.config.device, "latency_budget_ms", None)
        if configured_budget is not None:
            return float(configured_budget)
        stride = self._resolve_dataset_numeric_setting("stride")
        sampling_rate_hz = self._resolve_dataset_numeric_setting("sampling_rate_hz")
        return (stride / sampling_rate_hz) * 1000

    def _configured_runtime_mode(self) -> str:
        """Return the normalized configured runtime mode."""
        return str(getattr(self.config.device, "runtime_mode", "back_to_back")).strip().lower()

    def _run_arduino_cadenced_second_pass(
        self,
        *,
        runtime_device: object,
        prepared_dir: Path,
        base_metrics: dict,
        request_metrics_args,
        window_size: int,
        input_dim: int,
    ) -> dict:
        """Run and merge the Arduino-only cadenced second pass.

        Parameters
        ----------
        runtime_device : object
            Active runtime device.
        prepared_dir : Path
            Prepared candidate directory.
        base_metrics : dict
            Metrics emitted by the primary runtime pass.
        request_metrics_args : CollectMetricsRequest
            Shared primary metrics request payload.
        window_size : int
            Resolved runtime window size.
        input_dim : int
            Resolved runtime input dimension.

        Returns
        -------
        dict
            Updated metrics dictionary after the optional cadenced pass.
        """
        normalized_device_name = self._normalized_device_name()
        if self._configured_runtime_mode() != "cadenced":
            return base_metrics
        if normalized_device_name not in {"PORTENTA_H7", "ARDUINO_NANO_33_BLE_SENSE"}:
            return base_metrics
        if not request_metrics_args.hil_enabled:
            return base_metrics
        if int(base_metrics.get("error_code", -1)) != HIL_ERROR_OK:
            return base_metrics

        arena_bytes = int(base_metrics.get("arena_bytes", -1))
        if arena_bytes <= 0:
            return base_metrics

        runtime_device.set_input_mode(
            str(self.config.training.input_mode),
            outputs_dir=prepared_dir,
            config=self.config,
            sketches_dir=self.sketch_variants_dir,
            runtime_phase="cadenced",
        )
        sketch_candidate = prepared_dir / "tinyodom_tcn.ino"
        if sketch_candidate.is_file():
            self.active_sketch_path = sketch_candidate
            logger.info("Using sketch variant: %s", self.active_sketch_path)

        arena_kb = max(1, int(math.ceil(float(arena_bytes) / 1024.0)))
        harness = request_metrics_args.harness
        cadenced_result = runtime_device.evaluate(
            dirpath=prepared_dir,
            arena_kb=arena_kb,
            window_size=int(window_size),
            num_channels=int(input_dim),
            serial_port=request_metrics_args.serial_port,
            run_hil=True,
            serial_timeout_s=float(request_metrics_args.serial_timeout_s or 12.0),
            measured_inference_runs=int(request_metrics_args.measured_inference_runs),
            dut_ready_timeout_s=request_metrics_args.dut_ready_timeout_s,
            harness_serial_port=None if harness is None else harness.harness_serial_port,
            harness_fqbn=None if harness is None else harness.harness_fqbn,
            harness_auto_flash=None if harness is None else harness.harness_auto_flash,
            harness_arm_pin=None if harness is None else harness.harness_arm_pin,
            harness_trigger_pin=None if harness is None else harness.harness_trigger_pin,
            dut_arm_hold_ms=None if harness is None else harness.dut_arm_hold_ms,
            harness_stable_low_ms=None if harness is None else harness.harness_stable_low_ms,
            harness_ready_timeout_s=None if harness is None else harness.harness_ready_timeout_s,
            harness_arm_timeout_s=None if harness is None else harness.harness_arm_timeout_s,
            harness_active_timeout_s=None if harness is None else harness.harness_active_timeout_s,
            harness_done_timeout_s=None if harness is None else harness.harness_done_timeout_s,
        )

        base_metrics["runtime_mode"] = "cadenced"
        base_metrics["cadenced_error_code"] = int(cadenced_result.error_code)
        base_metrics["cadenced_error_label"] = describe_error_code(
            int(cadenced_result.error_code),
            prefer_master=False,
        )
        if int(cadenced_result.error_code) != HIL_ERROR_OK:
            return base_metrics

        raw_energy = -1.0
        if cadenced_result.power_metrics is not None:
            try:
                raw_energy = float(
                    cadenced_result.power_metrics.get("energy_mj_per_inference", -1.0)
                )
            except (AttributeError, TypeError, ValueError):
                raw_energy = -1.0
        energy_per_slot = raw_energy if math.isfinite(raw_energy) and raw_energy >= 0.0 else -1.0
        base_metrics["cadenced_energy_mj_per_window"] = energy_per_slot
        base_metrics["cadenced_energy_mj_per_trial"] = (
            energy_per_slot * float(request_metrics_args.measured_inference_runs)
            if energy_per_slot >= 0.0
            else -1.0
        )
        return base_metrics

    def start(self) -> None:
        """
        Start the ZeroMQ REP loop that evaluates incoming hyperparameters.

        The server blocks waiting for JSON messages, runs
        :meth:`determine_metrics`, and responds with a metrics dictionary for
        each request.

        Returns
        -------
        None
        """
        endpoint = f"tcp://{self.config.network.host}:{self.config.network.port}"
        self.socket.bind(endpoint)
        print(f"[HIL REP] Listening for hyperparameters on {endpoint}")

        try:
            while True:
                payload = self.socket.recv_json()
                print(f"[HIL REP] Received payload: {payload}")
                try:
                    hyperparams, device_options_overrides = self._normalize_request_payload(payload)
                    metrics = self.determine_metrics(
                        hyperparams,
                        device_options_overrides=device_options_overrides,
                    )
                except ValueError as exc:
                    logger.error("Invalid HIL request payload: %s", exc)
                    metrics = self._request_boundary_failure_metrics(
                        error_kind="request",
                        error_detail=str(exc),
                    )

                print(f"[HIL REP] Sending metrics: {metrics}")
                self.socket.send_json(metrics)
                if metrics.get("error_code") == HIL_MASTER_DEVICE_NOT_FOUND:
                    logger.error(
                        "Upload failed (device not found); stopping HIL server so the NAS run can be restarted."
                    )
                    break
        except KeyboardInterrupt:
            print("\n[HIL REP] Shutting down HIL REP server.")
        finally:
            self.socket.close(linger=0)
            self.context.term()

    @staticmethod
    def _normalize_request_payload(payload: object) -> tuple[Dict, dict | None]:
        """Normalize legacy and structured REQ payloads into one internal shape.

        Parameters
        ----------
        payload : object
            Inbound JSON-decoded request payload.

        Returns
        -------
        tuple[Dict, dict | None]
            Model hyperparameters plus optional device option overrides.
        """
        if not isinstance(payload, Mapping):
            raise ValueError("HIL request payload must be a JSON object.")
        if "hyperparams" in payload:
            raw_hyperparams = payload.get("hyperparams", {})
            if not isinstance(raw_hyperparams, Mapping):
                raise ValueError("HIL request field `hyperparams` must be a JSON object.")
            raw_overrides = payload.get("device_options_overrides", None)
            if raw_overrides is not None and not isinstance(raw_overrides, Mapping):
                raise ValueError(
                    "HIL request field `device_options_overrides` must be a JSON object when provided."
                )
            overrides = dict(raw_overrides) if raw_overrides is not None else None
            return Dict(dict(raw_hyperparams)), overrides
        return Dict(dict(payload)), None

    def _request_boundary_failure_metrics(
        self,
        *,
        error_kind: str,
        error_detail: str,
    ) -> dict:
        """Build failure metrics for malformed or invalid client requests."""
        try:
            latency_budget_ms = self._effective_latency_budget_ms()
        except ValueError:
            latency_budget_ms = -1.0
        effective_hil_enabled = bool(self.config.device.hil)
        effective_energy_aware = bool(self.config.training.energy_aware and effective_hil_enabled)
        return _build_backend_failure_metrics(
            error_kind=error_kind,
            error_detail=error_detail,
            latency_budget_ms=latency_budget_ms,
            hil_enabled=effective_hil_enabled,
            energy_aware=effective_energy_aware,
        )

    def determine_metrics(
        self,
        hyperparams: Dict,
        device_options_overrides: dict | None = None,
        checkpoint_path: Path | str | None = None,
        model_variant: str = APPROX_TRAINED_VARIANT_NAME,
    ) -> dict:
        """
        Build/select a model variant, compile/export it, and collect HIL metrics.

        Parameters
        ----------
        hyperparams : Dict
            Inbound request payload containing both model-family hyperparameters
            and runner-owned runtime metadata such as ``flops``.
        checkpoint_path : Path | str | None, optional
            Checkpoint path used when ``model_variant`` starts with ``"trained"``.
        model_variant : str, optional
            Model source/variant selector, by default ``"approx_trained"``. Other
             values are for analysis and debug only. Supported values are
            ``"approx_trained"``, ``"untrained"``, or any value that starts with
            ``"trained"``. ``"approx_trained"`` applies deterministic full-BN +
            non-BN-bias perturbation to approximate trained-model exported op
            structure without running training. ``"untrained"`` uses raw
            initializer defaults and typically exports fewer operations than
            ``"trained"``/``"approx_trained"``.

        Returns
        -------
        dict
            Metrics dictionary returned by ``collect_metrics``.

        Raises
        ------
        ValueError
            If ``model_variant`` is unsupported or a trained variant is selected
            without a checkpoint path.
        FileNotFoundError
            If a requested trained checkpoint does not exist.
        """
        family_hparams, runtime_metadata = self._split_request_hparams(hyperparams)
        try:
            runtime_metadata = self._normalize_runtime_metadata(runtime_metadata)
        except ValueError as exc:
            return self._request_boundary_failure_metrics(
                error_kind="request",
                error_detail=str(exc),
            )
        try:
            self._ensure_pipeline_bootstrapped()
            latency_budget_ms = self._effective_latency_budget_ms()
        except ValueError as exc:
            return self._request_boundary_failure_metrics(
                error_kind="config",
                error_detail=str(exc),
            )
        try:
            self._validate_request_dimensions(
                requested_timesteps=runtime_metadata.timesteps,
                requested_input_dim=runtime_metadata.input_dim,
            )
        except ValueError as exc:
            return self._request_boundary_failure_metrics(
                error_kind="request",
                error_detail=str(exc),
            )
        resolved_device_options = resolve_device_options(self._normalized_device_name(), self.config.device) or {}
        filtered_device_options_overrides = {
            key: value
            for key, value in dict(device_options_overrides or {}).items()
            if value is not None
        }
        requested_cpu_clock_mhz = -1
        if "cpu_clock_mhz" in filtered_device_options_overrides:
            raw_requested_cpu_clock_mhz = filtered_device_options_overrides["cpu_clock_mhz"]
            if isinstance(raw_requested_cpu_clock_mhz, bool):
                return self._request_boundary_failure_metrics(
                    error_kind="request",
                    error_detail="device_options_overrides.cpu_clock_mhz must be an integer.",
                )
            try:
                requested_cpu_clock_mhz = int(raw_requested_cpu_clock_mhz)
            except (TypeError, ValueError):
                return self._request_boundary_failure_metrics(
                    error_kind="request",
                    error_detail="device_options_overrides.cpu_clock_mhz must be an integer.",
                )
            filtered_device_options_overrides["cpu_clock_mhz"] = requested_cpu_clock_mhz
        merged_device_options = {
            **resolved_device_options,
            **filtered_device_options_overrides,
        }
        merged_device_options["latency_budget_ms"] = float(latency_budget_ms)
        try:
            runtime_device = get_microcontroller_device(
                self._normalized_device_name(),
                serial_port=getattr(self.config.device, "serial_port", None),
                device_options=merged_device_options,
            )
        except ValueError as exc:
            logger.error("Invalid runtime device options: %s", exc)
            return self._request_boundary_failure_metrics(
                error_kind="request" if filtered_device_options_overrides else "config",
                error_detail=str(exc),
            )

        window_size, input_dim = self._resolve_runtime_dimensions()
        model = None
        calibration_split: DataSplit | None = None
        if runtime_device.requires_candidate_model():
            model = self._materialize_candidate_model(
                family_hparams,
                model_variant=model_variant,
                checkpoint_path=checkpoint_path,
            )
            if runtime_device.requires_training_data():
                try:
                    calibration_split = self._require_calibration_split()
                except ValueError as exc:
                    logger.error("Export calibration data unavailable: %s", exc)
                    return self._request_boundary_failure_metrics(
                        error_kind="config",
                        error_detail=str(exc),
                    )

        effective_hil_enabled = bool(
            self.config.device.hil and runtime_device.supports_runtime_measurement()
        )
        harness_serial_port = getattr(self.config.device, "harness_serial_port", None)
        energy_requires_explicit_harness = self._normalized_device_name() in {
            "PORTENTA_H7",
            "ARDUINO_NANO_33_BLE_SENSE",
        }
        effective_energy_aware = bool(
            self.config.training.energy_aware
            and effective_hil_enabled
            and runtime_device.supports_energy_measurement()
            and (
                bool(harness_serial_port)
                or not energy_requires_explicit_harness
            )
        )
        prepared_dir: Path | None = None
        try:
            prepared_dir = runtime_device.prepare_candidate(
                request=self._build_candidate_prepare_request(
                    model=model,
                    model_variant=model_variant,
                    checkpoint_path=checkpoint_path,
                    calibration_split=calibration_split,
                ),
            )
            prepared_dir = Path(prepared_dir)
            sketch_candidate = prepared_dir / "tinyodom_tcn.ino"
            if runtime_device.requires_candidate_model() and sketch_candidate.is_file():
                self.active_sketch_path = sketch_candidate
                logger.info("Using sketch variant: %s", self.active_sketch_path)
            elif runtime_device.requires_candidate_model():
                self.active_sketch_path = None

            print("Starting metric collection")
            try:
                request_metrics_args = build_collect_metrics_request(
                    config=self.config,
                    hyperparams=runtime_metadata,
                    latency_budget_ms=latency_budget_ms,
                    dirpath=prepared_dir,
                    device_options=merged_device_options,
                    hil_enabled=effective_hil_enabled,
                    energy_aware=effective_energy_aware,
                    window_size=window_size,
                    input_dim=input_dim,
                )
            except RuntimeError as exc:
                logger.error("Runtime metrics request failure: %s", exc)
                metrics = _build_backend_failure_metrics(
                    error_kind="config",
                    error_detail=str(exc),
                    latency_budget_ms=latency_budget_ms,
                    hil_enabled=effective_hil_enabled,
                    energy_aware=effective_energy_aware,
                )
            else:
                metrics = collect_metrics(request_metrics_args)
                metrics = self._run_arduino_cadenced_second_pass(
                    runtime_device=runtime_device,
                    prepared_dir=prepared_dir,
                    base_metrics=metrics,
                    request_metrics_args=request_metrics_args,
                    window_size=window_size,
                    input_dim=input_dim,
                )
        except (
            stm32_cube_clt.WorkflowError,
            stm32_runtime.STM32RuntimeProtocolError,
            serial.SerialException,
        ) as exc:
            if self._normalized_device_name() != STM32_BOARD_NAME:
                raise
            logger.error("STM backend failure: %s", exc)
            metrics = _build_backend_failure_metrics(
                error_kind=classify_stm32_backend_error(str(exc)),
                error_detail=str(exc),
                latency_budget_ms=latency_budget_ms,
                hil_enabled=effective_hil_enabled,
                energy_aware=effective_energy_aware,
            )
        finally:
            runtime_device.cleanup_prepared_candidate(prepared_dir)

        metrics["cpu_clock_mhz_requested"] = requested_cpu_clock_mhz
        if self.config.device.hil:
            metrics["latency_budget_ms"] = latency_budget_ms

        print(
            "[HIL REP] Runtime clock details: "
            f"requested_cpu_clock_mhz={requested_cpu_clock_mhz}, "
            f"effective_cpu_clock_mhz={merged_device_options.get('cpu_clock_mhz', -1)}, "
            f"reported_clock_hz={metrics.get('clock_hz', -1.0)}"
        )
        print("Metric collection complete")
        return metrics

    def _sync_sketch_variant(self) -> Path:
        """
        Copy the selected Arduino sketch variant into the active build directory.

        Selection depends on ``device.name``, optional Portenta
        ``device.portenta.target_core``, ``training.energy_aware``, and
        ``training.input_mode`` in the loaded config. Uniform sketches use the
        shared root ``sketches/tinyodom_tcn_*.ino`` assets, while
        representative/real analysis variants continue to use
        ``sketches/analysis_sketches``.

        Returns
        -------
        Path
            Path to the synchronized ``tinyodom_tcn.ino`` sketch.

        Raises
        ------
        ValueError
            If the configured input mode is unsupported.
        FileNotFoundError
            If a required variant sketch or input header is missing.
        """
        return _sync_arduino_sketch_variant_for_config(
            self.config,
            Path(self.config.outputs.tcn_dir),
            sketches_dir=self.sketch_variants_dir,
        )

    def set_input_mode(self, input_mode: str, *, runtime_phase: str = "back_to_back") -> Path:
        """
        Set the input mode and resynchronize the active Arduino sketch variant.

        Parameters
        ----------
        input_mode : str
            Desired input mode (``"uniform"``, ``"representative"``, or
            ``"real"``).
        runtime_phase : str, optional
            Runtime sketch phase (``"back_to_back"`` or ``"cadenced"``).

        Returns
        -------
        Path
            Path to the active synchronized sketch file.

        """
        runtime_device = get_microcontroller_device(
            self._normalized_device_name(),
            serial_port=getattr(self.config.device, "serial_port", None),
            device_options=resolve_device_options(self._normalized_device_name(), self.config.device),
        )
        self.active_sketch_path = runtime_device.set_input_mode(
            input_mode,
            outputs_dir=Path(self.config.outputs.tcn_dir),
            config=self.config,
            sketches_dir=self.sketch_variants_dir,
            runtime_phase=runtime_phase,
        )
        logger.info("Using sketch variant: %s", self.active_sketch_path)
        return self.active_sketch_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the TinyODOM HIL server.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to config YAML.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    config = load_config(cfg_path)
    _configure_logging(config.logging.level)

    server = HILServer(config_path=cfg_path, config=config)
    server.start()
