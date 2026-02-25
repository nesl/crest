import csv
import itertools
import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import tensorflow as tf
import yaml
from addict import Dict
from sklearn.metrics import mean_squared_error

# import optuna
from tcn import TCN
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten, MaxPooling1D, Reshape
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)

from .hardware import (
    HIL_controller,
    describe_error_code,
    normalize_power_metrics,
)
from .microcontrollers import get_device as get_microcontroller_device
from .data import OxIODSplitData

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "nas_config.yaml"
MIN_TCN_LAYERS = 3
MAX_TCN_LAYERS = 8
DILATION_POOL = [1, 2, 4, 8, 16, 32, 64, 128, 256]
DILATION_CANDIDATES = [
    list(combo)
    for layer_count in range(MIN_TCN_LAYERS, MAX_TCN_LAYERS + 1)
    for combo in itertools.combinations(DILATION_POOL, layer_count)
]
DROP_RATE_CHOICES = [0.0, 0.1, 0.2, 0.3, 0.4]
logger = logging.getLogger(__name__)


class TrialLike(Protocol):
    """Minimal Optuna Trial interface used by log_trial."""

    def set_user_attr(self, key: str, value: Any) -> None:
        ...


@dataclass(frozen=True)
class HarnessConfig:
    """Energy-aware harness settings forwarded to ``HIL_controller``.

    Parameters
    ----------
    harness_serial_port : str | None
        Serial port for the INA228 harness.
    harness_fqbn : str | None
        FQBN used to compile/upload the harness sketch.
    harness_auto_flash : str | None
        Harness flashing policy (``once``, ``always``, ``never``).
    harness_arm_pin : int | None
        Harness arming GPIO pin.
    harness_trigger_pin : int | None
        Harness trigger GPIO pin.
    dut_arm_hold_ms : int | None
        Time to hold DUT arm low before trigger observation.
    harness_stable_low_ms : int | None
        Required stable-low arming duration.
    harness_ready_timeout_s : float | None
        Timeout waiting for ``HARNESS READY``.
    harness_arm_timeout_s : float | None
        Timeout waiting for a valid arm/trigger edge.
    harness_active_timeout_s : float | None
        Maximum active measurement window.
    harness_done_timeout_s : float | None
        Timeout waiting for ``DONE``.
    """

    harness_serial_port: str | None
    harness_fqbn: str | None
    harness_auto_flash: str | None
    harness_arm_pin: int | None
    harness_trigger_pin: int | None
    dut_arm_hold_ms: int | None
    harness_stable_low_ms: int | None
    harness_ready_timeout_s: float | None
    harness_arm_timeout_s: float | None
    harness_active_timeout_s: float | None
    harness_done_timeout_s: float | None


@dataclass(frozen=True)
class CollectMetricsRequest:
    """Normalized request used by :func:`collect_metrics`.

    Parameters
    ----------
    hil_enabled : bool
        Whether to run HIL upload/measurement (vs compile-only proxy mode).
    energy_aware : bool
        Whether harness-assisted power measurement is enabled.
    flops : float
        Model FLOP estimate for trial bookkeeping.
    device_name : str
        Target hardware name.
    window_size : int
        Input window length compiled into firmware.
    input_dim : int
        Number of input channels compiled into firmware.
    dirpath : pathlib.Path
        Firmware project directory containing generated model artifacts.
    latency_proxy_max_flops : float
        Maximum FLOPs used by proxy latency normalization.
    serial_port : str | None
        DUT serial port used for upload/latency capture during HIL runs.
    latency_budget_ms : float | None, optional
        Target inference cadence in milliseconds for normalized latency checks.
    dut_ready_timeout_s : float | None, optional
        Timeout waiting for DUT ready handshake.
    harness : HarnessConfig | None, optional
        Harness settings for energy-aware runs. ``None`` for non-energy-aware runs.
    device_options : dict[str, Any] | None, optional
        Optional board-specific options forwarded to the device factory.
    """

    hil_enabled: bool
    energy_aware: bool
    flops: float
    device_name: str
    window_size: int
    input_dim: int
    dirpath: Path
    latency_proxy_max_flops: float
    serial_port: str | None
    latency_budget_ms: float | None = None
    dut_ready_timeout_s: float | None = None
    harness: HarnessConfig | None = None
    device_options: dict[str, Any] | None = None


def set_error_code(metrics: dict, code: int) -> None:
    """Attach a numeric error code and its descriptive label to `metrics`."""
    metrics["error_code"] = code
    metrics["error_label"] = describe_error_code(code)


def validate_loaded_model_input_shape(model: tf.keras.Model, hyperparams: Dict) -> None:
    """Validate that a loaded checkpoint input shape matches HIL expectations.

    Parameters
    ----------
    model : tf.keras.Model
        Loaded Keras model from a checkpoint.
    hyperparams : Dict
        Hyperparameter set that provides expected ``timesteps`` and
        ``input_dim``.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the checkpoint has an unexpected number of inputs or incompatible
        input dimensions.
    """
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        if len(input_shape) != 1:
            raise ValueError(
                f"Expected a single-input model but checkpoint exposes {len(input_shape)} inputs."
            )
        input_shape = input_shape[0]
    if not isinstance(input_shape, tuple) or len(input_shape) != 3:
        raise ValueError(f"Unexpected checkpoint input shape: {input_shape}")

    expected_timesteps = int(hyperparams.timesteps)
    expected_input_dim = int(hyperparams.input_dim)
    actual_timesteps = input_shape[1]
    actual_input_dim = input_shape[2]
    if (
        actual_timesteps not in (None, expected_timesteps)
        or actual_input_dim not in (None, expected_input_dim)
    ):
        raise ValueError(
            "Checkpoint input shape mismatch: "
            f"expected (None, {expected_timesteps}, {expected_input_dim}), "
            f"got {input_shape}."
        )


def iter_layers(model: tf.keras.Model) -> list[tf.keras.layers.Layer]:
    """Flatten model layers across TensorFlow/Keras versions.

    Parameters
    ----------
    model : tf.keras.Model
        Model whose layers should be traversed.

    Returns
    -------
    list[tf.keras.layers.Layer]
        De-duplicated list of all layers reachable from the model graph.
    """
    layers: list[tf.keras.layers.Layer] = []
    seen_nodes: set[int] = set()

    if hasattr(model, "_flatten_layers"):
        try:
            for layer in model._flatten_layers(include_self=False, recursive=True):
                if id(layer) in seen_nodes:
                    continue
                seen_nodes.add(id(layer))
                layers.append(layer)
            if layers:
                return layers
        except TypeError:
            pass

    queue: deque[object] = deque([model])
    while queue:
        node = queue.popleft()
        node_id = id(node)
        if node_id in seen_nodes:
            continue
        seen_nodes.add(node_id)
        if isinstance(node, tf.keras.layers.Layer):
            layers.append(node)
        for child in getattr(node, "layers", []) or []:
            queue.append(child)
    return layers


def collect_bn_layers(model: tf.keras.Model) -> list[tf.keras.layers.BatchNormalization]:
    """Collect all BatchNormalization layers from a model.

    Parameters
    ----------
    model : tf.keras.Model
        Model to inspect.

    Returns
    -------
    list[tf.keras.layers.BatchNormalization]
        BatchNormalization layers found in the model graph.
    """
    out: list[tf.keras.layers.BatchNormalization] = []
    for layer in iter_layers(model):
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            out.append(layer)
    return out


def collect_non_bn_bias_layers(model: tf.keras.Model) -> list[tf.keras.layers.Layer]:
    """Collect non-BN layers that expose a ``bias`` variable.

    Parameters
    ----------
    model : tf.keras.Model
        Model to inspect.

    Returns
    -------
    list[tf.keras.layers.Layer]
        Layers that are not BatchNormalization and have a bias tensor.
    """
    out: list[tf.keras.layers.Layer] = []
    for layer in iter_layers(model):
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            continue
        bias = getattr(layer, "bias", None)
        if bias is None:
            continue
        out.append(layer)
    return out


def apply_combined_perturbation(
    model: tf.keras.Model,
    seed: int = 1337,
) -> tuple[int, int]:
    """Apply BN-full + non-BN-bias perturbations.

    Parameters
    ----------
    model : tf.keras.Model
        Model whose BN statistics/affine parameters and non-BN biases are
        perturbed in place.
    seed : int, optional
        Random seed used to make perturbation deterministic.

    Returns
    -------
    tuple[int, int]
        (number of BN layers touched, number of non-BN bias layers touched)
    """
    rng = np.random.default_rng(int(seed))
    bn_touched = 0
    for layer in collect_bn_layers(model):
        if layer.gamma is not None:
            gamma_values = rng.uniform(0.7, 1.3, size=layer.gamma.shape).astype(np.float32)
            layer.gamma.assign(gamma_values)
        if layer.beta is not None:
            beta_values = rng.uniform(-0.3, 0.3, size=layer.beta.shape).astype(np.float32)
            layer.beta.assign(beta_values)
        if layer.moving_mean is not None:
            mean_values = rng.uniform(-0.5, 0.5, size=layer.moving_mean.shape).astype(np.float32)
            layer.moving_mean.assign(mean_values)
        if layer.moving_variance is not None:
            var_values = rng.uniform(0.3, 2.0, size=layer.moving_variance.shape).astype(np.float32)
            layer.moving_variance.assign(var_values)
        bn_touched += 1

    bias_touched = 0
    for layer in collect_non_bn_bias_layers(model):
        bias = getattr(layer, "bias", None)
        if bias is None:
            continue
        bias_values = rng.uniform(-0.3, 0.3, size=bias.shape).astype(np.float32)
        bias.assign(bias_values)
        bias_touched += 1
    return bn_touched, bias_touched


def load_config(
    config_path: str | Path | None = None,
) -> Dict:
    """
    Load the NAS configuration from YAML, derive convenience paths/names,
    and return an addict.Dict for dot-attribute access.

    Parameters
    ----------
    config_path : str | Path | None
        Optional override for the YAML location. Defaults to src/nas_config.yaml.

    Returns
    -------
    addict.Dict
        Configuration tree with derived paths (models_dir, checkpoint paths, etc.).
    """
    cfg_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")

    config = Dict(yaml.safe_load(cfg_path.read_text()))

    # Validate required sections early so callers get actionable errors.
    for section in ("device", "outputs", "training"):
        if section not in config:
            raise KeyError(f"Missing '{section}' section in {cfg_path}")

    device_name = config.device.get("name")
    if not device_name:
        raise KeyError("Expected 'device.name' to be set in the configuration.")

    # Derive paths and names for models/checkpoints based on device name.
    outputs = config.outputs
    models_dir = Path(outputs.get("models_dir", "../models")).resolve()
    tcn_dir = Path(outputs.get("tcn_dir", "../tinyodom_tcn")).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    tcn_dir.mkdir(parents=True, exist_ok=True)

    # Stores derived model/checkpoint names and paths into the config
    model_stem = f"TinyOdomEx_OxIOD_{device_name}"
    outputs.model_name = f"{model_stem}.tflite"
    outputs.checkpoint_name = f"{model_stem}.keras"
    outputs.models_dir = models_dir
    outputs.tcn_dir = tcn_dir
    outputs.tflite_model_path = models_dir / outputs.model_name
    outputs.checkpoint_path = models_dir / outputs.checkpoint_name

    # Populate training choices from constants
    training = config.training
    if "nas_trials" not in training:
        raise KeyError("Expected 'training.nas_trials' to be set in the configuration.")
    if "max_total_trials" not in training:
        # Allow pruned/failed runs without risking an infinite loop.
        training.max_total_trials = int(training.nas_trials * 2)
    # If not explicitly set, disable training by default for faster debugging.
    training.energy_aware = bool(training.get("energy_aware", False))
    # Input mode selects which Arduino sketch variant is used during HIL runs.
    training.input_mode = str(training.get("input_mode", "uniform")).lower()
    config.training.drop_rate_choices = DROP_RATE_CHOICES

    device = config.device
    device.harness_fqbn = device.get("harness_fqbn", "arduino:mbed_nano:nano33ble")
    device.harness_auto_flash = str(device.get("harness_auto_flash", "once")).lower()
    device.harness_arm_pin = int(device.get("harness_arm_pin", 3))
    device.harness_trigger_pin = int(device.get("harness_trigger_pin", 2))
    device.dut_arm_hold_ms = int(device.get("dut_arm_hold_ms", 600))
    device.harness_stable_low_ms = int(device.get("harness_stable_low_ms", 500))
    device.harness_ready_timeout_s = float(device.get("harness_ready_timeout_s", 5.0))
    device.harness_arm_timeout_s = float(device.get("harness_arm_timeout_s", 5.0))
    device.harness_active_timeout_s = float(device.get("harness_active_timeout_s", 30.0))
    device.harness_done_timeout_s = float(device.get("harness_done_timeout_s", 5.0))
    device.dut_ready_timeout_s = float(device.get("dut_ready_timeout_s", 5.0))

    if training.energy_aware and not device.get("harness_serial_port"):
        raise RuntimeError(
            "Set device.harness_serial_port when energy_aware is True so the harness can be used."
        )
    if device.harness_arm_pin <= 0:
        raise ValueError("device.harness_arm_pin must be a positive integer.")
    if device.harness_trigger_pin <= 0:
        raise ValueError("device.harness_trigger_pin must be a positive integer.")
    if device.dut_arm_hold_ms <= 0:
        raise ValueError("device.dut_arm_hold_ms must be a positive integer.")
    if device.harness_stable_low_ms <= 0:
        raise ValueError("device.harness_stable_low_ms must be a positive integer.")
    if device.dut_arm_hold_ms <= device.harness_stable_low_ms:
        raise ValueError(
            "device.dut_arm_hold_ms must be greater than device.harness_stable_low_ms."
        )
    if device.harness_auto_flash not in {"once", "always", "never"}:
        raise ValueError(
            "device.harness_auto_flash must be one of: once, always, never."
        )

    # Logging level for runtime observability (used by hil_server.py and scripts).
    logging_section = Dict(config.get("logging", {}))
    level_name = str(logging_section.get("level", "INFO")).upper()
    valid_levels = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
    if level_name not in valid_levels:
        raise ValueError(
            "logging.level must be one of: CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET."
        )
    config.logging = Dict()
    config.logging.level = level_name

    return config


def count_flops(model, input_shape):
    """Estimate model FLOPs by profiling a frozen forward graph. 
    Replaces keras-flops.get_flops (deprecated).

    Parameters
    ----------
    model : tf.keras.Model
        Keras model with defined input signatures.
    input_shape : tuple[int]
        Input tensor shape excluding the batch dimension.

    Returns
    -------
    int
        Total floating point operations for a single forward pass with batch size 1.
    """
    concrete = tf.function(model).get_concrete_function(
        tf.TensorSpec([1, *input_shape], tf.float32)
    )
    frozen = convert_variables_to_constants_v2(concrete)
    graph_def = frozen.graph.as_graph_def()

    with tf.Graph().as_default() as graph:
        tf.compat.v1.import_graph_def(graph_def, name="")
        options = (
            tf.compat.v1.profiler.ProfileOptionBuilder(
                tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            )
            .with_empty_output()
            .build()
        )
        flops = tf.compat.v1.profiler.profile(graph, options=options)
    return flops.total_float_ops

def build_collect_metrics_request(
    config: Dict,
    hyperparams: Dict,
    latency_budget_ms: float,
) -> CollectMetricsRequest:
    """Build a :class:`CollectMetricsRequest` from full config and hyperparameters.

    Parameters
    ----------
    config : addict.Dict
        Loaded runtime configuration.
    hyperparams : addict.Dict
        Trial/model hyperparameters containing at least ``flops`` and ``input_dim``.
    latency_budget_ms : float
        Per-inference latency budget in milliseconds, derived from stride cadence.

    Returns
    -------
    CollectMetricsRequest
        Normalized request payload for :func:`collect_metrics`.

    Raises
    ------
    RuntimeError
        If ``training.energy_aware`` is enabled but ``device.harness_serial_port``
        is not configured, or when ``device.name`` is ``PORTENTA_H7`` and
        ``device.portenta.target_core`` is missing.
    """
    def _cfg_get(container: Any, key: str, default: Any = None) -> Any:
        getter = getattr(container, "get", None)
        if callable(getter):
            return getter(key, default)
        return getattr(container, key, default)

    energy_aware = bool(config.training.energy_aware)
    harness = None
    device_options: dict[str, Any] | None = None

    if str(config.device.name).strip().upper() == "PORTENTA_H7":
        portenta_cfg = _cfg_get(config.device, "portenta", None)
        target_core = _cfg_get(portenta_cfg, "target_core", None) if portenta_cfg is not None else None
        split = _cfg_get(portenta_cfg, "split", None) if portenta_cfg is not None else None
        security = _cfg_get(portenta_cfg, "security", None) if portenta_cfg is not None else None
        if not target_core:
            raise RuntimeError(
                "Set device.portenta.target_core to 'cm7' or 'cm4' when device.name is PORTENTA_H7."
            )
        device_options = {"target_core": str(target_core)}
        if split:
            device_options["split"] = str(split)
        if security:
            device_options["security"] = str(security)

    runtime_mode = "direct_serial"
    if bool(config.device.hil):
        try:
            runtime_device = get_microcontroller_device(
                str(config.device.name),
                serial_port=_cfg_get(config.device, "serial_port", None),
                device_options=device_options,
            )
        except ValueError:
            runtime_device = None
        if runtime_device is not None:
            runtime_mode_fn = getattr(runtime_device, "runtime_measure_mode", None)
            if callable(runtime_mode_fn):
                runtime_mode = str(runtime_mode_fn())

    if energy_aware or runtime_mode == "harness_only":
        harness_serial_port = _cfg_get(config.device, "harness_serial_port", None)
        if not harness_serial_port:
            raise RuntimeError(
                "Set device.harness_serial_port when runtime measurement requires the harness."
            )
        harness = HarnessConfig(
            harness_serial_port=harness_serial_port,
            harness_fqbn=_cfg_get(config.device, "harness_fqbn", None),
            harness_auto_flash=_cfg_get(config.device, "harness_auto_flash", None),
            harness_arm_pin=_cfg_get(config.device, "harness_arm_pin", None),
            harness_trigger_pin=_cfg_get(config.device, "harness_trigger_pin", None),
            dut_arm_hold_ms=_cfg_get(config.device, "dut_arm_hold_ms", None),
            harness_stable_low_ms=_cfg_get(config.device, "harness_stable_low_ms", None),
            harness_ready_timeout_s=_cfg_get(config.device, "harness_ready_timeout_s", None),
            harness_arm_timeout_s=_cfg_get(config.device, "harness_arm_timeout_s", None),
            harness_active_timeout_s=_cfg_get(config.device, "harness_active_timeout_s", None),
            harness_done_timeout_s=_cfg_get(config.device, "harness_done_timeout_s", None),
        )

    dut_ready_timeout = _cfg_get(config.device, "dut_ready_timeout_s", 5.0)
    if dut_ready_timeout is None:
        dut_ready_timeout = 5.0

    return CollectMetricsRequest(
        hil_enabled=bool(config.device.hil),
        energy_aware=energy_aware,
        flops=hyperparams.flops,
        device_name=config.device.name,
        window_size=config.data.window_size,
        input_dim=hyperparams.input_dim,
        dirpath=config.outputs.tcn_dir,
        latency_proxy_max_flops=config.training.latency_proxy_max_flops,
        serial_port=config.device.serial_port,
        latency_budget_ms=latency_budget_ms,
        dut_ready_timeout_s=float(dut_ready_timeout),
        harness=harness,
        device_options=device_options,
    )


def collect_metrics(request: CollectMetricsRequest) -> dict:
    """Gather RAM/flash/latency metrics from the controller for both HIL and proxy runs.

    Parameters
    ----------
    request : CollectMetricsRequest
        Normalized request containing all required/optional controller inputs.

    Returns
    -------
    dict
        RAM/flash/latency/arena metrics plus error codes shared across the trial.

    Raises
    ------
    RuntimeError
        If runtime measurement requires a harness but ``request.harness`` is missing.
    """
    # Prepare controller kwargs (ease of use and readability)
    controller_kwargs = {
        "dirpath": request.dirpath,
        "chosen_device": request.device_name,
        "window_size": request.window_size,
        "number_of_channels": request.input_dim,
    }
    if request.device_options is not None:
        controller_kwargs["device_options"] = request.device_options

    runtime_mode = "direct_serial"
    if request.hil_enabled:
        try:
            runtime_device = get_microcontroller_device(
                str(request.device_name),
                serial_port=request.serial_port,
                device_options=request.device_options,
            )
        except ValueError:
            runtime_device = None
        if runtime_device is not None:
            runtime_mode_fn = getattr(runtime_device, "runtime_measure_mode", None)
            if callable(runtime_mode_fn):
                runtime_mode = str(runtime_mode_fn())

    if request.energy_aware and request.harness is None:
        raise RuntimeError(
            "energy_aware=True requires harness configuration; do not run without harness."
        )
    if request.hil_enabled and runtime_mode == "harness_only" and request.harness is None:
        raise RuntimeError(
            "Runtime mode requires harness configuration. Set device.harness_serial_port."
        )
    
    if request.hil_enabled and request.serial_port is not None:
        controller_kwargs["serial_port"] = request.serial_port
    elif request.hil_enabled and request.serial_port is None:
        raise RuntimeError(
            "Set serial_port before enabling HIL runs so uploads know which DUT to target."
        )

    if request.hil_enabled and request.dut_ready_timeout_s is not None:
        controller_kwargs["dut_ready_timeout_s"] = request.dut_ready_timeout_s

    if (
        request.hil_enabled
        and request.harness is not None
        and (request.energy_aware or runtime_mode == "harness_only")
    ):
        controller_kwargs["harness_serial_port"] = request.harness.harness_serial_port
        controller_kwargs["harness_fqbn"] = request.harness.harness_fqbn
        controller_kwargs["harness_auto_flash"] = request.harness.harness_auto_flash
        controller_kwargs["harness_arm_pin"] = request.harness.harness_arm_pin
        controller_kwargs["harness_trigger_pin"] = request.harness.harness_trigger_pin
        controller_kwargs["dut_arm_hold_ms"] = request.harness.dut_arm_hold_ms
        controller_kwargs["harness_stable_low_ms"] = request.harness.harness_stable_low_ms
        controller_kwargs["harness_ready_timeout_s"] = request.harness.harness_ready_timeout_s
        controller_kwargs["harness_arm_timeout_s"] = request.harness.harness_arm_timeout_s
        controller_kwargs["harness_active_timeout_s"] = request.harness.harness_active_timeout_s
        controller_kwargs["harness_done_timeout_s"] = request.harness.harness_done_timeout_s
    
    # Run the HIL controller to get metrics. HIL_controller handles both HIL and proxy runs.
    logger.info(
        "collect_metrics: invoking HIL_controller (hil=%s, serial_port=%s, harness_port=%s)",
        request.hil_enabled,
        request.serial_port,
        request.harness.harness_serial_port if request.harness is not None else None,
    )
    (
        ram_bytes,
        flash_bytes,
        latency_s,
        arena_bytes,
        error_code,
        power_metrics,
    ) = HIL_controller(
        run_hil=request.hil_enabled,
        **controller_kwargs,
    )
    logger.info(
        "collect_metrics: HIL_controller finished (error_code=%s, latency_s=%s, arena_bytes=%s)",
        error_code,
        latency_s,
        arena_bytes,
    )

    # Normalize None returns to -1 for CSV compatibility
    ram_bytes = ram_bytes if ram_bytes is not None else -1
    flash_bytes = flash_bytes if flash_bytes is not None else -1
    latency_ms = latency_s * 1000.0 if latency_s is not None else -1  # convert seconds → milliseconds

    # Normalize latency so downstream scoring logic can remain scale-invariant.
    latency_budget_entry = -1.0
    if request.hil_enabled:
        if request.latency_budget_ms is None:
            raise ValueError(
                "latency_budget_ms must be provided when hil_enabled is True so the"
                " normalized latency penalty has consistent units."
            )
        if request.latency_budget_ms <= 0:
            raise ValueError("latency_budget_ms must be a positive value")
        latency_budget_entry = request.latency_budget_ms
    elif request.latency_proxy_max_flops <= 0:
        raise ValueError("latency_proxy_max_flops must be a positive value")

    # Creates the metrics dict to return
    normalized_power = normalize_power_metrics(power_metrics)
    harness_latency_ms = -1.0
    if normalized_power.get("harness_latency_s", -1.0) >= 0:
        harness_latency_ms = normalized_power["harness_latency_s"] * 1000.0
    metrics = {
        "ram_bytes": ram_bytes,
        "flash_bytes": flash_bytes,
        "latency_ms": latency_ms if request.hil_enabled else -1,
        "latency_budget_ms": latency_budget_entry,
        "arena_bytes": arena_bytes,
        "hil_enabled": request.hil_enabled,
        "inference_seq": int(normalized_power["sequence"]) if normalized_power["sequence"] >= 0 else -1,
        "energy_mj_per_inference": normalized_power["energy_mj_per_inference"],
        "avg_power_mw": normalized_power["avg_power_mw"],
        "avg_current_ma": normalized_power["avg_current_ma"],
        "bus_voltage_v": normalized_power["bus_voltage_v"],
        "idle_power_mw": normalized_power["idle_power_mw"],
        "harness_latency_ms": harness_latency_ms,
    }
    set_error_code(metrics, error_code)

    return metrics

def log_trial(score, 
              rmse_vel_x, 
              rmse_vel_y, 
              metrics: dict, 
              hyperparams: dict, 
              trial: TrialLike, 
              log_file_name: str, 
              study_name: str="",
              pruned: bool=False,
              prune_reason: str=""):
    """Writes the summary row to CSV and store metrics on the Optuna trial for later filtering/visualization.

    Parameters
    ----------
    score : float
        Computed Optuna objective value.
    rmse_vel_x : float
        Validation RMSE along X.
    rmse_vel_y : float
        Validation RMSE along Y.
    metrics : dict
        Resource metrics dict.
    hyperparams : dict
        Selected hyperparameters for the trial.
    trial : TrialLike
        Trial-like object (e.g., optuna.Trial) to annotate.
    log_file_name : str
        Path to the CSV log file.
    study_name : str
        Name of the Optuna study, by default None.
    pruned : bool
        Whether the trial was pruned, by default False.
    prune_reason : str
        Reason for pruning, by default "".
    """
    log_path = Path(log_file_name)
    header = [
        "study_name",
        "timestamp_unix",  # Added: Unix timestamp (seconds since epoch, float)
        "timestamp_readable",  # Added: Human-readable timestamp (MM-DD-YYYY HH:MM:SS)
        "score",
        "rmse_vel_x",
        "rmse_vel_y",
        "ram_bytes",
        "flash_bytes",
        "flops",
        "latency_ms",
        "energy_mj_per_inference",
        "avg_power_mw",
        "avg_current_ma",
        "bus_voltage_v",
        "nb_filters",
        "kernel_size",
        "dilations",
        "dropout_rate",
        "use_skip_connections",
        "norm_flag",
        "error_code",
        "error_label",
        "pruned",
        "prune_reason",
    ]
    
    if not log_path.exists() or log_path.stat().st_size == 0:
        # Seed the CSV with a header row so downstream tooling can rely on names
        with open(log_path, "w", newline="") as csvfile:
            csv.writer(csvfile).writerow(header)
    # Row mirrors TinyODOM CSV schema so downstream tooling stays compatible
    row_write = [
        study_name,
        time.time(),  # Added: Current Unix timestamp (float)
        time.strftime('%m-%d-%Y %H:%M:%S'),  # Added: Human-readable timestamp
        score,
        rmse_vel_x,
        rmse_vel_y,
        metrics["ram_bytes"],
        metrics["flash_bytes"],
        hyperparams["flops"],
        metrics["latency_ms"],
        metrics["energy_mj_per_inference"],
        metrics["avg_power_mw"],
        metrics["avg_current_ma"],
        metrics["bus_voltage_v"],
        hyperparams["nb_filters"],
        hyperparams["kernel_size"],
        hyperparams["dilations"],
        hyperparams["dropout_rate"],
        hyperparams["use_skip_connections"],
        hyperparams["norm_flag"],
        metrics["error_code"],
        metrics.get("error_label", describe_error_code(metrics["error_code"])),
        pruned,
        prune_reason,
    ]
    print("Design choice:", row_write)
    with open(log_path, "a", newline="") as csvfile:
        csv.writer(csvfile).writerow(row_write)

    trial.set_user_attr("ram_bytes", metrics["ram_bytes"])
    trial.set_user_attr("flash_bytes", metrics["flash_bytes"])
    trial.set_user_attr("latency_ms", metrics["latency_ms"])
    trial.set_user_attr("latency_budget_ms", metrics["latency_budget_ms"])
    trial.set_user_attr("energy_mj_per_inference", metrics["energy_mj_per_inference"])
    trial.set_user_attr("rmse_vel_x", rmse_vel_x)
    trial.set_user_attr("rmse_vel_y", rmse_vel_y)
    trial.set_user_attr("hil_error_code", metrics["error_code"])
    trial.set_user_attr("arena_bytes", metrics["arena_bytes"])
    trial.set_user_attr("flops", hyperparams["flops"])
    trial.set_user_attr("error_code", metrics["error_code"])
    error_label = metrics.get("error_label", describe_error_code(metrics["error_code"]))
    trial.set_user_attr("error_code_label", error_label)
    trial.set_user_attr("pruned", pruned)
    trial.set_user_attr("prune_reason", prune_reason)



def train_and_score(model, batch_size: int, hyperparams: Dict, metrics: dict, max_ram: float, max_flash: float, training_data: OxIODSplitData, validation_data: OxIODSplitData, config: Dict):
    """Train the model, compute validation RMSE, and return the composite Optuna score.

    Parameters
    ----------
    model : tf.keras.Model
        Model instance to train.
    batch_size : int
        Mini-batch size for SGD.
    hyperparams : addict.Dict
        Trial hyperparameters. Required to have flops key.
    metrics : dict
        Shared resource metrics dict updated in-place.
    max_ram : float
        Maximum usable RAM on the device.
    max_flash : float
        Maximum usable flash on the device.
    training_data : OxIODSplitData
        Training dataset.
    validation_data : OxIODSplitData
        Validation dataset.
    config : addict.Dict
        NAS configuration tree.

    Returns
    -------
    tuple : (float, float, float, float)
        (rmse_vel_x, rmse_vel_y, score, latency_or_energy).
    """

    if config.training.energy_aware:
        latency_or_energy = metrics["energy_mj_per_inference"]
    else:
        latency_or_energy = metrics["latency_ms"]

    if not config.training.train:
        # Skip training for debugging: use default values
        rmse_vel_x = -1
        rmse_vel_y = -1
        metrics["rmse_vel_x"] = rmse_vel_x
        metrics["rmse_vel_y"] = rmse_vel_y
        model_acc = -(rmse_vel_x + rmse_vel_y)  # = 2
        resource_usage = (metrics["ram_bytes"] / max_ram) + (metrics["flash_bytes"] / max_flash)
        # Note score is different here since no training
        latency_penalty = hyperparams["flops"] / config.training.latency_proxy_max_flops
        score = model_acc + 0.01 * resource_usage - 0.5 * latency_penalty
        if (not metrics["hil_enabled"]) and metrics.get("error_code", 0) == 0:
            metrics["latency_ms"] = -1  # keep CSV compatibility for non-HIL trials
            set_error_code(metrics, 1)
        return rmse_vel_x, rmse_vel_y, score, latency_or_energy
    
    # Train the model with early stopping and checkpointing
    checkpoint = ModelCheckpoint(
        str(config.outputs.checkpoint_path),
        monitor="val_loss",
        mode="min",
        verbose=1,
        save_best_only=True,
    )
    # Early stopping to prevent overfitting
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=40,
        mode="min",
        verbose=1,
        restore_best_weights=True,
    )

    # Fit the model with validation loss
    model.fit(
        x=training_data.inputs,
        y=[training_data.x_vel, training_data.y_vel],
        epochs=config.training.nas_epochs,
        shuffle=True,
        callbacks=[checkpoint, early_stop],
        batch_size=batch_size,
        validation_data=(validation_data.inputs, [validation_data.x_vel, validation_data.y_vel]),
    )

    # Load the best model from checkpoint
    model = load_model(str(config.outputs.checkpoint_path), custom_objects={"TCN": TCN})
    
    # Compute validation RMSE
    y_pred = model.predict(validation_data.inputs)
    rmse_vel_x = mean_squared_error(validation_data.x_vel, y_pred[0], squared=False)
    rmse_vel_y = mean_squared_error(validation_data.y_vel, y_pred[1], squared=False)
    metrics["rmse_vel_x"] = rmse_vel_x
    metrics["rmse_vel_y"] = rmse_vel_y
    model_acc = -(rmse_vel_x + rmse_vel_y)

    # Compute resource usage penalty
    resource_usage = (metrics["ram_bytes"] / max_ram) + (metrics["flash_bytes"] / max_flash)
    
    # Compute latency penalty
    if metrics["hil_enabled"]:
        latency_over_ratio = max(0.0, (metrics["latency_ms"] - metrics["latency_budget_ms"]) / metrics["latency_budget_ms"])
        latency_penalty = min(2.0, latency_over_ratio)
    else:
        latency_penalty = hyperparams["flops"] / config.training.latency_proxy_max_flops
    
    # Compute energy penalty with sane caps so a noisy measurement does not dominate the score
    # Target energy per inference is derived from latency budget and target power (default 50 mW)
    energy_penalty = 0.0
    energy_mj = metrics.get("energy_mj_per_inference", -1.0)
    latency_budget_ms = metrics.get("latency_budget_ms", -1.0)
    if energy_mj >= 0.0 and latency_budget_ms > 0.0:
        # Note: target power can be adjusted here if needed
        target_power_mw = 50.0 
        target_energy_mj = latency_budget_ms * target_power_mw / 1000.0
        raw_energy_penalty = 0.15 * (energy_mj - target_energy_mj)  # Target energy per inference
        # Clamp energy penalty between -0.5 and 2.0
        energy_penalty = max(-0.5, min(2.0, raw_energy_penalty))

    # Score the trial run
    score = model_acc + 0.01 * resource_usage - latency_penalty - energy_penalty
    
    # Set error code for non-HIL trials that passed resource checks
    if (not metrics["hil_enabled"]) and metrics.get("error_code", 0) == 0:
        metrics["latency_ms"] = -1  # keep CSV compatibility for non-HIL trials
        set_error_code(metrics, 1)
    return rmse_vel_x, rmse_vel_y, score, latency_or_energy


def build_tinyodom_model(hyperparams: Dict) -> Model:
    """Build a TinyODOM Keras model based on given hyperparameters.

    Parameters
    ----------
    hyperparams : addict.Dict
        Dictionary containing model hyperparameters, including:
        - timesteps : int
            Number of time steps in the input.
        - input_dim : int
            Number of input features per time step.
        - nb_filters : int
            Number of filters in the TCN layers.
        - kernel_size : int
            Kernel size for the TCN layers.
        - dilations : list of int
            Dilation rates for the TCN layers.
        - dropout_rate : float
            Dropout rate for the TCN layers.
        - use_skip_connections : bool
            Whether to use skip connections in the TCN.
        - norm_flag : bool
            Whether to use batch normalization in the TCN.

    Returns
    -------
    tf.keras.Model
        Compiled Keras model with TCN layers and post-processing for velocity prediction.
    """
    inputs = Input(shape=(hyperparams.timesteps, hyperparams.input_dim))

    features = TCN(
        nb_filters=hyperparams.nb_filters,
        kernel_size=hyperparams.kernel_size,
        dilations=hyperparams.dilations,
        dropout_rate=hyperparams.dropout_rate,
        use_skip_connections=hyperparams.use_skip_connections,
        use_batch_norm=hyperparams.norm_flag,
    )(inputs)

    # Mirror the TinyODOM post-processing stack. 
    # Each step updates `features` to the next layer's output
    features = Reshape((hyperparams.nb_filters, 1))(features)  # preserves symbolic tensor by staying in Keras space
    features = MaxPooling1D(pool_size=2)(features)
    features = Flatten()(features)
    features = Dense(32, activation="linear", name="pre")(features)
    
    # Outputs for velocity in X and Y directions
    vel_x = Dense(1, activation="linear", name="velx")(features)
    vel_y = Dense(1, activation="linear", name="vely")(features)

    # Define the Keras Model with specified inputs and outputs. Traces the layers from the given outputs
    model = Model(inputs=[inputs], outputs=[vel_x, vel_y])
    return model
