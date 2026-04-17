import csv
import json
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
from .microcontrollers import (
    get_device as get_microcontroller_device,
    resolve_device_options,
)
from .data import OxIODSplitData

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "nas_config.yaml"
REPO_ROOT = Path(__file__).resolve().parents[2]
# Keep the legacy constant name for compatibility with older callers, but point
# it at the canonical STM32 FSBL template that now lives under ``sketches/``.
STM32_DEFAULT_PROJECT_ROOT = (
    REPO_ROOT
    / "sketches"
    / "stm32"
    / "tinyodom_tcn_stm32"
    / "FSBL"
)
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

VALID_SCORE_TYPES = {"scoring-function", "multi-objective"}
VALID_DERIVED_METRIC_TYPES = {"add", "energy-budget-from-power"}
VALID_TERM_TYPES = {"weighted", "normalized-weighted", "boundary", "target"}
VALID_OBJECTIVE_DIRECTIONS = {"maximize", "minimize"}
VALID_PRUNE_CONDITIONS = {"gt", "gte", "lt", "lte"}
TRAINING_ONLY_METRICS = {"rmse_vel_x", "rmse_vel_y", "rmse_total"}
NONNEGATIVE_METRICS = {
    "rmse_vel_x",
    "rmse_vel_y",
    "rmse_total",
    "ram_bytes",
    "flash_bytes",
    "max_ram_bytes",
    "max_flash_bytes",
    "external_flash_bytes",
    "flops",
    "latency_ms",
    "energy_mj_per_inference",
    "avg_power_mw",
    "avg_current_ma",
    "bus_voltage_v",
    "cpu_clock_mhz_requested",
    "clock_hz",
    "latency_budget_ms",
    "arena_bytes",
}
BUILTIN_SCORE_METRICS = {
    "rmse_vel_x",
    "rmse_vel_y",
    "rmse_total",
    "ram_bytes",
    "flash_bytes",
    "max_ram_bytes",
    "max_flash_bytes",
    "external_flash_bytes",
    "flops",
    "latency_ms",
    "energy_mj_per_inference",
    "avg_power_mw",
    "avg_current_ma",
    "bus_voltage_v",
    "cpu_clock_mhz_requested",
    "clock_hz",
    "latency_budget_ms",
    "arena_bytes",
    "error_code",
}


class TrialLike(Protocol):
    """Minimal Optuna Trial interface used by log_trial."""

    def set_user_attr(self, key: str, value: Any) -> None:
        ...


@dataclass(frozen=True)
class ScoringResult:
    """Resolved scalar or multi-objective trial result.

    Parameters
    ----------
    rmse_vel_x : float
        Validation RMSE along X.
    rmse_vel_y : float
        Validation RMSE along Y.
    score : float | None
        Scalar score for single-objective runs. ``None`` for multi-objective runs.
    objective_names : list[str]
        Ordered objective labels.
    objective_values : list[float]
        Ordered objective values.
    objective_directions : list[str]
        Ordered objective directions matching Optuna conventions.
    """

    rmse_vel_x: float
    rmse_vel_y: float
    score: float | None
    objective_names: list[str]
    objective_values: list[float]
    objective_directions: list[str]


class ScoreConfigEvaluationError(ValueError):
    """Raised when a configured score cannot be evaluated at runtime.

    This is reserved for score-config resolution failures such as unavailable
    metrics, invalid runtime references, or derived metrics that cannot be
    computed from the current trial context.
    """


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
    serial_timeout_s : float | None, optional
        Post-``START`` runtime timeout forwarded to direct-serial backends.
    measured_inference_runs : int, optional
        Number of on-device inference invokes averaged into one measured HIL
        attempt.
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
    serial_timeout_s: float | None = None
    measured_inference_runs: int = 10
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


def is_multiobjective_score_config(score_config: Any) -> bool:
    """Return whether a resolved score config uses multi-objective optimization.

    Parameters
    ----------
    score_config : object
        Score configuration object with a ``type`` field.

    Returns
    -------
    bool
        ``True`` when ``score.type == "multi-objective"``.
    """
    score_type = getattr(score_config, "type", None)
    if score_type is None and hasattr(score_config, "get"):
        score_type = score_config.get("type")
    return str(score_type).strip().lower() == "multi-objective"


def get_score_config_directions(score_config: Any) -> list[str]:
    """Return Optuna directions for a resolved score configuration.

    Parameters
    ----------
    score_config : object
        Score configuration tree.

    Returns
    -------
    list[str]
        One direction for scalar mode or one per objective for multi-objective mode.
    """
    if not is_multiobjective_score_config(score_config):
        return ["maximize"]
    return [
        str(objective.direction)
        for objective in getattr(score_config.params, "objectives", [])
    ]


def _metric_unavailable(metric_name: str, value: Any) -> bool:
    """Return whether a metric value is unavailable for scoring.

    Parameters
    ----------
    metric_name : str
        Metric registry name.
    value : object
        Candidate runtime value for the metric.

    Returns
    -------
    bool
        ``True`` when the metric is missing, non-finite, or a negative
        sentinel for a metric that is expected to stay non-negative.
    """
    if value is None:
        return True
    if isinstance(value, str):
        return False
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return True
    if not np.isfinite(numeric_value):
        return True
    return metric_name in NONNEGATIVE_METRICS and numeric_value < 0.0


def _resolve_metric_value(
    metric_name: str,
    context: dict[str, Any],
    score_config: Dict,
    stack: tuple[str, ...] = (),
) -> float:
    """Resolve a metric from the runtime scoring context.

    Parameters
    ----------
    metric_name : str
        Metric registry key to resolve.
    context : dict[str, Any]
        Runtime scoring context populated from metrics, hyperparameters, and
        previously resolved derived metrics.
    score_config : addict.Dict
        Validated score configuration tree.
    stack : tuple[str, ...], optional
        Active recursion stack used to detect cycles in derived metrics.

    Returns
    -------
    float
        Resolved metric value.

    Raises
    ------
    ValueError
        If the metric is unavailable, undefined, or participates in a cyclic
        derived-metric graph.
    """
    if metric_name in context:
        value = context[metric_name]
        if _metric_unavailable(metric_name, value):
            raise ValueError(f"Metric '{metric_name}' is unavailable for scoring.")
        return float(value)

    score_metrics = getattr(score_config, "metrics", Dict())
    if metric_name not in score_metrics:
        raise ValueError(f"Metric '{metric_name}' is not defined in the scoring registry.")
    if metric_name in stack:
        raise ValueError(f"Cycle detected while resolving score metric '{metric_name}'.")

    metric_config = score_metrics[metric_name]
    try:
        if metric_config.type == "add":
            # Derived metrics intentionally stay simple in v1: resolve each
            # child metric and sum the results.
            resolved = sum(
                _resolve_metric_value(child_name, context, score_config, stack + (metric_name,))
                for child_name in metric_config.metrics
            )
        elif metric_config.type == "energy-budget-from-power":
            power_mw = _typed_reference_value(metric_config.power_mw, context, score_config)
            duration_ms = _typed_reference_value(metric_config.duration_ms, context, score_config)
            if duration_ms <= 0.0:
                raise ValueError(
                    f"Derived metric '{metric_name}' requires a positive duration reference."
                )
            if power_mw < 0.0:
                raise ValueError(
                    f"Derived metric '{metric_name}' requires a non-negative power reference."
                )
            # Power is expressed in mW and duration in ms, so dividing by 1000
            # yields energy in mJ.
            resolved = (power_mw * duration_ms) / 1000.0
        else:
            raise ValueError(f"Unsupported derived metric type '{metric_config.type}'.")
    except ValueError:
        # Cache the standardized unavailable sentinel so repeated lookups for the
        # same derived metric stay consistent within one trial.
        context[metric_name] = -1.0
        raise
    context[metric_name] = float(resolved)
    return float(resolved)


def _typed_reference_value(reference: Dict, context: dict[str, Any], score_config: Dict) -> float:
    """Resolve a typed literal-or-metric reference entry.

    Parameters
    ----------
    reference : addict.Dict
        Typed reference config with ``type`` equal to ``literal`` or ``metric``.
    context : dict[str, Any]
        Runtime scoring context.
    score_config : addict.Dict
        Validated score configuration tree.

    Returns
    -------
    float
        Numeric reference value used by scalar score terms that compare or
        normalize against another value.
    """
    if reference.type == "literal":
        return float(reference.value)
    return float(_resolve_metric_value(reference.metric, context, score_config))


def _evaluate_score_config(
    rmse_vel_x: float,
    rmse_vel_y: float,
    metrics: dict[str, Any],
    hyperparams: Dict,
    score_config: Dict,
) -> ScoringResult:
    """Evaluate scalar or multi-objective scores from the resolved config.

    Parameters
    ----------
    rmse_vel_x : float
        Validation RMSE along X.
    rmse_vel_y : float
        Validation RMSE along Y.
    metrics : dict[str, Any]
        Runtime metrics collected for the trial.
    hyperparams : addict.Dict
        Sampled hyperparameters for the trial.
    score_config : addict.Dict
        Validated score configuration tree.

    Returns
    -------
    ScoringResult
        Structured scalar or multi-objective result ready for Optuna and CSV
        logging.

    Raises
    ------
    ScoreConfigEvaluationError
        If any configured metric or derived metric is unavailable, or if a
        normalized reference resolves to a non-positive value.
    """
    context = dict(metrics)
    context["rmse_vel_x"] = rmse_vel_x
    context["rmse_vel_y"] = rmse_vel_y
    context["rmse_total"] = metrics.get("rmse_total", -1.0)
    context["flops"] = hyperparams["flops"]

    def _resolve_score_metric(metric_name: str) -> float:
        try:
            return _resolve_metric_value(metric_name, context, score_config)
        except ValueError as exc:
            raise ScoreConfigEvaluationError(str(exc)) from exc

    def _resolve_score_reference(reference: Dict) -> float:
        try:
            return _typed_reference_value(reference, context, score_config)
        except ValueError as exc:
            raise ScoreConfigEvaluationError(str(exc)) from exc

    if not is_multiobjective_score_config(score_config):
        score_total = 0.0
        for term in score_config.params.terms:
            metric_value = _resolve_score_metric(term.metric)
            weight = float(term.get("weight", 1.0))
            if term.type == "weighted":
                score_total += weight * metric_value
            elif term.type == "normalized-weighted":
                reference_value = _resolve_score_reference(term.reference)
                if reference_value <= 0.0:
                    raise ScoreConfigEvaluationError(
                        f"Normalized reference for metric '{term.metric}' must be greater than zero."
                    )
                score_total += weight * (metric_value / reference_value)
            elif term.type == "boundary":
                reference_value = _resolve_score_reference(term.reference)
                score_total -= weight * max(0.0, metric_value - reference_value)
            elif term.type == "target":
                reference_value = _resolve_score_reference(term.reference)
                score_total -= weight * abs(metric_value - reference_value)
            else:
                raise ValueError(f"Unsupported scalar score term '{term.type}'.")
        return ScoringResult(
            rmse_vel_x=rmse_vel_x,
            rmse_vel_y=rmse_vel_y,
            score=float(score_total),
            objective_names=["score"],
            objective_values=[float(score_total)],
            objective_directions=["maximize"],
        )

    objective_names: list[str] = []
    objective_values: list[float] = []
    objective_directions: list[str] = []
    for objective in score_config.params.objectives:
        objective_names.append(str(objective.metric))
        objective_values.append(_resolve_score_metric(objective.metric))
        objective_directions.append(str(objective.direction))
    return ScoringResult(
        rmse_vel_x=rmse_vel_x,
        rmse_vel_y=rmse_vel_y,
        score=None,
        objective_names=objective_names,
        objective_values=objective_values,
        objective_directions=objective_directions,
    )


def _validate_typed_reference(reference: Any, allowed_metrics: set[str], context_name: str) -> Dict:
    """Validate and normalize a typed reference entry.

    Parameters
    ----------
    reference : object
        Raw reference configuration.
    allowed_metrics : set[str]
        Metric names that are valid in the current score configuration.
    context_name : str
        Human-readable label used in validation errors.

    Returns
    -------
    addict.Dict
        Normalized typed reference configuration.

    Raises
    ------
    ValueError
        If the reference shape or values are invalid.
    """
    if not isinstance(reference, (dict, Dict)):
        raise ValueError(f"{context_name} reference must be a mapping.")
    normalized_reference = Dict(reference)
    ref_type = str(normalized_reference.get("type", "")).strip().lower()
    if ref_type not in {"metric", "literal"}:
        raise ValueError(f"{context_name} reference.type must be 'metric' or 'literal'.")
    normalized_reference.type = ref_type
    if ref_type == "metric":
        metric_name = str(normalized_reference.get("metric", "")).strip()
        if metric_name not in allowed_metrics:
            raise ValueError(f"{context_name} references unknown metric '{metric_name}'.")
        normalized_reference.metric = metric_name
    else:
        try:
            normalized_reference.value = float(normalized_reference["value"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{context_name} literal reference must define a numeric value.") from exc
    return normalized_reference


def _metric_depends_on_training(
    metric_name: str,
    score_metrics: Dict,
    stack: tuple[str, ...] = (),
) -> bool:
    """Return whether a metric depends on training-only quantities.

    Parameters
    ----------
    metric_name : str
        Metric name to inspect.
    score_metrics : addict.Dict
        Normalized derived metrics from ``nas.score.metrics``.
    stack : tuple[str, ...], optional
        Active recursion stack used to detect cycles.

    Returns
    -------
    bool
        ``True`` when the metric or any derived dependency requires
        post-training values such as RMSE.
    """
    if metric_name in TRAINING_ONLY_METRICS:
        return True
    if metric_name not in score_metrics or metric_name in stack:
        return False

    metric_cfg = score_metrics[metric_name]
    child_metrics: list[str] = []
    if metric_cfg.type == "add":
        child_metrics = list(metric_cfg.metrics)
    elif metric_cfg.type == "energy-budget-from-power":
        if metric_cfg.power_mw.type == "metric":
            child_metrics.append(str(metric_cfg.power_mw.metric))
        if metric_cfg.duration_ms.type == "metric":
            child_metrics.append(str(metric_cfg.duration_ms.metric))
    return any(
        _metric_depends_on_training(child_metric, score_metrics, stack + (metric_name,))
        for child_metric in child_metrics
    )


def score_config_uses_training_metrics(score_config: Dict) -> bool:
    """Return whether a score config depends on post-training RMSE metrics.

    Parameters
    ----------
    score_config : addict.Dict
        Normalized score configuration.

    Returns
    -------
    bool
        ``True`` when any active term, objective, or typed metric reference
        depends directly or indirectly on training-only metrics.
    """
    score_metrics = getattr(score_config, "metrics", Dict())

    def _reference_depends_on_training(reference: Any) -> bool:
        return (
            reference is not None
            and getattr(reference, "type", None) == "metric"
            and _metric_depends_on_training(str(reference.metric), score_metrics)
        )

    if is_multiobjective_score_config(score_config):
        return any(
            _metric_depends_on_training(str(objective.metric), score_metrics)
            for objective in getattr(score_config.params, "objectives", [])
        )

    for term in getattr(score_config.params, "terms", []):
        if _metric_depends_on_training(str(term.metric), score_metrics):
            return True
        if _reference_depends_on_training(getattr(term, "reference", None)):
            return True
    return False


def _validate_score_config(score_input: Any, has_legacy_multiobjective: bool = False) -> Dict:
    """Validate and normalize the NAS score configuration.

    Parameters
    ----------
    score_input : object
        Raw ``nas.score`` configuration.
    has_legacy_multiobjective : bool, optional
        Whether the deprecated ``training.nas_multiobjective`` field is still
        present in the source configuration.

    Returns
    -------
    addict.Dict
        Normalized score configuration.

    Raises
    ------
    KeyError
        If the required score section is missing or the deprecated
        ``training.nas_multiobjective`` field is still present.
    ValueError
        If any score type, metric, term, reference, or objective entry is
        invalid.
    """
    if score_input is None:
        raise KeyError("Missing required 'nas.score' section in the configuration.")
    if has_legacy_multiobjective:
        raise KeyError(
            "training.nas_multiobjective is no longer supported; define nas.score.type instead."
        )

    score_config = Dict(score_input)
    score_type = str(score_config.get("type", "")).strip().lower()
    if score_type not in VALID_SCORE_TYPES:
        raise ValueError("score.type must be one of: scoring-function, multi-objective.")
    score_config.type = score_type
    score_config.metrics = Dict(score_config.get("metrics", {}))
    score_config.params = Dict(score_config.get("params", {}))

    custom_metric_names = set(score_config.metrics.keys())
    duplicate_names = BUILTIN_SCORE_METRICS & custom_metric_names
    if duplicate_names:
        duplicate_name = sorted(duplicate_names)[0]
        raise ValueError(f"score.metrics may not redefine built-in metric '{duplicate_name}'.")
    allowed_metric_names = BUILTIN_SCORE_METRICS | custom_metric_names

    normalized_metrics = Dict()
    for metric_name, raw_metric in score_config.metrics.items():
        metric_cfg = Dict(raw_metric)
        metric_type = str(metric_cfg.get("type", "")).strip().lower()
        if metric_type not in VALID_DERIVED_METRIC_TYPES:
            raise ValueError(
                f"score.metrics.{metric_name}.type must be one of: {sorted(VALID_DERIVED_METRIC_TYPES)}."
            )
        metric_cfg.type = metric_type
        if metric_type == "add":
            metric_list = metric_cfg.get("metrics")
            if not isinstance(metric_list, list) or len(metric_list) == 0:
                raise ValueError(f"score.metrics.{metric_name}.metrics must be a non-empty list.")
            normalized_metric_list = []
            for child_metric in metric_list:
                child_name = str(child_metric).strip()
                if child_name not in allowed_metric_names:
                    raise ValueError(
                        f"score.metrics.{metric_name} references unknown metric '{child_name}'."
                    )
                normalized_metric_list.append(child_name)
            metric_cfg.metrics = normalized_metric_list
        elif metric_type == "energy-budget-from-power":
            if "power" in metric_cfg and "power_mw" in metric_cfg:
                raise ValueError(
                    f"score.metrics.{metric_name} may define only one of power or power_mw."
                )
            if "duration" in metric_cfg and "duration_ms" in metric_cfg:
                raise ValueError(
                    f"score.metrics.{metric_name} may define only one of duration or duration_ms."
                )

            # Keep the runtime representation explicit about units even when
            # reading older configs that still use the shorter key names.
            metric_cfg.power_mw = _validate_typed_reference(
                metric_cfg.get("power_mw", metric_cfg.get("power")),
                allowed_metric_names,
                f"score.metrics.{metric_name}.power_mw",
            )
            metric_cfg.duration_ms = _validate_typed_reference(
                metric_cfg.get("duration_ms", metric_cfg.get("duration")),
                allowed_metric_names,
                f"score.metrics.{metric_name}.duration_ms",
            )
            if (
                metric_cfg.duration_ms.type == "literal"
                and float(metric_cfg.duration_ms.value) <= 0.0
            ):
                raise ValueError(
                    f"score.metrics.{metric_name}.duration_ms literal reference must be greater than zero."
                )
            if metric_cfg.power_mw.type == "literal" and float(metric_cfg.power_mw.value) < 0.0:
                raise ValueError(
                    f"score.metrics.{metric_name}.power_mw literal reference must be non-negative."
                )
        normalized_metrics[metric_name] = metric_cfg
    score_config.metrics = normalized_metrics

    if score_type == "scoring-function":
        raw_terms = score_config.params.get("terms")
        if not isinstance(raw_terms, list) or len(raw_terms) == 0:
            raise ValueError("score.params.terms must be a non-empty list for scoring-function mode.")
        normalized_terms = []
        for idx, raw_term in enumerate(raw_terms):
            term = Dict(raw_term)
            term_type = str(term.get("type", "")).strip().lower()
            if term_type not in VALID_TERM_TYPES:
                raise ValueError(
                    f"score.params.terms[{idx}].type must be one of: {sorted(VALID_TERM_TYPES)}."
                )
            metric_name = str(term.get("metric", "")).strip()
            if metric_name not in allowed_metric_names:
                raise ValueError(f"score.params.terms[{idx}] references unknown metric '{metric_name}'.")
            term.type = term_type
            term.metric = metric_name
            term.weight = float(term.get("weight", 1.0))
            if term_type in {"normalized-weighted", "boundary", "target"}:
                term.reference = _validate_typed_reference(
                    term.get("reference"),
                    allowed_metric_names,
                    f"score.params.terms[{idx}]",
                )
                if (
                    term_type == "normalized-weighted"
                    and term.reference.type == "literal"
                    and float(term.reference.value) <= 0.0
                ):
                    raise ValueError(
                        f"score.params.terms[{idx}] normalized-weighted literal reference must be greater than zero."
                    )
            normalized_terms.append(term)
        score_config.params.terms = normalized_terms
    else:
        raw_objectives = score_config.params.get("objectives")
        if not isinstance(raw_objectives, list) or len(raw_objectives) == 0:
            raise ValueError("score.params.objectives must be a non-empty list for multi-objective mode.")
        normalized_objectives = []
        for idx, raw_objective in enumerate(raw_objectives):
            objective = Dict(raw_objective)
            metric_name = str(objective.get("metric", "")).strip()
            direction = str(objective.get("direction", "")).strip().lower()
            if metric_name not in allowed_metric_names:
                raise ValueError(f"score.params.objectives[{idx}] references unknown metric '{metric_name}'.")
            if direction not in VALID_OBJECTIVE_DIRECTIONS:
                raise ValueError(
                    f"score.params.objectives[{idx}].direction must be one of: {sorted(VALID_OBJECTIVE_DIRECTIONS)}."
                )
            objective.metric = metric_name
            objective.direction = direction
            normalized_objectives.append(objective)
        score_config.params.objectives = normalized_objectives
    return score_config


def _validate_prune_config(prune_input: Any, score_config: Dict) -> Dict:
    """Validate and normalize NAS prune policy.

    Parameters
    ----------
    prune_input : object
        Raw ``nas.prune`` configuration.
    score_config : addict.Dict
        Normalized ``nas.score`` configuration.

    Returns
    -------
    addict.Dict
        Normalized prune configuration with a ``rules`` list.

    Raises
    ------
    ValueError
        If any prune rule is invalid or incompatible with the score mode.
    """
    prune_config = Dict(prune_input or {})
    raw_rules = prune_config.get("rules", [])
    if raw_rules is None:
        raw_rules = []
    if not isinstance(raw_rules, list):
        raise ValueError("nas.prune.rules must be a list when provided.")
    if is_multiobjective_score_config(score_config) and len(raw_rules) > 0:
        raise ValueError("nas.prune.rules is only supported when nas.score.type is scoring-function.")

    allowed_metric_names = BUILTIN_SCORE_METRICS | set(getattr(score_config, "metrics", Dict()).keys())
    normalized_rules = []
    for idx, raw_rule in enumerate(raw_rules):
        rule_cfg = Dict(raw_rule)
        metric_name = str(rule_cfg.get("metric", "")).strip()
        condition = str(rule_cfg.get("condition", "")).strip().lower()
        if metric_name not in allowed_metric_names:
            raise ValueError(f"nas.prune.rules[{idx}] references unknown metric '{metric_name}'.")
        if condition not in VALID_PRUNE_CONDITIONS:
            raise ValueError(
                f"nas.prune.rules[{idx}].condition must be one of: {sorted(VALID_PRUNE_CONDITIONS)}."
            )
        if _metric_depends_on_training(metric_name, getattr(score_config, "metrics", Dict())):
            raise ValueError(
                f"nas.prune.rules[{idx}] may not use training-only metric '{metric_name}'."
            )
        reference = _validate_typed_reference(
            rule_cfg.get("reference"),
            allowed_metric_names,
            f"nas.prune.rules[{idx}]",
        )
        if reference.type == "metric" and _metric_depends_on_training(
            str(reference.metric),
            getattr(score_config, "metrics", Dict()),
        ):
            raise ValueError(
                f"nas.prune.rules[{idx}] may not reference training-only metric '{reference.metric}'."
            )
        rule_cfg.metric = metric_name
        rule_cfg.condition = condition
        rule_cfg.reference = reference
        rule_cfg.reason = str(rule_cfg.get("reason", "")).strip()
        raw_rule_id = str(rule_cfg.get("rule", "")).strip()
        rule_cfg.rule = raw_rule_id if raw_rule_id else f"rule_{idx}"
        normalized_rules.append(rule_cfg)
    prune_config.rules = normalized_rules
    return prune_config


def _validate_nas_config(config: Dict) -> Dict:
    """Validate and normalize the top-level NAS policy configuration.

    Parameters
    ----------
    config : addict.Dict
        Parsed YAML configuration tree.

    Returns
    -------
    addict.Dict
        Normalized ``nas`` subtree containing ``score`` and ``prune``.

    Raises
    ------
    KeyError
        If the NAS policy section is missing or legacy score fields are used.
    """
    if "score" in config:
        raise KeyError("Top-level 'score' is no longer supported; move it to 'nas.score'.")
    if "nas" not in config:
        raise KeyError("Missing required top-level 'nas' section in the configuration.")

    nas_config = Dict(config.nas)
    score_config = _validate_score_config(
        nas_config.get("score"),
        has_legacy_multiobjective="nas_multiobjective" in config.training,
    )
    prune_config = _validate_prune_config(nas_config.get("prune", {}), score_config)
    nas_config.score = score_config
    nas_config.prune = prune_config
    return nas_config


def evaluate_prune_rules(
    metrics: dict[str, Any],
    hyperparams: Dict,
    score_config: Dict,
    prune_config: Dict,
) -> tuple[str, str] | None:
    """Evaluate pre-training prune rules against the current trial context.

    Parameters
    ----------
    metrics : dict[str, Any]
        Runtime metrics available before training.
    hyperparams : addict.Dict
        Trial hyperparameters.
    score_config : addict.Dict
        Normalized score configuration that owns the derived metric registry.
    prune_config : addict.Dict
        Normalized prune configuration.

    Returns
    -------
    tuple[str, str] | None
        ``(prune_rule, prune_reason)`` when a rule matches, otherwise ``None``.
    """
    context = dict(metrics)
    context["flops"] = hyperparams["flops"]

    for rule_cfg in getattr(prune_config, "rules", []):
        try:
            metric_value = _resolve_metric_value(rule_cfg.metric, context, score_config)
            reference_value = _typed_reference_value(rule_cfg.reference, context, score_config)
        except ValueError:
            return rule_cfg.rule, f"Configured prune metric unavailable: {rule_cfg.metric}"

        condition_matched = {
            "gt": metric_value > reference_value,
            "gte": metric_value >= reference_value,
            "lt": metric_value < reference_value,
            "lte": metric_value <= reference_value,
        }[rule_cfg.condition]
        if condition_matched:
            reason = (
                rule_cfg.reason
                or f"Prune rule '{rule_cfg.rule}' matched: {rule_cfg.metric} {rule_cfg.condition} {reference_value}"
            )
            return rule_cfg.rule, reason
    return None


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
    normalized_device_name = str(device_name).strip().upper()

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

    measured_inference_runs_raw = device.get("measured_inference_runs", 10)
    if isinstance(measured_inference_runs_raw, bool):
        raise ValueError("device.measured_inference_runs must be an integer >= 1.")
    if isinstance(measured_inference_runs_raw, float) and not measured_inference_runs_raw.is_integer():
        raise ValueError("device.measured_inference_runs must be an integer >= 1.")
    try:
        device.measured_inference_runs = int(measured_inference_runs_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("device.measured_inference_runs must be an integer >= 1.") from exc
    if device.measured_inference_runs < 1:
        raise ValueError("device.measured_inference_runs must be >= 1.")

    cpu_clock_mhz_options_raw = device.get("cpu_clock_mhz_options", None)
    if cpu_clock_mhz_options_raw is None:
        device.cpu_clock_mhz_options = None
    else:
        if not isinstance(cpu_clock_mhz_options_raw, list) or len(cpu_clock_mhz_options_raw) == 0:
            raise ValueError("device.cpu_clock_mhz_options must be a non-empty list of integers or null.")
        normalized_cpu_clock_mhz_options = []
        for value in cpu_clock_mhz_options_raw:
            if isinstance(value, bool):
                raise ValueError("device.cpu_clock_mhz_options entries must be integers, not booleans.")
            if isinstance(value, float) and not value.is_integer():
                raise ValueError("device.cpu_clock_mhz_options entries must be integers.")
            try:
                normalized_value = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("device.cpu_clock_mhz_options entries must be integers.") from exc
            normalized_cpu_clock_mhz_options.append(normalized_value)
        if normalized_device_name == "STM32_NUCLEO_N657X0_Q":
            from .microcontrollers.stm32_nucleo_n657x0 import SUPPORTED_CPU_CLOCK_MHZ

            allowed_cpu_clock_mhz_options = sorted(SUPPORTED_CPU_CLOCK_MHZ)
            for normalized_value in normalized_cpu_clock_mhz_options:
                if normalized_value not in SUPPORTED_CPU_CLOCK_MHZ:
                    raise ValueError(
                        "device.cpu_clock_mhz_options entries for STM32_NUCLEO_N657X0_Q "
                        f"must be one of: {allowed_cpu_clock_mhz_options}."
                    )
        device.cpu_clock_mhz_options = normalized_cpu_clock_mhz_options

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
    config.nas = _validate_nas_config(config)

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
    *,
    dirpath: Path,
    device_options: dict[str, Any] | None,
    hil_enabled: bool | None = None,
    energy_aware: bool | None = None,
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
        If runtime measurement requires a harness but ``device.harness_serial_port``
        is not configured.
    """
    def _cfg_get(container: Any, key: str, default: Any = None) -> Any:
        """Read a value from either an ``addict.Dict`` or a namespace-like object.

        Parameters
        ----------
        container : Any
            Config subtree or namespace object to read from.
        key : str
            Field name to resolve.
        default : Any, optional
            Fallback value when the field is absent.

        Returns
        -------
        Any
            Resolved field value or ``default``.
        """
        getter = getattr(container, "get", None)
        if callable(getter):
            return getter(key, default)
        return getattr(container, key, default)

    effective_energy_aware = bool(config.training.energy_aware) if energy_aware is None else bool(energy_aware)
    harness = None
    normalized_device_name = str(config.device.name).strip().upper()
    effective_hil_enabled = bool(config.device.hil) if hil_enabled is None else bool(hil_enabled)

    runtime_mode = "direct_serial"
    if effective_hil_enabled:
        try:
            runtime_device = get_microcontroller_device(
                normalized_device_name,
                serial_port=_cfg_get(config.device, "serial_port", None),
                device_options=device_options,
            )
        except ValueError:
            runtime_device = None
        if runtime_device is not None:
            runtime_mode_fn = getattr(runtime_device, "runtime_measure_mode", None)
            if callable(runtime_mode_fn):
                runtime_mode = str(runtime_mode_fn())

    if effective_energy_aware or runtime_mode == "harness_only":
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
    serial_timeout = _cfg_get(config.device, "serial_timeout_s", 12.0)
    if serial_timeout is None:
        serial_timeout = 12.0

    return CollectMetricsRequest(
        hil_enabled=effective_hil_enabled,
        energy_aware=effective_energy_aware,
        flops=hyperparams.flops,
        device_name=normalized_device_name,
        window_size=config.data.window_size,
        input_dim=hyperparams.input_dim,
        dirpath=Path(dirpath).resolve(),
        latency_proxy_max_flops=config.training.latency_proxy_max_flops,
        serial_port=_cfg_get(config.device, "serial_port", None),
        latency_budget_ms=latency_budget_ms,
        dut_ready_timeout_s=float(dut_ready_timeout),
        serial_timeout_s=float(serial_timeout),
        measured_inference_runs=int(_cfg_get(config.device, "measured_inference_runs", 10)),
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
        "measured_inference_runs": request.measured_inference_runs,
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
    if request.hil_enabled and request.serial_timeout_s is not None:
        controller_kwargs["serial_timeout_s"] = request.serial_timeout_s

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
    backend_error_kind = None
    backend_error_detail = None
    external_flash_bytes = -1
    weight_storage_mode = "embedded"
    if power_metrics:
        backend_error_kind = power_metrics.get("backend_error_kind")
        backend_error_detail = power_metrics.get("backend_error_detail")
        raw_external_flash_bytes = power_metrics.get("external_flash_bytes")
        if raw_external_flash_bytes is not None:
            try:
                parsed_external_flash_bytes = int(float(raw_external_flash_bytes))
            except (TypeError, ValueError):
                parsed_external_flash_bytes = -1
            if parsed_external_flash_bytes >= 0:
                external_flash_bytes = parsed_external_flash_bytes
        raw_weight_storage_mode = power_metrics.get("weight_storage_mode")
        if raw_weight_storage_mode:
            weight_storage_mode = str(raw_weight_storage_mode)
    normalized_power = normalize_power_metrics(power_metrics)
    harness_latency_ms = -1.0
    if normalized_power.get("harness_latency_s", -1.0) >= 0:
        harness_latency_ms = normalized_power["harness_latency_s"] * 1000.0
    metrics = {
        "ram_bytes": ram_bytes,
        "flash_bytes": flash_bytes,
        "external_flash_bytes": external_flash_bytes,
        "latency_ms": latency_ms if request.hil_enabled else -1,
        "latency_budget_ms": latency_budget_entry,
        "arena_bytes": arena_bytes,
        "hil_enabled": request.hil_enabled,
        "energy_aware": request.energy_aware,
        "weight_storage_mode": weight_storage_mode,
        "inference_seq": int(normalized_power["sequence"]) if normalized_power["sequence"] >= 0 else -1,
        "energy_mj_per_inference": normalized_power["energy_mj_per_inference"],
        "avg_power_mw": normalized_power["avg_power_mw"],
        "avg_current_ma": normalized_power["avg_current_ma"],
        "bus_voltage_v": normalized_power["bus_voltage_v"],
        "idle_power_mw": normalized_power["idle_power_mw"],
        "clock_hz": normalized_power["clock_hz"],
        "harness_latency_ms": harness_latency_ms,
    }
    set_error_code(metrics, error_code)
    if backend_error_kind is not None:
        metrics["backend_error_kind"] = str(backend_error_kind)
    if backend_error_detail is not None:
        metrics["backend_error_detail"] = str(backend_error_detail)

    return metrics

def log_trial(
    scoring_result: ScoringResult,
    metrics: dict,
    hyperparams: dict,
    trial: TrialLike,
    log_file_name: str,
    study_name: str = "",
    pruned: bool = False,
    prune_reason: str = "",
    prune_rule: str = "",
):
    """Write one trial summary row to CSV and annotate the Optuna trial.

    Parameters
    ----------
    scoring_result : ScoringResult
        Structured scalar or multi-objective score output.
    metrics : dict
        Resource metrics dict.
    hyperparams : dict
        Selected hyperparameters for the trial.
    trial : TrialLike
        Trial-like object (e.g., optuna.Trial) to annotate.
    log_file_name : str
        Path to the CSV log file.
    study_name : str, optional
        Name of the Optuna study.
    pruned : bool, optional
        Whether the trial was pruned.
    prune_reason : str, optional
        Reason for pruning.
    prune_rule : str, optional
        Stable machine-readable pruning rule identifier.

    Returns
    -------
    None
        Writes one CSV row and mutates ``trial`` user attributes in place.
    """
    log_path = Path(log_file_name)
    score_type = "multi-objective" if scoring_result.score is None else "scoring-function"
    header = [
        "study_name",
        "timestamp_unix",  # Added: Unix timestamp (seconds since epoch, float)
        "timestamp_readable",  # Added: Human-readable timestamp (MM-DD-YYYY HH:MM:SS)
        "score",
        "rmse_vel_x",
        "rmse_vel_y",
        "rmse_total",
        "ram_bytes",
        "flash_bytes",
        "external_flash_bytes",
        "weight_storage_mode",
        "flops",
        "latency_ms",
        "energy_mj_per_inference",
        "avg_power_mw",
        "avg_current_ma",
        "bus_voltage_v",
        "cpu_clock_mhz_requested",
        "clock_hz",
        "nb_filters",
        "kernel_size",
        "dilations",
        "dropout_rate",
        "use_skip_connections",
        "norm_flag",
        "error_code",
        "error_label",
        "score_type",
        "objective_names_json",
        "objective_values_json",
        "objective_directions_json",
        "pruned",
        "prune_reason",
        "prune_rule",
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
        "" if scoring_result.score is None else scoring_result.score,
        scoring_result.rmse_vel_x,
        scoring_result.rmse_vel_y,
        metrics.get("rmse_total", -1.0),
        metrics["ram_bytes"],
        metrics["flash_bytes"],
        metrics.get("external_flash_bytes", -1),
        metrics.get("weight_storage_mode", "embedded"),
        hyperparams["flops"],
        metrics["latency_ms"],
        metrics["energy_mj_per_inference"],
        metrics["avg_power_mw"],
        metrics["avg_current_ma"],
        metrics["bus_voltage_v"],
        metrics.get("cpu_clock_mhz_requested", -1),
        metrics.get("clock_hz", -1.0),
        hyperparams["nb_filters"],
        hyperparams["kernel_size"],
        hyperparams["dilations"],
        hyperparams["dropout_rate"],
        hyperparams["use_skip_connections"],
        hyperparams["norm_flag"],
        metrics["error_code"],
        metrics.get("error_label", describe_error_code(metrics["error_code"])),
        score_type,
        json.dumps(scoring_result.objective_names),
        json.dumps(scoring_result.objective_values),
        json.dumps(scoring_result.objective_directions),
        pruned,
        prune_reason,
        prune_rule,
    ]
    with open(log_path, "a", newline="") as csvfile:
        csv.writer(csvfile).writerow(row_write)

    trial.set_user_attr("ram_bytes", metrics["ram_bytes"])
    trial.set_user_attr("flash_bytes", metrics["flash_bytes"])
    trial.set_user_attr("external_flash_bytes", metrics.get("external_flash_bytes", -1))
    trial.set_user_attr("weight_storage_mode", metrics.get("weight_storage_mode", "embedded"))
    trial.set_user_attr("latency_ms", metrics["latency_ms"])
    trial.set_user_attr("latency_budget_ms", metrics["latency_budget_ms"])
    trial.set_user_attr("energy_mj_per_inference", metrics["energy_mj_per_inference"])
    trial.set_user_attr("cpu_clock_mhz_requested", metrics.get("cpu_clock_mhz_requested", -1))
    trial.set_user_attr("clock_hz", metrics.get("clock_hz", -1.0))
    trial.set_user_attr("rmse_vel_x", scoring_result.rmse_vel_x)
    trial.set_user_attr("rmse_vel_y", scoring_result.rmse_vel_y)
    trial.set_user_attr("rmse_total", metrics.get("rmse_total", -1.0))
    trial.set_user_attr("hil_error_code", metrics["error_code"])
    trial.set_user_attr("arena_bytes", metrics["arena_bytes"])
    trial.set_user_attr("flops", hyperparams["flops"])
    trial.set_user_attr("error_code", metrics["error_code"])
    trial.set_user_attr("score_type", score_type)
    trial.set_user_attr("objective_names", list(scoring_result.objective_names))
    trial.set_user_attr("objective_values", list(scoring_result.objective_values))
    trial.set_user_attr("objective_directions", list(scoring_result.objective_directions))
    error_label = metrics.get("error_label", describe_error_code(metrics["error_code"]))
    trial.set_user_attr("error_code_label", error_label)
    trial.set_user_attr("pruned", pruned)
    trial.set_user_attr("prune_reason", prune_reason)
    trial.set_user_attr("prune_rule", prune_rule)



def train_and_score(
    model,
    batch_size: int,
    hyperparams: Dict,
    metrics: dict,
    max_ram: float,
    max_flash: float,
    training_data: OxIODSplitData,
    validation_data: OxIODSplitData,
    config: Dict,
):
    """Train the model, compute validation RMSE, and evaluate the configured score.

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
        Maximum usable RAM on the device. Exposed to scoring as
        ``max_ram_bytes``.
    max_flash : float
        Maximum usable flash on the device. Exposed to scoring as
        ``max_flash_bytes``.
    training_data : OxIODSplitData
        Training dataset.
    validation_data : OxIODSplitData
        Validation dataset.
    config : addict.Dict
        NAS configuration tree.

    Returns
    -------
    ScoringResult
        Structured scalar or multi-objective result.

    Raises
    ------
    ValueError
        If the active scoring configuration references unavailable metrics.
    """

    # Surface device resource caps inside the scoring context so scalar terms
    # can normalize usage directly against the target hardware limits.
    metrics["max_ram_bytes"] = float(max_ram)
    metrics["max_flash_bytes"] = float(max_flash)

    if not config.training.train:
        rmse_vel_x = -1.0
        rmse_vel_y = -1.0
        metrics["rmse_vel_x"] = rmse_vel_x
        metrics["rmse_vel_y"] = rmse_vel_y
        # Keep the aggregate RMSE sentinel aligned with the individual RMSE
        # sentinels so config-driven scoring sees one consistent unavailable value.
        metrics["rmse_total"] = -1.0
        if (not metrics["hil_enabled"]) and metrics.get("error_code", 0) == 0:
            metrics["latency_ms"] = -1  # keep CSV compatibility for non-HIL trials
            set_error_code(metrics, 1)
        return _evaluate_score_config(rmse_vel_x, rmse_vel_y, metrics, hyperparams, config.nas.score)
    
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
    # Populate the built-in aggregate RMSE metric once so all downstream score
    # terms and objectives reference the same value.
    metrics["rmse_total"] = float(rmse_vel_x + rmse_vel_y)
    
    # Set error code for non-HIL trials that passed resource checks
    if (not metrics["hil_enabled"]) and metrics.get("error_code", 0) == 0:
        metrics["latency_ms"] = -1  # keep CSV compatibility for non-HIL trials
        set_error_code(metrics, 1)
    return _evaluate_score_config(rmse_vel_x, rmse_vel_y, metrics, hyperparams, config.nas.score)


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
