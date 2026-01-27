import csv
import itertools
import os
import sys
import time
from pathlib import Path
from typing import Any, Protocol

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

sys.path.insert(0, os.path.abspath("src"))
from hardware_utils_ex import (
    HIL_controller,
    describe_error_code,
    normalize_power_metrics,
)
from data_utils_ex import OxIODSplitData

DEFAULT_CONFIG_PATH = Path(__file__).with_name("nas_config.yaml")
MIN_TCN_LAYERS = 3
MAX_TCN_LAYERS = 8
DILATION_POOL = [1, 2, 4, 8, 16, 32, 64, 128, 256]
DILATION_CANDIDATES = [
    list(combo)
    for layer_count in range(MIN_TCN_LAYERS, MAX_TCN_LAYERS + 1)
    for combo in itertools.combinations(DILATION_POOL, layer_count)
]
DROP_RATE_CHOICES = [0.0, 0.1, 0.2, 0.3, 0.4]


class TrialLike(Protocol):
    """Minimal Optuna Trial interface used by log_trial."""

    def set_user_attr(self, key: str, value: Any) -> None:
        ...


def set_error_code(metrics: dict, code: int) -> None:
    """Attach a numeric error code and its descriptive label to `metrics`."""
    metrics["error_code"] = code
    metrics["error_label"] = describe_error_code(code)


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
    config.training.drop_rate_choices = DROP_RATE_CHOICES

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

def collect_metrics(
    hil_enabled: bool,
    flops: float,
    device_name: str,
    window_size: int,
    input_dim: int,
    dirpath: Path,
    latency_proxy_max_flops: float,
    serial_port: str | None,
    latency_budget_ms: float | None = None,
) -> dict:
    """ Gather RAM/flash/latency metrics from the controller for both HIL and proxy runs.

    Parameters
    ----------
    hil_enabled : bool
        Flag that indicates whether hardware-in-the-loop is active.
    flops : float
        FLOP count for the current architecture, used as a latency proxy when HIL is off.
    device_name : str
        Target hardware identifier (e.g., Arduino Nano BLE).
    window_size : int
        Inference window length used by the firmware.
    input_dim : int
        Number of channels per window.
    dirpath : Path
        Path to the directory containing compiled C output.
    latency_proxy_max_flops : float
        Maximum FLOPs for the latency proxy normalization.
    serial_port : str or None
        Serial port for HIL testing.
    latency_budget_ms : float | None, optional
        Upper-bound latency (in milliseconds) that defines the target cadence when
        HIL measurements are available. Required if ``hil_enabled`` is True.

    Returns
    -------
    dict
        RAM/flash/latency/arena metrics plus error codes shared across the trial.
    """
    # Prepare controller kwargs (ease of use and readability)
    controller_kwargs = {
        "dirpath": dirpath,
        "chosen_device": device_name,
        "window_size": window_size,
        "number_of_channels": input_dim,
    }
    
    if hil_enabled and serial_port is not None:
        controller_kwargs["serial_port"] = serial_port
    elif hil_enabled and serial_port is None:
        raise RuntimeError("Set serial_port before enabling HIL runs so uploads know which device to target.")
    
    # Run the HIL controller to get metrics. HIL_controller handles both HIL and proxy runs.
    (
        ram_bytes,
        flash_bytes,
        latency_s,
        arena_bytes,
        error_code,
        power_metrics,
    ) = HIL_controller(
        run_hil=hil_enabled,
        **controller_kwargs,
    )

    # Normalize None returns to -1 for CSV compatibility
    ram_bytes = ram_bytes if ram_bytes is not None else -1
    flash_bytes = flash_bytes if flash_bytes is not None else -1
    latency_ms = latency_s * 1000.0 if latency_s is not None else -1  # convert seconds → milliseconds

    # Normalize latency so downstream scoring logic can remain scale-invariant.
    latency_budget_entry = -1.0
    if hil_enabled:
        if latency_budget_ms is None:
            raise ValueError(
                "latency_budget_ms must be provided when hil_enabled is True so the"
                " normalized latency penalty has consistent units."
            )
        if latency_budget_ms <= 0:
            raise ValueError("latency_budget_ms must be a positive value")
        latency_budget_entry = latency_budget_ms
    elif latency_proxy_max_flops <= 0:
        raise ValueError("latency_proxy_max_flops must be a positive value")

    # Creates the metrics dict to return
    normalized_power = normalize_power_metrics(power_metrics)
    metrics = {
        "ram_bytes": ram_bytes,
        "flash_bytes": flash_bytes,
        "latency_ms": latency_ms if hil_enabled else -1,
        "latency_budget_ms": latency_budget_entry,
        "arena_bytes": arena_bytes,
        "hil_enabled": hil_enabled,
        "inference_seq": int(normalized_power["sequence"]) if normalized_power["sequence"] >= 0 else -1,
        "energy_mj_per_inference": normalized_power["energy_mj_per_inference"],
        "avg_power_mw": normalized_power["avg_power_mw"],
        "avg_current_ma": normalized_power["avg_current_ma"],
        "bus_voltage_v": normalized_power["bus_voltage_v"],
        "idle_power_mw": normalized_power["idle_power_mw"],
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
