import argparse
import csv
import json
import importlib
import logging
import os
import sys
import time
import socket
import shutil
from pathlib import Path
from datetime import datetime
import zmq
from addict import Dict


import absl.logging
import numpy as np
import optuna
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error  # , root_mean_squared_error
from tcn import TCN
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from optuna.trial import TrialState
# from tensorflow.keras.layers import Dense, Flatten, MaxPooling1D, Reshape
from tensorflow.keras.models import load_model

sys.path.insert(0, os.path.abspath("src"))
import hardware_utils_ex
from data_utils_ex import import_oxiod_dataset

importlib.reload(hardware_utils_ex)

from hardware_utils_ex import (
    HIL_MASTER_ARENA_EXHAUSTED,
    HIL_MASTER_DEVICE_NOT_FOUND,
    HIL_MASTER_FATAL,
    HIL_MASTER_FLASH_OVERFLOW,
    HIL_MASTER_RAM_OVERFLOW,
    HIL_MASTER_SUCCESS,
    convert_to_tflite_model,
    return_hardware_specs,
)
from nas_utils_ex import build_tinyodom_model, train_and_score, count_flops, log_trial, load_config, DEFAULT_CONFIG_PATH, DILATION_CANDIDATES, DROP_RATE_CHOICES, set_error_code

tf.get_logger().setLevel(logging.ERROR)
absl.logging.set_verbosity(absl.logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

logger = logging.getLogger(__name__)

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
    latency_or_energy_name : str
    Label for the second objective in multi-objective mode ("latency [ms]"
    or "energy per inference [mJ]"), derived from ``config.training.energy_aware``.
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
    - Multi-objective NAS is enabled via ``config.training.nas_multiobjective``;
      when true, NSGA-II is used with directions [maximize accuracy, minimize
      latency/energy]. Otherwise, a single-objective TPE sampler is used using
      the scoring method described in nas_utils_ex's `train_and_score` function.
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
        self.config_path = Path(config_path)
        self.config = load_config(self.config_path)
        
        if self.config.device.hil is False:
            logger.warning("HIL is disabled in the configuration.")
        if self.config.training.nas_multiobjective:
            if self.config.training.energy_aware:
                logger.info("Using multi-objective NAS with energy awareness.")
                self.latency_or_energy_name = "energy per inference [mJ]"
            else:
                logger.info("Using multi-objective NAS with latency.")
                self.latency_or_energy_name = "latency [ms]"
        else:
            logger.info("Using single-objective NAS.")

        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.RCVTIMEO = self.config.network.recv_timeout_sec * 1000
        self.socket.SNDTIMEO = self.config.network.send_timeout_sec * 1000  # Avoid hanging forever during tunnel hiccups

        self.training_data = import_oxiod_dataset(type_flag=2, 
                                                  useMagnetometer=True, 
                                                  useStepCounter=True, 
                                                  AugmentationCopies=0,
                                                  dataset_folder=self.config.data.directory,
                                                  sub_folders=['handbag/','handheld/','pocket/','running/','slow_walking/','trolley/'],
                                                  sampling_rate=self.config.data.sampling_rate_hz, 
                                                  window_size=self.config.data.window_size, 
                                                  stride=self.config.data.stride, 
                                                  verbose=False)
        print("Imported Training Data")

        self.validation_data = import_oxiod_dataset(type_flag = 3, 
                                                    useMagnetometer = True, 
                                                    useStepCounter = True, 
                                                    AugmentationCopies = 0,
                                                    dataset_folder = self.config.data.directory,
                                                    sub_folders = ['handbag/','handheld/','pocket/','running/','slow_walking/','trolley/'],
                                                    sampling_rate = self.config.data.sampling_rate_hz, 
                                                    window_size = self.config.data.window_size, 
                                                    stride = self.config.data.stride, 
                                                    verbose=False)
        print("Imported Validation Data")

        self.test_data = import_oxiod_dataset(type_flag = 4,
                                              dataset_folder = self.config.data.directory,
                                              sub_folders = ['handbag/','handheld/','pocket/','running/','slow_walking/','trolley/'],
                                              sampling_rate = self.config.data.sampling_rate_hz, 
                                              window_size = self.config.data.window_size, 
                                              stride = self.config.data.stride, 
                                              verbose=False)
        print("Imported Test Data")

        endpoint = f"tcp://{self.config.network.host}:{self.config.network.port}"
        self.socket.connect(endpoint)
        print(f"[REQ] Connected to HIL server at {endpoint}")

        self.study_name = "default_study"

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


    def _hil_request(self, hyperparams):
        """Send hyperparameters to the HIL server and receive metrics.

        Parameters
        ----------
        hyperparams : dict
            Dictionary containing model hyperparameters such as nb_filters,
            kernel_size, dilations, etc.

        Returns
        -------
        dict or None
            Dictionary containing metrics like ram_bytes, flash_bytes, latency_ms,
            etc., or None if the request times out.
        """
        print(f"[REQ] Sending hyperparameters to {self.config.network.host}:{self.config.network.port}: {hyperparams}")

        try:
            self.socket.send_json(hyperparams)
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
            - If ``config.training.nas_multiobjective`` is False: returns a
                single scalar ``score`` (higher is better) that blends accuracy
                with latency/energy according to configuration.
            - If ``config.training.nas_multiobjective`` is True: returns a
                2-tuple ``(model_acc, latency_or_energy)`` where
                ``model_acc = -(rmse_vel_x + rmse_vel_y)`` and the second element
                is either latency (ms) or energy depending on
                ``config.training.energy_aware``.

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
        # Sample the CNN/TCN architecture knobs for this Optuna trial.
        # ~8 million combinations
        nb_filters = trial.suggest_int("nb_filters", 2, 63)
        kernel_size = trial.suggest_int("kernel_size", 2, 15)
        dropout_rate = trial.suggest_categorical("dropout_rate", DROP_RATE_CHOICES)
        use_skip_connections = trial.suggest_categorical("use_skip_connections", [True, False])
        norm_flag = trial.suggest_categorical("norm_flag", [True, False])
        dilations_index = trial.suggest_int("dilations_index", 0, len(DILATION_CANDIDATES) - 1)
        dilations = DILATION_CANDIDATES[dilations_index]

        # Build a model that matches the sampled hyperparameters so we can count FLOPs.
        batch_size, timesteps, input_dim = 256, self.config.data.window_size, self.training_data.inputs.shape[2]
        hyperparams = {
            "nb_filters": nb_filters,
            "kernel_size": kernel_size,
            "dropout_rate": dropout_rate,
            "use_skip_connections": use_skip_connections,
            "norm_flag": norm_flag,
            "dilations": dilations,
            "batch_size": batch_size,
            "timesteps": timesteps,
            "input_dim": input_dim,
        }
        
        model = build_tinyodom_model(Dict(hyperparams))
        optimizer = optimizers.Adam()
        model.compile(loss={"velx": "mse", "vely": "mse"}, optimizer=optimizer)

        # Returns total number of flops
        flops = count_flops(model, (timesteps, input_dim))

        hyperparams["flops"] = flops

        # Ask the HIL server to evaluate the candidate for resource usage and latency.
        metrics = self._hil_request(hyperparams)

        # Gets the hardware *estimated* specifications for the target device
        max_ram, max_flash = return_hardware_specs(self.config.device.name)

        rmse_vel_x = float("inf")
        rmse_vel_y = float("inf")
        penalty_acc = -100.0
        # Set a high penalty for energy or latency depending on the mode
        if self.config.training.energy_aware:
            penalty_energy_latency = 100.0
        else:
            # latency has been seen to go up to 1 second on the extreme, so set penalty accordingly
            penalty_energy_latency = 10000.0
        score = penalty_acc

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
            if not self.config.training.nas_multiobjective:
                trial.report(value, step=0)

        def _fail_with_penalty(prune_reason: str):
            """Helper to prune with a penalty score and log the failure."""
            # Multi-objective: return a dominated pair so the trial completes; single-objective still prunes.
            # Ensure required metrics are present for logging
            metrics.setdefault("latency_ms", penalty_energy_latency)
            metrics.setdefault("energy_mj_per_inference", penalty_energy_latency)
            metrics.setdefault("avg_power_mw", -1.0)
            metrics.setdefault("avg_current_ma", -1.0)
            metrics.setdefault("bus_voltage_v", -1.0)
            metrics.setdefault("latency_budget_ms", -1.0)
            metrics.setdefault("arena_bytes", -1)

            log_trial(
                score=penalty_acc,
                rmse_vel_x=rmse_vel_x,
                rmse_vel_y=rmse_vel_y,
                metrics=metrics,
                hyperparams=hyperparams,
                trial=trial,
                log_file_name=str(log_path),
                study_name=self.study_name,
                pruned=True,
                prune_reason=prune_reason,
            )

            if self.config.training.nas_multiobjective:
                # penalty_secondary = metrics["energy_mj_per_inference"] if self.config.training.energy_aware else metrics["latency_ms"]
                return penalty_acc, penalty_energy_latency

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
        arena_ok = metrics["arena_bytes"] != -1

        # Shouldn't get to here, still included for completeness
        if flash_failure or not resources_ok or not arena_ok:
            # Treat missing/invalid resource numbers as fatal so Optuna can move on.
            if not flash_failure and not resources_ok:
                set_error_code(metrics, HIL_MASTER_FATAL)
            elif not flash_failure and not arena_ok:
                set_error_code(metrics, HIL_MASTER_ARENA_EXHAUSTED)
            return _fail_with_penalty("Resource or arena check failed")

        # Only train/evaluate models that pass all resource checks.
        rmse_vel_x, rmse_vel_y, score, latency_or_energy = train_and_score(
            model,
            batch_size=batch_size,
            hyperparams=Dict(hyperparams),
            metrics=metrics,
            max_ram=max_ram,
            max_flash=max_flash,
            training_data=self.training_data,
            validation_data=self.validation_data,
            config=self.config
        )
        
        model_acc = -(rmse_vel_x + rmse_vel_y)

        # Return either single-objective score or multi-objective tuple
        if self.config.training.nas_multiobjective:
            if (
                not np.isfinite(model_acc)
                or not np.isfinite(latency_or_energy)
                or latency_or_energy < 0
                or model_acc == -5.0
            ):
                # If training produced invalid objectives, fall back to penalty pair.
                return _fail_with_penalty("Training failed to produce valid metrics")
            log_trial(
                score=score,
                rmse_vel_x=rmse_vel_x,
                rmse_vel_y=rmse_vel_y,
                metrics=metrics,
                hyperparams=hyperparams,
                trial=trial,
                log_file_name=str(log_path),
                study_name=self.study_name,
            )
            return model_acc, latency_or_energy
        else:
            log_trial(
                score=score,
                rmse_vel_x=rmse_vel_x,
                rmse_vel_y=rmse_vel_y,
                metrics=metrics,
                hyperparams=hyperparams,
                trial=trial,
                log_file_name=str(log_path),
                study_name=self.study_name,
            )
            return score

    def smoke_test(
        self,
        train: bool=True,
        hil: bool=False,
        trials: int=5,
        epochs: int=5,
        study_name: str="smoke_test_study",
        multiobjective: bool | None = None,
    ) -> None:
        """Run a quick Optuna smoke test with configurable training and HIL settings.

        Parameters
        ----------
        train : bool, optional
            Whether to enable model training during the test (default is True).
        hil : bool, optional
            Whether to enable hardware-in-the-loop testing (default is False).
        trials : int, optional
            Number of trials to run in the smoke test (default is 5).
        epochs : int, optional 
            Number of epochs to train during the smoke test (default is 5).
        study_name : str, optional
            Name of the Optuna study (default is "smoke_test_study").
        multiobjective : bool | None, optional
            Whether to run the smoke test in multi-objective mode. Defaults to
            the current configuration if not provided.
            
        Returns
        -------
        None
            Prints the best trial value, parameters, and runtime metrics.
        """
        self.study_name = study_name
        artifacts_dir = self._artifacts_dir()
        storage_uri = f"sqlite:///{artifacts_dir / 'optuna_smoke_test.db'}"
        multiobjective = self.config.training.nas_multiobjective if multiobjective is None else multiobjective
        _previous_hil = self.config.device.hil
        _previous_train = self.config.training.train
        _previous_epochs = self.config.training.nas_epochs
        _previous_multiobjective = self.config.training.nas_multiobjective
        try:
            self.config.device.hil = hil
            self.config.training.train = train
            self.config.training.nas_epochs = epochs  # Speed up smoke test
            self.config.training.nas_multiobjective = multiobjective
            if multiobjective:
                sampler = optuna.samplers.NSGAIISampler(
                    population_size=self.config.training.nas_multiobjective_population_size,
                    seed=42,
                )
                single_trial_study = optuna.create_study(
                    directions=["maximize", "minimize"],
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
            self.config.training.nas_multiobjective = _previous_multiobjective

        complete_trials = [t for t in single_trial_study.trials if t.state == TrialState.COMPLETE]
        if not complete_trials:
            print("[SMOKE] No completed trials to report.")
            return

        if multiobjective:
            pareto = single_trial_study.best_trials
            print(f"Pareto front ({len(pareto)} trial(s)):")
            for trial in pareto:
                print(f"  Trial {trial.number} values: {trial.values}")
                print("  Params:")
                for name, value in trial.params.items():
                    print(f"    {name}: {value}")
                print("  Runtime metrics (user attrs):")
                for key in ("ram_bytes", "flash_bytes", "latency_ms", "rmse_vel_x", "rmse_vel_y", "hil_error_code", "arena_bytes"):
                    print(f"    {key}: {trial.user_attrs.get(key)}")
        else:
            best_trial = single_trial_study.best_trial
            print(f"Single-trial value: {best_trial.value}")
            print("Best params:")
            for name, value in best_trial.params.items():
                print(f"  {name}: {value}")
            print("Runtime metrics (user attrs):")
            for key in ("ram_bytes", "flash_bytes", "latency_ms", "rmse_vel_x", "rmse_vel_y", "hil_error_code", "arena_bytes"):
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

        if self.config.training.nas_multiobjective:
            sampler = optuna.samplers.NSGAIISampler(
                        population_size=self.config.training.nas_multiobjective_population_size,
                        seed=42)
            study = optuna.create_study(
                    directions=["maximize", "minimize"],  # maximize the accuracy, minimize latency/energy
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
        storage_uri = f"sqlite:///{artifacts_dir / 'optuna.db'}"

        # 1) Run NAS with configured HIL/train settings.
        study = self.run_nas(study_name=study_name, storage=storage_uri)
        print(f"[run_scoring_nas] Completed NAS study with {len(study.trials)} total trials")
        
        # Persist raw Optuna trial history so convergence plots can be rebuilt later.
        trials_df = study.trials_dataframe()
        trials_csv = artifacts_dir / "trials.csv"
        trials_df.to_csv(trials_csv, index=False)
        print(f"[run_scoring_nas] Saved trials dataframe to {trials_csv}")

        if self.config.training.nas_multiobjective:
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

        # Derive full hyperparameter set expected by the model builder.
        # The search space stores only indices/choices; fill in derived values here.
        dilations = DILATION_CANDIDATES[best_params["dilations_index"]]
        batch_size = 256  # Use the same fixed batch size as in NAS search.
        hyperparams = Dict(
            {
                "nb_filters": best_params["nb_filters"],
                "kernel_size": best_params["kernel_size"],
                "dropout_rate": best_params["dropout_rate"],
                "use_skip_connections": best_params["use_skip_connections"],
                "norm_flag": best_params["norm_flag"],
                "dilations": dilations,
                "timesteps": self.config.data.window_size,
                "input_dim": self.training_data.inputs.shape[2],
                "batch_size": batch_size,
            }
        )

        # Rebuild a fresh model and compile with a standard optimizer/loss.
        model = build_tinyodom_model(hyperparams)
        model.compile(loss={"velx": "mse", "vely": "mse"}, optimizer=optimizers.Adam())

        # Decide whether to hold out validation or fold it into training for the final fit.
        if combine_train_val:
            # Concatenate train and validation to squeeze the most data; monitor training loss instead.
            train_inputs = np.concatenate([self.training_data.inputs, self.validation_data.inputs], axis=0)
            train_x_vel = np.concatenate([self.training_data.x_vel, self.validation_data.x_vel], axis=0)
            train_y_vel = np.concatenate([self.training_data.y_vel, self.validation_data.y_vel], axis=0)
            val_data = None
            monitor_metric = "loss"
        else:
            train_inputs = self.training_data.inputs
            train_x_vel = self.training_data.x_vel
            train_y_vel = self.training_data.y_vel
            val_data = (self.validation_data.inputs, [self.validation_data.x_vel, self.validation_data.y_vel])
            monitor_metric = "val_loss"

        # Configure callbacks for checkpointing and early stopping.
        checkpoint_cb = ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor=monitor_metric,
            mode="min",
            verbose=1,
            save_best_only=True,
        )
        early_stop_cb = EarlyStopping(
            monitor=monitor_metric,
            patience=patience,
            mode="min",
            verbose=1,
            restore_best_weights=True,
        )

        # Kick off the long training run with the configured schedule and callbacks.
        fit_kwargs = {
            "x": train_inputs,
            "y": [train_x_vel, train_y_vel],
            "epochs": self.config.training.model_epochs,
            "batch_size": batch_size,
            "shuffle": True,
            "callbacks": [checkpoint_cb, early_stop_cb],
        }
        if val_data is not None:
            fit_kwargs["validation_data"] = val_data
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

        # Load checkpoint and compute test-set RMSEs.
        model = load_model(str(ckpt_path), custom_objects={"TCN": TCN})
        preds = model.predict(self.test_data.inputs)
        rmse_vel_x = mean_squared_error(self.test_data.x_vel, preds[0], squared=False)
        rmse_vel_y = mean_squared_error(self.test_data.y_vel, preds[1], squared=False)

        # Gather hyperparameters for record-keeping if the study is available.
        best_params = None
        if study_storage and study_name:
            study = optuna.load_study(study_name=study_name, storage=study_storage)
            best_params = study.best_trial.params

        # Optionally emit a TFLite artifact for downstream deployment.
        tflite_written = None
        if export_tflite:
            tflite_path = Path(tflite_path) if tflite_path else Path(self.config.outputs.tflite_model_path)
            convert_to_tflite_model(
                model=model,
                training_data=self.training_data.inputs,
                quantization=self.config.training.quantization,
                output_name=tflite_path,
            )
            tflite_written = str(tflite_path)

        metrics = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "checkpoint_path": str(ckpt_path),
            "study_name": study_name,
            "study_storage": study_storage,
            "rmse_vel_x": float(rmse_vel_x),
            "rmse_vel_y": float(rmse_vel_y),
            "hyperparameters": best_params,
            "tflite_path": tflite_written,
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
            Sliding window stride used during preprocessing. Defaults to `config.data.stride`.
        window_size : int | None, optional
            Sliding window length used during preprocessing. Defaults to `config.data.window_size`.
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
        stride = stride if stride is not None else self.config.data.stride
        window_size = window_size if window_size is not None else self.config.data.window_size

        model = load_model(str(ckpt_path), custom_objects={"TCN": TCN})
        preds = model.predict(self.test_data.inputs)

        # Helper to integrate velocities into XY tracks.
        samples_per_window = max((window_size - stride) / stride, 1)

        def integrate_track(vx, vy, start_x, start_y):
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
        for i, length in enumerate(self.test_data.size_of_each):
            idx_end = idx_start + length
            # Flatten in case datasets store (n, 1) vectors.
            gt_vx = self.test_data.x_vel[idx_start:idx_end].ravel()
            gt_vy = self.test_data.y_vel[idx_start:idx_end].ravel()
            pred_vx = preds[0][idx_start:idx_end].ravel()
            pred_vy = preds[1][idx_start:idx_end].ravel()
            start_x = self.test_data.x0[i]
            start_y = self.test_data.y0[i]

            gt_x, gt_y = integrate_track(gt_vx, gt_vy, start_x, start_y)
            pd_x, pd_y = integrate_track(pred_vx, pred_vy, start_x, start_y)

            # Absolute Trajectory Error (mean distance).
            errs = np.sqrt((gt_x - pd_x) ** 2 + (gt_y - pd_y) ** 2)
            ate = float(np.mean(errs))
            ate_per_traj.append(ate)

            # Relative Trajectory Error over ~60s segments (heuristic).
            window_seconds = 60
            # Windows per second: sampling_rate_hz / stride (stride is in samples).
            samples_per_sec = max(self.config.data.sampling_rate_hz / stride, 1)
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

        metrics = {
            "ate_mean": float(np.mean(ate_per_traj)),
            "ate_median": float(np.median(ate_per_traj)),
            "ate_per_traj": ate_per_traj,
            "rte_median": float(np.nanmedian(rte_per_traj)),
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
    parser.add_argument(
        "--smoke-test-multiobjective",
        action="store_true",
        help="Run the smoke test in multi-objective mode (overrides config).",
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
                multiobjective=args.smoke_test_multiobjective if args.smoke_test_multiobjective else None,
            )
            print("[MAIN] Smoke test complete.")
        else:
            print(f"[MAIN] Starting full NAS workflow with study name '{args.study_name}'...")
            client.run_scoring_nas(study_name=args.study_name, storage_uri=storage_uri)
            print("[MAIN] Full NAS workflow complete.")
    finally:
        if client is not None:
            client.close()
