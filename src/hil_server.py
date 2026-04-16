import argparse
import logging
import shutil
from dataclasses import replace
from pathlib import Path

import absl.logging
# import optuna
import serial
import tensorflow as tf
# import tensorflow_model_optimization as tfmot
import zmq
from addict import Dict
# from sklearn.metrics import mean_squared_error  # , root_mean_squared_error
from tcn import TCN
from tensorflow.keras import optimizers
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from tensorflow.keras.layers import Dense, Flatten, MaxPooling1D, Reshape
from tensorflow.keras.models import load_model

from tinyodom.data import import_oxiod_dataset
from tinyodom.devices import _sync_arduino_sketch_variant_for_config
from tinyodom.hardware import (
    HIL_MASTER_DEVICE_NOT_FOUND,
)
from tinyodom.errors import HIL_MASTER_FATAL
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
    apply_combined_perturbation,
    build_collect_metrics_request,
    build_tinyodom_model,
    collect_metrics,
    load_config,
    set_error_code,
    validate_loaded_model_input_shape,
)

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
        "harness_latency_ms": -1.0,
        "weight_storage_mode": "embedded",
        "backend_error_kind": str(error_kind),
        "backend_error_detail": str(error_detail),
    }
    set_error_code(metrics, HIL_MASTER_FATAL)
    return metrics

class HILServer:
    def __init__(
        self,
        config_path: Path = DEFAULT_CONFIG_PATH,
        config: Dict | None = None,
    ) -> None:
        """
        Initialize the HIL server state and preload calibration data.

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
        self.config = config if config is not None else load_config(config_path)

        # Resolve repository root once so sketch variants can be copied before each compile.
        self.repo_root = Path(__file__).resolve().parent.parent
        self.sketch_variants_dir = self.repo_root / "sketches"
        self.active_sketch_path: Path | None = None
        self.training_data = None

        if self.config.device.hil is False:
            logger.warning("HIL is disabled in the configuration.")

        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.REP)

    def _normalized_device_name(self) -> str:
        """Return the normalized configured device name.

        Returns
        -------
        str
            Upper-cased device name from the loaded configuration.
        """
        return str(getattr(self.config.device, "name", "")).strip().upper()

    def _ensure_training_data(self):
        """Load training data lazily for backends that still need model export.

        Returns
        -------
        Any
            Cached OxIOD training/calibration dataset.
        """
        if self.training_data is not None:
            return self.training_data
        calibration_windows = self.config.data.calibration_windows
        self.training_data = import_oxiod_dataset(
            type_flag=2,
            useMagnetometer=True,
            useStepCounter=True,
            AugmentationCopies=0,
            dataset_folder=self.config.data.directory,
            sub_folders=['handbag/', 'handheld/', 'pocket/', 'running/', 'slow_walking/', 'trolley/'],
            sampling_rate=self.config.data.sampling_rate_hz,
            window_size=self.config.data.window_size,
            stride=self.config.data.stride,
            verbose=False,
            max_windows=calibration_windows,
        )
        print("Imported Training Data")
        return self.training_data

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
                hyperparams = self.socket.recv_json()
                print(f"[HIL REP] Received hyperparameters: {hyperparams}")

                metrics = self.determine_metrics(Dict(hyperparams))

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

    def determine_metrics(
        self,
        hyperparams: Dict,
        checkpoint_path: Path | str | None = None,
        model_variant: str = APPROX_TRAINED_VARIANT_NAME,
    ) -> dict:
        """
        Build/select a model variant, compile/export it, and collect HIL metrics.

        Parameters
        ----------
        hyperparams : Dict
            Hyperparameter bundle used to build the model and to drive hardware
            metric collection (for example ``flops`` and ``input_dim``).
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
        latency_budget_ms = (self.config.data.stride / self.config.data.sampling_rate_hz) * 1000
        device_options = resolve_device_options(self._normalized_device_name(), self.config.device)
        runtime_device = get_microcontroller_device(
            self._normalized_device_name(),
            serial_port=getattr(self.config.device, "serial_port", None),
            device_options=device_options,
        )

        variant = str(model_variant).strip().lower()
        model = None
        training_data = None
        if runtime_device.requires_candidate_model():
            is_approx_trained = variant in {
                APPROX_TRAINED_VARIANT_NAME,
                REPRESENTATIVE_VARIANT_LEGACY_NAME,
                PERTURBED_VARIANT_LEGACY_NAME,
            }
            if variant == "untrained":
                model = build_tinyodom_model(hyperparams)
                print("Model created from untrained architecture")
            elif is_approx_trained:
                model = build_tinyodom_model(hyperparams)
                bn_touched, bias_touched = apply_combined_perturbation(model=model, seed=1337)
                print(
                    "Model created from approx_trained architecture variant (combined perturbation) "
                    f"(bn_layers={bn_touched}, non_bn_bias_layers={bias_touched})"
                )
            elif variant.startswith("trained"):
                if checkpoint_path is None:
                    raise ValueError(
                        f"model_variant '{model_variant}' requires checkpoint_path to be provided."
                    )
                ckpt_path = Path(checkpoint_path)
                if not ckpt_path.exists():
                    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
                model = load_model(str(ckpt_path), custom_objects={"TCN": TCN})
                validate_loaded_model_input_shape(model, hyperparams)
                print(f"Model loaded from checkpoint: {ckpt_path}")
            else:
                raise ValueError(
                    f"Unsupported model_variant '{model_variant}'. "
                    f"Use '{APPROX_TRAINED_VARIANT_NAME}', 'untrained', "
                    "or a variant that starts with 'trained'. "
                    f"(Legacy aliases '{REPRESENTATIVE_VARIANT_LEGACY_NAME}' and "
                    f"'{PERTURBED_VARIANT_LEGACY_NAME}' are also accepted.)"
                )

            optimizer = optimizers.Adam()
            model.compile(loss={"velx": "mse", "vely": "mse"}, optimizer=optimizer)

            if runtime_device.requires_training_data():
                training_data = self._ensure_training_data()

        effective_hil_enabled = bool(
            self.config.device.hil and runtime_device.supports_runtime_measurement()
        )
        effective_energy_aware = bool(
            self.config.training.energy_aware
            and effective_hil_enabled
            and runtime_device.supports_energy_measurement()
        )
        try:
            prepared_dir = runtime_device.prepare_candidate(
                config=self.config,
                hyperparams=hyperparams,
                model=model,
                outputs_dir=Path(self.config.outputs.tcn_dir),
                tflite_model_path=Path(self.config.outputs.tflite_model_path),
                training_data=training_data,
                model_variant=model_variant,
                checkpoint_path=checkpoint_path,
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
                    hyperparams=hyperparams,
                    latency_budget_ms=latency_budget_ms,
                    dirpath=prepared_dir,
                    device_options=device_options,
                    hil_enabled=effective_hil_enabled,
                    energy_aware=effective_energy_aware,
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

        if self.config.device.hil:
            metrics["latency_budget_ms"] = latency_budget_ms

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

    def set_input_mode(self, input_mode: str) -> Path:
        """
        Set the input mode and resynchronize the active Arduino sketch variant.

        Parameters
        ----------
        input_mode : str
            Desired input mode (``"uniform"``, ``"representative"``, or
            ``"real"``).

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
