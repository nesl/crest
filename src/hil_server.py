import argparse
import logging
import shutil
from pathlib import Path

import absl.logging
# import optuna
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
from tinyodom.hardware import (
    convert_to_cpp_model,
    convert_to_tflite_model,
    HIL_MASTER_DEVICE_NOT_FOUND,
)
from tinyodom.model import (
    DEFAULT_CONFIG_PATH,
    apply_combined_perturbation,
    build_collect_metrics_request,
    build_tinyodom_model,
    collect_metrics,
    load_config,
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
        self.active_sketch_path = self._sync_sketch_variant()
        logger.info("Using sketch variant: %s", self.active_sketch_path)

        if self.config.device.hil is False:
            logger.warning("HIL is disabled in the configuration.")

        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.REP)

        calibration_windows = self.config.data.calibration_windows
        self.training_data = import_oxiod_dataset(type_flag=2, 
                                            useMagnetometer=True, 
                                            useStepCounter=True, 
                                            AugmentationCopies=0,
                                            dataset_folder=self.config.data.directory,
                                            sub_folders=['handbag/','handheld/','pocket/','running/','slow_walking/','trolley/'],
                                            sampling_rate=self.config.data.sampling_rate_hz, 
                                            window_size=self.config.data.window_size, 
                                            stride=self.config.data.stride, 
                                            verbose=False,
                                            max_windows=calibration_windows)
        print("Imported Training Data")

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
        variant = str(model_variant).strip().lower()
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

        # Convert the model to a TFLite format for deployment on the target device and save to OUTPUT_PATH
        convert_to_tflite_model(
            model=model,
            training_data=self.training_data.inputs,
            quantization=self.config.training.quantization,
            output_name=str(self.config.outputs.tflite_model_path),
        )

        print("Converted to TFLite")

        # Converts the TFLite model to C code for deployment on the target device
        convert_to_cpp_model(tflite_path=self.config.outputs.tflite_model_path, output_dir=self.config.outputs.tcn_dir)

        print("Converted to C++")

        print("Starting metric collection")

        # Compute the latency budget once so the downstream metrics/logging logic can reuse it.
        latency_budget_ms = (self.config.data.stride / self.config.data.sampling_rate_hz) * 1000

        request_metrics_args = build_collect_metrics_request(
            config=self.config,
            hyperparams=hyperparams,
            # Stride=20 at 100 Hz emits an inference roughly every 0.2s, so normalize
            # latency by the stride cadence rather than the full window length.
            latency_budget_ms=latency_budget_ms,
        )
        metrics = collect_metrics(request_metrics_args)
        
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
        def _normalize_target_core(raw_value: object) -> str:
            """Normalize and validate a Portenta target-core token.

            Parameters
            ----------
            raw_value : object
                Raw target-core setting from config.

            Returns
            -------
            str
                Lower-cased target core token (``cm7`` or ``cm4``).

            Raises
            ------
            ValueError
                If the provided value is missing or unsupported.
            """
            normalized = str(raw_value).strip().lower() if raw_value is not None else ""
            if normalized not in {"cm7", "cm4"}:
                raise ValueError(
                    "Set device.portenta.target_core to 'cm7' or 'cm4' when device.name is PORTENTA_H7."
                )
            return normalized

        def _resolve_uniform_variant_dir() -> Path:
            """Resolve the folder containing shared uniform sketches.

            Returns
            -------
            Path
                Directory containing ``tinyodom_tcn_energy.ino`` and
                ``tinyodom_tcn_no_energy.ino`` shared across device profiles.
            """
            device_name = str(getattr(self.config.device, "name", "")).strip().upper()
            if device_name == "PORTENTA_H7":
                portenta_cfg = getattr(self.config.device, "portenta", None)
                _normalize_target_core(
                    getattr(portenta_cfg, "target_core", None) if portenta_cfg is not None else None
                )
                # Validation only: uniform sketch source is shared.
            # Uniform variants are shared for all boards.
            return self.sketch_variants_dir

        if not bool(self.config.training.energy_aware):
            variant_dir = _resolve_uniform_variant_dir()
            variant_name = "tinyodom_tcn_no_energy.ino"
        else:
            input_mode = str(getattr(self.config.training, "input_mode", "uniform")).lower()
            variants = {
                "uniform": ("tinyodom_tcn_energy.ino", _resolve_uniform_variant_dir()),
                "representative": (
                    "tinyodom_tcn_energy_representative.ino",
                    self.sketch_variants_dir / "analysis_sketches",
                ),
                "real": ("tinyodom_tcn_energy_real_data.ino", self.sketch_variants_dir / "analysis_sketches"),
            }
            if input_mode not in variants:
                allowed = ", ".join(sorted(variants))
                raise ValueError(
                    f"Unsupported input_mode '{input_mode}'. Expected one of: {allowed}."
                )
            variant_name, variant_dir = variants[input_mode]
        variant_source = variant_dir / variant_name
        if not variant_source.exists():
            raise FileNotFoundError(f"Sketch variant not found: {variant_source}")

        sketch_dir = Path(self.config.outputs.tcn_dir)
        sketch_dir.mkdir(parents=True, exist_ok=True)
        sketch_target = sketch_dir / "tinyodom_tcn.ino"
        shutil.copyfile(variant_source, sketch_target)

        common_source = self.sketch_variants_dir / "common"
        if common_source.exists():
            shutil.copytree(common_source, sketch_dir / "common", dirs_exist_ok=True)
        # Allow board-specific overrides while keeping shared-copy behavior.
        variant_common_source = variant_dir / "common"
        if variant_common_source.exists() and variant_common_source != common_source:
            shutil.copytree(variant_common_source, sketch_dir / "common", dirs_exist_ok=True)

        needs_header = variant_name in {
            "tinyodom_tcn_energy_representative.ino",
            "tinyodom_tcn_energy_real_data.ino",
        }
        header_source = self.sketch_variants_dir / "analysis_sketches" / "tinyodom_tcn_input_data.h"
        if needs_header:
            if not header_source.exists():
                raise FileNotFoundError(f"Input header not found: {header_source}")
            shutil.copyfile(header_source, sketch_dir / header_source.name)
        return sketch_target

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
        self.config.training.input_mode = str(input_mode).lower()
        self.active_sketch_path = self._sync_sketch_variant()
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
