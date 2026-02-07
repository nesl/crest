import argparse
import logging
import shutil
from pathlib import Path

import absl.logging
# import numpy as np
# import optuna
import tensorflow as tf
# import tensorflow_model_optimization as tfmot
import zmq
from addict import Dict
# from sklearn.metrics import mean_squared_error  # , root_mean_squared_error
# from tcn import TCN
from tensorflow.keras import optimizers
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from tensorflow.keras.layers import Dense, Flatten, MaxPooling1D, Reshape
# from tensorflow.keras.models import load_model

from tinyodom.data import import_oxiod_dataset
from tinyodom.hardware import (
    convert_to_cpp_model,
    convert_to_tflite_model,
    HIL_MASTER_DEVICE_NOT_FOUND,
)
from tinyodom.model import (
    DEFAULT_CONFIG_PATH,
    build_tinyodom_model,
    collect_metrics,
    load_config,
    count_flops
)

tf.get_logger().setLevel(logging.ERROR)
absl.logging.set_verbosity(absl.logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

logger = logging.getLogger(__name__)


def _configure_logging(level_name: str) -> None:
    """
    Configure root logging for the HIL server process.

    Parameters
    ----------
    level_name : str
        Logging level name (e.g., INFO, DEBUG).
    """
    level_value = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level_value,
        format="%(levelname)s:%(name)s:%(message)s",
    )

class HILServer:
    def __init__(self, config_path: Path=DEFAULT_CONFIG_PATH) -> None:
        self.config = load_config(config_path)

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

    def determine_metrics(self, hyperparams: Dict) -> dict:
        model = build_tinyodom_model(hyperparams)
        optimizer = optimizers.Adam()
        model.compile(loss={"velx": "mse", "vely": "mse"}, optimizer=optimizer)

        print("Model created")

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

        # Collect RAM/flash/latency/arena metrics from the controller
        metrics = collect_metrics(
            hil_enabled=self.config.device.hil,
            flops=hyperparams.flops,
            device_name=self.config.device.name,
            window_size=self.config.data.window_size,
            input_dim=hyperparams.input_dim,
            dirpath=self.config.outputs.tcn_dir,
            latency_proxy_max_flops=self.config.training.latency_proxy_max_flops,
            serial_port=self.config.device.serial_port,
            # Stride=20 at 100 Hz emits an inference roughly every 0.2s, so normalize
            # latency by the stride cadence rather than the full window length.
            latency_budget_ms=latency_budget_ms,
            dut_ready_timeout_s=getattr(self.config.device, "dut_ready_timeout_s", 5.0),
            harness_serial_port=(
                self.config.device.harness_serial_port
                if self.config.training.energy_aware
                else None
            ),
            harness_fqbn=(
                self.config.device.harness_fqbn
                if self.config.training.energy_aware
                else None
            ),
            harness_auto_flash=(
                self.config.device.harness_auto_flash
                if self.config.training.energy_aware
                else None
            ),
            harness_arm_pin=(
                self.config.device.harness_arm_pin
                if self.config.training.energy_aware
                else None
            ),
            harness_trigger_pin=(
                self.config.device.harness_trigger_pin
                if self.config.training.energy_aware
                else None
            ),
            dut_arm_hold_ms=(
                self.config.device.dut_arm_hold_ms
                if self.config.training.energy_aware
                else None
            ),
            harness_stable_low_ms=(
                self.config.device.harness_stable_low_ms
                if self.config.training.energy_aware
                else None
            ),
            harness_ready_timeout_s=(
                self.config.device.harness_ready_timeout_s
                if self.config.training.energy_aware
                else None
            ),
            harness_arm_timeout_s=(
                self.config.device.harness_arm_timeout_s
                if self.config.training.energy_aware
                else None
            ),
            harness_active_timeout_s=(
                self.config.device.harness_active_timeout_s
                if self.config.training.energy_aware
                else None
            ),
            harness_done_timeout_s=(
                self.config.device.harness_done_timeout_s
                if self.config.training.energy_aware
                else None
            ),
        )
        if self.config.device.hil:
            metrics["latency_budget_ms"] = latency_budget_ms

        print("Metric collection complete")
        return metrics

    def _sync_sketch_variant(self) -> Path:
        """Copy the requested sketch variant into the Arduino build directory."""

        if not bool(self.config.training.energy_aware):
            variant_dir = self.sketch_variants_dir
            variant_name = "tinyodom_tcn_no_energy.ino"
        else:
            input_mode = str(getattr(self.config.training, "input_mode", "standard")).lower()
            variants = {
                "standard": ("tinyodom_tcn_energy.ino", self.sketch_variants_dir),
                "uniform": ("tinyodom_tcn_energy_uniform.ino", self.sketch_variants_dir / "analysis_sketches"),
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
        """Update the sketch input mode and resync the Arduino sketch variant."""
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
    preview_config = load_config(cfg_path)
    _configure_logging(preview_config.logging.level)

    server = HILServer(config_path=cfg_path)
    server.start()
