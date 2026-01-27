import importlib
import logging
import os
import shutil
import sys
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

sys.path.insert(0, os.path.abspath("src"))
import hardware_utils_ex
from data_utils_ex import import_oxiod_dataset

importlib.reload(hardware_utils_ex)

from hardware_utils_ex import (
    convert_to_cpp_model,
    convert_to_tflite_model,
    HIL_MASTER_DEVICE_NOT_FOUND,
)
from nas_utils_ex import (
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
logger.setLevel(logging.INFO)

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
        )
        if self.config.device.hil:
            metrics["latency_budget_ms"] = latency_budget_ms

        print("Metric collection complete")
        return metrics

    def _sync_sketch_variant(self) -> Path:
        """Copy the requested sketch variant into the Arduino build directory."""

        variants = {
            True: "tinyodom_tcn_energy.ino",
            False: "tinyodom_tcn_no_energy.ino",
        }
        variant_name = variants.get(bool(self.config.training.energy_aware))
        variant_source = self.sketch_variants_dir / variant_name
        if not variant_source.exists():
            raise FileNotFoundError(f"Sketch variant not found: {variant_source}")

        sketch_dir = Path(self.config.outputs.tcn_dir)
        sketch_dir.mkdir(parents=True, exist_ok=True)
        sketch_target = sketch_dir / "tinyodom_tcn.ino"
        shutil.copyfile(variant_source, sketch_target)
        return sketch_target
if __name__ == "__main__":
    server = HILServer()

    server.start()

    # Energy noise scan (uncomment to run)
    # import time
    # import csv
    # import tqdm
    # runs = 20
    # cooldown_s = 20.0
    # csv_path = "hil_noise_scan.csv"
    # # csv_path.parent.mkdir(parents=True, exist_ok=True)

    # hyperparams = Dict(
    #     nb_filters=10,
    #     kernel_size=12,
    #     dilations=[1, 4, 8, 64],
    #     dropout_rate=0.0,
    #     use_skip_connections=False,
    #     norm_flag=True,
    #     batch_size=256,
    #     timesteps=server.config.data.window_size,
    #     input_dim=server.training_data.inputs.shape[2],
    # )
    # model = build_tinyodom_model(hyperparams)
    # hyperparams.flops = count_flops(model, (hyperparams.timesteps, hyperparams.input_dim))

    # logger.info("Starting noise scan: %s runs, cooldown %.1fs", runs, cooldown_s)
    # with open(csv_path, "w", newline="") as csvfile:
    #     writer = None
    #     for run_idx in tqdm(range(1, runs + 1), desc="HIL noise scan"):
    #         logger.info("Noise scan run %d/%d", run_idx, runs)
    #         metrics = server.determine_metrics(hyperparams)
    #         if writer is None:
    #             fieldnames = ["run_index", *metrics.keys()]
    #             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #             writer.writeheader()
    #         row = {"run_index": run_idx, **metrics}
    #         writer.writerow(row)
    #         csvfile.flush()
    #         if run_idx < runs:
    #             logger.info("Cooling down for %.1f seconds", cooldown_s)
    #             for _ in tqdm(range(int(cooldown_s)), desc="Cooldown", leave=False):
    #                 time.sleep(1)

    # logger.info("Noise scan complete. Metrics saved to %s", csv_path)
