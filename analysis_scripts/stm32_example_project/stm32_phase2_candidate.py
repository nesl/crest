from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

from addict import Dict


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_CONFIG_PATH = REPO_ROOT / "src" / "nas_config.yaml"
PERTURBED_VARIANT_NAME = "approx_trained"
OXIOD_SUB_FOLDERS = [
    "handbag/",
    "handheld/",
    "pocket/",
    "running/",
    "slow_walking/",
    "trolley/",
]


@dataclass(frozen=True)
class Phase2CandidateBundle:
    """Shared fixed candidate bundle used by STM Phase 2 scripts."""

    config: Dict
    training_data: Any
    hyperparams: Dict
    model: Any
    metadata: Dict


def build_fixed_hyperparams(*, window_size: int, input_dim: int) -> Dict:
    """Build the fixed perturbed-model hyperparameter bundle."""
    from tinyodom.model import build_tinyodom_model, count_flops

    hyperparams = Dict(
        nb_filters=10,
        kernel_size=12,
        dilations=[1, 4, 8, 64],
        dropout_rate=0.0,
        use_skip_connections=False,
        norm_flag=True,
        batch_size=256,
        timesteps=window_size,
        input_dim=input_dim,
    )
    model = build_tinyodom_model(hyperparams)
    hyperparams.flops = count_flops(model, (hyperparams.timesteps, hyperparams.input_dim))
    return hyperparams


def load_training_data(config: Dict):
    """Load the calibration/training windows used for STM candidate export."""
    from tinyodom.data import import_oxiod_dataset

    return import_oxiod_dataset(
        type_flag=2,
        useMagnetometer=True,
        useStepCounter=True,
        AugmentationCopies=0,
        dataset_folder=config.data.directory,
        sub_folders=OXIOD_SUB_FOLDERS,
        sampling_rate=config.data.sampling_rate_hz,
        window_size=config.data.window_size,
        stride=config.data.stride,
        verbose=False,
        max_windows=config.data.calibration_windows,
    )


def load_or_build_perturbed_candidate(config_path: Path) -> Phase2CandidateBundle:
    """Build the fixed perturbed TinyODOM candidate used by STM Phase 2 flows."""
    from tensorflow.keras import optimizers

    from tinyodom.model import (
        apply_combined_perturbation,
        build_tinyodom_model,
        load_config,
    )

    config = load_config(config_path)
    training_data = load_training_data(config)
    hyperparams = build_fixed_hyperparams(
        window_size=int(config.data.window_size),
        input_dim=int(training_data.inputs.shape[2]),
    )
    model = build_tinyodom_model(hyperparams)
    bn_touched, bias_touched = apply_combined_perturbation(model=model, seed=1337)
    model.compile(loss={"velx": "mse", "vely": "mse"}, optimizer=optimizers.Adam())

    metadata = Dict(
        model_variant=PERTURBED_VARIANT_NAME,
        input_mode="uniform",
        energy_aware=True,
        quantization=bool(config.training.quantization),
        bn_layers_touched=int(bn_touched),
        non_bn_bias_layers_touched=int(bias_touched),
        hyperparams={
            "nb_filters": int(hyperparams.nb_filters),
            "kernel_size": int(hyperparams.kernel_size),
            "dilations": [int(value) for value in hyperparams.dilations],
            "dropout_rate": float(hyperparams.dropout_rate),
            "use_skip_connections": bool(hyperparams.use_skip_connections),
            "norm_flag": bool(hyperparams.norm_flag),
            "batch_size": int(hyperparams.batch_size),
            "timesteps": int(hyperparams.timesteps),
            "input_dim": int(hyperparams.input_dim),
            "flops": float(hyperparams.flops),
        },
    )
    return Phase2CandidateBundle(
        config=config,
        training_data=training_data,
        hyperparams=hyperparams,
        model=model,
        metadata=metadata,
    )


def export_perturbed_candidate_tflite(config_path: Path, output_root: Path) -> tuple[Path, Dict]:
    """Build the shared perturbed candidate and export it as TFLite."""
    from tinyodom.hardware import convert_to_tflite_model

    bundle = load_or_build_perturbed_candidate(config_path)
    model_dir = output_root / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    device_name = str(bundle.config.device.name).strip().upper()
    tflite_path = model_dir / f"TinyOdomEx_OxIOD_{device_name}_{PERTURBED_VARIANT_NAME}.tflite"
    convert_to_tflite_model(
        model=bundle.model,
        training_data=bundle.training_data.inputs,
        quantization=bool(bundle.config.training.quantization),
        output_name=tflite_path,
    )
    return tflite_path, bundle.metadata
