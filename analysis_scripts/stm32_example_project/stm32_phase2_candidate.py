from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

from addict import Dict

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tinyodom.analysis_support import (
    build_fixed_tinyodom_hyperparams,
    require_calibration_inputs,
)
from tinyodom.builtin_components import ensure_builtin_components_registered
from tinyodom.model import configured_quantization_mode, load_config, quantization_requires_calibration
from tinyodom.runtime_bootstrap import bootstrap_pipeline

DEFAULT_CONFIG_PATH = REPO_ROOT / "src" / "config" / "nas_config_stm32.yaml"
PERTURBED_VARIANT_NAME = "approx_trained"


@dataclass(frozen=True)
class Phase2CandidateBundle:
    """Shared fixed candidate bundle used by STM Phase 2 scripts."""

    config: Dict
    calibration_inputs: Any | None
    hyperparams: Dict
    model: Any
    metadata: Dict
    window_size: int
    input_dim: int


def load_or_build_perturbed_candidate(config_path: Path) -> Phase2CandidateBundle:
    """Build the fixed perturbed TinyODOM candidate used by STM Phase 2 flows."""
    ensure_builtin_components_registered()
    config = load_config(config_path)
    pipeline = bootstrap_pipeline(config)
    quantization_mode = configured_quantization_mode(config)
    calibration_split = pipeline.dataset.make_calibration_data(
        pipeline.bundle,
        pipeline.selection["dataset_config"],
    )
    calibration_inputs = (
        require_calibration_inputs(None if calibration_split is None else calibration_split.inputs)
        if quantization_requires_calibration(quantization_mode)
        else None
    )
    window_size = int(pipeline.model_build_context.input_shape[0])
    input_dim = int(pipeline.model_build_context.input_shape[1])
    hyperparams = build_fixed_tinyodom_hyperparams(
        window_size=window_size,
        input_dim=input_dim,
    )
    model = pipeline.model_family.materialize_export_model(
        dict(hyperparams),
        pipeline.model_build_context,
        pipeline.selection["model_config"],
        model_variant=PERTURBED_VARIANT_NAME,
    )
    pipeline.task.compile_model(model, pipeline.selection["task_config"], pipeline.target_spec)

    metadata = Dict(
        model_variant=PERTURBED_VARIANT_NAME,
        input_mode="uniform",
        energy_aware=True,
        quantization_mode=quantization_mode,
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
        calibration_inputs=calibration_inputs,
        hyperparams=hyperparams,
        model=model,
        metadata=metadata,
        window_size=window_size,
        input_dim=input_dim,
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
        training_data=bundle.calibration_inputs,
        quantization_mode=configured_quantization_mode(bundle.config),
        output_name=tflite_path,
    )
    return tflite_path, bundle.metadata
