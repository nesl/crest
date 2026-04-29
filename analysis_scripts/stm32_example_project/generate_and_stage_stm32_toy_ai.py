#!/usr/bin/env python3
"""
Generate and stage the STM32N6 toy AI network sources.

Default behavior rebuilds the TinyODOM ``approx_trained`` / perturbed TFLite
model using the same hyperparameters as
``analysis_scripts/hil_single_run/run_single_hil_perturbed.py``, then runs:

    stedgeai generate --c-api legacy

and stages the generated STM32 sources into the toy FSBL project.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))

from addict import Dict
from stm32_phase2_candidate import export_perturbed_candidate_tflite

DEFAULT_PROJECT_ROOT = SCRIPT_DIR / "stm32_toy_ai_project" / "FSBL"
DEFAULT_CONFIG_PATH = REPO_ROOT / "src" / "config" / "nas_config.yaml"
DEFAULT_OUTPUT_ROOT = Path("/tmp/tinyodom_stm32_toy_generate")
EXPECTED_OUTPUTS = [
    "network.c",
    "network.h",
    "network_config.h",
    "network_data.c",
    "network_data.h",
    "network_data_params.c",
    "network_data_params.h",
]
WEIGHTS_BLOB_NAME = "network_data.bin"
DEFAULT_WEIGHT_STORAGE_MODE = "embedded"
DEFAULT_WEIGHTS_FLASH_ADDRESS = "0x71000000"
DEFAULT_WEIGHTS_MEMORY_POOL = (
    REPO_ROOT
    / "analysis_scripts"
    / "stm32_example_project"
    / "nucleo_mypool.json"
)
STAGING_MANIFEST_NAME = "staging_manifest.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and stage the STM32N6 toy AI network sources."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="TinyODOM config YAML used to rebuild the perturbed model.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Optional prebuilt .tflite path. When omitted, rebuild the perturbed TinyODOM model.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="Path to the toy FSBL project root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Temporary generation root used for TinyODOM and ST Edge AI outputs.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the temporary output root before generation.",
    )
    parser.add_argument(
        "--weight-storage-mode",
        choices=["embedded", "external_flash"],
        default=DEFAULT_WEIGHT_STORAGE_MODE,
        help="Where generated weights should live for the toy STM32 flow.",
    )
    parser.add_argument(
        "--weights-flash-address",
        default=DEFAULT_WEIGHTS_FLASH_ADDRESS,
        help="Absolute external flash address used in external_flash mode.",
    )
    parser.add_argument(
        "--weights-memory-pool",
        type=Path,
        default=DEFAULT_WEIGHTS_MEMORY_POOL,
        help="Memory-pool JSON used in external_flash mode.",
    )
    parser.add_argument(
        "--weights-external-loader",
        type=Path,
        default=None,
        help="Optional STM32CubeProgrammer external loader override recorded in the manifest for handoff/debugging.",
    )
    return parser


def _export_perturbed_tflite(config_path: Path, output_root: Path) -> tuple[Path, Dict]:
    """Rebuild the perturbed TinyODOM TFLite artifact.

    Parameters
    ----------
    config_path : Path
        Path to the TinyODOM YAML configuration file.
    output_root : Path
        Temporary root used for exported model artifacts.

    Returns
    -------
    tuple[Path, Dict]
        Path to the generated TFLite file and metadata describing the
        perturbed model variant and resolved hyperparameters.
    """
    return export_perturbed_candidate_tflite(config_path, output_root)


def _run_generate(
    model_path: Path,
    output_root: Path,
    *,
    weight_storage_mode: str,
    weights_flash_address: str,
    weights_memory_pool: Path,
) -> Path:
    """Run ST Edge AI code generation for one TFLite model.

    Parameters
    ----------
    model_path : Path
        Path to the input TFLite model.
    output_root : Path
        Root directory used for the ST Edge AI workspace and outputs.

    Returns
    -------
    Path
        Directory containing the generated STM32 network sources.
    """
    out_dir = output_root / "out"
    ws_dir = output_root / "ws"
    cmd = [
        "stedgeai",
        "generate",
        "-m",
        str(model_path),
        "-t",
        "tflite",
        "--target",
        "stm32n6",
        "--c-api",
        "legacy",
    ]
    if weight_storage_mode == "external_flash":
        cmd.extend(
            [
                "--binary",
                "--address",
                weights_flash_address,
                "--memory-pool",
                str(weights_memory_pool),
            ]
        )
    cmd.extend(
        [
            "--quiet",
            "--workspace",
            str(ws_dir),
            "--output",
            str(out_dir),
        ]
    )
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    return out_dir


def _verify_outputs(out_dir: Path, *, weight_storage_mode: str) -> Path | None:
    """Validate that ST Edge AI emitted the expected network files.

    Parameters
    ----------
    out_dir : Path
        Generation output directory to inspect.

    Returns
    -------
    None

    Returns
    -------
    Path | None
        Path to the weights blob in ``external_flash`` mode, otherwise ``None``.
    """
    missing = [name for name in EXPECTED_OUTPUTS if not (out_dir / name).is_file()]
    if missing:
        missing_text = ", ".join(missing)
        raise RuntimeError(f"Missing generated outputs: {missing_text}")
    if weight_storage_mode != "external_flash":
        return None
    weights_blob_path = out_dir / WEIGHTS_BLOB_NAME
    if not weights_blob_path.is_file():
        raise RuntimeError(f"Missing generated weights blob: {weights_blob_path}")
    return weights_blob_path


def _write_manifest(
    output_root: Path,
    *,
    weight_storage_mode: str,
    generated_output_dir: Path,
    weights_blob_path: Path | None,
    weights_flash_address: str | None,
    weights_external_loader: Path | None,
) -> Path:
    """Persist the fixed-path staging manifest for downstream tooling."""

    manifest = {
        "weight_storage_mode": weight_storage_mode,
        "generated_output_dir": str(generated_output_dir),
        "weights_blob_path": str(weights_blob_path) if weights_blob_path is not None else None,
        "weights_blob_size": weights_blob_path.stat().st_size if weights_blob_path is not None else None,
        "weights_flash_address": weights_flash_address,
        "weights_external_loader": (
            str(weights_external_loader.resolve()) if weights_external_loader is not None else None
        ),
    }
    manifest_path = output_root / STAGING_MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def _stage_outputs(out_dir: Path, project_root: Path) -> list[Path]:
    """Copy generated STM32 network sources into the toy FSBL project.

    Parameters
    ----------
    out_dir : Path
        Directory containing generated ST Edge AI source/header files.
    project_root : Path
        STM32 FSBL project root that receives the staged files.

    Returns
    -------
    list[Path]
        Destination paths written under the STM32 project.
    """
    staged: list[Path] = []
    src_dir = project_root / "Src"
    inc_dir = project_root / "Inc"
    src_dir.mkdir(parents=True, exist_ok=True)
    inc_dir.mkdir(parents=True, exist_ok=True)

    for name in EXPECTED_OUTPUTS:
        destination_dir = src_dir if name.endswith(".c") else inc_dir
        destination = destination_dir / name
        shutil.copy2(out_dir / name, destination)
        staged.append(destination)
    return staged


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    config_path = args.config.resolve()
    model_override = args.model.resolve() if args.model is not None else None
    project_root = args.project_root.resolve()
    output_root = args.output_root.resolve()
    weights_memory_pool = args.weights_memory_pool.resolve()
    weights_external_loader = (
        args.weights_external_loader.resolve() if args.weights_external_loader is not None else None
    )

    if not config_path.is_file():
        parser.error(f"Config not found: {config_path}")
    if model_override is not None and not model_override.is_file():
        parser.error(f"Model not found: {model_override}")
    if not project_root.is_dir():
        parser.error(f"Project root not found: {project_root}")
    if args.weight_storage_mode == "external_flash" and not weights_memory_pool.is_file():
        parser.error(f"Weights memory-pool JSON not found: {weights_memory_pool}")
    if weights_external_loader is not None and not weights_external_loader.is_file():
        parser.error(f"Weights external loader not found: {weights_external_loader}")

    if args.clean and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    metadata = Dict()
    if model_override is not None:
        model_path = model_override
        metadata.model_variant = "external_tflite"
        metadata.input_mode = "unknown"
        metadata.energy_aware = None
        metadata.quantization = None
    else:
        model_path, metadata = _export_perturbed_tflite(config_path, output_root)

    out_dir = _run_generate(
        model_path,
        output_root,
        weight_storage_mode=args.weight_storage_mode,
        weights_flash_address=args.weights_flash_address,
        weights_memory_pool=weights_memory_pool,
    )
    weights_blob_path = _verify_outputs(out_dir, weight_storage_mode=args.weight_storage_mode)
    staged = _stage_outputs(out_dir, project_root)
    manifest_path = _write_manifest(
        output_root,
        weight_storage_mode=args.weight_storage_mode,
        generated_output_dir=out_dir,
        weights_blob_path=weights_blob_path,
        weights_flash_address=(
            args.weights_flash_address if args.weight_storage_mode == "external_flash" else None
        ),
        weights_external_loader=weights_external_loader,
    )

    print("STM32 toy AI staging complete.")
    print(f"Config: {config_path}")
    print(f"Model: {model_path}")
    print(f"Model variant: {metadata.model_variant}")
    print(f"Weight storage mode: {args.weight_storage_mode}")
    print(f"Workspace: {output_root / 'ws'}")
    print(f"Outputs: {out_dir}")
    print(f"Manifest: {manifest_path}")
    if weights_blob_path is not None:
        print(f"Weights blob: {weights_blob_path} ({weights_blob_path.stat().st_size} bytes)")
    print("Staged files:")
    for path in staged:
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
