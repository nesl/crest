#!/usr/bin/env python3
"""Generate and stage STM32N6 LRUN toy AI sources into the LRUN App project."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from addict import Dict

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))

from stm32_phase2_candidate import export_perturbed_candidate_tflite
from stm32_lrun_common import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PROJECT_ROOT,
    DEFAULT_WEIGHTS_FLASH_ADDRESS,
    DEFAULT_WEIGHTS_MEMORY_POOL,
    EXPECTED_OUTPUTS,
    resolve_stedgeai_cli,
    stage_generated_outputs,
    verify_generate_outputs,
    write_staging_manifest,
)
from tinyodom.microcontrollers import stm32_cube_clt


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and stage STM32N6 toy AI sources into the LRUN App project."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument(
        "--weight-storage-mode",
        choices=["embedded", "external_flash"],
        default="external_flash",
    )
    parser.add_argument("--weights-flash-address", default=DEFAULT_WEIGHTS_FLASH_ADDRESS)
    parser.add_argument("--weights-memory-pool", type=Path, default=DEFAULT_WEIGHTS_MEMORY_POOL)
    parser.add_argument("--weights-external-loader", type=Path, default=None)
    return parser


def _run_generate(
    model_path: Path,
    output_root: Path,
    *,
    weight_storage_mode: str,
    weights_flash_address: str,
    weights_memory_pool: Path,
) -> Path:
    out_dir = output_root / "out"
    ws_dir = output_root / "ws"
    cmd = [
        resolve_stedgeai_cli(),
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
    cmd.extend(["--quiet", "--workspace", str(ws_dir), "--output", str(out_dir)])
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise stm32_cube_clt.WorkflowError(
            f"Failed to execute ST Edge AI generation: {' '.join(cmd)}\n\n{exc}"
        ) from exc
    if proc.returncode != 0:
        raise stm32_cube_clt.WorkflowError(
            f"ST Edge AI generation failed with exit code {proc.returncode}: {' '.join(cmd)}\n\n"
            f"{(proc.stdout or '')}{(proc.stderr or '')}".strip()
        )
    return out_dir


def _export_perturbed_tflite(config_path: Path, output_root: Path) -> tuple[Path, Dict]:
    return export_perturbed_candidate_tflite(config_path, output_root)


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
    else:
        model_path, metadata = _export_perturbed_tflite(config_path, output_root)

    out_dir = _run_generate(
        model_path,
        output_root,
        weight_storage_mode=args.weight_storage_mode,
        weights_flash_address=args.weights_flash_address,
        weights_memory_pool=weights_memory_pool,
    )
    weights_blob_path = verify_generate_outputs(out_dir, weight_storage_mode=args.weight_storage_mode)
    staged = stage_generated_outputs(out_dir, project_root)
    manifest_path = write_staging_manifest(
        output_root,
        weight_storage_mode=args.weight_storage_mode,
        generated_output_dir=out_dir,
        weights_blob_path=weights_blob_path,
        weights_flash_address=(
            args.weights_flash_address if args.weight_storage_mode == "external_flash" else None
        ),
        weights_external_loader=weights_external_loader,
    )

    print("STM32 LRUN toy AI staging complete.")
    print(f"Config: {config_path}")
    print(f"Model: {model_path}")
    print(f"Model variant: {metadata.get('model_variant', 'unknown')}")
    print(f"Weight storage mode: {args.weight_storage_mode}")
    print(f"Workspace: {output_root / 'ws'}")
    print(f"Outputs: {out_dir}")
    print(f"Manifest: {manifest_path}")
    if weights_blob_path is not None:
        print(f"Weights blob: {weights_blob_path} ({weights_blob_path.stat().st_size} bytes)")
    print("Expected generated files:")
    for name in EXPECTED_OUTPUTS:
        print(f"  - {out_dir / name}")
    print("Staged files:")
    for path in staged:
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
