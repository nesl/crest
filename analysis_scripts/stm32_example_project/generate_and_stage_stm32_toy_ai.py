#!/usr/bin/env python3
"""
Generate and stage the STM32N6 toy AI network sources.

This script runs the Phase 0 safe generation path:

    stedgeai generate --c-api legacy

It then stages the generated network sources directly into the copied toy
project so the existing committed make metadata can build them without a
CubeIDE regeneration step.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPO_ROOT / "models" / "TinyOdomEx_OxIOD_PORTENTA_H7.tflite"
DEFAULT_PROJECT_ROOT = (
    REPO_ROOT
    / "analysis_scripts"
    / "stm32_example_project"
    / "stm32_toy_ai_project"
    / "FSBL"
)
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and stage the STM32N6 toy AI network sources."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to the representative TinyODOM .tflite model.",
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
        help="Temporary generation root used for ST Edge AI outputs.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the temporary output root before generation.",
    )
    return parser


def _run_generate(model_path: Path, output_root: Path) -> Path:
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
        "--quiet",
        "--workspace",
        str(ws_dir),
        "--output",
        str(out_dir),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    return out_dir


def _verify_outputs(out_dir: Path) -> None:
    missing = [name for name in EXPECTED_OUTPUTS if not (out_dir / name).is_file()]
    if missing:
        missing_text = ", ".join(missing)
        raise RuntimeError(f"Missing generated outputs: {missing_text}")


def _stage_outputs(out_dir: Path, project_root: Path) -> list[Path]:
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

    model_path = args.model.resolve()
    project_root = args.project_root.resolve()
    output_root = args.output_root.resolve()

    if not model_path.is_file():
        parser.error(f"Model not found: {model_path}")
    if not project_root.is_dir():
        parser.error(f"Project root not found: {project_root}")

    if args.clean and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    out_dir = _run_generate(model_path, output_root)
    _verify_outputs(out_dir)
    staged = _stage_outputs(out_dir, project_root)

    print("STM32 toy AI staging complete.")
    print(f"Model: {model_path}")
    print(f"Workspace: {output_root / 'ws'}")
    print(f"Outputs: {out_dir}")
    print("Staged files:")
    for path in staged:
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
