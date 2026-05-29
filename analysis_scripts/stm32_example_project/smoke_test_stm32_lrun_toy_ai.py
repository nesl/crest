#!/usr/bin/env python3
"""Smoke-test the LRUN STM32 toy AI path with stub and real-app passes."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from stm32_lrun_common import STAGING_MANIFEST_NAME

RUNNER = SCRIPT_DIR / "run_stm32_lrun_toy_ai_hil.py"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test the LRUN STM32 toy AI path.")
    parser.add_argument("--project-root", type=Path, default=SCRIPT_DIR / "stm32_lrun_toy_ai_project")
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR.parents[1] / "src" / "config" / "nas_config_stm32.yaml")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "out" / "stm32_lrun_smoke")
    parser.add_argument("--stage-output-root", type=Path, default=Path("/tmp/tinyodom_stm32_lrun_smoke"))
    parser.add_argument("--phase", choices=["back_to_back", "cadenced"], default="cadenced")
    parser.add_argument("--weight-storage-mode", choices=["embedded", "external_flash"], default="external_flash")
    parser.add_argument("--cpu-clock-mhz", type=int, default=600)
    parser.add_argument("--with-harness", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _run_one(
    label: str,
    args: argparse.Namespace,
    *,
    stub_only: bool,
    reuse_staged_model: bool,
    use_harness: bool,
) -> int:
    output_path = args.output_dir / f"{label}.json"
    cmd = [
        sys.executable,
        str(RUNNER),
        "--project-root",
        str(args.project_root.resolve()),
        "--config",
        str(args.config.resolve()),
        "--output",
        str(output_path),
        "--stage-output-root",
        str(args.stage_output_root.resolve()),
        "--phase",
        args.phase,
        "--weight-storage-mode",
        args.weight_storage_mode,
        "--cpu-clock-mhz",
        str(args.cpu_clock_mhz),
    ]
    if not use_harness:
        cmd.append("--skip-harness")
    if stub_only:
        cmd.append("--stub-only")
    if reuse_staged_model:
        cmd.append("--reuse-staged-model")
    if args.verbose:
        cmd.append("--verbose")
    return subprocess.run(cmd, cwd=SCRIPT_DIR.parents[1], check=False).returncode


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stub_rc = _run_one(
        "stub_smoke",
        args,
        stub_only=True,
        reuse_staged_model=False,
        use_harness=False,
    )
    if stub_rc != 0:
        return stub_rc

    real_rc = _run_one(
        "real_smoke",
        args,
        stub_only=False,
        reuse_staged_model=(args.stage_output_root.resolve() / STAGING_MANIFEST_NAME).is_file(),
        use_harness=args.with_harness,
    )
    return real_rc


if __name__ == "__main__":
    raise SystemExit(main())
