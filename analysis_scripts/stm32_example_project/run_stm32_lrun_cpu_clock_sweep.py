#!/usr/bin/env python3
"""Sweep CPU clock / phase / weight mode combinations for the LRUN STM32 track."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from stm32_lrun_common import DEFAULT_CONFIG_PATH, device_defaults, load_config

RUNNER = SCRIPT_DIR / "run_stm32_lrun_toy_ai_hil.py"


def _run_case(
    *,
    project_root: Path,
    config: Path,
    stage_root: Path,
    output_path: Path,
    phase: str,
    weight_storage_mode: str,
    cpu_clock_mhz: int,
    skip_harness: bool,
    verbose: bool,
    reuse_staged_model: bool,
) -> int:
    cmd = [
        sys.executable,
        str(RUNNER),
        "--project-root",
        str(project_root),
        "--config",
        str(config),
        "--output",
        str(output_path),
        "--stage-output-root",
        str(stage_root),
        "--phase",
        phase,
        "--weight-storage-mode",
        weight_storage_mode,
        "--cpu-clock-mhz",
        str(cpu_clock_mhz),
    ]
    if skip_harness:
        cmd.append("--skip-harness")
    if reuse_staged_model:
        cmd.append("--reuse-staged-model")
    if verbose:
        cmd.append("--verbose")
    return subprocess.run(cmd, cwd=SCRIPT_DIR.parents[1], check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the LRUN STM32 CPU clock sweep.")
    parser.add_argument("--project-root", type=Path, default=SCRIPT_DIR / "stm32_lrun_toy_ai_project")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "out" / "stm32_lrun_cpu_clock_sweep")
    parser.add_argument("--skip-harness", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    defaults = device_defaults(load_config(args.config.resolve()))
    cpu_clocks = [int(value) for value in defaults["cpu_clock_mhz_options"]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    overall_rc = 0
    reuse_staged_model = {"embedded": False, "external_flash": False}

    for weight_storage_mode in ("embedded", "external_flash"):
        for phase in ("back_to_back", "cadenced"):
            for cpu_clock_mhz in cpu_clocks:
                run_label = f"{weight_storage_mode}_{phase}_{cpu_clock_mhz}mhz"
                output_path = args.output_dir / f"{run_label}.json"
                stage_root = args.output_dir / "stage" / weight_storage_mode
                stage_root.mkdir(parents=True, exist_ok=True)
                rc = _run_case(
                    project_root=args.project_root.resolve(),
                    config=args.config.resolve(),
                    stage_root=stage_root,
                    output_path=output_path,
                    phase=phase,
                    weight_storage_mode=weight_storage_mode,
                    cpu_clock_mhz=cpu_clock_mhz,
                    skip_harness=args.skip_harness,
                    verbose=args.verbose,
                    reuse_staged_model=reuse_staged_model[weight_storage_mode],
                )
                row: dict[str, object] = {
                    "run_label": run_label,
                    "phase": phase,
                    "weight_storage_mode": weight_storage_mode,
                    "cpu_clock_mhz": cpu_clock_mhz,
                    "boot_mode": "dev_boot",
                    "return_code": rc,
                    "output_path": str(output_path),
                }
                if output_path.is_file():
                    metrics = json.loads(output_path.read_text(encoding="utf-8"))
                    row.update(metrics)
                if (stage_root / "staging_manifest.json").is_file():
                    reuse_staged_model[weight_storage_mode] = True
                summary_rows.append(row)
                overall_rc = overall_rc or rc

    fieldnames = sorted({key for row in summary_rows for key in row.keys()})
    summary_path = args.output_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {summary_path}")
    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
