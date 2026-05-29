#!/usr/bin/env python3
"""Compare back-to-back and cadenced LRUN runs for the STM32 toy AI project."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER = SCRIPT_DIR / "run_stm32_lrun_toy_ai_hil.py"


def _run_case(
    *,
    project_root: Path,
    config: Path,
    output_path: Path,
    stage_root: Path,
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
    parser = argparse.ArgumentParser(description="Compare LRUN back-to-back and cadenced runs.")
    parser.add_argument("--project-root", type=Path, default=SCRIPT_DIR / "stm32_lrun_toy_ai_project")
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR.parents[1] / "src" / "config" / "nas_config_stm32.yaml")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "out" / "stm32_lrun_cadenced_comparison")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--weight-storage-mode", choices=["embedded", "external_flash"], default="external_flash")
    parser.add_argument("--cpu-clock-mhz", type=int, default=600)
    parser.add_argument("--skip-harness", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    overall_rc = 0
    reuse_staged_model = False

    for phase in ("back_to_back", "cadenced"):
        for attempt in range(1, args.attempts + 1):
            label = f"{phase}_attempt{attempt}"
            output_path = args.output_dir / f"{label}.json"
            stage_root = args.output_dir / "stage" / args.weight_storage_mode
            stage_root.mkdir(parents=True, exist_ok=True)
            rc = _run_case(
                project_root=args.project_root.resolve(),
                config=args.config.resolve(),
                output_path=output_path,
                stage_root=stage_root,
                phase=phase,
                weight_storage_mode=args.weight_storage_mode,
                cpu_clock_mhz=args.cpu_clock_mhz,
                skip_harness=args.skip_harness,
                verbose=args.verbose,
                reuse_staged_model=reuse_staged_model,
            )
            row: dict[str, object] = {
                "label": label,
                "phase": phase,
                "attempt": attempt,
                "return_code": rc,
                "output_path": str(output_path),
            }
            if output_path.is_file():
                row.update(json.loads(output_path.read_text(encoding="utf-8")))
            if (stage_root / "staging_manifest.json").is_file():
                reuse_staged_model = True
            rows.append(row)
            overall_rc = overall_rc or rc

    fieldnames = sorted({key for row in rows for key in row.keys()})
    summary_path = args.output_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {summary_path}")
    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
