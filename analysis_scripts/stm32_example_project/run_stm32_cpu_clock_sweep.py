#!/usr/bin/env python3
"""Run an archival STM32 CPU-clock sweep and preserve every per-run artifact."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
RUNNER_PATH = SCRIPT_DIR / "run_stm32_toy_ai_hil.py"
DEFAULT_PROJECT_ROOT = SCRIPT_DIR / "stm32_cadenced_toy_ai_project" / "FSBL"
DEFAULT_CONFIG = REPO_ROOT / "src" / "nas_config.yaml"
DEFAULT_RESULTS_ROOT = SCRIPT_DIR / "results"
DEFAULT_STAGE_OUTPUT_ROOT = Path("/tmp/tinyodom_stm32_toy_generate")
DEFAULT_WEIGHTS_MEMORY_POOL = SCRIPT_DIR / "nucleo_mypool.json"
VALID_CPU_CLOCK_MHZ = (200, 300, 400, 600, 800)
VALID_PHASES = ("back_to_back", "cadenced")
DEFAULT_FREQUENCIES = VALID_CPU_CLOCK_MHZ
DEFAULT_REPEATS = 5

SUMMARY_COLUMNS = [
    "run_index",
    "scenario_name",
    "cpu_clock_mhz_requested",
    "phase",
    "weight_storage_mode",
    "repeat",
    "started_at_utc",
    "finished_at_utc",
    "duration_s",
    "return_code",
    "metrics_path",
    "diagnostics_path",
    "stdout_log_path",
    "stderr_log_path",
    "error_code",
    "error_label",
    "clock_hz",
    "latency_ms",
    "active_inference_latency_ms",
    "window_latency_ms",
    "energy_mj_per_inference",
    "energy_mj_per_window",
    "rtc_sleep_ms",
]


@dataclass(frozen=True)
class SweepScenario:
    """One archived run scenario."""

    name: str
    phase: str
    weight_storage_mode: str


DEFAULT_SCENARIOS = (
    SweepScenario(name="embedded_back_to_back", phase="back_to_back", weight_storage_mode="embedded"),
    SweepScenario(name="embedded_cadenced", phase="cadenced", weight_storage_mode="embedded"),
    SweepScenario(name="external_back_to_back", phase="back_to_back", weight_storage_mode="external_flash"),
    SweepScenario(name="external_cadenced", phase="cadenced", weight_storage_mode="external_flash"),
)


def _default_python_executable() -> Path:
    """Return the interpreter that should launch child STM32 runs.

    When the current shell is wrapped by ``conda run`` on this machine, the
    ``python`` command can resolve to ST Edge AI's shim instead of the active
    conda environment. Prefer the real ``$CONDA_PREFIX/bin/python`` when
    available.
    """
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        conda_python = Path(conda_prefix) / "bin" / "python"
        if conda_python.is_file():
            try:
                Path(sys.executable).resolve().relative_to(Path(conda_prefix).resolve())
            except ValueError:
                return conda_python.resolve()
    return Path(sys.executable).resolve()


def _diagnostic_output_path(output_path: Path) -> Path:
    """Return the sidecar diagnostics path for one metrics JSON path."""
    if output_path.suffix:
        return output_path.with_name(f"{output_path.stem}.diagnostics{output_path.suffix}")
    return output_path.with_name(f"{output_path.name}.diagnostics.json")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the sweep summary CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in SUMMARY_COLUMNS})


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the archival CPU-clock sweep."""
    parser = argparse.ArgumentParser(
        description=(
            "Run an archival STM32 CPU-clock sweep across embedded/external "
            "back-to-back and cadenced scenarios."
        ),
        epilog=(
            "Examples:\n"
            "  python analysis_scripts/stm32_example_project/run_stm32_cpu_clock_sweep.py\n"
            "  python analysis_scripts/stm32_example_project/run_stm32_cpu_clock_sweep.py "
            "--repeats 5 --frequencies 200 300 400 600 800\n"
            "  /home/joe/miniforge3/envs/tinyodomex/bin/python "
            "analysis_scripts/stm32_example_project/run_stm32_cpu_clock_sweep.py --clean-first"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help=f"Cadenced STM32 FSBL project root. Default: {DEFAULT_PROJECT_ROOT}.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"TinyODOM NAS config YAML. Default: {DEFAULT_CONFIG}.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help=f"Root directory that receives the timestamped sweep folder. Default: {DEFAULT_RESULTS_ROOT}.",
    )
    parser.add_argument(
        "--python-executable",
        type=Path,
        default=_default_python_executable(),
        help=(
            "Python interpreter used to launch child run_stm32_toy_ai_hil.py processes. "
            "Defaults to the active interpreter, preferring $CONDA_PREFIX/bin/python when needed."
        ),
    )
    parser.add_argument(
        "--frequencies",
        type=int,
        nargs="+",
        choices=VALID_CPU_CLOCK_MHZ,
        default=list(DEFAULT_FREQUENCIES),
        help="CPU clock presets to sweep. Default: 200 300 400 600 800.",
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=VALID_PHASES,
        default=list(VALID_PHASES),
        help="Phases to include in the sweep. Default: back_to_back cadenced.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=f"Repeat count per scenario and frequency. Default: {DEFAULT_REPEATS}.",
    )
    parser.add_argument(
        "--clean-first",
        action="store_true",
        help="Pass --clean on the first child run to force a fresh initial staging/build.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop the sweep immediately after the first failed child run.",
    )
    parser.add_argument(
        "--stage-output-root",
        type=Path,
        default=DEFAULT_STAGE_OUTPUT_ROOT,
        help=f"Shared staging root forwarded to child runs. Default: {DEFAULT_STAGE_OUTPUT_ROOT}.",
    )
    parser.add_argument("--dut-port", default="/dev/ttyACM0", help="DUT serial port. Default: /dev/ttyACM0.")
    parser.add_argument(
        "--harness-port",
        default="/dev/ttyACM1",
        help="Harness serial port. Default: /dev/ttyACM1.",
    )
    parser.add_argument("--baud", type=int, default=115200, help="Shared baud rate. Default: 115200.")
    parser.add_argument(
        "--latency-budget-ms",
        type=float,
        default=200.0,
        help="Latency budget forwarded to child runs. Default: 200.0.",
    )
    parser.add_argument(
        "--measured-runs",
        type=int,
        default=10,
        help="Measured inference count per child run. Default: 10.",
    )
    parser.add_argument(
        "--wake-margin-us",
        type=int,
        default=5000,
        help="Cadenced wake margin forwarded to child runs. Default: 5000.",
    )
    parser.add_argument(
        "--min-sleep-us",
        type=int,
        default=5000,
        help="Cadenced min sleep forwarded to child runs. Default: 5000.",
    )
    parser.add_argument(
        "--serial-timeout",
        type=float,
        default=30.0,
        help="Base serial timeout for child runs. Default: 30.0.",
    )
    parser.add_argument(
        "--dut-ready-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for DUT READY. Default: 30.0.",
    )
    parser.add_argument(
        "--harness-ready-timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for HARNESS READY. Default: 5.0.",
    )
    parser.add_argument(
        "--harness-done-timeout",
        type=float,
        default=10.0,
        help="Extra seconds added while waiting for DONE. Default: 10.0.",
    )
    parser.add_argument(
        "--harness-fqbn",
        default="arduino:mbed_nano:nano33ble",
        help="Harness board FQBN. Default: arduino:mbed_nano:nano33ble.",
    )
    parser.add_argument(
        "--harness-auto-flash",
        choices=["once", "always", "never"],
        default="once",
        help="Harness flashing policy. Default: once.",
    )
    parser.add_argument("--harness-arm-pin", type=int, default=3, help="Harness ARM pin. Default: 3.")
    parser.add_argument("--harness-trigger-pin", type=int, default=2, help="Harness trigger pin. Default: 2.")
    parser.add_argument("--dut-arm-hold-ms", type=int, default=600, help="DUT arm hold. Default: 600 ms.")
    parser.add_argument(
        "--harness-stable-low-ms",
        type=int,
        default=500,
        help="Stable-low window for harness completion. Default: 500 ms.",
    )
    parser.add_argument(
        "--weights-flash-address",
        default="0x71000000",
        help="External-flash base address for weight blobs. Default: 0x71000000.",
    )
    parser.add_argument(
        "--weights-memory-pool",
        type=Path,
        default=DEFAULT_WEIGHTS_MEMORY_POOL,
        help=f"Memory-pool JSON for external weights. Default: {DEFAULT_WEIGHTS_MEMORY_POOL}.",
    )
    parser.add_argument(
        "--weights-external-loader",
        type=Path,
        default=None,
        help="Optional explicit .stldr loader path.",
    )
    parser.add_argument("--gdbserver", type=Path, default=None, help="Optional ST-LINK_gdbserver override.")
    parser.add_argument("--gdb", type=Path, default=None, help="Optional arm-none-eabi-gdb override.")
    parser.add_argument("--cubeprog-bin", type=Path, default=None, help="Optional STM32CubeProgrammer bin override.")
    parser.add_argument("--gdb-port", type=int, default=61234, help="GDB server port. Default: 61234.")
    parser.add_argument("--apid", type=int, default=1, help="ST-Link APID. Default: 1.")
    parser.add_argument(
        "--server-ready-timeout",
        type=float,
        default=30.0,
        help="ST-LINK gdbserver readiness timeout. Default: 30.0.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Parallel make jobs for child builds. Default: child-runner default.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose child-runner logging.")
    return parser


def _append_path_arg(cmd: list[str], flag: str, value: Path | None) -> None:
    """Append one optional path flag."""
    if value is not None:
        cmd.extend([flag, str(value)])


def _build_child_command(
    *,
    args: argparse.Namespace,
    scenario: SweepScenario,
    cpu_clock_mhz: int,
    output_path: Path,
    reuse_staged_model: bool,
    clean: bool,
) -> list[str]:
    """Assemble one child run command."""
    cmd = [
        str(args.python_executable),
        str(RUNNER_PATH),
        "--project-root",
        str(args.project_root),
        "--config",
        str(args.config),
        "--output",
        str(output_path),
        "--stage-output-root",
        str(args.stage_output_root),
        "--phase",
        scenario.phase,
        "--cpu-clock-mhz",
        str(cpu_clock_mhz),
        "--latency-budget-ms",
        str(args.latency_budget_ms),
        "--measured-runs",
        str(args.measured_runs),
        "--wake-margin-us",
        str(args.wake_margin_us),
        "--min-sleep-us",
        str(args.min_sleep_us),
        "--dut-port",
        str(args.dut_port),
        "--harness-port",
        str(args.harness_port),
        "--baud",
        str(args.baud),
        "--serial-timeout",
        str(args.serial_timeout),
        "--dut-ready-timeout",
        str(args.dut_ready_timeout),
        "--harness-ready-timeout",
        str(args.harness_ready_timeout),
        "--harness-done-timeout",
        str(args.harness_done_timeout),
        "--harness-fqbn",
        str(args.harness_fqbn),
        "--harness-auto-flash",
        str(args.harness_auto_flash),
        "--harness-arm-pin",
        str(args.harness_arm_pin),
        "--harness-trigger-pin",
        str(args.harness_trigger_pin),
        "--dut-arm-hold-ms",
        str(args.dut_arm_hold_ms),
        "--harness-stable-low-ms",
        str(args.harness_stable_low_ms),
        "--weight-storage-mode",
        scenario.weight_storage_mode,
        "--weights-flash-address",
        str(args.weights_flash_address),
        "--weights-memory-pool",
        str(args.weights_memory_pool),
        "--gdb-port",
        str(args.gdb_port),
        "--apid",
        str(args.apid),
        "--server-ready-timeout",
        str(args.server_ready_timeout),
    ]
    _append_path_arg(cmd, "--weights-external-loader", args.weights_external_loader)
    _append_path_arg(cmd, "--gdbserver", args.gdbserver)
    _append_path_arg(cmd, "--gdb", args.gdb)
    _append_path_arg(cmd, "--cubeprog-bin", args.cubeprog_bin)
    if args.jobs is not None:
        cmd.extend(["--jobs", str(args.jobs)])
    if reuse_staged_model:
        cmd.append("--reuse-staged-model")
    if clean:
        cmd.append("--clean")
    if args.verbose:
        cmd.append("--verbose")
    return cmd


def _parse_metrics(metrics_path: Path) -> dict[str, Any]:
    """Read one metrics JSON if it exists."""
    if not metrics_path.is_file():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def main() -> int:
    """Run the archival sweep and preserve every child artifact."""
    parser = _build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s:%(message)s")

    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    if not args.python_executable.is_file():
        raise SystemExit(f"Python executable not found: {args.python_executable}")
    if not args.project_root.is_dir():
        raise SystemExit(f"Project root not found: {args.project_root}")
    if not args.config.is_file():
        raise SystemExit(f"Config not found: {args.config}")
    if not args.weights_memory_pool.is_file():
        raise SystemExit(f"Weights memory pool not found: {args.weights_memory_pool}")
    if args.weights_external_loader is not None and not args.weights_external_loader.is_file():
        raise SystemExit(f"Weights external loader not found: {args.weights_external_loader}")

    timestamp_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sweep_dir = args.results_root.resolve() / f"stm32_cpu_clock_sweep_{timestamp_tag}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    summary_json_path = sweep_dir / "sweep_summary.json"
    summary_csv_path = sweep_dir / "sweep_summary.csv"

    logging.info("Sweep results will be written under %s", sweep_dir)
    logging.info("Child python executable: %s", args.python_executable)

    records: list[dict[str, Any]] = []
    run_index = 0
    prior_weight_storage_mode: str | None = None
    prior_stage_succeeded = False
    had_failure = False

    selected_scenarios = [scenario for scenario in DEFAULT_SCENARIOS if scenario.phase in args.phases]
    if not selected_scenarios:
        raise SystemExit("No scenarios matched the requested phases.")

    total_runs = len(selected_scenarios) * len(args.frequencies) * args.repeats
    with tqdm(total=total_runs, desc="STM32 sweep", unit="run") as pbar:
        for scenario in selected_scenarios:
            for cpu_clock_mhz in args.frequencies:
                for repeat_idx in range(1, args.repeats + 1):
                    run_index += 1
                    run_stem = (
                        f"{run_index:03d}_{cpu_clock_mhz}mhz_{scenario.weight_storage_mode}"
                        f"_{scenario.phase}_repeat{repeat_idx:02d}"
                    )
                    output_path = sweep_dir / f"{run_stem}.json"
                    diagnostic_path = _diagnostic_output_path(output_path)
                    stdout_log_path = sweep_dir / f"{run_stem}.stdout.log"
                    stderr_log_path = sweep_dir / f"{run_stem}.stderr.log"

                    reuse_staged_model = (
                        prior_stage_succeeded and prior_weight_storage_mode == scenario.weight_storage_mode
                    )
                    clean = bool(args.clean_first and run_index == 1)
                    cmd = _build_child_command(
                        args=args,
                        scenario=scenario,
                        cpu_clock_mhz=cpu_clock_mhz,
                        output_path=output_path,
                        reuse_staged_model=reuse_staged_model,
                        clean=clean,
                    )

                    started_at = datetime.now(timezone.utc)
                    logging.info(
                        "Run %03d/%03d: %s %s %d MHz repeat %d",
                        run_index,
                        len(selected_scenarios) * len(args.frequencies) * args.repeats,
                        scenario.weight_storage_mode,
                        scenario.phase,
                        cpu_clock_mhz,
                        repeat_idx,
                    )
                    pbar.set_postfix(
                        phase=scenario.phase,
                        mode=scenario.weight_storage_mode,
                        mhz=cpu_clock_mhz,
                        repeat=repeat_idx,
                    )
                    proc = subprocess.run(
                        cmd,
                        cwd=REPO_ROOT,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    finished_at = datetime.now(timezone.utc)
                    duration_s = (finished_at - started_at).total_seconds()
                    stdout_log_path.write_text(proc.stdout, encoding="utf-8")
                    stderr_log_path.write_text(proc.stderr, encoding="utf-8")

                    metrics = _parse_metrics(output_path)
                    record = {
                        "run_index": run_index,
                        "scenario_name": scenario.name,
                        "cpu_clock_mhz_requested": cpu_clock_mhz,
                        "phase": scenario.phase,
                        "weight_storage_mode": scenario.weight_storage_mode,
                        "repeat": repeat_idx,
                        "started_at_utc": started_at.isoformat(),
                        "finished_at_utc": finished_at.isoformat(),
                        "duration_s": duration_s,
                        "return_code": proc.returncode,
                        "command": cmd,
                        "metrics_path": str(output_path),
                        "diagnostics_path": str(diagnostic_path),
                        "stdout_log_path": str(stdout_log_path),
                        "stderr_log_path": str(stderr_log_path),
                        "metrics_exists": output_path.is_file(),
                        "diagnostics_exists": diagnostic_path.is_file(),
                        "error_code": metrics.get("error_code"),
                        "error_label": metrics.get("error_label"),
                        "clock_hz": metrics.get("clock_hz"),
                        "latency_ms": metrics.get("latency_ms"),
                        "active_inference_latency_ms": metrics.get("active_inference_latency_ms"),
                        "window_latency_ms": metrics.get("window_latency_ms"),
                        "energy_mj_per_inference": metrics.get("energy_mj_per_inference"),
                        "energy_mj_per_window": metrics.get("energy_mj_per_window"),
                        "rtc_sleep_ms": metrics.get("rtc_sleep_ms"),
                    }
                    records.append(record)
                    pbar.update(1)
                    summary_json_path.write_text(
                    json.dumps(
                        {
                            "metadata": {
                                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                                "results_dir": str(sweep_dir),
                                "project_root": str(args.project_root),
                                "config": str(args.config),
                                "python_executable": str(args.python_executable),
                                "frequencies": list(args.frequencies),
                                "repeats": int(args.repeats),
                                "phases": list(args.phases),
                                "scenarios": [scenario.__dict__ for scenario in selected_scenarios],
                            },
                            "runs": records,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                    _write_csv(summary_csv_path, records)

                stage_succeeded = proc.returncode in (0, 1) and output_path.is_file()
                prior_weight_storage_mode = scenario.weight_storage_mode
                prior_stage_succeeded = stage_succeeded

                if proc.returncode != 0:
                    had_failure = True
                    logging.error(
                        "Run %03d failed with code %d. stdout=%s stderr=%s",
                        run_index,
                        proc.returncode,
                        stdout_log_path,
                        stderr_log_path,
                    )
                    if args.fail_fast:
                        logging.error("Stopping early because --fail-fast is enabled.")
                        return 1

    logging.info("Sweep complete. Summary JSON: %s", summary_json_path)
    logging.info("Sweep complete. Summary CSV: %s", summary_csv_path)
    print(f"Sweep directory: {sweep_dir}")
    print(f"Summary JSON: {summary_json_path}")
    print(f"Summary CSV: {summary_csv_path}")
    return 1 if had_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
