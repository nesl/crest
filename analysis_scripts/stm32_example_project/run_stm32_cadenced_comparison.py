#!/usr/bin/env python3
"""Run back-to-back and cadenced STM32 toy-AI HIL comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
RUNNER_PATH = SCRIPT_DIR / "run_stm32_toy_ai_hil.py"
DEFAULT_PROJECT_ROOT = SCRIPT_DIR.parents[1] / "sketches" / "stm32" / "tinyodom_tcn_stm32" / "FSBL"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results"
VALID_PHASES = ("back_to_back", "cadenced")
VALID_CPU_CLOCK_MHZ = (200, 300, 400, 600, 800)
DEFAULT_CPU_CLOCK_MHZ = 600
MODEL_VARIANT_NAME = "approx_trained"
MASTER_SUCCESS_CODE = 1

CSV_COLUMNS = [
    "timestamp_utc",
    "phase",
    "repeat",
    "model_variant",
    "latency_budget_ms_requested",
    "latency_budget_ms_reported",
    "latency_ms",
    "window_latency_ms",
    "harness_latency_ms",
    "active_inference_latency_ms",
    "cpu_clock_mhz_requested",
    "energy_mj_per_inference",
    "energy_mj_per_window",
    "avg_power_mw",
    "avg_current_ma",
    "bus_voltage_v",
    "idle_power_mw",
    "rtc_sleep_ms",
    "deadline_miss_count",
    "measured_runs",
    "clock_hz",
    "dwt_cycles_per_inference",
    "wake_recovery_us_mean",
    "wake_overshoot_us_mean",
    "rtc_clock_source",
    "stop_mode_variant",
    "inference_seq",
    "error_code",
    "error_label",
    "timing_sanity_warning",
    "back_to_back_energy_mj_per_inference",
    "back_to_back_latency_ms_per_window",
    "cadenced_energy_mj_per_window",
    "cadenced_latency_ms_per_window",
]

AGG_NUMERIC_FIELDS = (
    "latency_ms",
    "window_latency_ms",
    "harness_latency_ms",
    "active_inference_latency_ms",
    "energy_mj_per_inference",
    "energy_mj_per_window",
    "avg_power_mw",
    "avg_current_ma",
    "bus_voltage_v",
    "idle_power_mw",
    "rtc_sleep_ms",
    "clock_hz",
    "dwt_cycles_per_inference",
    "wake_recovery_us_mean",
    "wake_overshoot_us_mean",
)


def _safe_float(value: Any) -> float:
    """Convert a value to a finite float or return a sentinel.

    Parameters
    ----------
    value : Any
        Candidate numeric value extracted from JSON or parsed CLI output.

    Returns
    -------
    float
        Parsed finite float value, or ``-1.0`` when the input is missing,
        not numeric, or not finite.
    """
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return -1.0
    return numeric if math.isfinite(numeric) else -1.0


def _is_successful_attempt(attempt: dict[str, Any]) -> bool:
    """Return whether one attempt completed with the success status code.

    Parameters
    ----------
    attempt : dict[str, Any]
        Attempt record assembled from the single-run HIL metrics contract.

    Returns
    -------
    bool
        ``True`` when the attempt error code matches
        ``HIL_MASTER_SUCCESS``, otherwise ``False``.
    """
    return int(_safe_float(attempt.get("error_code", -1))) == MASTER_SUCCESS_CODE


def _build_attempt_record(
    *,
    timestamp_utc: str,
    phase: str,
    repeat_idx: int,
    latency_budget_ms: float,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Build one flattened comparison-row record from single-run metrics.

    Parameters
    ----------
    timestamp_utc : str
        ISO-8601 UTC timestamp shared across the comparison run.
    phase : str
        Attempt phase label, either ``"back_to_back"`` or ``"cadenced"``.
    repeat_idx : int
        One-based repeat index for the current phase.
    latency_budget_ms : float
        Requested cadence budget used to launch the attempt.
    metrics : dict[str, Any]
        Parsed metrics JSON emitted by ``run_stm32_toy_ai_hil.py``.

    Returns
    -------
    dict[str, Any]
        Flat attempt record suitable for JSON serialization and CSV export.
    """
    record = {
        "timestamp_utc": timestamp_utc,
        "phase": phase,
        "repeat": repeat_idx,
        "model_variant": MODEL_VARIANT_NAME,
        "latency_budget_ms_requested": latency_budget_ms,
        "latency_budget_ms_reported": _safe_float(metrics.get("latency_budget_ms", -1.0)),
        "latency_ms": _safe_float(metrics.get("latency_ms", -1.0)),
        "window_latency_ms": _safe_float(metrics.get("window_latency_ms", -1.0)),
        "harness_latency_ms": _safe_float(metrics.get("harness_latency_ms", -1.0)),
        "active_inference_latency_ms": _safe_float(metrics.get("active_inference_latency_ms", -1.0)),
        "cpu_clock_mhz_requested": int(_safe_float(metrics.get("cpu_clock_mhz_requested", -1))),
        "energy_mj_per_inference": _safe_float(metrics.get("energy_mj_per_inference", -1.0)),
        "energy_mj_per_window": _safe_float(metrics.get("energy_mj_per_window", -1.0)),
        "avg_power_mw": _safe_float(metrics.get("avg_power_mw", -1.0)),
        "avg_current_ma": _safe_float(metrics.get("avg_current_ma", -1.0)),
        "bus_voltage_v": _safe_float(metrics.get("bus_voltage_v", -1.0)),
        "idle_power_mw": _safe_float(metrics.get("idle_power_mw", -1.0)),
        "rtc_sleep_ms": _safe_float(metrics.get("rtc_sleep_ms", -1.0)),
        "deadline_miss_count": int(_safe_float(metrics.get("deadline_miss_count", -1))),
        "measured_runs": int(_safe_float(metrics.get("measured_runs", -1))),
        "clock_hz": _safe_float(metrics.get("clock_hz", -1.0)),
        "dwt_cycles_per_inference": _safe_float(metrics.get("dwt_cycles_per_inference", -1.0)),
        "wake_recovery_us_mean": _safe_float(metrics.get("wake_recovery_us_mean", -1.0)),
        "wake_overshoot_us_mean": _safe_float(metrics.get("wake_overshoot_us_mean", -1.0)),
        "rtc_clock_source": metrics.get("rtc_clock_source"),
        "stop_mode_variant": metrics.get("stop_mode_variant"),
        "inference_seq": int(_safe_float(metrics.get("inference_seq", -1))),
        "error_code": int(_safe_float(metrics.get("error_code", -1))),
        "error_label": metrics.get("error_label"),
        "timing_sanity_warning": metrics.get("timing_sanity_warning"),
        "back_to_back_energy_mj_per_inference": "",
        "back_to_back_latency_ms_per_window": "",
        "cadenced_energy_mj_per_window": "",
        "cadenced_latency_ms_per_window": "",
    }
    if phase == "back_to_back":
        record["back_to_back_energy_mj_per_inference"] = record["energy_mj_per_inference"]
        record["back_to_back_latency_ms_per_window"] = (
            record["window_latency_ms"] if record["window_latency_ms"] >= 0.0 else record["latency_ms"]
        )
    else:
        record["cadenced_energy_mj_per_window"] = record["energy_mj_per_window"]
        record["cadenced_latency_ms_per_window"] = (
            record["window_latency_ms"] if record["window_latency_ms"] >= 0.0 else record["latency_ms"]
        )
    return record


def _valid_metric_values(attempts: list[dict[str, Any]], key: str) -> list[float]:
    """Collect valid non-negative numeric values for one aggregate field.

    Parameters
    ----------
    attempts : list[dict[str, Any]]
        Attempt records for one phase.
    key : str
        Metric field to extract.

    Returns
    -------
    list[float]
        Successful non-negative numeric values for the requested metric.
    """
    values: list[float] = []
    for attempt in attempts:
        if not _is_successful_attempt(attempt):
            continue
        value = _safe_float(attempt.get(key, -1.0))
        if value >= 0.0:
            values.append(value)
    return values


def _summarize_group(phase: str, attempts: list[dict[str, Any]], latency_budget_ms: float) -> dict[str, Any]:
    """Summarize repeated attempts for one comparison phase.

    Parameters
    ----------
    phase : str
        Phase label, either ``"back_to_back"`` or ``"cadenced"``.
    attempts : list[dict[str, Any]]
        Attempt records belonging to the phase.
    latency_budget_ms : float
        Requested cadence budget used to compute over-budget counts for
        baseline runs when needed.

    Returns
    -------
    dict[str, Any]
        Aggregate statistics including counts plus mean/std/min/max for each
        numeric metric field.
    """
    summary: dict[str, Any] = {
        "phase": phase,
        "attempt_count": len(attempts),
        "success_count": sum(1 for attempt in attempts if _is_successful_attempt(attempt)),
        "failure_count": sum(1 for attempt in attempts if not _is_successful_attempt(attempt)),
        "over_budget_count": sum(
            1
            for attempt in attempts
            if _is_successful_attempt(attempt)
            and (
                int(_safe_float(attempt.get("deadline_miss_count", -1))) > 0
                if phase == "cadenced"
                else _safe_float(attempt.get("window_latency_ms", -1.0))
                > latency_budget_ms * max(1, int(_safe_float(attempt.get("measured_runs", 1))))
            )
        ),
    }
    for key in AGG_NUMERIC_FIELDS:
        values = _valid_metric_values(attempts, key)
        if not values:
            summary[f"{key}_mean"] = -1.0
            summary[f"{key}_std"] = -1.0
            summary[f"{key}_min"] = -1.0
            summary[f"{key}_max"] = -1.0
            summary[f"{key}_n"] = 0
            continue
        summary[f"{key}_mean"] = statistics.fmean(values)
        summary[f"{key}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
        summary[f"{key}_min"] = min(values)
        summary[f"{key}_max"] = max(values)
        summary[f"{key}_n"] = len(values)
    return summary


def _write_csv(path: Path, attempts: list[dict[str, Any]]) -> None:
    """Write flattened attempt rows to a CSV file.

    Parameters
    ----------
    path : Path
        Destination CSV path.
    attempts : list[dict[str, Any]]
        Attempt records to serialize.

    Returns
    -------
    None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in attempts:
            writer.writerow({column: record.get(column, "") for column in CSV_COLUMNS})


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the cadenced STM32 comparison runner.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser with output paths, phase timing, hardware ports,
        and build/debug-tool options.
    """
    parser = argparse.ArgumentParser(
        description="Run STM32 toy-AI back-to-back and cadenced comparisons and write JSON + CSV summaries.",
        epilog=(
            "Examples:\n"
            "  python analysis_scripts/stm32_example_project/run_stm32_cadenced_comparison.py\n"
            "  python analysis_scripts/stm32_example_project/run_stm32_cadenced_comparison.py "
            "--repeats 3 --latency-budget-ms 200\n"
            "  python analysis_scripts/stm32_example_project/run_stm32_cadenced_comparison.py "
            "--weight-storage-mode external_flash --reuse-staged-model\n"
            "  python analysis_scripts/stm32_example_project/run_stm32_cadenced_comparison.py "
            "--output-json /tmp/stm32_compare.json --output-csv /tmp/stm32_compare.csv"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help=(
            "STM32 FSBL project root for the copied cadenced toy-AI project. "
            f"Default: {DEFAULT_PROJECT_ROOT}. "
            "Example: analysis_scripts/stm32_example_project/stm32_cadenced_toy_ai_project/FSBL"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "src" / "nas_config.yaml",
        help=(
            "TinyODOM NAS configuration YAML used for staging the perturbed model. "
            "Example: src/nas_config.yaml"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help=(
            "Destination JSON summary path. Defaults to a timestamped file under "
            f"{DEFAULT_OUTPUT_DIR}."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=(
            "Destination CSV summary path. Defaults to a timestamped file under "
            f"{DEFAULT_OUTPUT_DIR}."
        ),
    )
    parser.add_argument("--repeats", type=int, default=1, help="Number of repeats to run per phase. Default: 1.")
    parser.add_argument(
        "--latency-budget-ms",
        type=float,
        default=200.0,
        help="Cadence period in milliseconds for cadenced runs. Default: 200.0.",
    )
    parser.add_argument(
        "--measured-runs",
        type=int,
        default=10,
        help="Number of inference slots or back-to-back inferences per attempt. Default: 10.",
    )
    parser.add_argument(
        "--cpu-clock-mhz",
        type=int,
        choices=VALID_CPU_CLOCK_MHZ,
        default=DEFAULT_CPU_CLOCK_MHZ,
        help=(
            "Fixed CPU clock preset forwarded to the child runner. "
            f"Allowed values: {', '.join(str(value) for value in VALID_CPU_CLOCK_MHZ)}. "
            f"Default: {DEFAULT_CPU_CLOCK_MHZ}. "
            "Use 800 only as the dedicated overdrive preset."
        ),
    )
    parser.add_argument(
        "--wake-margin-us",
        type=int,
        default=5000,
        help="Wake margin in microseconds written into the generated phase config header. Default: 5000.",
    )
    parser.add_argument(
        "--min-sleep-us",
        type=int,
        default=5000,
        help="Minimum slack in microseconds required before the DUT enters Stop mode. Default: 5000.",
    )
    parser.add_argument(
        "--dut-port",
        default="/dev/ttyACM0",
        help="DUT serial port. Default: /dev/ttyACM0. Example: /dev/ttyUSB0",
    )
    parser.add_argument(
        "--harness-port",
        default="/dev/ttyACM1",
        help="Harness serial port. Default: /dev/ttyACM1. Example: /dev/ttyUSB1",
    )
    parser.add_argument("--baud", type=int, default=115200, help="Shared serial baud rate. Default: 115200.")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean the staging directory and rebuild the STM32 project before the first attempt.",
    )
    parser.add_argument(
        "--reuse-staged-model",
        action="store_true",
        help="Reuse the already staged STM32 model artifacts instead of regenerating them.",
    )
    parser.add_argument(
        "--weight-storage-mode",
        choices=["embedded", "external_flash"],
        default="embedded",
        help="Target weight-storage mode. Default: embedded.",
    )
    parser.add_argument(
        "--weights-flash-address",
        default="0x71000000",
        help="Base NOR-flash address for external weights. Default: 0x71000000.",
    )
    parser.add_argument(
        "--weights-memory-pool",
        type=Path,
        default=SCRIPT_DIR / "nucleo_mypool.json",
        help="ST Edge AI memory-pool JSON used for external weights.",
    )
    parser.add_argument(
        "--weights-external-loader",
        type=Path,
        default=None,
        help="Optional explicit .stldr loader path for external-flash programming.",
    )
    parser.add_argument("--gdbserver", type=Path, default=None, help="Optional path override for ST-LINK_gdbserver.")
    parser.add_argument("--gdb", type=Path, default=None, help="Optional path override for arm-none-eabi-gdb.")
    parser.add_argument(
        "--cubeprog-bin",
        type=Path,
        default=None,
        help="Optional STM32CubeProgrammer bin directory override.",
    )
    parser.add_argument("--gdb-port", type=int, default=61234, help="ST-LINK GDB server port. Default: 61234.")
    parser.add_argument("--apid", type=int, default=1, help="ST-Link access port ID. Default: 1.")
    parser.add_argument(
        "--server-ready-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for ST-LINK_gdbserver readiness. Default: 30.0.",
    )
    parser.add_argument(
        "--dut-ready-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for 'DUT READY' after loading firmware. Default: 30.0.",
    )
    parser.add_argument(
        "--harness-ready-timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for 'HARNESS READY'. Default: 5.0.",
    )
    parser.add_argument(
        "--harness-done-timeout",
        type=float,
        default=10.0,
        help="Extra seconds added when waiting for the harness DONE line. Default: 10.0.",
    )
    parser.add_argument(
        "--harness-fqbn",
        default="arduino:mbed_nano:nano33ble",
        help="Arduino FQBN for the harness firmware. Default: arduino:mbed_nano:nano33ble.",
    )
    parser.add_argument(
        "--harness-auto-flash",
        choices=["once", "always", "never"],
        default="once",
        help="Harness flashing policy. Default: once.",
    )
    parser.add_argument("--harness-arm-pin", type=int, default=3, help="Harness ARM GPIO pin. Default: 3.")
    parser.add_argument(
        "--harness-trigger-pin",
        type=int,
        default=2,
        help="Harness trigger-detect GPIO pin. Default: 2.",
    )
    parser.add_argument(
        "--dut-arm-hold-ms",
        type=int,
        default=600,
        help="Milliseconds to hold the harness ARM signal before release. Default: 600.",
    )
    parser.add_argument(
        "--harness-stable-low-ms",
        type=int,
        default=500,
        help="Milliseconds of stable-low trigger required before harness completion. Default: 500.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Parallel make job count for STM32 builds. Default: use the runner default.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug-level logging.")
    return parser


def _append_path_arg(cmd: list[str], flag: str, value: Path | None) -> None:
    """Append a CLI flag and path argument when the path is present.

    Parameters
    ----------
    cmd : list[str]
        Mutable subprocess command being assembled.
    flag : str
        CLI flag name to append, such as ``"--gdb"``.
    value : Path or None
        Optional path value to append after the flag.

    Returns
    -------
    None
    """
    if value is not None:
        cmd.extend([flag, str(value)])


def _expected_serial_timeout_s(phase: str, latency_budget_ms: float, measured_runs: int) -> float:
    """Estimate a base serial timeout for one phase attempt.

    Parameters
    ----------
    phase : str
        Phase label, either ``"back_to_back"`` or ``"cadenced"``.
    latency_budget_ms : float
        Cadence period in milliseconds.
    measured_runs : int
        Number of inference slots or repeated inferences in the attempt.

    Returns
    -------
    float
        Base timeout in seconds passed to the single-run STM32 HIL runner.
    """
    if phase == "cadenced":
        return max(30.0, (latency_budget_ms * measured_runs) / 1000.0 + 10.0)
    return 30.0


def _run_phase_attempt(args: argparse.Namespace, phase: str, repeat_idx: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Launch one phase attempt through the single-run STM32 HIL runner.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments for the comparison runner.
    phase : str
        Phase label to run.
    repeat_idx : int
        One-based repeat index for the phase.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        Parsed metrics JSON and diagnostics JSON emitted by the child runner.

    Raises
    ------
    RuntimeError
        Raised if the child process exits unexpectedly or the expected JSON
        artifacts cannot be consumed.
    """
    with tempfile.TemporaryDirectory(prefix=f"stm32_{phase}_{repeat_idx}_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        output_json = tmpdir_path / "metrics.json"
        cmd = [
            sys.executable,
            str(RUNNER_PATH),
            "--project-root",
            str(args.project_root),
            "--config",
            str(args.config),
            "--output",
            str(output_json),
            "--phase",
            phase,
            "--latency-budget-ms",
            str(args.latency_budget_ms),
            "--measured-runs",
            str(args.measured_runs),
            "--cpu-clock-mhz",
            str(args.cpu_clock_mhz),
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
            str(_expected_serial_timeout_s(phase, args.latency_budget_ms, args.measured_runs)),
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
            str(args.weight_storage_mode),
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
        if args.reuse_staged_model:
            cmd.append("--reuse-staged-model")
        if args.verbose:
            cmd.append("--verbose")
        if args.clean and phase == VALID_PHASES[0] and repeat_idx == 1:
            cmd.append("--clean")

        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
        if proc.returncode not in (0, 1):
            raise RuntimeError(
                f"Attempt subprocess failed for phase={phase} repeat={repeat_idx}.\n"
                f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
            )

        diagnostic_path = output_json.with_name("metrics.diagnostics.json")
        if not output_json.is_file() or not diagnostic_path.is_file():
            raise RuntimeError(
                f"Attempt subprocess exited before producing metrics artifacts for phase={phase} "
                f"repeat={repeat_idx} (returncode={proc.returncode}).\n"
                f"metrics_exists={output_json.is_file()} diagnostics_exists={diagnostic_path.is_file()}\n\n"
                f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
            )

        metrics = json.loads(output_json.read_text(encoding="utf-8"))
        diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
        return metrics, diagnostic


def main() -> int:
    """Run the full back-to-back versus cadenced comparison workflow.

    Returns
    -------
    int
        ``0`` on successful comparison-run completion.
    """
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    if args.measured_runs < 1:
        raise SystemExit("--measured-runs must be >= 1")

    timestamp_utc = datetime.now(timezone.utc).isoformat()
    timestamp_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.output_json is None:
        args.output_json = DEFAULT_OUTPUT_DIR / f"stm32_cadenced_comparison_{timestamp_tag}.json"
    if args.output_csv is None:
        args.output_csv = DEFAULT_OUTPUT_DIR / f"stm32_cadenced_comparison_{timestamp_tag}.csv"

    attempts: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    for phase in VALID_PHASES:
        for repeat_idx in range(1, args.repeats + 1):
            metrics, diagnostic = _run_phase_attempt(args, phase, repeat_idx)
            attempts.append(
                _build_attempt_record(
                    timestamp_utc=timestamp_utc,
                    phase=phase,
                    repeat_idx=repeat_idx,
                    latency_budget_ms=args.latency_budget_ms,
                    metrics=metrics,
                )
            )
            diagnostics.append(
                {
                    "phase": phase,
                    "repeat": repeat_idx,
                    "metrics": metrics,
                    "diagnostic": diagnostic,
                }
            )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for attempt in attempts:
        grouped[str(attempt["phase"])].append(attempt)
    aggregates = [
        _summarize_group(phase, phase_attempts, args.latency_budget_ms)
        for phase, phase_attempts in sorted(grouped.items())
    ]

    payload = {
        "metadata": {
            "timestamp_utc": timestamp_utc,
            "config_path": str(args.config),
            "project_root": str(args.project_root),
            "model_variant": MODEL_VARIANT_NAME,
            "phases": list(VALID_PHASES),
            "repeats": int(args.repeats),
            "latency_budget_ms_requested": float(args.latency_budget_ms),
            "measured_runs": int(args.measured_runs),
            "cpu_clock_mhz_requested": int(args.cpu_clock_mhz),
            "wake_margin_us": int(args.wake_margin_us),
            "min_sleep_us": int(args.min_sleep_us),
            "weight_storage_mode": args.weight_storage_mode,
            "rtc_clock_source": attempts[0].get("rtc_clock_source") if attempts else None,
            "rtc_clock_hz_nominal": diagnostics[0]["metrics"].get("rtc_clock_hz_nominal") if diagnostics else None,
            "cadence_timing_quality": diagnostics[0]["metrics"].get("cadence_timing_quality") if diagnostics else None,
            "stop_mode_variant": attempts[0].get("stop_mode_variant") if attempts else None,
            "notes": [
                "Cadenced phase uses Stop mode with RTC wake-up timer in the copied STM32 toy project.",
                "Harness energy output remains per inference; energy_mj_per_window is derived as per_inference * measured_runs.",
            ],
        },
        "attempts": attempts,
        "aggregates": aggregates,
        "diagnostics": diagnostics,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_csv(args.output_csv, attempts)
    print(f"Wrote JSON: {args.output_json}")
    print(f"Wrote CSV: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
