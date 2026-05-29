#!/usr/bin/env python3
"""
Run Portenta H7 baseline load tests (heavy vs sleep) with harness telemetry.

This runner executes a 2x2 matrix over selected cores/workloads:
1. CM7 + heavy (10 x 200 ms busy iterations)
2. CM7 + sleep (10 x 200 ms delay iterations)
3. CM4 + heavy
4. CM4 + sleep

Results are exported as per-attempt CSV rows plus JSON attempts/aggregates.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import shutil
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

try:
    import serial  # type: ignore
except Exception:  # pragma: no cover - import availability depends on runtime env
    serial = None  # type: ignore

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"

VALID_CORES = ("cm7", "cm4")
VALID_WORKLOADS = ("heavy", "sleep")
PORTENTA_FQBN = "arduino:mbed_portenta:envie_m7"
RUNS_PER_WINDOW = 10

WORKLOAD_TEMPLATE_FILES = {
    "heavy": "portenta_heavy_10x200ms.ino",
    "sleep": "portenta_sleep_10x200ms.ino",
}

ERROR_LABELS = {
    0: "ok",
    1: "compile_failed",
    2: "upload_failed",
    3: "harness_ready_timeout",
    4: "harness_done_timeout",
    5: "runtime_prepare_failed",
    6: "serial_error",
    7: "telemetry_parse_failed",
}

CSV_COLUMNS = [
    "timestamp_utc",
    "core",
    "workload",
    "repeat",
    "runs",
    "latency_ms_per_iter",
    "energy_mj_per_iter",
    "avg_power_mw",
    "avg_current_ma",
    "bus_voltage_v",
    "idle_power_mw",
    "error_code",
    "error_label",
    "runtime_mode",
    "board_options",
]

AGG_NUMERIC_FIELDS = (
    "latency_ms_per_iter",
    "energy_mj_per_iter",
    "avg_power_mw",
    "avg_current_ma",
    "bus_voltage_v",
    "idle_power_mw",
)

_RUNS_RE = re.compile(r"^runs:\s*(?P<value>\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class RuntimeSettings:
    """Resolved runtime settings for baseline load tests.

    Parameters
    ----------
    config_path : Path
        Configuration path used for optional defaults.
    dut_port : str
        DUT serial port.
    harness_port : str
        Harness serial port.
    harness_fqbn : str
        Harness FQBN for compile/upload.
    harness_auto_flash : str
        Harness flash policy (`once`, `always`, `never`).
    harness_arm_pin : int
        Arm pin shared between DUT and harness.
    harness_trigger_pin : int
        Trigger pin shared between DUT and harness.
    dut_arm_hold_ms : int
        Arm hold duration before trigger high.
    harness_stable_low_ms : int
        Stable-low arming window required by harness.
    harness_ready_timeout_s : float
        Timeout waiting for harness ready after ping.
    harness_arm_timeout_s : float
        Harness arm timeout (compiled into harness firmware).
    harness_active_timeout_s : float
        Active measurement timeout.
    harness_done_timeout_s : float
        Additional done timeout.
    repeats : int
        Repeats per core x workload.
    cores : list[str]
        Selected cores.
    output_json : Path
        JSON output path.
    output_csv : Path
        CSV output path.
    """

    config_path: Path
    dut_port: str
    harness_port: str
    harness_fqbn: str
    harness_auto_flash: str
    harness_arm_pin: int
    harness_trigger_pin: int
    dut_arm_hold_ms: int
    harness_stable_low_ms: int
    harness_ready_timeout_s: float
    harness_arm_timeout_s: float
    harness_active_timeout_s: float
    harness_done_timeout_s: float
    repeats: int
    cores: list[str]
    output_json: Path
    output_csv: Path


def _configure_logging(level_name: str) -> None:
    """Configure timestamped process logging.

    Parameters
    ----------
    level_name : str
        Logging level name (for example ``INFO`` or ``DEBUG``).
    """

    level = getattr(logging, str(level_name).upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _ensure_src_import_path() -> None:
    """Ensure the repository ``src`` directory is importable."""

    src_text = str(SRC_DIR)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)


def _load_runtime_defaults_from_config(config_path: Path) -> Dict[str, Any]:
    """Load device/harness defaults from TinyODOM config when available.

    Parameters
    ----------
    config_path : Path
        Candidate config path.

    Returns
    -------
    dict[str, Any]
        Optional defaults dictionary; empty when unavailable.
    """

    _ensure_src_import_path()
    try:
        from tinyodom.model import load_config  # type: ignore
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "Unable to import tinyodom.model.load_config; using CLI/fallback defaults (%s).",
            exc,
        )
        return {}

    try:
        cfg = load_config(config_path)
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "Unable to load config '%s'; using CLI/fallback defaults (%s).",
            config_path,
            exc,
        )
        return {}

    return {
        "dut_port": str(cfg.device.get("serial_port", "")) if cfg.device.get("serial_port") else None,
        "harness_port": str(cfg.device.get("harness_serial_port", "")) if cfg.device.get("harness_serial_port") else None,
        "harness_fqbn": str(cfg.device.get("harness_fqbn", "arduino:mbed_nano:nano33ble")),
        "harness_auto_flash": str(cfg.device.get("harness_auto_flash", "once")),
        "harness_arm_pin": int(cfg.device.get("harness_arm_pin", 3)),
        "harness_trigger_pin": int(cfg.device.get("harness_trigger_pin", 2)),
        "dut_arm_hold_ms": int(cfg.device.get("dut_arm_hold_ms", 600)),
        "harness_stable_low_ms": int(cfg.device.get("harness_stable_low_ms", 500)),
        "harness_ready_timeout_s": float(cfg.device.get("harness_ready_timeout_s", 3.6)),
        "harness_arm_timeout_s": float(cfg.device.get("harness_arm_timeout_s", 0.0)),
        "harness_active_timeout_s": float(cfg.device.get("harness_active_timeout_s", 12.0)),
        "harness_done_timeout_s": float(cfg.device.get("harness_done_timeout_s", 3.6)),
    }


def _resolve_settings(args: argparse.Namespace) -> RuntimeSettings:
    """Resolve runtime settings from CLI arguments and optional config defaults.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    RuntimeSettings
        Final runtime settings for the run.
    """

    config_path = Path(args.config)
    cfg_defaults = _load_runtime_defaults_from_config(config_path)

    cores = [str(core).strip().lower() for core in args.cores]
    invalid_cores = [core for core in cores if core not in VALID_CORES]
    if invalid_cores:
        raise ValueError(f"Invalid cores: {invalid_cores}. Expected subset of {VALID_CORES}.")
    loop_count = int(args.loops) if args.loops is not None else int(args.repeats)
    if loop_count < 1:
        raise ValueError("--loops/--repeats must be >= 1.")

    timestamp_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    default_stem = SCRIPT_DIR / "results" / f"portenta_baseline_load_{timestamp_tag}"

    harness_auto_flash = "never" if args.skip_harness_flash else str(
        args.harness_auto_flash or cfg_defaults.get("harness_auto_flash") or "once"
    ).lower()
    if harness_auto_flash not in {"once", "always", "never"}:
        raise ValueError("--harness-auto-flash must be one of: once, always, never.")

    return RuntimeSettings(
        config_path=config_path,
        dut_port=args.dut_port or cfg_defaults.get("dut_port") or "/dev/ttyACM0",
        harness_port=args.harness_port or cfg_defaults.get("harness_port") or "/dev/ttyACM1",
        harness_fqbn=args.harness_fqbn or cfg_defaults.get("harness_fqbn") or "arduino:mbed_nano:nano33ble",
        harness_auto_flash=harness_auto_flash,
        harness_arm_pin=int(args.harness_arm_pin if args.harness_arm_pin is not None else cfg_defaults.get("harness_arm_pin", 3)),
        harness_trigger_pin=int(args.harness_trigger_pin if args.harness_trigger_pin is not None else cfg_defaults.get("harness_trigger_pin", 2)),
        dut_arm_hold_ms=int(args.dut_arm_hold_ms if args.dut_arm_hold_ms is not None else cfg_defaults.get("dut_arm_hold_ms", 600)),
        harness_stable_low_ms=int(
            args.harness_stable_low_ms
            if args.harness_stable_low_ms is not None
            else cfg_defaults.get("harness_stable_low_ms", 500)
        ),
        harness_ready_timeout_s=float(
            args.harness_ready_timeout_s
            if args.harness_ready_timeout_s is not None
            else cfg_defaults.get("harness_ready_timeout_s", 3.6)
        ),
        harness_arm_timeout_s=float(
            args.harness_arm_timeout_s
            if args.harness_arm_timeout_s is not None
            else cfg_defaults.get("harness_arm_timeout_s", 0.0)
        ),
        harness_active_timeout_s=float(
            args.harness_active_timeout_s
            if args.harness_active_timeout_s is not None
            else cfg_defaults.get("harness_active_timeout_s", 12.0)
        ),
        harness_done_timeout_s=float(
            args.harness_done_timeout_s
            if args.harness_done_timeout_s is not None
            else cfg_defaults.get("harness_done_timeout_s", 3.6)
        ),
        repeats=loop_count,
        cores=cores,
        output_json=Path(args.output_json) if args.output_json else default_stem.with_suffix(".json"),
        output_csv=Path(args.output_csv) if args.output_csv else default_stem.with_suffix(".csv"),
    )


def _stage_sketch(workload: str) -> Path:
    """Stage one workload sketch into a compile-ready Arduino sketch folder.

    Parameters
    ----------
    workload : str
        Workload key (`heavy` or `sleep`).

    Returns
    -------
    Path
        Staged sketch directory path.
    """

    if workload not in WORKLOAD_TEMPLATE_FILES:
        raise ValueError(f"Unsupported workload '{workload}'. Expected one of {VALID_WORKLOADS}.")

    sketch_name = "portenta_baseline_load"
    stage_dir = SCRIPT_DIR / ".staged_sketch" / sketch_name
    stage_dir.mkdir(parents=True, exist_ok=True)

    template_path = SCRIPT_DIR / WORKLOAD_TEMPLATE_FILES[workload]
    if not template_path.exists():
        raise FileNotFoundError(f"Missing workload template: {template_path}")

    staged_ino = stage_dir / f"{sketch_name}.ino"
    staged_ino.write_text(template_path.read_text())

    shared_common = REPO_ROOT / "sketches" / "common"
    shutil.copytree(shared_common, stage_dir / "common", dirs_exist_ok=True)
    return stage_dir


def _parse_runs(lines: Iterable[str]) -> int:
    """Parse ``runs: <N>`` from harness log lines.

    Parameters
    ----------
    lines : Iterable[str]
        Harness serial log lines.

    Returns
    -------
    int
        Parsed runs count, or ``-1`` when unavailable.
    """

    for raw_line in lines:
        line = raw_line.strip()
        match = _RUNS_RE.match(line)
        if match:
            try:
                return int(match.group("value"))
            except (TypeError, ValueError):
                return -1
    return -1


def _safe_float(value: Any) -> float:
    """Convert values to finite float with ``-1`` fallback.

    Parameters
    ----------
    value : Any
        Candidate scalar.

    Returns
    -------
    float
        Finite numeric value or ``-1.0``.
    """

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return -1.0
    return parsed if parsed == parsed and parsed not in (float("inf"), float("-inf")) else -1.0


def _build_harness_defines(settings: RuntimeSettings) -> Dict[str, int]:
    """Build compile-time defines for harness firmware.

    Parameters
    ----------
    settings : RuntimeSettings
        Resolved run settings.

    Returns
    -------
    dict[str, int]
        Harness build defines.
    """

    return {
        "TINYODOM_HARNESS_ARM_PIN": int(settings.harness_arm_pin),
        "TINYODOM_HARNESS_TRIGGER_PIN": int(settings.harness_trigger_pin),
        "TINYODOM_DUT_ARM_HOLD_MS": int(settings.dut_arm_hold_ms),
        "TINYODOM_HARNESS_STABLE_LOW_MS": int(settings.harness_stable_low_ms),
        "TINYODOM_HARNESS_ARM_TIMEOUT_MS": max(0, int(round(settings.harness_arm_timeout_s * 1000.0))),
        "TINYODOM_HARNESS_ACTIVE_TIMEOUT_MS": max(0, int(round(settings.harness_active_timeout_s * 1000.0))),
        "TINYODOM_INFERENCE_RUNS": RUNS_PER_WINDOW,
    }


def _build_dut_defines(settings: RuntimeSettings, runtime_mode_build_defines: Dict[str, int]) -> Dict[str, int]:
    """Build compile-time defines for DUT sketches.

    Parameters
    ----------
    settings : RuntimeSettings
        Resolved run settings.
    runtime_mode_build_defines : dict[str, int]
        Core-specific runtime defines from Portenta wrapper.

    Returns
    -------
    dict[str, int]
        DUT build defines.
    """

    defines = {
        "TINYODOM_AUTOSTART": 1,
        "TINYODOM_HARNESS_ARM_PIN": int(settings.harness_arm_pin),
        "TINYODOM_HARNESS_TRIGGER_PIN": int(settings.harness_trigger_pin),
        "TINYODOM_DUT_ARM_HOLD_MS": int(settings.dut_arm_hold_ms),
        "TINYODOM_HARNESS_STABLE_LOW_MS": int(settings.harness_stable_low_ms),
        "TINYODOM_INFERENCE_RUNS": RUNS_PER_WINDOW,
    }
    defines.update(runtime_mode_build_defines)
    return defines


def _run_attempt(
    *,
    settings: RuntimeSettings,
    core: str,
    workload: str,
    repeat_idx: int,
) -> Dict[str, Any]:
    """Execute one core/workload attempt and return a flat metrics record.

    Parameters
    ----------
    settings : RuntimeSettings
        Resolved run settings.
    core : str
        Target core (`cm7` or `cm4`).
    workload : str
        Workload key (`heavy` or `sleep`).
    repeat_idx : int
        One-based repeat index.

    Returns
    -------
    dict[str, Any]
        Attempt record suitable for JSON/CSV outputs.
    """

    _ensure_src_import_path()
    if serial is None:
        raise RuntimeError("pyserial is required to run baseline load attempts.")
    from tinyodom import hil_protocol  # type: ignore
    from tinyodom.microcontrollers import arduino_base  # type: ignore
    from tinyodom.microcontrollers.arduino_portenta_h7 import (  # type: ignore
        ArduinoPortentaH7Device,
    )

    timestamp_utc = datetime.now(timezone.utc).isoformat()
    attempt: Dict[str, Any] = {
        "timestamp_utc": timestamp_utc,
        "core": core,
        "workload": workload,
        "repeat": repeat_idx,
        "runs": -1,
        "latency_ms_per_iter": -1.0,
        "energy_mj_per_iter": -1.0,
        "avg_power_mw": -1.0,
        "avg_current_ma": -1.0,
        "bus_voltage_v": -1.0,
        "idle_power_mw": -1.0,
        "error_code": 0,
        "error_label": ERROR_LABELS[0],
        "runtime_mode": "unknown",
        "board_options": "",
    }

    sketch_dir = _stage_sketch(workload)
    device = ArduinoPortentaH7Device(
        serial_port=settings.dut_port,
        device_options={"target_core": core},
    )
    runtime_mode = device.runtime_measure_mode()
    board_options = device.resolved_options.to_board_options()
    attempt["runtime_mode"] = runtime_mode
    attempt["board_options"] = ",".join(f"{k}={v}" for k, v in sorted(board_options.items()))

    try:
        device.prepare_for_runtime(runtime_mode=runtime_mode, serial_port=settings.dut_port)
    except Exception as exc:
        logging.error("Runtime preparation failed (core=%s workload=%s): %s", core, workload, exc)
        attempt["error_code"] = 5
        attempt["error_label"] = ERROR_LABELS[5]
        return attempt

    harness_defines = _build_harness_defines(settings)
    try:
        arduino_base.ensure_harness_firmware(
            harness_serial_port=settings.harness_port,
            harness_fqbn=settings.harness_fqbn,
            harness_auto_flash=settings.harness_auto_flash,
            build_defines=harness_defines,
        )
    except Exception as exc:
        logging.error("Harness flash/compile failed: %s", exc)
        # Distinguish likely compile vs upload failures based on exception message.
        msg = str(exc).lower()
        if any(keyword in msg for keyword in ("compile", "compilation", "build")):
            attempt["error_code"] = 1
            attempt["error_label"] = ERROR_LABELS[1]
        else:
            attempt["error_code"] = 2
            attempt["error_label"] = ERROR_LABELS[2]
        return attempt

    dut_defines = _build_dut_defines(settings, device.runtime_mode_build_defines())
    compile_result = arduino_base.compile_sketch(
        sketch_path=sketch_dir,
        fqbn=PORTENTA_FQBN,
        build_defines=dut_defines,
        board_options=board_options,
    )
    if not compile_result.success or compile_result.build_dir is None:
        logging.error("DUT compile failed (core=%s workload=%s).", core, workload)
        attempt["error_code"] = 1
        attempt["error_label"] = ERROR_LABELS[1]
        return attempt

    try:
        with serial.Serial(settings.harness_port, baudrate=115200, timeout=0.1) as harness:
            prime_result = hil_protocol.prime_harness_session(
                harness=harness,
                harness_ready_timeout_s=settings.harness_ready_timeout_s,
                harness_log=[],
                flush_input=True,
            )
            if not prime_result.harness_ready:
                logging.error("Harness did not become ready (core=%s workload=%s).", core, workload)
                attempt["error_code"] = 3
                attempt["error_label"] = ERROR_LABELS[3]
                return attempt

            upload_result = arduino_base.upload_sketch(
                sketch_path=sketch_dir,
                fqbn=PORTENTA_FQBN,
                build_dir=compile_result.build_dir,
                serial_port=settings.dut_port,
                board_options=board_options,
            )
            if not upload_result.success:
                logging.error("DUT upload failed (core=%s workload=%s).", core, workload)
                attempt["error_code"] = 2
                attempt["error_label"] = ERROR_LABELS[2]
                return attempt

            done_result = hil_protocol.wait_for_harness_done(
                harness=harness,
                harness_active_timeout_s=settings.harness_active_timeout_s,
                harness_done_timeout_s=settings.harness_done_timeout_s,
                harness_log=prime_result.harness_log,
            )

    except serial.SerialException as exc:
        logging.error("Serial failure during attempt (core=%s workload=%s): %s", core, workload, exc)
        attempt["error_code"] = 6
        attempt["error_label"] = ERROR_LABELS[6]
        return attempt

    if not done_result.harness_done:
        logging.error("Harness DONE timeout (core=%s workload=%s).", core, workload)
        attempt["error_code"] = 4
        attempt["error_label"] = ERROR_LABELS[4]
        attempt["runs"] = _parse_runs(done_result.harness_log)
        return attempt

    power_metrics = arduino_base._parse_power_metrics(done_result.harness_log)
    if power_metrics is None:
        logging.error("Unable to parse harness telemetry (core=%s workload=%s).", core, workload)
        attempt["error_code"] = 7
        attempt["error_label"] = ERROR_LABELS[7]
        attempt["runs"] = _parse_runs(done_result.harness_log)
        return attempt

    normalized = arduino_base.normalize_power_metrics(power_metrics)
    attempt["runs"] = done_result.runs_harness if done_result.runs_harness is not None else _parse_runs(done_result.harness_log)
    harness_latency_s = _safe_float(normalized.get("harness_latency_s", -1.0))
    attempt["latency_ms_per_iter"] = harness_latency_s * 1000.0 if harness_latency_s >= 0 else -1.0
    attempt["energy_mj_per_iter"] = _safe_float(normalized.get("energy_mj_per_inference", -1.0))
    attempt["avg_power_mw"] = _safe_float(normalized.get("avg_power_mw", -1.0))
    attempt["avg_current_ma"] = _safe_float(normalized.get("avg_current_ma", -1.0))
    attempt["bus_voltage_v"] = _safe_float(normalized.get("bus_voltage_v", -1.0))
    attempt["idle_power_mw"] = _safe_float(normalized.get("idle_power_mw", -1.0))
    return attempt


def _valid_values(attempts: Iterable[Dict[str, Any]], key: str) -> list[float]:
    """Collect valid non-negative numeric values for one metric key.

    Parameters
    ----------
    attempts : Iterable[dict[str, Any]]
        Attempt records.
    key : str
        Metric key.

    Returns
    -------
    list[float]
        Filtered numeric values.
    """

    values: list[float] = []
    for attempt in attempts:
        value = _safe_float(attempt.get(key, -1))
        if value >= 0:
            values.append(value)
    return values


def _summarize_group(core: str, workload: str, attempts: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate statistics for one ``core x workload`` group.

    Parameters
    ----------
    core : str
        Core label.
    workload : str
        Workload label.
    attempts : list[dict[str, Any]]
        Group attempt rows.

    Returns
    -------
    dict[str, Any]
        Aggregate summary dictionary.
    """

    summary: Dict[str, Any] = {
        "core": core,
        "workload": workload,
        "attempt_count": len(attempts),
        "success_count": sum(1 for attempt in attempts if int(attempt.get("error_code", -1)) == 0),
        "failure_count": sum(1 for attempt in attempts if int(attempt.get("error_code", -1)) != 0),
    }
    for key in AGG_NUMERIC_FIELDS:
        values = _valid_values(attempts, key)
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


def _write_csv(path: Path, attempts: list[Dict[str, Any]]) -> None:
    """Write attempt rows to CSV.

    Parameters
    ----------
    path : Path
        CSV output path.
    attempts : list[dict[str, Any]]
        Attempt rows.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for attempt in attempts:
            writer.writerow({column: attempt.get(column, "") for column in CSV_COLUMNS})


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Run Portenta H7 baseline load tests (heavy vs sleep) over CM7/CM4 "
            "using harness timing/energy telemetry."
        )
    )
    parser.add_argument("--config", default=str(SRC_DIR / "config" / "nas_config_stm32.yaml"), help="TinyODOM config path (optional defaults source).")
    parser.add_argument("--dut-port", default=None, help="DUT serial port (default from config or /dev/ttyACM0).")
    parser.add_argument("--harness-port", default=None, help="Harness serial port (default from config or /dev/ttyACM1).")
    parser.add_argument("--harness-fqbn", default=None, help="Harness board FQBN (default from config or arduino:mbed_nano:nano33ble).")
    parser.add_argument("--harness-auto-flash", default=None, help="Harness flash policy: once, always, never.")
    parser.add_argument("--skip-harness-flash", action="store_true", help="Set harness flash policy to never for this run.")
    parser.add_argument("--harness-arm-pin", type=int, default=None, help="Harness arm pin (default from config or 3).")
    parser.add_argument("--harness-trigger-pin", type=int, default=None, help="Harness trigger pin (default from config or 2).")
    parser.add_argument("--dut-arm-hold-ms", type=int, default=None, help="DUT arm-hold duration in ms.")
    parser.add_argument("--harness-stable-low-ms", type=int, default=None, help="Harness stable-low arming window in ms.")
    parser.add_argument("--harness-ready-timeout-s", type=float, default=None, help="Harness ready timeout in seconds.")
    parser.add_argument("--harness-arm-timeout-s", type=float, default=None, help="Harness arm timeout in seconds.")
    parser.add_argument("--harness-active-timeout-s", type=float, default=None, help="Harness active timeout in seconds.")
    parser.add_argument("--harness-done-timeout-s", type=float, default=None, help="Harness done timeout in seconds.")
    parser.add_argument(
        "--loops",
        type=int,
        default=None,
        help="Number of loops per core x workload (defaults to --repeats value, default 1).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Backward-compatible alias for --loops (default: 1).",
    )
    parser.add_argument("--cores", nargs="+", default=list(VALID_CORES), help="Target cores to run (cm7 cm4).")
    parser.add_argument("--output-json", default=None, help="JSON output path (default: analysis_scripts/portenta_baseline_load/results/...).")
    parser.add_argument("--output-csv", default=None, help="CSV output path (default: analysis_scripts/portenta_baseline_load/results/...).")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")
    return parser


def main() -> int:
    """Run the baseline load matrix and export results.

    Returns
    -------
    int
        Exit code (0 on completion).
    """

    parser = _build_parser()
    args = parser.parse_args()
    _configure_logging(args.log_level)
    if serial is None:
        raise RuntimeError(
            "pyserial is not available in this environment. Install it before running this script."
        )
    settings = _resolve_settings(args)

    logging.info("DUT port: %s | Harness port: %s", settings.dut_port, settings.harness_port)
    logging.info("Cores: %s | repeats: %d", settings.cores, settings.repeats)

    attempts: list[Dict[str, Any]] = []
    for core in settings.cores:
        for workload in VALID_WORKLOADS:
            for repeat_idx in range(1, settings.repeats + 1):
                logging.info("Running attempt: core=%s workload=%s repeat=%d", core, workload, repeat_idx)
                attempt = _run_attempt(
                    settings=settings,
                    core=core,
                    workload=workload,
                    repeat_idx=repeat_idx,
                )
                attempts.append(attempt)
                logging.info(
                    "Attempt done: core=%s workload=%s repeat=%d error=%s latency_ms=%.3f energy_mJ=%.3f",
                    core,
                    workload,
                    repeat_idx,
                    attempt["error_label"],
                    _safe_float(attempt["latency_ms_per_iter"]),
                    _safe_float(attempt["energy_mj_per_iter"]),
                )

    grouped: dict[tuple[str, str], list[Dict[str, Any]]] = defaultdict(list)
    for attempt in attempts:
        grouped[(str(attempt["core"]), str(attempt["workload"]))].append(attempt)
    aggregates = [
        _summarize_group(core, workload, rows)
        for (core, workload), rows in sorted(grouped.items())
    ]

    payload = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "config_path": str(settings.config_path),
            "dut_port": settings.dut_port,
            "harness_port": settings.harness_port,
            "harness_fqbn": settings.harness_fqbn,
            "harness_auto_flash": settings.harness_auto_flash,
            "cores": settings.cores,
            "workloads": list(VALID_WORKLOADS),
            "repeats": settings.repeats,
            "runs_per_window": RUNS_PER_WINDOW,
            "iteration_budget_ms": 200,
        },
        "attempts": attempts,
        "aggregates": aggregates,
    }

    settings.output_json.parent.mkdir(parents=True, exist_ok=True)
    settings.output_json.write_text(json.dumps(payload, indent=2))
    _write_csv(settings.output_csv, attempts)
    logging.info("Wrote JSON: %s", settings.output_json)
    logging.info("Wrote CSV: %s", settings.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
