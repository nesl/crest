#!/usr/bin/env python3
"""Run synthetic MCU workload energy probes through the CREST HIL harness."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import shutil
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

try:
    import serial  # type: ignore
except Exception:  # pragma: no cover - import availability depends on runtime env
    serial = None  # type: ignore

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is a declared project dependency
    tqdm = None  # type: ignore

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"

VALID_BOARDS = ("stm32", "portenta_m4", "portenta_m7", "ble")
VALID_WORKLOADS = ("sleep", "wait", "poll", "float", "int")
WORKLOAD_MODE = {"sleep": 0, "wait": 1, "poll": 2, "float": 3, "int": 4}
ARDUINO_SKETCH_NAME = "micro_workload_probe"
ARDUINO_TEMPLATE = SCRIPT_DIR / "micro_workload_probe.ino"
STM32_RUNNER_TEMPLATE = SCRIPT_DIR / "stm32_synthetic_dut_runner.c"
DEFAULT_CONFIG_PATH = SRC_DIR / "config" / "nas_config.yaml"
DEFAULT_BAUD = 115200
DEFAULT_WINDOW_MS = 1000
DEFAULT_HARNESS_FQBN = "arduino:mbed_nano:nano33ble"
DEFAULT_HARNESS_AUTO_FLASH = "once"
DEFAULT_DUT_PORT = "/dev/ttyACM0"
DEFAULT_HARNESS_PORT = "/dev/ttyACM1"
DEFAULT_STM32_CPU_CLOCK_MHZ = 600
DEFAULT_STM32_APPLI_FLASH_ADDRESS = "0x70100000"
DEFAULT_STM32_SIGNING_HEADER_VERSION = "2.3"
DEFAULT_STM32_SIGNING_LOAD_OFFSET = "0x80000000"
DEFAULT_STM32_EXTERNAL_LOADER_NAME = "MX25UM51245G_STM32N6570-NUCLEO.stldr"

ERROR_LABELS = {
    0: "ok",
    1: "compile_failed",
    2: "upload_failed",
    3: "harness_ready_timeout",
    4: "harness_done_timeout",
    5: "runtime_prepare_failed",
    6: "serial_error",
    7: "telemetry_parse_failed",
    8: "stm32_build_failed",
    9: "stm32_program_failed",
    10: "invalid_configuration",
    11: "window_duration_invalid",
}

CSV_COLUMNS = [
    "timestamp_utc",
    "board",
    "workload",
    "repeat",
    "requested_window_ms",
    "measured_harness_window_ms",
    "energy_mj_per_window",
    "avg_power_mw",
    "avg_current_ma",
    "bus_voltage_v",
    "idle_baseline_mw",
    "dut_iterations",
    "dut_work_units",
    "dut_work_unit_label",
    "dut_elapsed_us",
    "dut_cycles",
    "dut_sleep_ms",
    "dut_sleep_mode",
    "error_code",
    "error_label",
    "serial_log_path",
    "build_metadata",
]

AGG_NUMERIC_FIELDS = (
    "energy_mj_per_window",
    "avg_power_mw",
    "measured_harness_window_ms",
    "dut_iterations",
    "dut_work_units",
    "dut_elapsed_us",
    "dut_cycles",
    "dut_sleep_ms",
)

DERIVED_AGG_FIELDS = (
    "energy_over_sleep_mj_mean",
    "energy_over_wait_mj_mean",
    "energy_over_poll_mj_mean",
    "energy_per_work_unit_nj_mean",
    "payload_energy_per_work_unit_nj_mean",
)

DUT_TELEMETRY_DEFAULTS = {
    "dut_iterations": -1,
    "dut_work_units": -1,
    "dut_work_unit_label": "",
    "dut_elapsed_us": -1,
    "dut_cycles": -1,
    "dut_sleep_ms": -1.0,
    "dut_sleep_mode": "",
}

DUT_INT_PATTERNS = {
    "dut_iterations": re.compile(r"^dut iterations output:\s*(?P<value>-?\d+)\s*$", re.IGNORECASE),
    "dut_work_units": re.compile(r"^dut work units output:\s*(?P<value>-?\d+)\s*$", re.IGNORECASE),
    "dut_elapsed_us": re.compile(r"^dut elapsed us output:\s*(?P<value>-?\d+)\s*$", re.IGNORECASE),
    "dut_cycles": re.compile(r"^dut cycles output:\s*(?P<value>-?\d+)\s*$", re.IGNORECASE),
}

DUT_FLOAT_PATTERNS = {
    "dut_sleep_ms": re.compile(
        r"^dut sleep ms output:\s*(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*$",
        re.IGNORECASE,
    ),
}

DUT_STRING_PATTERNS = {
    "dut_work_unit_label": re.compile(r"^dut work unit label output:\s*(?P<value>\S+)\s*$", re.IGNORECASE),
    "dut_sleep_mode": re.compile(r"^dut sleep mode output:\s*(?P<value>\S+)\s*$", re.IGNORECASE),
}

REQUIRED_POWER_METRICS = (
    "harness_latency_s",
    "energy_mj_per_inference",
    "avg_power_mw",
    "avg_current_ma",
    "bus_voltage_v",
    "idle_power_mw",
)

AGG_CSV_COLUMNS = [
    "board",
    "workload",
    "attempt_count",
    "success_count",
    "failure_count",
]
for _field in AGG_NUMERIC_FIELDS:
    AGG_CSV_COLUMNS.extend(
        [
            f"{_field}_mean",
            f"{_field}_std",
            f"{_field}_min",
            f"{_field}_max",
            f"{_field}_n",
        ]
    )
AGG_CSV_COLUMNS.extend(DERIVED_AGG_FIELDS)
AGG_CSV_COLUMNS.append("aggregate_warning")


@dataclass(frozen=True)
class BoardSpec:
    """Resolved board-token behavior for one target.

    Parameters
    ----------
    token : str
        User-facing board token.
    family : str
        Runner family, either ``arduino`` or ``stm32``.
    fqbn : str | None
        Arduino FQBN when the target uses Arduino CLI.
    target_core : str | None
        Portenta target core when applicable.

    Attributes
    ----------
    token : str
        Token identifying the workload in generated artifacts.
    family : str
        Microcontroller family for the board target.
    fqbn : str | None
        Fully qualified board name used by Arduino tooling.
    target_core : str | None
        Target processor core for the workload.
    """

    token: str
    family: str
    fqbn: str | None
    target_core: str | None = None


@dataclass(frozen=True)
class RuntimeSettings:
    """Fully resolved runner settings.

    Parameters
    ----------
    config_path : Path
        Config path used only as an optional source of defaults.
    boards : list[BoardSpec]
        Board targets in execution order.
    workloads : list[str]
        Workload names in execution order.
    repeats : int
        Number of attempts per board/workload pair.
    window_ms : int
        Requested active measurement window in milliseconds.
    dut_port : str
        DUT serial port.
    harness_port : str
        Harness serial port.
    harness_fqbn : str
        Harness Arduino FQBN.
    harness_auto_flash : str
        Harness flash policy.
    harness_arm_pin : int
        Harness-side arm pin.
    harness_trigger_pin : int
        Harness-side trigger pin.
    dut_arm_hold_ms : int
        DUT arm-hold delay before trigger high.
    harness_stable_low_ms : int
        Harness stable-low arming window.
    harness_ready_timeout_s : float
        Timeout waiting for harness ready.
    harness_arm_timeout_s : float
        Harness firmware arm timeout.
    harness_active_timeout_s : float
        Harness active-window timeout.
    harness_done_timeout_s : float
        Host wait after active phase.
    baud_rate : int
        Serial baud rate.
    output_json : Path
        JSON output path.
    output_csv : Path
        CSV output path.
    log_dir : Path
        Directory for serial/build diagnostics.
    stm32_stage_root : Path
        Root for generated STM32 staging workspaces.
    run_tag : str
        Stable tag for generated artifacts within one runner invocation.
    stm32_project_root : Path
        Production STM32 LRUN template root.
    stm32_cpu_clock_mhz : int
        STM32 CPU clock preset.
    stm32_wake_margin_us : int
        STM32 wake margin for sleep workload.
    stm32_min_sleep_us : int
        STM32 minimum STOP sleep request.
    stm32_jobs : int
        STM32 build parallelism.
    stm32_appli_flash_address : str
        External flash address for signed app image.
    stm32_cubeprog_bin : Path | None
        Optional STM32CubeProgrammer bin override.
    stm32_gdbserver : Path | None
        Optional ST-LINK GDB server override.
    stm32_gdb : Path | None
        Optional GDB override.
    stm32_gdb_port : int
        GDB server TCP port.
    stm32_apid : int
        STM32 access-port identifier.
    stm32_server_ready_timeout_s : float
        GDB server ready timeout.
    stm32_signing_tool : Path | None
        Optional signing-tool override.
    stm32_signing_header_version : str
        Signing header version.
    stm32_signing_load_offset : str
        Signing load offset.

    Attributes
    ----------
    config_path : Path
        Path to the workload configuration file.
    boards : list[BoardSpec]
        Board configurations included in the workload run.
    workloads : list[str]
        Workload definitions scheduled for measurement.
    repeats : int
        Number of repeated measurements per workload.
    window_ms : int
        Measurement window length in milliseconds.
    dut_port : str
        Serial port connected to the device under test.
    harness_port : str
        Serial port connected to the measurement harness.
    harness_fqbn : str
        Fully qualified board name for the measurement harness.
    harness_auto_flash : str
        Whether the harness firmware should be flashed automatically.
    harness_arm_pin : int
        GPIO pin used to arm the measurement harness.
    harness_trigger_pin : int
        GPIO pin used to trigger the measurement harness.
    dut_arm_hold_ms : int
        Hold time in milliseconds before arming the DUT.
    harness_stable_low_ms : int
        Stable-low interval in milliseconds before triggering.
    harness_ready_timeout_s : float
        Timeout in seconds while waiting for harness readiness.
    harness_arm_timeout_s : float
        Timeout in seconds while arming the harness.
    harness_active_timeout_s : float
        Timeout in seconds while waiting for active harness state.
    harness_done_timeout_s : float
        Timeout in seconds while waiting for harness completion.
    baud_rate : int
        Serial baud rate for board and harness connections.
    output_json : Path
        JSON output path for workload measurements.
    output_csv : Path
        CSV output path for workload measurements.
    log_dir : Path
        Directory where workload logs are written.
    stm32_stage_root : Path
        Directory used for staged STM32 build artifacts.
    run_tag : str
        Tag used to identify the workload run.
    stm32_project_root : Path
        Root directory of the STM32 project.
    stm32_cpu_clock_mhz : int
        STM32 CPU clock frequency in MHz.
    stm32_wake_margin_us : int
        Wake margin in microseconds for STM32 workload timing.
    stm32_min_sleep_us : int
        Minimum STM32 sleep interval in microseconds.
    stm32_jobs : int
        Number of parallel jobs used for STM32 builds.
    stm32_appli_flash_address : str
        Flash address where the STM32 application image is loaded.
    stm32_cubeprog_bin : Path | None
        Path to the STM32CubeProgrammer executable.
    stm32_gdbserver : Path | None
        Path to the STM32 GDB server executable.
    stm32_gdb : Path | None
        Path to the GDB executable used for STM32 debugging.
    stm32_gdb_port : int
        TCP port used by the STM32 GDB server.
    stm32_apid : int
        STM32 access port identifier used by debug tooling.
    stm32_server_ready_timeout_s : float
        Timeout in seconds while waiting for the STM32 debug server.
    stm32_signing_tool : Path | None
        Path to the STM32 signing tool executable.
    stm32_signing_header_version : str
        Header version passed to the STM32 signing tool.
    stm32_signing_load_offset : str
        Load offset passed to the STM32 signing tool.
    """

    config_path: Path
    boards: list[BoardSpec]
    workloads: list[str]
    repeats: int
    window_ms: int
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
    baud_rate: int
    output_json: Path
    output_csv: Path
    log_dir: Path
    stm32_stage_root: Path
    run_tag: str
    stm32_project_root: Path
    stm32_cpu_clock_mhz: int
    stm32_wake_margin_us: int
    stm32_min_sleep_us: int
    stm32_jobs: int
    stm32_appli_flash_address: str
    stm32_cubeprog_bin: Path | None
    stm32_gdbserver: Path | None
    stm32_gdb: Path | None
    stm32_gdb_port: int
    stm32_apid: int
    stm32_server_ready_timeout_s: float
    stm32_signing_tool: Path | None
    stm32_signing_header_version: str
    stm32_signing_load_offset: str


def ensure_import_paths() -> None:
    """Ensure repository helper modules are importable.

    Returns
    -------
    None
        ``sys.path`` is updated in place when required.
    """
    for path in (str(SRC_DIR),):
        if path not in sys.path:
            sys.path.insert(0, path)


def configure_logging(level_name: str) -> None:
    """Configure process logging.

    Parameters
    ----------
    level_name : str
        Logging level name such as ``INFO`` or ``DEBUG``.
    """
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _which_path(name: str) -> Path | None:
    """Resolve a command on ``PATH``.

    Parameters
    ----------
    name : str
        Executable name.

    Returns
    -------
    Path | None
        Resolved path or ``None``.
    """
    resolved = shutil.which(name)
    return Path(resolved).resolve() if resolved else None


def _optional_path(value: object | None) -> Path | None:
    """Normalize an optional path argument.

    Parameters
    ----------
    value : object | None
        Optional path-like value.

    Returns
    -------
    Path | None
        Expanded path when provided.
    """
    if value in (None, ""):
        return None
    return Path(str(value)).expanduser().resolve()


def safe_float(value: Any, default: float = -1.0) -> float:
    """Convert a value to a finite float.

    Parameters
    ----------
    value : Any
        Candidate scalar.
    default : float, default=-1.0
        Fallback for missing or invalid values.

    Returns
    -------
    float
        Finite value or ``default``.
    """
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if math.isfinite(parsed) else float(default)


def safe_int(value: Any, default: int = -1) -> int:
    """Convert a value to an integer.

    Parameters
    ----------
    value : Any
        Candidate scalar.
    default : int, default=-1
        Fallback for missing or invalid values.

    Returns
    -------
    int
        Parsed integer or ``default``.
    """
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return int(default)


def _strip_log_prefix(line: str) -> str:
    """Remove optional stored-log stream prefixes from one telemetry line.

    Parameters
    ----------
    line : str
        Raw line from a live serial stream or stored diagnostic log.

    Returns
    -------
    str
        Line with a leading ``DUT:`` or ``HARNESS:`` marker removed.
    """
    text = str(line).strip()
    for prefix in ("DUT: ", "HARNESS: "):
        if text.startswith(prefix):
            return text[len(prefix):].strip()
    return text


def parse_dut_telemetry(lines: Sequence[str]) -> dict[str, Any]:
    """Parse DUT telemetry emitted after the measured trigger window.

    Parameters
    ----------
    lines : Sequence[str]
        DUT serial lines, optionally prefixed with stored-log stream labels.

    Returns
    -------
    dict[str, Any]
        Telemetry fields filled from the log, with missing fields set to
        ``DUT_TELEMETRY_DEFAULTS``.
    """
    telemetry: dict[str, Any] = dict(DUT_TELEMETRY_DEFAULTS)
    for raw_line in lines:
        line = _strip_log_prefix(str(raw_line))
        for key, pattern in DUT_INT_PATTERNS.items():
            match = pattern.match(line)
            if match:
                telemetry[key] = safe_int(match.group("value"))
        for key, pattern in DUT_FLOAT_PATTERNS.items():
            match = pattern.match(line)
            if match:
                telemetry[key] = safe_float(match.group("value"))
        for key, pattern in DUT_STRING_PATTERNS.items():
            match = pattern.match(line)
            if match:
                telemetry[key] = match.group("value")
    return telemetry


def fill_dut_telemetry(attempt: dict[str, Any], lines: Sequence[str]) -> None:
    """Copy parsed DUT-side telemetry into an attempt row.

    Parameters
    ----------
    attempt : dict[str, Any]
        Attempt row to update in place.
    lines : Sequence[str]
        DUT serial lines to parse.

    Returns
    -------
    None
        ``attempt`` is mutated in place.
    """
    attempt.update(parse_dut_telemetry(lines))


def direct_dut_telemetry_valid(attempt: Mapping[str, Any]) -> bool:
    """Return whether a direct-serial attempt has usable DUT telemetry.

    Parameters
    ----------
    attempt : Mapping[str, Any]
        Attempt row with parsed DUT telemetry fields.

    Returns
    -------
    bool
        True when required workload-specific telemetry is present.
    """
    workload = str(attempt.get("workload", ""))
    if safe_int(attempt.get("dut_iterations")) < 0:
        return False
    if safe_int(attempt.get("dut_elapsed_us")) <= 0:
        return False
    if workload in {"wait", "poll", "float", "int"}:
        if safe_int(attempt.get("dut_work_units")) <= 0:
            return False
        if not str(attempt.get("dut_work_unit_label", "")).strip():
            return False
    if workload == "sleep":
        if safe_float(attempt.get("dut_sleep_ms")) < 0.0:
            return False
        if not str(attempt.get("dut_sleep_mode", "")).strip():
            return False
    return True


def validate_direct_dut_telemetry(attempt: dict[str, Any], *, direct_serial: bool) -> None:
    """Fail direct-serial attempts that completed but lost DUT telemetry.

    Parameters
    ----------
    attempt : dict[str, Any]
        Attempt row to update in place.
    direct_serial : bool
        Whether the target runtime exposes a DUT serial stream.

    Returns
    -------
    None
        ``attempt`` is mutated only when telemetry validation fails.
    """
    if direct_serial and int(attempt.get("error_code", -1)) == 0 and not direct_dut_telemetry_valid(attempt):
        set_attempt_error(attempt, 7)


def validate_harness_window_duration(attempt: dict[str, Any], requested_window_ms: int) -> None:
    """Fail attempts whose measured trigger-high window is implausible.

    Parameters
    ----------
    attempt : dict[str, Any]
        Attempt row to update in place.
    requested_window_ms : int
        Requested trigger-high window duration.

    Returns
    -------
    None
        ``attempt`` is mutated only when the measured window is outside the
        allowed tolerance.
    """
    if int(attempt.get("error_code", -1)) != 0:
        return
    measured_ms = safe_float(attempt.get("measured_harness_window_ms"))
    requested_ms = float(requested_window_ms)
    lower_ms = max(1.0, requested_ms * 0.80)
    upper_ms = requested_ms * 1.20
    if measured_ms < lower_ms or measured_ms > upper_ms:
        set_attempt_error(attempt, 11)


def load_config_defaults(config_path: Path) -> dict[str, Any]:
    """Load optional CREST device defaults.

    Parameters
    ----------
    config_path : Path
        Candidate CREST config path.

    Returns
    -------
    dict[str, Any]
        Default values found in config, or an empty mapping.
    """
    ensure_import_paths()
    try:
        from crest.model import load_config  # type: ignore
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "Unable to import crest.model.load_config; using CLI defaults (%s).",
            exc,
        )
        return {}
    try:
        cfg = load_config(config_path)
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "Unable to load config %s; using CLI defaults (%s).",
            config_path,
            exc,
        )
        return {}

    device = dict(getattr(cfg, "device", {}) or {})
    stm32 = dict(device.get("stm32") or {})
    return {
        "dut_port": device.get("serial_port"),
        "harness_port": device.get("harness_serial_port"),
        "harness_fqbn": device.get("harness_fqbn"),
        "harness_auto_flash": device.get("harness_auto_flash"),
        "harness_arm_pin": device.get("harness_arm_pin"),
        "harness_trigger_pin": device.get("harness_trigger_pin"),
        "dut_arm_hold_ms": device.get("dut_arm_hold_ms"),
        "harness_stable_low_ms": device.get("harness_stable_low_ms"),
        "harness_ready_timeout_s": device.get("harness_ready_timeout_s"),
        "harness_arm_timeout_s": device.get("harness_arm_timeout_s"),
        "harness_active_timeout_s": device.get("harness_active_timeout_s"),
        "harness_done_timeout_s": device.get("harness_done_timeout_s"),
        "stm32_project_root": stm32.get("project_root"),
        "stm32_cpu_clock_mhz": device.get("cpu_clock_mhz", stm32.get("cpu_clock_mhz")),
        "stm32_wake_margin_us": stm32.get("wake_margin_us"),
        "stm32_min_sleep_us": stm32.get("min_sleep_us"),
        "stm32_cubeprog_bin": stm32.get("cubeprog_bin"),
        "stm32_gdbserver": stm32.get("gdbserver"),
        "stm32_gdb": stm32.get("gdb"),
        "stm32_gdb_port": stm32.get("gdb_port"),
        "stm32_apid": stm32.get("apid"),
        "stm32_server_ready_timeout_s": stm32.get("server_ready_timeout_s"),
        "stm32_signing_tool": stm32.get("signing_tool"),
        "stm32_signing_header_version": stm32.get("signing_header_version"),
        "stm32_signing_load_offset": stm32.get("signing_load_offset"),
    }


def resolve_board_specs(board_tokens: Sequence[str]) -> list[BoardSpec]:
    """Resolve user-facing board tokens into backend behavior.

    Parameters
    ----------
    board_tokens : Sequence[str]
        Requested board tokens.

    Returns
    -------
    list[BoardSpec]
        Resolved board specifications.

    Raises
    ------
    ValueError
        If an unsupported token is present.
    """
    specs: list[BoardSpec] = []
    for raw_token in board_tokens:
        token = str(raw_token).strip().lower()
        if token == "stm32":
            specs.append(BoardSpec(token=token, family="stm32", fqbn=None))
        elif token == "portenta_m4":
            specs.append(
                BoardSpec(
                    token=token,
                    family="arduino",
                    fqbn="arduino:mbed_portenta:envie_m7",
                    target_core="cm4",
                )
            )
        elif token == "portenta_m7":
            specs.append(
                BoardSpec(
                    token=token,
                    family="arduino",
                    fqbn="arduino:mbed_portenta:envie_m7",
                    target_core="cm7",
                )
            )
        elif token == "ble":
            specs.append(
                BoardSpec(
                    token=token,
                    family="arduino",
                    fqbn="arduino:mbed_nano:nano33ble",
                )
            )
        else:
            raise ValueError(f"Invalid board '{raw_token}'. Expected one of {VALID_BOARDS}.")
    return specs


def validate_workloads(workloads: Sequence[str]) -> list[str]:
    """Validate and normalize workload names.

    Parameters
    ----------
    workloads : Sequence[str]
        Requested workload names.

    Returns
    -------
    list[str]
        Normalized workload names.

    Raises
    ------
    ValueError
        If existing validation or execution checks fail.
    """
    normalized = [str(workload).strip().lower() for workload in workloads]
    invalid = [workload for workload in normalized if workload not in VALID_WORKLOADS]
    if invalid:
        raise ValueError(f"Invalid workloads {invalid}. Expected subset of {VALID_WORKLOADS}.")
    return normalized


def resolve_output_path(raw_value: str | None, default_path: Path) -> Path:
    """Resolve an output path, accepting either a file path or directory path.

    Parameters
    ----------
    raw_value : str | None
        User-provided path string. A missing value uses ``default_path``.
    default_path : Path
        Timestamped default file path, used directly or as the filename when
        ``raw_value`` points to a directory.

    Returns
    -------
    Path
        Absolute output file path.
    """
    if not raw_value:
        return default_path
    raw_text = str(raw_value)
    candidate = Path(raw_text).expanduser().resolve()
    if raw_text.endswith(("/", "\\")) or candidate.is_dir():
        return candidate / default_path.name
    return candidate


def resolve_settings(args: argparse.Namespace) -> RuntimeSettings:
    """Resolve runtime settings from CLI arguments and config defaults.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    RuntimeSettings
        Fully normalized settings.

    Raises
    ------
    ValueError
        If existing validation or execution checks fail.
    """
    config_path = Path(args.config).expanduser().resolve()
    defaults = load_config_defaults(config_path)
    timestamp_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    default_stem = SCRIPT_DIR / "results" / f"micro_workload_energy_probe_{timestamp_tag}"
    output_json = resolve_output_path(args.output_json, default_stem.with_suffix(".json"))
    output_csv = resolve_output_path(args.output_csv, default_stem.with_suffix(".csv"))
    harness_auto_flash = "never" if args.skip_harness_flash else str(
        args.harness_auto_flash or defaults.get("harness_auto_flash") or DEFAULT_HARNESS_AUTO_FLASH
    ).lower()
    if harness_auto_flash not in {"once", "always", "never"}:
        raise ValueError("--harness-auto-flash must be one of: once, always, never.")
    window_ms = int(args.window_ms)
    if window_ms < 1:
        raise ValueError("--window-ms must be >= 1.")
    repeats = int(args.repeats)
    if repeats < 1:
        raise ValueError("--repeats must be >= 1.")

    active_timeout_default = max(5.0, (float(window_ms) / 1000.0) + 5.0)
    stm32_project_default = REPO_ROOT / "sketches" / "stm32" / "crest_stm32_lrun"
    return RuntimeSettings(
        config_path=config_path,
        boards=resolve_board_specs(args.boards),
        workloads=validate_workloads(args.workloads),
        repeats=repeats,
        window_ms=window_ms,
        dut_port=str(args.dut_port or defaults.get("dut_port") or DEFAULT_DUT_PORT),
        harness_port=str(args.harness_port or defaults.get("harness_port") or DEFAULT_HARNESS_PORT),
        harness_fqbn=str(args.harness_fqbn or defaults.get("harness_fqbn") or DEFAULT_HARNESS_FQBN),
        harness_auto_flash=harness_auto_flash,
        harness_arm_pin=int(args.harness_arm_pin if args.harness_arm_pin is not None else defaults.get("harness_arm_pin") or 3),
        harness_trigger_pin=int(args.harness_trigger_pin if args.harness_trigger_pin is not None else defaults.get("harness_trigger_pin") or 2),
        dut_arm_hold_ms=int(args.dut_arm_hold_ms if args.dut_arm_hold_ms is not None else defaults.get("dut_arm_hold_ms") or 600),
        harness_stable_low_ms=int(args.harness_stable_low_ms if args.harness_stable_low_ms is not None else defaults.get("harness_stable_low_ms") or 500),
        harness_ready_timeout_s=float(args.harness_ready_timeout_s if args.harness_ready_timeout_s is not None else defaults.get("harness_ready_timeout_s") or 5.0),
        harness_arm_timeout_s=float(args.harness_arm_timeout_s if args.harness_arm_timeout_s is not None else defaults.get("harness_arm_timeout_s") or 0.0),
        harness_active_timeout_s=float(args.harness_active_timeout_s if args.harness_active_timeout_s is not None else defaults.get("harness_active_timeout_s") or active_timeout_default),
        harness_done_timeout_s=float(args.harness_done_timeout_s if args.harness_done_timeout_s is not None else defaults.get("harness_done_timeout_s") or 5.0),
        baud_rate=int(args.baud),
        output_json=output_json,
        output_csv=output_csv,
        log_dir=output_json.with_suffix("").parent / f"{output_json.with_suffix('').name}_logs",
        stm32_stage_root=Path(args.stm32_stage_root).expanduser().resolve(),
        run_tag=timestamp_tag,
        stm32_project_root=Path(args.stm32_project_root or defaults.get("stm32_project_root") or stm32_project_default).expanduser().resolve(),
        stm32_cpu_clock_mhz=int(args.stm32_cpu_clock_mhz or defaults.get("stm32_cpu_clock_mhz") or DEFAULT_STM32_CPU_CLOCK_MHZ),
        stm32_wake_margin_us=int(args.stm32_wake_margin_us if args.stm32_wake_margin_us is not None else defaults.get("stm32_wake_margin_us") or 5000),
        stm32_min_sleep_us=int(args.stm32_min_sleep_us if args.stm32_min_sleep_us is not None else defaults.get("stm32_min_sleep_us") or 5000),
        stm32_jobs=int(args.stm32_jobs or os.cpu_count() or 1),
        stm32_appli_flash_address=str(args.stm32_appli_flash_address or DEFAULT_STM32_APPLI_FLASH_ADDRESS),
        stm32_cubeprog_bin=_optional_path(args.stm32_cubeprog_bin or defaults.get("stm32_cubeprog_bin")),
        stm32_gdbserver=_optional_path(args.stm32_gdbserver or defaults.get("stm32_gdbserver")) or _which_path("ST-LINK_gdbserver"),
        stm32_gdb=_optional_path(args.stm32_gdb or defaults.get("stm32_gdb")) or _which_path("arm-none-eabi-gdb"),
        stm32_gdb_port=int(args.stm32_gdb_port or defaults.get("stm32_gdb_port") or 61234),
        stm32_apid=int(args.stm32_apid or defaults.get("stm32_apid") or 1),
        stm32_server_ready_timeout_s=float(args.stm32_server_ready_timeout_s or defaults.get("stm32_server_ready_timeout_s") or 10.0),
        stm32_signing_tool=_optional_path(args.stm32_signing_tool or defaults.get("stm32_signing_tool")) or _which_path("STM32_SigningTool_CLI") or _which_path("STM32TrustedPackageCreator_CLI"),
        stm32_signing_header_version=str(args.stm32_signing_header_version or defaults.get("stm32_signing_header_version") or DEFAULT_STM32_SIGNING_HEADER_VERSION),
        stm32_signing_load_offset=str(args.stm32_signing_load_offset or defaults.get("stm32_signing_load_offset") or DEFAULT_STM32_SIGNING_LOAD_OFFSET),
    )


def build_harness_defines(settings: RuntimeSettings) -> dict[str, int]:
    """Build compile-time defines for the shared HIL harness.

    Parameters
    ----------
    settings : RuntimeSettings
        Resolved runtime settings.

    Returns
    -------
    dict[str, int]
        Harness compile-time defines.
    """
    return {
        "CREST_INFERENCE_RUNS": 1,
        "CREST_HARNESS_ARM_PIN": int(settings.harness_arm_pin),
        "CREST_HARNESS_TRIGGER_PIN": int(settings.harness_trigger_pin),
        "CREST_DUT_ARM_HOLD_MS": int(settings.dut_arm_hold_ms),
        "CREST_HARNESS_STABLE_LOW_MS": int(settings.harness_stable_low_ms),
        "CREST_HARNESS_ARM_TIMEOUT_MS": max(0, int(round(settings.harness_arm_timeout_s * 1000.0))),
        "CREST_HARNESS_ACTIVE_TIMEOUT_MS": max(0, int(round(settings.harness_active_timeout_s * 1000.0))),
    }


def build_dut_defines(settings: RuntimeSettings, workload: str, extra: Mapping[str, int] | None = None) -> dict[str, int]:
    """Build compile-time defines for Arduino DUT sketches.

    Parameters
    ----------
    settings : RuntimeSettings
        Resolved runtime settings.
    workload : str
        Workload key.
    extra : Mapping[str, int] | None, optional
        Board-specific extra defines.

    Returns
    -------
    dict[str, int]
        DUT compile-time defines.
    """
    defines = {
        "CREST_AUTOSTART": 1,
        "CREST_INFERENCE_RUNS": 1,
        "CREST_HARNESS_ARM_PIN": int(settings.harness_arm_pin),
        "CREST_HARNESS_TRIGGER_PIN": int(settings.harness_trigger_pin),
        "CREST_DUT_ARM_HOLD_MS": int(settings.dut_arm_hold_ms),
        "CREST_HARNESS_STABLE_LOW_MS": int(settings.harness_stable_low_ms),
        "CREST_MICRO_WORKLOAD_MODE": int(WORKLOAD_MODE[workload]),
        "CREST_MICRO_WINDOW_MS": int(settings.window_ms),
    }
    if extra:
        defines.update({str(key): int(value) for key, value in extra.items()})
    return defines


def stage_arduino_sketch(stage_root: Path | None = None) -> Path:
    """Stage the Arduino sketch and shared headers into a compile-ready folder.

    Parameters
    ----------
    stage_root : Path | None, optional
        Optional stage root used by tests.

    Returns
    -------
    Path
        Directory containing the staged sketch.
    """
    root = stage_root or (SCRIPT_DIR / ".staged_sketch")
    stage_dir = root / ARDUINO_SKETCH_NAME
    stage_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ARDUINO_TEMPLATE, stage_dir / f"{ARDUINO_SKETCH_NAME}.ino")
    shutil.copytree(REPO_ROOT / "sketches" / "common", stage_dir / "common", dirs_exist_ok=True)
    return stage_dir


def write_serial_log(log_dir: Path, stem: str, lines: Iterable[str]) -> Path:
    """Write one serial diagnostic log.

    Parameters
    ----------
    log_dir : Path
        Output log directory.
    stem : str
        Filename stem.
    lines : Iterable[str]
        Lines to write.

    Returns
    -------
    Path
        Written log path.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{stem}.log"
    path.write_text("\n".join(str(line) for line in lines) + "\n", encoding="utf-8")
    return path


def base_attempt(settings: RuntimeSettings, board: BoardSpec, workload: str, repeat_idx: int) -> dict[str, Any]:
    """Create a default attempt row.

    Parameters
    ----------
    settings : RuntimeSettings
        Resolved settings.
    board : BoardSpec
        Board target.
    workload : str
        Workload name.
    repeat_idx : int
        One-based repeat index.

    Returns
    -------
    dict[str, Any]
        Attempt row initialized with failure sentinels.
    """
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "board": board.token,
        "workload": workload,
        "repeat": int(repeat_idx),
        "requested_window_ms": int(settings.window_ms),
        "measured_harness_window_ms": -1.0,
        "energy_mj_per_window": -1.0,
        "avg_power_mw": -1.0,
        "avg_current_ma": -1.0,
        "bus_voltage_v": -1.0,
        "idle_baseline_mw": -1.0,
        **DUT_TELEMETRY_DEFAULTS,
        "error_code": 0,
        "error_label": ERROR_LABELS[0],
        "serial_log_path": "",
        "build_metadata": "",
    }


def set_attempt_error(attempt: dict[str, Any], code: int) -> None:
    """Set an attempt error code and label in place.

    Parameters
    ----------
    attempt : dict[str, Any]
        Attempt row to update.
    code : int
        Error code.
    """
    attempt["error_code"] = int(code)
    attempt["error_label"] = ERROR_LABELS.get(int(code), f"unknown_{code}")


def required_power_metrics_available(power_metrics: Mapping[str, Any]) -> bool:
    """Return whether all required harness power fields are usable.

    Parameters
    ----------
    power_metrics : Mapping[str, Any]
        Normalized metrics from existing CREST parsers.

    Returns
    -------
    bool
        True when all required metrics are finite and non-negative.
    """
    return all(safe_float(power_metrics.get(key), default=-1.0) >= 0.0 for key in REQUIRED_POWER_METRICS)


def fill_metrics_from_power(attempt: dict[str, Any], power_metrics: Mapping[str, Any]) -> None:
    """Copy normalized harness power metrics into an attempt row.

    Parameters
    ----------
    attempt : dict[str, Any]
        Attempt row to update.
    power_metrics : Mapping[str, Any]
        Normalized metrics from existing CREST parsers.
    """
    harness_latency_s = safe_float(power_metrics.get("harness_latency_s"))
    attempt["measured_harness_window_ms"] = harness_latency_s * 1000.0 if harness_latency_s >= 0.0 else -1.0
    attempt["energy_mj_per_window"] = safe_float(power_metrics.get("energy_mj_per_inference"))
    attempt["avg_power_mw"] = safe_float(power_metrics.get("avg_power_mw"))
    attempt["avg_current_ma"] = safe_float(power_metrics.get("avg_current_ma"))
    attempt["bus_voltage_v"] = safe_float(power_metrics.get("bus_voltage_v"))
    attempt["idle_baseline_mw"] = safe_float(power_metrics.get("idle_power_mw"))


def finish_attempt_from_harness_log(
    attempt: dict[str, Any],
    *,
    harness_done: bool,
    harness_log: Sequence[str],
    parse_power_metrics: Callable[[Sequence[str]], Mapping[str, Any] | None],
    normalize_power_metrics: Callable[[Mapping[str, Any] | None], Mapping[str, Any]],
) -> None:
    """Finalize an attempt from harness DONE state and telemetry lines.

    Parameters
    ----------
    attempt : dict[str, Any]
        Attempt row to update in place.
    harness_done : bool
        Whether the harness emitted ``DONE``.
    harness_log : Sequence[str]
        Harness serial lines.
    parse_power_metrics : callable
        Existing CREST parser for raw harness telemetry.
    normalize_power_metrics : callable
        Existing CREST normalizer for parsed telemetry.
    """
    if not harness_done:
        set_attempt_error(attempt, 4)
        return
    parsed = parse_power_metrics(harness_log)
    if parsed is None:
        set_attempt_error(attempt, 7)
        return
    normalized = normalize_power_metrics(parsed)
    if not required_power_metrics_available(normalized):
        set_attempt_error(attempt, 7)
        return
    fill_metrics_from_power(attempt, normalized)


def run_arduino_attempt(settings: RuntimeSettings, board: BoardSpec, workload: str, repeat_idx: int) -> dict[str, Any]:
    """Run one Arduino-backed board/workload attempt.

    Parameters
    ----------
    settings : RuntimeSettings
        Resolved settings.
    board : BoardSpec
        Arduino board target.
    workload : str
        Workload name.
    repeat_idx : int
        One-based repeat index.

    Returns
    -------
    dict[str, Any]
        Attempt row.

    Raises
    ------
    RuntimeError
        If existing validation or execution checks fail.
    """
    ensure_import_paths()
    if serial is None:
        raise RuntimeError("pyserial is required for hardware attempts.")
    from crest import hil_protocol  # type: ignore
    from crest.microcontrollers import arduino_base  # type: ignore
    from crest.microcontrollers import stm32_runtime  # type: ignore
    from crest.microcontrollers.arduino_ble33 import ArduinoBLE33Device  # type: ignore
    from crest.microcontrollers.arduino_portenta_h7 import ArduinoPortentaH7Device  # type: ignore

    attempt = base_attempt(settings, board, workload, repeat_idx)
    log_stem = f"{board.token}_{workload}_repeat{repeat_idx:02d}"
    sketch_dir = stage_arduino_sketch()
    if board.token.startswith("portenta"):
        device = ArduinoPortentaH7Device(serial_port=settings.dut_port, device_options={"target_core": str(board.target_core)})
    else:
        device = ArduinoBLE33Device(serial_port=settings.dut_port)
    runtime_mode = device.runtime_measure_mode()
    board_options = getattr(device, "resolved_options", None)
    board_option_map = board_options.to_board_options() if board_options is not None else {}
    direct_serial = runtime_mode != "harness_only"
    attempt["build_metadata"] = ",".join(
        f"{key}={value}" for key, value in sorted({"fqbn": board.fqbn or "", **board_option_map}.items())
    )
    runtime_header = [
        f"runtime_mode: {runtime_mode}",
        f"direct_serial: {direct_serial}",
        f"build_metadata: {attempt['build_metadata']}",
    ]

    try:
        device.prepare_for_runtime(runtime_mode=runtime_mode, serial_port=settings.dut_port)
    except Exception as exc:
        set_attempt_error(attempt, 5)
        attempt["serial_log_path"] = str(write_serial_log(settings.log_dir, log_stem, [*runtime_header, f"runtime_prepare_failed: {exc}"]))
        return attempt

    try:
        arduino_base.ensure_harness_firmware(
            harness_serial_port=settings.harness_port,
            harness_fqbn=settings.harness_fqbn,
            harness_auto_flash=settings.harness_auto_flash,
            build_defines=build_harness_defines(settings),
        )
    except Exception as exc:
        set_attempt_error(attempt, 1 if "compile" in str(exc).lower() else 2)
        attempt["serial_log_path"] = str(write_serial_log(settings.log_dir, log_stem, [*runtime_header, f"harness_prepare_failed: {exc}"]))
        return attempt

    extra_defines = dict(device.runtime_mode_build_defines())
    if direct_serial:
        extra_defines["CREST_AUTOSTART"] = 0
    dut_defines = build_dut_defines(settings, workload, extra_defines)
    log_header = [
        *runtime_header,
        f"dut_build_defines: {json.dumps(dut_defines, sort_keys=True)}",
    ]
    compile_result = arduino_base.compile_sketch(
        sketch_path=sketch_dir,
        fqbn=str(board.fqbn),
        build_defines=dut_defines,
        board_options=board_option_map,
    )
    if not compile_result.success:
        set_attempt_error(attempt, 1)
        attempt["serial_log_path"] = str(write_serial_log(settings.log_dir, log_stem, [*log_header, compile_result.log]))
        return attempt

    try:
        with serial.Serial(settings.harness_port, baudrate=settings.baud_rate, timeout=0.1) as harness:
            prime_result = hil_protocol.prime_harness_session(
                harness=harness,
                harness_ready_timeout_s=settings.harness_ready_timeout_s,
                harness_log=[],
                flush_input=True,
            )
            if not prime_result.harness_ready:
                set_attempt_error(attempt, 3)
                attempt["serial_log_path"] = str(write_serial_log(settings.log_dir, log_stem, [*log_header, *prime_result.harness_log]))
                return attempt

            upload_result = arduino_base.upload_sketch(
                sketch_path=sketch_dir,
                fqbn=str(board.fqbn),
                build_dir=compile_result.build_dir,
                serial_port=settings.dut_port,
                board_options=board_option_map,
            )
            if not upload_result.success:
                set_attempt_error(attempt, 2)
                attempt["serial_log_path"] = str(write_serial_log(settings.log_dir, log_stem, [*log_header, upload_result.log]))
                return attempt

            dut_lines: list[str] = []
            if direct_serial:
                with stm32_runtime.SerialMonitor(settings.dut_port, settings.baud_rate, "arduino-dut") as monitor:
                    ready_line, cursor = monitor.wait_for_match(
                        lambda line: line.strip() == "DUT READY",
                        settings.harness_ready_timeout_s,
                        "DUT READY",
                    )
                    if ready_line is None:
                        set_attempt_error(attempt, 6)
                        dut_lines = monitor.snapshot_lines()
                        attempt["serial_log_path"] = str(
                            write_serial_log(
                                settings.log_dir,
                                log_stem,
                                [*log_header, *(f"DUT: {line}" for line in dut_lines), *(f"HARNESS: {line}" for line in prime_result.harness_log)],
                            )
                        )
                        return attempt
                    monitor.write_line("START")
                    done_line, _ = monitor.wait_for_match(
                        lambda line: line.strip().lower() == "micro workload run: ok",
                        settings.harness_active_timeout_s + settings.harness_done_timeout_s,
                        "micro workload completion",
                        start_index=cursor,
                    )
                    time.sleep(0.2)
                    dut_lines = monitor.snapshot_lines()
                    if done_line is None:
                        set_attempt_error(attempt, 6)
                        attempt["serial_log_path"] = str(
                            write_serial_log(
                                settings.log_dir,
                                log_stem,
                                [*log_header, *(f"DUT: {line}" for line in dut_lines), *(f"HARNESS: {line}" for line in prime_result.harness_log)],
                            )
                        )
                        return attempt

                harness_result = hil_protocol.wait_for_harness_done(
                    harness=harness,
                    harness_active_timeout_s=settings.harness_active_timeout_s,
                    harness_done_timeout_s=settings.harness_done_timeout_s,
                    harness_log=prime_result.harness_log,
                )
                fill_dut_telemetry(attempt, dut_lines)
                validate_direct_dut_telemetry(attempt, direct_serial=direct_serial)
            else:
                time.sleep(0.25)
                prime_result = hil_protocol.prime_harness_session(
                    harness=harness,
                    harness_ready_timeout_s=settings.harness_ready_timeout_s,
                    harness_log=[],
                    flush_input=True,
                )
                if not prime_result.harness_ready:
                    set_attempt_error(attempt, 3)
                    attempt["serial_log_path"] = str(write_serial_log(settings.log_dir, log_stem, [*log_header, *prime_result.harness_log]))
                    return attempt
                harness_result = hil_protocol.wait_for_harness_done(
                    harness=harness,
                    harness_active_timeout_s=settings.harness_active_timeout_s,
                    harness_done_timeout_s=settings.harness_done_timeout_s,
                    harness_log=prime_result.harness_log,
                )
    except serial.SerialException as exc:
        set_attempt_error(attempt, 6)
        attempt["serial_log_path"] = str(write_serial_log(settings.log_dir, log_stem, [*log_header, f"serial_error: {exc}"]))
        return attempt
    except Exception as exc:
        set_attempt_error(attempt, 6)
        attempt["serial_log_path"] = str(write_serial_log(settings.log_dir, log_stem, [*log_header, f"serial_error: {exc}"]))
        return attempt

    if not harness_result.harness_done:
        set_attempt_error(attempt, 4)
        attempt["serial_log_path"] = str(
            write_serial_log(
                settings.log_dir,
                log_stem,
                [*log_header, *(f"DUT: {line}" for line in dut_lines), *(f"HARNESS: {line}" for line in harness_result.harness_log)],
            )
        )
        return attempt
    finish_attempt_from_harness_log(
        attempt,
        harness_done=harness_result.harness_done,
        harness_log=harness_result.harness_log,
        parse_power_metrics=arduino_base._parse_power_metrics,
        normalize_power_metrics=arduino_base.normalize_power_metrics,
    )
    validate_harness_window_duration(attempt, settings.window_ms)
    if int(attempt.get("error_code", -1)) != 0:
        attempt["serial_log_path"] = str(
            write_serial_log(
                settings.log_dir,
                log_stem,
                [*log_header, *(f"DUT: {line}" for line in dut_lines), *(f"HARNESS: {line}" for line in harness_result.harness_log)],
            )
        )
    return attempt


def write_stm32_phase_config(stage_root: Path, *, workload: str, window_ms: int, cpu_clock_mhz: int, wake_margin_us: int, min_sleep_us: int) -> Path:
    """Write the staged STM32 synthetic phase config.

    Parameters
    ----------
    stage_root : Path
        Staged STM32 workspace root.
    workload : str
        Workload name.
    window_ms : int
        Requested window in milliseconds.
    cpu_clock_mhz : int
        STM32 CPU clock preset.
    wake_margin_us : int
        RTC wake margin.
    min_sleep_us : int
        Minimum STOP sleep duration.

    Returns
    -------
    Path
        Generated header path.
    """
    selected_phase = "CREST_DUT_PHASE_CADENCED" if workload == "sleep" else "CREST_DUT_PHASE_BACK_TO_BACK"
    header_text = (
        "#ifndef CREST_DUT_PHASE_CONFIG_H\n"
        "#define CREST_DUT_PHASE_CONFIG_H\n\n"
        "#define CREST_DUT_PHASE_BACK_TO_BACK 0\n"
        "#define CREST_DUT_PHASE_CADENCED 1\n\n"
        f"#define CREST_DUT_SELECTED_PHASE {selected_phase}\n"
        f"#define CREST_DUT_LATENCY_BUDGET_MS {max(1, int(window_ms))}\n"
        "#define CREST_DUT_MEASURED_RUNS 1\n"
        f"#define CREST_DUT_CPU_CLOCK_MHZ {int(cpu_clock_mhz)}\n"
        f"#define CREST_DUT_WAKE_MARGIN_US {max(0, int(wake_margin_us))}\n"
        f"#define CREST_DUT_MIN_SLEEP_US {max(0, int(min_sleep_us))}\n"
        f"#define CREST_MICRO_WORKLOAD_MODE {int(WORKLOAD_MODE[workload])}\n"
        f"#define CREST_MICRO_WINDOW_MS {max(1, int(window_ms))}\n\n"
        "#endif /* CREST_DUT_PHASE_CONFIG_H */\n"
    )
    header_path = stage_root / "Appli" / "Inc" / "crest_dut_phase_config.h"
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(header_text, encoding="utf-8")
    return header_path


def patch_stm32_synthetic_build_recipes(stage_root: Path) -> list[Path]:
    """Patch staged STM32 AppS recipes to remove model/runtime dependencies.

    Parameters
    ----------
    stage_root : Path
        Staged STM32 workspace root.

    Returns
    -------
    list[Path]
        Recipe files changed.
    """
    changed: list[Path] = []
    objects_mk = stage_root / "STM32CubeIDE" / "AppS" / "Debug" / "objects.mk"
    if objects_mk.is_file():
        objects_mk.write_text(
            "################################################################################\n"
            "# Synthetic micro-workload probe: no ST Edge AI runtime libraries are linked.\n"
            "################################################################################\n\n"
            "USER_OBJS :=\n\n"
            "LIBS :=\n",
            encoding="utf-8",
        )
        changed.append(objects_mk)

    for relative in (
        Path("STM32CubeIDE/AppS/Debug/Src/subdir.mk"),
        Path("STM32CubeIDE/AppS/Debug/objects.list"),
    ):
        path = stage_root / relative
        if not path.is_file():
            continue
        lines = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if "stedgeai.mk" in line or "STEDGEAI_INC" in line or "network" in line:
                continue
            lines.append(line)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        changed.append(path)
    return changed


def stm32_stage_workspace_path(settings: RuntimeSettings, workload: str) -> Path:
    """Return the staged STM32 workspace path for one run/workload/config.

    Parameters
    ----------
    settings : RuntimeSettings
        Resolved settings containing the run tag and STM32 config.
    workload : str
        Workload name.

    Returns
    -------
    Path
        Deterministic staged STM32 workspace root for this runner invocation.
    """
    stage_name = (
        f"{workload}_{settings.window_ms}ms_"
        f"{settings.stm32_cpu_clock_mhz}mhz_"
        f"wake{settings.stm32_wake_margin_us}us_"
        f"min{settings.stm32_min_sleep_us}us"
    )
    return settings.stm32_stage_root / settings.run_tag / stage_name / settings.stm32_project_root.name


def stage_stm32_workspace(settings: RuntimeSettings, workload: str) -> Path:
    """Stage or reuse one synthetic STM32 workspace for a workload.

    Parameters
    ----------
    settings : RuntimeSettings
        Resolved settings.
    workload : str
        Workload name.

    Returns
    -------
    Path
        Staged STM32 workspace root.
    """
    stage_root = stm32_stage_workspace_path(settings, workload)
    if stage_root.exists():
        return stage_root
    shutil.copytree(settings.stm32_project_root, stage_root)
    shutil.copy2(STM32_RUNNER_TEMPLATE, stage_root / "Appli" / "Src" / "crest_dut_runner.c")
    write_stm32_phase_config(
        stage_root,
        workload=workload,
        window_ms=settings.window_ms,
        cpu_clock_mhz=settings.stm32_cpu_clock_mhz,
        wake_margin_us=settings.stm32_wake_margin_us,
        min_sleep_us=settings.stm32_min_sleep_us,
    )
    patch_stm32_synthetic_build_recipes(stage_root)
    return stage_root


@dataclass(frozen=True)
class STM32BuildArtifacts:
    """Minimal STM32 LRUN build artifacts needed by the synthetic probe.

    Attributes
    ----------
    app_bin : Path
        Signed application binary selected for flashing.
    """

    app_bin: Path


def _resolve_stm32_bin_path(elf_path: Path) -> Path:
    """Resolve the binary emitted beside an STM32CubeIDE ELF artifact.

    Parameters
    ----------
    elf_path : Path
        Path to the ELF used by the helper.

    Returns
    -------
    Path
        Resolved STM32 bin path.

    Raises
    ------
    RuntimeError
        If existing validation or execution checks fail.
    """
    candidate = elf_path.with_suffix(".bin")
    if candidate.is_file():
        return candidate
    siblings = sorted(elf_path.parent.glob("*.bin"))
    if len(siblings) == 1:
        return siblings[0]
    if siblings:
        names = ", ".join(path.name for path in siblings)
        raise RuntimeError(f"Could not determine matching BIN artifact for {elf_path}; found: {names}")
    raise RuntimeError(f"Missing BIN artifact after build: {candidate}")


def build_stm32_workspace(project_root: Path, *, clean: bool, jobs: int | None) -> STM32BuildArtifacts:
    """Build the staged production STM32 LRUN workspace used by the probe.

    Parameters
    ----------
    project_root : Path
        Root directory for project artifacts.
    clean : bool
        Whether to clean projects before building them.
    jobs : int | None
        Optional parallel build job count.

    Returns
    -------
    STM32BuildArtifacts
        Constructed STM32 workspace.

    Raises
    ------
    WorkflowError
        If existing validation or execution checks fail.
    """
    from crest.microcontrollers import stm32_cube_clt  # type: ignore

    boot_project_root = project_root / "STM32CubeIDE" / "Boot"
    app_project_root = project_root / "STM32CubeIDE" / "AppS"
    for required_dir in (project_root / "FSBL", project_root / "Appli", boot_project_root, app_project_root):
        if not required_dir.is_dir():
            raise stm32_cube_clt.WorkflowError(f"Missing STM32 LRUN workspace path: {required_dir}")
    stm32_cube_clt.build_project(project_root=boot_project_root, jobs=jobs, clean=clean)
    app_build = stm32_cube_clt.build_project(project_root=app_project_root, jobs=jobs, clean=clean)
    return STM32BuildArtifacts(app_bin=_resolve_stm32_bin_path(app_build.elf_path))


def update_stm32_lrun_source_size(
    project_root: Path,
    *,
    trusted_app_size: int,
    alignment: int = 0x400,
) -> tuple[Path, bool, int]:
    """Update the LRUN boot copy-window size from the signed app size.

    Parameters
    ----------
    project_root : Path
        Root directory for project artifacts.
    trusted_app_size : int
        Signed application size used to derive the copy-window size.
    alignment : int
        Byte alignment applied to the generated copy-window size.

    Returns
    -------
    tuple[Path, bool, int]
        Updated header path, whether it changed, and the aligned size.

    Raises
    ------
    WorkflowError
        If existing validation or execution checks fail.
    """
    from crest.microcontrollers import stm32_cube_clt  # type: ignore

    header_path = project_root / "FSBL" / "Inc" / "stm32_extmem_conf.h"
    if not header_path.is_file():
        raise stm32_cube_clt.WorkflowError(f"Missing FSBL extmem config: {header_path}")
    aligned_size = ((int(trusted_app_size) + alignment - 1) // alignment) * alignment
    pattern = re.compile(r"^(#define\s+EXTMEM_LRUN_SOURCE_SIZE\s+)0x[0-9A-Fa-f]+\s*$", re.MULTILINE)
    original_text = header_path.read_text(encoding="utf-8")
    updated_text, replacements = pattern.subn(rf"\g<1>0x{aligned_size:08X}", original_text, count=1)
    if replacements != 1:
        raise stm32_cube_clt.WorkflowError(f"Could not update EXTMEM_LRUN_SOURCE_SIZE in {header_path}")
    changed = updated_text != original_text
    if changed:
        header_path.write_text(updated_text, encoding="utf-8")
    return header_path, changed, aligned_size


def resolve_stm32_external_loader(cubeprog_bin: Path | None) -> Path:
    """Resolve the STM32CubeProgrammer external loader for the Nucleo flash.

    Parameters
    ----------
    cubeprog_bin : Path | None
        Optional STM32CubeProgrammer executable path used as the search anchor.

    Returns
    -------
    Path
        Resolved STM32 external loader.

    Raises
    ------
    WorkflowError
        If existing validation or execution checks fail.
    """
    from crest.microcontrollers import stm32_cube_clt  # type: ignore

    cubeprog_dir = cubeprog_bin or stm32_cube_clt.default_cubeprog_bin()
    if cubeprog_dir is None:
        raise stm32_cube_clt.WorkflowError("STM32CubeProgrammer bin directory was not provided.")
    candidates = (
        cubeprog_dir / "ExternalLoader" / DEFAULT_STM32_EXTERNAL_LOADER_NAME,
        cubeprog_dir.parent / "ExternalLoader" / DEFAULT_STM32_EXTERNAL_LOADER_NAME,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = "\n".join(f"  - {path}" for path in candidates)
    raise stm32_cube_clt.WorkflowError(
        f"Could not find external loader {DEFAULT_STM32_EXTERNAL_LOADER_NAME}.\nSearched:\n{searched}"
    )


def run_stm32_attempt(settings: RuntimeSettings, board: BoardSpec, workload: str, repeat_idx: int) -> dict[str, Any]:
    """Run one STM32 synthetic workload attempt.

    Parameters
    ----------
    settings : RuntimeSettings
        Resolved settings.
    board : BoardSpec
        STM32 board spec.
    workload : str
        Workload name.
    repeat_idx : int
        One-based repeat index.

    Returns
    -------
    dict[str, Any]
        Attempt row.

    Raises
    ------
    RuntimeError
        If existing validation or execution checks fail.
    """
    ensure_import_paths()
    if serial is None:
        raise RuntimeError("pyserial is required for hardware attempts.")
    from crest import hil_protocol  # type: ignore
    from crest.microcontrollers import arduino_base, stm32_cube_clt  # type: ignore
    from crest.microcontrollers import stm32_runtime  # type: ignore

    attempt = base_attempt(settings, board, workload, repeat_idx)
    log_stem = f"{board.token}_{workload}_repeat{repeat_idx:02d}"
    stage_root = stage_stm32_workspace(settings, workload)
    attempt["build_metadata"] = f"stage_root={stage_root},cpu_clock_mhz={settings.stm32_cpu_clock_mhz}"

    try:
        arduino_base.ensure_harness_firmware(
            harness_serial_port=settings.harness_port,
            harness_fqbn=settings.harness_fqbn,
            harness_auto_flash=settings.harness_auto_flash,
            build_defines=build_harness_defines(settings),
        )
    except Exception as exc:
        set_attempt_error(attempt, 2)
        attempt["serial_log_path"] = str(write_serial_log(settings.log_dir, log_stem, [f"harness_prepare_failed: {exc}"]))
        return attempt

    try:
        build = build_stm32_workspace(stage_root, clean=(repeat_idx == 1), jobs=settings.stm32_jobs)
        app_trusted = build.app_bin.with_name(f"{build.app_bin.stem}-trusted.bin")
        stm32_cube_clt.sign_binary(
            signing_tool=settings.stm32_signing_tool,
            input_bin=build.app_bin,
            output_bin=app_trusted,
            load_offset=settings.stm32_signing_load_offset,
            header_version=settings.stm32_signing_header_version,
        )
        _header, copy_window_changed, _copy_window_bytes = update_stm32_lrun_source_size(
            stage_root,
            trusted_app_size=app_trusted.stat().st_size,
        )
        if copy_window_changed:
            build_stm32_workspace(stage_root, clean=False, jobs=settings.stm32_jobs)
        external_loader = resolve_stm32_external_loader(settings.stm32_cubeprog_bin)
    except Exception as exc:
        set_attempt_error(attempt, 8)
        attempt["serial_log_path"] = str(write_serial_log(settings.log_dir, log_stem, [f"stm32_build_failed: {exc}"]))
        return attempt

    dut_lines: list[str] = []
    harness_lines: list[str] = []
    try:
        stm32_cube_clt.program_external_image(
            cubeprog_bin=settings.stm32_cubeprog_bin,
            apid=settings.stm32_apid,
            image_path=app_trusted,
            flash_address=settings.stm32_appli_flash_address,
            external_loader=external_loader,
        )
        with stm32_runtime.SerialMonitor(settings.dut_port, settings.baud_rate, "dut") as monitor:
            with serial.Serial(settings.harness_port, baudrate=settings.baud_rate, timeout=0.1) as harness:
                prime_result = hil_protocol.prime_harness_session(
                    harness=harness,
                    harness_ready_timeout_s=settings.harness_ready_timeout_s,
                    harness_log=[],
                    flush_input=True,
                )
                if not prime_result.harness_ready:
                    set_attempt_error(attempt, 3)
                    attempt["serial_log_path"] = str(write_serial_log(settings.log_dir, log_stem, prime_result.harness_log))
                    return attempt
                stm32_cube_clt.debug_load_elf(
                    elf_path=build.fsbl_build.elf_path,
                    gdbserver=settings.stm32_gdbserver,
                    gdb=settings.stm32_gdb,
                    cubeprog_bin=settings.stm32_cubeprog_bin,
                    gdb_port=settings.stm32_gdb_port,
                    apid=settings.stm32_apid,
                    server_ready_timeout_s=settings.stm32_server_ready_timeout_s,
                    run_after_load=True,
                )
                telemetry = stm32_runtime.execute_runtime_session(
                    monitor,
                    boot_timeout_s=max(12.0, settings.harness_ready_timeout_s),
                    run_timeout_s=settings.harness_active_timeout_s,
                )
                harness_result = hil_protocol.wait_for_harness_done(
                    harness=harness,
                    harness_active_timeout_s=settings.harness_active_timeout_s,
                    harness_done_timeout_s=settings.harness_done_timeout_s,
                    harness_log=prime_result.harness_log,
                )
                dut_lines = telemetry.serial_log
                harness_lines = harness_result.harness_log
    except serial.SerialException as exc:
        set_attempt_error(attempt, 6)
        attempt["serial_log_path"] = str(write_serial_log(settings.log_dir, log_stem, [f"serial_error: {exc}"]))
        return attempt
    except Exception as exc:
        set_attempt_error(attempt, 9)
        attempt["serial_log_path"] = str(write_serial_log(settings.log_dir, log_stem, [f"stm32_program_or_runtime_failed: {exc}", *dut_lines, *harness_lines]))
        return attempt

    fill_dut_telemetry(attempt, dut_lines)
    validate_direct_dut_telemetry(attempt, direct_serial=True)
    finish_attempt_from_harness_log(
        attempt,
        harness_done=harness_result.harness_done,
        harness_log=harness_lines,
        parse_power_metrics=arduino_base._parse_power_metrics,
        normalize_power_metrics=arduino_base.normalize_power_metrics,
    )
    validate_harness_window_duration(attempt, settings.window_ms)
    if int(attempt.get("error_code", -1)) != 0:
        attempt["serial_log_path"] = str(
            write_serial_log(
                settings.log_dir,
                log_stem,
                [
                    "runtime_mode: direct_serial",
                    f"dut_build_defines: workload={workload}, window_ms={settings.window_ms}",
                    *(f"DUT: {line}" for line in dut_lines),
                    *(f"HARNESS: {line}" for line in harness_lines),
                ],
            )
        )
    return attempt


def run_attempt(settings: RuntimeSettings, board: BoardSpec, workload: str, repeat_idx: int) -> dict[str, Any]:
    """Dispatch one attempt to the correct board family runner.

    Parameters
    ----------
    settings : RuntimeSettings
        Resolved settings.
    board : BoardSpec
        Board target.
    workload : str
        Workload name.
    repeat_idx : int
        One-based repeat index.

    Returns
    -------
    dict[str, Any]
        Attempt row.
    """
    if board.family == "stm32":
        return run_stm32_attempt(settings, board, workload, repeat_idx)
    return run_arduino_attempt(settings, board, workload, repeat_idx)


def valid_numeric_values(rows: Iterable[Mapping[str, Any]], key: str) -> list[float]:
    """Collect non-negative finite numeric values for aggregation.

    Parameters
    ----------
    rows : Iterable[Mapping[str, Any]]
        Attempt rows.
    key : str
        Metric key.

    Returns
    -------
    list[float]
        Valid values.
    """
    values: list[float] = []
    for row in rows:
        value = safe_float(row.get(key), default=-1.0)
        if value >= 0.0:
            values.append(value)
    return values


def summarize_group(board: str, workload: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize one board/workload group.

    Parameters
    ----------
    board : str
        Board token.
    workload : str
        Workload name.
    rows : list[dict[str, Any]]
        Attempt rows.

    Returns
    -------
    dict[str, Any]
        Aggregate row.
    """
    summary: dict[str, Any] = {
        "board": board,
        "workload": workload,
        "attempt_count": len(rows),
        "success_count": sum(1 for row in rows if int(row.get("error_code", -1)) == 0),
        "failure_count": sum(1 for row in rows if int(row.get("error_code", -1)) != 0),
    }
    for key in AGG_NUMERIC_FIELDS:
        values = valid_numeric_values(rows, key)
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
    for key in DERIVED_AGG_FIELDS:
        summary[key] = -1.0
    summary["aggregate_warning"] = ""
    return summary


def summarize_attempts(attempts: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate attempts by board/workload.

    Parameters
    ----------
    attempts : Sequence[dict[str, Any]]
        Attempt rows.

    Returns
    -------
    list[dict[str, Any]]
        Aggregate rows.
    """
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for attempt in attempts:
        grouped[(str(attempt["board"]), str(attempt["workload"]))].append(attempt)
    summaries = [
        summarize_group(board, workload, rows)
        for (board, workload), rows in sorted(grouped.items())
    ]
    by_key = {(str(row["board"]), str(row["workload"])): row for row in summaries}
    for row in summaries:
        board = str(row["board"])
        workload = str(row["workload"])
        energy_mean = safe_float(row.get("energy_mj_per_window_mean"))
        work_units_mean = safe_float(row.get("dut_work_units_mean"))
        sleep_row = by_key.get((board, "sleep"))
        wait_row = by_key.get((board, "wait"))
        poll_row = by_key.get((board, "poll"))
        sleep_energy = safe_float(sleep_row.get("energy_mj_per_window_mean") if sleep_row else None)
        wait_energy = safe_float(wait_row.get("energy_mj_per_window_mean") if wait_row else None)
        poll_energy = safe_float(poll_row.get("energy_mj_per_window_mean") if poll_row else None)

        if energy_mean >= 0.0 and sleep_energy >= 0.0:
            row["energy_over_sleep_mj_mean"] = energy_mean - sleep_energy
        if workload in {"wait", "poll", "float", "int"} and energy_mean >= 0.0 and wait_energy >= 0.0:
            row["energy_over_wait_mj_mean"] = energy_mean - wait_energy
        if workload in {"float", "int"} and energy_mean >= 0.0 and poll_energy >= 0.0:
            row["energy_over_poll_mj_mean"] = energy_mean - poll_energy
        elif workload == "poll" and energy_mean >= 0.0 and poll_energy >= 0.0:
            row["energy_over_poll_mj_mean"] = 0.0

        energy_over_sleep = safe_float(row.get("energy_over_sleep_mj_mean"))
        energy_over_wait = safe_float(row.get("energy_over_wait_mj_mean"))
        if work_units_mean > 0.0 and energy_over_sleep >= 0.0:
            row["energy_per_work_unit_nj_mean"] = (energy_over_sleep * 1_000_000.0) / work_units_mean
        if workload in {"float", "int"} and work_units_mean > 0.0 and energy_over_wait >= 0.0:
            row["payload_energy_per_work_unit_nj_mean"] = (energy_over_wait * 1_000_000.0) / work_units_mean
    for row in summaries:
        board = str(row["board"])
        workload = str(row["workload"])
        if workload not in {"poll", "float", "int"}:
            continue
        poll_row = by_key.get((board, "poll"))
        if poll_row is None:
            continue
        poll_energy = safe_float(poll_row.get("energy_mj_per_window_mean"))
        warnings: list[str] = []
        for compute_workload in ("float", "int"):
            compute_row = by_key.get((board, compute_workload))
            compute_energy = safe_float(compute_row.get("energy_mj_per_window_mean") if compute_row else None)
            if poll_energy >= 0.0 and compute_energy >= 0.0 and poll_energy > compute_energy:
                if workload in {"poll", compute_workload}:
                    warnings.append(f"poll_exceeds_{compute_workload}")
        if warnings:
            row["aggregate_warning"] = ",".join(sorted(set(warnings)))
    return summaries


def aggregate_csv_path(output_csv: Path) -> Path:
    """Return the aggregate CSV path paired with an attempt CSV path.

    Parameters
    ----------
    output_csv : Path
        Attempt CSV output path.

    Returns
    -------
    Path
        Aggregate CSV output path.
    """
    suffix = output_csv.suffix or ".csv"
    stem = output_csv.stem if output_csv.suffix else output_csv.name
    return output_csv.with_name(f"{stem}_aggregates{suffix}")


def attempt_jsonl_path(output_json: Path) -> Path:
    """Return the streaming JSONL attempt path paired with the final JSON path.

    Parameters
    ----------
    output_json : Path
        Final JSON output path.

    Returns
    -------
    Path
        JSONL path for incrementally persisted attempt rows.
    """
    suffix = output_json.suffix or ".json"
    stem = output_json.stem if output_json.suffix else output_json.name
    return output_json.with_name(f"{stem}_attempts.jsonl")


def flush_to_disk(handle: Any) -> None:
    """Flush an open output handle through the operating-system buffer.

    Parameters
    ----------
    handle : Any
        Open file-like object with ``flush`` and ``fileno`` methods.
    """
    handle.flush()
    os.fsync(handle.fileno())


def verify_writable_file_path(path: Path) -> None:
    """Create and fsync an output file to catch write failures before hardware runs.

    Parameters
    ----------
    path : Path
        Output file path to validate with an append-mode open.
    """
    ensure_writable_file_path(path)
    with path.open("a", encoding="utf-8") as handle:
        flush_to_disk(handle)


def ensure_writable_file_path(path: Path) -> None:
    """Create a file parent directory and reject directory output targets.

    Parameters
    ----------
    path : Path
        Output file path to validate.

    Raises
    ------
    ValueError
        If ``path`` already exists as a directory.
    """
    if path.exists() and path.is_dir():
        raise ValueError(f"Output path is a directory, expected a file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


class StreamingAttemptWriter:
    """Persist attempt rows as they complete during long HIL runs.

    Parameters
    ----------
    settings : RuntimeSettings
        Resolved runtime settings containing output paths and metadata.
    """

    def __init__(self, settings: RuntimeSettings) -> None:
        """Initialize the streaming writer without opening files.

        Parameters
        ----------
        settings : RuntimeSettings
            Resolved runtime settings containing output paths and metadata.
        """
        self.settings = settings
        self.csv_handle: Any | None = None
        self.csv_writer: csv.DictWriter[Any] | None = None
        self.jsonl_handle: Any | None = None

    def open(self) -> None:
        """Open attempt CSV and JSONL files before hardware execution starts.

        Returns
        -------
        None
            Output handles are stored on this writer and metadata is flushed.

        Raises
        ------
        ValueError
            If either output path resolves to an existing directory.
        """
        ensure_writable_file_path(self.settings.output_csv)
        ensure_writable_file_path(attempt_jsonl_path(self.settings.output_json))
        self.csv_handle = self.settings.output_csv.open("w", newline="", encoding="utf-8")
        self.csv_writer = csv.DictWriter(self.csv_handle, fieldnames=CSV_COLUMNS)
        self.csv_writer.writeheader()
        self.jsonl_handle = attempt_jsonl_path(self.settings.output_json).open("w", encoding="utf-8")
        self.jsonl_handle.write(json.dumps({"type": "metadata", "metadata": build_output_payload(self.settings, [])["metadata"]}) + "\n")
        flush_to_disk(self.csv_handle)
        flush_to_disk(self.jsonl_handle)

    def append(self, attempt: Mapping[str, Any]) -> None:
        """Write one completed attempt row and flush it to disk.

        Parameters
        ----------
        attempt : Mapping[str, Any]
            Completed attempt row.

        Raises
        ------
        RuntimeError
            If existing validation or execution checks fail.
        """
        if self.csv_handle is None or self.csv_writer is None or self.jsonl_handle is None:
            raise RuntimeError("StreamingAttemptWriter.open() must be called before append().")
        self.csv_writer.writerow({column: attempt.get(column, "") for column in CSV_COLUMNS})
        self.jsonl_handle.write(json.dumps({"type": "attempt", "attempt": dict(attempt)}) + "\n")
        flush_to_disk(self.csv_handle)
        flush_to_disk(self.jsonl_handle)

    def close(self) -> None:
        """Close any open streaming output handles.

        Returns
        -------
        None
            Handles are closed if present and cleared from this writer.
        """
        for handle in (self.csv_handle, self.jsonl_handle):
            if handle is not None:
                handle.close()
        self.csv_handle = None
        self.csv_writer = None
        self.jsonl_handle = None


def preflight_output_paths(settings: RuntimeSettings) -> None:
    """Validate all final and streaming output paths before hardware runs.

    Parameters
    ----------
    settings : RuntimeSettings
        Resolved runtime settings containing output paths.
    """
    verify_writable_file_path(settings.output_json)
    verify_writable_file_path(settings.output_csv)
    verify_writable_file_path(aggregate_csv_path(settings.output_csv))
    verify_writable_file_path(attempt_jsonl_path(settings.output_json))


def write_csv(path: Path, attempts: Sequence[Mapping[str, Any]]) -> None:
    """Write attempt rows to CSV.

    Parameters
    ----------
    path : Path
        CSV output path.
    attempts : Sequence[Mapping[str, Any]]
        Attempt rows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for attempt in attempts:
            writer.writerow({column: attempt.get(column, "") for column in CSV_COLUMNS})


def write_aggregate_csv(path: Path, aggregates: Sequence[Mapping[str, Any]]) -> None:
    """Write aggregate rows to CSV.

    Parameters
    ----------
    path : Path
        Aggregate CSV output path.
    aggregates : Sequence[Mapping[str, Any]]
        Aggregate rows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AGG_CSV_COLUMNS)
        writer.writeheader()
        for aggregate in aggregates:
            writer.writerow({column: aggregate.get(column, "") for column in AGG_CSV_COLUMNS})


def build_output_payload(settings: RuntimeSettings, attempts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Build the JSON output payload.

    Parameters
    ----------
    settings : RuntimeSettings
        Resolved settings.
    attempts : Sequence[dict[str, Any]]
        Attempt rows.

    Returns
    -------
    dict[str, Any]
        JSON-serializable output payload.
    """
    return {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "config_path": str(settings.config_path),
            "boards": [board.token for board in settings.boards],
            "workloads": list(settings.workloads),
            "repeats": settings.repeats,
            "window_ms": settings.window_ms,
            "dut_port": settings.dut_port,
            "harness_port": settings.harness_port,
            "harness_fqbn": settings.harness_fqbn,
            "harness_auto_flash": settings.harness_auto_flash,
            "stm32_project_root": str(settings.stm32_project_root),
            "stm32_stage_root": str(settings.stm32_stage_root),
            "run_tag": str(getattr(settings, "run_tag", "")),
            "aggregate_csv_path": str(aggregate_csv_path(settings.output_csv)),
            "attempt_jsonl_path": str(attempt_jsonl_path(settings.output_json)),
            "notes": "Energy is one measured harness window because DUT run count is fixed to 1. Payload diagnostics subtract the board-local wait baseline; tight poll/spin is retained as a separate phase workload.",
        },
        "attempts": list(attempts),
        "aggregates": summarize_attempts(attempts),
    }


def write_outputs(settings: RuntimeSettings, attempts: Sequence[dict[str, Any]]) -> None:
    """Write JSON and CSV outputs.

    Parameters
    ----------
    settings : RuntimeSettings
        Resolved settings.
    attempts : Sequence[dict[str, Any]]
        Attempt rows.
    """
    payload = build_output_payload(settings, attempts)
    settings.output_json.parent.mkdir(parents=True, exist_ok=True)
    settings.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_csv(settings.output_csv, attempts)
    write_aggregate_csv(aggregate_csv_path(settings.output_csv), payload["aggregates"])


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """
    examples = """examples:
  # First smoke test on the installed STM32 board.
  python analysis_scripts/micro_workload_energy_probe/run_micro_workload_energy_probe.py --boards stm32 --workloads sleep --window-ms 200 --repeats 1

  # Run the full requested matrix with default serial ports.
  python analysis_scripts/micro_workload_energy_probe/run_micro_workload_energy_probe.py --boards stm32 portenta_m4 portenta_m7 ble --workloads sleep wait poll float int --repeats 3

  # Reuse an already-flashed harness and write explicit outputs.
  python analysis_scripts/micro_workload_energy_probe/run_micro_workload_energy_probe.py --boards ble --workloads wait poll float int --skip-harness-flash --output-json results/probe.json --output-csv results/probe.csv
"""
    parser = argparse.ArgumentParser(
        description="Run synthetic MCU sleep/wait/poll/float/int energy windows through the CREST HIL harness.",
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--boards", nargs="+", default=["stm32"], help="Board tokens to run: stm32, portenta_m4, portenta_m7, ble. Portenta tokens encode the core.")
    parser.add_argument("--workloads", nargs="+", default=list(VALID_WORKLOADS), help="Workloads to run: sleep, wait, poll, float, int.")
    parser.add_argument("--repeats", type=int, default=1, help="Number of attempts per board/workload pair.")
    parser.add_argument("--window-ms", type=int, default=DEFAULT_WINDOW_MS, help="Requested DUT trigger-high measurement window in milliseconds.")
    parser.add_argument("--dut-port", default=None, help="DUT serial port; defaults to config device.serial_port or /dev/ttyACM0.")
    parser.add_argument("--harness-port", default=None, help="Harness serial port; defaults to config device.harness_serial_port or /dev/ttyACM1.")
    parser.add_argument("--harness-fqbn", default=None, help="Harness Arduino FQBN; default is arduino:mbed_nano:nano33ble.")
    parser.add_argument("--harness-auto-flash", default=None, help="Harness flash policy: once, always, or never.")
    parser.add_argument("--skip-harness-flash", action="store_true", help="Use the currently flashed harness firmware and set harness flash policy to never.")
    parser.add_argument("--harness-arm-pin", type=int, default=None, help="Harness active-low arm pin number; default D3.")
    parser.add_argument("--harness-trigger-pin", type=int, default=None, help="Harness trigger pin number; default D2.")
    parser.add_argument("--dut-arm-hold-ms", type=int, default=None, help="DUT delay between arm LOW and trigger HIGH.")
    parser.add_argument("--harness-stable-low-ms", type=int, default=None, help="Stable-low duration required by the harness before a trigger rising edge.")
    parser.add_argument("--harness-ready-timeout-s", type=float, default=None, help="Host timeout waiting for HARNESS READY after PING.")
    parser.add_argument("--harness-arm-timeout-s", type=float, default=None, help="Harness firmware timeout while waiting for the trigger rising edge; 0 disables.")
    parser.add_argument("--harness-active-timeout-s", type=float, default=None, help="Host and harness timeout for the active trigger-high window.")
    parser.add_argument("--harness-done-timeout-s", type=float, default=None, help="Additional host timeout waiting for DONE after the active window.")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD, help="Serial baud rate for DUT and harness sessions.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Optional CREST config used as a source of port/tool defaults.")
    parser.add_argument("--output-json", default=None, help="Final JSON output path, or a directory for a timestamped JSON file; default is analysis_scripts/micro_workload_energy_probe/results/<timestamp>.json.")
    parser.add_argument("--output-csv", default=None, help="Attempt CSV output path, or a directory for a timestamped CSV file; rows are flushed after each attempt.")
    parser.add_argument("--log-level", default="INFO", help="Logging level: DEBUG, INFO, WARNING, or ERROR.")
    parser.add_argument("--stm32-stage-root", default=str(SCRIPT_DIR / ".staged_stm32"), help="Root for generated STM32 LRUN staging workspaces.")
    parser.add_argument("--stm32-project-root", default=None, help="Production STM32 LRUN template root to copy for staged synthetic builds.")
    parser.add_argument("--stm32-cpu-clock-mhz", type=int, default=None, help="STM32 CPU clock preset, matching the production LRUN clock define path.")
    parser.add_argument("--stm32-wake-margin-us", type=int, default=None, help="STM32 sleep wake margin used by RTC-cadenced STOP/WFI workload.")
    parser.add_argument("--stm32-min-sleep-us", type=int, default=None, help="Minimum STM32 STOP sleep request in microseconds.")
    parser.add_argument("--stm32-jobs", type=int, default=os.cpu_count() or 1, help="Parallel jobs for STM32 LRUN make builds.")
    parser.add_argument("--stm32-appli-flash-address", default=DEFAULT_STM32_APPLI_FLASH_ADDRESS, help="External flash address for the signed STM32 App image.")
    parser.add_argument("--stm32-cubeprog-bin", default=None, help="Optional STM32CubeProgrammer bin directory override.")
    parser.add_argument("--stm32-gdbserver", default=None, help="Optional ST-LINK_gdbserver executable override.")
    parser.add_argument("--stm32-gdb", default=None, help="Optional arm-none-eabi-gdb executable override.")
    parser.add_argument("--stm32-gdb-port", type=int, default=None, help="TCP port used by ST-LINK_gdbserver.")
    parser.add_argument("--stm32-apid", type=int, default=None, help="STM32 access-port identifier used by programmer/debug tools.")
    parser.add_argument("--stm32-server-ready-timeout-s", type=float, default=None, help="Timeout waiting for ST-LINK_gdbserver readiness.")
    parser.add_argument("--stm32-signing-tool", default=None, help="Optional STM32 trusted-image signing tool override.")
    parser.add_argument("--stm32-signing-header-version", default=None, help="STM32 signing header version.")
    parser.add_argument("--stm32-signing-load-offset", default=None, help="STM32 signing load offset.")
    return parser


def main() -> int:
    """Run the requested matrix and write outputs.

    Returns
    -------
    int
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)
    try:
        settings = resolve_settings(args)
    except ValueError as exc:
        parser.error(str(exc))

    preflight_output_paths(settings)
    attempts: list[dict[str, Any]] = []
    logging.info(
        "Running boards=%s workloads=%s repeats=%d window_ms=%d",
        [board.token for board in settings.boards],
        settings.workloads,
        settings.repeats,
        settings.window_ms,
    )
    attempt_specs = [
        (board, workload, repeat_idx)
        for board in settings.boards
        for workload in settings.workloads
        for repeat_idx in range(1, settings.repeats + 1)
    ]
    progress_iter = attempt_specs
    if tqdm is not None:
        progress_iter = tqdm(
            attempt_specs,
            total=len(attempt_specs),
            desc="micro workload HIL attempts",
            unit="attempt",
            dynamic_ncols=True,
        )

    stream_writer = StreamingAttemptWriter(settings)
    stream_writer.open()
    try:
        for board, workload, repeat_idx in progress_iter:
            if tqdm is not None and hasattr(progress_iter, "set_postfix_str"):
                progress_iter.set_postfix_str(f"{board.token}/{workload}/r{repeat_idx}")
            logging.info("Attempt start: board=%s workload=%s repeat=%d", board.token, workload, repeat_idx)
            try:
                attempt = run_attempt(settings, board, workload, repeat_idx)
            except Exception as exc:
                attempt = base_attempt(settings, board, workload, repeat_idx)
                set_attempt_error(attempt, 10)
                attempt["serial_log_path"] = str(
                    write_serial_log(settings.log_dir, f"{board.token}_{workload}_repeat{repeat_idx:02d}", [f"unhandled_error: {exc}"])
                )
            attempts.append(attempt)
            stream_writer.append(attempt)
            logging.info(
                "Attempt done: board=%s workload=%s repeat=%d error=%s energy_mj=%.6f window_ms=%.3f",
                board.token,
                workload,
                repeat_idx,
                attempt["error_label"],
                safe_float(attempt["energy_mj_per_window"]),
                safe_float(attempt["measured_harness_window_ms"]),
            )
    finally:
        stream_writer.close()

    write_outputs(settings, attempts)
    logging.info("Wrote JSON: %s", settings.output_json)
    logging.info("Wrote CSV: %s", settings.output_csv)
    logging.info("Wrote aggregate CSV: %s", aggregate_csv_path(settings.output_csv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
