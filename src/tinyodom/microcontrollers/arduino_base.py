from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import serial


# Resolve the Arduino CLI executable once so every subprocess call uses the same path.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_REPO_ARDUINO_CLI = _PROJECT_ROOT / "tools" / "bin" / "arduino-cli"
ARDUINO_CLI_BIN = os.environ.get("ARDUINO_CLI_BIN")
ARDUINO_CLI_CONFIG = str(_PROJECT_ROOT / "tools" / "arduino-cli.yaml")
if not ARDUINO_CLI_BIN:
    if _REPO_ARDUINO_CLI.exists():
        ARDUINO_CLI_BIN = str(_REPO_ARDUINO_CLI)
    else:
        # Fallback to PATH lookup so developer installations still work.
        ARDUINO_CLI_BIN = shutil.which("arduino-cli") or "arduino-cli"
print(f"Using Arduino CLI at: {ARDUINO_CLI_BIN}")


FLASH_USAGE_RE = re.compile(
    r"Sketch uses (\d+) bytes.*?Maximum is (\d+)", re.IGNORECASE | re.DOTALL
)
RAM_USAGE_RE = re.compile(
    r"Global variables use (\d+) bytes.*?Maximum is (\d+)", re.IGNORECASE | re.DOTALL
)
FLASH_OVERFLOW_PATTERNS = [
    re.compile(r"section [`']?\.text[`']?\s+will not fit in region [`']?flash[`']?", re.IGNORECASE),
    re.compile(r"region [`']?flash[`']?\s+overflowed", re.IGNORECASE),
]
RAM_OVERFLOW_PATTERNS = [
    re.compile(r"region [`']?ram[`']?\s+overflowed", re.IGNORECASE),
    re.compile(r"region [`']?sram[`']?\s+overflowed", re.IGNORECASE),
    re.compile(r"cannot move location counter backwards", re.IGNORECASE),
]
ARENA_TOO_SMALL_PATTERNS = [
    re.compile(r"size is too small for all buffers", re.IGNORECASE),
    re.compile(r"failed\s+to\s+allocate", re.IGNORECASE),
    re.compile(r"buffer\s+missing", re.IGNORECASE),
]
MISSING_BYTES_RE = re.compile(r"missing:\s*(\d+)", re.IGNORECASE)
REQUESTED_BYTES_RE = re.compile(r"requested:\s*(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class CompileResult:
    """Result of an Arduino CLI compile invocation.

    Parameters
    ----------
    success : bool
        True when the compile returned zero.
    log : str
        Combined stdout/stderr from the compile step.
    flash_bytes : int | None
        Parsed flash usage in bytes, if available.
    ram_bytes : int | None
        Parsed RAM usage in bytes, if available.
    overflow_kind : str | None
        "flash" or "ram" when an overflow is detected.
    build_dir : pathlib.Path
        Build cache directory used by the Arduino CLI.
    """

    success: bool
    log: str
    flash_bytes: Optional[int]
    ram_bytes: Optional[int]
    overflow_kind: Optional[str]
    build_dir: Path


@dataclass(frozen=True)
class UploadResult:
    """Result of an Arduino CLI upload invocation.

    Parameters
    ----------
    success : bool
        True when the upload returned zero.
    log : str
        Combined stdout/stderr from the upload step.
    """

    success: bool
    log: str


@dataclass(frozen=True)
class MeasureResult:
    """Parsed metrics from a firmware serial log.

    Parameters
    ----------
    latency_s : float | None
        Inference latency in seconds, if found.
    arena_error_line : str | None
        First arena allocation error line, if found.
    serial_log : list[str]
        Full decoded serial log captured during measurement.
    power_metrics : dict[str, float | None] | None
        Parsed power telemetry fields, if present.
    """

    latency_s: Optional[float]
    arena_error_line: Optional[str]
    serial_log: List[str]
    power_metrics: Optional[Dict[str, Optional[float]]]


def _compute_retry_hint_bytes(
    current_arena_bytes: int, arena_error_line: Optional[str]
) -> Optional[int]:
    """Derive a suggested arena size (bytes) from the device log when available.

    Parameters
    ----------
    current_arena_bytes : int
        Current tensor arena size in bytes.
    arena_error_line : str | None
        Log line that includes arena allocation details.

    Returns
    -------
    int | None
        Suggested arena size in bytes, or None if unavailable.
    """
    if not arena_error_line:
        return None
    missing_match = MISSING_BYTES_RE.search(arena_error_line)
    requested_match = REQUESTED_BYTES_RE.search(arena_error_line)
    target_bytes: Optional[int] = None

    if missing_match:
        missing = int(missing_match.group(1))
        if missing > 0:
            target_bytes = current_arena_bytes + missing
    elif requested_match:
        requested = int(requested_match.group(1))
        if requested > current_arena_bytes:
            target_bytes = requested

    if target_bytes is None or target_bytes <= current_arena_bytes:
        return None

    # Add a small cushion to avoid oscillating on an exact boundary.
    return target_bytes + 2048


def _replace_define(text: str, name: str, value: str) -> str:
    """Replace a single `#define` directive within the sketch source.

    Parameters
    ----------
    text : str
        Sketch contents to mutate.
    name : str
        Macro symbol to replace.
    value : str
        Replacement literal inserted after the symbol.

    Returns
    -------
    str
        Updated sketch text with the new macro definition.
    """
    pattern = re.compile(rf"(#define\s+{re.escape(name)}\s+)([^\n]+)")
    if not pattern.search(text):
        raise ValueError(f"Unable to locate definition for {name}.")
    return pattern.sub(lambda match: f"{match.group(1)}{value}", text, count=1)


def _patch_sketch_constants(
    sketch_path: Path, arena_kb: int, window_size: int, num_channels: int
) -> None:
    """Rewrite TinyODOM deployment constants inside the Arduino sketch.

    Parameters
    ----------
    sketch_path : pathlib.Path
        Directory containing the target `.ino` file.
    arena_kb : int
        Tensor arena size expressed in KiB.
    window_size : int
        Sliding window length used by the model.
    num_channels : int
        Number of sensor channels captured per window.
    """
    ino_files = sorted(sketch_path.glob("*.ino"))
    if not ino_files:
        raise FileNotFoundError(f"No .ino file found in {sketch_path}")
    ino_path = ino_files[0]
    text = ino_path.read_text()
    text = _replace_define(text, "TINYODOM_WINDOW_SIZE", str(window_size))
    text = _replace_define(text, "TINYODOM_NUM_CHANNELS", str(num_channels))
    text = _replace_define(text, "TINYODOM_TENSOR_ARENA_BYTES", f"({arena_kb} * 1024)")
    ino_path.write_text(text)


def _parse_memory_from_compile(output: str) -> Tuple[Optional[int], Optional[int]]:
    """Extract flash and RAM usage from Arduino CLI compile output.

    Parameters
    ----------
    output : str
        stdout emitted by `arduino-cli compile`.

    Returns
    -------
    tuple[Optional[int], Optional[int]]
        Flash bytes and RAM bytes when parseable, otherwise None placeholders.
    """
    flash_match = FLASH_USAGE_RE.search(output)
    ram_match = RAM_USAGE_RE.search(output)
    flash_bytes = int(flash_match.group(1)) if flash_match else None
    ram_bytes = int(ram_match.group(1)) if ram_match else None
    return flash_bytes, ram_bytes


def _classify_compile_failure(log_text: str) -> Optional[str]:
    """Determine whether output indicates a flash or RAM overflow.

    Parameters
    ----------
    log_text : str
        Concatenated stdout/stderr from `arduino-cli compile`.

    Returns
    -------
    str | None
        "flash" when program storage overflowed, "ram" for RAM overflow,
        otherwise None.
    """
    normalized = log_text.lower()
    for pattern in FLASH_OVERFLOW_PATTERNS:
        if pattern.search(normalized):
            return "flash"
    for pattern in RAM_OVERFLOW_PATTERNS:
        if pattern.search(normalized):
            return "ram"
    return None


def _resolve_build_dir(sketch_path: Path, fqbn: str) -> Path:
    """Return the Arduino build cache directory for a sketch and FQBN."""
    build_cache_root = sketch_path / ".arduino-build"
    return build_cache_root / fqbn.replace(":", "_")


def compile_sketch(*, sketch_path: Path, fqbn: str) -> CompileResult:
    """Compile an Arduino sketch with the Arduino CLI.

    Parameters
    ----------
    sketch_path : pathlib.Path
        Directory containing the Arduino sketch.
    fqbn : str
        Fully qualified board name.

    Returns
    -------
    CompileResult
        Parsed compile results including RAM/flash usage and overflow status.
    """
    build_dir = _resolve_build_dir(sketch_path, fqbn)
    build_dir.mkdir(parents=True, exist_ok=True)
    compile_cmd = [
        ARDUINO_CLI_BIN,
        "--config-file",
        ARDUINO_CLI_CONFIG,
        "compile",
        "--fqbn",
        fqbn,
        "--build-path",
        str(build_dir),
        str(sketch_path),
    ]
    compile_proc = subprocess.run(
        compile_cmd, capture_output=True, text=True, check=False
    )
    compile_log = f"{compile_proc.stdout}\n{compile_proc.stderr}"
    flash_bytes, ram_bytes = _parse_memory_from_compile(compile_log)
    overflow_kind = _classify_compile_failure(compile_log)
    return CompileResult(
        success=compile_proc.returncode == 0,
        log=compile_log,
        flash_bytes=flash_bytes,
        ram_bytes=ram_bytes,
        overflow_kind=overflow_kind,
        build_dir=build_dir,
    )


def upload_sketch(
    *,
    sketch_path: Path,
    fqbn: str,
    build_dir: Path,
    serial_port: str,
) -> UploadResult:
    """Upload a compiled Arduino sketch to a device.

    Parameters
    ----------
    sketch_path : pathlib.Path
        Directory containing the Arduino sketch.
    fqbn : str
        Fully qualified board name.
    build_dir : pathlib.Path
        Build cache directory containing the compiled binary.
    serial_port : str
        Serial port used for upload.

    Returns
    -------
    UploadResult
        Upload success flag and captured output.
    """
    upload_cmd = [
        ARDUINO_CLI_BIN,
        "--config-file",
        ARDUINO_CLI_CONFIG,
        "upload",
        "-p",
        serial_port,
        "--fqbn",
        fqbn,
        "--build-path",
        str(build_dir),
        str(sketch_path),
    ]
    upload_proc = subprocess.run(upload_cmd, capture_output=True, text=True, check=False)
    upload_log = f"{upload_proc.stdout}\n{upload_proc.stderr}"
    return UploadResult(success=upload_proc.returncode == 0, log=upload_log)


# regex patterns for parsing power metrics from serial log.
_FLOAT_CAPTURE = r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
POWER_FIELD_SPECS: Dict[str, Tuple[str, re.Pattern[str]]] = {
    "sequence": (
        "int",
        re.compile(r"^inference seq:\s*(?P<value>\d+)$", re.IGNORECASE),
    ),
    "energy_mj_per_inference": (
        "float",
        re.compile(rf"^energy output.*?:\s*{_FLOAT_CAPTURE}$", re.IGNORECASE),
    ),
    "avg_power_mw": (
        "float",
        re.compile(rf"^avg power output.*?:\s*{_FLOAT_CAPTURE}$", re.IGNORECASE),
    ),
    "avg_current_ma": (
        "float",
        re.compile(rf"^avg current output.*?:\s*{_FLOAT_CAPTURE}$", re.IGNORECASE),
    ),
    "bus_voltage_v": (
        "float",
        re.compile(rf"^bus voltage output.*?:\s*{_FLOAT_CAPTURE}$", re.IGNORECASE),
    ),
    "idle_power_mw": (
        "float",
        re.compile(rf"^idle power baseline.*?:\s*{_FLOAT_CAPTURE}$", re.IGNORECASE),
    ),
}

POWER_METRIC_DEFAULTS: Dict[str, float] = {
    "sequence": -1.0,
    "energy_mj_per_inference": -1.0,
    "avg_power_mw": -1.0,
    "avg_current_ma": -1.0,
    "bus_voltage_v": -1.0,
    "idle_power_mw": -1.0,
}


def _parse_power_metrics(lines: Sequence[str]) -> Optional[Dict[str, Optional[float]]]:
    """Extract structured telemetry from the firmware serial log.

    Parameters
    ----------
    lines : Sequence[str]
        Sequence of decoded serial log lines.

    Returns
    -------
    dict[str, float | None] | None
        Parsed power metrics if any match was found, otherwise None.
    """
    candidates: Dict[str, Optional[float]] = {key: None for key in POWER_FIELD_SPECS}
    matched = False
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        for key, (dtype, pattern) in POWER_FIELD_SPECS.items():
            if candidates[key] is not None:
                continue
            match = pattern.match(line)
            if not match:
                continue
            value = match.group("value")
            try:
                if dtype == "int":
                    candidates[key] = float(int(value))
                else:
                    candidates[key] = float(value)
                matched = True
            except (TypeError, ValueError):
                candidates[key] = None
    return candidates if matched else None


def normalize_power_metrics(
    power_metrics: Optional[Dict[str, Optional[float]]]
) -> Dict[str, float]:
    """Return power metrics with defaults so downstream code can rely on keys.

    Parameters
    ----------
    power_metrics : dict[str, float | None] | None
        Parsed power metrics from the serial log.

    Returns
    -------
    dict[str, float]
        Normalized power metrics with missing fields set to defaults.
    """
    normalized = POWER_METRIC_DEFAULTS.copy()
    if not power_metrics:
        return normalized
    for key, value in power_metrics.items():
        if key not in normalized or value is None:
            continue
        normalized[key] = value
    return normalized


def _collect_latency_seconds(
    port: str, baud: int, timeout_s: float
) -> Tuple[Optional[float], Optional[str], List[str]]:
    """Read the first `timer output:` line produced by the firmware.

    Parameters
    ----------
    port : str
        Serial port identifier.
    baud : int
        Serial baud rate.
    timeout_s : float
        Maximum time in seconds to wait for output.

    Returns
    -------
    tuple[float | None, str | None, list[str]]
        Latency (seconds), arena error line (if any), and captured serial log.
    """
    decoded_lines: List[str] = []
    try:
        with serial.Serial(port, baudrate=baud, timeout=1.0) as ser:  # type: ignore[arg-type]
            start_time = time.time()
            while time.time() - start_time < timeout_s:
                raw = ser.readline()
                if not raw:
                    continue
                try:
                    line = raw.decode("utf-8", errors="ignore").strip()
                except UnicodeDecodeError:
                    continue
                if not line:
                    continue
                decoded_lines.append(line)
    except serial.SerialException as exc:  # type: ignore[attr-defined]
        raise RuntimeError(f"Failed to read serial port {port}: {exc}") from exc
    for line in decoded_lines:
        lower_line = line.lower()
        if lower_line.startswith("timer output:"):
            _, _, value = line.partition(":")
            try:
                return float(value.strip()), None, decoded_lines
            except ValueError:
                return None, None, decoded_lines
        if any(pattern.search(lower_line) for pattern in ARENA_TOO_SMALL_PATTERNS):
            return None, line, decoded_lines
    return None, None, decoded_lines


def measure_serial(
    *,
    serial_port: str,
    baud_rate: int,
    serial_timeout_s: float,
) -> MeasureResult:
    """Capture latency and power metrics from a device serial log.

    Parameters
    ----------
    serial_port : str
        Serial port used to read the log.
    baud_rate : int
        Serial baud rate.
    serial_timeout_s : float
        Timeout when waiting for output.

    Returns
    -------
    MeasureResult
        Parsed latency, arena errors, and power metrics.
    """
    latency_s, arena_error_line, serial_log = _collect_latency_seconds(
        serial_port, baud_rate, serial_timeout_s
    )
    power_metrics = _parse_power_metrics(serial_log)
    return MeasureResult(
        latency_s=latency_s,
        arena_error_line=arena_error_line,
        serial_log=serial_log,
        power_metrics=power_metrics,
    )


__all__ = [
    "ARDUINO_CLI_BIN",
    "ARDUINO_CLI_CONFIG",
    "ARENA_TOO_SMALL_PATTERNS",
    "CompileResult",
    "UploadResult",
    "MeasureResult",
    "POWER_FIELD_SPECS",
    "POWER_METRIC_DEFAULTS",
    "_compute_retry_hint_bytes",
    "_classify_compile_failure",
    "_collect_latency_seconds",
    "_parse_memory_from_compile",
    "_parse_power_metrics",
    "_patch_sketch_constants",
    "_replace_define",
    "compile_sketch",
    "measure_serial",
    "normalize_power_metrics",
    "upload_sketch",
]
