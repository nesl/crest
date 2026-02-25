from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import serial

from .. import hil_protocol

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

logger = logging.getLogger(__name__)

_HARNESS_SKETCH_DIR = _PROJECT_ROOT / "sketches" / "harness"
_SHARED_COMMON_DIR = _PROJECT_ROOT / "sketches" / "common"
_SHARED_HEADER_NAMES = ("tinyodom_hil_config.h", "tinyodom_power.h")
_HARNESS_FLASHED_SIGNATURES: Dict[str, str] = {}


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
UPLOAD_PERMISSION_PATTERNS = [
    re.compile(r"LIBUSB_ERROR_ACCESS", re.IGNORECASE),
    re.compile(r"dfu-util:.*permission denied", re.IGNORECASE),
]


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
    serial_log: list[str]
    power_metrics: Optional[dict[str, Optional[float]]]


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


def _resolve_platform_txt_path(build_dir: Path) -> Optional[Path]:
    """Resolve the active platform.txt from Arduino build metadata.

    Parameters
    ----------
    build_dir : pathlib.Path
        Arduino CLI build directory.

    Returns
    -------
    pathlib.Path | None
        Path to ``platform.txt`` when discoverable, else ``None``.
    """
    options_path = build_dir / "build.options.json"
    if not options_path.exists():
        return None
    try:
        metadata = json.loads(options_path.read_text())
    except (OSError, ValueError):
        return None
    hardware_folders = str(metadata.get("hardwareFolders", "")).split(",")
    for folder in hardware_folders:
        folder_path = Path(folder.strip())
        if not folder_path:
            continue
        platform_txt = folder_path / "platform.txt"
        if platform_txt.exists():
            return platform_txt
    return None


def _load_platform_properties(platform_txt: Path) -> Dict[str, str]:
    """Parse ``platform.txt`` key/value entries.

    Parameters
    ----------
    platform_txt : pathlib.Path
        Platform configuration file.

    Returns
    -------
    dict[str, str]
        Parsed property map.
    """
    properties: Dict[str, str] = {}
    for raw_line in platform_txt.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        properties[key.strip()] = value.strip()
    return properties


def _sum_size_regex_matches(output: str, pattern_text: Optional[str]) -> Optional[int]:
    """Sum section sizes matched by a platform ``recipe.size.regex`` pattern."""
    if not pattern_text:
        return None
    try:
        pattern = re.compile(pattern_text, re.MULTILINE)
    except re.error:
        return None
    if pattern.groups < 1:
        return None

    total = 0
    matched = False
    for match in pattern.finditer(output):
        try:
            raw_size = match.group(1)
        except IndexError:
            return None
        try:
            total += int(raw_size)
            matched = True
        except (TypeError, ValueError):
            continue
    if not matched:
        return None
    return total


def _parse_memory_from_size_recipe(build_dir: Path) -> Tuple[Optional[int], Optional[int]]:
    """Parse memory usage using board-defined ``platform.txt`` size regexes.

    This mirrors Arduino's classic ``recipe.size.regex`` semantics used to
    produce ``Sketch uses ...`` and ``Global variables use ...`` lines.

    Parameters
    ----------
    build_dir : pathlib.Path
        Arduino CLI build directory containing ``build.options.json`` and ELF.

    Returns
    -------
    tuple[int | None, int | None]
        ``(flash_bytes, ram_bytes)`` when regex parsing succeeds.
    """
    platform_txt = _resolve_platform_txt_path(build_dir)
    if platform_txt is None:
        return None, None
    try:
        properties = _load_platform_properties(platform_txt)
    except OSError:
        return None, None

    flash_regex = properties.get("recipe.size.regex")
    ram_regex = properties.get("recipe.size.regex.data")
    if not flash_regex and not ram_regex:
        return None, None

    size_bin = _resolve_arm_size_binary()
    elf_path = _find_compiled_elf(build_dir)
    if not size_bin or elf_path is None:
        return None, None

    proc = subprocess.run(
        [size_bin, "-A", str(elf_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None, None

    flash_total = _sum_size_regex_matches(proc.stdout, flash_regex)
    ram_total = _sum_size_regex_matches(proc.stdout, ram_regex)
    return flash_total, ram_total


def _format_board_options(board_options: Optional[Mapping[str, str]]) -> str:
    """Convert board options map to Arduino CLI ``--board-options`` payload.

    Parameters
    ----------
    board_options : Mapping[str, str] | None
        Board option key/value pairs.

    Returns
    -------
    str
        Comma-separated ``key=value`` payload accepted by Arduino CLI.
    """
    if not board_options:
        return ""
    parts: List[str] = []
    for key in sorted(board_options):
        value = str(board_options[key]).strip()
        if not value:
            continue
        parts.append(f"{key}={value}")
    return ",".join(parts)


def _find_compiled_elf(build_dir: Path) -> Optional[Path]:
    """Locate the compiled ELF artifact within an Arduino build directory.

    Parameters
    ----------
    build_dir : pathlib.Path
        Arduino CLI build directory.

    Returns
    -------
    pathlib.Path | None
        Path to the preferred ELF file when present, otherwise ``None``.
    """
    candidates = sorted(build_dir.glob("*.elf"))
    if not candidates:
        candidates = sorted(build_dir.rglob("*.elf"))
    if not candidates:
        return None
    # Prefer ino ELF artifacts when multiple files exist.
    ino_candidates = [path for path in candidates if path.name.endswith(".ino.elf")]
    return ino_candidates[0] if ino_candidates else candidates[0]


def _resolve_arm_size_binary() -> Optional[str]:
    """Resolve an ``arm-none-eabi-size`` binary for ELF section accounting.

    Returns
    -------
    str | None
        Resolved executable path, or ``None`` when unavailable.
    """
    env_bin = os.environ.get("ARM_NONE_EABI_SIZE_BIN")
    if env_bin:
        return env_bin
    path_bin = shutil.which("arm-none-eabi-size")
    if path_bin:
        return path_bin

    search_root = _PROJECT_ROOT / "tools" / "arduino-data" / "packages"
    if not search_root.exists():
        return None
    candidates = list(search_root.rglob("arm-none-eabi-size"))
    if os.name == "nt":
        candidates.extend(search_root.rglob("arm-none-eabi-size.exe"))
    if not candidates:
        return None
    return str(sorted(candidates)[-1])


def _augment_upload_error(log: str) -> str:
    """Append actionable Linux DFU guidance to permission-related upload failures.

    Parameters
    ----------
    log : str
        Raw upload log text.

    Returns
    -------
    str
        Original or augmented log with Linux udev remediation steps.
    """
    for pattern in UPLOAD_PERMISSION_PATTERNS:
        if pattern.search(log):
            return (
                f"{log}\n"
                "Upload failed due to insufficient USB DFU permissions.\n"
                "On Linux, install the appropriate Portenta udev rules for the board's DFU "
                "USB IDs as documented in this project's README (Portenta DFU/udev section), "
                "reload the rules "
                "(`sudo udevadm control --reload-rules && sudo udevadm trigger`), "
                "then replug the board and retry.\n"
                "macOS typically does not require this step."
            )
    return log


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


def _resolve_build_dir(
    sketch_path: Path,
    fqbn: str,
    build_defines: Optional[Dict[str, int]] = None,
    board_options: Optional[Mapping[str, str]] = None,
) -> Path:
    """Return the Arduino build cache directory for a sketch/FQBN/define set.

    The cache path includes a short hash of compile-time defines so changing
    ``--build-property compiler.cpp.extra_flags`` cannot accidentally reuse
    artifacts from a previous define set.
    """
    build_cache_root = sketch_path / ".arduino-build"
    fqbn_key = fqbn.replace(":", "_")
    define_flags = _build_define_flags(build_defines)
    board_options_text = _format_board_options(board_options)
    hash_input = "|".join(part for part in (define_flags, board_options_text) if part)
    if not hash_input:
        return build_cache_root / fqbn_key
    defines_hash = hashlib.sha1(hash_input.encode("utf-8")).hexdigest()[:12]
    return build_cache_root / f"{fqbn_key}_{defines_hash}"


def _build_define_flags(build_defines: Optional[Dict[str, int]]) -> str:
    """Format Arduino compile-time define flags from a key/value map."""
    if not build_defines:
        return ""
    parts: List[str] = []
    for name in sorted(build_defines):
        parts.append(f"-D{name}={int(build_defines[name])}")
    return " ".join(parts)


def compile_sketch(
    *,
    sketch_path: Path,
    fqbn: str,
    build_defines: Optional[Dict[str, int]] = None,
    board_options: Optional[Mapping[str, str]] = None,
) -> CompileResult:
    """Compile an Arduino sketch with the Arduino CLI.

    Parameters
    ----------
    sketch_path : pathlib.Path
        Directory containing the Arduino sketch.
    fqbn : str
        Fully qualified board name.
    build_defines : dict[str, int] | None, optional
        Preprocessor defines forwarded to ``compiler.cpp.extra_flags``.
    board_options : Mapping[str, str] | None, optional
        Arduino board options forwarded via ``--board-options``.

    Returns
    -------
    CompileResult
        Parsed compile results including RAM/flash usage and overflow status.
    """
    build_dir = _resolve_build_dir(
        sketch_path,
        fqbn,
        build_defines,
        board_options=board_options,
    )
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
    board_options_text = _format_board_options(board_options)
    if board_options_text:
        compile_cmd.extend(["--board-options", board_options_text])
    define_flags = _build_define_flags(build_defines)
    if define_flags:
        compile_cmd.extend(
            [
                "--build-property",
                f"compiler.cpp.extra_flags={define_flags}",
            ]
        )
    compile_proc = subprocess.run(
        compile_cmd, capture_output=True, text=True, check=False
    )
    compile_log = f"{compile_proc.stdout}\n{compile_proc.stderr}"
    flash_bytes, ram_bytes = _parse_memory_from_compile(compile_log)
    if compile_proc.returncode == 0 and (flash_bytes is None or ram_bytes is None):
        recipe_flash_bytes, recipe_ram_bytes = _parse_memory_from_size_recipe(build_dir)
        if flash_bytes is None:
            flash_bytes = recipe_flash_bytes
        if ram_bytes is None:
            ram_bytes = recipe_ram_bytes
    overflow_kind = _classify_compile_failure(compile_log)
    return CompileResult(
        success=compile_proc.returncode == 0,
        log=compile_log,
        flash_bytes=flash_bytes,
        ram_bytes=ram_bytes,
        overflow_kind=overflow_kind,
        build_dir=build_dir,
    )


def compile_harness_sketch(
    *,
    sketch_path: Optional[Path] = None,
    fqbn: str,
    build_defines: Optional[Dict[str, int]] = None,
) -> CompileResult:
    """Compile the harness sketch without patching deployment constants.

    The harness keeps local `common/` headers for Arduino include compatibility.
    Before each compile, shared protocol/power headers from `sketches/common/`
    are copied into the harness `common/` directory so there is one source of
    truth and no manual sync step.
    """
    target_path = sketch_path or _HARNESS_SKETCH_DIR
    harness_common_dir = target_path / "common"
    harness_common_dir.mkdir(parents=True, exist_ok=True)
    for header_name in _SHARED_HEADER_NAMES:
        shared_header = _SHARED_COMMON_DIR / header_name
        if not shared_header.exists():
            raise FileNotFoundError(f"Shared header not found: {shared_header}")
        shutil.copy2(shared_header, harness_common_dir / header_name)
    return compile_sketch(
        sketch_path=target_path,
        fqbn=fqbn,
        build_defines=build_defines,
    )


def upload_harness_sketch(
    *,
    sketch_path: Optional[Path],
    fqbn: str,
    build_dir: Path,
    serial_port: str,
) -> UploadResult:
    """Upload a compiled harness sketch."""
    target_path = sketch_path or _HARNESS_SKETCH_DIR
    return upload_sketch(
        sketch_path=target_path,
        fqbn=fqbn,
        build_dir=build_dir,
        serial_port=serial_port,
    )


def upload_sketch(
    *,
    sketch_path: Path,
    fqbn: str,
    build_dir: Path,
    serial_port: str,
    board_options: Optional[Mapping[str, str]] = None,
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
    board_options : Mapping[str, str] | None, optional
        Arduino board options forwarded via ``--board-options``.

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
    board_options_text = _format_board_options(board_options)
    if board_options_text:
        upload_cmd.extend(["--board-options", board_options_text])
    upload_proc = subprocess.run(upload_cmd, capture_output=True, text=True, check=False)
    upload_log = _augment_upload_error(f"{upload_proc.stdout}\n{upload_proc.stderr}")
    return UploadResult(success=upload_proc.returncode == 0, log=upload_log)


def ensure_harness_firmware(
    *,
    harness_serial_port: str,
    harness_fqbn: str = "arduino:mbed_nano:nano33ble",
    harness_auto_flash: str = "once",
    build_defines: Optional[Dict[str, int]] = None,
    force: bool = False,
) -> bool:
    """Compile/upload harness firmware when required by flash policy.

    Parameters
    ----------
    harness_serial_port : str
        Serial port for the harness board.
    harness_fqbn : str, optional
        Harness board FQBN.
    harness_auto_flash : str, optional
        Flash policy (``once``, ``always``, ``never``).
    build_defines : dict[str, int] | None, optional
        Compile-time defines forwarded to harness build.
    force : bool, optional
        Ignore ``once`` cache and force a reflash.

    Returns
    -------
    bool
        ``True`` when a flash was performed, otherwise ``False``.
    """
    harness_auto_flash = (harness_auto_flash or "once").lower()
    define_flags = _build_define_flags(build_defines)
    flashed_signature = f"{harness_fqbn}|{define_flags}"
    if harness_auto_flash == "never":
        return False
    if harness_auto_flash == "once" and not force:
        if _HARNESS_FLASHED_SIGNATURES.get(harness_serial_port, "") == flashed_signature:
            return False

    logger.info(
        "ensure_harness_firmware: compiling/uploading harness (port=%s, force=%s)",
        harness_serial_port,
        force,
    )
    compile_result = compile_harness_sketch(
        fqbn=harness_fqbn,
        build_defines=build_defines,
    )
    if not compile_result.success or compile_result.build_dir is None:
        raise RuntimeError("Harness compile failed.")
    upload_result = upload_harness_sketch(
        sketch_path=None,
        fqbn=harness_fqbn,
        build_dir=compile_result.build_dir,
        serial_port=harness_serial_port,
    )
    if not upload_result.success:
        raise RuntimeError("Harness upload failed.")

    _HARNESS_FLASHED_SIGNATURES[harness_serial_port] = flashed_signature
    time.sleep(0.5)
    return True


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
    "harness_latency_s": (
        "float",
        re.compile(rf"^harness timer output.*?:\s*{_FLOAT_CAPTURE}$", re.IGNORECASE),
    ),
}

POWER_METRIC_DEFAULTS: Dict[str, float] = {
    "sequence": -1.0,
    "energy_mj_per_inference": -1.0,
    "avg_power_mw": -1.0,
    "avg_current_ma": -1.0,
    "bus_voltage_v": -1.0,
    "idle_power_mw": -1.0,
    "harness_latency_s": -1.0,
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
    latency_s, arena_error_line = _parse_latency_from_log(decoded_lines)
    return latency_s, arena_error_line, decoded_lines


def _parse_latency_from_log(
    decoded_lines: Sequence[str],
) -> Tuple[Optional[float], Optional[str]]:
    """Parse latency and arena errors from decoded serial lines."""
    for line in decoded_lines:
        lower_line = line.lower()
        if lower_line.startswith("timer output:"):
            _, _, value = line.partition(":")
            try:
                return float(value.strip()), None
            except ValueError:
                return None, None
        if any(pattern.search(lower_line) for pattern in ARENA_TOO_SMALL_PATTERNS):
            return None, line
    return None, None


def measure_serial(
    *,
    serial_port: str,
    baud_rate: int,
    serial_timeout_s: float,
    dut_ready_timeout_s: float = 5.0,
    harness_serial_port: Optional[str] = None,
    harness_fqbn: str = "arduino:mbed_nano:nano33ble",
    harness_auto_flash: str = "once",
    harness_arm_pin: int = 3,
    harness_trigger_pin: int = 2,
    dut_arm_hold_ms: int = 600,
    harness_stable_low_ms: int = 500,
    harness_ready_timeout_s: float = 5.0,
    harness_arm_timeout_s: float = 5.0,
    harness_active_timeout_s: float = 30.0,
    harness_done_timeout_s: float = 5.0,
) -> MeasureResult:
    """Capture latency and power metrics from a device serial log.

    Parameters
    ----------
    serial_port : str
        DUT serial port used for upload-time handshake and timer capture.
    baud_rate : int
        Serial baud rate.
    serial_timeout_s : float
        Timeout waiting for DUT ``timer output:``.
    dut_ready_timeout_s : float
        Time to wait for DUT READY before sending START.
    harness_serial_port : str | None
        Harness serial port. When None, run DUT-only measurement without harness
        handshake or energy parsing.
    harness_fqbn : str
        Arduino FQBN used to compile/upload the harness sketch.
    harness_auto_flash : str
        Harness flashing policy: ``once``, ``always``, or ``never``.
    harness_arm_pin : int
        D3 arm pin number used by DUT/harness.
    harness_trigger_pin : int
        D2 trigger pin number used by DUT/harness.
    dut_arm_hold_ms : int
        DUT delay between asserting arm LOW and raising trigger HIGH.
    harness_stable_low_ms : int
        Harness stable-low window before arming.
    harness_ready_timeout_s : float
        Timeout waiting for ``HARNESS READY`` after PING.
    harness_arm_timeout_s : float
        Harness arm timeout in seconds. This value is converted to milliseconds
        and compiled into the harness firmware (`TINYODOM_HARNESS_ARM_TIMEOUT_MS`).
        Set to ``0`` to disable harness-side arm timeout.
    harness_active_timeout_s : float
        Maximum allowed active measurement window in seconds. This value is used
        both for host-side wait bounds and for the harness firmware timeout
        (`TINYODOM_HARNESS_ACTIVE_TIMEOUT_MS`).
    harness_done_timeout_s : float
        Timeout waiting for harness ``DONE`` after the active window completes.

    Returns
    -------
    MeasureResult
        Parsed latency, arena errors, merged serial log, and optional power
        metrics.
    """
    if not harness_serial_port:
        logger.info("measure_serial: running DUT-only flow on %s", serial_port)
        dut_log = hil_protocol.run_dut_only(
            dut_port=serial_port,
            baud_rate=baud_rate,
            dut_ready_timeout_s=dut_ready_timeout_s,
            dut_timer_timeout_s=serial_timeout_s,
        )
        latency_s, arena_error_line = _parse_latency_from_log(dut_log)
        power_metrics = _parse_power_metrics(dut_log)
        return MeasureResult(
            latency_s=latency_s,
            arena_error_line=arena_error_line,
            serial_log=dut_log,
            power_metrics=power_metrics,
        )
    harness_arm_timeout_ms = max(0, int(round(float(harness_arm_timeout_s) * 1000.0)))
    harness_active_timeout_ms = max(0, int(round(float(harness_active_timeout_s) * 1000.0)))

    build_defines = {
        "TINYODOM_HARNESS_ARM_PIN": int(harness_arm_pin),
        "TINYODOM_HARNESS_TRIGGER_PIN": int(harness_trigger_pin),
        "TINYODOM_DUT_ARM_HOLD_MS": int(dut_arm_hold_ms),
        "TINYODOM_HARNESS_STABLE_LOW_MS": int(harness_stable_low_ms),
        "TINYODOM_HARNESS_ARM_TIMEOUT_MS": harness_arm_timeout_ms,
        "TINYODOM_HARNESS_ACTIVE_TIMEOUT_MS": harness_active_timeout_ms,
    }

    def _flash_harness(force: bool = False) -> bool:
        return ensure_harness_firmware(
            harness_serial_port=harness_serial_port,
            harness_fqbn=harness_fqbn,
            harness_auto_flash=harness_auto_flash,
            build_defines=build_defines,
            force=force,
        )

    def _run_handshake() -> hil_protocol.HandshakeResult:
        logger.info(
            "measure_serial: starting DUT+harness handshake (dut=%s, harness=%s)",
            serial_port,
            harness_serial_port,
        )
        return hil_protocol.run_handshake(
            dut_port=serial_port,
            harness_port=harness_serial_port,
            baud_rate=baud_rate,
            dut_ready_timeout_s=dut_ready_timeout_s,
            dut_timer_timeout_s=serial_timeout_s,
            harness_ready_timeout_s=harness_ready_timeout_s,
            harness_active_timeout_s=harness_active_timeout_s,
            harness_done_timeout_s=harness_done_timeout_s,
        )

    try:
        _flash_harness()
        result = _run_handshake()
        if result.error:
            logger.warning("Harness handshake failed: %s", result.error)
            logger.info(
                "measure_serial: skipping harness reflash for protocol error '%s'",
                result.error,
            )
            logger.info("measure_serial: falling back to DUT-only flow after harness error")
            dut_log = hil_protocol.run_dut_only(
                dut_port=serial_port,
                baud_rate=baud_rate,
                dut_ready_timeout_s=dut_ready_timeout_s,
                dut_timer_timeout_s=serial_timeout_s,
            )
            latency_s, arena_error_line = _parse_latency_from_log(dut_log)
            serial_log = [f"HANDSHAKE_ERROR: {result.error}"]
            serial_log.extend(f"HARNESS_PRE: {line}" for line in result.harness_log)
            serial_log.extend(f"DUT_PRE: {line}" for line in result.dut_log)
            serial_log.extend(f"DUT: {line}" for line in dut_log)
            return MeasureResult(
                latency_s=latency_s,
                arena_error_line=arena_error_line,
                serial_log=serial_log,
                power_metrics=None,
            )

        latency_s, arena_error_line = _parse_latency_from_log(result.dut_log)
        serial_log = [f"DUT: {line}" for line in result.dut_log] + [
            f"HARNESS: {line}" for line in result.harness_log
        ]
        power_metrics = None
        if not result.dut_timer_found or latency_s is None:
            logger.warning("DUT timer output missing; ignoring harness energy metrics.")
        elif not result.harness_done:
            logger.warning("Harness did not report DONE; ignoring energy metrics.")
        elif result.runs_dut is None or result.runs_harness is None:
            logger.warning("Missing DUT/harness run count; ignoring energy metrics.")
        elif result.runs_dut != result.runs_harness:
            logger.warning(
                "Harness run-count mismatch (dut=%s harness=%s)",
                result.runs_dut,
                result.runs_harness,
            )
        else:
            power_metrics = _parse_power_metrics(result.harness_log)
        return MeasureResult(
            latency_s=latency_s,
            arena_error_line=arena_error_line,
            serial_log=serial_log,
            power_metrics=power_metrics,
        )
    except (RuntimeError, serial.SerialException) as exc:  # type: ignore[attr-defined]
        logger.warning("Harness measurement failed: %s", exc)
        if isinstance(exc, serial.SerialException):
            try:
                logger.info(
                    "measure_serial: serial-port error detected; attempting harness reflash recovery"
                )
                _flash_harness(force=True)
            except RuntimeError as reflash_exc:
                logger.warning("Harness reflash after serial error failed: %s", reflash_exc)
        logger.info("measure_serial: exception fallback to DUT-only flow")
        dut_log = hil_protocol.run_dut_only(
            dut_port=serial_port,
            baud_rate=baud_rate,
            dut_ready_timeout_s=dut_ready_timeout_s,
            dut_timer_timeout_s=serial_timeout_s,
        )
        latency_s, arena_error_line = _parse_latency_from_log(dut_log)
        serial_log = [f"DUT: {line}" for line in dut_log]
        return MeasureResult(
            latency_s=latency_s,
            arena_error_line=arena_error_line,
            serial_log=serial_log,
            power_metrics=None,
        )


def measure_harness_only_open_session(
    *,
    harness: serial.Serial,
    harness_active_timeout_s: float = 30.0,
    harness_done_timeout_s: float = 5.0,
    harness_log: Optional[list[str]] = None,
) -> MeasureResult:
    """Collect metrics from an already-open harness session.

    Parameters
    ----------
    harness : serial.Serial
        Open harness serial connection that remained active through DUT upload.
    harness_active_timeout_s : float, optional
        Active-window timeout used to bound DONE wait.
    harness_done_timeout_s : float, optional
        Additional DONE timeout after active window.
    harness_log : list[str] | None, optional
        Existing harness log lines captured before this call.

    Returns
    -------
    MeasureResult
        Parsed latency/power from harness output only.
    """
    log = list(harness_log or [])
    session_result = hil_protocol.wait_for_harness_done(
        harness=harness,
        harness_active_timeout_s=harness_active_timeout_s,
        harness_done_timeout_s=harness_done_timeout_s,
        harness_log=log,
    )
    log = session_result.harness_log
    power_metrics = _parse_power_metrics(log)
    latency_s: Optional[float] = None
    if power_metrics is not None:
        harness_latency = power_metrics.get("harness_latency_s")
        if harness_latency is not None and harness_latency >= 0.0:
            latency_s = float(harness_latency)

    serial_log = [f"HARNESS: {line}" for line in log]
    if session_result.error is not None:
        serial_log.insert(0, f"HARNESS_ERROR: {session_result.error}")
    return MeasureResult(
        latency_s=latency_s if session_result.harness_done else None,
        arena_error_line=None,
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
    "_parse_latency_from_log",
    "_parse_memory_from_compile",
    "_parse_power_metrics",
    "_patch_sketch_constants",
    "_replace_define",
    "compile_harness_sketch",
    "compile_sketch",
    "ensure_harness_firmware",
    "measure_harness_only_open_session",
    "measure_serial",
    "normalize_power_metrics",
    "upload_harness_sketch",
    "upload_sketch",
]
