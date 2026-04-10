#!/usr/bin/env python3
"""
Run one end-to-end STM32 toy AI HIL attempt and emit a compact metrics JSON.

This script keeps the STM32 flow standalone from the main TinyODOM backend
while still reusing the existing safe debug-load path and the shared Arduino
harness firmware contract.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import serial

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))

from build_and_upload_stm32_blink import (  # noqa: E402
    WorkflowError,
    _build_project,
    _configure_logging,
    _default_cubeprog_bin,
    _resolve_required_tool_path,
    _run_gdb_load,
    _start_gdb_server,
    _summarize_server_log,
    _validate_paths,
    _wait_for_server_ready,
    DEFAULT_APID,
    DEFAULT_GDB_PORT,
    SERVER_READY_TIMEOUT_S,
    SERVER_SETTLE_DELAY_S,
)
from tinyodom.microcontrollers.arduino_base import ensure_harness_firmware  # noqa: E402

DEFAULT_PROJECT_ROOT = SCRIPT_DIR / "stm32_toy_ai_project" / "FSBL"
DEFAULT_DUT_PORT = "/dev/ttyACM0"
DEFAULT_HARNESS_PORT = "/dev/ttyACM1"
DEFAULT_BAUD = 115200
DEFAULT_LATENCY_BUDGET_MS = 200.0
DEFAULT_OUTPUT = SCRIPT_DIR / "stm32_toy_ai_metrics.json"
DEFAULT_CONFIG = REPO_ROOT / "src" / "nas_config.yaml"
DEFAULT_STAGE_OUTPUT_ROOT = Path("/tmp/tinyodom_stm32_toy_generate")
DEFAULT_HARNESS_FQBN = "arduino:mbed_nano:nano33ble"
DEFAULT_HARNESS_AUTO_FLASH = "once"
DEFAULT_WEIGHT_STORAGE_MODE = "embedded"
DEFAULT_WEIGHTS_FLASH_ADDRESS = "0x71000000"
DEFAULT_WEIGHTS_MEMORY_POOL = SCRIPT_DIR / "nucleo_mypool.json"
DEFAULT_WEIGHTS_EXTERNAL_LOADER_NAME = "MX25UM51245G_STM32N6570-NUCLEO.stldr"
DEFAULT_WEIGHTS_PROGRAMMER_MODE = "hotplug"
DEFAULT_WEIGHTS_PROGRAMMER_FREQ_KHZ = "500"
DEFAULT_WEIGHTS_RECOVERY_MODE = "powerdown"
DEFAULT_WEIGHTS_RECOVERY_APID = "0"
STAGING_MANIFEST_NAME = "staging_manifest.json"

MASTER_PENDING = 0
MASTER_SUCCESS = 1
MASTER_ARENA_EXHAUSTED = 2
MASTER_FATAL = 3
MASTER_FLASH_OVERFLOW = 4
MASTER_RAM_OVERFLOW = 5
MASTER_DEVICE_NOT_FOUND = 6
ERROR_LABELS = {
    MASTER_PENDING: "HIL_MASTER_PENDING",
    MASTER_SUCCESS: "HIL_MASTER_SUCCESS",
    MASTER_ARENA_EXHAUSTED: "HIL_MASTER_ARENA_EXHAUSTED",
    MASTER_FATAL: "HIL_MASTER_FATAL",
    MASTER_FLASH_OVERFLOW: "HIL_MASTER_FLASH_OVERFLOW",
    MASTER_RAM_OVERFLOW: "HIL_MASTER_RAM_OVERFLOW",
    MASTER_DEVICE_NOT_FOUND: "HIL_MASTER_DEVICE_NOT_FOUND",
}

LOGGER = logging.getLogger(__name__)

SIZE_RE = re.compile(
    r"^\s*(?P<text>\d+)\s+(?P<data>\d+)\s+(?P<bss>\d+)\s+(?P<dec>\d+)\s+(?P<hex>[0-9a-fA-F]+)\s+",
    re.MULTILINE,
)
ARENA_RE = re.compile(r"AI_NETWORK_DATA_ACTIVATIONS_SIZE\s+\((?P<value>\d+)\)")
HEX_DEFINE_RE = re.compile(r"(?P<name>_Min_(?:Heap|Stack)_Size)\s*=\s*(?P<value>0x[0-9A-Fa-f]+)")
RUNS_RE = re.compile(r"^runs:\s*(?P<value>\d+)$", re.IGNORECASE)
FLOAT_CAPTURE = r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
HARNESS_FLOAT_PATTERNS = {
    "energy_mj_per_inference": re.compile(rf"^energy output.*?:\s*{FLOAT_CAPTURE}$", re.IGNORECASE),
    "avg_power_mw": re.compile(rf"^avg power output.*?:\s*{FLOAT_CAPTURE}$", re.IGNORECASE),
    "avg_current_ma": re.compile(rf"^avg current output.*?:\s*{FLOAT_CAPTURE}$", re.IGNORECASE),
    "bus_voltage_v": re.compile(rf"^bus voltage output.*?:\s*{FLOAT_CAPTURE}$", re.IGNORECASE),
    "idle_power_mw": re.compile(rf"^idle power baseline.*?:\s*{FLOAT_CAPTURE}$", re.IGNORECASE),
    "harness_latency_s": re.compile(rf"^harness timer output.*?:\s*{FLOAT_CAPTURE}$", re.IGNORECASE),
}
DUT_FLOAT_PATTERNS = {
    "clock_hz": re.compile(rf"^clock hz output.*?:\s*{FLOAT_CAPTURE}$", re.IGNORECASE),
    "dwt_cycles_per_inference": re.compile(
        rf"^dwt cycles per inference output.*?:\s*{FLOAT_CAPTURE}$",
        re.IGNORECASE,
    ),
    "timer_output_s": re.compile(rf"^timer output.*?:\s*{FLOAT_CAPTURE}$", re.IGNORECASE),
}


class SerialMonitor:
    """Background serial reader with queued line delivery and write support."""

    def __init__(self, port: str, baud: int, label: str) -> None:
        """Open the serial port and start the background reader thread.

        Parameters
        ----------
        port : str
            Serial device path, e.g. ``"/dev/ttyACM0"`` or ``"COM3"``.
        baud : int
            Baud rate, e.g. ``115200``.
        label : str
            Human-readable label used in log messages (e.g. ``"dut"`` or ``"harness"``).
        """
        self.port = port
        self.baud = baud
        self.label = label
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._lines: list[str] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._serial = serial.Serial(port, baud, timeout=0.2)
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        """Background thread: read lines from the serial port into the queue."""
        try:
            while not self._stop.is_set():
                raw = self._serial.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line:
                    continue
                with self._lock:
                    self._lines.append(line)
                self._queue.put(line)
        except serial.SerialException as exc:
            LOGGER.error("%s serial error: %s", self.label, exc)
        finally:
            self._queue.put(None)

    def write_line(self, text: str) -> None:
        """Write a text line followed by a newline to the serial port.

        Parameters
        ----------
        text : str
            Command string to send (a ``\\n`` is appended automatically).
        """
        payload = f"{text}\n".encode("utf-8")
        self._serial.write(payload)
        self._serial.flush()

    def wait_for(self, predicate, timeout_s: float, stage: str) -> str | None:
        """Block until a line matching *predicate* arrives or the timeout elapses.

        Parameters
        ----------
        predicate : callable[[str], bool]
            Function that receives each incoming line and returns ``True``
            when the desired condition is met.
        timeout_s : float
            Maximum seconds to wait before returning ``None``.
        stage : str
            Human-readable name for the awaited event, used in error messages.

        Returns
        -------
        str or None
            The first matching line, or ``None`` if the timeout elapsed without a match.

        Raises
        ------
        RuntimeError
            Raised if the serial reader thread exits unexpectedly while waiting.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                line = self._queue.get(timeout=min(0.5, max(0.01, deadline - time.monotonic())))
            except queue.Empty:
                continue
            if line is None:
                raise RuntimeError(f"{self.label} serial reader exited while waiting for {stage}.")
            LOGGER.info("[%s] %s", self.label, line)
            if predicate(line):
                return line
        return None

    def snapshot_lines(self) -> list[str]:
        """Return a thread-safe snapshot of all lines received so far.

        Returns
        -------
        list[str]
            Copy of all lines captured since the monitor was opened.
        """
        with self._lock:
            return list(self._lines)

    def close(self) -> None:
        """Signal the reader thread to stop and close the serial port."""
        self._stop.set()
        try:
            self._thread.join(timeout=2.0)
        finally:
            if self._serial.is_open:
                self._serial.close()


def _which_path(name: str) -> Path | None:
    """Resolve a command name to its absolute path using the system PATH.

    Parameters
    ----------
    name : str
        Executable name to look up, e.g. ``"arm-none-eabi-gdb"``.

    Returns
    -------
    Path or None
        Absolute path to the executable if found on PATH, otherwise ``None``.
    """
    value = shutil.which(name)
    return Path(value) if value is not None else None


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the STM32 toy AI HIL runner.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser with all CLI options for paths, timeouts, serial
        ports, hardware pins, and weight storage settings.
    """
    parser = argparse.ArgumentParser(description="Run one STM32 toy AI HIL attempt.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help=(
            "STM32 FSBL project root that contains the Makefile and Src/Inc directories. "
            f"Default: {DEFAULT_PROJECT_ROOT}. "
            "Examples: path/to/stm32_toy_ai_project/FSBL, /workspace/myproject/FSBL"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=(
            "TinyODOM NAS configuration YAML used to generate and export the perturbed model. "
            f"Default: {DEFAULT_CONFIG}. "
            "Examples: src/nas_config.yaml, configs/my_experiment.yaml"
        ),
    )
    parser.add_argument(
        "--dut-port",
        default=DEFAULT_DUT_PORT,
        help=(
            "Serial port of the device under test (STM32 NUCLEO board). "
            f"Default: {DEFAULT_DUT_PORT}. "
            "Examples: /dev/ttyACM0, /dev/ttyUSB0, COM3"
        ),
    )
    parser.add_argument(
        "--harness-port",
        default=DEFAULT_HARNESS_PORT,
        help=(
            "Serial port of the Arduino energy-measurement harness. "
            f"Default: {DEFAULT_HARNESS_PORT}. "
            "Examples: /dev/ttyACM1, /dev/ttyUSB1, COM4"
        ),
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=DEFAULT_BAUD,
        help=(
            "Baud rate for both the DUT and harness serial connections. "
            f"Default: {DEFAULT_BAUD}. "
            "Examples: 115200, 9600, 921600"
        ),
    )
    parser.add_argument(
        "--latency-budget-ms",
        type=float,
        default=DEFAULT_LATENCY_BUDGET_MS,
        help=(
            "Inference latency budget in milliseconds written into the output metrics. "
            f"Default: {DEFAULT_LATENCY_BUDGET_MS}. "
            "Examples: 200.0, 100.0, 50.0"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Destination path for the compact metrics JSON file. "
            f"Default: {DEFAULT_OUTPUT}. "
            "Examples: results/run1.json, /tmp/metrics.json"
        ),
    )
    parser.add_argument(
        "--stage-output-root",
        type=Path,
        default=DEFAULT_STAGE_OUTPUT_ROOT,
        help=(
            "Temporary directory used by the staging script for generated model and ST Edge AI artifacts. "
            f"Default: {DEFAULT_STAGE_OUTPUT_ROOT}. "
            "Examples: /tmp/my_staging, /workspace/staging_out"
        ),
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Wipe the staging output root and perform a clean STM32 build before running.",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help=(
            "Skip the STM32 build step and use the existing ELF. "
            "Requires --reuse-staged-model since the default flow regenerates sources before building."
        ),
    )
    parser.add_argument(
        "--reuse-staged-model",
        action="store_true",
        help="Reuse the currently staged STM32 network sources instead of regenerating a fresh perturbed TinyODOM model.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=os.cpu_count() or 1,
        help=(
            "Number of parallel make jobs for the STM32 build. "
            f"Default: number of CPU cores ({os.cpu_count() or 1}). "
            "Examples: 4, 8, 1"
        ),
    )
    parser.add_argument(
        "--serial-timeout",
        type=float,
        default=30.0,
        help=(
            "Seconds to allow for the DUT inference run to complete (base timeout for harness DONE). "
            "Default: 30.0. "
            "Examples: 30.0, 60.0, 120.0"
        ),
    )
    parser.add_argument(
        "--dut-ready-timeout",
        type=float,
        default=30.0,
        help=(
            "Seconds to wait for the DUT to emit 'DUT READY' after GDB loads the firmware. "
            "Default: 30.0. "
            "Examples: 30.0, 15.0, 60.0"
        ),
    )
    parser.add_argument(
        "--harness-ready-timeout",
        type=float,
        default=5.0,
        help=(
            "Seconds to wait for 'HARNESS READY' in response to the initial PING. "
            "Default: 5.0. "
            "Examples: 5.0, 2.0, 10.0"
        ),
    )
    parser.add_argument(
        "--harness-done-timeout",
        type=float,
        default=10.0,
        help=(
            "Extra seconds added to --serial-timeout when waiting for the harness 'DONE' message. "
            "Default: 10.0. "
            "Examples: 10.0, 5.0, 30.0"
        ),
    )
    parser.add_argument(
        "--harness-fqbn",
        default=DEFAULT_HARNESS_FQBN,
        help=(
            "Arduino Fully Qualified Board Name (FQBN) for compiling and flashing harness firmware. "
            f"Default: {DEFAULT_HARNESS_FQBN}. "
            "Examples: arduino:mbed_nano:nano33ble, arduino:avr:uno, arduino:samd:mkr1000"
        ),
    )
    parser.add_argument(
        "--harness-auto-flash",
        choices=["once", "always", "never"],
        default=DEFAULT_HARNESS_AUTO_FLASH,
        help=(
            "Controls when the harness Arduino firmware is recompiled and flashed. "
            "'once' flashes only if not already up-to-date, 'always' forces a reflash, "
            f"'never' skips flashing entirely. Default: {DEFAULT_HARNESS_AUTO_FLASH}."
        ),
    )
    parser.add_argument(
        "--harness-arm-pin",
        type=int,
        default=3,
        help=(
            "Arduino GPIO pin number used to send the ARM signal to the DUT. "
            "Default: 3. "
            "Examples: 3, 4, 7"
        ),
    )
    parser.add_argument(
        "--harness-trigger-pin",
        type=int,
        default=2,
        help=(
            "Arduino GPIO pin number that detects the DUT trigger pulse to start/stop timing. "
            "Default: 2. "
            "Examples: 2, 5, 8"
        ),
    )
    parser.add_argument(
        "--dut-arm-hold-ms",
        type=int,
        default=600,
        help=(
            "Milliseconds to hold the ARM signal high before releasing, giving the DUT time to prepare. "
            "Default: 600. "
            "Examples: 600, 300, 1000"
        ),
    )
    parser.add_argument(
        "--harness-stable-low-ms",
        type=int,
        default=500,
        help=(
            "Milliseconds the DUT trigger pin must remain low before the harness confirms completion. "
            "Default: 500. "
            "Examples: 500, 200, 1000"
        ),
    )
    parser.add_argument(
        "--weight-storage-mode",
        choices=["embedded", "external_flash"],
        default=DEFAULT_WEIGHT_STORAGE_MODE,
        help=(
            "Where model weights are stored on the target. "
            "'embedded' packs weights into the ELF flash image; "
            "'external_flash' writes a separate blob to NOR flash via STM32_Programmer_CLI. "
            f"Default: {DEFAULT_WEIGHT_STORAGE_MODE}."
        ),
    )
    parser.add_argument(
        "--weights-flash-address",
        default=DEFAULT_WEIGHTS_FLASH_ADDRESS,
        help=(
            "Base address in NOR flash where the external weight blob is programmed. "
            "Only used when --weight-storage-mode=external_flash. "
            f"Default: {DEFAULT_WEIGHTS_FLASH_ADDRESS}. "
            "Examples: 0x71000000, 0x70000000"
        ),
    )
    parser.add_argument(
        "--weights-memory-pool",
        type=Path,
        default=DEFAULT_WEIGHTS_MEMORY_POOL,
        help=(
            "JSON memory-pool specification passed to ST Edge AI when externalizing weights. "
            "Only used when --weight-storage-mode=external_flash. "
            f"Default: {DEFAULT_WEIGHTS_MEMORY_POOL}. "
            "Examples: nucleo_mypool.json, /path/to/custom_pool.json"
        ),
    )
    parser.add_argument(
        "--weights-external-loader",
        type=Path,
        default=None,
        help=(
            "Path to the .stldr external loader file for NOR flash programming. "
            "Auto-discovered under --cubeprog-bin/ExternalLoader if omitted. "
            "Example: /opt/STM32CubeProgrammer/bin/ExternalLoader/MX25UM51245G_STM32N6570-NUCLEO.stldr"
        ),
    )
    parser.add_argument(
        "--gdbserver",
        type=Path,
        default=_which_path("ST-LINK_gdbserver"),
        help=(
            "Path to the ST-LINK_gdbserver executable. Auto-detected from PATH if omitted. "
            "Examples: /opt/st/stm32cubeide/plugins/.../ST-LINK_gdbserver, /usr/local/bin/ST-LINK_gdbserver"
        ),
    )
    parser.add_argument(
        "--gdb",
        type=Path,
        default=_which_path("arm-none-eabi-gdb"),
        help=(
            "Path to the arm-none-eabi-gdb executable. Auto-detected from PATH if omitted. "
            "Examples: /usr/bin/arm-none-eabi-gdb, /opt/gcc-arm/bin/arm-none-eabi-gdb"
        ),
    )
    parser.add_argument(
        "--cubeprog-bin",
        type=Path,
        default=_default_cubeprog_bin(),
        help=(
            "STM32CubeProgrammer bin directory (must contain STM32_Programmer_CLI). "
            "Auto-detected from common install paths if omitted. "
            "Examples: /opt/STM32CubeProgrammer/bin, /usr/local/STM32CubeProgrammer/bin"
        ),
    )
    parser.add_argument(
        "--gdb-port",
        type=int,
        default=DEFAULT_GDB_PORT,
        help=(
            "TCP port for the ST-LINK GDB server to listen on. "
            f"Default: {DEFAULT_GDB_PORT}. "
            "Examples: 61234, 3333, 2331"
        ),
    )
    parser.add_argument(
        "--apid",
        type=int,
        default=DEFAULT_APID,
        help=(
            "ST-Link Access Port ID (AP index) for the main application core. "
            f"Default: {DEFAULT_APID}. "
            "Examples: 1, 0, 2"
        ),
    )
    parser.add_argument(
        "--server-ready-timeout",
        type=float,
        default=SERVER_READY_TIMEOUT_S,
        help=(
            "Seconds to wait for the GDB server to report ready after launch. "
            f"Default: {SERVER_READY_TIMEOUT_S}. "
            "Examples: 30.0, 15.0, 60.0"
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging for more detailed output.",
    )
    return parser


def _run_size(elf_path: Path) -> dict[str, int]:
    """Run ``arm-none-eabi-size`` on an ELF and return parsed section sizes.

    Parameters
    ----------
    elf_path : Path
        Path to the compiled ELF file.

    Returns
    -------
    dict[str, int]
        Parsed section sizes with keys ``text``, ``data``, ``bss``, ``dec``,
        ``hex``, ``elf_flash_bytes`` (text + data), and ``ram_bytes`` (data + bss).

    Raises
    ------
    WorkflowError
        Raised if ``arm-none-eabi-size`` exits non-zero or its output cannot be parsed.
    """
    proc = subprocess.run(
        ["arm-none-eabi-size", str(elf_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise WorkflowError(f"arm-none-eabi-size failed for {elf_path}:\n{proc.stderr}")
    match = SIZE_RE.search(proc.stdout)
    if not match:
        raise WorkflowError(f"Unable to parse arm-none-eabi-size output:\n{proc.stdout}")
    values = {key: int(value, 16) if key == "hex" else int(value) for key, value in match.groupdict().items()}
    values["elf_flash_bytes"] = values["text"] + values["data"]
    values["ram_bytes"] = values["data"] + values["bss"]
    return values


def _parse_linker_reservations(linker_script: Path) -> dict[str, int]:
    """Extract heap and stack size reservations from a linker script.

    Parameters
    ----------
    linker_script : Path
        Path to the ``.ld`` linker script file.

    Returns
    -------
    dict[str, int]
        Dictionary with keys ``heap_bytes`` and/or ``stack_bytes`` parsed
        from ``_Min_Heap_Size`` and ``_Min_Stack_Size`` hex defines.
    """
    text = linker_script.read_text(encoding="utf-8")
    out: dict[str, int] = {}
    for match in HEX_DEFINE_RE.finditer(text):
        key = "heap_bytes" if "Heap" in match.group("name") else "stack_bytes"
        out[key] = int(match.group("value"), 16)
    return out


def _find_linker_script(project_root: Path) -> Path:
    """Locate the first linker script (``.ld``) in the project root.

    Parameters
    ----------
    project_root : Path
        Root directory of the STM32 FSBL project.

    Returns
    -------
    Path
        Path to the first ``.ld`` file found (sorted lexicographically).

    Raises
    ------
    WorkflowError
        Raised if no linker script is found under *project_root*.
    """
    linker_scripts = sorted(project_root.glob("*.ld"))
    if not linker_scripts:
        raise WorkflowError(f"Unable to locate linker script under {project_root}")
    return linker_scripts[0]


def _parse_arena_bytes(network_data_params: Path) -> int:
    """Parse the ST Edge AI activation arena size from a generated header.

    Parameters
    ----------
    network_data_params : Path
        Path to the generated ``network_data_params.h`` header file.

    Returns
    -------
    int
        Value of ``AI_NETWORK_DATA_ACTIVATIONS_SIZE`` in bytes.

    Raises
    ------
    WorkflowError
        Raised if the macro cannot be found or parsed in the file.
    """
    text = network_data_params.read_text(encoding="utf-8")
    match = ARENA_RE.search(text)
    if not match:
        raise WorkflowError(f"Unable to parse activations size from {network_data_params}")
    return int(match.group("value"))


def _parse_runs(lines: list[str]) -> int | None:
    """Extract the last ``runs: <N>`` count reported by a serial monitor.

    Parameters
    ----------
    lines : list[str]
        Lines captured from a serial port (DUT or harness).

    Returns
    -------
    int or None
        The last run count seen, or ``None`` if no ``runs:`` line was found.
    """
    value = None
    for line in lines:
        match = RUNS_RE.match(line.strip())
        if match:
            value = int(match.group("value"))
    return value


def _parse_last_float_fields(lines: list[str], patterns: dict[str, re.Pattern[str]]) -> dict[str, float]:
    """Extract the last match for each named float pattern from serial lines.

    Parameters
    ----------
    lines : list[str]
        Lines captured from a serial port.
    patterns : dict[str, re.Pattern[str]]
        Mapping of metric name to compiled regex. Each pattern must expose a
        ``value`` named group matching the float literal.

    Returns
    -------
    dict[str, float]
        Latest float value seen for each pattern key. Keys without any match
        are absent from the result.
    """
    values: dict[str, float] = {}
    for line in lines:
        stripped = line.strip()
        for key, pattern in patterns.items():
            match = pattern.match(stripped)
            if match:
                values[key] = float(match.group("value"))
    return values


def _diagnostic_output_path(output_path: Path) -> Path:
    """Derive the sibling diagnostics JSON path from the main output path.

    Parameters
    ----------
    output_path : Path
        Path to the primary metrics JSON output file.

    Returns
    -------
    Path
        Sibling path with ``.diagnostics`` inserted before the extension,
        e.g. ``metrics.json`` → ``metrics.diagnostics.json``.
    """
    if output_path.suffix:
        return output_path.with_name(f"{output_path.stem}.diagnostics{output_path.suffix}")
    return output_path.with_name(f"{output_path.name}.diagnostics.json")


def _stage_stm32_network(
    *,
    config_path: Path,
    project_root: Path,
    output_root: Path,
    clean: bool,
    weight_storage_mode: str,
    weights_flash_address: str,
    weights_memory_pool: Path,
    weights_external_loader: Path | None,
) -> None:
    """Regenerate and stage fresh STM32 network sources.

    Parameters
    ----------
    config_path : Path
        TinyODOM YAML configuration used for perturbed-model export.
    project_root : Path
        STM32 FSBL project root that receives the generated sources.
    output_root : Path
        Temporary root used by the staging script for model and ST Edge AI
        artifacts.
    clean : bool
        Whether to clear the staging output root before regeneration.

    Returns
    -------
    None

    Raises
    ------
    WorkflowError
        Raised when the staging subprocess fails.
    """
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "generate_and_stage_stm32_toy_ai.py"),
        "--config",
        str(config_path),
        "--project-root",
        str(project_root),
        "--output-root",
        str(output_root),
        "--weight-storage-mode",
        weight_storage_mode,
        "--weights-flash-address",
        weights_flash_address,
        "--weights-memory-pool",
        str(weights_memory_pool),
    ]
    if weights_external_loader is not None:
        cmd.extend(["--weights-external-loader", str(weights_external_loader)])
    if clean:
        cmd.append("--clean")
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        detail = []
        if proc.stdout.strip():
            detail.append(f"stdout:\n{proc.stdout.strip()}")
        if proc.stderr.strip():
            detail.append(f"stderr:\n{proc.stderr.strip()}")
        joined = "\n\n".join(detail) if detail else "No subprocess output captured."
        raise WorkflowError(f"STM32 model staging failed.\n\n{joined}")
    if proc.stdout.strip():
        LOGGER.info(proc.stdout.strip())


def _read_staging_manifest(output_root: Path) -> dict[str, object]:
    """Load and validate the staging manifest produced by the staging script.

    Parameters
    ----------
    output_root : Path
        Staging output root directory that should contain
        ``staging_manifest.json``.

    Returns
    -------
    dict[str, object]
        Parsed manifest dictionary.

    Raises
    ------
    WorkflowError
        Raised if the manifest file is missing, not valid JSON, or not a dict.
    """
    manifest_path = output_root / STAGING_MANIFEST_NAME
    if not manifest_path.is_file():
        raise WorkflowError(f"Staging manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WorkflowError(f"Unable to parse staging manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise WorkflowError(f"Unexpected staging manifest payload: {manifest_path}")
    return manifest


def _resolve_weights_external_loader(
    cubeprog_bin: Path,
    weights_external_loader: Path | None,
) -> Path:
    """Resolve the NOR-flash external loader path, auto-discovering if needed.

    Parameters
    ----------
    cubeprog_bin : Path
        STM32CubeProgrammer bin directory used for auto-discovery.
    weights_external_loader : Path or None
        Explicit loader path supplied by the caller, or ``None`` to trigger
        auto-discovery under *cubeprog_bin*.

    Returns
    -------
    Path
        Resolved, validated path to the ``.stldr`` external loader file.

    Raises
    ------
    WorkflowError
        Raised if an explicit path does not exist, or if auto-discovery fails
        to find the loader under either expected location.
    """
    if weights_external_loader is not None:
        loader_path = weights_external_loader.resolve()
        if not loader_path.is_file():
            raise WorkflowError(f"Weights external loader not found: {loader_path}")
        return loader_path
    candidate_paths = [
        cubeprog_bin / "ExternalLoader" / DEFAULT_WEIGHTS_EXTERNAL_LOADER_NAME,
        cubeprog_bin.parent / "ExternalLoader" / DEFAULT_WEIGHTS_EXTERNAL_LOADER_NAME,
    ]
    for loader_path in candidate_paths:
        if loader_path.is_file():
            return loader_path
    raise WorkflowError(
        "Required Nucleo external loader was not found in either expected location:\n"
        f"  - {candidate_paths[0]}\n"
        f"  - {candidate_paths[1]}\n"
        "Pass --weights-external-loader explicitly."
    )


def _program_weights_blob(
    *,
    cubeprog_bin: Path,
    apid: int,
    weights_blob_path: Path,
    weights_flash_address: str,
    external_loader: Path,
) -> None:
    """Program the model-weights blob into NOR flash via STM32_Programmer_CLI.

    First issues a powerdown recovery command to bring the target to a known
    state, then programs the blob at the specified flash address using the
    supplied external loader.

    Parameters
    ----------
    cubeprog_bin : Path
        STM32CubeProgrammer bin directory (must contain ``STM32_Programmer_CLI``).
    apid : int
        ST-Link Access Port ID for the application core.
    weights_blob_path : Path
        Path to the binary weights blob file to program.
    weights_flash_address : str
        Hex address string (e.g. ``"0x71000000"``) where the blob is written.
    external_loader : Path
        Path to the ``.stldr`` external loader for the target NOR flash chip.

    Raises
    ------
    WorkflowError
        Raised if the recovery command or the programming command exits non-zero.
    """
    programmer = cubeprog_bin / "STM32_Programmer_CLI"
    if not programmer.is_file():
        raise WorkflowError(f"STM32_Programmer_CLI not found under {cubeprog_bin}")
    recovery_cmd = [
        str(programmer),
        "-q",
        "-c",
        "port=SWD",
        f"mode={DEFAULT_WEIGHTS_RECOVERY_MODE}",
        f"freq={DEFAULT_WEIGHTS_PROGRAMMER_FREQ_KHZ}",
        f"ap={DEFAULT_WEIGHTS_RECOVERY_APID}",
    ]
    recovery_proc = subprocess.run(recovery_cmd, capture_output=True, text=True, check=False)
    if recovery_proc.returncode != 0:
        detail = []
        if recovery_proc.stdout.strip():
            detail.append(f"stdout:\n{recovery_proc.stdout.strip()}")
        if recovery_proc.stderr.strip():
            detail.append(f"stderr:\n{recovery_proc.stderr.strip()}")
        joined = "\n\n".join(detail) if detail else "No programmer output captured."
        raise WorkflowError(
            "Recovering STM32 target state before external weight programming failed.\n\n"
            f"Recovery command: {' '.join(recovery_cmd)}\n\n{joined}"
        )
    time.sleep(1.0)
    cmd = [
        str(programmer),
        "-q",
        "-c",
        f"port=SWD",
        f"mode={DEFAULT_WEIGHTS_PROGRAMMER_MODE}",
        f"freq={DEFAULT_WEIGHTS_PROGRAMMER_FREQ_KHZ}",
        f"ap={apid}",
        "-halt",
        "-el",
        str(external_loader),
        "-d",
        str(weights_blob_path),
        weights_flash_address,
        "-v",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        detail = []
        if proc.stdout.strip():
            detail.append(f"stdout:\n{proc.stdout.strip()}")
        if proc.stderr.strip():
            detail.append(f"stderr:\n{proc.stderr.strip()}")
        joined = "\n\n".join(detail) if detail else "No programmer output captured."
        raise WorkflowError(
            "Programming external weights blob failed.\n\n"
            f"Blob: {weights_blob_path}\n"
            f"Address: {weights_flash_address}\n"
            f"Loader: {external_loader}\n\n{joined}"
        )


def _load_and_run(
    *,
    gdbserver: Path,
    gdb: Path,
    cubeprog_bin: Path,
    gdb_port: int,
    apid: int,
    server_ready_timeout: float,
    verbose: bool,
    elf_path: Path,
) -> None:
    """Start the GDB server, load the ELF onto the target, and resume execution.

    Starts the ST-LINK GDB server, waits for it to become ready, loads the
    compiled ELF via arm-none-eabi-gdb, and lets the firmware run. The GDB
    server is terminated in the finally block regardless of outcome.

    Parameters
    ----------
    gdbserver : Path
        Path to the ``ST-LINK_gdbserver`` executable.
    gdb : Path
        Path to the ``arm-none-eabi-gdb`` executable.
    cubeprog_bin : Path
        STM32CubeProgrammer bin directory (used by the GDB server).
    gdb_port : int
        TCP port for the GDB server to listen on.
    apid : int
        ST-Link Access Port ID for the application core.
    server_ready_timeout : float
        Seconds to wait for the GDB server to report ready.
    verbose : bool
        If ``True``, pass verbose flags to the GDB server and GDB client.
    elf_path : Path
        Path to the ELF file to flash and run.

    Raises
    ------
    WorkflowError
        Raised if the GDB server or GDB load step fails.
    """
    server_proc = None
    server_log = None
    try:
        server_proc, server_log = _start_gdb_server(
            gdbserver=gdbserver,
            cubeprog_bin=cubeprog_bin,
            gdb_port=gdb_port,
            apid=apid,
            verbose=verbose,
        )
        try:
            _wait_for_server_ready(server_proc, server_log, server_ready_timeout)
        except WorkflowError as exc:
            raise WorkflowError(f"{exc}\n\n{_summarize_server_log(server_log)}") from exc
        time.sleep(SERVER_SETTLE_DELAY_S)
        if server_proc.poll() is not None:
            raise WorkflowError(
                "GDB server exited before GDB connected.\n\n" + _summarize_server_log(server_log)
            )
        _run_gdb_load(
            gdb=gdb,
            elf_path=elf_path,
            gdb_port=gdb_port,
            run_after_load=True,
            verbose=verbose,
        )
    finally:
        if server_proc is not None and server_proc.poll() is None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5.0)
            except Exception:
                server_proc.kill()
                server_proc.wait(timeout=5.0)
        if server_log is not None:
            server_log.close()


def _build_contract(
    *,
    size_info: dict[str, int],
    arena_bytes: int,
    latency_budget_ms: float,
    dut_metrics: dict[str, float],
    harness_metrics: dict[str, float],
    error_code: int,
    weight_storage_mode: str,
    weights_flash_bytes: int,
    weights_flash_address: str | None,
    weights_programmed: bool,
) -> dict[str, float | int | bool | str | None]:
    """Assemble the compact metrics contract dict from raw HIL results.

    Parameters
    ----------
    size_info : dict[str, int]
        Output of :func:`_run_size`: section sizes and derived flash/RAM totals.
    arena_bytes : int
        ST Edge AI activation arena size in bytes.
    latency_budget_ms : float
        Configured latency budget in milliseconds.
    dut_metrics : dict[str, float]
        Float metrics parsed from the DUT serial output.
    harness_metrics : dict[str, float]
        Float metrics parsed from the Arduino energy harness serial output.
    error_code : int
        One of the ``MASTER_*`` status constants indicating run outcome.
    weight_storage_mode : str
        ``"embedded"`` or ``"external_flash"``.
    weights_flash_bytes : int
        Size of the external weights blob in bytes (0 if embedded).
    weights_flash_address : str or None
        Hex address string for the external blob, or ``None`` if embedded.
    weights_programmed : bool
        Whether the external weights blob was successfully programmed this run.

    Returns
    -------
    dict[str, float | int | bool | str | None]
        Flat metrics dictionary conforming to the TinyODOM HIL contract schema.
    """
    harness_latency_ms = -1.0
    if harness_metrics.get("harness_latency_s", -1.0) >= 0.0:
        harness_latency_ms = harness_metrics["harness_latency_s"] * 1000.0
    elf_flash_bytes = int(size_info["elf_flash_bytes"])
    return {
        "ram_bytes": int(size_info["ram_bytes"]),
        "flash_bytes": int(elf_flash_bytes + weights_flash_bytes),
        "elf_flash_bytes": elf_flash_bytes,
        "weights_flash_bytes": int(weights_flash_bytes),
        "weight_storage_mode": weight_storage_mode,
        "weights_flash_address": weights_flash_address,
        "weights_programmed": bool(weights_programmed),
        "latency_ms": harness_latency_ms,
        "latency_budget_ms": float(latency_budget_ms),
        "arena_bytes": int(arena_bytes),
        "hil_enabled": True,
        "dwt_cycles_per_inference": float(dut_metrics.get("dwt_cycles_per_inference", -1.0)),
        "energy_mj_per_inference": float(harness_metrics.get("energy_mj_per_inference", -1.0)),
        "avg_power_mw": float(harness_metrics.get("avg_power_mw", -1.0)),
        "avg_current_ma": float(harness_metrics.get("avg_current_ma", -1.0)),
        "bus_voltage_v": float(harness_metrics.get("bus_voltage_v", -1.0)),
        "idle_power_mw": float(harness_metrics.get("idle_power_mw", -1.0)),
        "harness_latency_ms": harness_latency_ms,
        "error_code": int(error_code),
        "error_label": ERROR_LABELS.get(int(error_code), f"UNKNOWN_{error_code}"),
    }


def main() -> int:
    """Parse arguments, orchestrate the full STM32 toy AI HIL run, and emit metrics.

    The full workflow is:

    1. Regenerate and stage the perturbed TinyODOM model (unless ``--reuse-staged-model``).
    2. Build the STM32 FSBL project (unless ``--no-build``).
    3. Optionally program external model weights into NOR flash.
    4. Ensure the Arduino harness firmware is up-to-date.
    5. Open serial monitors on both the DUT and harness ports.
    6. Load the ELF onto the target via GDB and wait for ``DUT READY``.
    7. Trigger the harness, collect energy and latency telemetry, and verify run counts.
    8. Write the compact metrics JSON and a full diagnostics JSON side-car.

    Returns
    -------
    int
        0 on success (``MASTER_SUCCESS``), 1 on any failure.
    """
    parser = _build_parser()
    args = parser.parse_args()
    _configure_logging(args.verbose)

    project_root = args.project_root.resolve()
    config_path = args.config.resolve()
    output_path = args.output.resolve()
    diagnostic_path = _diagnostic_output_path(output_path)
    stage_output_root = args.stage_output_root.resolve()
    weights_memory_pool = args.weights_memory_pool.resolve()
    weights_external_loader = (
        args.weights_external_loader.resolve() if args.weights_external_loader is not None else None
    )
    gdbserver = _resolve_required_tool_path(args.gdbserver, "ST-LINK_gdbserver", "ST-LINK_gdbserver")
    gdb = _resolve_required_tool_path(args.gdb, "arm-none-eabi-gdb", "arm-none-eabi-gdb")
    cubeprog_bin = _resolve_required_tool_path(
        args.cubeprog_bin,
        "STM32CubeProgrammer bin directory",
        "STM32_Programmer_CLI",
    )
    _debug_dir, elf_path = _validate_paths(
        project_root=project_root,
        gdbserver=gdbserver,
        gdb=gdb,
        cubeprog_bin=cubeprog_bin,
    )

    if not config_path.is_file():
        raise WorkflowError(f"Config not found: {config_path}")
    if args.weight_storage_mode == "external_flash" and not weights_memory_pool.is_file():
        raise WorkflowError(f"Weights memory-pool JSON not found: {weights_memory_pool}")
    if weights_external_loader is not None and not weights_external_loader.is_file():
        raise WorkflowError(f"Weights external loader not found: {weights_external_loader}")
    if args.no_build and not args.reuse_staged_model:
        raise WorkflowError(
            "--no-build requires --reuse-staged-model because the default STM32 flow "
            "regenerates and stages a fresh perturbed TinyODOM model before building."
        )

    if not args.reuse_staged_model:
        LOGGER.info("Rebuilding perturbed TinyODOM model and staging STM32 network sources")
        _stage_stm32_network(
            config_path=config_path,
            project_root=project_root,
            output_root=stage_output_root,
            clean=args.clean,
            weight_storage_mode=args.weight_storage_mode,
            weights_flash_address=args.weights_flash_address,
            weights_memory_pool=weights_memory_pool,
            weights_external_loader=weights_external_loader,
        )

    staging_manifest = _read_staging_manifest(stage_output_root)
    weight_storage_mode = str(staging_manifest.get("weight_storage_mode", args.weight_storage_mode))
    weights_blob_path = staging_manifest.get("weights_blob_path")
    weights_flash_address = staging_manifest.get("weights_flash_address")
    weights_flash_bytes = int(staging_manifest.get("weights_blob_size") or 0)
    weights_programmed = False

    if not args.no_build:
        LOGGER.info("Building STM32 toy AI project in %s", project_root)
        _build_project(project_root, jobs=args.jobs, clean=args.clean, verbose=args.verbose)

    if not elf_path.is_file():
        raise WorkflowError(f"ELF not found after build: {elf_path}")

    linker_script = _find_linker_script(project_root)
    size_info = _run_size(elf_path)
    linker_info = _parse_linker_reservations(linker_script)
    arena_bytes = _parse_arena_bytes(project_root / "Inc" / "network_data_params.h")

    if weight_storage_mode == "external_flash":
        if not weights_blob_path or not weights_flash_address:
            raise WorkflowError(
                "external_flash staging manifest is missing weights_blob_path or weights_flash_address."
            )
        weights_blob = Path(str(weights_blob_path)).resolve()
        if not weights_blob.is_file():
            raise WorkflowError(f"Weights blob not found: {weights_blob}")
        external_loader = _resolve_weights_external_loader(cubeprog_bin, weights_external_loader)
        LOGGER.info("Programming external weights blob %s at %s", weights_blob, weights_flash_address)
        _program_weights_blob(
            cubeprog_bin=cubeprog_bin,
            apid=args.apid,
            weights_blob_path=weights_blob,
            weights_flash_address=str(weights_flash_address),
            external_loader=external_loader,
        )
        weights_programmed = True

    ensure_harness_firmware(
        harness_serial_port=args.harness_port,
        harness_fqbn=args.harness_fqbn,
        harness_auto_flash=args.harness_auto_flash,
        build_defines={
            "TINYODOM_HARNESS_ARM_PIN": int(args.harness_arm_pin),
            "TINYODOM_HARNESS_TRIGGER_PIN": int(args.harness_trigger_pin),
            "TINYODOM_DUT_ARM_HOLD_MS": int(args.dut_arm_hold_ms),
            "TINYODOM_HARNESS_STABLE_LOW_MS": int(args.harness_stable_low_ms),
        },
    )

    dut_monitor: SerialMonitor | None = None
    harness_monitor: SerialMonitor | None = None
    error_code = MASTER_PENDING
    dut_lines: list[str] = []
    harness_lines: list[str] = []
    dut_metrics: dict[str, float] = {}
    harness_metrics: dict[str, float] = {}

    try:
        LOGGER.info("Opening DUT serial monitor on %s", args.dut_port)
        dut_monitor = SerialMonitor(args.dut_port, args.baud, "dut")
        LOGGER.info("Opening harness serial monitor on %s", args.harness_port)
        harness_monitor = SerialMonitor(args.harness_port, args.baud, "harness")

        harness_monitor.write_line("PING")
        if harness_monitor.wait_for(lambda line: line.strip().lower() == "harness ready", args.harness_ready_timeout, "HARNESS READY") is None:
            raise RuntimeError("Timed out waiting for HARNESS READY.")

        _load_and_run(
            gdbserver=gdbserver,
            gdb=gdb,
            cubeprog_bin=cubeprog_bin,
            gdb_port=args.gdb_port,
            apid=args.apid,
            server_ready_timeout=args.server_ready_timeout,
            verbose=args.verbose,
            elf_path=elf_path,
        )

        if dut_monitor.wait_for(lambda line: line.strip() == "DUT READY", args.dut_ready_timeout, "DUT READY") is None:
            raise RuntimeError("Timed out waiting for DUT READY.")
        dut_monitor.write_line("START")

        if harness_monitor.wait_for(lambda line: line.strip() == "DONE", args.serial_timeout + args.harness_done_timeout, "DONE") is None:
            raise RuntimeError("Timed out waiting for harness DONE.")

        time.sleep(0.5)
        dut_lines = dut_monitor.snapshot_lines()
        harness_lines = harness_monitor.snapshot_lines()
        dut_metrics = _parse_last_float_fields(dut_lines, DUT_FLOAT_PATTERNS)
        harness_metrics = _parse_last_float_fields(harness_lines, HARNESS_FLOAT_PATTERNS)

        runs_dut = _parse_runs(dut_lines)
        runs_harness = _parse_runs(harness_lines)
        if runs_dut != 10 or runs_harness != 10:
            raise RuntimeError(f"Run-count mismatch: dut={runs_dut} harness={runs_harness}")
        if "harness_latency_s" not in harness_metrics:
            raise RuntimeError("Harness latency telemetry missing.")
        error_code = MASTER_SUCCESS
    except (WorkflowError, RuntimeError, serial.SerialException) as exc:
        LOGGER.error("%s", exc)
        if isinstance(exc, serial.SerialException):
            error_code = MASTER_DEVICE_NOT_FOUND
        elif isinstance(exc, WorkflowError):
            error_code = MASTER_DEVICE_NOT_FOUND
        else:
            error_code = MASTER_FATAL
        if dut_monitor is not None:
            dut_lines = dut_monitor.snapshot_lines()
        if harness_monitor is not None:
            harness_lines = harness_monitor.snapshot_lines()
        dut_metrics = _parse_last_float_fields(dut_lines, DUT_FLOAT_PATTERNS)
        harness_metrics = _parse_last_float_fields(harness_lines, HARNESS_FLOAT_PATTERNS)
    finally:
        if dut_monitor is not None:
            dut_monitor.close()
        if harness_monitor is not None:
            harness_monitor.close()

    contract = _build_contract(
        size_info=size_info,
        arena_bytes=arena_bytes,
        latency_budget_ms=args.latency_budget_ms,
        dut_metrics=dut_metrics,
        harness_metrics=harness_metrics,
        error_code=error_code,
        weight_storage_mode=weight_storage_mode,
        weights_flash_bytes=weights_flash_bytes,
        weights_flash_address=str(weights_flash_address) if weights_flash_address else None,
        weights_programmed=weights_programmed,
    )

    dut_latency_ms_from_cycles = -1.0
    if dut_metrics.get("clock_hz", -1.0) > 0.0 and dut_metrics.get("dwt_cycles_per_inference", -1.0) >= 0.0:
        dut_latency_ms_from_cycles = (
            dut_metrics["dwt_cycles_per_inference"] / dut_metrics["clock_hz"]
        ) * 1000.0
    latency_delta_ms = -1.0
    if contract["harness_latency_ms"] >= 0 and dut_latency_ms_from_cycles >= 0:
        latency_delta_ms = abs(float(contract["harness_latency_ms"]) - dut_latency_ms_from_cycles)

    diagnostic = {
        "raw_size": size_info,
        "linker": linker_info,
        "staging_manifest": staging_manifest,
        "arena_bytes": arena_bytes,
        "latency_budget_ms": args.latency_budget_ms,
        "dut_metrics": dut_metrics,
        "harness_metrics": harness_metrics,
        "dut_latency_ms_from_cycles": dut_latency_ms_from_cycles,
        "latency_cross_check_delta_ms": latency_delta_ms,
        "dut_runs": _parse_runs(dut_lines),
        "harness_runs": _parse_runs(harness_lines),
        "dut_lines": dut_lines,
        "harness_lines": harness_lines,
        "elf_path": str(elf_path),
        "linker_script": str(linker_script),
        "config_path": str(config_path),
        "stage_output_root": str(stage_output_root),
        "reuse_staged_model": bool(args.reuse_staged_model),
        "weights_programmed": bool(weights_programmed),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    diagnostic_path.write_text(json.dumps(diagnostic, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(contract, indent=2))
    print(f"\nDiagnostics: {diagnostic_path}")
    return 0 if error_code == MASTER_SUCCESS else 1


if __name__ == "__main__":
    raise SystemExit(main())
