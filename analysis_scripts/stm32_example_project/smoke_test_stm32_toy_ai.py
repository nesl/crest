#!/usr/bin/env python3
"""
Phase 0 smoke test for the STM32N6 toy AI project.

Builds (optionally), loads, and runs the toy AI ELF, then captures the
virtual COM port and confirms the expected Phase 0 telemetry tokens:

    STM32_AI_INIT=OK
    STM32_AI_RUN=OK
    STM32_AI_LATENCY_CYCLES=<value>

The serial reader thread is started before the load so no early output is
missed.  All received lines are printed to the terminal in real time.

Exit status:
    0  all expected tokens received within the timeout
    1  one or more tokens missing, or a build/load failure
"""

from __future__ import annotations

import argparse
import logging
import os
import queue
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

import serial

# Pull in the build/load helpers from the existing blink wrapper rather than
# duplicating them.  The wrapper lives in the same directory, so we add that
# directory to sys.path before importing.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_and_upload_stm32_blink import (
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
    DEFAULT_GDB_PORT,
    DEFAULT_APID,
    SERVER_READY_TIMEOUT_S,
    SERVER_SETTLE_DELAY_S,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = SCRIPT_DIR / "stm32_toy_ai_project" / "FSBL"
# ttyACM0 is the ST-LINK virtual COM port on a typical Linux host.
DEFAULT_SERIAL_PORT = "/dev/ttyACM0"
DEFAULT_BAUD = 115200
# How long to wait for all three tokens before declaring failure.
DEFAULT_SERIAL_TIMEOUT_S = 30.0

# The three lines the toy AI shell must emit over UART for Phase 0 to pass.
# LATENCY_CYCLES is matched as a prefix because the cycle count varies.
PHASE0_TOKENS = [
    "STM32_AI_INIT=OK",
    "STM32_AI_RUN=OK",
    "STM32_AI_LATENCY_CYCLES=",
]

LOGGER = logging.getLogger(__name__)


def _which_path(name: str) -> Path | None:
    value = shutil.which(name)
    return Path(value) if value is not None else None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 0 smoke test: build, load, and verify UART telemetry."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="Path to the toy FSBL project root.",
    )
    parser.add_argument(
        "--port",
        default=DEFAULT_SERIAL_PORT,
        help=f"Serial port for UART capture (default: {DEFAULT_SERIAL_PORT}).",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=DEFAULT_BAUD,
        help=f"Baud rate (default: {DEFAULT_BAUD}).",
    )
    parser.add_argument(
        "--serial-timeout",
        type=float,
        default=DEFAULT_SERIAL_TIMEOUT_S,
        help=f"Seconds to wait for all expected tokens (default: {DEFAULT_SERIAL_TIMEOUT_S}).",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Skip the make build step.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Run make clean before building.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="Parallel make jobs (default: cpu count).",
    )
    parser.add_argument(
        "--gdbserver",
        type=Path,
        default=_which_path("ST-LINK_gdbserver"),
    )
    parser.add_argument(
        "--gdb",
        type=Path,
        default=_which_path("arm-none-eabi-gdb"),
    )
    parser.add_argument(
        "--cubeprog-bin",
        type=Path,
        default=_default_cubeprog_bin(),
    )
    parser.add_argument(
        "--gdb-port",
        type=int,
        default=DEFAULT_GDB_PORT,
    )
    parser.add_argument(
        "--apid",
        type=int,
        default=DEFAULT_APID,
    )
    parser.add_argument(
        "--server-ready-timeout",
        type=float,
        default=SERVER_READY_TIMEOUT_S,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
    )
    return parser


def _serial_reader(
    ser: serial.Serial,
    line_queue: "queue.Queue[str | None]",
    stop_event: threading.Event,
) -> None:
    """Read lines from the serial port and push them onto line_queue.

    Runs on a background thread so the main thread can drive the load
    and token-matching loop concurrently.  The read timeout of 0.2 s lets
    the thread check stop_event regularly without busy-waiting.  Pushes
    None as a sentinel when the thread exits so the main thread can detect
    an unexpected reader death.
    """
    try:
        while not stop_event.is_set():
            raw = ser.readline()
            if raw:
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                line_queue.put(line)
    except serial.SerialException as exc:
        LOGGER.error("Serial error: %s", exc)
    finally:
        # Sentinel: tells the main thread the reader is gone.
        line_queue.put(None)


def _load_and_run(
    *,
    project_root: Path,
    gdbserver: Path,
    gdb: Path,
    cubeprog_bin: Path,
    gdb_port: int,
    apid: int,
    server_ready_timeout: float,
    verbose: bool,
    elf_path: Path,
) -> None:
    server_proc = None
    server_log = None
    try:
        LOGGER.info("Starting ST-LINK GDB server")
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
                "GDB server exited before GDB connected.\n\n"
                + _summarize_server_log(server_log)
            )
        LOGGER.info("Loading %s to target memory", elf_path.name)
        _run_gdb_load(
            gdb=gdb,
            elf_path=elf_path,
            gdb_port=gdb_port,
            run_after_load=True,
            verbose=verbose,
        )
        LOGGER.info("Load complete, target resumed.")
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


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    _configure_logging(args.verbose)

    project_root = args.project_root.resolve()
    gdbserver = _resolve_required_tool_path(args.gdbserver, "ST-LINK_gdbserver", "ST-LINK_gdbserver")
    gdb = _resolve_required_tool_path(args.gdb, "arm-none-eabi-gdb", "arm-none-eabi-gdb")
    cubeprog_bin = _resolve_required_tool_path(
        args.cubeprog_bin,
        "STM32CubeProgrammer bin directory",
        "STM32_Programmer_CLI",
    )
    _, elf_path = _validate_paths(
        project_root=project_root,
        gdbserver=gdbserver,
        gdb=gdb,
        cubeprog_bin=cubeprog_bin,
    )

    if not args.no_build:
        LOGGER.info("Building toy AI project in %s", project_root)
        _build_project(project_root, jobs=args.jobs, clean=args.clean, verbose=args.verbose)

    if not elf_path.is_file():
        LOGGER.error("ELF not found after build: %s", elf_path)
        return 1

    # Start the serial reader thread *before* triggering the load.  The
    # firmware starts printing immediately after the jump-to-entry-point, so
    # any reader that opens the port after the load risks missing the first
    # line or two.
    LOGGER.info("Opening serial port %s at %d baud", args.port, args.baud)
    try:
        with serial.Serial(args.port, args.baud, timeout=0.2) as ser:
            line_queue = queue.Queue()
            stop_event = threading.Event()
            reader_thread = threading.Thread(
                target=_serial_reader,
                args=(ser, line_queue, stop_event),
                daemon=True,
            )
            reader_thread.start()

            try:
                _load_and_run(
                    project_root=project_root,
                    gdbserver=gdbserver,
                    gdb=gdb,
                    cubeprog_bin=cubeprog_bin,
                    gdb_port=args.gdb_port,
                    apid=args.apid,
                    server_ready_timeout=args.server_ready_timeout,
                    verbose=args.verbose,
                    elf_path=elf_path,
                )
            except WorkflowError as exc:
                LOGGER.error("%s", exc)
                stop_event.set()
                reader_thread.join(timeout=2.0)
                return 1

            remaining_tokens = list(PHASE0_TOKENS)
            deadline = time.monotonic() + args.serial_timeout
            latency_line: str | None = None
            start_sent = False

            print(f"\n--- serial capture ({args.port} @ {args.baud}) ---")
            while remaining_tokens and time.monotonic() < deadline:
                timeout_left = deadline - time.monotonic()
                try:
                    line = line_queue.get(timeout=min(timeout_left, 0.5))
                except queue.Empty:
                    continue
                if line is None:
                    LOGGER.error("Serial reader thread exited unexpectedly.")
                    break
                print(line)
                if line == "DUT READY" and not start_sent:
                    ser.write(b"START\n")
                    ser.flush()
                    start_sent = True
                    LOGGER.info("Sent START to DUT.")
                for token in list(remaining_tokens):
                    if line.startswith(token):
                        remaining_tokens.remove(token)
                        if token == "STM32_AI_LATENCY_CYCLES=":
                            latency_line = line
            print("--- end serial capture ---\n")

            stop_event.set()
            reader_thread.join(timeout=2.0)
    except serial.SerialException as exc:
        LOGGER.error("Failed to open serial port %s: %s", args.port, exc)
        return 1

    if remaining_tokens:
        LOGGER.error("Phase 0 FAIL — missing tokens: %s", remaining_tokens)
        return 1

    LOGGER.info("Phase 0 PASS")
    if latency_line:
        LOGGER.info("%s", latency_line)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except WorkflowError as exc:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
        LOGGER.error("%s", exc)
        raise SystemExit(1)
