#!/usr/bin/env python3
"""
Smoke-test the TinyODOM STM32 Phase 1 backend against a fixed FSBL project.

Default behavior:

1. Validate the selected STM32 FSBL project root
2. Compile the project through the TinyODOM STM backend
3. Debug-load the produced ELF through ST-LINK
4. Optionally verify UART output tokens when requested

This script is intentionally backend-centric. It does not route through the
NAS or HIL server flows and does not attempt any ST Edge AI staging.
"""

from __future__ import annotations

import argparse
import logging
import os
import queue
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import serial

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tinyodom.devices import CompileResult  # noqa: E402
from tinyodom.microcontrollers import get_device  # noqa: E402
from tinyodom.microcontrollers import stm32_cube_clt  # noqa: E402

BOARD_NAME = "STM32_NUCLEO_N657X0_Q"
DEFAULT_PROJECT_ROOT = SCRIPT_DIR / "stm32_blink_example_project" / "FSBL"
DEFAULT_SERIAL_PORT = "/dev/ttyACM0"
DEFAULT_BAUD = 115200
DEFAULT_SERIAL_TIMEOUT_S = 15.0

EXIT_SUCCESS = 0
EXIT_COMPILE_FAILURE = 1
EXIT_UPLOAD_FAILURE = 2
EXIT_SERIAL_FAILURE = 3
EXIT_CONFIG_FAILURE = 4

LOGGER = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    """Configure process logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s:%(name)s:%(message)s",
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the backend smoke script."""
    parser = argparse.ArgumentParser(
        description="Smoke-test the TinyODOM STM32 Phase 1 backend."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help=f"STM32 FSBL project root. Default: {DEFAULT_PROJECT_ROOT}",
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Compile only; skip the upload/debug-load step.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Run a clean build before compiling.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="Parallel jobs passed to the STM32 make build.",
    )
    parser.add_argument(
        "--gdbserver",
        type=Path,
        default=None,
        help="Path to ST-LINK_gdbserver.",
    )
    parser.add_argument(
        "--gdb",
        type=Path,
        default=None,
        help="Path to arm-none-eabi-gdb.",
    )
    parser.add_argument(
        "--cubeprog-bin",
        type=Path,
        default=None,
        help="Path to the STM32CubeProgrammer bin directory.",
    )
    parser.add_argument(
        "--gdb-port",
        type=int,
        default=stm32_cube_clt.DEFAULT_GDB_PORT,
        help="TCP port used by the ST-LINK GDB server.",
    )
    parser.add_argument(
        "--apid",
        type=int,
        default=stm32_cube_clt.DEFAULT_APID,
        help="Access port / core ID passed to ST-LINK_gdbserver.",
    )
    parser.add_argument(
        "--server-ready-timeout",
        type=float,
        default=stm32_cube_clt.SERVER_READY_TIMEOUT_S,
        help="Seconds to wait for ST-LINK_gdbserver readiness.",
    )
    parser.add_argument(
        "--serial-port",
        default=DEFAULT_SERIAL_PORT,
        help=f"Optional UART device path. Default: {DEFAULT_SERIAL_PORT}",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=DEFAULT_BAUD,
        help=f"UART baud rate. Default: {DEFAULT_BAUD}",
    )
    parser.add_argument(
        "--serial-timeout",
        type=float,
        default=DEFAULT_SERIAL_TIMEOUT_S,
        help=f"Seconds to wait for expected UART tokens. Default: {DEFAULT_SERIAL_TIMEOUT_S}",
    )
    parser.add_argument(
        "--expect-token",
        action="append",
        default=[],
        help="Repeatable UART token substring to require after upload.",
    )
    parser.add_argument(
        "--no-serial-check",
        action="store_true",
        help="Disable UART token checking even when tokens are provided.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def _build_device_options(args: argparse.Namespace) -> dict[str, object]:
    """Build STM backend options from CLI args."""
    return {
        "project_root": args.project_root.resolve(),
        "gdbserver": args.gdbserver.resolve() if args.gdbserver is not None else None,
        "gdb": args.gdb.resolve() if args.gdb is not None else None,
        "cubeprog_bin": args.cubeprog_bin.resolve() if args.cubeprog_bin is not None else None,
        "gdb_port": int(args.gdb_port),
        "apid": int(args.apid),
        "server_ready_timeout_s": float(args.server_ready_timeout),
    }


def _clean_compile(project_root: Path, jobs: int) -> CompileResult:
    """Run a clean STM build through the helper module and normalize the result."""
    try:
        build_result = stm32_cube_clt.build_project(
            project_root=project_root,
            jobs=jobs,
            clean=True,
        )
        size_result = stm32_cube_clt.parse_size_output(build_result.elf_path)
        return CompileResult(
            success=True,
            log="\n".join(
                text for text in (build_result.log.strip(), size_result.raw_output.strip()) if text
            ),
            flash_bytes=size_result.elf_flash_bytes,
            ram_bytes=size_result.ram_bytes,
            overflow_kind=None,
            build_dir=build_result.debug_dir,
        )
    except stm32_cube_clt.WorkflowError as exc:
        log_text = str(exc)
        return CompileResult(
            success=False,
            log=log_text,
            flash_bytes=None,
            ram_bytes=None,
            overflow_kind=stm32_cube_clt.classify_build_failure(log_text),
            build_dir=project_root / "Debug",
        )


class _SerialTokenMonitor:
    """Background serial reader that streams lines and matches expected tokens."""

    def __init__(self, port: str, baud: int) -> None:
        self.port = port
        self.baud = baud
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._stop = threading.Event()
        self._serial = serial.Serial(port, baud, timeout=0.2)
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        """Read lines from the serial port until stopped."""
        try:
            while not self._stop.is_set():
                raw = self._serial.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line:
                    continue
                print(f"[serial] {line}")
                self._queue.put(line)
        except serial.SerialException as exc:
            LOGGER.error("Serial error: %s", exc)
        finally:
            self._queue.put(None)

    def wait_for_tokens(self, tokens: Sequence[str], timeout_s: float) -> tuple[bool, list[str]]:
        """Wait for all expected token substrings to appear before timeout."""
        remaining = list(tokens)
        deadline = time.monotonic() + timeout_s
        while remaining and time.monotonic() < deadline:
            timeout_left = max(0.0, deadline - time.monotonic())
            try:
                line = self._queue.get(timeout=min(timeout_left, 0.25))
            except queue.Empty:
                continue
            if line is None:
                break
            remaining = [token for token in remaining if token not in line]
        return (len(remaining) == 0, remaining)

    def close(self) -> None:
        """Stop the background reader and close the serial port."""
        self._stop.set()
        try:
            self._thread.join(timeout=1.0)
        finally:
            self._serial.close()


def _print_summary(summary: SimpleNamespace) -> None:
    """Print a short phase-by-phase result summary."""
    print("\nSTM32 Phase 1 backend smoke summary")
    print(f"project_root: {summary.project_root}")
    print(
        f"compile: {summary.compile_status}"
        f" ram_bytes={summary.ram_bytes} flash_bytes={summary.flash_bytes}"
    )
    print(f"upload: {summary.upload_status}")
    print(f"serial_check: {summary.serial_status}")


def main() -> int:
    """Run the STM32 backend smoke test and return a process exit code."""
    parser = _build_parser()
    args = parser.parse_args()
    _configure_logging(args.verbose)

    device_options = _build_device_options(args)
    project_root = Path(device_options["project_root"])
    run_serial_check = bool(args.expect_token) and not bool(args.no_serial_check)

    summary = SimpleNamespace(
        project_root=project_root,
        compile_status="NOT_RUN",
        upload_status="SKIPPED" if args.compile_only else "NOT_RUN",
        serial_status="DISABLED" if not run_serial_check else "NOT_RUN",
        ram_bytes=-1,
        flash_bytes=-1,
    )

    serial_monitor: _SerialTokenMonitor | None = None

    try:
        validated_project_root = stm32_cube_clt.validate_project_root(project_root)
        device = get_device(
            BOARD_NAME,
            device_options=device_options,
            serial_port=args.serial_port,
        )

        if args.clean:
            compile_result = _clean_compile(validated_project_root, jobs=max(args.jobs, 1))
        else:
            compile_result = device.compile(
                sketch_path=validated_project_root,
                arena_kb=-1,
                window_size=0,
                num_channels=0,
            )

        summary.compile_status = "OK" if compile_result.success else "FAIL"
        summary.ram_bytes = compile_result.ram_bytes if compile_result.ram_bytes is not None else -1
        summary.flash_bytes = (
            compile_result.flash_bytes if compile_result.flash_bytes is not None else -1
        )

        if not compile_result.success:
            LOGGER.error("Compile failed.\n%s", compile_result.log)
            _print_summary(summary)
            return EXIT_COMPILE_FAILURE

        if args.compile_only:
            _print_summary(summary)
            return EXIT_SUCCESS

        if run_serial_check:
            serial_monitor = _SerialTokenMonitor(args.serial_port, args.baud)

        upload_result = device.upload(
            sketch_path=validated_project_root,
            build_dir=compile_result.build_dir,
            serial_port=args.serial_port,
        )
        summary.upload_status = "OK" if upload_result.success else "FAIL"
        if not upload_result.success:
            LOGGER.error("Upload failed.\n%s", upload_result.log)
            _print_summary(summary)
            return EXIT_UPLOAD_FAILURE

        if run_serial_check:
            matched, missing = serial_monitor.wait_for_tokens(
                args.expect_token,
                timeout_s=args.serial_timeout,
            )
            summary.serial_status = "OK" if matched else "FAIL"
            if not matched:
                LOGGER.error("Serial token check failed. Missing tokens: %s", ", ".join(missing))
                _print_summary(summary)
                return EXIT_SERIAL_FAILURE

        _print_summary(summary)
        return EXIT_SUCCESS
    except (ValueError, serial.SerialException, stm32_cube_clt.WorkflowError) as exc:
        LOGGER.error("Configuration/setup failure: %s", exc)
        _print_summary(summary)
        return EXIT_CONFIG_FAILURE
    finally:
        if serial_monitor is not None:
            serial_monitor.close()


if __name__ == "__main__":
    raise SystemExit(main())
