#!/usr/bin/env python3
"""
Build and debug-load the STM32 N6 blink example.

This wrapper automates the currently understood safe STM32N6 development flow:

1. Optionally clean the generated make build
2. Build the existing FSBL project with ``make``
3. Start ``ST-LINK_gdbserver``
4. Connect with ``arm-none-eabi-gdb``
5. Load the ELF into target memory
6. Optionally continue execution, then detach and exit

This script is intentionally scoped to development boot mode debug loading. It
does not attempt to sign binaries, program external flash, or change device
security state.
"""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = SCRIPT_DIR / "stm32_blink_example_project" / "FSBL"
DEFAULT_GDB_PORT = 61234
DEFAULT_APID = int(os.environ.get("STM32_APID", "1"))
SERVER_READY_TIMEOUT_S = 15.0
SERVER_POLL_INTERVAL_S = 0.1
SERVER_SETTLE_DELAY_S = 0.5
DEFAULT_LOG_LEVEL = "1"
DEFAULT_REFRESH_DELAY_S = "15"
LOGGER = logging.getLogger(__name__)


class WorkflowError(RuntimeError):
    """Raised when the build/upload workflow cannot proceed."""


def _which_path(name: str) -> Path | None:
    """Return a Path for an executable found on PATH."""

    value = shutil.which(name)
    if value is None:
        return None
    return Path(value)


def _default_cubeprog_bin() -> Path | None:
    """Infer the STM32CubeProgrammer bin directory from STM32_Programmer_CLI."""

    cli = _which_path("STM32_Programmer_CLI")
    if cli is None:
        return None
    return cli.resolve().parent


def _configure_logging(verbose: bool) -> None:
    """Configure process logging.

    Parameters
    ----------
    verbose : bool
        When ``True``, enable debug-level logging. Otherwise use info-level
        logging.
    """

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s:%(name)s:%(message)s",
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser for the STM32 build/upload workflow.
    """

    parser = argparse.ArgumentParser(
        description="Build and debug-load the STM32 N6 blink FSBL project."
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Run `make -C Debug clean` before building.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="Parallel jobs passed to make (default: cpu count).",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="Path to the FSBL project root.",
    )
    parser.add_argument(
        "--gdbserver",
        type=Path,
        default=_which_path("ST-LINK_gdbserver"),
        help="Path to ST-LINK_gdbserver (defaults to PATH lookup).",
    )
    parser.add_argument(
        "--gdb",
        type=Path,
        default=_which_path("arm-none-eabi-gdb"),
        help="Path to arm-none-eabi-gdb (defaults to PATH lookup).",
    )
    parser.add_argument(
        "--cubeprog-bin",
        type=Path,
        default=_default_cubeprog_bin(),
        help="Path to the STM32CubeProgrammer bin directory (defaults to PATH lookup via STM32_Programmer_CLI).",
    )
    parser.add_argument(
        "--gdb-port",
        type=int,
        default=DEFAULT_GDB_PORT,
        help="TCP port used by the GDB server.",
    )
    parser.add_argument(
        "--apid",
        type=int,
        default=DEFAULT_APID,
        help="Access port / core ID passed to ST-LINK_gdbserver (default: 1).",
    )
    parser.add_argument(
        "--server-ready-timeout",
        type=float,
        default=SERVER_READY_TIMEOUT_S,
        help="Seconds to wait for ST-LINK_gdbserver to open its TCP port.",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Skip the make build step.",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Load the ELF but do not continue execution after download.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print subprocess commands before executing them.",
    )
    return parser


def _print_command(argv: list[str], cwd: Path | None = None) -> None:
    """Log a shell-style rendering of a subprocess command.

    Parameters
    ----------
    argv : list[str]
        Command and arguments to render.
    cwd : Path | None, optional
        Working directory for the command, by default ``None``.
    """

    rendered = " ".join(shlex.quote(part) for part in argv)
    if cwd is not None:
        LOGGER.debug("[cmd] (cd %s && %s)", cwd, rendered)
    else:
        LOGGER.debug("[cmd] %s", rendered)


def _run_command(
    argv: list[str],
    *,
    cwd: Path | None = None,
    verbose: bool = False,
    stream_output: bool = False,
) -> None:
    """Run a subprocess and raise on failure.

    Parameters
    ----------
    argv : list[str]
        Command and arguments to execute.
    cwd : Path | None, optional
        Working directory for the command, by default ``None``.
    verbose : bool, optional
        When ``True``, log the rendered command before execution, by default
        ``False``.
    stream_output : bool, optional
        When ``True``, inherit stdout and stderr so command output is shown
        directly in the terminal, by default ``False``.

    Raises
    ------
    WorkflowError
        Raised when the subprocess exits with a nonzero return code.
    """

    if verbose:
        _print_command(argv, cwd=cwd)
    if stream_output:
        proc = subprocess.run(argv, cwd=cwd, check=False)
    else:
        proc = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        if stream_output:
            raise WorkflowError(
                f"Command failed with exit code {proc.returncode}: {' '.join(argv)}"
            )
        detail = []
        if proc.stdout.strip():
            detail.append(f"stdout:\n{proc.stdout.strip()}")
        if proc.stderr.strip():
            detail.append(f"stderr:\n{proc.stderr.strip()}")
        joined = "\n\n".join(detail) if detail else "No subprocess output captured."
        raise WorkflowError(
            f"Command failed with exit code {proc.returncode}: {' '.join(argv)}\n\n{joined}"
        )


def _wait_for_server_ready(
    proc: subprocess.Popen[str],
    log_file: tempfile._TemporaryFileWrapper[str],
    timeout_s: float,
) -> None:
    """Wait for ST-LINK_gdbserver to reach its ready state.

    Parameters
    ----------
    proc : subprocess.Popen[str]
        Running GDB server process.
    log_file : tempfile._TemporaryFileWrapper[str]
        Temporary file capturing the GDB server output.
    timeout_s : float
        Maximum number of seconds to wait.

    Raises
    ------
    WorkflowError
        Raised if the server never reaches the debugger-waiting state or exits
        early.
    """

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        log_file.flush()
        log_file.seek(0)
        content = log_file.read()
        if "Waiting for debugger connection..." in content:
            return
        if "Error in initializing ST-LINK device." in content:
            raise WorkflowError("GDB server failed to initialize the ST-LINK device.")
        if proc.poll() is not None:
            raise WorkflowError("GDB server exited before reaching debugger-waiting state.")
        time.sleep(SERVER_POLL_INTERVAL_S)
    raise WorkflowError(f"GDB server did not become ready within {timeout_s:.1f}s.")


def _validate_paths(
    *,
    project_root: Path,
    gdbserver: Path,
    gdb: Path,
    cubeprog_bin: Path,
) -> tuple[Path, Path]:
    """Validate project and tool paths.

    Parameters
    ----------
    project_root : Path
        FSBL project root.
    gdbserver : Path
        Path to ``ST-LINK_gdbserver``.
    gdb : Path
        Path to ``arm-none-eabi-gdb``.
    cubeprog_bin : Path
        Path to the STM32CubeProgrammer bin directory.

    Returns
    -------
    tuple[Path, Path]
        The ``Debug`` directory path and the expected ELF path.

    Raises
    ------
    WorkflowError
        Raised when a required path is missing or not executable.
    """

    if not project_root.exists():
        raise WorkflowError(f"Project root does not exist: {project_root}")

    debug_dir = project_root / "Debug"
    if not debug_dir.is_dir():
        raise WorkflowError(f"Debug directory does not exist: {debug_dir}")

    makefile = debug_dir / "makefile"
    if not makefile.is_file():
        raise WorkflowError(f"Generated makefile does not exist: {makefile}")

    if not gdbserver.is_file():
        raise WorkflowError(f"ST-LINK_gdbserver was not found: {gdbserver}")
    if not os.access(gdbserver, os.X_OK):
        raise WorkflowError(f"ST-LINK_gdbserver is not executable: {gdbserver}")
    if not gdb.is_file():
        raise WorkflowError(f"arm-none-eabi-gdb was not found: {gdb}")
    if not os.access(gdb, os.X_OK):
        raise WorkflowError(f"arm-none-eabi-gdb is not executable: {gdb}")
    if not cubeprog_bin.is_dir():
        raise WorkflowError(f"STM32CubeProgrammer bin directory was not found: {cubeprog_bin}")

    return debug_dir, debug_dir / "stm32_blink_example_project_FSBL.elf"


def _resolve_required_tool_path(path: Path | None, label: str, hint: str) -> Path:
    """Resolve a required STM32 tool path and provide a setup hint on failure."""

    if path is None:
        raise WorkflowError(
            f"{label} was not provided.\n"
            f"Install STM32CubeCLT and ensure `{hint}` is on PATH, or pass the matching CLI flag."
        )
    return path.resolve()


def _build_project(project_root: Path, jobs: int, clean: bool, verbose: bool) -> None:
    """Run the generated make build for the FSBL project.

    Parameters
    ----------
    project_root : Path
        FSBL project root.
    jobs : int
        Parallel make job count.
    clean : bool
        When ``True``, run ``make clean`` first.
    verbose : bool
        When ``True``, log subprocess commands before execution.
    """

    if clean:
        _run_command(
            ["make", "-C", "Debug", "clean"],
            cwd=project_root,
            verbose=verbose,
            stream_output=True,
        )
    _run_command(
        ["make", "-C", "Debug", f"-j{max(jobs, 1)}", "all"],
        cwd=project_root,
        verbose=verbose,
        stream_output=True,
    )


def _gdb_commands(gdb_port: int, run_after_load: bool) -> list[str]:
    """Build the batch GDB command list.

    Parameters
    ----------
    gdb_port : int
        TCP port exposed by the GDB server.
    run_after_load : bool
        When ``True``, resume execution after ``load``.

    Returns
    -------
    list[str]
        Ordered GDB commands for batch execution.
    """

    commands = [
        "set pagination off",
        "set confirm off",
        f"target remote localhost:{gdb_port}",
        "load",
    ]
    if run_after_load:
        commands.extend(
            [
                "continue&",
                "detach",
            ]
        )
    commands.append("quit")
    return commands


def _run_gdb_load(
    *,
    gdb: Path,
    elf_path: Path,
    gdb_port: int,
    run_after_load: bool,
    verbose: bool,
) -> None:
    """Run batch-mode GDB to load the ELF.

    Parameters
    ----------
    gdb : Path
        Path to ``arm-none-eabi-gdb``.
    elf_path : Path
        Path to the built ELF image.
    gdb_port : int
        TCP port exposed by the GDB server.
    run_after_load : bool
        When ``True``, resume execution after the download.
    verbose : bool
        When ``True``, log subprocess commands before execution.
    """

    argv = [str(gdb), "--quiet", "--batch"]
    for command in _gdb_commands(gdb_port, run_after_load):
        argv.extend(["-ex", command])
    argv.append(str(elf_path))
    _run_command(argv, verbose=verbose)


def _start_gdb_server(
    *,
    gdbserver: Path,
    cubeprog_bin: Path,
    gdb_port: int,
    apid: int,
    verbose: bool,
) -> tuple[subprocess.Popen[str], tempfile._TemporaryFileWrapper[str]]:
    """Start the ST-LINK GDB server process.

    Parameters
    ----------
    gdbserver : Path
        Path to ``ST-LINK_gdbserver``.
    cubeprog_bin : Path
        Path to the STM32CubeProgrammer bin directory.
    apid : int
        Access port / core ID passed to ``ST-LINK_gdbserver``.
    verbose : bool
        When ``True``, log the subprocess command before execution.

    Returns
    -------
    tuple[subprocess.Popen[str], tempfile._TemporaryFileWrapper[str]]
        The running server process and its temporary log file handle.
    """

    log_file = tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8", prefix="stm32_gdbserver_", suffix=".log")
    argv = [
        str(gdbserver),
        "-p",
        str(gdb_port),
        "-d",
        "-m",
        str(apid),
        "-k",
        "-l",
        DEFAULT_LOG_LEVEL,
        "-r",
        DEFAULT_REFRESH_DELAY_S,
        "-cp",
        str(cubeprog_bin),
    ]
    if verbose:
        _print_command(argv)
    proc = subprocess.Popen(
        argv,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc, log_file


def _summarize_server_log(log_file: tempfile._TemporaryFileWrapper[str]) -> str:
    """Summarize captured GDB server output.

    Parameters
    ----------
    log_file : tempfile._TemporaryFileWrapper[str]
        Temporary file used to capture GDB server output.

    Returns
    -------
    str
        Human-readable summary of the captured server log.
    """

    log_file.flush()
    log_file.seek(0)
    content = log_file.read().strip()
    if not content:
        return "No GDB server output captured."
    lines = content.splitlines()
    tail = "\n".join(lines[-40:])
    return f"GDB server output (tail):\n{tail}"


def main() -> int:
    """Run the STM32 build-and-upload workflow.

    Returns
    -------
    int
        Process exit status. Zero indicates success.

    Raises
    ------
    WorkflowError
        Raised when any validation, build, server, or load step fails.
    """

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
        LOGGER.info("Building STM32 project in %s", project_root)
        _build_project(project_root, jobs=args.jobs, clean=args.clean, verbose=args.verbose)

    if not elf_path.is_file():
        raise WorkflowError(
            f"Expected ELF does not exist after build step: {elf_path}\n"
            "If you passed --no-build, make sure the project has already been built."
        )

    LOGGER.info("Starting ST-LINK GDB server")
    server_proc: subprocess.Popen[str] | None = None
    server_log: tempfile._TemporaryFileWrapper[str] | None = None
    try:
        server_proc, server_log = _start_gdb_server(
            gdbserver=gdbserver,
            cubeprog_bin=cubeprog_bin,
            gdb_port=args.gdb_port,
            apid=args.apid,
            verbose=args.verbose,
        )
        try:
            _wait_for_server_ready(server_proc, server_log, args.server_ready_timeout)
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
            gdb_port=args.gdb_port,
            run_after_load=not args.no_run,
            verbose=args.verbose,
        )
        if args.no_run:
            LOGGER.info("Load complete. Target was not resumed.")
        else:
            LOGGER.info("Load complete and target resumed.")
        return 0
    finally:
        if server_proc is not None and server_proc.poll() is None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                server_proc.kill()
                server_proc.wait(timeout=5.0)
        if server_log is not None:
            server_log.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except WorkflowError as exc:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
        LOGGER.error("%s", exc)
        raise SystemExit(1)
