from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# Default TCP port used for the ST-LINK GDB server session.
DEFAULT_GDB_PORT = 61234
# Access port identifier for the main application core; overridable for lab setups.
DEFAULT_APID = int(os.environ.get("STM32_APID", "1"))
# Upper bound for waiting until the GDB server starts accepting connections.
SERVER_READY_TIMEOUT_S = 15.0
# Poll cadence while probing the GDB server readiness socket.
SERVER_POLL_INTERVAL_S = 0.1
# Small post-start delay so the server can finish initialization before GDB attaches.
SERVER_SETTLE_DELAY_S = 0.5
# ST-LINK_gdbserver verbosity level passed on the command line.
DEFAULT_LOG_LEVEL = "1"
# Delay string forwarded to the server's target-refresh option.
DEFAULT_REFRESH_DELAY_S = "15"
# Timeout for the initial GDB jump/continue sequence after symbols are loaded.
GDB_JUMP_TIMEOUT_S = 5.0
# Default STM32CubeProgrammer connection mode for normal programming flows.
DEFAULT_PROGRAMMER_MODE = "hotplug"
# Conservative SWD frequency to improve stability across host/debugger setups.
DEFAULT_PROGRAMMER_FREQ_KHZ = "500"
# Recovery connection mode used when trying to regain control of the target.
DEFAULT_RECOVERY_MODE = "powerdown"
# Recovery APID targets the boot-side access port used during board bring-up.
DEFAULT_RECOVERY_APID = "0"
DEFAULT_SIGNING_HEADER_VERSION = "2.3"
DEFAULT_SIGNING_LOAD_OFFSET = "0x80000000"
SIZE_RE = re.compile(
    r"^\s*(?P<text>\d+)\s+(?P<data>\d+)\s+(?P<bss>\d+)\s+(?P<dec>\d+)\s+(?P<hex>[0-9a-fA-F]+)\s+",
    re.MULTILINE,
)


class WorkflowError(RuntimeError):
    """Raised when the STM32 Cube CLI workflow cannot proceed.

    This exception is intentionally backend-internal. The STM device wrapper
    catches it and translates it into TinyODOM's shared compile/upload result
    dataclasses and public error codes.
    """


class SigningWorkflowError(WorkflowError):
    """Raised when trusted STM32 binary signing fails."""


@dataclass(frozen=True)
class BuildResult:
    """Build result for one STM32 subproject.

    Parameters
    ----------
    log : str
        Combined build log captured from the invoked ``make`` commands.
    debug_dir : pathlib.Path
        Resolved STM32CubeIDE ``Debug`` directory used for the build.
    elf_path : pathlib.Path
        ELF artifact produced by the build.
    """

    log: str
    debug_dir: Path
    elf_path: Path


@dataclass(frozen=True)
class SizeResult:
    """Parsed size information from ``arm-none-eabi-size``.

    Parameters
    ----------
    elf_flash_bytes : int
        Flash usage derived from ``text + data``.
    ram_bytes : int
        RAM usage derived from ``data + bss``.
    raw_output : str
        Raw stdout emitted by ``arm-none-eabi-size`` for logging/debugging.
    """

    elf_flash_bytes: int
    ram_bytes: int
    raw_output: str


@dataclass(frozen=True)
class SignedBinaryResult:
    """Result of producing one STM32 trusted binary image.

    Parameters
    ----------
    log : str
        Combined signing-tool output.
    output_bin : pathlib.Path
        Trusted binary emitted by the signing tool.
    """

    log: str
    output_bin: Path


def which_path(name: str) -> Optional[Path]:
    """Return a ``Path`` for an executable found on ``PATH``.

    Parameters
    ----------
    name : str
        Executable name to resolve.

    Returns
    -------
    pathlib.Path | None
        Resolved executable path when found, otherwise ``None``.
    """
    value = shutil.which(name)
    return Path(value).resolve() if value is not None else None


def default_cubeprog_bin() -> Optional[Path]:
    """Infer the STM32CubeProgrammer ``bin`` directory from the CLI path.

    Returns
    -------
    pathlib.Path | None
        Parent directory of ``STM32_Programmer_CLI`` when the tool is present
        on ``PATH``, otherwise ``None``.
    """
    cli = which_path("STM32_Programmer_CLI")
    return cli.parent if cli is not None else None


def resolve_signing_tool(path: Path | str | None = None) -> Path:
    """Resolve the STM32 signing-tool executable.

    Parameters
    ----------
    path : pathlib.Path | str | None, optional
        Explicit signing-tool path when callers provide one.

    Returns
    -------
    pathlib.Path
        Resolved signing-tool executable path.

    Raises
    ------
    WorkflowError
        If the signing tool cannot be resolved.
    """
    if path is not None:
        return resolve_required_tool_path(
            path,
            label="STM32 signing tool",
            hint="STM32_SigningTool_CLI",
        )
    for hint in ("STM32_SigningTool_CLI", "STM32TrustedPackageCreator_CLI"):
        candidate = which_path(hint)
        if candidate is not None:
            return candidate
    raise WorkflowError(
        "STM32 signing tool was not provided.\n"
        "Install STM32CubeCLT and ensure `STM32_SigningTool_CLI` or "
        "`STM32TrustedPackageCreator_CLI` is on PATH."
    )


def resolve_required_tool_path(
    path: Path | str | None,
    *,
    label: str,
    hint: str,
) -> Path:
    """Resolve a required STM32 tool path and provide a setup hint on failure.

    Parameters
    ----------
    path : pathlib.Path | str | None
        Explicit tool path from configuration, when provided.
    label : str
        Human-readable tool label used in error messages.
    hint : str
        Executable name used for ``PATH`` discovery when ``path`` is omitted.

    Returns
    -------
    pathlib.Path
        Resolved executable path.

    Raises
    ------
    WorkflowError
        If the tool cannot be found or is not executable.
    """
    candidate: Optional[Path]
    if path is None:
        candidate = which_path(hint)
    else:
        candidate = Path(path).expanduser().resolve()
    if candidate is None:
        raise WorkflowError(
            f"{label} was not provided.\n"
            f"Install STM32CubeCLT and ensure `{hint}` is on PATH, or configure the matching device.stm32 field."
        )
    if not candidate.exists():
        raise WorkflowError(f"{label} was not found: {candidate}")
    if candidate.is_file() and not os.access(candidate, os.X_OK):
        raise WorkflowError(f"{label} is not executable: {candidate}")
    return candidate


def validate_project_root(project_root: Path | str) -> Path:
    """Validate an STM32 CubeIDE project root and return the resolved path.

    Parameters
    ----------
    project_root : pathlib.Path | str
        Expected root directory of the STM32 subproject.

    Returns
    -------
    pathlib.Path
        Resolved project root.

    Raises
    ------
    WorkflowError
        If the project root, ``Debug`` directory, or generated makefile is
        missing.
    """
    resolved = Path(project_root).expanduser().resolve()
    if not resolved.is_dir():
        raise WorkflowError(f"STM32 project root does not exist: {resolved}")
    debug_dir = resolved / "Debug"
    if not debug_dir.is_dir():
        raise WorkflowError(f"STM32 Debug directory does not exist: {debug_dir}")
    makefile = debug_dir / "makefile"
    if not makefile.is_file():
        raise WorkflowError(f"STM32 generated makefile does not exist: {makefile}")
    return resolved


def _run_command(
    argv: list[str],
    *,
    cwd: Path | None = None,
) -> str:
    """Run a subprocess and return the combined stdout/stderr text.

    Parameters
    ----------
    argv : list[str]
        Command and arguments to execute.
    cwd : pathlib.Path | None, optional
        Working directory for the subprocess.

    Returns
    -------
    str
        Combined stdout/stderr text.

    Raises
    ------
    WorkflowError
        If the subprocess exits with a non-zero status or the host cannot
        launch the command at all.
    """
    try:
        proc = subprocess.run(
            argv,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise WorkflowError(
            f"Failed to execute command: {' '.join(argv)}\n\n{exc}"
        ) from exc
    output = f"{proc.stdout}{proc.stderr}"
    if proc.returncode != 0:
        detail = output.strip() if output.strip() else "No subprocess output captured."
        raise WorkflowError(
            f"Command failed with exit code {proc.returncode}: {' '.join(argv)}\n\n{detail}"
        )
    return output


def build_project(
    *,
    project_root: Path | str,
    jobs: int | None = None,
    clean: bool = False,
) -> BuildResult:
    """Build an STM32 subproject and resolve the produced ELF.

    Parameters
    ----------
    project_root : pathlib.Path | str
        STM32 project root.
    jobs : int | None, optional
        Optional parallel build job count. Defaults to CPU count.
    clean : bool, optional
        Whether to run ``make clean`` before the main build.

    Returns
    -------
    BuildResult
        Build log plus resolved output locations.
    """
    resolved_project = validate_project_root(project_root)
    logs: list[str] = []
    if clean:
        logs.append(
            _run_command(
                ["make", "-C", "Debug", "clean"],
                cwd=resolved_project,
            )
        )
    logs.append(
        _run_command(
            ["make", "-C", "Debug", f"-j{max(jobs or (os.cpu_count() or 1), 1)}", "all"],
            cwd=resolved_project,
        )
    )
    debug_dir = resolved_project / "Debug"
    # ELF discovery is intentionally separate so later phases can override the
    # artifact name without rewriting the build logic.
    elf_path = resolve_elf_path(debug_dir)
    return BuildResult(
        log="\n".join(text for text in logs if text),
        debug_dir=debug_dir,
        elf_path=elf_path,
    )


def resolve_elf_path(debug_dir: Path | str, elf_name: str | None = None) -> Path:
    """Resolve the built ELF path from a Debug directory.

    Parameters
    ----------
    debug_dir : pathlib.Path | str
        STM32CubeIDE ``Debug`` directory to inspect.
    elf_name : str | None, optional
        Explicit ELF name when the project emits more than one candidate or a
        caller wants deterministic selection.

    Returns
    -------
    pathlib.Path
        Resolved ELF path.

    Raises
    ------
    WorkflowError
        If the requested ELF is missing or ELF discovery is ambiguous.
    """
    resolved_debug_dir = Path(debug_dir).expanduser().resolve()
    if elf_name:
        candidate = resolved_debug_dir / elf_name
        if not candidate.is_file():
            raise WorkflowError(f"Expected ELF does not exist: {candidate}")
        return candidate
    candidates = sorted(resolved_debug_dir.glob("*.elf"))
    if not candidates:
        raise WorkflowError(f"Could not locate an ELF artifact under {resolved_debug_dir}")
    if len(candidates) > 1:
        raise WorkflowError(
            f"Multiple ELF artifacts found under {resolved_debug_dir}; configure an explicit STM ELF name."
        )
    return candidates[0]


def parse_size_output(elf_path: Path | str) -> SizeResult:
    """Run ``arm-none-eabi-size`` on an ELF and return parsed RAM/flash values.

    Parameters
    ----------
    elf_path : pathlib.Path | str
        ELF artifact to inspect.

    Returns
    -------
    SizeResult
        Parsed RAM/flash usage plus the raw size output.

    Raises
    ------
    WorkflowError
        If the size tool fails or its output cannot be parsed.
    """
    resolved_elf = Path(elf_path).expanduser().resolve()
    size_tool = resolve_required_tool_path(
        None,
        label="arm-none-eabi-size",
        hint="arm-none-eabi-size",
    )
    try:
        proc = subprocess.run(
            [str(size_tool), str(resolved_elf)],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise WorkflowError(
            f"Failed to execute arm-none-eabi-size for {resolved_elf}:\n{exc}"
        ) from exc
    if proc.returncode != 0:
        raise WorkflowError(f"arm-none-eabi-size failed for {resolved_elf}:\n{proc.stderr}")
    match = SIZE_RE.search(proc.stdout)
    if not match:
        raise WorkflowError(f"Unable to parse arm-none-eabi-size output:\n{proc.stdout}")
    values = {
        key: int(value, 16) if key == "hex" else int(value)
        for key, value in match.groupdict().items()
    }
    return SizeResult(
        elf_flash_bytes=values["text"] + values["data"],
        ram_bytes=values["data"] + values["bss"],
        raw_output=proc.stdout,
    )


def resolve_required_file_path(path: Path | str | None, *, label: str) -> Path:
    """Resolve a required non-executable file path.

    Parameters
    ----------
    path : pathlib.Path | str | None
        Explicit file path from configuration.
    label : str
        Human-readable label used in error messages.

    Returns
    -------
    pathlib.Path
        Resolved file path.

    Raises
    ------
    WorkflowError
        If the file path is missing or does not exist.
    """
    if path is None:
        raise WorkflowError(f"{label} was not provided.")
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file():
        raise WorkflowError(f"{label} was not found: {candidate}")
    return candidate


def program_external_flash_blob(
    *,
    cubeprog_bin: Path | str | None,
    apid: int,
    weights_blob_path: Path | str,
    weights_flash_address: str,
    external_loader: Path | str,
    recover_first: bool = True,
) -> str:
    """Program a staged external-weight blob through STM32CubeProgrammer.

    Parameters
    ----------
    cubeprog_bin : pathlib.Path | str | None
        STM32CubeProgrammer ``bin`` directory or ``None`` to discover it from
        ``PATH``.
    apid : int
        ST-LINK access port identifier for the application core.
    weights_blob_path : pathlib.Path | str
        Path to the staged binary weights blob.
    weights_flash_address : str
        Absolute external flash address where the blob is programmed.
    external_loader : pathlib.Path | str
        Path to the `.stldr` external loader for the mounted NOR flash device.
    recover_first : bool, default=True
        Whether to run the conservative recovery attach before programming.

    Returns
    -------
    str
        Combined recovery/programmer log text.

    Raises
    ------
    WorkflowError
        If recovery or programming fails.
    """
    return program_external_image(
        cubeprog_bin=cubeprog_bin,
        apid=apid,
        image_path=weights_blob_path,
        flash_address=weights_flash_address,
        external_loader=external_loader,
        programmer_mode=DEFAULT_PROGRAMMER_MODE,
        freq_khz=DEFAULT_PROGRAMMER_FREQ_KHZ,
        recover_first=recover_first,
    )


def sign_binary(
    *,
    signing_tool: Path | str | None,
    input_bin: Path | str,
    output_bin: Path | str,
    load_offset: str = DEFAULT_SIGNING_LOAD_OFFSET,
    header_version: str = DEFAULT_SIGNING_HEADER_VERSION,
) -> SignedBinaryResult:
    """Produce a trusted STM32 binary image.

    Parameters
    ----------
    signing_tool : pathlib.Path | str | None
        Optional explicit signing-tool path.
    input_bin : pathlib.Path | str
        Unsigned binary to sign.
    output_bin : pathlib.Path | str
        Trusted output path to write.
    load_offset : str, default="0x80000000"
        Load offset forwarded to the ST signing tool.
    header_version : str, default="2.3"
        Header version forwarded to the ST signing tool.

    Returns
    -------
    SignedBinaryResult
        Signing log and trusted output path.
    """
    resolved_tool = resolve_signing_tool(signing_tool)
    resolved_input = resolve_required_file_path(input_bin, label="STM32 unsigned binary")
    resolved_output = Path(output_bin).expanduser().resolve()
    if resolved_output.exists():
        resolved_output.unlink()
    base_cmd = [
        str(resolved_tool),
        "-bin",
        str(resolved_input),
        "-nk",
        "-of",
        str(load_offset),
        "-t",
        "fsbl",
        "-o",
        str(resolved_output),
        "-hv",
        str(header_version),
    ]
    try:
        try:
            log = _run_command(base_cmd + ["-align"])
        except WorkflowError:
            log = _run_command(base_cmd)
    except WorkflowError as exc:
        raise SigningWorkflowError(f"STM32 binary signing failed.\n\n{exc}") from exc
    return SignedBinaryResult(log=log, output_bin=resolved_output)


def program_external_image(
    *,
    cubeprog_bin: Path | str | None,
    apid: int,
    image_path: Path | str,
    flash_address: str,
    external_loader: Path | str,
    programmer_mode: str = DEFAULT_PROGRAMMER_MODE,
    freq_khz: str = DEFAULT_PROGRAMMER_FREQ_KHZ,
    recover_first: bool = True,
) -> str:
    """Program one image into external flash.

    Parameters
    ----------
    cubeprog_bin : pathlib.Path | str | None
        Optional STM32CubeProgrammer ``bin`` directory.
    apid : int
        Access-port identifier used for the normal programming attach.
    image_path : pathlib.Path | str
        Binary image to write.
    flash_address : str
        Destination external-flash address.
    external_loader : pathlib.Path | str
        Required `.stldr` loader for the target flash part.
    programmer_mode : str, default="hotplug"
        Normal-programming connect mode.
    freq_khz : str, default="500"
        SWD frequency forwarded to the programmer.
    recover_first : bool, default=True
        Whether to run the conservative recovery attach before programming.

    Returns
    -------
    str
        Combined programmer log text.
    """
    cubeprog_dir = (
        default_cubeprog_bin()
        if cubeprog_bin is None
        else Path(cubeprog_bin).expanduser().resolve()
    )
    if cubeprog_dir is None:
        raise WorkflowError(
            "STM32CubeProgrammer bin directory was not provided.\n"
            "Install STM32CubeProgrammer and ensure `STM32_Programmer_CLI` is on PATH, "
            "or configure device.stm32.cubeprog_bin."
        )
    programmer = resolve_required_tool_path(
        Path(cubeprog_dir) / "STM32_Programmer_CLI",
        label="STM32_Programmer_CLI",
        hint="STM32_Programmer_CLI",
    )
    resolved_image = resolve_required_file_path(image_path, label="STM32 external image")
    resolved_loader = resolve_required_file_path(external_loader, label="STM32 external loader")
    log_parts: list[str] = []
    if recover_first:
        recovery_cmd = [
            str(programmer),
            "-q",
            "-c",
            "port=SWD",
            f"mode={DEFAULT_RECOVERY_MODE}",
            f"freq={freq_khz}",
            f"ap={DEFAULT_RECOVERY_APID}",
        ]
        log_parts.append(_run_command(recovery_cmd).strip())
        time.sleep(1.0)
    program_cmd = [
        str(programmer),
        "-q",
        "-c",
        "port=SWD",
        f"mode={programmer_mode}",
        f"freq={freq_khz}",
        f"ap={int(apid)}",
        "-halt",
        "-el",
        str(resolved_loader),
        "-d",
        str(resolved_image),
        str(flash_address),
        "-v",
    ]
    log_parts.append(_run_command(program_cmd).strip())
    return "\n".join(text for text in log_parts if text)


def classify_build_failure(log_text: str) -> Optional[str]:
    """Classify common STM linker/build failures as flash or RAM overflow.

    Parameters
    ----------
    log_text : str
        Combined build log to classify.

    Returns
    -------
    str | None
        ``"flash"`` or ``"ram"`` for recognized overflow patterns, otherwise
        ``None``.
    """
    lowered = log_text.lower()
    flash_indicators = (
        "region `flash' overflowed",
        "will not fit in region `flash'",
        "region `rom' overflowed",
        "will not fit in region `rom'",
        ".text will not fit",
        "internal flash image exceeds available internal flash",
        "external weight blob exceeds available external flash",
    )
    if any(indicator in lowered for indicator in flash_indicators):
        return "flash"
    if (
        "region `ram' overflowed" in lowered
        or "will not fit in region `ram'" in lowered
        or "cannot move location counter backwards" in lowered
    ):
        return "ram"
    return None


def _wait_for_server_ready(
    proc: subprocess.Popen[str],
    log_file: tempfile._TemporaryFileWrapper[str],
    timeout_s: float,
) -> None:
    """Wait for ``ST-LINK_gdbserver`` to reach its debugger-waiting state.

    Parameters
    ----------
    proc : subprocess.Popen[str]
        Running GDB server process.
    log_file : tempfile._TemporaryFileWrapper[str]
        Temporary file capturing server stdout/stderr.
    timeout_s : float
        Maximum time to wait for the ready banner.

    Returns
    -------
    None

    Raises
    ------
    WorkflowError
        If the server exits, reports a startup error, or never reaches the
        debugger-waiting state.
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


def _summarize_server_log(log_file: tempfile._TemporaryFileWrapper[str]) -> str:
    """Summarize captured GDB server output.

    Parameters
    ----------
    log_file : tempfile._TemporaryFileWrapper[str]
        Temporary file capturing server stdout/stderr.

    Returns
    -------
    str
        Final log excerpt suitable for error messages.
    """
    log_file.flush()
    log_file.seek(0)
    content = log_file.read().strip()
    if not content:
        return "No GDB server output captured."
    lines = content.splitlines()
    return "\n".join(lines[-40:])


def _get_elf_entry_point(elf_path: Path, gdb: Path) -> int:
    """Return the ELF entry point address.

    Parameters
    ----------
    elf_path : pathlib.Path
        ELF artifact to inspect.
    gdb : pathlib.Path
        ``arm-none-eabi-gdb`` executable used to query the ELF metadata.

    Returns
    -------
    int
        ELF entry point address.

    Raises
    ------
    WorkflowError
        If the entry point cannot be determined.
    """
    proc = subprocess.run(
        [str(gdb), "--quiet", "--batch", "-ex", "info files", str(elf_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in proc.stdout.splitlines():
        if "Entry point" in line:
            return int(line.split(":")[-1].strip(), 16)
    raise WorkflowError(f"Could not determine ELF entry point from: {elf_path}")


def _gdb_commands(gdb_port: int, run_after_load: bool, entry_point: int | None = None) -> list[str]:
    """Build the batch GDB command list.

    Parameters
    ----------
    gdb_port : int
        TCP port exposed by the GDB server.
    run_after_load : bool
        Whether GDB should resume execution after loading the ELF.
    entry_point : int | None, optional
        Explicit entry point used for the post-load jump when
        ``run_after_load`` is enabled.

    Returns
    -------
    list[str]
        Ordered batch commands passed to GDB.
    """
    commands = [
        "set pagination off",
        "set confirm off",
        f"target remote localhost:{gdb_port}",
        "monitor halt",
        "load",
        "monitor halt",
    ]
    if run_after_load and entry_point is not None:
        commands.append(f"jump *{hex(entry_point)}")
    else:
        commands.append("quit")
    return commands


def _run_gdb_load(
    *,
    gdb: Path,
    elf_path: Path,
    gdb_port: int,
    run_after_load: bool,
) -> str:
    """Run batch-mode GDB to load the ELF and optionally resume execution.

    Parameters
    ----------
    gdb : pathlib.Path
        ``arm-none-eabi-gdb`` executable.
    elf_path : pathlib.Path
        ELF artifact to load.
    gdb_port : int
        TCP port exposed by the GDB server.
    run_after_load : bool
        Whether execution should resume after ``load`` completes.

    Returns
    -------
    str
        GDB output captured during the load/run sequence.

    Raises
    ------
    WorkflowError
        If GDB reports a hard failure before the expected resume timeout.
    """
    entry_point: int | None = None
    if run_after_load:
        entry_point = _get_elf_entry_point(elf_path, gdb)

    argv = [str(gdb), "--quiet", "--batch"]
    for command in _gdb_commands(gdb_port, run_after_load, entry_point):
        argv.extend(["-ex", command])
    argv.append(str(elf_path))

    if run_after_load:
        # The STM example flow relies on `jump *entry_point` and then expects
        # GDB to stop responding once execution is transferred. The timeout path
        # is therefore treated as success unless the partial output clearly
        # contains a failure signature.
        def _expected_post_jump_disconnect(output: str) -> bool:
            lowered = output.lower()
            loaded_ok = (
                "loading section" in lowered
                and "start address" in lowered
                and "transfer rate" in lowered
            )
            disconnected_after_jump = (
                "remote connection closed" in lowered
                or "lost target connection" in lowered
                or "target disconnected" in lowered
            )
            return loaded_ok and disconnected_after_jump

        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            stdout, _ = proc.communicate(timeout=GDB_JUMP_TIMEOUT_S)
        except subprocess.TimeoutExpired as exc:
            partial_stdout = exc.output or ""
            if isinstance(partial_stdout, bytes):
                partial_stdout = partial_stdout.decode("utf-8", errors="replace")
            proc.kill()
            remaining_stdout, _ = proc.communicate()
            stdout = partial_stdout + (remaining_stdout or "")
            if _expected_post_jump_disconnect(stdout):
                return stdout
            lowered = stdout.lower()
            for indicator in (
                "error",
                "failed",
                "cannot",
                "no such file",
                "remote communication error",
                "connection timed out",
                "connection refused",
                "target disconnected",
                "fault",
            ):
                if indicator in lowered:
                    detail = stdout.strip() if stdout.strip() else "No GDB output captured."
                    raise WorkflowError(
                        "GDB load/run appears to have failed before the expected jump timeout.\n\n"
                        f"Command: {' '.join(argv)}\n\n{detail}"
                    )
            return stdout
        if proc.returncode != 0:
            if _expected_post_jump_disconnect(stdout):
                return stdout
            detail = stdout.strip() if stdout.strip() else "No GDB output captured."
            raise WorkflowError(
                "GDB load/run failed before the jump timeout elapsed.\n\n"
                f"Command: {' '.join(argv)}\n\n{detail}"
            )
        return stdout
    return _run_command(argv)


def debug_load_elf(
    *,
    elf_path: Path | str,
    gdbserver: Path | str | None,
    gdb: Path | str | None,
    cubeprog_bin: Path | str | None,
    gdb_port: int = DEFAULT_GDB_PORT,
    apid: int = DEFAULT_APID,
    server_ready_timeout_s: float = SERVER_READY_TIMEOUT_S,
    run_after_load: bool = True,
) -> str:
    """Debug-load an ELF through ST-LINK and return the combined workflow log.

    Parameters
    ----------
    elf_path : pathlib.Path | str
        ELF artifact to load.
    gdbserver : pathlib.Path | str | None
        Optional explicit path to ``ST-LINK_gdbserver``.
    gdb : pathlib.Path | str | None
        Optional explicit path to ``arm-none-eabi-gdb``.
    cubeprog_bin : pathlib.Path | str | None
        Optional explicit path to the STM32CubeProgrammer ``bin`` directory.
    gdb_port : int, optional
        TCP port used by the GDB server.
    apid : int, optional
        ST-LINK access port identifier.
    server_ready_timeout_s : float, optional
        Timeout waiting for the GDB server to become ready.
    run_after_load : bool, optional
        Whether the target should resume execution after loading.

    Returns
    -------
    str
        Combined GDB and GDB-server log excerpt.

    Raises
    ------
    WorkflowError
        If the ELF is missing or the debug-load workflow fails.
    """
    resolved_elf = Path(elf_path).expanduser().resolve()
    if not resolved_elf.is_file():
        raise WorkflowError(f"ELF does not exist: {resolved_elf}")
    resolved_gdbserver = resolve_required_tool_path(
        gdbserver,
        label="ST-LINK_gdbserver",
        hint="ST-LINK_gdbserver",
    )
    resolved_gdb = resolve_required_tool_path(
        gdb,
        label="arm-none-eabi-gdb",
        hint="arm-none-eabi-gdb",
    )
    resolved_cubeprog_bin = resolve_required_tool_path(
        cubeprog_bin,
        label="STM32CubeProgrammer bin directory",
        hint="STM32_Programmer_CLI",
    )
    if resolved_cubeprog_bin.is_file():
        resolved_cubeprog_bin = resolved_cubeprog_bin.parent

    log_file = tempfile.NamedTemporaryFile(
        mode="w+",
        encoding="utf-8",
        prefix="stm32_gdbserver_",
        suffix=".log",
    )
    proc = subprocess.Popen(
        [
            str(resolved_gdbserver),
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
            str(resolved_cubeprog_bin),
        ],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        # Keep the entire server lifecycle inside this helper. The device class
        # should only need a single "debug-load this ELF" primitive in Phase 1.
        _wait_for_server_ready(proc, log_file, server_ready_timeout_s)
        time.sleep(SERVER_SETTLE_DELAY_S)
        if proc.poll() is not None:
            raise WorkflowError(
                "GDB server exited before GDB connected.\n\n"
                + _summarize_server_log(log_file)
            )
        gdb_output = _run_gdb_load(
            gdb=resolved_gdb,
            elf_path=resolved_elf,
            gdb_port=gdb_port,
            run_after_load=run_after_load,
        )
        server_log = _summarize_server_log(log_file)
        return "\n".join(text for text in (gdb_output.strip(), server_log.strip()) if text)
    except WorkflowError as exc:
        raise WorkflowError(f"{exc}\n\n{_summarize_server_log(log_file)}") from exc
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5.0)
        log_file.close()
