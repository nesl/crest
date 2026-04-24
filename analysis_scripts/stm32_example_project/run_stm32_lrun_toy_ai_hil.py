#!/usr/bin/env python3
"""Run one end-to-end STM32 LRUN toy AI HIL attempt."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import serial

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))

from tinyodom.microcontrollers import stm32_cube_clt  # noqa: E402
from tinyodom.microcontrollers.arduino_base import ensure_harness_firmware  # noqa: E402
from tinyodom.microcontrollers.stm32_runtime import SerialMonitor  # noqa: E402

from stm32_lrun_common import (  # noqa: E402
    DEFAULT_APID,
    DEFAULT_APPLI_FLASH_ADDRESS,
    DEFAULT_BAUD,
    DEFAULT_CONFIG_PATH,
    DEFAULT_CPU_CLOCK_MHZ,
    DEFAULT_DUT_PORT,
    DEFAULT_EXTERNAL_LOADER_NAME,
    DEFAULT_GDB_PORT,
    DEFAULT_HARNESS_PORT,
    DEFAULT_OUTPUT_JSON,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PHASE,
    DEFAULT_PROJECT_ROOT,
    DEFAULT_SERVER_READY_TIMEOUT_S,
    DEFAULT_SIGNING_HEADER_VERSION,
    DEFAULT_SIGNING_LOAD_OFFSET,
    DEFAULT_WEIGHT_STORAGE_MODE,
    DEFAULT_WEIGHTS_FLASH_ADDRESS,
    VALID_CPU_CLOCK_MHZ,
    SigningError,
    build_workspace,
    device_defaults,
    flash_external_image,
    load_config,
    read_staging_manifest,
    resolve_signing_tool,
    resolve_weights_external_loader,
    resolve_workspace_paths,
    sign_binary,
    update_fsbl_lrun_source_size,
    validate_flash_layout,
    write_phase_config_header,
)

LOGGER = logging.getLogger(__name__)
STAGING_SCRIPT = SCRIPT_DIR / "generate_and_stage_stm32_lrun_toy_ai.py"

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

FLOAT_CAPTURE = r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
STRING_CAPTURE = r"(?P<value>.+)"
RUNS_RE = re.compile(r"^runs:\s*(?P<value>\d+)$", re.IGNORECASE)
ARENA_RE = re.compile(r"AI_NETWORK_DATA_ACTIVATIONS_SIZE\s+\((?P<value>\d+)\)")
HEX_DEFINE_RE = re.compile(r"(?P<name>_Min_(?:Heap|Stack)_Size)\s*=\s*(?P<value>0x[0-9A-Fa-f]+)")
DUT_FLOAT_PATTERNS = {
    "clock_hz": re.compile(rf"^clock hz output.*?:\s*{FLOAT_CAPTURE}$", re.IGNORECASE),
    "inference_seq": re.compile(rf"^inference seq output.*?:\s*{FLOAT_CAPTURE}$", re.IGNORECASE),
    "dwt_cycles_per_inference": re.compile(
        rf"^dwt cycles per inference output.*?:\s*{FLOAT_CAPTURE}$", re.IGNORECASE
    ),
    "timer_output_s": re.compile(rf"^timer output.*?:\s*{FLOAT_CAPTURE}$", re.IGNORECASE),
    "timer_per_inference_s": re.compile(
        rf"^timer per inference output.*?:\s*{FLOAT_CAPTURE}$", re.IGNORECASE
    ),
    "timer_per_window_s": re.compile(
        rf"^timer per window output.*?:\s*{FLOAT_CAPTURE}$", re.IGNORECASE
    ),
    "wake_recovery_us": re.compile(rf"^wake recovery us.*?:\s*{FLOAT_CAPTURE}$", re.IGNORECASE),
    "wake_overshoot_us": re.compile(rf"^wake overshoot us.*?:\s*{FLOAT_CAPTURE}$", re.IGNORECASE),
    "rtc_sleep_total_ms": re.compile(rf"^rtc sleep total ms.*?:\s*{FLOAT_CAPTURE}$", re.IGNORECASE),
    "rtc_clock_hz_nominal": re.compile(
        rf"^rtc clock hz nominal output.*?:\s*{FLOAT_CAPTURE}$", re.IGNORECASE
    ),
    "cadence_deadline_misses": re.compile(
        rf"^cadence deadline misses.*?:\s*{FLOAT_CAPTURE}$", re.IGNORECASE
    ),
}
DUT_STRING_PATTERNS = {
    "phase": re.compile(rf"^phase output:\s*{STRING_CAPTURE}$", re.IGNORECASE),
    "rtc_clock_source": re.compile(rf"^rtc clock source output:\s*{STRING_CAPTURE}$", re.IGNORECASE),
    "cadence_timing_quality": re.compile(
        rf"^cadence timing quality output:\s*{STRING_CAPTURE}$", re.IGNORECASE
    ),
    "stop_mode_variant": re.compile(rf"^stop mode variant output:\s*{STRING_CAPTURE}$", re.IGNORECASE),
}
HARNESS_FLOAT_PATTERNS = {
    "inference_seq": re.compile(rf"^inference seq.*?:\s*{FLOAT_CAPTURE}$", re.IGNORECASE),
    "energy_mj_per_inference": re.compile(rf"^energy output.*?:\s*{FLOAT_CAPTURE}$", re.IGNORECASE),
    "avg_power_mw": re.compile(rf"^avg power output.*?:\s*{FLOAT_CAPTURE}$", re.IGNORECASE),
    "avg_current_ma": re.compile(rf"^avg current output.*?:\s*{FLOAT_CAPTURE}$", re.IGNORECASE),
    "bus_voltage_v": re.compile(rf"^bus voltage output.*?:\s*{FLOAT_CAPTURE}$", re.IGNORECASE),
    "idle_power_mw": re.compile(rf"^idle power baseline.*?:\s*{FLOAT_CAPTURE}$", re.IGNORECASE),
    "harness_latency_s": re.compile(rf"^harness timer output.*?:\s*{FLOAT_CAPTURE}$", re.IGNORECASE),
}


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )


def _which_path(name: str) -> Path | None:
    value = shutil.which(name)  # type: ignore[name-defined]
    return Path(value).resolve() if value is not None else None


def _build_parser() -> argparse.ArgumentParser:
    config_defaults = device_defaults(load_config(DEFAULT_CONFIG_PATH))
    parser = argparse.ArgumentParser(description="Run one STM32 LRUN toy AI HIL attempt.")
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--stage-output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--phase", choices=["back_to_back", "cadenced"], default=config_defaults["phase"])
    parser.add_argument("--cpu-clock-mhz", type=int, default=DEFAULT_CPU_CLOCK_MHZ)
    parser.add_argument("--weight-storage-mode", choices=["embedded", "external_flash"], default=config_defaults["weight_storage_mode"])
    parser.add_argument("--weights-flash-address", default=DEFAULT_WEIGHTS_FLASH_ADDRESS)
    parser.add_argument("--weights-memory-pool", type=Path, default=config_defaults["weights_memory_pool"])
    parser.add_argument("--weights-external-loader", type=Path, default=None)
    parser.add_argument("--appli-flash-address", default=DEFAULT_APPLI_FLASH_ADDRESS)
    parser.add_argument("--dut-port", default=config_defaults["dut_port"])
    parser.add_argument("--harness-port", default=config_defaults["harness_port"])
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--dut-ready-timeout", type=float, default=max(15.0, config_defaults["dut_ready_timeout_s"]))
    parser.add_argument("--serial-timeout", type=float, default=config_defaults["serial_timeout_s"])
    parser.add_argument("--harness-ready-timeout", type=float, default=config_defaults["harness_ready_timeout_s"])
    parser.add_argument("--harness-done-timeout", type=float, default=config_defaults["harness_done_timeout_s"])
    parser.add_argument("--latency-budget-ms", type=float, default=config_defaults["latency_budget_ms"])
    parser.add_argument("--measured-runs", type=int, default=config_defaults["measured_runs"])
    parser.add_argument("--wake-margin-us", type=int, default=config_defaults["wake_margin_us"])
    parser.add_argument("--min-sleep-us", type=int, default=config_defaults["min_sleep_us"])
    parser.add_argument("--reuse-staged-model", action="store_true")
    parser.add_argument("--skip-harness", action="store_true")
    parser.add_argument("--stub-only", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--gdbserver", type=Path, default=_which_path("ST-LINK_gdbserver"))
    parser.add_argument("--gdb", type=Path, default=_which_path("arm-none-eabi-gdb"))
    parser.add_argument("--cubeprog-bin", type=Path, default=stm32_cube_clt.default_cubeprog_bin())
    parser.add_argument(
        "--signing-tool",
        type=Path,
        default=_which_path("STM32_SigningTool_CLI") or _which_path("STM32TrustedPackageCreator_CLI"),
    )
    parser.add_argument("--gdb-port", type=int, default=DEFAULT_GDB_PORT)
    parser.add_argument("--apid", type=int, default=DEFAULT_APID)
    parser.add_argument("--server-ready-timeout", type=float, default=DEFAULT_SERVER_READY_TIMEOUT_S)
    parser.add_argument("--signing-header-version", default=DEFAULT_SIGNING_HEADER_VERSION)
    parser.add_argument("--signing-load-offset", default=DEFAULT_SIGNING_LOAD_OFFSET)
    parser.add_argument("--harness-fqbn", default=str(config_defaults["harness_fqbn"]))
    parser.add_argument("--harness-auto-flash", default=str(config_defaults["harness_auto_flash"]))
    parser.add_argument("--verbose", action="store_true")
    return parser


def _run_stage_generation(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(STAGING_SCRIPT),
        "--config",
        str(args.config.resolve()),
        "--project-root",
        str(args.project_root.resolve()),
        "--output-root",
        str(args.stage_output_root.resolve()),
        "--weight-storage-mode",
        args.weight_storage_mode,
        "--weights-flash-address",
        args.weights_flash_address,
        "--weights-memory-pool",
        str(args.weights_memory_pool.resolve()),
    ]
    if args.clean:
        cmd.append("--clean")
    if args.model is not None:
        cmd.extend(["--model", str(args.model.resolve())])
    if args.weights_external_loader is not None:
        cmd.extend(["--weights-external-loader", str(args.weights_external_loader.resolve())])
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise stm32_cube_clt.WorkflowError(
            f"LRUN staging helper failed with exit code {proc.returncode}: {' '.join(cmd)}\n\n"
            f"{(proc.stdout or '')}{(proc.stderr or '')}".strip()
        )


def _parse_last_float_fields(lines: list[str], patterns: dict[str, re.Pattern[str]]) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for line in lines:
        stripped = line.strip()
        for key, pattern in patterns.items():
            match = pattern.match(stripped)
            if match:
                parsed[key] = float(match.group("value"))
    return parsed


def _parse_last_string_fields(lines: list[str], patterns: dict[str, re.Pattern[str]]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        for key, pattern in patterns.items():
            match = pattern.match(stripped)
            if match:
                parsed[key] = match.group("value").strip()
    return parsed


def _parse_runs(lines: list[str]) -> int | None:
    value = None
    for line in lines:
        match = RUNS_RE.match(line.strip())
        if match:
            value = int(match.group("value"))
    return value


def _parse_arena_bytes(header_path: Path) -> int:
    if not header_path.is_file():
        return -1
    match = ARENA_RE.search(header_path.read_text(encoding="utf-8"))
    return int(match.group("value")) if match else -1


def _parse_linker_reservations(linker_script: Path) -> dict[str, int]:
    reservations: dict[str, int] = {}
    if not linker_script.is_file():
        return reservations
    for match in HEX_DEFINE_RE.finditer(linker_script.read_text(encoding="utf-8")):
        reservations[match.group("name")] = int(match.group("value"), 16)
    return reservations


def _diagnostic_output_path(output_path: Path) -> Path:
    if output_path.suffix:
        return output_path.with_name(f"{output_path.stem}.diagnostics{output_path.suffix}")
    return output_path.with_name(f"{output_path.name}.diagnostics.json")


def _wait_for_match(
    monitor: SerialMonitor,
    predicate,
    timeout_s: float,
    stage: str,
    cursor: int,
) -> tuple[str | None, int]:
    return monitor.wait_for_match(predicate, timeout_s, stage, start_index=cursor)


def _wait_for_protocol(
    dut_monitor: SerialMonitor,
    *,
    dut_ready_timeout: float,
    serial_timeout: float,
) -> tuple[int, list[str]]:
    cursor = 0
    matched, cursor = _wait_for_match(
        dut_monitor,
        lambda line: line.strip() == "STM32_AI_INIT=OK" or line.startswith("STM32_AI_INIT=FAIL"),
        dut_ready_timeout,
        "STM32_AI_INIT result",
        cursor,
    )
    if matched is None:
        raise RuntimeError("Timed out waiting for STM32_AI_INIT result.")
    if matched.startswith("STM32_AI_INIT=FAIL"):
        raise RuntimeError(f"DUT reported init failure: {matched}")

    matched, cursor = _wait_for_match(
        dut_monitor,
        lambda line: line.strip() == "DUT READY",
        dut_ready_timeout,
        "DUT READY",
        cursor,
    )
    if matched is None:
        raise RuntimeError("Timed out waiting for DUT READY.")

    dut_monitor.write_line("START")
    matched, cursor = _wait_for_match(
        dut_monitor,
        lambda line: line.strip() == "STM32_AI_RUN=OK" or line.startswith("STM32_AI_RUN=FAIL"),
        serial_timeout,
        "STM32_AI_RUN result",
        cursor,
    )
    if matched is None:
        raise RuntimeError("Timed out waiting for STM32_AI_RUN result.")
    if matched.startswith("STM32_AI_RUN=FAIL"):
        raise RuntimeError(f"DUT reported run failure: {matched}")
    return cursor, dut_monitor.snapshot_lines()


def _boot_target(
    *,
    args: argparse.Namespace,
    fsbl_elf: Path,
    appli_trusted: Path,
    weights_blob: Path | None,
    external_loader: Path,
) -> tuple[list[str], bool]:
    logs: list[str] = []
    weights_programmed = False
    logs.append(
        flash_external_image(
            cubeprog_bin=args.cubeprog_bin,
            apid=args.apid,
            image_path=appli_trusted,
            flash_address=args.appli_flash_address,
            external_loader=external_loader,
        )
    )
    if weights_blob is not None:
        logs.append(
            flash_external_image(
                cubeprog_bin=args.cubeprog_bin,
                apid=args.apid,
                image_path=weights_blob,
                flash_address=args.weights_flash_address,
                external_loader=external_loader,
            )
        )
        weights_programmed = True
    logs.append(
        stm32_cube_clt.debug_load_elf(
            elf_path=fsbl_elf,
            gdbserver=args.gdbserver,
            gdb=args.gdb,
            cubeprog_bin=args.cubeprog_bin,
            gdb_port=args.gdb_port,
            apid=args.apid,
            server_ready_timeout_s=args.server_ready_timeout,
            run_after_load=True,
        )
    )
    return logs, weights_programmed


def _latency_from_metrics(phase: str, dut_metrics: dict[str, float], harness_metrics: dict[str, float]) -> float:
    if "harness_latency_s" in harness_metrics:
        return harness_metrics["harness_latency_s"] * 1000.0
    if phase == "cadenced" and dut_metrics.get("timer_per_window_s", -1.0) >= 0.0:
        return dut_metrics["timer_per_window_s"] * 1000.0
    if dut_metrics.get("timer_per_inference_s", -1.0) >= 0.0:
        return dut_metrics["timer_per_inference_s"] * 1000.0
    if dut_metrics.get("timer_output_s", -1.0) >= 0.0:
        return dut_metrics["timer_output_s"] * 1000.0
    return -1.0


def _validate_effective_clock(requested_mhz: int, observed_hz: float) -> tuple[bool, str | None]:
    if observed_hz <= 0.0:
        return False, "DUT did not report a valid clock_hz value."
    requested_hz = float(requested_mhz) * 1_000_000.0
    tolerance_hz = max(1_000_000.0, requested_hz * 0.02)
    if abs(observed_hz - requested_hz) <= tolerance_hz:
        return True, None
    return False, (
        f"Observed clock_hz {observed_hz:.0f} does not match requested "
        f"{requested_hz:.0f} within tolerance {tolerance_hz:.0f}."
    )


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    _configure_logging(args.verbose)

    config = load_config(args.config.resolve())
    config_defaults = device_defaults(config)
    valid_cpu_clocks = tuple(int(value) for value in config_defaults["cpu_clock_mhz_options"])
    if args.cpu_clock_mhz not in valid_cpu_clocks:
        parser.error(f"--cpu-clock-mhz must be one of {valid_cpu_clocks}")
    project_root = args.project_root.resolve()
    output_path = args.output.resolve()
    diagnostic_path = _diagnostic_output_path(output_path)
    stage_output_root = args.stage_output_root.resolve()
    workspace = resolve_workspace_paths(project_root)
    phase_header = workspace.appli_source_root / "Inc" / "toy_ai_phase_config.h"
    phase_config_changed = False
    build = None
    fsbl_elf = workspace.fsbl_debug_dir / "unknown.elf"
    app_elf = workspace.app_debug_dir / "unknown.elf"
    app_trusted: Path | None = None
    sign_logs: list[str] = []
    staging_manifest = {
        "weight_storage_mode": args.weight_storage_mode,
        "weights_blob_path": None,
        "weights_blob_size": 0,
        "weights_flash_address": None,
    }
    resolved_weight_storage_mode = args.weight_storage_mode
    weights_blob: Path | None = None
    weights_blob_size = 0
    external_loader: Path | None = None

    if not args.skip_harness:
        ensure_harness_firmware(
            harness_serial_port=args.harness_port,
            harness_fqbn=args.harness_fqbn,
            harness_auto_flash=args.harness_auto_flash,
            build_defines={"TINYODOM_INFERENCE_RUNS": int(args.measured_runs)},
        )

    dut_monitor: SerialMonitor | None = None
    harness_monitor: SerialMonitor | None = None
    dut_lines: list[str] = []
    harness_lines: list[str] = []
    dut_metrics: dict[str, float] = {}
    dut_strings: dict[str, str] = {}
    harness_metrics: dict[str, float] = {}
    boot_logs: list[str] = []
    error_code = MASTER_PENDING
    error_kind: str | None = None
    error_detail: str | None = None
    weights_programmed = False
    fsbl_copy_window_bytes = -1

    try:
        if args.stub_only:
            LOGGER.info("Stub-only mode enabled; skipping ST Edge AI staging.")
        elif not args.reuse_staged_model:
            LOGGER.info("Rebuilding and staging the fixed LRUN toy model.")
            _run_stage_generation(args)

        phase_header, phase_config_changed = write_phase_config_header(
            project_root,
            phase=args.phase,
            latency_budget_ms=args.latency_budget_ms,
            measured_runs=args.measured_runs,
            cpu_clock_mhz=args.cpu_clock_mhz,
            wake_margin_us=args.wake_margin_us,
            min_sleep_us=args.min_sleep_us,
            stub_only=args.stub_only,
        )
        build_clean = bool(args.clean or phase_config_changed)
        if phase_config_changed and not args.clean:
            LOGGER.info("Phase config changed; forcing a clean rebuild.")

        build = build_workspace(project_root, clean=build_clean, jobs=args.jobs)
        fsbl_elf = build.fsbl_build.elf_path
        app_elf = build.app_build.elf_path
        signing_tool = resolve_signing_tool(args.signing_tool)
        app_trusted = build.app_bin.with_name(f"{build.app_bin.stem}-trusted.bin")
        sign_logs = [
            sign_binary(
                signing_tool=signing_tool,
                input_bin=build.app_bin,
                output_bin=app_trusted,
                load_offset=args.signing_load_offset,
                header_version=args.signing_header_version,
            )
        ]
        _, fsbl_copy_window_changed, fsbl_copy_window_bytes = update_fsbl_lrun_source_size(
            project_root,
            trusted_app_size=app_trusted.stat().st_size,
        )
        if fsbl_copy_window_changed:
            LOGGER.info(
                "Updated FSBL LRUN copy window to 0x%08X bytes; rebuilding Boot project.",
                fsbl_copy_window_bytes,
            )
            fsbl_build = stm32_cube_clt.build_project(
                project_root=project_root / "STM32CubeIDE" / "Boot",
                jobs=args.jobs,
                clean=False,
            )
            build = build.__class__(
                fsbl_build=fsbl_build,
                app_build=build.app_build,
                fsbl_size=stm32_cube_clt.parse_size_output(fsbl_build.elf_path),
                app_size=build.app_size,
                fsbl_bin=build.fsbl_bin.parent / f"{fsbl_build.elf_path.stem}.bin",
                app_bin=build.app_bin,
            )
            fsbl_elf = build.fsbl_build.elf_path
        staging_manifest = (
            {
                "weight_storage_mode": args.weight_storage_mode,
                "weights_blob_path": None,
                "weights_blob_size": 0,
                "weights_flash_address": None,
            }
            if args.stub_only
            else read_staging_manifest(stage_output_root)
        )
        resolved_weight_storage_mode = str(staging_manifest.get("weight_storage_mode", args.weight_storage_mode))
        weights_blob_path = staging_manifest.get("weights_blob_path")
        weights_blob = Path(str(weights_blob_path)).resolve() if weights_blob_path else None
        weights_blob_size = int(staging_manifest.get("weights_blob_size") or 0)
        if weights_blob is not None and not weights_blob.is_file():
            raise stm32_cube_clt.WorkflowError(f"Weights blob not found: {weights_blob}")

        external_loader = resolve_weights_external_loader(args.cubeprog_bin, args.weights_external_loader)
        validate_flash_layout(
            appli_trusted_size=app_trusted.stat().st_size,
            appli_flash_address=args.appli_flash_address,
            weights_flash_address=args.weights_flash_address,
        )

        LOGGER.info("Opening DUT serial monitor on %s", args.dut_port)
        dut_monitor = SerialMonitor(args.dut_port, args.baud, "dut")
        if not args.skip_harness:
            LOGGER.info("Opening harness serial monitor on %s", args.harness_port)
            harness_monitor = SerialMonitor(args.harness_port, args.baud, "harness")
            harness_monitor.write_line("PING")
            if (
                harness_monitor.wait_for(
                    lambda line: line.strip().lower() == "harness ready",
                    args.harness_ready_timeout,
                    "HARNESS READY",
                )
                is None
            ):
                raise RuntimeError("Timed out waiting for HARNESS READY.")

        boot_logs, weights_programmed = _boot_target(
            args=args,
            fsbl_elf=fsbl_elf,
            appli_trusted=app_trusted,
            weights_blob=weights_blob if resolved_weight_storage_mode == "external_flash" else None,
            external_loader=external_loader,
        )
        _, dut_lines = _wait_for_protocol(
            dut_monitor,
            dut_ready_timeout=args.dut_ready_timeout,
            serial_timeout=args.serial_timeout,
        )
        if harness_monitor is not None:
            if harness_monitor.wait_for(
                lambda line: line.strip() == "DONE",
                args.serial_timeout + args.harness_done_timeout,
                "HARNESS DONE",
            ) is None:
                raise RuntimeError("Timed out waiting for harness DONE.")
            time.sleep(0.5)
            harness_lines = harness_monitor.snapshot_lines()
        else:
            time.sleep(0.5)

        dut_lines = dut_monitor.snapshot_lines()
        dut_metrics = _parse_last_float_fields(dut_lines, DUT_FLOAT_PATTERNS)
        dut_strings = _parse_last_string_fields(dut_lines, DUT_STRING_PATTERNS)
        harness_metrics = _parse_last_float_fields(harness_lines, HARNESS_FLOAT_PATTERNS)

        runs_dut = _parse_runs(dut_lines)
        if runs_dut != args.measured_runs:
            raise RuntimeError(f"Run-count mismatch from DUT: expected {args.measured_runs}, got {runs_dut}")
        if harness_monitor is not None:
            runs_harness = _parse_runs(harness_lines)
            if runs_harness != args.measured_runs:
                raise RuntimeError(
                    f"Run-count mismatch from harness: expected {args.measured_runs}, got {runs_harness}"
                )
        error_code = MASTER_SUCCESS
    except (stm32_cube_clt.WorkflowError, RuntimeError, serial.SerialException, subprocess.CalledProcessError) as exc:
        LOGGER.error("%s", exc)
        if isinstance(exc, serial.SerialException):
            error_code = MASTER_DEVICE_NOT_FOUND
            error_kind = "serial"
        elif isinstance(exc, SigningError):
            error_code = MASTER_FATAL
            error_kind = "signing"
        else:
            error_code = MASTER_FATAL
            error_kind = "workflow" if isinstance(exc, stm32_cube_clt.WorkflowError) else "runtime"
        error_detail = str(exc)
        if dut_monitor is not None:
            dut_lines = dut_monitor.snapshot_lines()
        if harness_monitor is not None:
            harness_lines = harness_monitor.snapshot_lines()
        dut_metrics = _parse_last_float_fields(dut_lines, DUT_FLOAT_PATTERNS)
        dut_strings = _parse_last_string_fields(dut_lines, DUT_STRING_PATTERNS)
        harness_metrics = _parse_last_float_fields(harness_lines, HARNESS_FLOAT_PATTERNS)
    finally:
        if dut_monitor is not None:
            dut_monitor.close()
        if harness_monitor is not None:
            harness_monitor.close()

    latency_ms = _latency_from_metrics(args.phase, dut_metrics, harness_metrics)
    timer_per_inference_ms = (
        dut_metrics["timer_per_inference_s"] * 1000.0
        if dut_metrics.get("timer_per_inference_s", -1.0) >= 0.0
        else -1.0
    )
    timer_per_window_ms = (
        dut_metrics["timer_per_window_s"] * 1000.0
        if dut_metrics.get("timer_per_window_s", -1.0) >= 0.0
        else -1.0
    )
    harness_latency_ms = (
        harness_metrics["harness_latency_s"] * 1000.0
        if harness_metrics.get("harness_latency_s", -1.0) >= 0.0
        else -1.0
    )
    energy_mj_per_inference = float(harness_metrics.get("energy_mj_per_inference", -1.0))
    energy_mj_per_window = (
        energy_mj_per_inference * float(args.measured_runs)
        if energy_mj_per_inference >= 0.0
        else -1.0
    )
    observed_clock_hz = float(dut_metrics.get("clock_hz", -1.0))
    clock_validation_ok, clock_validation_message = _validate_effective_clock(
        int(args.cpu_clock_mhz),
        observed_clock_hz,
    )
    if not clock_validation_ok and error_kind is None:
        LOGGER.warning("%s", clock_validation_message)

    app_elf_flash_bytes = int(build.app_size.elf_flash_bytes) if build is not None else -1
    fsbl_elf_flash_bytes = int(build.fsbl_size.elf_flash_bytes) if build is not None else -1
    elf_flash_bytes = (
        int(build.app_size.elf_flash_bytes + build.fsbl_size.elf_flash_bytes)
        if build is not None
        else -1
    )
    signed_flash_bytes = (
        int(app_trusted.stat().st_size if app_trusted is not None and app_trusted.is_file() else 0)
    )
    contract = {
        "ram_bytes": int(build.app_size.ram_bytes) if build is not None else -1,
        "flash_bytes": int(elf_flash_bytes + weights_blob_size) if elf_flash_bytes >= 0 else -1,
        "elf_flash_bytes": int(elf_flash_bytes),
        "weights_flash_bytes": int(weights_blob_size),
        "weights_programmed": bool(weights_programmed),
        "external_flash_bytes": int(weights_blob_size),
        "appli_elf_flash_bytes": app_elf_flash_bytes,
        "fsbl_elf_flash_bytes": fsbl_elf_flash_bytes,
        "signed_flash_bytes": signed_flash_bytes,
        "weight_storage_mode": resolved_weight_storage_mode,
        "weights_flash_address": (
            args.weights_flash_address if resolved_weight_storage_mode == "external_flash" else None
        ),
        "fsbl_image_path": str(fsbl_elf),
        "appli_image_path": str(app_elf),
        "appli_signed_image_path": str(app_trusted) if app_trusted is not None else None,
        "appli_flash_address": args.appli_flash_address,
        "phase": args.phase,
        "boot_mode": "dev_boot",
        "fsbl_copy_window_bytes": fsbl_copy_window_bytes,
        "latency_ms": latency_ms,
        "latency_budget_ms": float(args.latency_budget_ms),
        "arena_bytes": int(_parse_arena_bytes(workspace.appli_source_root / "Inc" / "network_data_params.h")),
        "hil_enabled": not args.skip_harness,
        "cpu_clock_mhz_requested": int(args.cpu_clock_mhz),
        "clock_hz": observed_clock_hz,
        "clock_validation_ok": bool(clock_validation_ok),
        "dwt_cycles_per_inference": float(dut_metrics.get("dwt_cycles_per_inference", -1.0)),
        "energy_mj_per_inference": energy_mj_per_inference,
        "energy_mj_per_window": energy_mj_per_window,
        "avg_power_mw": float(harness_metrics.get("avg_power_mw", -1.0)),
        "avg_current_ma": float(harness_metrics.get("avg_current_ma", -1.0)),
        "bus_voltage_v": float(harness_metrics.get("bus_voltage_v", -1.0)),
        "idle_power_mw": float(harness_metrics.get("idle_power_mw", -1.0)),
        "harness_latency_ms": harness_latency_ms,
        "active_inference_latency_ms": timer_per_inference_ms,
        "window_latency_ms": timer_per_window_ms,
        "rtc_sleep_ms": float(dut_metrics.get("rtc_sleep_total_ms", -1.0)),
        "deadline_miss_count": int(dut_metrics.get("cadence_deadline_misses", -1.0)),
        "wake_recovery_us_mean": float(dut_metrics.get("wake_recovery_us", -1.0)),
        "wake_overshoot_us_mean": float(dut_metrics.get("wake_overshoot_us", -1.0)),
        "inference_seq": int(harness_metrics.get("inference_seq", dut_metrics.get("inference_seq", -1.0))),
        "measured_runs": int(args.measured_runs),
        "rtc_clock_source": dut_strings.get("rtc_clock_source"),
        "rtc_clock_hz_nominal": int(dut_metrics.get("rtc_clock_hz_nominal", -1.0)),
        "cadence_timing_quality": dut_strings.get("cadence_timing_quality"),
        "stop_mode_variant": dut_strings.get("stop_mode_variant"),
        "error_code": int(error_code),
        "error_label": ERROR_LABELS.get(int(error_code), f"UNKNOWN_{error_code}"),
    }

    diagnostic = {
        "phase_config_header": str(phase_header),
        "phase_config_changed": bool(phase_config_changed),
        "staging_manifest": staging_manifest,
        "fsbl_build_log": build.fsbl_build.log if build is not None else None,
        "app_build_log": build.app_build.log if build is not None else None,
        "sign_logs": sign_logs,
        "boot_logs": boot_logs,
        "fsbl_elf": str(fsbl_elf),
        "app_elf": str(app_elf),
        "fsbl_bin": str(build.fsbl_bin) if build is not None else None,
        "app_bin": str(build.app_bin) if build is not None else None,
        "app_trusted_image": str(app_trusted) if app_trusted is not None else None,
        "appli_flash_address": args.appli_flash_address,
        "weights_flash_address": args.weights_flash_address,
        "weights_programmed": bool(weights_programmed),
        "fsbl_size_raw": build.fsbl_size.raw_output if build is not None else None,
        "app_size_raw": build.app_size.raw_output if build is not None else None,
        "app_linker_reservations": _parse_linker_reservations(workspace.app_project_root / "STM32N657XX_LRUN.ld"),
        "boot_linker_reservations": _parse_linker_reservations(workspace.boot_project_root / "STM32N657XX_AXISRAM2_fsbl.ld"),
        "dut_lines": dut_lines,
        "harness_lines": harness_lines,
        "dut_metrics": dut_metrics,
        "dut_strings": dut_strings,
        "harness_metrics": harness_metrics,
        "skip_harness": bool(args.skip_harness),
        "stub_only": bool(args.stub_only),
        "external_loader": str(external_loader) if external_loader is not None else None,
        "boot_mode": "dev_boot",
        "fsbl_copy_window_bytes": fsbl_copy_window_bytes,
        "config_path": str(args.config.resolve()),
        "error_kind": error_kind,
        "error_detail": error_detail,
        "clock_validation_ok": bool(clock_validation_ok),
        "clock_validation_message": clock_validation_message,
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
