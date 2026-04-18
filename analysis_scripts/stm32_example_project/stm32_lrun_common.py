#!/usr/bin/env python3
"""Shared helpers for the parallel STM32 LRUN analysis track."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tinyodom.microcontrollers import stm32_cube_clt

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

DEFAULT_PROJECT_ROOT = SCRIPT_DIR / "stm32_lrun_toy_ai_project"
DEFAULT_CONFIG_PATH = REPO_ROOT / "src" / "nas_config.yaml"
DEFAULT_OUTPUT_ROOT = Path("/tmp/tinyodom_stm32_lrun_generate")
DEFAULT_OUTPUT_JSON = SCRIPT_DIR / "stm32_lrun_toy_ai_metrics.json"
DEFAULT_DUT_PORT = "/dev/ttyACM0"
DEFAULT_HARNESS_PORT = "/dev/ttyACM1"
DEFAULT_BAUD = 115200
DEFAULT_HARNESS_FQBN = "arduino:mbed_nano:nano33ble"
DEFAULT_HARNESS_AUTO_FLASH = "once"
DEFAULT_GDB_PORT = stm32_cube_clt.DEFAULT_GDB_PORT
DEFAULT_APID = stm32_cube_clt.DEFAULT_APID
DEFAULT_SERVER_READY_TIMEOUT_S = stm32_cube_clt.SERVER_READY_TIMEOUT_S
DEFAULT_PHASE = "cadenced"
DEFAULT_WEIGHT_STORAGE_MODE = "external_flash"
DEFAULT_MEASURED_RUNS = 10
DEFAULT_CPU_CLOCK_MHZ = 600
VALID_CPU_CLOCK_MHZ = (200, 300, 400, 600, 800)
DEFAULT_APPLI_FLASH_ADDRESS = "0x70100000"
DEFAULT_WEIGHTS_FLASH_ADDRESS = "0x71000000"
DEFAULT_SIGNING_HEADER_VERSION = "2.3"
DEFAULT_SIGNING_LOAD_OFFSET = "0x80000000"
DEFAULT_WEIGHTS_MEMORY_POOL = SCRIPT_DIR / "nucleo_mypool.json"
DEFAULT_EXTERNAL_LOADER_NAME = "MX25UM51245G_STM32N6570-NUCLEO.stldr"
DEFAULT_PROGRAMMER_MODE = stm32_cube_clt.DEFAULT_PROGRAMMER_MODE
DEFAULT_PROGRAMMER_FREQ_KHZ = stm32_cube_clt.DEFAULT_PROGRAMMER_FREQ_KHZ
DEFAULT_RECOVERY_MODE = stm32_cube_clt.DEFAULT_RECOVERY_MODE
DEFAULT_RECOVERY_APID = stm32_cube_clt.DEFAULT_RECOVERY_APID
STAGING_MANIFEST_NAME = "staging_manifest.json"
EXPECTED_OUTPUTS = (
    "network.c",
    "network.h",
    "network_config.h",
    "network_data.c",
    "network_data.h",
    "network_data_params.c",
    "network_data_params.h",
)
WEIGHTS_BLOB_NAME = "network_data.bin"
EXTMEM_COPY_WINDOW_ALIGN = 0x400


class SigningError(stm32_cube_clt.WorkflowError):
    """Raised when STM32 trusted-image signing fails."""


@dataclass(frozen=True)
class WorkspacePaths:
    root: Path
    fsbl_source_root: Path
    appli_source_root: Path
    boot_project_root: Path
    app_project_root: Path
    fsbl_debug_dir: Path
    app_debug_dir: Path


@dataclass(frozen=True)
class BuildArtifacts:
    fsbl_build: stm32_cube_clt.BuildResult
    app_build: stm32_cube_clt.BuildResult
    fsbl_size: stm32_cube_clt.SizeResult
    app_size: stm32_cube_clt.SizeResult
    fsbl_bin: Path
    app_bin: Path


def load_config(config_path: Path | str) -> dict[str, Any]:
    resolved = Path(config_path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise stm32_cube_clt.WorkflowError(f"Unexpected YAML root in config: {resolved}")
    return loaded


def default_latency_budget_ms(config: dict[str, Any]) -> float:
    device = dict(config.get("device") or {})
    if device.get("latency_budget_ms") is not None:
        return float(device["latency_budget_ms"])
    data = dict(config.get("data") or {})
    stride = float(data.get("stride", 20))
    sampling_rate_hz = float(data.get("sampling_rate_hz", 100))
    return (stride / sampling_rate_hz) * 1000.0


def device_defaults(config: dict[str, Any]) -> dict[str, Any]:
    device = dict(config.get("device") or {})
    stm32 = dict(device.get("stm32") or {})
    return {
        "phase": str(device.get("runtime_mode", DEFAULT_PHASE)),
        "latency_budget_ms": float(default_latency_budget_ms(config)),
        "measured_runs": int(device.get("measured_inference_runs", DEFAULT_MEASURED_RUNS)),
        "dut_port": str(device.get("serial_port", DEFAULT_DUT_PORT)),
        "dut_ready_timeout_s": float(device.get("dut_ready_timeout_s", 5.0)),
        "serial_timeout_s": float(device.get("serial_timeout_s", 12.0)),
        "harness_port": str(device.get("harness_serial_port", DEFAULT_HARNESS_PORT)),
        "harness_fqbn": str(device.get("harness_fqbn", DEFAULT_HARNESS_FQBN)),
        "harness_auto_flash": str(device.get("harness_auto_flash", DEFAULT_HARNESS_AUTO_FLASH)),
        "harness_ready_timeout_s": float(device.get("harness_ready_timeout_s", 5.0)),
        "harness_done_timeout_s": float(device.get("harness_done_timeout_s", 5.0)),
        "weight_storage_mode": str(stm32.get("weight_storage_mode", DEFAULT_WEIGHT_STORAGE_MODE)),
        "weights_memory_pool": Path(stm32.get("weights_memory_pool", DEFAULT_WEIGHTS_MEMORY_POOL)),
        "wake_margin_us": int(stm32.get("wake_margin_us", 5000)),
        "min_sleep_us": int(stm32.get("min_sleep_us", 5000)),
        "cpu_clock_mhz_options": tuple(device.get("cpu_clock_mhz_options", VALID_CPU_CLOCK_MHZ)),
    }


def resolve_workspace_paths(project_root: Path | str) -> WorkspacePaths:
    root = Path(project_root).expanduser().resolve()
    fsbl_source_root = root / "FSBL"
    appli_source_root = root / "Appli"
    boot_project_root = root / "STM32CubeIDE" / "Boot"
    app_project_root = root / "STM32CubeIDE" / "AppS"
    for candidate in (fsbl_source_root, appli_source_root, boot_project_root, app_project_root):
        if not candidate.is_dir():
            raise stm32_cube_clt.WorkflowError(f"Missing LRUN workspace path: {candidate}")
    fsbl_debug_dir = stm32_cube_clt.validate_project_root(boot_project_root) / "Debug"
    app_debug_dir = stm32_cube_clt.validate_project_root(app_project_root) / "Debug"
    return WorkspacePaths(
        root=root,
        fsbl_source_root=fsbl_source_root,
        appli_source_root=appli_source_root,
        boot_project_root=boot_project_root,
        app_project_root=app_project_root,
        fsbl_debug_dir=fsbl_debug_dir,
        app_debug_dir=app_debug_dir,
    )


def _resolve_bin_path(elf_path: Path) -> Path:
    candidate = elf_path.with_suffix(".bin")
    if candidate.is_file():
        return candidate
    siblings = sorted(elf_path.parent.glob("*.bin"))
    if len(siblings) == 1:
        return siblings[0]
    if siblings:
        raise stm32_cube_clt.WorkflowError(
            f"Could not determine matching BIN artifact for {elf_path}; found: "
            f"{', '.join(path.name for path in siblings)}"
        )
    raise stm32_cube_clt.WorkflowError(f"Missing BIN artifact after build: {candidate}")


def align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    return ((int(value) + alignment - 1) // alignment) * alignment


def build_workspace(
    project_root: Path | str,
    *,
    clean: bool,
    jobs: int | None,
) -> BuildArtifacts:
    paths = resolve_workspace_paths(project_root)
    fsbl_build = stm32_cube_clt.build_project(
        project_root=paths.boot_project_root,
        jobs=jobs,
        clean=clean,
    )
    app_build = stm32_cube_clt.build_project(
        project_root=paths.app_project_root,
        jobs=jobs,
        clean=clean,
    )
    fsbl_bin = _resolve_bin_path(fsbl_build.elf_path)
    app_bin = _resolve_bin_path(app_build.elf_path)
    return BuildArtifacts(
        fsbl_build=fsbl_build,
        app_build=app_build,
        fsbl_size=stm32_cube_clt.parse_size_output(fsbl_build.elf_path),
        app_size=stm32_cube_clt.parse_size_output(app_build.elf_path),
        fsbl_bin=fsbl_bin,
        app_bin=app_bin,
    )


def resolve_stedgeai_cli() -> str:
    for candidate in ("stedgeai", "st-edge-ai"):
        if shutil.which(candidate):
            return candidate
    raise stm32_cube_clt.WorkflowError(
        "Could not find ST Edge AI CLI. Expected `stedgeai` or `st-edge-ai` on PATH."
    )


def stage_generated_outputs(out_dir: Path, project_root: Path | str) -> list[Path]:
    paths = resolve_workspace_paths(project_root)
    staged: list[Path] = []
    src_dir = paths.appli_source_root / "Src"
    inc_dir = paths.appli_source_root / "Inc"
    for name in EXPECTED_OUTPUTS:
        destination_dir = src_dir if name.endswith(".c") else inc_dir
        destination = destination_dir / name
        shutil.copy2(out_dir / name, destination)
        staged.append(destination)
    return staged


def verify_generate_outputs(out_dir: Path, *, weight_storage_mode: str) -> Path | None:
    missing = [name for name in EXPECTED_OUTPUTS if not (out_dir / name).is_file()]
    if missing:
        raise stm32_cube_clt.WorkflowError(f"Missing generated outputs: {', '.join(missing)}")
    if weight_storage_mode != "external_flash":
        return None
    weights_blob_path = out_dir / WEIGHTS_BLOB_NAME
    if not weights_blob_path.is_file():
        raise stm32_cube_clt.WorkflowError(f"Missing generated weights blob: {weights_blob_path}")
    return weights_blob_path


def write_staging_manifest(
    output_root: Path,
    *,
    weight_storage_mode: str,
    generated_output_dir: Path,
    weights_blob_path: Path | None,
    weights_flash_address: str | None,
    weights_external_loader: Path | None,
) -> Path:
    manifest = {
        "weight_storage_mode": weight_storage_mode,
        "generated_output_dir": str(generated_output_dir),
        "weights_blob_path": str(weights_blob_path) if weights_blob_path is not None else None,
        "weights_blob_size": weights_blob_path.stat().st_size if weights_blob_path is not None else None,
        "weights_flash_address": weights_flash_address,
        "weights_external_loader": (
            str(weights_external_loader.resolve()) if weights_external_loader is not None else None
        ),
    }
    manifest_path = output_root / STAGING_MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def read_staging_manifest(output_root: Path | str) -> dict[str, Any]:
    manifest_path = Path(output_root).expanduser().resolve() / STAGING_MANIFEST_NAME
    if not manifest_path.is_file():
        raise stm32_cube_clt.WorkflowError(f"Missing staging manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def update_fsbl_lrun_source_size(
    project_root: Path | str,
    *,
    trusted_app_size: int,
    alignment: int = EXTMEM_COPY_WINDOW_ALIGN,
) -> tuple[Path, bool, int]:
    paths = resolve_workspace_paths(project_root)
    header_path = paths.fsbl_source_root / "Inc" / "stm32_extmem_conf.h"
    if not header_path.is_file():
        raise stm32_cube_clt.WorkflowError(f"Missing FSBL extmem config: {header_path}")

    aligned_size = align_up(int(trusted_app_size), int(alignment))
    source_define = re.compile(r"^(#define\s+EXTMEM_LRUN_SOURCE_SIZE\s+)0x[0-9A-Fa-f]+\s*$", re.MULTILINE)
    original_text = header_path.read_text(encoding="utf-8")
    replacement = rf"\g<1>0x{aligned_size:08X}"
    updated_text, replacements = source_define.subn(replacement, original_text, count=1)
    if replacements != 1:
        raise stm32_cube_clt.WorkflowError(
            f"Could not update EXTMEM_LRUN_SOURCE_SIZE in {header_path}"
        )
    changed = updated_text != original_text
    if changed:
        header_path.write_text(updated_text, encoding="utf-8")
    return header_path, changed, aligned_size


def write_phase_config_header(
    project_root: Path | str,
    *,
    phase: str,
    latency_budget_ms: float,
    measured_runs: int,
    cpu_clock_mhz: int,
    wake_margin_us: int,
    min_sleep_us: int,
    stub_only: bool,
) -> tuple[Path, bool]:
    paths = resolve_workspace_paths(project_root)
    phase_value = "TOY_AI_PHASE_CADENCED" if phase == "cadenced" else "TOY_AI_PHASE_BACK_TO_BACK"
    latency_budget_ms = max(1.0, float(latency_budget_ms))
    measured_runs = max(1, int(measured_runs))
    header_text = (
        "#ifndef TOY_AI_PHASE_CONFIG_H\n"
        "#define TOY_AI_PHASE_CONFIG_H\n\n"
        "#define TOY_AI_PHASE_BACK_TO_BACK 0\n"
        "#define TOY_AI_PHASE_CADENCED 1\n\n"
        f"#define TOY_AI_SELECTED_PHASE {phase_value}\n"
        f"#define TOY_AI_LATENCY_BUDGET_MS {int(round(latency_budget_ms))}\n"
        f"#define TOY_AI_MEASURED_RUNS {int(measured_runs)}\n"
        f"#define TOY_AI_CPU_CLOCK_MHZ {int(cpu_clock_mhz)}\n"
        f"#define TOY_AI_WAKE_MARGIN_US {int(wake_margin_us)}\n"
        f"#define TOY_AI_MIN_SLEEP_US {int(min_sleep_us)}\n"
        f"#define TOY_AI_USE_STUB {1 if stub_only else 0}\n\n"
        "#endif /* TOY_AI_PHASE_CONFIG_H */\n"
    )
    header_path = paths.appli_source_root / "Inc" / "toy_ai_phase_config.h"
    changed = True
    if header_path.is_file() and header_path.read_text(encoding="utf-8") == header_text:
        changed = False
    else:
        header_path.write_text(header_text, encoding="utf-8")
    return header_path, changed


def resolve_weights_external_loader(
    cubeprog_bin: Path | str | None,
    override: Path | str | None = None,
) -> Path:
    if override is not None:
        return stm32_cube_clt.resolve_required_file_path(override, label="STM32 external loader")
    cubeprog_dir = (
        stm32_cube_clt.default_cubeprog_bin()
        if cubeprog_bin is None
        else Path(cubeprog_bin).expanduser().resolve()
    )
    if cubeprog_dir is None:
        raise stm32_cube_clt.WorkflowError("STM32CubeProgrammer bin directory was not provided.")
    candidates = (
        cubeprog_dir / "ExternalLoader" / DEFAULT_EXTERNAL_LOADER_NAME,
        cubeprog_dir.parent / "ExternalLoader" / DEFAULT_EXTERNAL_LOADER_NAME,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise stm32_cube_clt.WorkflowError(
        f"Could not find external loader {DEFAULT_EXTERNAL_LOADER_NAME} under {cubeprog_dir}."
    )


def _run_checked(argv: list[str], *, cwd: Path | None = None) -> str:
    try:
        proc = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise stm32_cube_clt.WorkflowError(
            f"Failed to execute command: {' '.join(argv)}\n\n{exc}"
        ) from exc
    output = f"{proc.stdout}{proc.stderr}"
    if proc.returncode != 0:
        raise stm32_cube_clt.WorkflowError(
            f"Command failed with exit code {proc.returncode}: {' '.join(argv)}\n\n"
            f"{output.strip() or 'No subprocess output captured.'}"
        )
    return output


def resolve_signing_tool(path: Path | str | None = None) -> Path:
    if path is not None:
        return stm32_cube_clt.resolve_required_tool_path(
            path,
            label="STM32 signing tool",
            hint="STM32_SigningTool_CLI",
        )
    for hint in ("STM32_SigningTool_CLI", "STM32TrustedPackageCreator_CLI"):
        candidate = stm32_cube_clt.which_path(hint)
        if candidate is not None:
            return candidate
    raise stm32_cube_clt.WorkflowError(
        "STM32 signing tool was not provided.\n"
        "Install STM32CubeCLT and ensure `STM32_SigningTool_CLI` or "
        "`STM32TrustedPackageCreator_CLI` is on PATH."
    )


def sign_binary(
    *,
    signing_tool: Path | str | None,
    input_bin: Path | str,
    output_bin: Path | str,
    load_offset: str = DEFAULT_SIGNING_LOAD_OFFSET,
    header_version: str = DEFAULT_SIGNING_HEADER_VERSION,
) -> str:
    resolved_tool = resolve_signing_tool(signing_tool)
    resolved_input = stm32_cube_clt.resolve_required_file_path(input_bin, label="STM32 unsigned binary")
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
            return _run_checked(base_cmd + ["-align"])
        except stm32_cube_clt.WorkflowError:
            return _run_checked(base_cmd)
    except stm32_cube_clt.WorkflowError as exc:
        raise SigningError(str(exc)) from exc


def _resolve_programmer(cubeprog_bin: Path | str | None) -> Path:
    cubeprog_dir = (
        stm32_cube_clt.default_cubeprog_bin()
        if cubeprog_bin is None
        else Path(cubeprog_bin).expanduser().resolve()
    )
    if cubeprog_dir is None:
        raise stm32_cube_clt.WorkflowError("STM32CubeProgrammer bin directory was not provided.")
    return stm32_cube_clt.resolve_required_tool_path(
        Path(cubeprog_dir) / "STM32_Programmer_CLI",
        label="STM32_Programmer_CLI",
        hint="STM32_Programmer_CLI",
    )


def flash_external_image(
    *,
    cubeprog_bin: Path | str | None,
    apid: int,
    image_path: Path | str,
    flash_address: str,
    external_loader: Path | str,
    programmer_mode: str = DEFAULT_PROGRAMMER_MODE,
    freq_khz: str = DEFAULT_PROGRAMMER_FREQ_KHZ,
) -> str:
    programmer = _resolve_programmer(cubeprog_bin)
    resolved_image = stm32_cube_clt.resolve_required_file_path(image_path, label="STM32 external image")
    resolved_loader = stm32_cube_clt.resolve_required_file_path(
        external_loader,
        label="STM32 external loader",
    )
    recovery_cmd = [
        str(programmer),
        "-q",
        "-c",
        "port=SWD",
        f"mode={DEFAULT_RECOVERY_MODE}",
        f"freq={freq_khz}",
        f"ap={DEFAULT_RECOVERY_APID}",
    ]
    cmd = [
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
    recovery_log = _run_checked(recovery_cmd)
    time.sleep(1.0)
    program_log = _run_checked(cmd)
    return "\n".join(text for text in (recovery_log.strip(), program_log.strip()) if text)


def reset_target(
    *,
    cubeprog_bin: Path | str | None,
    apid: int,
    programmer_mode: str = DEFAULT_PROGRAMMER_MODE,
    freq_khz: str = DEFAULT_PROGRAMMER_FREQ_KHZ,
) -> str:
    programmer = _resolve_programmer(cubeprog_bin)
    cmd = [
        str(programmer),
        "-q",
        "-c",
        "port=SWD",
        f"mode={programmer_mode}",
        f"freq={freq_khz}",
        f"ap={int(apid)}",
        "-hardRst",
    ]
    return _run_checked(cmd)


def validate_flash_layout(
    *,
    appli_trusted_size: int,
    weights_blob_size: int,
    appli_flash_address: str,
    weights_flash_address: str,
) -> None:
    appli_addr = int(appli_flash_address, 16)
    weights_addr = int(weights_flash_address, 16)
    if not (appli_addr < weights_addr):
        raise stm32_cube_clt.WorkflowError(
            "Flash layout must satisfy Appli < weights addresses."
        )
    if appli_addr + appli_trusted_size > weights_addr:
        raise stm32_cube_clt.WorkflowError("Trusted Appli image overlaps the weights region.")
