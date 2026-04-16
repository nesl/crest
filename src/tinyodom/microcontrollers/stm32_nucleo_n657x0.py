from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import serial

from ..devices import (
    CompileResult,
    DeviceInterface,
    DeviceMetrics,
    DeviceSpec,
    MeasureResult,
    UploadResult,
)
from ..errors import (
    HIL_ERROR_COMPILE,
    HIL_ERROR_FLASH_OVERFLOW,
    HIL_ERROR_LATENCY,
    HIL_ERROR_OK,
    HIL_ERROR_RAM_OVERFLOW,
    HIL_ERROR_UPLOAD,
)
from .. import hil_protocol
from . import arduino_base
from . import stm32_cube_clt
from . import stm32_runtime


# Public device identifier used throughout config parsing and backend selection.
BOARD_NAME = "STM32_NUCLEO_N657X0_Q"
# Repository root used to resolve checked-in STM32 templates and helper assets.
REPO_ROOT = Path(__file__).resolve().parents[3]
# Canonical FSBL template copied into a per-candidate staging directory.
DEFAULT_TEMPLATE_ROOT = REPO_ROOT / "sketches" / "stm32" / "tinyodom_tcn_stm32" / "FSBL"
# ST STM32N657X0 product docs describe the part as having 4.2-Mbyte contiguous
# SRAM.
DEFAULT_MAX_RAM_BYTES = 4_194_304
# FIXME: legacy 8 MiB ceiling from the first STM backend bring-up; this does
# not match current ST N657X0 docs or our staged FSBL linker script, so treat
# it as unverified until a real deployment flash budget is chosen.
DEFAULT_MAX_FLASH_BYTES = 8_388_608
# ST UM3417 section 7.10 says the NUCLEO-N657X0-Q carries 512-Mbit Octo-SPI
# flash, which is 64 MiB usable capacity.
DEFAULT_MAX_EXTERNAL_FLASH_BYTES = 67_108_864
# Default fixed CPU clock preset written into the generated STM runtime config.
DEFAULT_CPU_CLOCK_MHZ = 600
# Weight placement mode used when callers do not override STM storage policy.
DEFAULT_WEIGHT_STORAGE_MODE = "embedded"
# Matches ST Edge AI STM32N6 CM55-validation examples, which generate external
# weights with `--address 0x71000000`; this is a workflow default, not a
# silicon-defined constant.
DEFAULT_WEIGHTS_FLASH_ADDRESS = "0x71000000"
# Default ST Edge AI memory-pool description for externalized weights.
DEFAULT_WEIGHTS_MEMORY_POOL = (
    REPO_ROOT / "analysis_scripts" / "stm32_example_project" / "nucleo_mypool.json"
)
# Supported fixed clock presets accepted by the STM staging pipeline.
SUPPORTED_CPU_CLOCK_MHZ = frozenset({200, 300, 400, 600, 800})
# Matches the checked-in FSBL linker script's `_Min_Heap_Size` reservation.
MIN_HEAP_BYTES = 0x2000
# Matches the checked-in FSBL linker script's `_Min_Stack_Size` reservation.
MIN_STACK_BYTES = 0x4000
# Generated ST Edge AI sources/headers that must exist after codegen.
EXPECTED_GENERATED_OUTPUTS = (
    "network.c",
    "network.h",
    "network_config.h",
    "network_data.c",
    "network_data.h",
    "network_data_params.c",
    "network_data_params.h",
)
# Filename emitted for externalized weight blobs.
WEIGHTS_BLOB_NAME = "network_data.bin"
# Default STM32CubeProgrammer external loader for the Nucleo flash device.
DEFAULT_WEIGHTS_EXTERNAL_LOADER_NAME = "MX25UM51245G_STM32N6570-NUCLEO.stldr"
# Manifest persisted in staged project roots to carry STM backend metadata.
STAGED_MANIFEST_NAME = "tinyodom_stm32_manifest.json"
# Build-recipe files expected at the staged project root after makefile generation.
DEBUG_RECIPE_ROOT_FILENAMES = frozenset(
    {
        "makefile",
        "sources.mk",
        "objects.mk",
        "objects.list",
        "stedgeai.mk",
    }
)
# Accept both the checked-in ST Edge AI header shape ``(47792)`` and the
# simpler bare integer form used by lightweight test doubles.
ARENA_RE = re.compile(
    r"#define\s+AI_NETWORK_DATA_ACTIVATIONS_SIZE\s+\(?(?P<value>\d+)\)?"
)
# Accept the real linker-script assignment syntax as well as the temporary
# preprocessor-style fixtures that older tests used.
HEX_DEFINE_RE = re.compile(
    r"(?m)(?:#define\s+)?(?P<name>_Min_(?:Heap|Stack)_Size)"
    r"(?:\s+\(\(size_t\)\))?\s*(?:=)?\s*0x(?P<value>[0-9A-Fa-f]+)\s*;?"
)
MEASURED_RUNS_RE = re.compile(r"(?m)^#define\s+TCN_DUT_MEASURED_RUNS\s+(?P<value>\d+)\s*$")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class STM32NucleoN657X0QOptions:
    """Normalized STM32 backend options.

    Parameters
    ----------
    project_root : pathlib.Path
        Canonical STM32 FSBL template root. The backend copies this template
        into a per-candidate staging directory before build/upload.
    gdbserver : pathlib.Path | None
        Optional explicit path to ``ST-LINK_gdbserver``.
    gdb : pathlib.Path | None
        Optional explicit path to ``arm-none-eabi-gdb``.
    cubeprog_bin : pathlib.Path | None
        Optional explicit path to the STM32CubeProgrammer ``bin`` directory.
    gdb_port : int
        TCP port used by the ST-LINK GDB server.
    apid : int
        ST-LINK access port identifier.
    server_ready_timeout_s : float
        Timeout waiting for the GDB server to report that it is ready.
    cpu_clock_mhz : int
        Requested fixed CPU clock preset written into the generated phase
        configuration header.
    weight_storage_mode : str
        Weight placement policy (`embedded` or `external_flash`).
    weights_flash_address : str
        Absolute external flash address used when externalizing weights.
    weights_memory_pool : pathlib.Path
        ST Edge AI memory-pool JSON used when externalizing weights.
    weights_external_loader : pathlib.Path | None
        Required `.stldr` loader path used to program external weight blobs.
    max_external_flash_bytes : int
        Maximum usable external flash bytes for staged weight blobs.
    """

    project_root: Path
    gdbserver: Path | None
    gdb: Path | None
    cubeprog_bin: Path | None
    gdb_port: int
    apid: int
    server_ready_timeout_s: float
    cpu_clock_mhz: int
    weight_storage_mode: str
    weights_flash_address: str
    weights_memory_pool: Path
    weights_external_loader: Path | None
    max_external_flash_bytes: int


def _resolve_optional_path(value: object | None) -> Path | None:
    """Normalize an optional filesystem path.

    Parameters
    ----------
    value : object | None
        Raw config or runtime value that may contain a path-like string.

    Returns
    -------
    pathlib.Path | None
        Resolved path when a non-empty value is provided, otherwise ``None``.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text).expanduser().resolve()


def _coerce_int_with_default(value: object | None, default: int) -> int:
    """Return ``default`` for missing values, otherwise coerce to ``int``.

    Parameters
    ----------
    value : object | None
        Raw numeric value from config or request plumbing.
    default : int
        Fallback used when ``value`` is omitted.

    Returns
    -------
    int
        Normalized integer value.
    """
    if value in (None, ""):
        return int(default)
    return int(value)


def _coerce_float_with_default(value: object | None, default: float) -> float:
    """Return ``default`` for missing values, otherwise coerce to ``float``.

    Parameters
    ----------
    value : object | None
        Raw numeric value from config or request plumbing.
    default : float
        Fallback used when ``value`` is omitted.

    Returns
    -------
    float
        Normalized floating-point value.
    """
    if value in (None, ""):
        return float(default)
    return float(value)


def _resolve_cpu_clock_mhz(raw_value: object | None) -> int:
    """Validate and normalize the requested STM CPU clock preset.

    Parameters
    ----------
    raw_value : object | None
        Raw clock preset value from config or runtime options.

    Returns
    -------
    int
        Supported CPU clock preset in MHz.

    Raises
    ------
    ValueError
        If the requested preset is not one of the supported STM clock values.
    """
    clock_mhz = _coerce_int_with_default(raw_value, DEFAULT_CPU_CLOCK_MHZ)
    if clock_mhz not in SUPPORTED_CPU_CLOCK_MHZ:
        allowed = ", ".join(str(value) for value in sorted(SUPPORTED_CPU_CLOCK_MHZ))
        raise ValueError(f"Unsupported STM CPU clock preset {clock_mhz}. Expected one of: {allowed}.")
    return clock_mhz


def _resolve_weight_storage_mode(raw_value: object | None) -> str:
    """Validate and normalize the requested STM weight storage mode.

    Parameters
    ----------
    raw_value : object | None
        Raw storage-mode value from config or runtime options.

    Returns
    -------
    str
        Normalized storage mode.
    """
    mode = str(raw_value or DEFAULT_WEIGHT_STORAGE_MODE).strip().lower()
    if mode not in {"embedded", "external_flash"}:
        raise ValueError(
            "Unsupported STM weight storage mode "
            f"{mode!r}. Expected 'embedded' or 'external_flash'."
        )
    return mode


def _resolve_cubeprog_cli_path(cubeprog_bin: Path | None) -> Path:
    """Resolve the STM32CubeProgrammer CLI executable path.

    Parameters
    ----------
    cubeprog_bin : pathlib.Path | None
        Optional configured STM32CubeProgrammer ``bin`` directory.

    Returns
    -------
    pathlib.Path
        Resolved ``STM32_Programmer_CLI`` executable path.
    """
    if cubeprog_bin is None:
        return stm32_cube_clt.resolve_required_tool_path(
            None,
            label="STM32_Programmer_CLI",
            hint="STM32_Programmer_CLI",
        )
    return stm32_cube_clt.resolve_required_tool_path(
        cubeprog_bin / "STM32_Programmer_CLI",
        label="STM32_Programmer_CLI",
        hint="STM32_Programmer_CLI",
    )


def resolve_stm32_nucleo_n657x0_q_options(
    device_options: Optional[Mapping[str, object]],
) -> STM32NucleoN657X0QOptions:
    """Resolve STM32 backend options from config/runtime input.

    Parameters
    ----------
    device_options : Mapping[str, object] | None
        Raw STM32 backend options supplied by config or request plumbing.

    Returns
    -------
    STM32NucleoN657X0QOptions
        Normalized options with defaults applied.
    """
    options = dict(device_options or {})
    project_root = _resolve_optional_path(options.get("template_root"))
    if project_root is None:
        project_root = _resolve_optional_path(options.get("project_root"))
    if project_root is None:
        project_root = DEFAULT_TEMPLATE_ROOT
    gdbserver = _resolve_optional_path(options.get("gdbserver"))
    gdb = _resolve_optional_path(options.get("gdb"))
    cubeprog_bin = _resolve_optional_path(options.get("cubeprog_bin"))
    gdb_port = _coerce_int_with_default(options.get("gdb_port"), stm32_cube_clt.DEFAULT_GDB_PORT)
    apid = _coerce_int_with_default(options.get("apid"), stm32_cube_clt.DEFAULT_APID)
    server_ready_timeout_s = _coerce_float_with_default(
        options.get("server_ready_timeout_s"),
        stm32_cube_clt.SERVER_READY_TIMEOUT_S,
    )
    cpu_clock_mhz = _resolve_cpu_clock_mhz(options.get("cpu_clock_mhz"))
    weight_storage_mode = _resolve_weight_storage_mode(options.get("weight_storage_mode"))
    weights_flash_address = str(
        options.get("weights_flash_address", DEFAULT_WEIGHTS_FLASH_ADDRESS)
    ).strip()
    weights_memory_pool = (
        _resolve_optional_path(options.get("weights_memory_pool")) or DEFAULT_WEIGHTS_MEMORY_POOL
    )
    weights_external_loader = _resolve_optional_path(options.get("weights_external_loader"))
    max_external_flash_bytes = _coerce_int_with_default(
        options.get("max_external_flash_bytes"),
        DEFAULT_MAX_EXTERNAL_FLASH_BYTES,
    )
    return STM32NucleoN657X0QOptions(
        project_root=project_root,
        gdbserver=gdbserver,
        gdb=gdb,
        cubeprog_bin=cubeprog_bin,
        gdb_port=gdb_port,
        apid=apid,
        server_ready_timeout_s=server_ready_timeout_s,
        cpu_clock_mhz=cpu_clock_mhz,
        weight_storage_mode=weight_storage_mode,
        weights_flash_address=weights_flash_address,
        weights_memory_pool=Path(weights_memory_pool).expanduser().resolve(),
        weights_external_loader=weights_external_loader,
        max_external_flash_bytes=max_external_flash_bytes,
    )


def build_stm32_nucleo_n657x0_q_spec(
    options: STM32NucleoN657X0QOptions | None = None,
) -> DeviceSpec:
    """Build the STM32 ``DeviceSpec`` exposed through the device registry.

    Parameters
    ----------
    options : STM32NucleoN657X0QOptions | None, optional
        Optional resolved board options that may override external-flash
        capacity metadata.

    Returns
    -------
    DeviceSpec
        Static STM32 board metadata used by the controller.
    """
    return DeviceSpec(
        name=BOARD_NAME,
        arena_sizes_kb=[-1],
        max_ram_bytes=DEFAULT_MAX_RAM_BYTES,
        max_flash_bytes=DEFAULT_MAX_FLASH_BYTES,
        max_external_flash_bytes=(
            DEFAULT_MAX_EXTERNAL_FLASH_BYTES
            if options is None
            else int(options.max_external_flash_bytes)
        ),
        toolchain="stm32-cube",
    )


def _find_linker_script(project_root: Path) -> Path:
    """Locate the first linker script under an STM32 FSBL root.

    Parameters
    ----------
    project_root : pathlib.Path
        STM32 FSBL project root to inspect.

    Returns
    -------
    pathlib.Path
        First linker script found under ``project_root``.

    Raises
    ------
    tinyodom.microcontrollers.stm32_cube_clt.WorkflowError
        If no linker script exists in the project root.
    """
    linker_scripts = sorted(project_root.glob("*.ld"))
    if not linker_scripts:
        raise stm32_cube_clt.WorkflowError(f"Unable to locate linker script under {project_root}")
    return linker_scripts[0]


def _parse_linker_reservations(linker_script: Path) -> dict[str, int]:
    """Parse linker-reserved heap and stack bytes.

    Parameters
    ----------
    linker_script : pathlib.Path
        STM32 linker script containing ``_Min_Heap_Size`` and
        ``_Min_Stack_Size`` declarations.

    Returns
    -------
    dict[str, int]
        Parsed ``heap_bytes`` and ``stack_bytes`` values when present.
    """
    text = linker_script.read_text(encoding="utf-8")
    values: dict[str, int] = {}
    for match in HEX_DEFINE_RE.finditer(text):
        key = "heap_bytes" if "Heap" in match.group("name") else "stack_bytes"
        values[key] = int(match.group("value"), 16)
    return values


def _parse_arena_bytes(network_data_params: Path) -> int:
    """Parse the ST Edge AI activation arena size from a generated header.

    Parameters
    ----------
    network_data_params : pathlib.Path
        Generated ``network_data_params.h`` header emitted by ST Edge AI.

    Returns
    -------
    int
        Parsed activation arena size in bytes.

    Raises
    ------
    tinyodom.microcontrollers.stm32_cube_clt.WorkflowError
        If the header is missing or does not contain the expected macro.
    """
    if not network_data_params.is_file():
        raise stm32_cube_clt.WorkflowError(f"Missing generated header: {network_data_params}")
    text = network_data_params.read_text(encoding="utf-8")
    match = ARENA_RE.search(text)
    if not match:
        raise stm32_cube_clt.WorkflowError(
            f"Unable to parse AI_NETWORK_DATA_ACTIVATIONS_SIZE from {network_data_params}"
        )
    return int(match.group("value"))


def _write_phase_config_header(
    *,
    project_root: Path,
    cpu_clock_mhz: int,
    latency_budget_ms: float = 1.0,
    measured_runs: int = 1,
    wake_margin_us: int = 5000,
    min_sleep_us: int = 5000,
) -> Path:
    """Write the generated STM phase-config header.

    Parameters
    ----------
    project_root : pathlib.Path
        Staged STM32 FSBL root that owns ``Inc/tcn_dut_phase_config.h``.
    cpu_clock_mhz : int
        Requested fixed CPU clock preset.
    latency_budget_ms : float, default=1.0
        Placeholder cadence budget written for deferred cadenced-mode compatibility.
    measured_runs : int, default=1
        Placeholder measured-run count written for deferred cadenced-mode compatibility.
    wake_margin_us : int, default=5000
        Wake margin in microseconds.
    min_sleep_us : int, default=5000
        Minimum sleep duration in microseconds.

    Returns
    -------
    pathlib.Path
        Path to the generated phase-config header.
    """
    header_path = project_root / "Inc" / "tcn_dut_phase_config.h"
    header_text = (
        "#ifndef TCN_DUT_PHASE_CONFIG_H\n"
        "#define TCN_DUT_PHASE_CONFIG_H\n\n"
        "#define TCN_DUT_PHASE_BACK_TO_BACK 0\n"
        "#define TCN_DUT_PHASE_CADENCED 1\n\n"
        "#define TCN_DUT_SELECTED_PHASE TCN_DUT_PHASE_BACK_TO_BACK\n"
        f"#define TCN_DUT_LATENCY_BUDGET_MS {max(1, int(round(latency_budget_ms)))}\n"
        f"#define TCN_DUT_MEASURED_RUNS {max(1, int(measured_runs))}\n"
        f"#define TCN_DUT_CPU_CLOCK_MHZ {int(cpu_clock_mhz)}\n"
        f"#define TCN_DUT_WAKE_MARGIN_US {max(0, int(wake_margin_us))}\n"
        f"#define TCN_DUT_MIN_SLEEP_US {max(0, int(min_sleep_us))}\n\n"
        "#endif /* TCN_DUT_PHASE_CONFIG_H */\n"
    )
    header_path.write_text(header_text, encoding="utf-8")
    return header_path


def _validate_phase_config_header(header_path: Path, *, cpu_clock_mhz: int) -> None:
    """Confirm the generated phase-config header contains the expected values.

    Parameters
    ----------
    header_path : pathlib.Path
        Generated phase-config header path.
    cpu_clock_mhz : int
        Requested CPU clock preset expected in the generated header.

    Returns
    -------
    None

    Raises
    ------
    tinyodom.microcontrollers.stm32_cube_clt.WorkflowError
        If the header does not contain the required STM runtime settings.
    """
    text = header_path.read_text(encoding="utf-8")
    if "#define TCN_DUT_SELECTED_PHASE TCN_DUT_PHASE_BACK_TO_BACK" not in text:
        raise stm32_cube_clt.WorkflowError(
            f"Generated phase config did not select back_to_back mode: {header_path}"
        )
    expected_clock = f"#define TCN_DUT_CPU_CLOCK_MHZ {int(cpu_clock_mhz)}"
    if expected_clock not in text:
        raise stm32_cube_clt.WorkflowError(
            f"Generated phase config is missing requested clock preset {cpu_clock_mhz}: {header_path}"
        )


def _read_phase_config_measured_runs(project_root: Path) -> int:
    """Read the staged DUT measured-run count from the generated phase header.

    Parameters
    ----------
    project_root : pathlib.Path
        Staged STM32 FSBL root that owns ``Inc/tcn_dut_phase_config.h``.

    Returns
    -------
    int
        Positive DUT run count compiled into the staged project.

    Raises
    ------
    tinyodom.microcontrollers.stm32_cube_clt.WorkflowError
        If the generated phase-config header is missing or does not declare
        ``TCN_DUT_MEASURED_RUNS``.
    """
    header_path = project_root / "Inc" / "tcn_dut_phase_config.h"
    if not header_path.is_file():
        raise stm32_cube_clt.WorkflowError(f"Missing generated phase config header: {header_path}")
    header_text = header_path.read_text(encoding="utf-8")
    match = MEASURED_RUNS_RE.search(header_text)
    if not match:
        raise stm32_cube_clt.WorkflowError(
            f"Generated phase config is missing TCN_DUT_MEASURED_RUNS: {header_path}"
        )
    return max(1, int(match.group("value")))


def _stage_generated_outputs(generated_output_dir: Path, project_root: Path) -> None:
    """Copy generated ST Edge AI sources and headers into the staged FSBL tree.

    Parameters
    ----------
    generated_output_dir : pathlib.Path
        Directory containing generated ``network*.c`` and ``network*.h`` files.
    project_root : pathlib.Path
        Staged STM32 FSBL root that receives the generated sources.

    Returns
    -------
    None

    Raises
    ------
    tinyodom.microcontrollers.stm32_cube_clt.WorkflowError
        If any required generated output is missing.
    """
    src_dir = project_root / "Src"
    inc_dir = project_root / "Inc"
    src_dir.mkdir(parents=True, exist_ok=True)
    inc_dir.mkdir(parents=True, exist_ok=True)
    missing = [name for name in EXPECTED_GENERATED_OUTPUTS if not (generated_output_dir / name).is_file()]
    if missing:
        missing_text = ", ".join(missing)
        raise stm32_cube_clt.WorkflowError(f"Missing generated outputs: {missing_text}")
    for name in EXPECTED_GENERATED_OUTPUTS:
        destination_dir = src_dir if name.endswith(".c") else inc_dir
        shutil.copy2(generated_output_dir / name, destination_dir / name)


def _generated_weights_blob_path(
    generated_output_dir: Path,
    *,
    weight_storage_mode: str,
) -> Path | None:
    """Return the generated external-weight blob path when present.

    Parameters
    ----------
    generated_output_dir : pathlib.Path
        Directory containing ST Edge AI outputs.
    weight_storage_mode : str
        Weight placement policy for the staged candidate.

    Returns
    -------
    pathlib.Path | None
        Generated blob path in `external_flash` mode, otherwise `None`.
    """
    if weight_storage_mode != "external_flash":
        return None
    blob_path = generated_output_dir / WEIGHTS_BLOB_NAME
    if not blob_path.is_file():
        raise stm32_cube_clt.WorkflowError(f"Missing generated weights blob: {blob_path}")
    return blob_path


def _resolve_weights_external_loader(
    cubeprog_bin: Path | None,
    weights_external_loader: Path | None,
) -> Path:
    """Resolve the Nucleo N657 external-loader path.

    Parameters
    ----------
    cubeprog_bin : pathlib.Path | None
        Optional STM32CubeProgrammer ``bin`` directory from config.
    weights_external_loader : pathlib.Path | None
        Explicit override from config or CLI, when provided.

    Returns
    -------
    pathlib.Path
        Resolved `.stldr` path for the board's external NOR flash.

    Raises
    ------
    tinyodom.microcontrollers.stm32_cube_clt.WorkflowError
        If STM32CubeProgrammer is unavailable or the required loader cannot be
        found under the expected `ExternalLoader` directories.
    """
    if weights_external_loader is not None:
        return stm32_cube_clt.resolve_required_file_path(
            weights_external_loader,
            label="STM external loader",
        )
    cubeprog_dir = cubeprog_bin or stm32_cube_clt.default_cubeprog_bin()
    if cubeprog_dir is None:
        raise stm32_cube_clt.WorkflowError(
            "STM32CubeProgrammer not found; required for external_flash weight storage mode."
        )
    candidate_paths = (
        cubeprog_dir / "ExternalLoader" / DEFAULT_WEIGHTS_EXTERNAL_LOADER_NAME,
        cubeprog_dir.parent / "ExternalLoader" / DEFAULT_WEIGHTS_EXTERNAL_LOADER_NAME,
    )
    for loader_path in candidate_paths:
        if loader_path.is_file():
            return loader_path.resolve()
    candidate_text = "\n".join(f"  - {path}" for path in candidate_paths)
    raise stm32_cube_clt.WorkflowError(
        "Required Nucleo external loader was not found in the STM32CubeProgrammer install.\n"
        f"Searched:\n{candidate_text}\n"
        "Install STM32CubeProgrammer with ExternalLoader files, add "
        "`STM32_Programmer_CLI` to PATH, or configure device.stm32.weights_external_loader."
    )


def _manifest_path(project_root: Path) -> Path:
    """Return the fixed path of the staged STM manifest.

    Parameters
    ----------
    project_root : pathlib.Path
        Staged STM32 FSBL root.

    Returns
    -------
    pathlib.Path
        Sidecar manifest path stored inside the staged project root.
    """
    return project_root / STAGED_MANIFEST_NAME


def _write_staged_manifest(
    *,
    project_root: Path,
    candidate_root: Path,
    generated_output_dir: Path,
    weight_storage_mode: str,
    weights_blob_path: Path | None,
    weights_flash_address: str,
    weights_external_loader: Path | None,
) -> Path:
    """Write the staged STM manifest used across compile and evaluate.

    The manifest is valid only for the lifetime of one staged candidate root.
    Callers must keep the staged root intact until `evaluate()` finishes.

    Parameters
    ----------
    project_root : pathlib.Path
        Staged STM32 FSBL root that owns the manifest.
    candidate_root : pathlib.Path
        Per-candidate root directory that contains generated outputs.
    generated_output_dir : pathlib.Path
        ST Edge AI output directory for this candidate.
    weight_storage_mode : str
        Weight placement policy.
    weights_blob_path : pathlib.Path | None
        Generated external-weight blob path when present.
    weights_flash_address : str
        Absolute external flash address for the blob.
        weights_external_loader : pathlib.Path | None
        Optional explicit `.stldr` loader path from config. When omitted, the
        backend auto-discovers the board-specific loader under the detected
        STM32CubeProgrammer install.

    Returns
    -------
    pathlib.Path
        Written manifest path.
    """
    manifest = {
        "candidate_root": str(candidate_root.resolve()),
        "generated_output_dir": str(generated_output_dir.resolve()),
        "weight_storage_mode": str(weight_storage_mode),
        "weights_blob_path": str(weights_blob_path.resolve()) if weights_blob_path is not None else None,
        "weights_blob_size": weights_blob_path.stat().st_size if weights_blob_path is not None else None,
        "weights_flash_address": str(weights_flash_address),
        "weights_external_loader": (
            str(weights_external_loader.resolve()) if weights_external_loader is not None else None
        ),
    }
    manifest_path = _manifest_path(project_root)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def _read_staged_manifest(project_root: Path) -> dict[str, object]:
    """Read the staged STM manifest from a staged project root.

    The manifest is expected to survive only for one staged candidate
    lifecycle. Cleanup must not remove it before `evaluate()` finishes.

    Parameters
    ----------
    project_root : pathlib.Path
        Staged STM32 FSBL root containing the manifest.

    Returns
    -------
    dict[str, object]
        Parsed manifest payload.
    """
    manifest_path = _manifest_path(project_root)
    if not manifest_path.is_file():
        raise stm32_cube_clt.WorkflowError(f"STM staged manifest not found: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise stm32_cube_clt.WorkflowError(f"Unable to parse STM staged manifest: {manifest_path}") from exc
    if not isinstance(payload, dict):
        raise stm32_cube_clt.WorkflowError(f"Unexpected STM staged manifest payload: {manifest_path}")
    return payload


def _validate_project_structure(project_root: Path) -> Path:
    """Validate the staged STM32 FSBL project shape and required headers.

    Parameters
    ----------
    project_root : pathlib.Path
        STM32 FSBL project root to validate.

    Returns
    -------
    pathlib.Path
        Resolved and validated project root.

    Raises
    ------
    tinyodom.microcontrollers.stm32_cube_clt.WorkflowError
        If the project root is missing required directories, headers, or the
        generated CubeIDE makefile.
    """
    resolved = stm32_cube_clt.validate_project_root(project_root)
    for required_dir in ("Src", "Inc"):
        candidate = resolved / required_dir
        if not candidate.is_dir():
            raise stm32_cube_clt.WorkflowError(f"STM32 project is missing required directory: {candidate}")
    for required_header in ("stm32n6xx_hal_conf.h", "stm32n6xx_nucleo_conf.h", "tcn_dut_runner.h"):
        candidate = resolved / "Inc" / required_header
        if not candidate.is_file():
            raise stm32_cube_clt.WorkflowError(f"STM32 project is missing required CubeN6 header: {candidate}")
    _find_linker_script(resolved)
    return resolved


def _validate_memory_reservations(project_root: Path) -> tuple[int, int]:
    """Validate linker-reserved heap and stack sizes against minimum floors.

    Parameters
    ----------
    project_root : pathlib.Path
        STM32 FSBL project root containing the linker script.

    Returns
    -------
    tuple[int, int]
        Parsed ``(heap_bytes, stack_bytes)`` tuple.

    Raises
    ------
    tinyodom.microcontrollers.stm32_cube_clt.WorkflowError
        If the reservations cannot be parsed or fall below the accepted floor.
    """
    linker_script = _find_linker_script(project_root)
    reservations = _parse_linker_reservations(linker_script)
    heap_bytes = reservations.get("heap_bytes")
    stack_bytes = reservations.get("stack_bytes")
    if heap_bytes is None or stack_bytes is None:
        raise stm32_cube_clt.WorkflowError(
            f"Unable to parse linker heap/stack reservations from {linker_script}"
        )
    if heap_bytes < MIN_HEAP_BYTES:
        raise stm32_cube_clt.WorkflowError(
            f"STM32 linker heap reservation is too small ({heap_bytes} bytes < {MIN_HEAP_BYTES})."
        )
    if stack_bytes < MIN_STACK_BYTES:
        raise stm32_cube_clt.WorkflowError(
            f"STM32 linker stack reservation is too small ({stack_bytes} bytes < {MIN_STACK_BYTES})."
        )
    return heap_bytes, stack_bytes


def _strip_stale_debug_artifacts(project_root: Path) -> None:
    """Remove copied STM build outputs while preserving committed recipes.

    Parameters
    ----------
    project_root : pathlib.Path
        STM32 FSBL project root whose ``Debug`` directory should be scrubbed.
    """
    debug_dir = project_root / "Debug"
    if not debug_dir.is_dir():
        return

    for child in debug_dir.iterdir():
        if child.name in {"Src", "Startup"}:
            for nested in child.iterdir():
                if nested.name == "subdir.mk":
                    continue
                if nested.is_dir():
                    shutil.rmtree(nested)
                else:
                    nested.unlink()
            continue
        if child.name in DEBUG_RECIPE_ROOT_FILENAMES:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _ensure_staging_tools() -> None:
    """Fail early when STM staging tools are unavailable.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    stm32_cube_clt.resolve_required_tool_path(
        None,
        label="stedgeai",
        hint="stedgeai",
    )


def _run_stedgeai_analyze(
    *,
    model_path: Path,
    workspace_dir: Path,
    output_dir: Path,
) -> None:
    """Run ST Edge AI analyze as a cheap compatibility preflight.

    Parameters
    ----------
    model_path : pathlib.Path
        Candidate-specific exported TFLite model.
    workspace_dir : pathlib.Path
        Per-candidate ST Edge AI workspace directory.
    output_dir : pathlib.Path
        Per-candidate output directory receiving analyze artifacts.
    """
    stedgeai = stm32_cube_clt.resolve_required_tool_path(
        None,
        label="stedgeai",
        hint="stedgeai",
    )
    workspace_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(stedgeai),
        "analyze",
        "-m",
        str(model_path),
        "-t",
        "tflite",
        "--target",
        "stm32n6",
        "--quiet",
        "--workspace",
        str(workspace_dir),
        "--output",
        str(output_dir),
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise stm32_cube_clt.WorkflowError(
            f"Failed to execute ST Edge AI analyze command: {' '.join(cmd)}\n\n{exc}"
        ) from exc
    output = f"{proc.stdout}{proc.stderr}".strip()
    if proc.returncode != 0:
        detail = output if output else "No ST Edge AI output captured."
        raise stm32_cube_clt.WorkflowError(
            f"ST Edge AI analyze failed with exit code {proc.returncode}.\n\n{detail}"
        )


def _run_stedgeai_generate(
    *,
    model_path: Path,
    workspace_dir: Path,
    output_dir: Path,
    weight_storage_mode: str,
    weights_flash_address: str,
    weights_memory_pool: Path,
) -> None:
    """Run ST Edge AI code generation for one staged candidate model.

    Parameters
    ----------
    model_path : pathlib.Path
        Candidate-specific exported TFLite model.
    workspace_dir : pathlib.Path
        Per-candidate ST Edge AI workspace directory.
    output_dir : pathlib.Path
        Per-candidate output directory receiving generated network files.
    weight_storage_mode : str
        Weight placement policy (`embedded` or `external_flash`).
    weights_flash_address : str
        Absolute external flash address used when externalizing weights.
    weights_memory_pool : pathlib.Path
        Memory-pool JSON used when externalizing weights.

    Returns
    -------
    None

    Raises
    ------
    tinyodom.microcontrollers.stm32_cube_clt.WorkflowError
        If the ST Edge AI CLI cannot be started or generation fails.
    """
    stedgeai = stm32_cube_clt.resolve_required_tool_path(
        None,
        label="stedgeai",
        hint="stedgeai",
    )
    workspace_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(stedgeai),
        "generate",
        "-m",
        str(model_path),
        "-t",
        "tflite",
        "--target",
        "stm32n6",
        "--c-api",
        "legacy",
    ]
    if weight_storage_mode == "external_flash":
        cmd.extend(
            [
                "--binary",
                "--address",
                str(weights_flash_address),
                "--memory-pool",
                str(weights_memory_pool),
            ]
        )
    cmd.extend(
        [
            "--quiet",
            "--workspace",
            str(workspace_dir),
            "--output",
            str(output_dir),
        ]
    )
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise stm32_cube_clt.WorkflowError(
            f"Failed to execute ST Edge AI generation command: {' '.join(cmd)}\n\n{exc}"
        ) from exc
    output = f"{proc.stdout}{proc.stderr}".strip()
    if proc.returncode != 0:
        detail = output if output else "No ST Edge AI output captured."
        raise stm32_cube_clt.WorkflowError(
            f"ST Edge AI generation failed with exit code {proc.returncode}.\n\n{detail}"
        )


def classify_stm32_backend_error(detail: str) -> str:
    """Return a stable STM backend error kind for a diagnostic message.

    Parameters
    ----------
    detail : str
        Diagnostic text from a staging, toolchain, upload, or runtime failure.

    Returns
    -------
    str
        Stable STM-specific backend error kind.
    """
    lowered = str(detail).lower()
    if "st edge ai" in lowered or "stedgeai" in lowered:
        unsupported_markers = (
            "unsupported",
            "not supported",
            "not yet supported",
            "operator",
            "op type",
            "layer",
            "incompatible",
            "not compatible",
        )
        if any(marker in lowered for marker in unsupported_markers):
            return "unsupported_model"
        return "codegen"
    if "external loader" in lowered or ".stldr" in lowered:
        return "external_flash_loader"
    if "weights blob" in lowered or "external weight" in lowered or "network_data.bin" in lowered:
        if "overflow" in lowered or "too large" in lowered:
            return "external_flash_overflow"
        return "external_flash_programming"
    if "harness ready" in lowered:
        return "harness_ready_timeout"
    if "harness done" in lowered:
        return "harness_done_timeout"
    if "run-count mismatch" in lowered or "run count mismatch" in lowered:
        return "harness_run_count_mismatch"
    if "gdb" in lowered or "st-link" in lowered or "cubeprog" in lowered:
        return "upload"
    if "dut ready" in lowered or "stm32_ai_" in lowered or "runtime" in lowered:
        return "runtime_protocol"
    return "toolchain"


def _manifest_weight_storage_mode(manifest: Mapping[str, object]) -> str:
    """Return the normalized weight storage mode from a staged manifest.

    Parameters
    ----------
    manifest : Mapping[str, object]
        Parsed staged-manifest payload.

    Returns
    -------
    str
        Normalized staged weight storage mode.
    """
    return _resolve_weight_storage_mode(manifest.get("weight_storage_mode"))


def _manifest_external_flash_bytes(manifest: Mapping[str, object]) -> int | None:
    """Return staged external-weight bytes from a manifest.

    Parameters
    ----------
    manifest : Mapping[str, object]
        Parsed staged-manifest payload.

    Returns
    -------
    int | None
        Staged external-weight blob size in bytes when present.
    """
    value = manifest.get("weights_blob_size")
    if value in (None, ""):
        return None
    return int(value)


def _build_storage_power_metrics(
    *,
    weight_storage_mode: str,
    external_flash_bytes: int | None,
) -> dict[str, Any]:
    """Return storage-related telemetry fields for downstream metrics.

    Parameters
    ----------
    weight_storage_mode : str
        Effective weight-placement policy.
    external_flash_bytes : int | None
        External weight-blob size in bytes when present.

    Returns
    -------
    dict[str, Any]
        Storage-related telemetry fields merged into final device metrics.
    """
    return {
        "weight_storage_mode": str(weight_storage_mode),
        "external_flash_bytes": (
            -1.0 if external_flash_bytes is None else float(external_flash_bytes)
        ),
    }


BOARD_DEFAULT_SPEC = build_stm32_nucleo_n657x0_q_spec()


class STM32NucleoN657X0QDevice(DeviceInterface):
    """TinyODOM-facing STM32 backend for the Nucleo N657x0 board.

    Notes
    -----
    The production backend supports compile, upload, back-to-back runtime
    measurement, harness-assisted energy capture, and optional external-flash
    weight staging. `cadenced` runtime remains deferred.
    """

    def __init__(
        self,
        *,
        serial_port: Optional[str] = None,
        device_options: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize the STM32 backend.

        Parameters
        ----------
        serial_port : str | None, optional
            Serial port used for upload/runtime measurement when callers do not
            override it per request.
        device_options : dict[str, Any] | None, optional
            Raw STM32 backend options from config or request plumbing.
        """
        self._serial_port = serial_port
        self._options = resolve_stm32_nucleo_n657x0_q_options(device_options)
        self._spec = build_stm32_nucleo_n657x0_q_spec(self._options)

    @property
    def spec(self) -> DeviceSpec:
        """Return the STM32 board specification.

        Returns
        -------
        DeviceSpec
            Static STM32 board metadata.
        """
        return self._spec

    @property
    def resolved_options(self) -> STM32NucleoN657X0QOptions:
        """Expose the normalized STM32 backend options.

        Returns
        -------
        STM32NucleoN657X0QOptions
            Resolved STM32 backend options.
        """
        return self._options

    def requires_candidate_model(self) -> bool:
        """Return whether the STM backend stages candidate-specific model artifacts.

        Returns
        -------
        bool
            Always ``True`` for STM candidate staging.
        """
        return True

    def requires_training_data(self) -> bool:
        """Return whether candidate export needs training/calibration data.

        Returns
        -------
        bool
            Always ``True`` for STM candidate export.
        """
        return True

    def requires_arena_validation(self) -> bool:
        """Return whether the backend requires arena-sentinel validation.

        Returns
        -------
        bool
            Always ``False`` because STM uses a single-shot arena sentinel
            contract and surfaces the real arena size from generated headers.
        """
        return False

    def supports_energy_measurement(self) -> bool:
        """Return whether the backend can produce real energy metrics.

        Returns
        -------
        bool
            ``True`` because STM can produce harness-assisted energy metrics.
        """
        return True

    def supports_runtime_measurement(self) -> bool:
        """Return whether the backend can run HIL latency/energy passes.

        Returns
        -------
        bool
            ``True`` because direct-serial runtime measurement is available.
        """
        return True

    def _build_candidate_root(self, outputs_dir: Path, model_variant: str) -> Path:
        """Return a unique per-candidate staging root.

        Parameters
        ----------
        outputs_dir : pathlib.Path
            Root output directory owned by the current TinyODOM run.
        model_variant : str
            Human-readable model variant name used to label the staging root.

        Returns
        -------
        pathlib.Path
            Unique per-candidate staging directory.
        """
        variant = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(model_variant).strip() or "candidate").strip("-")
        unique_suffix = uuid.uuid4().hex[:10]
        return outputs_dir.resolve() / "stm32" / f"{variant}-{unique_suffix}"

    def _read_storage_manifest(self, project_root: Path) -> dict[str, object]:
        """Read and validate the staged storage manifest.

        Parameters
        ----------
        project_root : pathlib.Path
            Staged STM32 FSBL root.

        Returns
        -------
        dict[str, object]
            Parsed manifest payload.
        """
        return _read_staged_manifest(project_root)

    def _storage_power_metrics(self, project_root: Path) -> dict[str, Any]:
        """Return storage metadata flattened for downstream metrics.

        Parameters
        ----------
        project_root : pathlib.Path
            Staged STM32 FSBL root.

        Returns
        -------
        dict[str, Any]
            Weight storage mode plus optional external flash bytes.
        """
        try:
            manifest = self._read_storage_manifest(project_root)
        except stm32_cube_clt.WorkflowError:
            return _build_storage_power_metrics(
                weight_storage_mode=self._options.weight_storage_mode,
                external_flash_bytes=None,
            )
        return _build_storage_power_metrics(
            weight_storage_mode=_manifest_weight_storage_mode(manifest),
            external_flash_bytes=_manifest_external_flash_bytes(manifest),
        )

    def _program_external_weights_if_needed(self, project_root: Path) -> dict[str, Any]:
        """Program staged external weights when the manifest requires it.

        Parameters
        ----------
        project_root : pathlib.Path
            Staged STM32 FSBL root.

        Returns
        -------
        dict[str, Any]
            Storage metadata to merge into final metrics.
        """
        try:
            manifest = self._read_storage_manifest(project_root)
        except stm32_cube_clt.WorkflowError:
            return _build_storage_power_metrics(
                weight_storage_mode=self._options.weight_storage_mode,
                external_flash_bytes=None,
            )
        weight_storage_mode = _manifest_weight_storage_mode(manifest)
        external_flash_bytes = _manifest_external_flash_bytes(manifest)
        power_metrics = _build_storage_power_metrics(
            weight_storage_mode=weight_storage_mode,
            external_flash_bytes=external_flash_bytes,
        )
        if weight_storage_mode != "external_flash":
            return power_metrics
        blob_path = manifest.get("weights_blob_path")
        if not blob_path:
            raise stm32_cube_clt.WorkflowError(
                "STM staged manifest is missing weights_blob_path for external_flash mode."
            )
        loader_path = manifest.get("weights_external_loader")
        resolved_loader = (
            _resolve_weights_external_loader(
                self._options.cubeprog_bin,
                self._options.weights_external_loader,
            )
            if not loader_path
            else Path(str(loader_path)).expanduser().resolve()
        )
        if external_flash_bytes is None or external_flash_bytes <= 0:
            raise stm32_cube_clt.WorkflowError(
                "STM staged manifest is missing a valid weights_blob_size for external_flash mode."
            )
        if external_flash_bytes > self._spec.max_external_flash_bytes:
            raise stm32_cube_clt.WorkflowError(
                "STM external weight blob exceeds available external flash "
                f"({external_flash_bytes} > {self._spec.max_external_flash_bytes})."
            )
        program_log = stm32_cube_clt.program_external_flash_blob(
            cubeprog_bin=self._options.cubeprog_bin,
            apid=self._options.apid,
            weights_blob_path=Path(str(blob_path)),
            weights_flash_address=str(manifest.get("weights_flash_address") or self._options.weights_flash_address),
            external_loader=resolved_loader,
        )
        power_metrics["external_flash_program_log"] = program_log
        return power_metrics

    def prepare_candidate(
        self,
        *,
        config: Any,
        hyperparams: Any,
        model: Any,
        outputs_dir: Path,
        tflite_model_path: Path,
        training_data: Any,
        model_variant: str,
        checkpoint_path: Path | str | None,
    ) -> Path:
        """Stage a candidate-specific STM32 FSBL project and return its root.

        Parameters
        ----------
        config : Any
            Loaded TinyODOM config object.
        hyperparams : Any
            Trial hyperparameters retained for interface compatibility after
            the Keras model has been built.
        model : Any
            Candidate Keras model to export as TFLite.
        outputs_dir : pathlib.Path
            Root output directory for the current run.
        tflite_model_path : pathlib.Path
            Shared controller argument retained for interface compatibility.
        training_data : Any
            Calibration or training data used for TFLite export.
        model_variant : str
            Human-readable model variant label.
        checkpoint_path : pathlib.Path | str | None
            Shared controller argument retained for interface compatibility.

        Returns
        -------
        pathlib.Path
            Per-candidate staged STM32 FSBL root.

        Raises
        ------
        ValueError
            If the candidate model or training data is missing.
        tinyodom.microcontrollers.stm32_cube_clt.WorkflowError
            If template validation, compatibility preflight, ST Edge AI
            generation, or staged-file validation fails.
        """
        del hyperparams, tflite_model_path, checkpoint_path
        if model is None:
            raise ValueError("STM32 candidate preparation requires a built Keras model.")
        if training_data is None or not hasattr(training_data, "inputs"):
            raise ValueError("STM32 candidate preparation requires calibration/training data.")

        from tinyodom.hardware import convert_to_tflite_model

        _ensure_staging_tools()
        canonical_template = _validate_project_structure(self._options.project_root)
        _validate_memory_reservations(canonical_template)
        resolved_external_loader: Path | None = None
        if self._options.weight_storage_mode == "external_flash":
            if (
                self._options.cubeprog_bin is None
                and stm32_cube_clt.default_cubeprog_bin() is None
            ):
                raise stm32_cube_clt.WorkflowError(
                    "STM32CubeProgrammer not found; required for external_flash weight storage mode."
                )
            if not self._options.weights_memory_pool.is_file():
                raise stm32_cube_clt.WorkflowError(
                    f"STM weights memory-pool JSON was not found: {self._options.weights_memory_pool}"
                )
            resolved_external_loader = _resolve_weights_external_loader(
                self._options.cubeprog_bin,
                self._options.weights_external_loader,
            )

        candidate_root = self._build_candidate_root(Path(outputs_dir), model_variant)
        staged_project_root = candidate_root / "FSBL"
        model_dir = candidate_root / "model"
        analyze_workspace_dir = candidate_root / "stedgeai_analyze_ws"
        analyze_output_dir = candidate_root / "stedgeai_analyze_out"
        stedgeai_workspace_dir = candidate_root / "stedgeai_ws"
        generated_output_dir = candidate_root / "stedgeai_out"
        model_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(canonical_template, staged_project_root)
        _strip_stale_debug_artifacts(staged_project_root)
        staged_project_root = _validate_project_structure(staged_project_root)
        _validate_memory_reservations(staged_project_root)

        tflite_path = model_dir / "tinyodom_candidate.tflite"
        convert_to_tflite_model(
            model=model,
            training_data=training_data.inputs,
            quantization=bool(config.training.quantization),
            output_name=tflite_path,
        )
        _run_stedgeai_analyze(
            model_path=tflite_path,
            workspace_dir=analyze_workspace_dir,
            output_dir=analyze_output_dir,
        )
        _run_stedgeai_generate(
            model_path=tflite_path,
            workspace_dir=stedgeai_workspace_dir,
            output_dir=generated_output_dir,
            weight_storage_mode=self._options.weight_storage_mode,
            weights_flash_address=self._options.weights_flash_address,
            weights_memory_pool=self._options.weights_memory_pool,
        )
        _stage_generated_outputs(generated_output_dir, staged_project_root)
        weights_blob_path = _generated_weights_blob_path(
            generated_output_dir,
            weight_storage_mode=self._options.weight_storage_mode,
        )
        _write_staged_manifest(
            project_root=staged_project_root,
            candidate_root=candidate_root,
            generated_output_dir=generated_output_dir,
            weight_storage_mode=self._options.weight_storage_mode,
            weights_blob_path=weights_blob_path,
            weights_flash_address=self._options.weights_flash_address,
            weights_external_loader=resolved_external_loader,
        )
        _parse_arena_bytes(staged_project_root / "Inc" / "network_data_params.h")
        header_path = _write_phase_config_header(
            project_root=staged_project_root,
            cpu_clock_mhz=self._options.cpu_clock_mhz,
        )
        _validate_phase_config_header(header_path, cpu_clock_mhz=self._options.cpu_clock_mhz)
        return staged_project_root

    def set_input_mode(
        self,
        input_mode: str,
        *,
        outputs_dir: Path,
        config: Any,
        sketches_dir: Path | None = None,
    ) -> Path | None:
        """STM32 ignores input-mode selection and returns the canonical template.

        Parameters
        ----------
        input_mode : str
            Requested input mode.
        outputs_dir : pathlib.Path
            Runtime outputs directory.
        config : Any
            Loaded TinyODOM config.
        sketches_dir : pathlib.Path | None, optional
            Optional sketch-root override retained for interface compatibility.

        Returns
        -------
        pathlib.Path | None
            Canonical STM32 template root.
        """
        del input_mode, outputs_dir, config, sketches_dir
        return self._options.project_root

    def compile(
        self,
        *,
        sketch_path: Path,
        arena_kb: int,
        window_size: int,
        num_channels: int,
        build_defines: Optional[dict[str, int]] = None,
    ) -> CompileResult:
        """Build the staged STM32 project and parse compile-time diagnostics.

        Parameters
        ----------
        sketch_path : pathlib.Path
            Staged STM32 FSBL root to compile.
        arena_kb : int
            Shared interface argument retained for compatibility.
        window_size : int
            Shared interface argument retained for compatibility.
        num_channels : int
            Shared interface argument retained for compatibility.
        build_defines : dict[str, int] | None, optional
            Shared interface argument retained for compatibility.

        Returns
        -------
        CompileResult
            Parsed build result including flash, RAM, activation arena, and
            linker heap/stack diagnostics when successful.
        """
        del arena_kb, window_size, num_channels, build_defines
        project_root = Path(sketch_path).expanduser().resolve()
        external_flash_bytes: int | None = None
        weight_storage_mode = self._options.weight_storage_mode
        try:
            project_root = _validate_project_structure(project_root)
            heap_bytes, stack_bytes = _validate_memory_reservations(project_root)
            try:
                manifest = _read_staged_manifest(project_root)
            except stm32_cube_clt.WorkflowError:
                manifest = None
            if manifest is not None:
                external_flash_bytes = _manifest_external_flash_bytes(manifest)
                weight_storage_mode = _manifest_weight_storage_mode(manifest)
            if external_flash_bytes is not None and external_flash_bytes > self._spec.max_external_flash_bytes:
                raise stm32_cube_clt.WorkflowError(
                    "STM external weight blob exceeds available external flash "
                    f"({external_flash_bytes} > {self._spec.max_external_flash_bytes})."
                )
            build_result = stm32_cube_clt.build_project(
                project_root=project_root,
                jobs=os.cpu_count() or 1,
                clean=True,
            )
            size_result = stm32_cube_clt.parse_size_output(build_result.elf_path)
            if size_result.elf_flash_bytes > self._spec.max_flash_bytes:
                raise stm32_cube_clt.WorkflowError(
                    "STM ELF flash image exceeds available internal flash "
                    f"({size_result.elf_flash_bytes} > {self._spec.max_flash_bytes})."
                )
            arena_bytes = _parse_arena_bytes(project_root / "Inc" / "network_data_params.h")
            diagnostics = (
                f"stm32_arena_bytes={arena_bytes}\n"
                f"stm32_heap_bytes={heap_bytes}\n"
                f"stm32_stack_bytes={stack_bytes}\n"
                f"stm32_weight_storage_mode={weight_storage_mode}\n"
                f"stm32_external_flash_bytes={external_flash_bytes if external_flash_bytes is not None else -1}"
            )
            return CompileResult(
                success=True,
                log="\n".join(
                    text
                    for text in (
                        build_result.log.strip(),
                        size_result.raw_output.strip(),
                        diagnostics,
                    )
                    if text
                ),
                flash_bytes=size_result.elf_flash_bytes,
                ram_bytes=size_result.ram_bytes,
                overflow_kind=None,
                build_dir=build_result.debug_dir,
                arena_bytes=arena_bytes,
                heap_bytes=heap_bytes,
                stack_bytes=stack_bytes,
                external_flash_bytes=external_flash_bytes,
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
                external_flash_bytes=external_flash_bytes,
            )

    def upload(
        self,
        *,
        sketch_path: Path,
        build_dir: Optional[Path],
        serial_port: Optional[str],
    ) -> UploadResult:
        """Debug-load the staged STM32 ELF through ST-LINK.

        Parameters
        ----------
        sketch_path : pathlib.Path
            Shared interface argument retained for compatibility.
        build_dir : pathlib.Path | None
            Staged ``Debug`` directory produced by the compile step.
        serial_port : str | None
            Shared interface argument retained for compatibility.

        Returns
        -------
        UploadResult
            Result of the ST-LINK debug-load workflow.
        """
        project_root = Path(sketch_path).expanduser().resolve()
        del serial_port
        if build_dir is None:
            return UploadResult(
                success=False,
                log="STM32 upload requires a staged build directory from the preceding compile step.",
            )
        try:
            storage_metrics = self._program_external_weights_if_needed(project_root)
            elf_path = stm32_cube_clt.resolve_elf_path(build_dir)
            log = stm32_cube_clt.debug_load_elf(
                elf_path=elf_path,
                gdbserver=self._options.gdbserver,
                gdb=self._options.gdb,
                cubeprog_bin=self._options.cubeprog_bin,
                gdb_port=self._options.gdb_port,
                apid=self._options.apid,
                server_ready_timeout_s=self._options.server_ready_timeout_s,
                run_after_load=True,
            )
            storage_lines = [
                f"{key}={value}"
                for key, value in storage_metrics.items()
                if key in {"weight_storage_mode", "external_flash_bytes"}
            ]
            return UploadResult(
                success=True,
                log="\n".join(text for text in (log, "\n".join(storage_lines)) if text),
            )
        except stm32_cube_clt.WorkflowError as exc:
            return UploadResult(success=False, log=str(exc))

    def measure(
        self,
        *,
        serial_port: Optional[str],
        baud_rate: int,
        serial_timeout_s: float,
        dut_ready_timeout_s: Optional[float] = None,
        harness_serial_port: Optional[str] = None,
        harness_fqbn: Optional[str] = None,
        harness_auto_flash: Optional[str] = None,
        harness_arm_pin: Optional[int] = None,
        harness_trigger_pin: Optional[int] = None,
        dut_arm_hold_ms: Optional[int] = None,
        harness_stable_low_ms: Optional[int] = None,
        harness_ready_timeout_s: Optional[float] = None,
        harness_arm_timeout_s: Optional[float] = None,
        harness_active_timeout_s: Optional[float] = None,
        harness_done_timeout_s: Optional[float] = None,
    ) -> MeasureResult:
        """Capture direct-serial runtime telemetry from an already-loaded DUT.

        Parameters
        ----------
        serial_port : str | None
            Serial port associated with the target.
        baud_rate : int
            Serial baud rate.
        serial_timeout_s : float
            Post-``START`` runtime timeout in seconds.
        dut_ready_timeout_s : float | None, optional
            Pre-``START`` timeout for init and ready tokens.
        harness_serial_port : str | None, optional
            Retained for interface compatibility.
        harness_fqbn : str | None, optional
            Retained for interface compatibility.
        harness_auto_flash : str | None, optional
            Retained for interface compatibility.
        harness_arm_pin : int | None, optional
            Retained for interface compatibility.
        harness_trigger_pin : int | None, optional
            Retained for interface compatibility.
        dut_arm_hold_ms : int | None, optional
            Retained for interface compatibility.
        harness_stable_low_ms : int | None, optional
            Retained for interface compatibility.
        harness_ready_timeout_s : float | None, optional
            Retained for interface compatibility.
        harness_arm_timeout_s : float | None, optional
            Retained for interface compatibility.
        harness_active_timeout_s : float | None, optional
            Retained for interface compatibility.
        harness_done_timeout_s : float | None, optional
            Retained for interface compatibility.

        Returns
        -------
        MeasureResult
            Parsed latency, serial log, and backend error details when present.
        """
        del (
            harness_serial_port,
            harness_fqbn,
            harness_auto_flash,
            harness_arm_pin,
            harness_trigger_pin,
            dut_arm_hold_ms,
            harness_stable_low_ms,
            harness_ready_timeout_s,
            harness_arm_timeout_s,
            harness_active_timeout_s,
            harness_done_timeout_s,
        )
        use_serial_port = self._serial_port if serial_port is None else serial_port
        if use_serial_port is None:
            raise ValueError("serial_port must be provided for STM32 runtime measurement.")
        try:
            telemetry = stm32_runtime.measure_direct_serial(
                serial_port=use_serial_port,
                baud_rate=baud_rate,
                boot_timeout_s=dut_ready_timeout_s,
                run_timeout_s=serial_timeout_s,
            )
        except stm32_runtime.STM32RuntimeProtocolError as exc:
            return MeasureResult(
                latency_s=None,
                arena_error_line=None,
                serial_log=exc.serial_log,
                power_metrics=stm32_runtime.build_backend_error_metrics(exc.kind, exc.detail),
            )
        except (serial.SerialException, RuntimeError) as exc:
            return MeasureResult(
                latency_s=None,
                arena_error_line=None,
                serial_log=[],
                power_metrics=stm32_runtime.build_backend_error_metrics("runtime_io", str(exc)),
            )
        return MeasureResult(
            latency_s=telemetry.latency_s,
            arena_error_line=None,
            serial_log=telemetry.serial_log,
            power_metrics=telemetry.power_metrics,
        )

    def evaluate(
        self,
        *,
        dirpath: str | Path,
        arena_kb: int,
        window_size: int,
        num_channels: int,
        serial_port: Optional[str] = None,
        run_hil: bool = True,
        baud_rate: int = 115200,
        serial_timeout_s: float = 12.0,
        dut_ready_timeout_s: Optional[float] = None,
        harness_serial_port: Optional[str] = None,
        harness_fqbn: Optional[str] = None,
        harness_auto_flash: Optional[str] = None,
        harness_arm_pin: Optional[int] = None,
        harness_trigger_pin: Optional[int] = None,
        dut_arm_hold_ms: Optional[int] = None,
        harness_stable_low_ms: Optional[int] = None,
        harness_ready_timeout_s: Optional[float] = None,
        harness_arm_timeout_s: Optional[float] = None,
        harness_active_timeout_s: Optional[float] = None,
        harness_done_timeout_s: Optional[float] = None,
    ) -> DeviceMetrics:
        """Run the staged STM32 compile/upload/runtime path and return metrics.

        Parameters
        ----------
        dirpath : str | pathlib.Path
            Staged STM32 FSBL root to compile.
        arena_kb : int
            Shared interface argument retained for compatibility.
        window_size : int
            Shared interface argument retained for compatibility.
        num_channels : int
            Shared interface argument retained for compatibility.
        serial_port : str | None, optional
            Serial port retained for interface compatibility.
        run_hil : bool, default=True
            Whether the caller requested runtime measurement.
        baud_rate : int, default=115200
            Serial baud rate retained for interface compatibility.
        serial_timeout_s : float, default=12.0
            Serial timeout retained for interface compatibility.
        dut_ready_timeout_s : float | None, optional
            DUT-ready timeout retained for interface compatibility.
        harness_serial_port : str | None, optional
            Harness serial port retained for interface compatibility.
        harness_fqbn : str | None, optional
            Harness FQBN retained for interface compatibility.
        harness_auto_flash : str | None, optional
            Harness flashing policy retained for interface compatibility.
        harness_arm_pin : int | None, optional
            Harness arm pin retained for interface compatibility.
        harness_trigger_pin : int | None, optional
            Harness trigger pin retained for interface compatibility.
        dut_arm_hold_ms : int | None, optional
            DUT arm hold retained for interface compatibility.
        harness_stable_low_ms : int | None, optional
            Stable-low period retained for interface compatibility.
        harness_ready_timeout_s : float | None, optional
            Harness ready timeout retained for interface compatibility.
        harness_arm_timeout_s : float | None, optional
            Harness arm timeout retained for interface compatibility.
        harness_active_timeout_s : float | None, optional
            Harness active timeout retained for interface compatibility.
        harness_done_timeout_s : float | None, optional
            Harness done timeout retained for interface compatibility.

        Returns
        -------
        DeviceMetrics
            Normalized STM32 metrics with compile-time sizes and optional runtime
            latency when HIL is enabled.
        """
        del arena_kb
        project_root = Path(dirpath).expanduser().resolve()
        try:
            storage_metrics = self._storage_power_metrics(project_root)
        except stm32_cube_clt.WorkflowError:
            storage_metrics = _build_storage_power_metrics(
                weight_storage_mode=self._options.weight_storage_mode,
                external_flash_bytes=None,
            )

        compile_result = self.compile(
            sketch_path=project_root,
            arena_kb=-1,
            window_size=window_size,
            num_channels=num_channels,
            build_defines=None,
        )
        compile_external_flash_bytes = getattr(compile_result, "external_flash_bytes", None)

        def _merge_metrics(extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
            """Merge storage metrics with optional runtime/backend detail fields.

            Parameters
            ----------
            extra : Mapping[str, Any] | None, optional
                Additional metrics to merge over the storage baseline.

            Returns
            -------
            dict[str, Any]
                Combined metrics payload.
            """
            merged = dict(storage_metrics)
            if extra:
                merged.update(extra)
            return merged

        def _latency_failure(kind: str, detail: str) -> DeviceMetrics:
            """Build a normalized latency-failure result for the current candidate.

            Parameters
            ----------
            kind : str
                Backend-specific latency failure category.
            detail : str
                Human-readable failure detail.

            Returns
            -------
            DeviceMetrics
                Failure metrics tagged with ``HIL_ERROR_LATENCY``.
            """
            return DeviceMetrics(
                ram_bytes=compile_result.ram_bytes or -1,
                flash_bytes=compile_result.flash_bytes or -1,
                latency_s=-1.0,
                arena_bytes=compile_result.arena_bytes or -1,
                error_code=HIL_ERROR_LATENCY,
                power_metrics=_merge_metrics(
                    stm32_runtime.build_backend_error_metrics(kind, detail)
                ),
                external_flash_bytes=compile_external_flash_bytes,
            )

        def _upload_failure(kind: str, detail: str) -> DeviceMetrics:
            """Build a normalized upload-failure result for the current candidate.

            Parameters
            ----------
            kind : str
                Backend-specific upload failure category.
            detail : str
                Human-readable failure detail.

            Returns
            -------
            DeviceMetrics
                Failure metrics tagged with ``HIL_ERROR_UPLOAD``.
            """
            return DeviceMetrics(
                ram_bytes=compile_result.ram_bytes or -1,
                flash_bytes=compile_result.flash_bytes or -1,
                latency_s=-1.0,
                arena_bytes=compile_result.arena_bytes or -1,
                error_code=HIL_ERROR_UPLOAD,
                power_metrics=_merge_metrics(
                    stm32_runtime.build_backend_error_metrics(kind, detail)
                ),
                external_flash_bytes=compile_external_flash_bytes,
            )

        overflow_kind = compile_result.overflow_kind
        if overflow_kind is not None:
            error_code = (
                HIL_ERROR_FLASH_OVERFLOW if overflow_kind == "flash" else HIL_ERROR_RAM_OVERFLOW
            )
            return DeviceMetrics(
                ram_bytes=compile_result.ram_bytes or -1,
                flash_bytes=compile_result.flash_bytes or -1,
                latency_s=-1.0,
                arena_bytes=compile_result.arena_bytes or -1,
                error_code=error_code,
                power_metrics=_merge_metrics(),
                external_flash_bytes=compile_external_flash_bytes,
            )
        if not compile_result.success:
            return DeviceMetrics(
                ram_bytes=compile_result.ram_bytes or -1,
                flash_bytes=compile_result.flash_bytes or -1,
                latency_s=-1.0,
                arena_bytes=compile_result.arena_bytes or -1,
                error_code=HIL_ERROR_COMPILE,
                power_metrics=_merge_metrics(),
                external_flash_bytes=compile_external_flash_bytes,
            )
        if not run_hil:
            return DeviceMetrics(
                ram_bytes=compile_result.ram_bytes or -1,
                flash_bytes=compile_result.flash_bytes or -1,
                latency_s=-1.0,
                arena_bytes=compile_result.arena_bytes or -1,
                error_code=HIL_ERROR_OK,
                power_metrics=_merge_metrics(),
                external_flash_bytes=compile_external_flash_bytes,
            )
        use_serial_port = self._serial_port if serial_port is None else serial_port
        if use_serial_port is None:
            raise ValueError("serial_port must be provided when running STM32 HIL uploads.")
        try:
            elf_path = stm32_cube_clt.resolve_elf_path(compile_result.build_dir)
        except stm32_cube_clt.WorkflowError as exc:
            return _upload_failure("upload", str(exc))
        harness_enabled = bool(harness_serial_port)
        if harness_enabled:
            try:
                measured_runs = _read_phase_config_measured_runs(project_root)
                arduino_base.ensure_harness_firmware(
                    harness_serial_port=str(harness_serial_port),
                    harness_fqbn="arduino:mbed_nano:nano33ble"
                    if harness_fqbn is None
                    else str(harness_fqbn),
                    harness_auto_flash="once"
                    if harness_auto_flash is None
                    else str(harness_auto_flash),
                    build_defines={
                        "TINYODOM_HARNESS_ARM_PIN": 3 if harness_arm_pin is None else int(harness_arm_pin),
                        "TINYODOM_HARNESS_TRIGGER_PIN": 2
                        if harness_trigger_pin is None
                        else int(harness_trigger_pin),
                        "TINYODOM_DUT_ARM_HOLD_MS": 600 if dut_arm_hold_ms is None else int(dut_arm_hold_ms),
                        "TINYODOM_HARNESS_STABLE_LOW_MS": 500
                        if harness_stable_low_ms is None
                        else int(harness_stable_low_ms),
                        "TINYODOM_HARNESS_ARM_TIMEOUT_MS": max(
                            0,
                            int(round((5.0 if harness_arm_timeout_s is None else float(harness_arm_timeout_s)) * 1000.0)),
                        ),
                        "TINYODOM_HARNESS_ACTIVE_TIMEOUT_MS": max(
                            0,
                            int(round((30.0 if harness_active_timeout_s is None else float(harness_active_timeout_s)) * 1000.0)),
                        ),
                        "TINYODOM_INFERENCE_RUNS": measured_runs,
                    },
                )
            except (RuntimeError, stm32_cube_clt.WorkflowError) as exc:
                return _upload_failure("harness_prepare", str(exc))
        try:
            # Canonical combined ordering:
            # 1. build
            # 2. program external weight blob
            # 3. open DUT serial
            # 4. open harness serial
            # 5. prime harness / send PING
            # 6. ELF load / resume
            # 7. wait for DUT READY
            # 8. send DUT START
            # 9. wait for harness DONE
            runtime_storage_metrics = self._program_external_weights_if_needed(project_root)
            with stm32_runtime.SerialMonitor(use_serial_port, baud_rate, "dut") as monitor:
                if harness_enabled:
                    harness_log: list[str] = []
                    with serial.Serial(
                        str(harness_serial_port),
                        baudrate=baud_rate,
                        timeout=0.1,
                    ) as harness:
                        prime_result = hil_protocol.prime_harness_session(
                            harness=harness,
                            harness_ready_timeout_s=(
                                5.0 if harness_ready_timeout_s is None else float(harness_ready_timeout_s)
                            ),
                            harness_log=harness_log,
                            flush_input=True,
                        )
                        if not prime_result.harness_ready:
                            return _latency_failure(
                                "harness_ready_timeout",
                                "Timed out waiting for HARNESS READY.",
                            )
                        stm32_cube_clt.debug_load_elf(
                            elf_path=elf_path,
                            gdbserver=self._options.gdbserver,
                            gdb=self._options.gdb,
                            cubeprog_bin=self._options.cubeprog_bin,
                            gdb_port=self._options.gdb_port,
                            apid=self._options.apid,
                            server_ready_timeout_s=self._options.server_ready_timeout_s,
                            run_after_load=True,
                        )
                        telemetry = stm32_runtime.execute_runtime_session(
                            monitor,
                            boot_timeout_s=(
                                stm32_runtime.DEFAULT_BOOT_TIMEOUT_S
                                if dut_ready_timeout_s is None
                                else float(dut_ready_timeout_s)
                            ),
                            run_timeout_s=float(serial_timeout_s),
                        )
                        harness_result = hil_protocol.wait_for_harness_done(
                            harness=harness,
                            harness_active_timeout_s=(
                                30.0 if harness_active_timeout_s is None else float(harness_active_timeout_s)
                            ),
                            harness_done_timeout_s=(
                                5.0 if harness_done_timeout_s is None else float(harness_done_timeout_s)
                            ),
                            harness_log=prime_result.harness_log,
                        )
                    if not harness_result.harness_done:
                        return _latency_failure(
                            "harness_done_timeout",
                            "Timed out waiting for harness DONE.",
                        )
                    harness_power = arduino_base._parse_power_metrics(harness_result.harness_log)
                    if harness_power is None:
                        return _latency_failure(
                            "harness_metrics_missing",
                            "Harness completed without reporting energy telemetry.",
                        )
                    dut_runs = int(telemetry.power_metrics.get("runs", -1))
                    if harness_result.runs_harness is None or harness_result.runs_harness != dut_runs:
                        return _latency_failure(
                            "harness_run_count_mismatch",
                            f"Run-count mismatch: dut={dut_runs} harness={harness_result.runs_harness}",
                        )
                    merged_power_metrics = arduino_base._merge_power_metrics(
                        primary=harness_power,
                        secondary=telemetry.power_metrics,
                    ) or dict(telemetry.power_metrics)
                    merged_power_metrics["phase"] = telemetry.power_metrics.get("phase", "back_to_back")
                    merged_power_metrics["runs"] = telemetry.power_metrics.get("runs", dut_runs)
                    for timer_field in (
                        "timer_output_s",
                        "timer_per_inference_s",
                        "timer_per_window_s",
                    ):
                        timer_value = telemetry.power_metrics.get(timer_field)
                        if timer_value is not None:
                            merged_power_metrics[timer_field] = timer_value
                    merged_power_metrics.update(runtime_storage_metrics)
                    return DeviceMetrics(
                        ram_bytes=compile_result.ram_bytes or -1,
                        flash_bytes=compile_result.flash_bytes or -1,
                        latency_s=telemetry.latency_s,
                        arena_bytes=compile_result.arena_bytes or -1,
                        error_code=HIL_ERROR_OK,
                        power_metrics=merged_power_metrics,
                        external_flash_bytes=compile_external_flash_bytes,
                    )
                stm32_cube_clt.debug_load_elf(
                    elf_path=elf_path,
                    gdbserver=self._options.gdbserver,
                    gdb=self._options.gdb,
                    cubeprog_bin=self._options.cubeprog_bin,
                    gdb_port=self._options.gdb_port,
                    apid=self._options.apid,
                    server_ready_timeout_s=self._options.server_ready_timeout_s,
                    run_after_load=True,
                )
                telemetry = stm32_runtime.execute_runtime_session(
                    monitor,
                    boot_timeout_s=(
                        stm32_runtime.DEFAULT_BOOT_TIMEOUT_S
                        if dut_ready_timeout_s is None
                        else float(dut_ready_timeout_s)
                    ),
                    run_timeout_s=float(serial_timeout_s),
                )
        except stm32_cube_clt.WorkflowError as exc:
            return _upload_failure(classify_stm32_backend_error(str(exc)), str(exc))
        except stm32_runtime.STM32RuntimeProtocolError as exc:
            return _latency_failure(exc.kind, exc.detail)
        except (serial.SerialException, RuntimeError) as exc:
            return _latency_failure("runtime_io", str(exc))
        return DeviceMetrics(
            ram_bytes=compile_result.ram_bytes or -1,
            flash_bytes=compile_result.flash_bytes or -1,
            latency_s=telemetry.latency_s,
            arena_bytes=compile_result.arena_bytes or -1,
            error_code=HIL_ERROR_OK,
            power_metrics=_merge_metrics(telemetry.power_metrics),
            external_flash_bytes=compile_external_flash_bytes,
        )
