from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

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
    HIL_ERROR_OK,
    HIL_ERROR_RAM_OVERFLOW,
)
from . import stm32_cube_clt


BOARD_NAME = "STM32_NUCLEO_N657X0_Q"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TEMPLATE_ROOT = REPO_ROOT / "sketches" / "stm32" / "tinyodom_tcn_stm32" / "FSBL"
DEFAULT_MAX_RAM_BYTES = 4_194_304
DEFAULT_MAX_FLASH_BYTES = 8_388_608
DEFAULT_CPU_CLOCK_MHZ = 600
SUPPORTED_CPU_CLOCK_MHZ = frozenset({200, 300, 400, 600, 800})
MIN_HEAP_BYTES = 0x2000
MIN_STACK_BYTES = 0x4000
EXPECTED_GENERATED_OUTPUTS = (
    "network.c",
    "network.h",
    "network_config.h",
    "network_data.c",
    "network_data.h",
    "network_data_params.c",
    "network_data_params.h",
)
WEIGHTS_BLOB_NAME = "network_data.bin"
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
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class STM32NucleoN657X0QOptions:
    """Normalized STM32 Phase 2 backend options.

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
    """

    project_root: Path
    gdbserver: Path | None
    gdb: Path | None
    cubeprog_bin: Path | None
    gdb_port: int
    apid: int
    server_ready_timeout_s: float
    cpu_clock_mhz: int


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
        If the requested preset is not one of the supported Phase 2 values.
    """
    clock_mhz = _coerce_int_with_default(raw_value, DEFAULT_CPU_CLOCK_MHZ)
    if clock_mhz not in SUPPORTED_CPU_CLOCK_MHZ:
        allowed = ", ".join(str(value) for value in sorted(SUPPORTED_CPU_CLOCK_MHZ))
        raise ValueError(f"Unsupported STM CPU clock preset {clock_mhz}. Expected one of: {allowed}.")
    return clock_mhz


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
    return STM32NucleoN657X0QOptions(
        project_root=project_root,
        gdbserver=gdbserver,
        gdb=gdb,
        cubeprog_bin=cubeprog_bin,
        gdb_port=gdb_port,
        apid=apid,
        server_ready_timeout_s=server_ready_timeout_s,
        cpu_clock_mhz=cpu_clock_mhz,
    )


def build_stm32_nucleo_n657x0_q_spec(
    options: STM32NucleoN657X0QOptions | None = None,
) -> DeviceSpec:
    """Build the STM32 ``DeviceSpec`` exposed through the device registry.

    Parameters
    ----------
    options : STM32NucleoN657X0QOptions | None, optional
        Reserved for future per-option spec overrides. Phase 2 currently uses
        a static hardware profile.

    Returns
    -------
    DeviceSpec
        Static STM32 board metadata used by the controller.
    """
    del options
    return DeviceSpec(
        name=BOARD_NAME,
        arena_sizes_kb=[-1],
        max_ram_bytes=DEFAULT_MAX_RAM_BYTES,
        max_flash_bytes=DEFAULT_MAX_FLASH_BYTES,
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
    """Write the generated Phase 2 phase-config header.

    Parameters
    ----------
    project_root : pathlib.Path
        Staged STM32 FSBL root that owns ``Inc/tcn_dut_phase_config.h``.
    cpu_clock_mhz : int
        Requested fixed CPU clock preset.
    latency_budget_ms : float, default=1.0
        Placeholder cadence budget written for Phase 3 compatibility.
    measured_runs : int, default=1
        Placeholder measured-run count written for Phase 3 compatibility.
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
        If the header does not contain the required Phase 2 settings.
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
    """Fail early when Phase 2 staging tools are unavailable.

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


def _run_stedgeai_generate(
    *,
    model_path: Path,
    workspace_dir: Path,
    output_dir: Path,
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
            f"Failed to execute ST Edge AI generation command: {' '.join(cmd)}\n\n{exc}"
        ) from exc
    output = f"{proc.stdout}{proc.stderr}".strip()
    if proc.returncode != 0:
        detail = output if output else "No ST Edge AI output captured."
        raise stm32_cube_clt.WorkflowError(
            f"ST Edge AI generation failed with exit code {proc.returncode}.\n\n{detail}"
        )


BOARD_DEFAULT_SPEC = build_stm32_nucleo_n657x0_q_spec()


class STM32NucleoN657X0QDevice(DeviceInterface):
    """TinyODOM-facing STM32 Phase 2 backend.

    Notes
    -----
    Phase 2 is intentionally compile-first. The backend stages a
    candidate-specific STM32 project and compiles it, but runtime HIL,
    cadence validation, and energy measurement remain Phase 3 work.
    """

    def __init__(
        self,
        *,
        serial_port: Optional[str] = None,
        device_options: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize the STM32 Phase 2 backend.

        Parameters
        ----------
        serial_port : str | None, optional
            Serial port reserved for future runtime measurement support.
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
            Always ``True`` for Phase 2.
        """
        return True

    def requires_training_data(self) -> bool:
        """Return whether candidate export needs training/calibration data.

        Returns
        -------
        bool
            Always ``True`` for Phase 2.
        """
        return True

    def requires_arena_validation(self) -> bool:
        """Return whether the backend requires arena-sentinel validation.

        Returns
        -------
        bool
            Always ``False`` because Phase 2 can produce real arena bytes, but
            the shared NAS path must still tolerate sentinel-style behavior.
        """
        return False

    def supports_energy_measurement(self) -> bool:
        """Return whether the backend can produce real energy metrics.

        Returns
        -------
        bool
            Always ``False`` for Phase 2.
        """
        return False

    def supports_runtime_measurement(self) -> bool:
        """Return whether the backend can run HIL latency/energy passes.

        Returns
        -------
        bool
            Always ``False`` for Phase 2.
        """
        return False

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
            Trial hyperparameters. Phase 2 does not currently use them
            directly after the Keras model has been built.
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
            If template validation, ST Edge AI generation, or staged-file
            validation fails.
        """
        del hyperparams, tflite_model_path, checkpoint_path
        if model is None:
            raise ValueError("STM32 candidate preparation requires a built Keras model.")
        if training_data is None or not hasattr(training_data, "inputs"):
            raise ValueError("STM32 candidate preparation requires calibration/training data.")

        from tinyodom.hardware import convert_to_tflite_model

        # Phase 2 is compile-first, so staging only requires ``stedgeai``.
        # Upload/debug tools are resolved later by the upload path itself.
        _ensure_staging_tools()
        canonical_template = _validate_project_structure(self._options.project_root)
        _validate_memory_reservations(canonical_template)

        candidate_root = self._build_candidate_root(Path(outputs_dir), model_variant)
        staged_project_root = candidate_root / "FSBL"
        model_dir = candidate_root / "model"
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
        _run_stedgeai_generate(
            model_path=tflite_path,
            workspace_dir=stedgeai_workspace_dir,
            output_dir=generated_output_dir,
        )
        _stage_generated_outputs(generated_output_dir, staged_project_root)
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
        try:
            project_root = _validate_project_structure(project_root)
            heap_bytes, stack_bytes = _validate_memory_reservations(project_root)
            build_result = stm32_cube_clt.build_project(
                project_root=project_root,
                jobs=os.cpu_count() or 1,
                clean=True,
            )
            size_result = stm32_cube_clt.parse_size_output(build_result.elf_path)
            arena_bytes = _parse_arena_bytes(project_root / "Inc" / "network_data_params.h")
            diagnostics = (
                f"stm32_arena_bytes={arena_bytes}\n"
                f"stm32_heap_bytes={heap_bytes}\n"
                f"stm32_stack_bytes={stack_bytes}"
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
        del sketch_path, serial_port
        if build_dir is None:
            return UploadResult(
                success=False,
                log="STM32 upload requires a staged build directory from the preceding compile step.",
            )
        try:
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
            return UploadResult(success=True, log=log)
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
        """Reject runtime measurement until Phase 3.

        Parameters
        ----------
        serial_port : str | None
            Serial port associated with the target.
        baud_rate : int
            Serial baud rate.
        serial_timeout_s : float
            Serial read timeout in seconds.
        dut_ready_timeout_s : float | None, optional
            DUT-ready timeout.
        harness_serial_port : str | None, optional
            Harness serial port.
        harness_fqbn : str | None, optional
            Harness FQBN.
        harness_auto_flash : str | None, optional
            Harness flashing policy.
        harness_arm_pin : int | None, optional
            Harness arm pin.
        harness_trigger_pin : int | None, optional
            Harness trigger pin.
        dut_arm_hold_ms : int | None, optional
            DUT arm hold time.
        harness_stable_low_ms : int | None, optional
            Required stable-low period.
        harness_ready_timeout_s : float | None, optional
            Harness ready timeout.
        harness_arm_timeout_s : float | None, optional
            Harness arm timeout.
        harness_active_timeout_s : float | None, optional
            Harness active timeout.
        harness_done_timeout_s : float | None, optional
            Harness done timeout.

        Returns
        -------
        MeasureResult
            This method never returns in Phase 2.

        Raises
        ------
        RuntimeError
            Always raised because runtime measurement is a Phase 3 feature.
        """
        del (
            serial_port,
            baud_rate,
            serial_timeout_s,
            dut_ready_timeout_s,
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
        raise RuntimeError("STM runtime measurement is not implemented in Phase 2.")

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
        """Run the staged STM32 compile path and return normalized metrics.

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
            Normalized compile-only STM32 metrics.
        """
        del (
            arena_kb,
            serial_port,
            baud_rate,
            serial_timeout_s,
            dut_ready_timeout_s,
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
        compile_result = self.compile(
            sketch_path=Path(dirpath),
            arena_kb=-1,
            window_size=window_size,
            num_channels=num_channels,
            build_defines=None,
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
            )
        if not compile_result.success:
            return DeviceMetrics(
                ram_bytes=compile_result.ram_bytes or -1,
                flash_bytes=compile_result.flash_bytes or -1,
                latency_s=-1.0,
                arena_bytes=compile_result.arena_bytes or -1,
                error_code=HIL_ERROR_COMPILE,
            )
        if not run_hil:
            return DeviceMetrics(
                ram_bytes=compile_result.ram_bytes or -1,
                flash_bytes=compile_result.flash_bytes or -1,
                latency_s=-1.0,
                arena_bytes=compile_result.arena_bytes or -1,
                error_code=HIL_ERROR_OK,
            )
        logger.warning(
            "STM runtime measurement is not implemented in Phase 2; returning compile-only failure semantics."
        )
        return DeviceMetrics(
            ram_bytes=compile_result.ram_bytes or -1,
            flash_bytes=compile_result.flash_bytes or -1,
            latency_s=-1.0,
            arena_bytes=compile_result.arena_bytes or -1,
            error_code=HIL_ERROR_COMPILE,
        )
