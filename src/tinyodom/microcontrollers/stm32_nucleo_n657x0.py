from __future__ import annotations

import logging
import os
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
# FIXME: This default points at the toy STM32 example project under
# `analysis_scripts/` purely so the Phase 1 backend has a concrete project to
# build against while the production STM firmware tree is still being promoted
# into `src/`. Replace this with the real TinyODOM STM project root as soon as
# that project exists. Do not treat the toy project path as a long-term default.
DEFAULT_PROJECT_ROOT = (
    REPO_ROOT
    / "analysis_scripts"
    / "stm32_example_project"
    / "stm32_toy_ai_project"
    / "FSBL"
)
DEFAULT_MAX_RAM_BYTES = 4_194_304
DEFAULT_MAX_FLASH_BYTES = 8_388_608
logger = logging.getLogger(__name__)
ALLOW_TOY_PROJECT_FALLBACK_ENV = "TINYODOM_ALLOW_STM32_TOY_PROJECT_FALLBACK"


@dataclass(frozen=True)
class STM32NucleoN657X0QOptions:
    """Normalized STM32 backend options.

    Parameters
    ----------
    project_root : pathlib.Path
        Root of the STM32CubeIDE-generated FSBL project used for Phase 1
        compile and debug-load operations.
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
    cpu_clock_mhz : int | None
        Placeholder Phase 1 CPU clock selection carried through config and
        device options. The Phase 1 backend preserves this value for future
        use but does not apply it yet.
    """

    project_root: Path
    gdbserver: Path | None
    gdb: Path | None
    cubeprog_bin: Path | None
    gdb_port: int
    apid: int
    server_ready_timeout_s: float
    cpu_clock_mhz: int | None


def _resolve_optional_path(value: object | None) -> Path | None:
    """Normalize an optional filesystem path.

    Parameters
    ----------
    value : object | None
        Raw config/runtime value that may contain a path-like string.

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
        Raw option value from config or request plumbing.
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
        Raw option value from config or request plumbing.
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


def resolve_stm32_nucleo_n657x0_q_options(
    device_options: Optional[Mapping[str, object]],
) -> STM32NucleoN657X0QOptions:
    """Resolve STM32 backend options from config/runtime input.

    Parameters
    ----------
    device_options : Mapping[str, object] | None
        Raw option mapping passed in from config parsing or controller code.

    Returns
    -------
    STM32NucleoN657X0QOptions
        Normalized, fully-resolved STM32 backend options.

    Raises
    ------
    ValueError
        If no STM project root is provided and the development-only toy
        fallback has not been explicitly enabled via
        ``TINYODOM_ALLOW_STM32_TOY_PROJECT_FALLBACK=1``.
    """
    options = dict(device_options or {})
    # Phase 1 uses a fixed STM32 project tree, so normalize everything up front
    # here and keep the rest of the device methods free of config-shape logic.
    project_root = _resolve_optional_path(options.get("project_root"))
    if project_root is None:
        if os.environ.get(ALLOW_TOY_PROJECT_FALLBACK_ENV) == "1":
            logger.warning(
                "Using STM Phase 1 toy project fallback from analysis_scripts/. "
                "Set device.stm32.project_root to target a real STM firmware tree."
            )
            project_root = DEFAULT_PROJECT_ROOT
        else:
            raise ValueError(
                "STM32_NUCLEO_N657X0_Q requires device.stm32.project_root to be set. "
                f"The temporary toy-project fallback is disabled unless {ALLOW_TOY_PROJECT_FALLBACK_ENV}=1."
            )
    gdbserver = _resolve_optional_path(options.get("gdbserver"))
    gdb = _resolve_optional_path(options.get("gdb"))
    cubeprog_bin = _resolve_optional_path(options.get("cubeprog_bin"))
    gdb_port = _coerce_int_with_default(options.get("gdb_port"), stm32_cube_clt.DEFAULT_GDB_PORT)
    apid = _coerce_int_with_default(options.get("apid"), stm32_cube_clt.DEFAULT_APID)
    server_ready_timeout_s = _coerce_float_with_default(
        options.get("server_ready_timeout_s"),
        stm32_cube_clt.SERVER_READY_TIMEOUT_S,
    )
    raw_cpu_clock = options.get("cpu_clock_mhz")
    cpu_clock_mhz = int(raw_cpu_clock) if raw_cpu_clock not in (None, "") else None
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
    """Build the STM32 ``DeviceSpec`` used by the existing controller path.

    Parameters
    ----------
    options : STM32NucleoN657X0QOptions | None, optional
        Reserved for future backend-specific spec customization. Phase 1 uses a
        static spec regardless of options.

    Returns
    -------
    DeviceSpec
        STM32 board metadata exposed through the existing device registry.
    """
    del options
    return DeviceSpec(
        name=BOARD_NAME,
        # The controller currently expects every board to provide at least one
        # arena size candidate. STM does not use TFLM tensor-arena sweep
        # semantics, so `-1` is a compatibility sentinel that forces a single
        # pass while making it explicit that arena sizing is not meaningful for
        # this backend yet.
        arena_sizes_kb=[-1],
        max_ram_bytes=DEFAULT_MAX_RAM_BYTES,
        max_flash_bytes=DEFAULT_MAX_FLASH_BYTES,
        toolchain="stm32-cube",
    )


BOARD_DEFAULT_SPEC = build_stm32_nucleo_n657x0_q_spec()


class STM32NucleoN657X0QDevice(DeviceInterface):
    """Concrete STM32 Nucleo N657X0-Q backend wrapper.

    This class is the TinyODOM-facing STM backend. It translates the shared
    :class:`DeviceInterface` contract into STM32CubeCLT build/debug-load calls
    and normalizes the resulting metrics/error codes back into TinyODOM's
    backend-agnostic result dataclasses.
    """

    def __init__(
        self,
        *,
        serial_port: Optional[str] = None,
        device_options: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize the STM backend wrapper.

        Parameters
        ----------
        serial_port : str | None, optional
            Serial port associated with the target device. Phase 1 keeps this
            for interface compatibility, but compile-only STM flows do not use
            it yet.
        device_options : dict[str, Any] | None, optional
            Raw STM backend options from config/request plumbing.
        """
        self._serial_port = serial_port
        self._options = resolve_stm32_nucleo_n657x0_q_options(device_options)
        self._spec = build_stm32_nucleo_n657x0_q_spec(self._options)

    @property
    def spec(self) -> DeviceSpec:
        """Return the device specification metadata.

        Returns
        -------
        DeviceSpec
            Static board metadata used by the controller and registry APIs.
        """
        return self._spec

    @property
    def resolved_options(self) -> STM32NucleoN657X0QOptions:
        """Expose the normalized STM backend options.

        Returns
        -------
        STM32NucleoN657X0QOptions
            Resolved Phase 1 STM backend options.
        """
        return self._options

    def requires_candidate_model(self) -> bool:
        """Return whether STM Phase 1 consumes generated TinyODOM model artifacts."""
        return False

    def requires_training_data(self) -> bool:
        """Return whether STM Phase 1 needs calibration/training data."""
        return False

    def requires_arena_validation(self) -> bool:
        """Return whether STM Phase 1 treats ``arena_bytes`` as a required metric."""
        return False

    def supports_energy_measurement(self) -> bool:
        """Return whether STM Phase 1 can produce real energy metrics."""
        return False

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
        """Validate and return the backend-owned STM project root for Phase 1."""
        del (
            config,
            hyperparams,
            model,
            outputs_dir,
            tflite_model_path,
            training_data,
            model_variant,
            checkpoint_path,
        )
        return stm32_cube_clt.validate_project_root(self._options.project_root)

    def set_input_mode(
        self,
        input_mode: str,
        *,
        outputs_dir: Path,
        config: Any,
        sketches_dir: Path | None = None,
    ) -> Path | None:
        """Accept input-mode changes as a no-op and return the STM project root."""
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
        """Build the STM project and parse compile-time RAM/flash usage.

        Parameters
        ----------
        sketch_path : pathlib.Path
            Shared interface argument for Arduino compatibility. STM Phase 1
            ignores this and instead builds ``self._options.project_root``.
        arena_kb : int
            Compatibility sentinel from the current controller path. STM Phase 1
            ignores this value because STM does not use TFLM tensor-arena sweep
            semantics.
        window_size : int
            Shared interface argument reserved for future STM staging work.
        num_channels : int
            Shared interface argument reserved for future STM staging work.
        build_defines : dict[str, int] | None, optional
            Shared interface argument reserved for future STM staging work.

        Returns
        -------
        CompileResult
            Normalized compile result containing RAM/flash usage when the build
            succeeds, or classified overflow information on failure.
        """
        del sketch_path, arena_kb, window_size, num_channels, build_defines
        try:
            # Phase 1 intentionally compiles a fixed STM project. The future
            # model-staging path will hook in before this build step.
            build_result = stm32_cube_clt.build_project(
                project_root=self._options.project_root,
                jobs=os.cpu_count() or 1,
                clean=False,
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
                build_dir=self._options.project_root / "Debug",
            )

    def upload(
        self,
        *,
        sketch_path: Path,
        build_dir: Optional[Path],
        serial_port: Optional[str],
    ) -> UploadResult:
        """Debug-load the built STM ELF through ST-LINK.

        Parameters
        ----------
        sketch_path : pathlib.Path
            Shared interface argument for Arduino compatibility. The STM backend
            ignores this and instead loads the ELF produced from the configured
            STM project root.
        build_dir : pathlib.Path | None
            Optional explicit build directory. When omitted, the backend uses
            ``<project_root>/Debug``.
        serial_port : str | None
            Shared interface argument retained for interface compatibility.

        Returns
        -------
        UploadResult
            Result of the ST-LINK debug-load workflow.
        """
        del sketch_path, serial_port
        resolved_build_dir = build_dir or (self._options.project_root / "Debug")
        try:
            elf_path = stm32_cube_clt.resolve_elf_path(resolved_build_dir)
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
        """Reject STM runtime measurement in Phase 1.

        Parameters
        ----------
        serial_port : str | None
            Serial port for the DUT.
        baud_rate : int
            Serial baud rate.
        serial_timeout_s : float
            Serial read timeout.
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
            DUT arming hold time.
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
            This method never returns in Phase 1.

        Raises
        ------
        RuntimeError
            Always raised because runtime measurement is deferred until a later
            STM backend phase.
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
        raise RuntimeError("STM runtime measurement is not implemented in Phase 1.")

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
        """Run the Phase 1 STM compile path and return normalized metrics.

        Parameters
        ----------
        dirpath : str | pathlib.Path
            Shared interface argument. Phase 1 ignores this because the backend
            always uses the configured STM project root.
        arena_kb : int
            Compatibility sentinel from the current controller path. STM ignores
            it and always reports ``arena_bytes=-1``.
        window_size : int
            Shared interface argument reserved for future STM staging work.
        num_channels : int
            Shared interface argument reserved for future STM staging work.
        serial_port : str | None, optional
            Serial port for runtime measurement. Retained for interface
            compatibility.
        run_hil : bool, optional
            Whether the caller requested upload/measurement. Phase 1 only
            supports compile-only semantics.
        baud_rate : int, optional
            Serial baud rate.
        serial_timeout_s : float, optional
            Serial read timeout.
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
            DUT arming hold time.
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
        DeviceMetrics
            Compile-only Phase 1 STM metrics normalized to the shared TinyODOM
            backend contract.
        """
        del (
            dirpath,
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
        # The controller still calls every backend through the same evaluate
        # path. For STM Phase 1, normalize that shared call into a compile-only
        # backend run against the fixed STM project.
        compile_result = self.compile(
            sketch_path=self._options.project_root,
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
                arena_bytes=-1,
                error_code=error_code,
            )
        if not compile_result.success:
            return DeviceMetrics(
                ram_bytes=compile_result.ram_bytes or -1,
                flash_bytes=compile_result.flash_bytes or -1,
                latency_s=-1.0,
                arena_bytes=-1,
                error_code=HIL_ERROR_COMPILE,
            )
        if not run_hil:
            return DeviceMetrics(
                ram_bytes=compile_result.ram_bytes or -1,
                flash_bytes=compile_result.flash_bytes or -1,
                latency_s=-1.0,
                arena_bytes=-1,
                error_code=HIL_ERROR_OK,
            )
        logger.warning(
            "STM runtime measurement is not implemented in Phase 1; returning compile-only failure semantics."
        )
        # Keep using HIL_ERROR_COMPILE here, even though the compile step
        # succeeded, because HIL_ERROR_UPLOAD is currently interpreted by the
        # shared controller as "device missing" and tears down the HIL server.
        # This is a Phase 1 compromise until STM runtime measurement has its
        # own non-disruptive error path.
        return DeviceMetrics(
            ram_bytes=compile_result.ram_bytes or -1,
            flash_bytes=compile_result.flash_bytes or -1,
            latency_s=-1.0,
            arena_bytes=-1,
            error_code=HIL_ERROR_COMPILE,
        )
