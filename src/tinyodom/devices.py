from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional, Sequence

import numpy as np
import logging
import serial

from .errors import (
    HIL_ERROR_OK,
    HIL_ERROR_COMPILE,
    HIL_ERROR_LATENCY,
    HIL_ERROR_UNDER_SIZED,
    HIL_ERROR_FLASH_OVERFLOW,
    HIL_ERROR_RAM_OVERFLOW,
    HIL_ERROR_UPLOAD,
)
from . import hil_protocol
from .microcontrollers import arduino_base

logger = logging.getLogger(__name__)
RuntimeMeasureMode = Literal["direct_serial", "harness_only"]


@dataclass(frozen=True)
class DeviceSpec:
    """Static hardware capabilities and constraints.

    Parameters
    ----------
    name : str
        Human-readable device identifier.
    arena_sizes_kb : Sequence[int]
        Candidate tensor arena sizes in KiB.
    max_ram_bytes : int
        Maximum usable RAM on the device in bytes.
    max_flash_bytes : int
        Maximum usable flash on the device in bytes.
    fqbn : str | None, optional
        Fully qualified board name (Arduino CLI only).
    toolchain : str | None, optional
        Toolchain identifier (e.g., "arduino-cli", "stm32-cube").
    """

    name: str
    arena_sizes_kb: Sequence[int]
    max_ram_bytes: int
    max_flash_bytes: int
    fqbn: Optional[str] = None
    toolchain: Optional[str] = None


@dataclass(frozen=True)
class DeviceMetrics:
    """Standardized telemetry captured after a HIL or proxy run.

    Parameters
    ----------
    ram_bytes : int
        RAM usage in bytes (or -1 if unavailable).
    flash_bytes : int
        Flash usage in bytes (or -1 if unavailable).
    latency_s : float
        Inference latency in seconds (or -1.0 if unavailable).
    arena_bytes : int
        Tensor arena size used in bytes.
    error_code : int
        Error code emitted by the device evaluation.
    power_metrics : dict[str, Any] | None, optional
        Optional raw power metrics parsed from the serial log.
    retry_hint_bytes : int | None, optional
        Suggested arena size in bytes for the next attempt, when available.
    """

    ram_bytes: int
    flash_bytes: int
    latency_s: float
    arena_bytes: int
    error_code: int
    power_metrics: Optional[Dict[str, Any]] = None
    retry_hint_bytes: Optional[int] = None


@dataclass(frozen=True)
class CompileResult:
    """Result of a device-specific compile step.

    Parameters
    ----------
    success : bool
        True when compilation succeeded.
    log : str
        Combined stdout/stderr from the compile step.
    flash_bytes : int | None
        Parsed flash usage in bytes, if available.
    ram_bytes : int | None
        Parsed RAM usage in bytes, if available.
    overflow_kind : str | None
        "flash" or "ram" when an overflow is detected.
    build_dir : pathlib.Path | None
        Build cache directory if applicable.
    """

    success: bool
    log: str
    flash_bytes: Optional[int]
    ram_bytes: Optional[int]
    overflow_kind: Optional[str]
    build_dir: Optional[Path] = None


@dataclass(frozen=True)
class UploadResult:
    """Result of a device-specific upload step.

    Parameters
    ----------
    success : bool
        True when upload succeeded.
    log : str
        Combined stdout/stderr from the upload step.
    """

    success: bool
    log: str


@dataclass(frozen=True)
class MeasureResult:
    """Parsed telemetry from a device measurement pass.

    Parameters
    ----------
    latency_s : float | None
        Inference latency in seconds, if available.
    arena_error_line : str | None
        First arena allocation error line, if found.
    serial_log : list[str]
        Decoded serial log captured during measurement.
    power_metrics : dict[str, float | None] | None
        Parsed power metrics fields, if present.
    """

    latency_s: Optional[float]
    arena_error_line: Optional[str]
    serial_log: list[str]
    power_metrics: Optional[Dict[str, Optional[float]]]


class DeviceInterface(ABC):
    """Contract for device-specific compile/upload/measure workflows."""

    def runtime_measure_mode(self) -> RuntimeMeasureMode:
        """Return runtime measurement mode for this device profile."""
        return "direct_serial"

    def runtime_mode_build_defines(self) -> Dict[str, int]:
        """Return additional compile-time defines for the runtime mode."""
        return {}

    def prepare_for_runtime(self, *, runtime_mode: RuntimeMeasureMode, serial_port: str) -> None:
        """Run optional board-specific setup before upload/measurement."""
        del runtime_mode, serial_port

    @property
    @abstractmethod
    def spec(self) -> DeviceSpec:
        """Return the device specification metadata."""

    @abstractmethod
    def compile(
        self,
        *,
        sketch_path: Path,
        arena_kb: int,
        window_size: int,
        num_channels: int,
        build_defines: Optional[Dict[str, int]] = None,
    ) -> CompileResult:
        """Compile the firmware for the target device.

        Parameters
        ----------
        sketch_path : pathlib.Path
            Path to the firmware project directory.
        arena_kb : int
            Tensor arena size in KiB for this attempt.
        window_size : int
            Sliding window length used by the model.
        num_channels : int
            Number of sensor channels per window.

        Returns
        -------
        CompileResult
            Parsed compile output and resource usage.
        """

    @abstractmethod
    def upload(
        self,
        *,
        sketch_path: Path,
        build_dir: Optional[Path],
        serial_port: Optional[str],
    ) -> UploadResult:
        """Upload compiled firmware to the device.

        Parameters
        ----------
        sketch_path : pathlib.Path
            Path to the firmware project directory.
        build_dir : pathlib.Path | None
            Build directory containing compiled artifacts.
        serial_port : str | None
            Serial port to upload to.

        Returns
        -------
        UploadResult
            Upload success flag and log output.
        """

    @abstractmethod
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
        """Capture latency/power metrics from the device.

        Parameters
        ----------
        serial_port : str | None
            Serial port used for log capture.
        baud_rate : int
            Serial baud rate.
        serial_timeout_s : float
            Timeout for log capture.
        dut_ready_timeout_s : float | None
            Time to wait for DUT READY before sending START.
        harness_serial_port : str | None
            Optional serial port for the INA228 harness.
        harness_fqbn : str | None
            FQBN for the harness board (Arduino CLI only).
        harness_auto_flash : str | None
            Harness flashing policy (once, always, never).
        harness_arm_pin : int | None
            Harness/DUT arm pin number (active-low).
        harness_trigger_pin : int | None
            Harness/DUT trigger pin number.
        dut_arm_hold_ms : int | None
            Arm-low hold duration before DUT sets trigger HIGH.
        harness_stable_low_ms : int | None
            Stable-low window required before harness arms.
        harness_ready_timeout_s : float | None
            Timeout waiting for HARNESS READY.
        harness_arm_timeout_s : float | None
            Harness-side arm timeout while waiting for a valid D2 trigger after
            D3 active-low arming. This value is compiled into harness firmware
            as `TINYODOM_HARNESS_ARM_TIMEOUT_MS`; set to 0 to disable it.
        harness_active_timeout_s : float | None
            Maximum time to wait for a harness measurement window.
        harness_done_timeout_s : float | None
            Timeout waiting for DONE response.

        Returns
        -------
        MeasureResult
            Parsed latency/power metrics.
        """

    @abstractmethod
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
        """Compile, upload, and measure a model on the target device.

        Parameters
        ----------
        dirpath : str | pathlib.Path
            Path to the sketch or project directory containing firmware sources.
        arena_kb : int
            Tensor arena size in KiB for this attempt.
        window_size : int
            Sliding window length used by the model.
        num_channels : int
            Number of sensor channels in each window.
        serial_port : str | None, optional
            Serial port for upload and log capture.
        run_hil : bool, optional
            When False, compile only (no upload/measurement).
        baud_rate : int, optional
            Serial baud rate for log capture.
        serial_timeout_s : float, optional
            Timeout for latency/power log capture.
        dut_ready_timeout_s : float | None, optional
            Time to wait for DUT READY before sending START.

        Returns
        -------
        DeviceMetrics
            Normalized metrics for the run.
        """


_LEGACY_DEVICE_SPECS = {
    "NUCLEO_F746ZG": {
        "arena_sizes": np.array([10, 30, 50, 75, 100, 150, 175, 200, 250, 280, 280]),
        "max_ram": 300_000,
        "max_flash": 800_000,
        "fqbn": None,
    },
    "NUCLEO_L476RG": {
        "arena_sizes": np.array([10, 25, 40, 70, 85, 100, 100]),
        "max_ram": 100_000,
        "max_flash": 800_000,
        "fqbn": None,
    },
    "NUCLEO_F446RE": {
        "arena_sizes": np.array([10, 25, 40, 70, 85, 100, 100]),
        "max_ram": 100_000,
        "max_flash": 400_000,
        "fqbn": None,
    },
    "ARCH_MAX": {
        "arena_sizes": np.array([10, 25, 40, 70, 95, 120, 140, 160, 170, 170]),
        "max_ram": 180_000,
        "max_flash": 400_000,
        "fqbn": None,
    },
    "ARDUINO_NANO_RP2040_CONNECT": {
        "arena_sizes": np.array([10, 25, 40, 70, 95, 120, 140, 150, 160, 180, 200, 210, 220]),
        "max_ram": 225_000,
        "max_flash": 15_000_000,
        "fqbn": "arduino:mbed_nano:nano_rp2040_connect",
    },
}


def _registry_device_specs() -> Dict[str, Dict[str, Any]]:
    """Load default device specs from microcontroller registry classes.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Compatibility-style spec map keyed by device name.
    """
    try:
        from .microcontrollers import list_device_specs
    except Exception:
        return {}

    loaded_specs = list_device_specs()
    normalized: Dict[str, Dict[str, Any]] = {}
    for device_name, spec in loaded_specs.items():
        normalized[device_name] = {
            "arena_sizes": np.array([int(value) for value in spec.get("arena_sizes", [])]),
            "max_ram": int(spec.get("max_ram", 0)),
            "max_flash": int(spec.get("max_flash", 0)),
            "fqbn": spec.get("fqbn"),
        }
    return normalized


def _rebuild_device_specs() -> Dict[str, Dict[str, Any]]:
    """Build compatibility DEVICE_SPECS from legacy + registry-backed metadata.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Merged compatibility map used by legacy call sites.
    """
    merged: Dict[str, Dict[str, Any]] = {}
    for device_name, raw in _LEGACY_DEVICE_SPECS.items():
        merged[device_name] = {
            "arena_sizes": np.array(raw["arena_sizes"]),
            "max_ram": int(raw["max_ram"]),
            "max_flash": int(raw["max_flash"]),
            "fqbn": raw.get("fqbn"),
        }
    merged.update(_registry_device_specs())
    return merged


# Initialize with legacy entries first; registry-backed entries are merged after
# ArduinoDevice is defined to avoid import cycles.
DEVICE_SPECS: Dict[str, Dict[str, Any]] = {
    device_name: {
        "arena_sizes": np.array(raw["arena_sizes"]),
        "max_ram": int(raw["max_ram"]),
        "max_flash": int(raw["max_flash"]),
        "fqbn": raw.get("fqbn"),
    }
    for device_name, raw in _LEGACY_DEVICE_SPECS.items()
}


def get_device_spec(name: str) -> DeviceSpec:
    """Return a DeviceSpec built from the compatibility DEVICE_SPECS registry.

    Parameters
    ----------
    name : str
        Device identifier that must exist in DEVICE_SPECS.

    Returns
    -------
    DeviceSpec
        Normalized device specification.
    """
    if name not in DEVICE_SPECS:
        # Retry loading registry defaults in case callers imported this module
        # before optional board modules became available.
        DEVICE_SPECS.update(_registry_device_specs())
    if name not in DEVICE_SPECS:
        raise ValueError(f"Unknown device '{name}'. Supported devices: {list(DEVICE_SPECS)}")
    raw = DEVICE_SPECS[name]
    arena_sizes = list(raw.get("arena_sizes", []))
    fqbn = raw.get("fqbn")
    toolchain = "arduino-cli" if fqbn else None
    return DeviceSpec(
        name=name,
        arena_sizes_kb=arena_sizes,
        max_ram_bytes=int(raw.get("max_ram", 0)),
        max_flash_bytes=int(raw.get("max_flash", 0)),
        fqbn=fqbn,
        toolchain=toolchain,
    )


class ArduinoDevice(DeviceInterface):
    """Temporary bridge that uses the Arduino CLI helpers.

    Notes
    -----
    FIXME: Remove this class once per-device microcontroller classes are wired
    into the HIL controller. It exists to keep current behavior intact while
    Phase 2 refactoring is introduced.
    """

    def __init__(
        self,
        device_name: str,
        *,
        serial_port: Optional[str] = None,
        device_options: Optional[Mapping[str, object]] = None,
        spec_override: Optional[DeviceSpec] = None,
    ) -> None:
        """Initialize an Arduino-backed device wrapper.

        Parameters
        ----------
        device_name : str
            Logical device identifier.
        serial_port : str | None, optional
            Serial port used for upload and measurement.
        device_options : Mapping[str, object] | None, optional
            Optional board-option mapping forwarded to subclasses.
        spec_override : DeviceSpec | None, optional
            Explicit spec override used by board-specific wrappers.
        """
        self._device_name = device_name
        self._device_options = dict(device_options or {})
        self._spec = spec_override or get_device_spec(device_name)
        if self._spec.fqbn is None:
            raise ValueError(f"Device '{device_name}' has no Arduino FQBN.")
        self._serial_port = serial_port
        self._board_options = self._resolve_board_options(self._device_options)

    @property
    def spec(self) -> DeviceSpec:
        """Return the device specification metadata."""
        return self._spec

    def runtime_measure_mode(self) -> RuntimeMeasureMode:
        """Return runtime measurement mode for this board wrapper."""
        return "direct_serial"

    def runtime_mode_build_defines(self) -> Dict[str, int]:
        """Return additional compile-time defines for runtime behavior."""
        return {}

    def prepare_for_runtime(self, *, runtime_mode: RuntimeMeasureMode, serial_port: str) -> None:
        """Run optional board-specific setup before upload/measurement."""
        del runtime_mode, serial_port

    def _resolve_board_options(
        self, device_options: Optional[Mapping[str, object]]
    ) -> Optional[Dict[str, str]]:
        """Resolve board options consumed by Arduino CLI compile/upload calls.

        Parameters
        ----------
        device_options : Mapping[str, object] | None
            Raw board option mapping supplied by callers.

        Returns
        -------
        Dict[str, str] | None
            Normalized board options for Arduino CLI, or ``None`` when unused.
        """
        del device_options
        return None

    def compile(
        self,
        *,
        sketch_path: Path,
        arena_kb: int,
        window_size: int,
        num_channels: int,
        build_defines: Optional[Dict[str, int]] = None,
    ) -> CompileResult:
        """Compile firmware using the Arduino CLI toolchain."""
        logger.info(
            "ArduinoDevice.compile: patching constants and compiling sketch at %s",
            sketch_path,
        )
        arduino_base._patch_sketch_constants(sketch_path, arena_kb, window_size, num_channels)
        result = arduino_base.compile_sketch(
            sketch_path=sketch_path,
            fqbn=self._spec.fqbn,
            build_defines=build_defines,
            board_options=self._board_options,
        )
        return CompileResult(
            success=result.success,
            log=result.log,
            flash_bytes=result.flash_bytes,
            ram_bytes=result.ram_bytes,
            overflow_kind=result.overflow_kind,
            build_dir=result.build_dir,
        )

    def upload(
        self,
        *,
        sketch_path: Path,
        build_dir: Optional[Path],
        serial_port: Optional[str],
    ) -> UploadResult:
        """Upload firmware using the Arduino CLI toolchain."""
        use_serial_port = self._serial_port if serial_port is None else serial_port
        if use_serial_port is None:
            raise ValueError("serial_port must be provided for Arduino upload.")
        if build_dir is None:
            raise ValueError("build_dir must be provided for Arduino upload.")
        logger.info("ArduinoDevice.upload: uploading sketch to %s", use_serial_port)
        result = arduino_base.upload_sketch(
            sketch_path=sketch_path,
            fqbn=self._spec.fqbn,
            build_dir=build_dir,
            serial_port=use_serial_port,
            board_options=self._board_options,
        )
        return UploadResult(success=result.success, log=result.log)

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
        """Measure latency/power using the Arduino serial log."""
        use_serial_port = self._serial_port if serial_port is None else serial_port
        if use_serial_port is None:
            raise ValueError("serial_port must be provided for Arduino measurement.")
        logger.info(
            "ArduinoDevice.measure: starting serial measurement (dut=%s, harness=%s)",
            use_serial_port,
            harness_serial_port,
        )
        _default = lambda value, fallback: fallback if value is None else value
        result = arduino_base.measure_serial(
            serial_port=use_serial_port,
            baud_rate=baud_rate,
            serial_timeout_s=serial_timeout_s,
            dut_ready_timeout_s=_default(dut_ready_timeout_s, 5.0),
            harness_serial_port=harness_serial_port,
            harness_fqbn=_default(harness_fqbn, "arduino:mbed_nano:nano33ble"),
            harness_auto_flash=_default(harness_auto_flash, "once"),
            harness_arm_pin=_default(harness_arm_pin, 3),
            harness_trigger_pin=_default(harness_trigger_pin, 2),
            dut_arm_hold_ms=_default(dut_arm_hold_ms, 600),
            harness_stable_low_ms=_default(harness_stable_low_ms, 500),
            harness_ready_timeout_s=_default(harness_ready_timeout_s, 5.0),
            harness_arm_timeout_s=_default(harness_arm_timeout_s, 5.0),
            harness_active_timeout_s=_default(harness_active_timeout_s, 30.0),
            harness_done_timeout_s=_default(harness_done_timeout_s, 5.0),
        )
        return MeasureResult(
            latency_s=result.latency_s,
            arena_error_line=result.arena_error_line,
            serial_log=result.serial_log,
            power_metrics=result.power_metrics,
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
        """Run a compile/upload/measure loop using the Arduino toolchain."""
        sketch_path = Path(dirpath).resolve()
        arena_bytes = arena_kb * 1024
        runtime_mode = self.runtime_measure_mode()
        build_defines = {
            "TINYODOM_HARNESS_ARM_PIN": 3 if harness_arm_pin is None else int(harness_arm_pin),
            "TINYODOM_HARNESS_TRIGGER_PIN": 2 if harness_trigger_pin is None else int(harness_trigger_pin),
            "TINYODOM_DUT_ARM_HOLD_MS": 600 if dut_arm_hold_ms is None else int(dut_arm_hold_ms),
            "TINYODOM_HARNESS_STABLE_LOW_MS": 500
            if harness_stable_low_ms is None
            else int(harness_stable_low_ms),
        }
        build_defines.update(self.runtime_mode_build_defines())
        compile_result = self.compile(
            sketch_path=sketch_path,
            arena_kb=arena_kb,
            window_size=window_size,
            num_channels=num_channels,
            build_defines=build_defines,
        )

        overflow_kind = compile_result.overflow_kind
        if compile_result.ram_bytes is not None and compile_result.ram_bytes > self._spec.max_ram_bytes:
            overflow_kind = "ram"

        if overflow_kind is not None:
            error_code = (
                HIL_ERROR_FLASH_OVERFLOW
                if overflow_kind == "flash"
                else HIL_ERROR_RAM_OVERFLOW
            )
            return DeviceMetrics(
                ram_bytes=compile_result.ram_bytes or -1,
                flash_bytes=compile_result.flash_bytes or -1,
                latency_s=-1.0,
                arena_bytes=arena_bytes,
                error_code=error_code,
            )

        if not compile_result.success:
            return DeviceMetrics(
                ram_bytes=compile_result.ram_bytes or -1,
                flash_bytes=compile_result.flash_bytes or -1,
                latency_s=-1.0,
                arena_bytes=arena_bytes,
                error_code=HIL_ERROR_COMPILE,
            )

        if not run_hil:
            return DeviceMetrics(
                ram_bytes=compile_result.ram_bytes or -1,
                flash_bytes=compile_result.flash_bytes or -1,
                latency_s=-1.0,
                arena_bytes=arena_bytes,
                error_code=HIL_ERROR_OK,
            )

        use_serial_port = self._serial_port if serial_port is None else serial_port
        if use_serial_port is None:
            raise ValueError("serial_port must be provided when running HIL uploads.")
        self.prepare_for_runtime(runtime_mode=runtime_mode, serial_port=use_serial_port)

        _default = lambda value, fallback: fallback if value is None else value
        measure_result: MeasureResult
        if runtime_mode == "harness_only":
            use_harness_port = harness_serial_port
            if not use_harness_port:
                raise ValueError(
                    "harness_serial_port must be provided for harness_only runtime measurement."
                )
            harness_fqbn_value = _default(harness_fqbn, "arduino:mbed_nano:nano33ble")
            harness_auto_flash_value = _default(harness_auto_flash, "once")
            harness_ready_timeout = float(_default(harness_ready_timeout_s, 5.0))
            harness_arm_timeout = float(_default(harness_arm_timeout_s, 5.0))
            harness_active_timeout = float(_default(harness_active_timeout_s, 30.0))
            harness_done_timeout = float(_default(harness_done_timeout_s, 5.0))
            harness_build_defines = {
                "TINYODOM_HARNESS_ARM_PIN": int(_default(harness_arm_pin, 3)),
                "TINYODOM_HARNESS_TRIGGER_PIN": int(_default(harness_trigger_pin, 2)),
                "TINYODOM_DUT_ARM_HOLD_MS": int(_default(dut_arm_hold_ms, 600)),
                "TINYODOM_HARNESS_STABLE_LOW_MS": int(_default(harness_stable_low_ms, 500)),
                "TINYODOM_HARNESS_ARM_TIMEOUT_MS": max(
                    0, int(round(harness_arm_timeout * 1000.0))
                ),
                "TINYODOM_HARNESS_ACTIVE_TIMEOUT_MS": max(
                    0, int(round(harness_active_timeout * 1000.0))
                ),
            }
            try:
                arduino_base.ensure_harness_firmware(
                    harness_serial_port=use_harness_port,
                    harness_fqbn=harness_fqbn_value,
                    harness_auto_flash=harness_auto_flash_value,
                    build_defines=harness_build_defines,
                )
                logger.info(
                    "ArduinoDevice.evaluate[harness_only]: opening harness session before DUT upload "
                    "(harness=%s, dut=%s)",
                    use_harness_port,
                    use_serial_port,
                )
                with serial.Serial(use_harness_port, baudrate=baud_rate, timeout=0.1) as harness:
                    prime_result = hil_protocol.prime_harness_session(
                        harness=harness,
                        harness_ready_timeout_s=harness_ready_timeout,
                        harness_log=[],
                        flush_input=True,
                    )
                    if not prime_result.harness_ready:
                        logger.warning(
                            "Harness session was not ready before DUT upload (error=%s).",
                            prime_result.error,
                        )
                        measure_result = MeasureResult(
                            latency_s=None,
                            arena_error_line=None,
                            serial_log=[f"HARNESS_ERROR: {prime_result.error}"]
                            + [f"HARNESS: {line}" for line in prime_result.harness_log],
                            power_metrics=None,
                        )
                    else:
                        logger.info(
                            "ArduinoDevice.evaluate[harness_only]: uploading DUT while harness session is open."
                        )
                        upload_result = self.upload(
                            sketch_path=sketch_path,
                            build_dir=compile_result.build_dir,
                            serial_port=use_serial_port,
                        )
                        if not upload_result.success:
                            return DeviceMetrics(
                                ram_bytes=compile_result.ram_bytes or -1,
                                flash_bytes=compile_result.flash_bytes or -1,
                                latency_s=-1.0,
                                arena_bytes=arena_bytes,
                                error_code=HIL_ERROR_UPLOAD,
                            )
                        logger.info(
                            "ArduinoDevice.evaluate[harness_only]: DUT upload complete, waiting for harness DONE."
                        )
                        # In Portenta CM4 harness-only runs, host DUT Serial is
                        # not a reliable diagnostics channel, so runtime error
                        # typing depends on harness-visible signals only.
                        measure_result = arduino_base.measure_harness_only_open_session(
                            harness=harness,
                            harness_active_timeout_s=harness_active_timeout,
                            harness_done_timeout_s=harness_done_timeout,
                            harness_log=prime_result.harness_log,
                        )
            except serial.SerialException as exc:
                logger.warning("Harness serial session failed: %s", exc)
                measure_result = MeasureResult(
                    latency_s=None,
                    arena_error_line=None,
                    serial_log=[f"HARNESS_ERROR: {exc}"],
                    power_metrics=None,
                )
            except RuntimeError as exc:
                logger.warning("Harness preparation failed: %s", exc)
                measure_result = MeasureResult(
                    latency_s=None,
                    arena_error_line=None,
                    serial_log=[f"HARNESS_ERROR: {exc}"],
                    power_metrics=None,
                )
        else:
            upload_result = self.upload(
                sketch_path=sketch_path,
                build_dir=compile_result.build_dir,
                serial_port=use_serial_port,
            )
            if not upload_result.success:
                return DeviceMetrics(
                    ram_bytes=compile_result.ram_bytes or -1,
                    flash_bytes=compile_result.flash_bytes or -1,
                    latency_s=-1.0,
                    arena_bytes=arena_bytes,
                    error_code=HIL_ERROR_UPLOAD,
                )

            measure_result = self.measure(
                serial_port=use_serial_port,
                baud_rate=baud_rate,
                serial_timeout_s=serial_timeout_s,
                dut_ready_timeout_s=dut_ready_timeout_s,
                harness_serial_port=harness_serial_port,
                harness_fqbn=harness_fqbn,
                harness_auto_flash=harness_auto_flash,
                harness_arm_pin=harness_arm_pin,
                harness_trigger_pin=harness_trigger_pin,
                dut_arm_hold_ms=dut_arm_hold_ms,
                harness_stable_low_ms=harness_stable_low_ms,
                harness_ready_timeout_s=harness_ready_timeout_s,
                harness_arm_timeout_s=harness_arm_timeout_s,
                harness_active_timeout_s=harness_active_timeout_s,
                harness_done_timeout_s=harness_done_timeout_s,
            )
        if measure_result.latency_s is None:
            if runtime_mode == "harness_only" and measure_result.arena_error_line is None:
                logger.warning(
                    "Harness-only runtime timeout without DUT arena diagnostics; "
                    "classifying as HIL_ERROR_LATENCY."
                )
            retry_hint = arduino_base._compute_retry_hint_bytes(
                arena_bytes, measure_result.arena_error_line
            )
            return DeviceMetrics(
                ram_bytes=compile_result.ram_bytes or -1,
                flash_bytes=compile_result.flash_bytes or -1,
                latency_s=-1.0,
                arena_bytes=arena_bytes,
                error_code=HIL_ERROR_UNDER_SIZED
                if measure_result.arena_error_line
                else HIL_ERROR_LATENCY,
                power_metrics=measure_result.power_metrics,
                retry_hint_bytes=retry_hint,
            )

        return DeviceMetrics(
            ram_bytes=compile_result.ram_bytes or -1,
            flash_bytes=compile_result.flash_bytes or -1,
            latency_s=measure_result.latency_s,
            arena_bytes=arena_bytes,
            error_code=HIL_ERROR_OK,
            power_metrics=measure_result.power_metrics,
        )


# Backward-compatible alias for the bridge class name.
# FIXME: Remove once the migration completes.
ArduinoLegacyDevice = ArduinoDevice

# Merge registry-backed class specs now that ArduinoDevice is fully defined.
DEVICE_SPECS.update(_registry_device_specs())


__all__ = [
    "RuntimeMeasureMode",
    "DeviceSpec",
    "DeviceMetrics",
    "CompileResult",
    "UploadResult",
    "MeasureResult",
    "DeviceInterface",
    "DEVICE_SPECS",
    "get_device_spec",
    "ArduinoDevice",
    "ArduinoLegacyDevice",
]
