from __future__ import annotations

import logging
from pathlib import Path
import shutil
from dataclasses import dataclass
from typing import Mapping, Optional

from ..devices import ArduinoDevice, DeviceSpec, RuntimeMeasureMode
from . import arduino_base

BOARD_NAME = "PORTENTA_H7"
BOARD_FQBN = "arduino:mbed_portenta:envie_m7"
PORTENTA_TOTAL_FLASH_BYTES = 2 * 1024 * 1024
PORTENTA_TOTAL_RAM_BYTES = 1 * 1024 * 1024
VALID_TARGET_CORES = {"cm7", "cm4"}
VALID_SPLITS = {"50_50", "75_25", "100_0"}
# `none` is validated on hardware; `default` and `secure` are allowed so users can
# pass through future/board-package variants without code changes.
VALID_SECURITY = {"none", "default", "secure"}
DEFAULT_SPLIT_BY_CORE = {"cm7": "75_25", "cm4": "50_50"}
# Intentionally 2001ms (not 2000) so CM4 autostart begins strictly after the
# 2s reset/USB-enumeration settling window used by host-side runtime setup.
CM4_AUTOSTART_DELAY_MS = 2001
CM7_BOOT_HELPER_SKETCH = (
    Path(__file__).resolve().parents[3]
    / "sketches"
    / "boot_m4_helper"
)
_CM4_BOOT_HELPER_SIGNATURES: dict[str, str] = {}
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PortentaH7BoardOptions:
    """Normalized Portenta H7 board-option tuple.

    Parameters
    ----------
    target_core : str
        Target core identifier (``cm7`` or ``cm4``).
    split : str
        On-chip memory split policy (for example ``75_25``).
    security : str
        Boot/security profile passed through to Arduino CLI board options.
    """

    target_core: str
    split: str
    security: str

    def to_board_options(self) -> dict[str, str]:
        """Return Arduino CLI ``--board-options`` key/value mapping.

        Returns
        -------
        dict[str, str]
            Mapping consumable by the shared Arduino compile/upload helpers.
        """
        return {
            "target_core": self.target_core,
            "split": self.split,
            "security": self.security,
        }


def _normalize_option(value: Optional[object]) -> Optional[str]:
    """Normalize optional string-like config values.

    Parameters
    ----------
    value : object | None
        Raw option value from config or runtime options.

    Returns
    -------
    str | None
        Lowercased, stripped value or ``None`` when empty.
    """
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return normalized or None


def resolve_portenta_h7_options(
    device_options: Optional[Mapping[str, object]],
) -> PortentaH7BoardOptions:
    """Resolve defaults and validate Portenta board options.

    Parameters
    ----------
    device_options : Mapping[str, object] | None
        Optional board options supplied by config/callers.

    Returns
    -------
    PortentaH7BoardOptions
        Validated option tuple with defaults applied.

    Raises
    ------
    ValueError
        If required options are missing or combinations are invalid.
    """
    options = device_options or {}
    target_core = _normalize_option(options.get("target_core"))
    if target_core is None:
        raise ValueError(
            "PORTENTA_H7 requires device option `target_core` (`cm7` or `cm4`)."
        )
    if target_core not in VALID_TARGET_CORES:
        raise ValueError(
            f"Invalid PORTENTA_H7 target_core '{target_core}'. "
            f"Supported values: {sorted(VALID_TARGET_CORES)}"
        )

    split = _normalize_option(options.get("split")) or DEFAULT_SPLIT_BY_CORE[target_core]
    if split not in VALID_SPLITS:
        raise ValueError(
            f"Invalid PORTENTA_H7 split '{split}'. Supported values: {sorted(VALID_SPLITS)}"
        )

    security = _normalize_option(options.get("security")) or "none"
    if security not in VALID_SECURITY:
        raise ValueError(
            f"Invalid PORTENTA_H7 security '{security}'. "
            f"Supported values: {sorted(VALID_SECURITY)}"
        )

    if target_core == "cm4" and split == "100_0":
        raise ValueError("PORTENTA_H7 does not support target_core=cm4 with split=100_0.")

    return PortentaH7BoardOptions(
        target_core=target_core,
        split=split,
        security=security,
    )


def _split_shares(split: str) -> tuple[float, float]:
    """Return (cm7_share, cm4_share) percentages for a split policy.

    Parameters
    ----------
    split : str
        Split policy token such as ``50_50``.

    Returns
    -------
    tuple[float, float]
        Memory ownership fractions for CM7 and CM4.
    """
    if split == "50_50":
        return 0.5, 0.5
    if split == "75_25":
        return 0.75, 0.25
    if split == "100_0":
        return 1.0, 0.0
    raise ValueError(f"Unsupported split '{split}'.")


def resolve_portenta_h7_limits(
    *,
    target_core: str,
    split: str,
) -> tuple[int, int]:
    """Resolve flash and RAM limits for a specific Portenta split/core tuple.

    Parameters
    ----------
    target_core : str
        Target core identifier (``cm7`` or ``cm4``).
    split : str
        Memory split policy.

    Returns
    -------
    tuple[int, int]
        ``(max_flash_bytes, max_ram_bytes)`` for the chosen core.
    """
    cm7_share, cm4_share = _split_shares(split)
    share = cm7_share if target_core == "cm7" else cm4_share
    max_flash_bytes = int(round(PORTENTA_TOTAL_FLASH_BYTES * share))
    max_ram_bytes = int(round(PORTENTA_TOTAL_RAM_BYTES * share))
    return max_flash_bytes, max_ram_bytes


def _portenta_h7_arena_candidates(max_ram_bytes: int) -> list[int]:
    """Generate default arena candidate sizes (KiB) for Portenta sweeps.

    Parameters
    ----------
    max_ram_bytes : int
        Runtime RAM ceiling for the selected core/split profile.

    Returns
    -------
    list[int]
        Arena sweep candidates expressed in KiB.
    """
    upper_kb = max(32, int((max_ram_bytes * 0.75) // 1024))
    baseline = [32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768]
    candidates = [value for value in baseline if value <= upper_kb]
    if not candidates:
        candidates = [32]
    return candidates


def build_portenta_h7_spec(
    *,
    options: Optional[PortentaH7BoardOptions] = None,
) -> DeviceSpec:
    """Construct a Portenta ``DeviceSpec`` for a resolved core/split policy.

    Parameters
    ----------
    options : PortentaH7BoardOptions | None, optional
        Resolved board options. When omitted, uses default ``cm7`` options.

    Returns
    -------
    DeviceSpec
        Device limits and arena sweep for the selected profile.
    """
    resolved_options = options or resolve_portenta_h7_options({"target_core": "cm7"})
    max_flash_bytes, max_ram_bytes = resolve_portenta_h7_limits(
        target_core=resolved_options.target_core,
        split=resolved_options.split,
    )
    return DeviceSpec(
        name=BOARD_NAME,
        arena_sizes_kb=_portenta_h7_arena_candidates(max_ram_bytes),
        max_ram_bytes=max_ram_bytes,
        max_flash_bytes=max_flash_bytes,
        fqbn=BOARD_FQBN,
        toolchain="arduino-cli",
    )


BOARD_DEFAULT_SPEC = build_portenta_h7_spec()


class ArduinoPortentaH7Device(ArduinoDevice):
    """Concrete Portenta H7 wrapper with target-core board options.

    Parameters
    ----------
    serial_port : str | None, optional
        Target serial port for upload and measurement.
    device_options : dict[str, str] | None, optional
        Portenta option map including ``target_core``, optional ``split``, and
        optional ``security``.
    """

    def __init__(
        self,
        *,
        serial_port: Optional[str] = None,
        device_options: Optional[dict[str, str]] = None,
    ) -> None:
        """Initialize a Portenta device wrapper with validated options.

        Parameters
        ----------
        serial_port : str | None, optional
            Target serial port used for upload/measurement.
        device_options : dict[str, str] | None, optional
            Portenta option map (``target_core``, ``split``, ``security``).
        """
        self._resolved_options = resolve_portenta_h7_options(device_options)
        spec = build_portenta_h7_spec(options=self._resolved_options)
        super().__init__(
            BOARD_NAME,
            serial_port=serial_port,
            device_options=self._resolved_options.to_board_options(),
            spec_override=spec,
        )

    @property
    def resolved_options(self) -> PortentaH7BoardOptions:
        """Expose validated options used by this device instance.

        Returns
        -------
        PortentaH7BoardOptions
            Canonicalized option tuple.
        """
        return self._resolved_options

    def _resolve_board_options(
        self, device_options: Optional[Mapping[str, object]]
    ) -> Optional[dict[str, str]]:
        """Translate resolved options to Arduino CLI board options.

        Parameters
        ----------
        device_options : Mapping[str, object] | None
            Unused. Board options are resolved during initialization.

        Returns
        -------
        dict[str, str] | None
            Normalized board options for Arduino CLI.
        """
        del device_options
        return self._resolved_options.to_board_options()

    def runtime_measure_mode(self) -> RuntimeMeasureMode:
        """Return runtime measurement mode for the selected target core."""
        if self._resolved_options.target_core == "cm4":
            # CM4 runtime telemetry is measured by the external harness because
            # host-visible USB CDC Serial is not reliable for this target core.
            return "harness_only"
        return "direct_serial"

    def runtime_mode_build_defines(self) -> dict[str, int]:
        """Return compile-time runtime defines for CM4 autostart mode."""
        if self._resolved_options.target_core != "cm4":
            return {}
        return {
            "TINYODOM_AUTOSTART": 1,
            "TINYODOM_AUTOSTART_DELAY_MS": CM4_AUTOSTART_DELAY_MS,
            "TINYODOM_SKIP_SERIAL_WAIT": 1,
        }

    def _cm7_boot_helper_board_options(self) -> dict[str, str]:
        """Return board options for CM7 boot-assist sketch upload."""
        return {
            "target_core": "cm7",
            "split": self._resolved_options.split,
            "security": self._resolved_options.security,
        }

    def _ensure_cm7_boot_helper(self, *, serial_port: str) -> None:
        """Ensure a CM7 helper sketch is running so M4 can boot reliably."""
        helper_signature = ",".join(
            f"{key}={value}" for key, value in sorted(self._cm7_boot_helper_board_options().items())
        )
        if _CM4_BOOT_HELPER_SIGNATURES.get(serial_port) == helper_signature:
            return
        if not CM7_BOOT_HELPER_SKETCH.exists():
            raise RuntimeError(f"Missing Portenta CM7 boot helper sketch: {CM7_BOOT_HELPER_SKETCH}")

        logger.info(
            "Portenta CM4 runtime: uploading CM7 boot helper (port=%s, options=%s)",
            serial_port,
            helper_signature,
        )
        compile_result = arduino_base.compile_sketch(
            sketch_path=CM7_BOOT_HELPER_SKETCH,
            fqbn=BOARD_FQBN,
            board_options=self._cm7_boot_helper_board_options(),
        )
        if not compile_result.success or compile_result.build_dir is None:
            raise RuntimeError(
                "Failed to compile Portenta CM7 boot helper sketch.\n"
                f"{compile_result.log.strip()}"
            )
        upload_result = arduino_base.upload_sketch(
            sketch_path=CM7_BOOT_HELPER_SKETCH,
            fqbn=BOARD_FQBN,
            build_dir=compile_result.build_dir,
            serial_port=serial_port,
            board_options=self._cm7_boot_helper_board_options(),
        )
        if not upload_result.success:
            logger.warning(
                "Portenta CM4 runtime: CM7 boot helper upload failed once; retrying with clean rebuild."
            )
            shutil.rmtree(compile_result.build_dir, ignore_errors=True)
            retry_compile = arduino_base.compile_sketch(
                sketch_path=CM7_BOOT_HELPER_SKETCH,
                fqbn=BOARD_FQBN,
                board_options=self._cm7_boot_helper_board_options(),
            )
            if not retry_compile.success or retry_compile.build_dir is None:
                raise RuntimeError(
                    "Failed to compile Portenta CM7 boot helper sketch during retry.\n"
                    f"{retry_compile.log.strip()}"
                )
            retry_upload = arduino_base.upload_sketch(
                sketch_path=CM7_BOOT_HELPER_SKETCH,
                fqbn=BOARD_FQBN,
                build_dir=retry_compile.build_dir,
                serial_port=serial_port,
                board_options=self._cm7_boot_helper_board_options(),
            )
            if not retry_upload.success:
                raise RuntimeError(
                    "Failed to upload Portenta CM7 boot helper sketch.\n"
                    f"{retry_upload.log.strip()}"
                )
        _CM4_BOOT_HELPER_SIGNATURES[serial_port] = helper_signature

    def prepare_for_runtime(self, *, runtime_mode: RuntimeMeasureMode, serial_port: str) -> None:
        """Prepare dual-core runtime prerequisites before DUT upload."""
        if runtime_mode == "harness_only" and self._resolved_options.target_core == "cm4":
            logger.info(
                "Portenta CM4 runtime uses harness-only telemetry; DUT Serial diagnostics are limited on host USB."
            )
            self._ensure_cm7_boot_helper(serial_port=serial_port)
