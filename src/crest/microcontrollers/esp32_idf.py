# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""ESP-IDF backend for the Freenove ESP32-S3 WROOM board."""

from __future__ import annotations

from dataclasses import dataclass, replace
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess

import serial
import time
from typing import Any, Mapping, Optional

from ..errors import HIL_ERROR_INTEGRITY
from ..devices import (
    ArduinoDevice,
    CandidatePrepareRequest,
    CompileResult,
    DeviceSpec,
    MeasureResult,
    UploadResult,
)


BOARD_NAME = "FREENOVE_ESP32_S3_WROOM"
DEFAULT_PROJECT_ROOT = Path("firmware/esp32/crest_esp32_tflm")


@dataclass(frozen=True)
class ESP32IDFOptions:
    """Resolved ESP-IDF workspace and board options."""

    project_root: Path = DEFAULT_PROJECT_ROOT
    build_dir: Path = DEFAULT_PROJECT_ROOT / "build"
    idf_py: str = "idf.py"
    target: str = "esp32s3"
    model_output_dir: Path = Path("main")
    model_source_name: str = "model_data.cc"
    model_header_name: str = "model_data.h"
    model_array_name: str = "g_model"
    arena_sizes_kb: tuple[int, ...] = (2048,)
    max_internal_ram_bytes: int = 512 * 1024
    max_flash_bytes: int = 4 * 1024 * 1024
    psram_bytes: int = 8 * 1024 * 1024
    harness_trigger_gpio: int = 21
    harness_arm_gpio: int = 47
    optimization_profile: str = "debug"
    esp_nn_enabled: bool = True
    runtime_mode: str = "back_to_back"
    cpu_clock_mhz: int = 240
    fixture_npz_path: Path | None = None
    fixture_manifest_path: Path | None = None
    warmup_runs: int = 1
    latency_aggregation: str = "median"
    build_timeout_s: float = 900.0
    upload_timeout_s: float = 120.0


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive integer.")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{field} must be a positive integer.")
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a positive integer.") from exc
    if resolved <= 0:
        raise ValueError(f"{field} must be a positive integer.")
    return resolved


def _positive_float(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite positive number.")
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite positive number.") from exc
    if not math.isfinite(resolved) or resolved <= 0:
        raise ValueError(f"{field} must be a finite positive number.")
    return resolved


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a nonnegative integer.")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{field} must be a nonnegative integer.")
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a nonnegative integer.") from exc
    if resolved < 0:
        raise ValueError(f"{field} must be a nonnegative integer.")
    return resolved


def resolve_esp32_idf_options(
    device_options: Optional[Mapping[str, object]],
) -> ESP32IDFOptions:
    """Validate and normalize ESP32 backend options."""

    raw = dict(device_options or {})
    internal_names = {"model_output_dir", "model_source_name", "model_header_name", "model_array_name"}
    unsupported_internal = sorted(internal_names.intersection(raw))
    if unsupported_internal:
        raise ValueError(
            "ESP32 generated firmware names are fixed and are not configurable: "
            + ", ".join(unsupported_internal)
        )
    project_root = Path(raw.get("project_root", DEFAULT_PROJECT_ROOT)).expanduser()
    build_dir_raw = raw.get("build_dir")
    build_dir = (
        Path(build_dir_raw).expanduser()
        if build_dir_raw not in (None, "")
        else project_root / "build"
    )
    arena_raw = raw.get("arena_sizes_kb", (2048,))
    if isinstance(arena_raw, (str, bytes)):
        raise ValueError("device.esp32.arena_sizes_kb must be a sequence of positive integers.")
    try:
        arena_sizes = tuple(
            _positive_int(value, field="device.esp32.arena_sizes_kb")
            for value in arena_raw  # type: ignore[union-attr]
        )
    except TypeError as exc:
        raise ValueError(
            "device.esp32.arena_sizes_kb must be a sequence of positive integers."
        ) from exc
    if not arena_sizes:
        raise ValueError("device.esp32.arena_sizes_kb must not be empty.")
    target = str(raw.get("target", "esp32s3")).strip().lower()
    if target != "esp32s3":
        raise ValueError("The Freenove backend supports only the ESP-IDF esp32s3 target.")
    profile = str(raw.get("optimization_profile", "debug")).strip().lower()
    if profile != "debug":
        raise ValueError("The verified ESP32 backend currently supports only the debug optimization profile.")
    esp_nn = raw.get("esp_nn_enabled", True)
    runtime_mode = str(raw.get("runtime_mode", "back_to_back")).strip().lower()
    if runtime_mode != "back_to_back":
        raise ValueError("The ESP32 backend supports only back_to_back runtime_mode.")
    cpu_clock_mhz = _positive_int(raw.get("cpu_clock_mhz", 240), field="device.esp32.cpu_clock_mhz")
    if cpu_clock_mhz != 240:
        raise ValueError("The verified ESP32 backend supports only a 240 MHz CPU clock.")
    if esp_nn is not True:
        raise ValueError("The verified ESP32 backend requires ESP-NN enabled.")
    fixture_npz_raw = raw.get("fixture_npz_path")
    fixture_manifest_raw = raw.get("fixture_manifest_path")
    if (fixture_npz_raw is None) != (fixture_manifest_raw is None):
        raise ValueError("ESP32 fixture_npz_path and fixture_manifest_path must be configured together.")
    latency_aggregation = str(raw.get("latency_aggregation", "median")).strip().lower()
    if latency_aggregation not in {"mean", "median"}:
        raise ValueError("device.esp32.latency_aggregation must be mean or median.")
    build_timeout_s = _positive_float(
        raw.get("build_timeout_s", 900.0), field="device.esp32.build_timeout_s"
    )
    upload_timeout_s = _positive_float(
        raw.get("upload_timeout_s", 120.0), field="device.esp32.upload_timeout_s"
    )
    max_internal_ram_bytes = _positive_int(
        raw.get("max_internal_ram_bytes", 512 * 1024), field="device.esp32.max_internal_ram_bytes"
    )
    max_flash_bytes = _positive_int(
        raw.get("max_flash_bytes", 4 * 1024 * 1024), field="device.esp32.max_flash_bytes"
    )
    psram_bytes = _positive_int(
        raw.get("psram_bytes", 8 * 1024 * 1024), field="device.esp32.psram_bytes"
    )
    if max_internal_ram_bytes > 512 * 1024:
        raise ValueError("ESP32 internal RAM limit cannot exceed the verified 512 KiB capacity.")
    if max_flash_bytes > 4 * 1024 * 1024:
        raise ValueError("ESP32 flash limit cannot exceed the configured 4 MiB app partition.")
    if psram_bytes > 8 * 1024 * 1024:
        raise ValueError("ESP32 PSRAM limit cannot exceed the verified 8 MiB capacity.")
    if max(arena_sizes) * 1024 > psram_bytes:
        raise ValueError("ESP32 tensor arena must fit within the configured PSRAM capacity.")
    return ESP32IDFOptions(
        project_root=project_root,
        build_dir=build_dir,
        idf_py=str(raw.get("idf_py", "idf.py")),
        target=target,
        model_output_dir=Path("main"),
        model_source_name="model_data.cc",
        model_header_name="model_data.h",
        model_array_name="g_model",
        arena_sizes_kb=arena_sizes,
        max_internal_ram_bytes=max_internal_ram_bytes,
        max_flash_bytes=max_flash_bytes,
        psram_bytes=psram_bytes,
        harness_trigger_gpio=_positive_int(
            raw.get("harness_trigger_gpio", 21),
            field="device.esp32.harness_trigger_gpio",
        ),
        harness_arm_gpio=_positive_int(
            raw.get("harness_arm_gpio", 47),
            field="device.esp32.harness_arm_gpio",
        ),
        optimization_profile=profile,
        esp_nn_enabled=esp_nn,
        runtime_mode=runtime_mode,
        cpu_clock_mhz=cpu_clock_mhz,
        fixture_npz_path=None if fixture_npz_raw is None else Path(fixture_npz_raw).expanduser(),
        fixture_manifest_path=None if fixture_manifest_raw is None else Path(fixture_manifest_raw).expanduser(),
        warmup_runs=_nonnegative_int(raw.get("warmup_runs", 1), field="device.esp32.warmup_runs"),
        latency_aggregation=latency_aggregation,
        build_timeout_s=build_timeout_s,
        upload_timeout_s=upload_timeout_s,
    )


def build_esp32_idf_spec(options: ESP32IDFOptions | None = None) -> DeviceSpec:
    """Build the shared device specification for an ESP32 configuration."""

    resolved = options or ESP32IDFOptions()
    return DeviceSpec(
        name=BOARD_NAME,
        arena_sizes_kb=resolved.arena_sizes_kb,
        max_ram_bytes=resolved.max_internal_ram_bytes,
        max_flash_bytes=resolved.max_flash_bytes,
        toolchain="esp-idf",
    )


BOARD_DEFAULT_SPEC = build_esp32_idf_spec()


_FLASH_PATTERNS = (
    re.compile(r"Total image size:\s*(\d+)\s*bytes", re.IGNORECASE),
    re.compile(r"Project app size:\s*(\d+)\s*bytes", re.IGNORECASE),
)
_RAM_PATTERNS = (
    re.compile(r"^\s*DIRAM\s+[^0-9\n]*(\d+)\s+", re.IGNORECASE | re.MULTILINE),
    re.compile(r"(?:Used static DRAM|DRAM used):\s*(\d+)\s*bytes", re.IGNORECASE),
)
_IDF55_IRAM_RE = re.compile(r"^\s*[^A-Za-z0-9\n]*IRAM\s+[^0-9\n]*(\d+)\s+", re.MULTILINE)
_IDF55_DRAM_RE = re.compile(r"^\s*[^A-Za-z0-9\n]*DRAM\s+[^0-9\n]*(\d+)\s+", re.MULTILINE)
_IDF55_DIRAM_RE = re.compile(r"^\s*[^A-Za-z0-9\n]*DIRAM\s+[^0-9\n]*(\d+)\s+", re.MULTILINE)


def parse_idf_size_output(output: str) -> tuple[int | None, int | None]:
    """Return ESP-IDF image and internal-static-RAM usage from build output."""

    def _first(patterns: tuple[re.Pattern[str], ...]) -> int | None:
        for pattern in patterns:
            match = pattern.search(output)
            if match:
                return int(match.group(1))
        return None

    ram_bytes = _first(_RAM_PATTERNS)
    iram_match = _IDF55_IRAM_RE.search(output)
    dram_match = _IDF55_DRAM_RE.search(output)
    diram_match = _IDF55_DIRAM_RE.search(output)
    data_ram_match = diram_match or dram_match
    if iram_match and data_ram_match:
        ram_bytes = int(iram_match.group(1)) + int(data_ram_match.group(1))
    return _first(_FLASH_PATTERNS), ram_bytes


def _acquire_lock(lock_file: Any, *, label: str, timeout_s: float = 60.0) -> None:
    """Acquire an advisory lock with a bounded wait."""

    deadline = time.monotonic() + timeout_s
    while True:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for the ESP32 {label} lock.")
            time.sleep(0.1)


def classify_idf_build_failure(output: str) -> str | None:
    """Classify known ESP-IDF linker capacity failures."""

    lowered = output.lower()
    if "does not fit" in lowered and ("flash" in lowered or "partition" in lowered):
        return "flash"
    if "region `dram" in lowered or "dram segment data does not fit" in lowered:
        return "ram"
    return None


class ESP32IDFDevice(ArduinoDevice):
    """Concrete ESP32-S3 backend using an incremental ESP-IDF workspace."""

    def __init__(
        self,
        *,
        serial_port: Optional[str] = None,
        device_options: Optional[dict[str, Any]] = None,
    ) -> None:
        self._options = resolve_esp32_idf_options(device_options)
        self._serial_port = serial_port
        self._spec = build_esp32_idf_spec(self._options)
        self._staged_model_sha256: str | None = None
        self._prepared_roots: set[Path] = set()
        self._active_manifest: dict[str, Any] | None = None
        self._active_staged_root: Path | None = None

    @property
    def spec(self) -> DeviceSpec:
        return self._spec

    def supports_energy_measurement(self) -> bool:
        return False

    def runtime_mode_build_defines(self) -> dict[str, int]:
        return {
            "CREST_HARNESS_TRIGGER_PIN": self._options.harness_trigger_gpio,
            "CREST_HARNESS_ARM_PIN": self._options.harness_arm_gpio,
        }

    def prepare_candidate(self, *, request: CandidatePrepareRequest) -> Path:
        """Export and embed a candidate in the reusable ESP-IDF workspace."""

        if request.model is None:
            raise ValueError("ESP32 candidate preparation requires a built Keras model.")
        if request.quantization_mode == "int8_ptq" and request.calibration_split is None:
            raise ValueError("ESP32 candidate preparation requires calibration data for int8_ptq.")
        project_root = self._options.project_root.resolve()
        if not (project_root / "CMakeLists.txt").is_file():
            raise ValueError(
                f"ESP32 ESP-IDF project was not found at {project_root}; "
                "configure device.esp32.project_root."
            )
        from ..hardware import convert_to_cpp_model, convert_to_tflite_model

        request.tflite_model_path.parent.mkdir(parents=True, exist_ok=True)
        convert_to_tflite_model(
            model=request.model,
            training_data=(
                None
                if request.calibration_split is None
                else request.calibration_split.inputs
            ),
            quantization_mode=request.quantization_mode,
            output_name=request.tflite_model_path,
        )
        model_bytes = request.tflite_model_path.read_bytes()
        model_sha256 = hashlib.sha256(model_bytes).hexdigest()
        candidate_label = re.sub(r"[^A-Za-z0-9_.-]+", "-", request.model_variant).strip(".-")
        if not candidate_label:
            raise ValueError("ESP32 model_variant must contain a filename-safe character.")
        candidate_root = request.artifact_root.resolve() / "esp32" / candidate_label
        generated_dir = candidate_root / self._options.model_output_dir
        generated_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(request.tflite_model_path, generated_dir / "crest_candidate.tflite")
        convert_to_cpp_model(
            request.tflite_model_path,
            generated_dir,
            array_name=self._options.model_array_name,
            source_name=self._options.model_source_name,
            header_name=self._options.model_header_name,
        )
        if self._options.fixture_npz_path is None or self._options.fixture_manifest_path is None:
            raise ValueError("ESP32 prepared fixtures must be configured for candidate measurement.")
        from .esp32_fixtures import stage_fixture_header
        fixture_evidence = stage_fixture_header(
            fixture_npz_path=self._options.fixture_npz_path,
            fixture_manifest_path=self._options.fixture_manifest_path,
            tflite_model_path=request.tflite_model_path,
            output_path=generated_dir / "fixtures.h",
        )
        device_config = request.config.device
        if isinstance(device_config, Mapping):
            measured_runs_raw = device_config.get("measured_inference_runs", 10)
        else:
            measured_runs_raw = vars(device_config).get("measured_inference_runs", 10)
        measured_runs = _positive_int(
            measured_runs_raw, field="device.measured_inference_runs"
        )
        manifest = {
            "tflite_sha256": model_sha256,
            "tflite_size_bytes": len(model_bytes),
            "array_name": self._options.model_array_name,
            "warmup_runs_per_fixture": self._options.warmup_runs,
            "measured_runs_per_fixture": measured_runs,
            "latency_aggregation": self._options.latency_aggregation,
            **fixture_evidence,
        }
        manifest_header = (
            "#pragma once\n"
            f"static constexpr const char *kCrestModelSha256 = \"{model_sha256}\";\n"
            f"static constexpr const char *kCrestFixtureSha256 = \"{fixture_evidence['fixture_sha256']}\";\n"
        )
        (generated_dir / "crest_manifest.h").write_text(manifest_header, encoding="utf-8")
        (generated_dir / "crest_model_manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        self._staged_model_sha256 = model_sha256
        self._prepared_roots.add(candidate_root.resolve())
        return candidate_root

    def cleanup_prepared_candidate(self, prepared_dir: Path | None) -> None:
        """Release tracking while preserving per-request build and measurement evidence."""
        if prepared_dir is None:
            return
        resolved = Path(prepared_dir).resolve()
        self._prepared_roots.discard(resolved)

    def set_input_mode(self, input_mode: str, **kwargs: Any) -> Path | None:
        """Reject Arduino sketch input-mode mutation for the ESP-IDF backend."""
        del kwargs
        if str(input_mode).lower() not in {"representative", "real"}:
            raise ValueError(f"Unsupported ESP32 input mode: {input_mode}")
        return None

    def evaluate(self, *, dirpath: str | Path, **kwargs: Any):
        """Serialize staging, incremental build, app upload, and measurement."""
        staged_root = Path(dirpath).resolve()
        manifest_path = staged_root / self._options.model_output_dir / "crest_model_manifest.json"
        if not manifest_path.is_file():
            raise ValueError("ESP32 evaluate requires a staged candidate manifest.")
        project_root = self._options.project_root.resolve()
        generated_source = staged_root / self._options.model_output_dir
        generated_target = project_root / self._options.model_output_dir
        lock_path = project_root / ".crest-workspace.lock"
        run_hil = bool(kwargs.get("run_hil", True))
        if kwargs.get("harness_serial_port") not in (None, ""):
            raise ValueError(
                "ESP32 classifier measurement is DUT-serial only; harness energy measurement is unsupported."
            )
        serial_value = kwargs.get("serial_port") or self._serial_port or "compile-only"
        serial_identity = str(Path(str(serial_value)).resolve()) if run_hil else "compile-only"
        board_lock_path = Path("/tmp") / (
            "crest-esp32-board-" + hashlib.sha256(serial_identity.encode("utf-8")).hexdigest()[:16] + ".lock"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        with board_lock_path.open("a+", encoding="utf-8") as board_lock, lock_path.open("a+", encoding="utf-8") as lock_file:
            _acquire_lock(board_lock, label="board")
            _acquire_lock(lock_file, label="workspace")
            self._active_manifest = manifest
            self._active_staged_root = staged_root
            try:
                generated_target.mkdir(parents=True, exist_ok=True)
                for filename in (
                    self._options.model_source_name,
                    self._options.model_header_name,
                    "crest_model_manifest.json",
                    "fixtures.h",
                    "crest_manifest.h",
                    "crest_candidate.tflite",
                ):
                    shutil.copy2(generated_source / filename, generated_target / filename)
                metrics = super().evaluate(dirpath=project_root, **kwargs)
                instrumentation_valid = bool(
                    (metrics.power_metrics or {}).get("measurement_instrumentation_valid", False)
                )
                if run_hil and metrics.latency_s >= 0.0 and not instrumentation_valid:
                    metrics = replace(metrics, error_code=HIL_ERROR_INTEGRITY)
                return metrics
            finally:
                self._active_manifest = None
                self._active_staged_root = None

    def measure(self, **kwargs: Any) -> MeasureResult:
        """Validate runtime identity and sample evidence from serial telemetry."""
        manifest = self._active_manifest or {}
        try:
            result = super().measure(**kwargs)
        except (serial.SerialException, OSError) as exc:
            return MeasureResult(
                latency_s=None, arena_error_line=None,
                serial_log=[f"ESP32_SERIAL_ERROR: {exc}"],
                power_metrics={"measurement_instrumentation_valid": False},
            )
        from .esp32_measurement import validate_measurement_log
        validated, diagnostics = validate_measurement_log(
            lines=result.serial_log,
            manifest=manifest,
            latency_s=result.latency_s,
            psram_bytes=self._options.psram_bytes,
        )
        evidence = dict(result.power_metrics or {})
        evidence.update(validated)
        if self._active_staged_root is not None:
            diagnostics["serial_log"] = list(result.serial_log)
            measurement_record = self._active_staged_root / "esp32_measurement.json"
            measurement_record.write_text(
                json.dumps(diagnostics, sort_keys=True, indent=2) + "\n", encoding="utf-8"
            )
            evidence["measurement_record"] = str(measurement_record.resolve())
        return MeasureResult(
            latency_s=result.latency_s,
            arena_error_line=result.arena_error_line,
            serial_log=result.serial_log,
            power_metrics=evidence,
        )

    def compile(
        self,
        *,
        sketch_path: Path,
        arena_kb: int,
        window_size: int,
        num_channels: int,
        build_defines: Optional[dict[str, int]] = None,
    ) -> CompileResult:
        """Build the project with ESP-IDF while preserving its build cache."""

        del window_size, num_channels
        project_root = Path(sketch_path).resolve()
        if project_root != self._options.project_root.resolve():
            raise ValueError("ESP32 compile must use the configured incremental project_root.")
        build_dir = self._options.build_dir.resolve()
        model_path = project_root / self._options.model_output_dir / "crest_candidate.tflite"
        source_path = project_root / self._options.model_output_dir / self._options.model_source_name
        if self._active_staged_root is not None:
            from .esp32_artifacts import verify_generated_model_source
            try:
                generated_sha256 = verify_generated_model_source(model_path=model_path, source_path=source_path)
            except (OSError, ValueError) as exc:
                return CompileResult(False, str(exc), None, None, None, build_dir=build_dir, arena_bytes=int(arena_kb) * 1024)
            if generated_sha256 != str((self._active_manifest or {}).get("tflite_sha256", "")):
                return CompileResult(False, "Staged ESP32 model hash does not match its manifest.", None, None, None, build_dir=build_dir, arena_bytes=int(arena_kb) * 1024)
        build_deadline = time.monotonic() + self._options.build_timeout_s
        sdkconfig_path = project_root / "sdkconfig"
        try:
            existing_config = sdkconfig_path.read_text(encoding="utf-8") if sdkconfig_path.is_file() else ""
        except OSError as exc:
            return CompileResult(False, f"Failed to read ESP-IDF configuration: {exc}", None, None, None, build_dir=build_dir, arena_bytes=int(arena_kb) * 1024)
        expected_target = f'CONFIG_IDF_TARGET="{self._options.target}"'
        if existing_config and expected_target not in existing_config:
            return CompileResult(False, "Existing ESP-IDF sdkconfig targets a different chip; clear the managed ESP32 build workspace before retrying.", None, None, None, build_dir=build_dir, arena_bytes=int(arena_kb) * 1024)
        if not existing_config:
            try:
                target_result = subprocess.run(
                    [self._options.idf_py, "-C", str(project_root), "-B", str(build_dir),
                     "set-target", self._options.target],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False,
                    timeout=max(0.001, build_deadline - time.monotonic()),
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                return CompileResult(False, f"Failed to configure ESP-IDF target: {exc}", None, None, None, build_dir=build_dir, arena_bytes=int(arena_kb) * 1024)
            if target_result.returncode != 0:
                return CompileResult(False, target_result.stdout or "ESP-IDF target configuration failed.", None, None, None, build_dir=build_dir, arena_bytes=int(arena_kb) * 1024)
        try:
            build_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return CompileResult(False, f"Failed to create ESP-IDF build directory: {exc}", None, None, None, build_dir=build_dir, arena_bytes=int(arena_kb) * 1024)
        defines = dict(build_defines or {})
        defines["CREST_TENSOR_ARENA_BYTES"] = int(arena_kb) * 1024
        defines["CREST_TENSOR_ARENA_IN_PSRAM"] = 1
        defines["CREST_ESP_NN_ENABLED"] = int(self._options.esp_nn_enabled)
        defines["CREST_WARMUP_RUNS"] = self._options.warmup_runs
        defines["CREST_AGGREGATE_MEDIAN"] = int(self._options.latency_aggregation == "median")
        defines["CREST_OPTIMIZE_FOR_SIZE"] = int(
            self._options.optimization_profile == "size"
        )
        command = [
            self._options.idf_py,
            "-C",
            str(project_root),
            "-B",
            str(build_dir),
        ]
        command.extend(f"-D{key}={int(value)}" for key, value in sorted(defines.items()))
        command.extend(("build", "size"))
        try:
            completed = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                timeout=max(0.001, build_deadline - time.monotonic()),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return CompileResult(
                success=False,
                log=f"Failed to execute ESP-IDF build: {exc}",
                flash_bytes=None,
                ram_bytes=None,
                overflow_kind=None,
                build_dir=build_dir,
                arena_bytes=int(arena_kb) * 1024,
            )
        log = completed.stdout or ""
        flash_bytes, ram_bytes = parse_idf_size_output(log)
        if completed.returncode == 0 and self._active_staged_root is not None:
            from .esp32_artifacts import resolve_built_application, verify_application_contains_model
            try:
                sdkconfig = (project_root / "sdkconfig").read_text(encoding="utf-8")
                required_config = (
                    'CONFIG_IDF_TARGET="esp32s3"',
                    "CONFIG_ESPTOOLPY_FLASHSIZE_8MB=y",
                    "CONFIG_COMPILER_OPTIMIZATION_DEBUG=y",
                    "CONFIG_SPIRAM_MODE_OCT=y",
                    "CONFIG_SPIRAM_SPEED_40M=y",
                    "CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ_240=y",
                    'CONFIG_PARTITION_TABLE_CUSTOM_FILENAME="partitions-4m.csv"',
                )
                missing_config = [line for line in required_config if line not in sdkconfig]
                if missing_config:
                    raise ValueError("Resolved sdkconfig lacks verified settings: " + ", ".join(missing_config))
                description = json.loads((build_dir / "project_description.json").read_text(encoding="utf-8"))
                if description.get("target") != "esp32s3" or description.get("git_revision") != "v5.5.1":
                    raise ValueError("ESP-IDF target/version does not match esp32s3 v5.5.1.")
                if "espressif__esp-nn" not in description.get("build_components", []):
                    raise ValueError("Resolved ESP-IDF build does not include ESP-NN.")
                app_path = resolve_built_application(build_dir)
                evidence_app = self._active_staged_root / "esp32_application.bin"
                shutil.copy2(app_path, evidence_app)
                identity = verify_application_contains_model(app_path=evidence_app, model_path=model_path)
                firmware_files = (
                    project_root / "CMakeLists.txt", project_root / "sdkconfig.defaults",
                    project_root / "partitions-4m.csv", project_root / "main" / "CMakeLists.txt",
                    project_root / "main" / "runner.cc", project_root / "main" / "idf_component.yml",
                )
                firmware_hash = hashlib.sha256()
                for firmware_file in firmware_files:
                    firmware_hash.update(firmware_file.relative_to(project_root).as_posix().encode("utf-8"))
                    firmware_hash.update(b"\0")
                    firmware_hash.update(firmware_file.read_bytes())
                identity.update({
                    "tflite_sha256": generated_sha256,
                    "arena_bytes_reserved": int(arena_kb) * 1024,
                    "optimization_profile": self._options.optimization_profile,
                    "esp_nn_enabled": self._options.esp_nn_enabled,
                    "cpu_clock_mhz": self._options.cpu_clock_mhz,
                    "idf_version": description["git_revision"],
                    "idf_target": description["target"],
                    "sdkconfig_sha256": hashlib.sha256(sdkconfig.encode("utf-8")).hexdigest(),
                    "firmware_sha256": firmware_hash.hexdigest(),
                })
                (self._active_staged_root / "esp32_build_identity.json").write_text(
                    json.dumps(identity, sort_keys=True, indent=2) + "\n", encoding="utf-8"
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                return CompileResult(False, f"{log}\nESP32 artifact identity failure: {exc}", flash_bytes, ram_bytes, None, build_dir=build_dir, arena_bytes=int(arena_kb) * 1024)
        return CompileResult(
            success=completed.returncode == 0,
            log=log,
            flash_bytes=flash_bytes,
            ram_bytes=ram_bytes,
            overflow_kind=(
                "flash" if flash_bytes is not None and flash_bytes > self._options.max_flash_bytes
                else classify_idf_build_failure(log)
            ),
            build_dir=build_dir,
            arena_bytes=int(arena_kb) * 1024,
        )

    def upload(
        self,
        *,
        sketch_path: Path,
        build_dir: Optional[Path],
        serial_port: Optional[str],
    ) -> UploadResult:
        """Flash the already-built ESP-IDF image."""

        use_port = self._serial_port if serial_port is None else serial_port
        if not use_port:
            raise ValueError("serial_port must be provided for ESP32 upload.")
        if build_dir is None:
            raise ValueError("build_dir must be provided for ESP32 upload.")
        command = [
            self._options.idf_py,
            "-C",
            str(Path(sketch_path).resolve()),
            "-B",
            str(Path(build_dir).resolve()),
            "-p",
            str(use_port),
            "app-flash",
        ]
        try:
            completed = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                timeout=self._options.upload_timeout_s,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return UploadResult(success=False, log=f"Failed to execute ESP-IDF flash: {exc}")
        return UploadResult(
            success=completed.returncode == 0,
            log=completed.stdout or "",
        )


__all__ = [
    "BOARD_DEFAULT_SPEC",
    "BOARD_NAME",
    "ESP32IDFDevice",
    "ESP32IDFOptions",
    "build_esp32_idf_spec",
    "classify_idf_build_failure",
    "parse_idf_size_output",
    "resolve_esp32_idf_options",
]
