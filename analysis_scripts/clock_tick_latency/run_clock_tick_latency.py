#!/usr/bin/env python3
"""
Run perturbed-model HIL repeats and normalize latency to ticks/inference.

Outputs per run:
- attempt-level CSV
- metadata + attempts + aggregates JSON
- latency-vs-ticks scatter PNG
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"

VALID_DEVICE = "PORTENTA_H7"
VALID_INPUT_MODES = ("uniform", "oxiod_representative", "oxiod_real")
VALID_PORTENTA_CORES = ("cm7", "cm4")
DEFAULT_ERROR_CODE_EXCEPTION = -999
MASTER_SUCCESS_CODE = 1
DEFAULT_FALLBACK_CLOCK_HZ_BY_CORE = {
    "cm7": 400_000_000.0,
    "cm4": 240_000_000.0,
}

CSV_COLUMNS = [
    "timestamp",
    "device",
    "core",
    "repeat",
    "latency_ms",
    "clock_hz",
    "clock_source",
    "ticks_per_inference",
    "ticks_source",
    "dwt_cycles_per_inference",
    "ram_bytes",
    "flash_bytes",
    "arena_bytes",
    "error_code",
    "error_label",
]


@dataclass(frozen=True)
class RunSettings:
    """Resolved runtime settings for one clock/tick run."""

    config_path: Path
    device_name: str
    core_label: str
    device_options: Dict[str, str]
    repeats: int
    cooldown_s: float
    input_mode: str
    model_variant: str
    trained_checkpoint: Optional[Path]
    fallback_clock_hz: float
    output_csv: Path
    output_json: Path
    output_plot: Path


def _configure_logging(level_name: str) -> None:
    """Configure process logging."""
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _safe_float(value: Any, default: float = -1.0) -> float:
    """Convert to float with a finite fallback."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if numeric != numeric or numeric in (float("inf"), float("-inf")):
        return default
    return numeric


def _write_attempt_csv(path: Path, attempts: list[Dict[str, Any]]) -> None:
    """Write attempt-level CSV output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in attempts:
            writer.writerow({name: record.get(name, "") for name in CSV_COLUMNS})


def _valid_success_values(attempts: Iterable[Dict[str, Any]], key: str) -> list[float]:
    """Collect non-negative numeric values from successful attempts only."""
    values: list[float] = []
    for attempt in attempts:
        if int(attempt.get("error_code", DEFAULT_ERROR_CODE_EXCEPTION)) != MASTER_SUCCESS_CODE:
            continue
        numeric = _safe_float(attempt.get(key, -1.0))
        if numeric >= 0:
            values.append(numeric)
    return values


def _aggregate_metric(attempts: list[Dict[str, Any]], key: str) -> Dict[str, Any]:
    """Aggregate one metric over successful attempts."""
    values = _valid_success_values(attempts, key)
    if not values:
        return {
            "mean": -1.0,
            "std": -1.0,
            "min": -1.0,
            "max": -1.0,
            "count": 0,
        }
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "count": len(values),
    }


def _write_plot(path: Path, attempts: list[Dict[str, Any]], *, title: str, core: str) -> None:
    """Render latency-vs-ticks scatter with per-core summary marker."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import ScalarFormatter
    except Exception as exc:  # pragma: no cover - dependency/runtime dependent
        raise RuntimeError("matplotlib is required to generate the plot output.") from exc

    valid_points = [
        row
        for row in attempts
        if int(row.get("error_code", DEFAULT_ERROR_CODE_EXCEPTION)) == MASTER_SUCCESS_CODE
        and _safe_float(row.get("latency_ms", -1.0)) >= 0
        and _safe_float(row.get("ticks_per_inference", -1.0)) >= 0
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    if valid_points:
        x = [_safe_float(row["latency_ms"]) for row in valid_points]
        y = [_safe_float(row["ticks_per_inference"]) for row in valid_points]
        ax.scatter(x, y, color="tab:blue", s=48, alpha=0.85, label="Attempts")

        mean_x = statistics.fmean(x)
        mean_y = statistics.fmean(y)
        ax.scatter(
            [mean_x],
            [mean_y],
            color="black",
            marker="X",
            s=120,
            label=f"{core.upper()} mean",
            zorder=3,
        )

    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Ticks / Inference")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.set_title(title)

    formatter_x = ScalarFormatter(useOffset=False)
    formatter_x.set_scientific(False)
    formatter_y = ScalarFormatter(useOffset=False)
    formatter_y.set_scientific(False)
    ax.xaxis.set_major_formatter(formatter_x)
    ax.yaxis.set_major_formatter(formatter_y)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="best")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Run perturbed-model HIL repeats and export latency/ticks outputs "
            "(CSV + JSON + PNG)."
        )
    )
    parser.add_argument(
        "--config",
        default=str(SRC_DIR / "config" / "nas_config.yaml"),
        help="Path to TinyODOM config YAML.",
    )
    parser.add_argument(
        "--device",
        default=VALID_DEVICE,
        help="Target device name. This workflow supports PORTENTA_H7 only.",
    )
    parser.add_argument(
        "--portenta-core",
        default=None,
        help="Portenta core selection for PORTENTA_H7 (cm7 or cm4).",
    )
    parser.add_argument(
        "--portenta-split",
        default=None,
        help="Optional Portenta split override (50_50, 75_25, 100_0).",
    )
    parser.add_argument(
        "--portenta-security",
        default=None,
        help="Optional Portenta security override (none, default, secure).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Number of HIL repeats.",
    )
    parser.add_argument(
        "--cooldown-s",
        type=float,
        default=0.0,
        help="Cooldown between attempts in seconds.",
    )
    parser.add_argument(
        "--model-variant",
        default="approx_trained",
        help="Model variant (default: approx_trained).",
    )
    parser.add_argument(
        "--trained-checkpoint",
        default=None,
        help="Checkpoint path required only for model variants that start with 'trained'.",
    )
    parser.add_argument(
        "--input-mode",
        default="uniform",
        help="Input mode override: uniform, oxiod_representative, or oxiod_real.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="CSV output path (defaults under analysis_scripts/clock_tick_latency/results).",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="JSON output path (defaults under analysis_scripts/clock_tick_latency/results).",
    )
    parser.add_argument(
        "--output-plot",
        default=None,
        help="PNG output path (defaults under analysis_scripts/clock_tick_latency/results).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level.",
    )
    return parser


def _resolve_fallback_clock_hz(server: Any, core_label: str) -> float:
    """Resolve fallback clock for selected core with optional config override."""
    portenta_cfg = getattr(getattr(server.config, "device", None), "portenta", None)
    key = f"clock_hz_{core_label}"
    value = getattr(portenta_cfg, key, None) if portenta_cfg is not None else None
    parsed = _safe_float(value, default=-1.0)
    if parsed > 0:
        return parsed
    return DEFAULT_FALLBACK_CLOCK_HZ_BY_CORE.get(core_label, -1.0)


def _resolve_settings(args: argparse.Namespace, server: Any, runtime_device: Any) -> RunSettings:
    """Resolve and validate run settings."""
    device_name = str(args.device).strip().upper()
    if device_name != VALID_DEVICE:
        raise ValueError(
            f"Unsupported --device '{device_name}'. This workflow supports {VALID_DEVICE} only."
        )

    input_mode = str(args.input_mode).strip().lower()
    if input_mode not in VALID_INPUT_MODES:
        raise ValueError(f"Unsupported --input-mode '{input_mode}'. Expected one of {VALID_INPUT_MODES}.")

    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1.")
    if args.cooldown_s < 0:
        raise ValueError("--cooldown-s must be >= 0.")

    model_variant = str(args.model_variant).strip()
    variant_lower = model_variant.lower()
    trained_checkpoint = Path(args.trained_checkpoint).resolve() if args.trained_checkpoint else None
    if variant_lower.startswith("trained") and trained_checkpoint is None:
        raise ValueError(
            "--trained-checkpoint is required when --model-variant starts with 'trained'."
        )
    if trained_checkpoint is not None and not trained_checkpoint.exists():
        raise FileNotFoundError(f"Trained checkpoint not found: {trained_checkpoint}")

    resolved = getattr(runtime_device, "resolved_options", None)
    if resolved is None:
        raise RuntimeError("Unable to resolve Portenta options from runtime device.")
    core_label = str(resolved.target_core)
    if core_label not in VALID_PORTENTA_CORES:
        raise ValueError(f"Unsupported core '{core_label}'. Expected one of {VALID_PORTENTA_CORES}.")

    fallback_clock_hz = _resolve_fallback_clock_hz(server, core_label)

    timestamp_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    default_stem = SCRIPT_DIR / "results" / f"clock_tick_latency_{device_name}_{core_label}_{timestamp_tag}"
    output_csv = Path(args.output_csv) if args.output_csv else default_stem.with_suffix(".csv")
    output_json = Path(args.output_json) if args.output_json else default_stem.with_suffix(".json")
    output_plot = Path(args.output_plot) if args.output_plot else default_stem.with_suffix(".png")

    return RunSettings(
        config_path=Path(args.config),
        device_name=device_name,
        core_label=core_label,
        device_options=dict(resolved.to_board_options()),
        repeats=int(args.repeats),
        cooldown_s=float(args.cooldown_s),
        input_mode=input_mode,
        model_variant=model_variant,
        trained_checkpoint=trained_checkpoint,
        fallback_clock_hz=fallback_clock_hz,
        output_csv=output_csv,
        output_json=output_json,
        output_plot=output_plot,
    )


def _prepare_model_artifacts(
    *,
    server: Any,
    hyperparams: Any,
    model_variant: str,
    trained_checkpoint: Optional[Path],
    convert_to_tflite_model_fn: Any,
    convert_to_cpp_model_fn: Any,
    require_calibration_inputs_fn: Any,
) -> None:
    """Build/select one model variant and export model artifacts once."""
    model = server.model_family.materialize_export_model(
        dict(hyperparams),
        server.model_build_context,
        server.model_config,
        model_variant=model_variant,
        checkpoint_path=trained_checkpoint,
    )
    server.task.compile_model(model, server.task_config, server.target_spec)
    calibration_inputs = require_calibration_inputs_fn(server.get_calibration_inputs())

    convert_to_tflite_model_fn(
        model=model,
        training_data=calibration_inputs,
        quantization=bool(server.config.training.quantization),
        output_name=str(server.config.outputs.tflite_model_path),
    )
    convert_to_cpp_model_fn(
        tflite_path=server.config.outputs.tflite_model_path,
        output_dir=server.config.outputs.candidate_dir,
    )


def _resolve_clock(runtime_clock_hz: float, fallback_clock_hz: float) -> tuple[float, str]:
    """Resolve final clock value/source with runtime-first precedence."""
    if runtime_clock_hz >= 0:
        return runtime_clock_hz, "runtime_reported"
    if fallback_clock_hz >= 0:
        return fallback_clock_hz, "fallback_estimate"
    return -1.0, "unavailable"


def _compute_ticks_per_inference(
    *,
    latency_s: float,
    clock_hz: float,
    dwt_cycles_per_inference: float,
) -> tuple[float, str]:
    """Compute ticks/inference from preferred metric then fallback."""
    if dwt_cycles_per_inference >= 0:
        return dwt_cycles_per_inference, "dwt_cycles_per_inference"
    if latency_s >= 0 and clock_hz >= 0:
        return latency_s * clock_hz, "latency_s_x_clock_hz"
    if latency_s < 0:
        return -1.0, "unavailable_missing_latency"
    if clock_hz < 0:
        return -1.0, "unavailable_missing_clock"
    return -1.0, "unavailable"


def _build_attempt_record(
    *,
    timestamp: str,
    settings: RunSettings,
    repeat: int,
    latency_s: float,
    clock_hz: float,
    clock_source: str,
    dwt_cycles_per_inference: float,
    ticks_per_inference: float,
    ticks_source: str,
    ram_bytes: int,
    flash_bytes: int,
    arena_bytes: int,
    error_code: int,
    error_label: str,
) -> Dict[str, Any]:
    """Convert one attempt result into a flat output row."""
    latency_ms = latency_s * 1000.0 if latency_s >= 0 else -1.0
    return {
        "timestamp": timestamp,
        "device": settings.device_name,
        "core": settings.core_label,
        "repeat": int(repeat),
        "latency_ms": latency_ms,
        "clock_hz": clock_hz,
        "clock_source": clock_source,
        "ticks_per_inference": ticks_per_inference,
        "ticks_source": ticks_source,
        "dwt_cycles_per_inference": dwt_cycles_per_inference,
        "ram_bytes": int(ram_bytes),
        "flash_bytes": int(flash_bytes),
        "arena_bytes": int(arena_bytes),
        "error_code": int(error_code),
        "error_label": error_label,
    }


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    _configure_logging(args.log_level)

    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    from addict import Dict as AddictDict
    from hil_server import HILServer
    from tinyodom.analysis_support import (
        build_fixed_tinyodom_hyperparams,
        require_calibration_inputs,
    )
    from tinyodom.hardware import (
        HIL_controller,
        describe_error_code,
        normalize_power_metrics,
    )
    from tinyodom.microcontrollers import get_device as get_microcontroller_device
    from tinyodom.hardware import convert_to_tflite_model, convert_to_cpp_model

    server = HILServer(config_path=Path(args.config))

    device_name = str(args.device).strip().upper()
    if device_name != VALID_DEVICE:
        raise ValueError(
            f"Unsupported --device '{device_name}'. This workflow supports {VALID_DEVICE} only."
        )
    server.config.device.name = device_name

    if getattr(server.config.device, "portenta", None) is None:
        server.config.device.portenta = AddictDict()

    core_value = (
        str(args.portenta_core).strip().lower()
        if args.portenta_core is not None
        else str(getattr(server.config.device.portenta, "target_core", "")).strip().lower()
    )
    if core_value not in VALID_PORTENTA_CORES:
        raise ValueError("--portenta-core is required for PORTENTA_H7 and must be one of cm7, cm4.")
    server.config.device.portenta.target_core = core_value

    if args.portenta_split is not None:
        server.config.device.portenta.split = str(args.portenta_split).strip().lower()
    if args.portenta_security is not None:
        server.config.device.portenta.security = str(args.portenta_security).strip().lower()

    input_mode = str(args.input_mode).strip().lower()
    if input_mode not in VALID_INPUT_MODES:
        raise ValueError(f"Unsupported --input-mode '{input_mode}'. Expected one of {VALID_INPUT_MODES}.")

    # Keep perturbed analysis behavior aligned with existing single-run perturbed script.
    server.config.training.energy_aware = True
    server.set_input_mode(input_mode)

    use_serial_port = str(getattr(server.config.device, "serial_port", "")).strip()
    if not use_serial_port:
        raise RuntimeError("device.serial_port must be set in the config for HIL runs.")

    device_options_for_factory: Dict[str, str] = {"target_core": core_value}
    split_value = getattr(server.config.device.portenta, "split", None)
    security_value = getattr(server.config.device.portenta, "security", None)
    if split_value:
        device_options_for_factory["split"] = str(split_value)
    if security_value:
        device_options_for_factory["security"] = str(security_value)

    runtime_device = get_microcontroller_device(
        device_name,
        serial_port=use_serial_port,
        device_options=device_options_for_factory,
    )
    settings = _resolve_settings(args, server, runtime_device)

    window_size, input_dim = server.get_runtime_dimensions()
    hyperparams = AddictDict(
        build_fixed_tinyodom_hyperparams(
            window_size=window_size,
            input_dim=input_dim,
        )
    )

    logging.info(
        "Preparing model export for variant '%s' (device=%s core=%s input_mode=%s).",
        settings.model_variant,
        settings.device_name,
        settings.core_label,
        settings.input_mode,
    )
    _prepare_model_artifacts(
        server=server,
        hyperparams=hyperparams,
        model_variant=settings.model_variant,
        trained_checkpoint=settings.trained_checkpoint,
        convert_to_tflite_model_fn=convert_to_tflite_model,
        convert_to_cpp_model_fn=convert_to_cpp_model,
        require_calibration_inputs_fn=require_calibration_inputs,
    )

    attempts: list[Dict[str, Any]] = []
    run_timestamp_utc = datetime.now(timezone.utc).isoformat()

    for repeat_idx in range(1, settings.repeats + 1):
        logging.info("Attempt %d/%d", repeat_idx, settings.repeats)
        try:
            (
                ram_bytes,
                flash_bytes,
                latency_s,
                arena_bytes,
                error_code,
                power_metrics,
            ) = HIL_controller(
                dirpath=server.config.outputs.candidate_dir,
                chosen_device=settings.device_name,
                device_options=settings.device_options,
                window_size=int(window_size),
                number_of_channels=int(hyperparams.input_dim),
                serial_port=use_serial_port,
                dut_ready_timeout_s=float(server.config.device.dut_ready_timeout_s),
                harness_serial_port=getattr(server.config.device, "harness_serial_port", None),
                harness_fqbn=getattr(server.config.device, "harness_fqbn", None),
                harness_auto_flash=getattr(server.config.device, "harness_auto_flash", None),
                harness_arm_pin=getattr(server.config.device, "harness_arm_pin", None),
                harness_trigger_pin=getattr(server.config.device, "harness_trigger_pin", None),
                dut_arm_hold_ms=getattr(server.config.device, "dut_arm_hold_ms", None),
                harness_stable_low_ms=getattr(server.config.device, "harness_stable_low_ms", None),
                harness_ready_timeout_s=getattr(server.config.device, "harness_ready_timeout_s", None),
                harness_arm_timeout_s=getattr(server.config.device, "harness_arm_timeout_s", None),
                harness_active_timeout_s=getattr(server.config.device, "harness_active_timeout_s", None),
                harness_done_timeout_s=getattr(server.config.device, "harness_done_timeout_s", None),
                run_hil=True,
                device=runtime_device,
            )
            normalized_power = normalize_power_metrics(power_metrics)
            error_label = describe_error_code(int(error_code), prefer_master=True)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            ram_bytes = -1
            flash_bytes = -1
            latency_s = -1.0
            arena_bytes = -1
            error_code = DEFAULT_ERROR_CODE_EXCEPTION
            error_label = f"EXCEPTION_{type(exc).__name__}"
            normalized_power = normalize_power_metrics(None)
            logging.exception("Attempt failed via exception (repeat=%d): %s", repeat_idx, exc)

        runtime_clock_hz = _safe_float(normalized_power.get("clock_hz", -1.0))
        dwt_cycles_per_inference = _safe_float(
            normalized_power.get("dwt_cycles_per_inference", -1.0)
        )
        clock_hz, clock_source = _resolve_clock(runtime_clock_hz, settings.fallback_clock_hz)
        ticks_per_inference, ticks_source = _compute_ticks_per_inference(
            latency_s=float(latency_s),
            clock_hz=clock_hz,
            dwt_cycles_per_inference=dwt_cycles_per_inference,
        )

        record = _build_attempt_record(
            timestamp=datetime.now(timezone.utc).isoformat(),
            settings=settings,
            repeat=repeat_idx,
            latency_s=float(latency_s),
            clock_hz=clock_hz,
            clock_source=clock_source,
            dwt_cycles_per_inference=dwt_cycles_per_inference,
            ticks_per_inference=ticks_per_inference,
            ticks_source=ticks_source,
            ram_bytes=int(ram_bytes),
            flash_bytes=int(flash_bytes),
            arena_bytes=int(arena_bytes),
            error_code=int(error_code),
            error_label=error_label,
        )
        attempts.append(record)

        logging.info(
            "Attempt result: repeat=%d error=%s latency_ms=%.3f ticks=%.3f clock_hz=%.3f source=%s",
            repeat_idx,
            record["error_label"],
            _safe_float(record["latency_ms"]),
            _safe_float(record["ticks_per_inference"]),
            _safe_float(record["clock_hz"]),
            record["clock_source"],
        )

        if settings.cooldown_s > 0 and repeat_idx < settings.repeats:
            time.sleep(settings.cooldown_s)

    _write_attempt_csv(settings.output_csv, attempts)

    payload = {
        "metadata": {
            "timestamp_utc": run_timestamp_utc,
            "config_path": str(settings.config_path),
            "device": settings.device_name,
            "core": settings.core_label,
            "device_options": settings.device_options,
            "model_variant": settings.model_variant,
            "trained_checkpoint": str(settings.trained_checkpoint) if settings.trained_checkpoint else "",
            "input_mode": settings.input_mode,
            "repeats": settings.repeats,
            "cooldown_s": settings.cooldown_s,
            "fallback_clock_hz": settings.fallback_clock_hz,
            "active_sketch_path": str(server.active_sketch_path),
            "candidate_dir": str(server.config.outputs.candidate_dir),
        },
        "attempts": attempts,
        "aggregates": {
            "latency_ms": _aggregate_metric(attempts, "latency_ms"),
            "ticks_per_inference": _aggregate_metric(attempts, "ticks_per_inference"),
        },
    }
    settings.output_json.parent.mkdir(parents=True, exist_ok=True)
    settings.output_json.write_text(json.dumps(payload, indent=2))

    title = (
        f"Clock-Tick Latency - {settings.device_name} "
        f"({settings.core_label}) - {settings.model_variant}"
    )
    _write_plot(settings.output_plot, attempts, title=title, core=settings.core_label)

    logging.info("Wrote CSV: %s", settings.output_csv)
    logging.info("Wrote JSON: %s", settings.output_json)
    logging.info("Wrote plot: %s", settings.output_plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
