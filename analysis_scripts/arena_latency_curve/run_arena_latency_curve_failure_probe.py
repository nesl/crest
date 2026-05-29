#!/usr/bin/env python3
"""
Run fixed-arena HIL sweeps and generate latency/energy curve outputs.

This script evaluates one model export against a list of tensor arena sizes and
captures attempt-level telemetry plus aggregated summaries:
- CSV attempts
- JSON metadata + aggregates
- PNG dual-axis latency/energy vs arena plot
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

VALID_DEVICES = ("ARDUINO_NANO_33_BLE_SENSE", "PORTENTA_H7")
VALID_INPUT_MODES = ("uniform", "oxiod_representative", "oxiod_real")
VALID_PORTENTA_CORES = ("cm7", "cm4")
DEFAULT_ERROR_CODE_EXCEPTION = -999

CSV_COLUMNS = [
    "timestamp",
    "device",
    "core",
    "arena_kb",
    "effective_arena_kb",
    "repeat",
    "latency_ms",
    "energy_mj_per_inference",
    "ram_bytes",
    "flash_bytes",
    "arena_bytes",
    "harness_latency_ms",
    "avg_power_mw",
    "avg_current_ma",
    "bus_voltage_v",
    "idle_power_mw",
    "error_code",
    "error_label",
]


@dataclass(frozen=True)
class SweepSettings:
    """Resolved runtime settings for one arena sweep run."""

    config_path: Path
    device_name: str
    core_label: str
    device_options: Optional[Dict[str, str]]
    arena_kb_list: list[int]
    effective_arena_kb_list: list[int]
    arena_underfill_kb: int
    repeats: int
    cooldown_s: float
    input_mode: str
    model_variant: str
    nb_filters: int
    trained_checkpoint: Optional[Path]
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


def _parse_arena_kb_list(raw: str) -> list[int]:
    """Parse and validate a comma-separated arena list."""
    values: list[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid arena entry '{token}'. Expected integers in KiB.") from exc
        if value <= 0:
            raise ValueError(f"Arena values must be positive. Got {value}.")
        values.append(value)
    if not values:
        raise ValueError("No arena sizes parsed from --arena-kb-list.")

    # Preserve user order while removing duplicates.
    deduped: list[int] = []
    seen: set[int] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


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
        if int(attempt.get("error_code", DEFAULT_ERROR_CODE_EXCEPTION)) != 0:
            continue
        numeric = _safe_float(attempt.get(key, -1.0))
        if numeric >= 0:
            values.append(numeric)
    return values


def _aggregate_by_arena(arena_kb_list: list[int], attempts: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    """Aggregate attempt rows by arena size."""
    grouped: dict[int, list[Dict[str, Any]]] = {value: [] for value in arena_kb_list}
    for record in attempts:
        grouped.setdefault(int(record["arena_kb"]), []).append(record)

    summary: list[Dict[str, Any]] = []
    for arena_kb in arena_kb_list:
        rows = grouped.get(arena_kb, [])
        success_count = sum(1 for row in rows if int(row.get("error_code", DEFAULT_ERROR_CODE_EXCEPTION)) == 0)
        failure_count = len(rows) - success_count
        entry: Dict[str, Any] = {
            "arena_kb": arena_kb,
            "attempt_count": len(rows),
            "success_count": success_count,
            "failure_count": failure_count,
        }

        for key in ("latency_ms", "energy_mj_per_inference"):
            values = _valid_success_values(rows, key)
            if not values:
                entry[f"{key}_mean"] = -1.0
                entry[f"{key}_std"] = -1.0
                entry[f"{key}_min"] = -1.0
                entry[f"{key}_max"] = -1.0
                entry[f"{key}_n"] = 0
                continue
            entry[f"{key}_mean"] = statistics.fmean(values)
            entry[f"{key}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
            entry[f"{key}_min"] = min(values)
            entry[f"{key}_max"] = max(values)
            entry[f"{key}_n"] = len(values)
        summary.append(entry)
    return summary


def _write_plot(path: Path, aggregates: list[Dict[str, Any]], *, title: str) -> None:
    """Render dual-axis latency/energy scatter vs arena size."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import ScalarFormatter
    except Exception as exc:  # pragma: no cover - dependency/runtime dependent
        raise RuntimeError(
            "matplotlib is required to generate the plot output."
        ) from exc

    success_rows = [row for row in aggregates if int(row.get("success_count", 0)) > 0]
    failed_rows = [row for row in aggregates if int(row.get("success_count", 0)) == 0]

    fig, ax_latency = plt.subplots(figsize=(10, 6))
    ax_energy = ax_latency.twinx()

    if success_rows:
        x = [int(row["arena_kb"]) for row in success_rows]
        latency_mean = [_safe_float(row["latency_ms_mean"]) for row in success_rows]
        energy_mean = [_safe_float(row["energy_mj_per_inference_mean"]) for row in success_rows]

        ax_latency.scatter(
            x,
            latency_mean,
            color="tab:blue",
            marker="o",
            s=48,
            label="Latency (ms)",
        )
        ax_energy.scatter(
            x,
            energy_mean,
            color="tab:orange",
            marker="s",
            s=48,
            label="Energy / Inference (mJ)",
        )

    for row in failed_rows:
        arena_kb = int(row["arena_kb"])
        ax_latency.axvline(arena_kb, linestyle="--", color="red", alpha=0.25)

    ax_latency.set_xlabel("Arena Size (KiB)")
    ax_latency.set_ylabel("Latency (ms)", color="tab:blue")
    ax_energy.set_ylabel("Energy / Inference (mJ)", color="tab:orange")
    ax_latency.grid(True, linestyle="--", alpha=0.25)
    ax_latency.set_title(title)

    # Keep axis labels in plain units (no scientific notation or offset text).
    for axis in (ax_latency, ax_energy):
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        axis.yaxis.set_major_formatter(formatter)
    x_formatter = ScalarFormatter(useOffset=False)
    x_formatter.set_scientific(False)
    ax_latency.xaxis.set_major_formatter(x_formatter)

    if failed_rows:
        top = ax_latency.get_ylim()[1]
        for row in failed_rows:
            ax_latency.annotate(
                "fail",
                (int(row["arena_kb"]), top * 0.98),
                color="red",
                fontsize=8,
                rotation=90,
                ha="center",
                va="top",
            )

    handles_latency, labels_latency = ax_latency.get_legend_handles_labels()
    handles_energy, labels_energy = ax_energy.get_legend_handles_labels()
    if handles_latency or handles_energy:
        ax_latency.legend(
            handles_latency + handles_energy,
            labels_latency + labels_energy,
            loc="best",
        )

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Run fixed-arena HIL sweeps and export latency/energy curve outputs "
            "(CSV + JSON + PNG)."
        )
    )
    parser.add_argument(
        "--config",
        default=str(SRC_DIR / "config" / "nas_config_stm32.yaml"),
        help="Path to TinyODOM config YAML.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Target device name (ARDUINO_NANO_33_BLE_SENSE or PORTENTA_H7). Defaults to config.",
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
        "--arena-kb-list",
        default=None,
        help="Optional comma-separated arena sizes in KiB. Defaults to device spec candidates.",
    )
    parser.add_argument(
        "--min-arena-kb",
        type=int,
        default=None,
        help="Optional minimum arena size filter (inclusive).",
    )
    parser.add_argument(
        "--max-arena-kb",
        type=int,
        default=None,
        help="Optional maximum arena size filter (inclusive).",
    )
    parser.add_argument(
        "--arena-underfill-kb",
        type=int,
        default=0,
        help=(
            "Subtract this many KiB from each requested arena for the actual tested "
            "arena. Useful for intentionally probing near-failure behavior."
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Attempts per arena size.",
    )
    parser.add_argument(
        "--cooldown-s",
        type=float,
        default=0.0,
        help="Cooldown between attempts in seconds.",
    )
    parser.add_argument(
        "--input-mode",
        default=None,
        help="Input mode override: uniform, oxiod_representative, oxiod_real.",
    )
    parser.add_argument(
        "--model-variant",
        default="approx_trained",
        help="Model variant (approx_trained, untrained, or trained*).",
    )
    parser.add_argument(
        "--nb-filters",
        type=int,
        default=10,
        help="Model width override for failure probing (default matches baseline: 10).",
    )
    parser.add_argument(
        "--trained-checkpoint",
        default=None,
        help="Checkpoint path required only for model variants that start with 'trained'.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="CSV output path (defaults under analysis_scripts/arena_latency_curve/results).",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="JSON output path (defaults under analysis_scripts/arena_latency_curve/results).",
    )
    parser.add_argument(
        "--output-plot",
        default=None,
        help="PNG output path (defaults under analysis_scripts/arena_latency_curve/results).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level.",
    )
    return parser


def _resolve_settings(args: argparse.Namespace, server: Any, runtime_device: Any) -> SweepSettings:
    """Resolve and validate sweep settings."""
    raw_device = args.device if args.device is not None else str(server.config.device.name)
    device_name = str(raw_device).strip().upper()
    if device_name not in VALID_DEVICES:
        raise ValueError(f"Unsupported --device '{device_name}'. Expected one of {VALID_DEVICES}.")

    input_mode = (
        str(args.input_mode).strip().lower()
        if args.input_mode is not None
        else str(server.config.training.input_mode).strip().lower()
    )
    if input_mode not in VALID_INPUT_MODES:
        raise ValueError(f"Unsupported --input-mode '{input_mode}'. Expected one of {VALID_INPUT_MODES}.")

    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1.")
    if args.cooldown_s < 0:
        raise ValueError("--cooldown-s must be >= 0.")
    if args.nb_filters < 1:
        raise ValueError("--nb-filters must be >= 1.")
    if args.arena_underfill_kb < 0:
        raise ValueError("--arena-underfill-kb must be >= 0.")
    if args.min_arena_kb is not None and args.min_arena_kb < 1:
        raise ValueError("--min-arena-kb must be >= 1 when provided.")
    if args.max_arena_kb is not None and args.max_arena_kb < 1:
        raise ValueError("--max-arena-kb must be >= 1 when provided.")
    if (
        args.min_arena_kb is not None
        and args.max_arena_kb is not None
        and args.min_arena_kb > args.max_arena_kb
    ):
        raise ValueError("--min-arena-kb cannot be greater than --max-arena-kb.")

    model_variant = str(args.model_variant).strip()
    variant_lower = model_variant.lower()
    trained_checkpoint = Path(args.trained_checkpoint).resolve() if args.trained_checkpoint else None
    if variant_lower.startswith("trained") and trained_checkpoint is None:
        raise ValueError(
            "--trained-checkpoint is required when --model-variant starts with 'trained'."
        )
    if trained_checkpoint is not None and not trained_checkpoint.exists():
        raise FileNotFoundError(f"Trained checkpoint not found: {trained_checkpoint}")

    if args.arena_kb_list:
        arena_kb_list = _parse_arena_kb_list(args.arena_kb_list)
    else:
        arena_kb_list = [int(value) for value in runtime_device.spec.arena_sizes_kb]
        if not arena_kb_list:
            raise ValueError(f"No default arena sizes available for {device_name}.")

    if args.min_arena_kb is not None:
        arena_kb_list = [value for value in arena_kb_list if value >= int(args.min_arena_kb)]
    if args.max_arena_kb is not None:
        arena_kb_list = [value for value in arena_kb_list if value <= int(args.max_arena_kb)]
    if not arena_kb_list:
        raise ValueError("Arena filters removed all candidates; widen --min/--max bounds.")

    arena_underfill_kb = int(args.arena_underfill_kb)
    effective_arena_kb_list = [max(1, value - arena_underfill_kb) for value in arena_kb_list]

    core_label = "n/a"
    device_options = None
    if device_name == "PORTENTA_H7":
        resolved = getattr(runtime_device, "resolved_options", None)
        if resolved is None:
            raise RuntimeError("Unable to resolve Portenta options from runtime device.")
        core_label = str(resolved.target_core)
        device_options = dict(resolved.to_board_options())

    timestamp_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    default_stem = SCRIPT_DIR / "results" / f"arena_failure_probe_{device_name}_{core_label}_{timestamp_tag}"
    output_csv = Path(args.output_csv) if args.output_csv else default_stem.with_suffix(".csv")
    output_json = Path(args.output_json) if args.output_json else default_stem.with_suffix(".json")
    output_plot = Path(args.output_plot) if args.output_plot else default_stem.with_suffix(".png")

    return SweepSettings(
        config_path=Path(args.config),
        device_name=device_name,
        core_label=core_label,
        device_options=device_options,
        arena_kb_list=arena_kb_list,
        effective_arena_kb_list=effective_arena_kb_list,
        arena_underfill_kb=arena_underfill_kb,
        repeats=int(args.repeats),
        cooldown_s=float(args.cooldown_s),
        input_mode=input_mode,
        model_variant=model_variant,
        nb_filters=int(args.nb_filters),
        trained_checkpoint=trained_checkpoint,
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
    quantization_mode = str(server.config.training.quantization.mode)
    calibration_inputs = (
        require_calibration_inputs_fn(server.get_calibration_inputs())
        if quantization_mode == "int8_ptq"
        else None
    )

    convert_to_tflite_model_fn(
        model=model,
        training_data=calibration_inputs,
        quantization_mode=quantization_mode,
        output_name=str(server.config.outputs.tflite_model_path),
    )
    convert_to_cpp_model_fn(
        tflite_path=server.config.outputs.tflite_model_path,
        output_dir=server.config.outputs.candidate_dir,
    )


def _build_attempt_record(
    *,
    timestamp: str,
    settings: SweepSettings,
    arena_kb: int,
    effective_arena_kb: int,
    repeat: int,
    ram_bytes: int,
    flash_bytes: int,
    latency_s: float,
    arena_bytes: int,
    error_code: int,
    error_label: str,
    normalized_power_metrics: Dict[str, float],
) -> Dict[str, Any]:
    """Convert one attempt result into a flat output row."""
    latency_ms = latency_s * 1000.0 if latency_s >= 0 else -1.0
    harness_latency_ms = (
        normalized_power_metrics["harness_latency_s"] * 1000.0
        if normalized_power_metrics["harness_latency_s"] >= 0
        else -1.0
    )
    return {
        "timestamp": timestamp,
        "device": settings.device_name,
        "core": settings.core_label,
        "arena_kb": int(arena_kb),
        "effective_arena_kb": int(effective_arena_kb),
        "repeat": int(repeat),
        "latency_ms": latency_ms,
        "energy_mj_per_inference": normalized_power_metrics["energy_mj_per_inference"],
        "ram_bytes": int(ram_bytes),
        "flash_bytes": int(flash_bytes),
        "arena_bytes": int(arena_bytes),
        "harness_latency_ms": harness_latency_ms,
        "avg_power_mw": normalized_power_metrics["avg_power_mw"],
        "avg_current_ma": normalized_power_metrics["avg_current_ma"],
        "bus_voltage_v": normalized_power_metrics["bus_voltage_v"],
        "idle_power_mw": normalized_power_metrics["idle_power_mw"],
        "error_code": int(error_code),
        "error_label": error_label,
    }


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    _configure_logging(args.log_level)

    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    from hil_server import HILServer
    from tinyodom.analysis_support import (
        build_fixed_tinyodom_hyperparams,
        require_calibration_inputs,
    )
    from tinyodom.hardware import HIL_spec, describe_error_code, normalize_power_metrics
    from tinyodom.microcontrollers import get_device as get_microcontroller_device
    from tinyodom.hardware import convert_to_tflite_model, convert_to_cpp_model

    server = HILServer(config_path=Path(args.config))

    device_name = str(args.device or server.config.device.name).strip().upper()
    if device_name not in VALID_DEVICES:
        raise ValueError(f"Unsupported --device '{device_name}'. Expected one of {VALID_DEVICES}.")
    server.config.device.name = device_name

    device_options_for_factory: Optional[Dict[str, str]] = None
    if device_name == "PORTENTA_H7":
        if getattr(server.config.device, "portenta", None) is None:
            from addict import Dict as AddictDict
            server.config.device.portenta = AddictDict()
        core_value = (
            str(args.portenta_core).strip().lower()
            if args.portenta_core is not None
            else str(getattr(server.config.device.portenta, "target_core", "")).strip().lower()
        )
        if core_value not in VALID_PORTENTA_CORES:
            raise ValueError(
                "--portenta-core is required for PORTENTA_H7 and must be one of cm7, cm4."
            )
        server.config.device.portenta.target_core = core_value
        if args.portenta_split is not None:
            server.config.device.portenta.split = str(args.portenta_split).strip().lower()
        if args.portenta_security is not None:
            server.config.device.portenta.security = str(args.portenta_security).strip().lower()

        device_options_for_factory = {"target_core": core_value}
        split_value = getattr(server.config.device.portenta, "split", None)
        security_value = getattr(server.config.device.portenta, "security", None)
        if split_value:
            device_options_for_factory["split"] = str(split_value)
        if security_value:
            device_options_for_factory["security"] = str(security_value)

    input_mode = (
        str(args.input_mode).strip().lower()
        if args.input_mode is not None
        else str(server.config.training.input_mode).strip().lower()
    )
    if input_mode not in VALID_INPUT_MODES:
        raise ValueError(f"Unsupported --input-mode '{input_mode}'. Expected one of {VALID_INPUT_MODES}.")
    server.set_input_mode(input_mode)

    use_serial_port = str(getattr(server.config.device, "serial_port", "")).strip()
    if not use_serial_port:
        raise RuntimeError("device.serial_port must be set in the config for HIL runs.")

    runtime_device = get_microcontroller_device(
        device_name,
        serial_port=use_serial_port,
        device_options=device_options_for_factory,
    )
    settings = _resolve_settings(args, server, runtime_device)

    # Build fixed hyperparameters and FLOPs once from the bootstrapped runtime.
    window_size, input_dim = server.get_runtime_dimensions()
    hyperparams = build_fixed_tinyodom_hyperparams(
        window_size=window_size,
        input_dim=input_dim,
        nb_filters=int(settings.nb_filters),
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
    total_attempts = len(settings.arena_kb_list) * settings.repeats
    attempt_index = 0

    for arena_idx, arena_kb in enumerate(settings.arena_kb_list):
        effective_arena_kb = int(settings.effective_arena_kb_list[arena_idx])
        for repeat_idx in range(1, settings.repeats + 1):
            attempt_index += 1
            logging.info(
                "Attempt %d/%d: requested_arena=%d KiB effective_arena=%d KiB repeat=%d",
                attempt_index,
                total_attempts,
                arena_kb,
                effective_arena_kb,
                repeat_idx,
            )
            try:
                (
                    ram_bytes,
                    flash_bytes,
                    latency_s,
                    arena_bytes,
                    error_code,
                    power_metrics,
                ) = HIL_spec(
                    dirpath=server.config.outputs.candidate_dir,
                    chosen_device=settings.device_name,
                    device_options=settings.device_options,
                    arenaSizes=settings.effective_arena_kb_list,
                    idx=arena_idx,
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
                    compile_only=False,
                    device=runtime_device,
                )
                normalized_power = normalize_power_metrics(power_metrics)
                error_label = describe_error_code(int(error_code), prefer_master=False)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                ram_bytes = -1
                flash_bytes = -1
                latency_s = -1.0
                arena_bytes = int(effective_arena_kb) * 1024
                error_code = DEFAULT_ERROR_CODE_EXCEPTION
                error_label = f"EXCEPTION_{type(exc).__name__}"
                normalized_power = normalize_power_metrics(None)
                logging.exception(
                    "Attempt failed via exception (requested=%d KiB effective=%d KiB, repeat=%d): %s",
                    arena_kb,
                    effective_arena_kb,
                    repeat_idx,
                    exc,
                )

            record = _build_attempt_record(
                timestamp=datetime.now(timezone.utc).isoformat(),
                settings=settings,
                arena_kb=int(arena_kb),
                effective_arena_kb=int(effective_arena_kb),
                repeat=repeat_idx,
                ram_bytes=int(ram_bytes),
                flash_bytes=int(flash_bytes),
                latency_s=float(latency_s),
                arena_bytes=int(arena_bytes),
                error_code=int(error_code),
                error_label=error_label,
                normalized_power_metrics=normalized_power,
            )
            attempts.append(record)

            logging.info(
                "Attempt result: requested=%d KiB effective=%d KiB repeat=%d error=%s latency_ms=%.3f energy_mj=%.3f",
                arena_kb,
                effective_arena_kb,
                repeat_idx,
                record["error_label"],
                _safe_float(record["latency_ms"]),
                _safe_float(record["energy_mj_per_inference"]),
            )

            if settings.cooldown_s > 0 and attempt_index < total_attempts:
                time.sleep(settings.cooldown_s)

    aggregates = _aggregate_by_arena(settings.arena_kb_list, attempts)

    _write_attempt_csv(settings.output_csv, attempts)

    payload = {
        "metadata": {
            "timestamp_utc": run_timestamp_utc,
            "config_path": str(settings.config_path),
            "device": settings.device_name,
            "core": settings.core_label,
            "device_options": settings.device_options or {},
            "model_variant": settings.model_variant,
            "nb_filters": settings.nb_filters,
            "trained_checkpoint": str(settings.trained_checkpoint) if settings.trained_checkpoint else "",
            "input_mode": settings.input_mode,
            "arena_kb_list": settings.arena_kb_list,
            "effective_arena_kb_list": settings.effective_arena_kb_list,
            "arena_underfill_kb": settings.arena_underfill_kb,
            "repeats": settings.repeats,
            "cooldown_s": settings.cooldown_s,
            "active_sketch_path": str(server.active_sketch_path),
            "candidate_dir": str(server.config.outputs.candidate_dir),
        },
        "attempts": attempts,
        "aggregates_by_arena": aggregates,
    }
    settings.output_json.parent.mkdir(parents=True, exist_ok=True)
    settings.output_json.write_text(json.dumps(payload, indent=2))

    title = (
        f"Arena Failure Probe - {settings.device_name}"
        f" ({settings.core_label}) - {settings.model_variant}"
    )
    _write_plot(settings.output_plot, aggregates, title=title)

    logging.info("Wrote CSV: %s", settings.output_csv)
    logging.info("Wrote JSON: %s", settings.output_json)
    logging.info("Wrote plot: %s", settings.output_plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
