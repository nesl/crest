#!/usr/bin/env python3
"""
Run cadenced-vs-back-to-back perturbed-model HIL experiments on Portenta H7.

This script executes a fixed four-cell matrix:
1. CM7 + back_to_back
2. CM7 + cadenced
3. CM4 + back_to_back
4. CM4 + cadenced

Each run compiles/uploads a phase-specific sketch template, collects HIL metrics,
and writes JSON + CSV outputs with per-attempt and aggregated summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import shutil
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

PHASE_BACK_TO_BACK = "back_to_back"
PHASE_CADENCED = "cadenced"
VALID_PHASES = (PHASE_BACK_TO_BACK, PHASE_CADENCED)
VALID_CORES = ("cm7", "cm4")
MODEL_VARIANT_NAME = "approx_trained"

PHASE_SKETCH_FILENAMES = {
    PHASE_BACK_TO_BACK: "tinyodom_tcn_energy_back_to_back.ino",
    PHASE_CADENCED: "tinyodom_tcn_energy_cadenced.ino",
}

CSV_COLUMNS = [
    "timestamp_utc",
    "core",
    "phase",
    "repeat",
    "model_variant",
    "sketch_template",
    "latency_budget_ms_requested",
    "latency_budget_ms_reported",
    "latency_ms",
    "harness_latency_ms",
    "energy_mj_per_inference",
    "avg_power_mw",
    "avg_current_ma",
    "bus_voltage_v",
    "idle_power_mw",
    "inference_seq",
    "error_code",
    "phase_a_energy_mj_per_inference",
    "phase_a_latency_ms_per_window",
    "phase_b_energy_mj_per_window",
    "phase_b_latency_ms_per_window",
]

AGG_NUMERIC_FIELDS = (
    "latency_ms",
    "harness_latency_ms",
    "energy_mj_per_inference",
    "avg_power_mw",
    "avg_current_ma",
    "bus_voltage_v",
    "idle_power_mw",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Resolved experiment configuration.

    Parameters
    ----------
    cores : list[str]
        Portenta target cores to run (subset of ``["cm7", "cm4"]``).
    repeats : int
        Number of repeated attempts per ``core x phase`` pair.
    latency_budget_ms : float
        Cadence period used by the cadenced sketch, in milliseconds.
    output_json : Path
        Destination JSON path.
    output_csv : Path
        Destination CSV path.
    config_path : Path
        TinyODOM config YAML path.
    """

    cores: list[str]
    repeats: int
    latency_budget_ms: float
    output_json: Path
    output_csv: Path
    config_path: Path


class AttrDict(dict):
    """Minimal dict with attribute access used for hyperparameter payloads."""

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _derive_default_latency_budget_ms(server: Any) -> float:
    """Compute cadence budget from stride and sampling rate.

    Parameters
    ----------
    server : HILServer
        Initialized HIL server with loaded config.

    Returns
    -------
    float
        ``(stride / sampling_rate_hz) * 1000`` in milliseconds.
    """

    return (float(server.config.data.stride) / float(server.config.data.sampling_rate_hz)) * 1000.0


def _build_hyperparams(server: Any, build_tinyodom_model_fn: Any, count_flops_fn: Any) -> AttrDict:
    """Construct the fixed TinyODOM hyperparameter set and FLOP count.

    Parameters
    ----------
    server : HILServer
        Active HIL server used to resolve shape metadata.

    Returns
    -------
    AttrDict
        Hyperparameter bundle consumed by ``determine_metrics``.
    """

    hyperparams = AttrDict(
        nb_filters=10,
        kernel_size=12,
        dilations=[1, 4, 8, 64],
        dropout_rate=0.0,
        use_skip_connections=False,
        norm_flag=True,
        batch_size=256,
        timesteps=server.config.data.window_size,
        input_dim=server.training_data.inputs.shape[2],
    )
    model = build_tinyodom_model_fn(hyperparams)
    hyperparams.flops = count_flops_fn(model, (hyperparams.timesteps, hyperparams.input_dim))
    return hyperparams


def _ensure_portenta_runtime_config(server: Any, core: str) -> None:
    """Force Portenta energy-aware runtime configuration for one core.

    Parameters
    ----------
    server : HILServer
        Active HIL server.
    core : str
        Target core token (``cm7`` or ``cm4``).
    """

    if core not in VALID_CORES:
        raise ValueError(f"Unsupported core '{core}'. Expected one of {VALID_CORES}.")
    server.config.device.name = "PORTENTA_H7"
    if getattr(server.config.device, "portenta", None) is None:
        server.config.device.portenta = AttrDict()
    server.config.device.portenta.target_core = core
    server.config.training.energy_aware = True


def _patch_cadence_define(sketch_text: str, latency_budget_ms: float) -> str:
    """Patch ``TINYODOM_LATENCY_BUDGET_MS`` inside a staged sketch.

    Parameters
    ----------
    sketch_text : str
        Sketch source text.
    latency_budget_ms : float
        Cadence period in milliseconds.

    Returns
    -------
    str
        Updated sketch text with an integer millisecond cadence define.
    """

    budget_ms = max(1, int(round(latency_budget_ms)))
    pattern = re.compile(r"(#define\s+TINYODOM_LATENCY_BUDGET_MS\s+)\d+")
    if not pattern.search(sketch_text):
        raise ValueError("Unable to locate TINYODOM_LATENCY_BUDGET_MS define in cadenced sketch.")
    return pattern.sub(rf"\g<1>{budget_ms}", sketch_text, count=1)


def _stage_phase_sketch(server: Any, phase: str, latency_budget_ms: float) -> Path:
    """Copy one phase template into the active TinyODOM sketch directory.

    Parameters
    ----------
    server : HILServer
        Active HIL server with output directory config.
    phase : str
        Experiment phase (``back_to_back`` or ``cadenced``).
    latency_budget_ms : float
        Cadence period used for the cadenced phase.

    Returns
    -------
    pathlib.Path
        Path to staged ``tinyodom_tcn.ino``.
    """

    if phase not in VALID_PHASES:
        raise ValueError(f"Unsupported phase '{phase}'. Expected one of {VALID_PHASES}.")

    template_path = SCRIPT_DIR / PHASE_SKETCH_FILENAMES[phase]
    if not template_path.exists():
        raise FileNotFoundError(f"Missing sketch template: {template_path}")

    sketch_dir = Path(server.config.outputs.tcn_dir)
    sketch_dir.mkdir(parents=True, exist_ok=True)

    # Keep shared HIL headers synchronized with the staged analysis sketch.
    common_source = REPO_ROOT / "sketches" / "common"
    shutil.copytree(common_source, sketch_dir / "common", dirs_exist_ok=True)

    sketch_text = template_path.read_text()
    if phase == PHASE_CADENCED:
        sketch_text = _patch_cadence_define(sketch_text, latency_budget_ms)

    staged_path = sketch_dir / "tinyodom_tcn.ino"
    staged_path.write_text(sketch_text)
    return staged_path


def _safe_float(value: Any) -> float:
    """Convert metric values to float with ``-1.0`` fallback."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return -1.0
    if math.isfinite(numeric):
        return numeric
    return -1.0


def _build_attempt_record(
    *,
    timestamp_utc: str,
    core: str,
    phase: str,
    repeat_idx: int,
    latency_budget_ms: float,
    sketch_template: str,
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Build one flat attempt record suitable for JSON and CSV.

    Parameters
    ----------
    timestamp_utc : str
        RFC3339 UTC timestamp.
    core : str
        Portenta target core.
    phase : str
        Experiment phase.
    repeat_idx : int
        One-based repeat index.
    latency_budget_ms : float
        Requested cadence budget.
    sketch_template : str
        Sketch template filename used for this run.
    metrics : dict[str, Any]
        Metrics dictionary returned by TinyODOM runtime.

    Returns
    -------
    dict[str, Any]
        Flat record with derived phase-specific columns.
    """

    record: Dict[str, Any] = {
        "timestamp_utc": timestamp_utc,
        "core": core,
        "phase": phase,
        "repeat": repeat_idx,
        "model_variant": MODEL_VARIANT_NAME,
        "sketch_template": sketch_template,
        "latency_budget_ms_requested": latency_budget_ms,
        "latency_budget_ms_reported": _safe_float(metrics.get("latency_budget_ms", -1.0)),
        "latency_ms": _safe_float(metrics.get("latency_ms", -1.0)),
        "harness_latency_ms": _safe_float(metrics.get("harness_latency_ms", -1.0)),
        "energy_mj_per_inference": _safe_float(metrics.get("energy_mj_per_inference", -1.0)),
        "avg_power_mw": _safe_float(metrics.get("avg_power_mw", -1.0)),
        "avg_current_ma": _safe_float(metrics.get("avg_current_ma", -1.0)),
        "bus_voltage_v": _safe_float(metrics.get("bus_voltage_v", -1.0)),
        "idle_power_mw": _safe_float(metrics.get("idle_power_mw", -1.0)),
        "inference_seq": int(_safe_float(metrics.get("inference_seq", -1))),
        "error_code": int(_safe_float(metrics.get("error_code", -1))),
        "phase_a_energy_mj_per_inference": "",
        "phase_a_latency_ms_per_window": "",
        "phase_b_energy_mj_per_window": "",
        "phase_b_latency_ms_per_window": "",
    }

    if phase == PHASE_BACK_TO_BACK:
        record["phase_a_energy_mj_per_inference"] = record["energy_mj_per_inference"]
        record["phase_a_latency_ms_per_window"] = record["latency_ms"]
    else:
        # Cadenced sketch reports per-window energy in the same telemetry field.
        record["phase_b_energy_mj_per_window"] = record["energy_mj_per_inference"]
        record["phase_b_latency_ms_per_window"] = record["latency_ms"]

    return record


def _valid_metric_values(attempts: Iterable[Dict[str, Any]], key: str) -> list[float]:
    """Collect valid non-negative finite values for one metric key."""

    values: list[float] = []
    for attempt in attempts:
        value = _safe_float(attempt.get(key, -1.0))
        if value >= 0.0:
            values.append(value)
    return values


def _summarize_group(core: str, phase: str, attempts: list[Dict[str, Any]], latency_budget_ms: float) -> Dict[str, Any]:
    """Aggregate one ``core x phase`` attempt group.

    Parameters
    ----------
    core : str
        Target core label.
    phase : str
        Experiment phase label.
    attempts : list[dict[str, Any]]
        Attempts in the group.
    latency_budget_ms : float
        Requested cadence budget used to compute over-budget counts.

    Returns
    -------
    dict[str, Any]
        Aggregate summary record.
    """

    summary: Dict[str, Any] = {
        "core": core,
        "phase": phase,
        "attempt_count": len(attempts),
        "success_count": sum(1 for attempt in attempts if int(attempt.get("error_code", -1)) == 0),
        "failure_count": sum(1 for attempt in attempts if int(attempt.get("error_code", -1)) != 0),
        "over_budget_count": sum(
            1
            for attempt in attempts
            if _safe_float(attempt.get("latency_ms", -1.0)) > latency_budget_ms
        ),
    }

    for key in AGG_NUMERIC_FIELDS:
        values = _valid_metric_values(attempts, key)
        if not values:
            summary[f"{key}_mean"] = -1.0
            summary[f"{key}_std"] = -1.0
            summary[f"{key}_min"] = -1.0
            summary[f"{key}_max"] = -1.0
            summary[f"{key}_n"] = 0
            continue
        summary[f"{key}_mean"] = statistics.fmean(values)
        summary[f"{key}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
        summary[f"{key}_min"] = min(values)
        summary[f"{key}_max"] = max(values)
        summary[f"{key}_n"] = len(values)
    return summary


def _write_csv(path: Path, attempts: list[Dict[str, Any]]) -> None:
    """Write per-attempt records to CSV.

    Parameters
    ----------
    path : pathlib.Path
        Output CSV path.
    attempts : list[dict[str, Any]]
        Flat attempt records.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in attempts:
            writer.writerow({column: record.get(column, "") for column in CSV_COLUMNS})


def _resolve_experiment_config(args: argparse.Namespace, server: Any) -> ExperimentConfig:
    """Resolve defaults and validate CLI arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    server : HILServer
        Active HIL server for default budget derivation.

    Returns
    -------
    ExperimentConfig
        Validated experiment settings.
    """

    cores = [str(core).strip().lower() for core in args.cores]
    invalid_cores = [core for core in cores if core not in VALID_CORES]
    if invalid_cores:
        raise ValueError(f"Invalid cores: {invalid_cores}. Expected subset of {VALID_CORES}.")
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1.")

    default_budget_ms = _derive_default_latency_budget_ms(server)
    latency_budget_ms = float(args.latency_budget_ms) if args.latency_budget_ms is not None else default_budget_ms
    if latency_budget_ms <= 0:
        raise ValueError("latency budget must be positive.")

    timestamp_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    default_output_stem = SCRIPT_DIR / "results" / f"cadenced_portenta_h7_{timestamp_tag}"
    output_json = Path(args.output_json) if args.output_json else default_output_stem.with_suffix(".json")
    output_csv = Path(args.output_csv) if args.output_csv else default_output_stem.with_suffix(".csv")

    return ExperimentConfig(
        cores=cores,
        repeats=int(args.repeats),
        latency_budget_ms=latency_budget_ms,
        output_json=output_json,
        output_csv=output_csv,
        config_path=Path(args.config),
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for this analysis runner."""

    parser = argparse.ArgumentParser(
        description=(
            "Run Portenta H7 perturbed-model experiments for back-to-back and cadenced "
            "phases on CM7/CM4, then export JSON + CSV summaries."
        )
    )
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "src" / "nas_config.yaml"),
        help="Path to TinyODOM config YAML.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional JSON output path. Default writes under analysis_scripts/cadenced_portenta_h7/results/.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional CSV output path. Default writes under analysis_scripts/cadenced_portenta_h7/results/.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of repeated attempts per core x phase pair.",
    )
    parser.add_argument(
        "--cores",
        nargs="+",
        default=list(VALID_CORES),
        help="Target cores to run (cm7 cm4).",
    )
    parser.add_argument(
        "--latency-budget-ms",
        type=float,
        default=None,
        help=(
            "Cadence budget in milliseconds for the cadenced phase. "
            "Defaults to stride cadence: (stride / sampling_rate_hz) * 1000."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser


def _configure_logging(level_name: str) -> None:
    """Configure root logging used by the analysis runner and TinyODOM modules.

    Parameters
    ----------
    level_name : str
        Logging level name (for example ``INFO`` or ``DEBUG``).
    """

    level = getattr(logging, str(level_name).upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> int:
    """Execute the full cadenced Portenta analysis workflow.

    Returns
    -------
    int
        Process exit code (0 on success).
    """

    parser = _build_arg_parser()
    args = parser.parse_args()
    _configure_logging(args.log_level)

    sys.path.insert(0, os.path.abspath(str(REPO_ROOT / "src")))
    from hil_server import HILServer
    from tinyodom.model import build_tinyodom_model, count_flops

    server = HILServer(config_path=Path(args.config))
    experiment_cfg = _resolve_experiment_config(args, server)
    hyperparams = _build_hyperparams(server, build_tinyodom_model, count_flops)

    attempts: list[Dict[str, Any]] = []
    timestamp_utc = datetime.now(timezone.utc).isoformat()

    for core in experiment_cfg.cores:
        _ensure_portenta_runtime_config(server, core=core)
        for phase in VALID_PHASES:
            for repeat_idx in range(1, experiment_cfg.repeats + 1):
                sketch_path = _stage_phase_sketch(
                    server,
                    phase=phase,
                    latency_budget_ms=experiment_cfg.latency_budget_ms,
                )
                metrics = server.determine_metrics(
                    hyperparams,
                    model_variant=MODEL_VARIANT_NAME,
                )
                record = _build_attempt_record(
                    timestamp_utc=timestamp_utc,
                    core=core,
                    phase=phase,
                    repeat_idx=repeat_idx,
                    latency_budget_ms=experiment_cfg.latency_budget_ms,
                    sketch_template=sketch_path.name,
                    metrics=metrics,
                )
                attempts.append(record)
                print(
                    "run complete:",
                    f"core={core}",
                    f"phase={phase}",
                    f"repeat={repeat_idx}",
                    f"error_code={record['error_code']}",
                    f"latency_ms={record['latency_ms']}",
                    f"energy_mj={record['energy_mj_per_inference']}",
                )

    grouped: dict[tuple[str, str], list[Dict[str, Any]]] = defaultdict(list)
    for record in attempts:
        grouped[(str(record["core"]), str(record["phase"]))].append(record)
    aggregates = [
        _summarize_group(core, phase, group_records, experiment_cfg.latency_budget_ms)
        for (core, phase), group_records in sorted(grouped.items())
    ]

    payload = {
        "metadata": {
            "timestamp_utc": timestamp_utc,
            "config_path": str(experiment_cfg.config_path),
            "model_variant": MODEL_VARIANT_NAME,
            "cores": experiment_cfg.cores,
            "phases": list(VALID_PHASES),
            "repeats": experiment_cfg.repeats,
            "latency_budget_ms_requested": experiment_cfg.latency_budget_ms,
            "notes": [
                "Cadenced phase reports per-window energy in energy_mj_per_inference.",
                "CM4 runs use harness-only runtime telemetry by design.",
            ],
        },
        "attempts": attempts,
        "aggregates": aggregates,
    }

    experiment_cfg.output_json.parent.mkdir(parents=True, exist_ok=True)
    experiment_cfg.output_json.write_text(json.dumps(payload, indent=2))
    _write_csv(experiment_cfg.output_csv, attempts)

    print(f"Wrote JSON: {experiment_cfg.output_json}")
    print(f"Wrote CSV: {experiment_cfg.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
