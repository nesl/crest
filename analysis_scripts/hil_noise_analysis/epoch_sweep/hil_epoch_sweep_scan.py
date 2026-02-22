#!/usr/bin/env python3
"""
Run HIL metrics over epoch-sweep checkpoints produced by train_epoch_sweep.py.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from analysis_scripts.hil_noise_analysis.noise_scan_model_spec import build_noise_scan_hyperparams
from hil_server import HILServer, logger
from tinyodom.model import DEFAULT_CONFIG_PATH


def _configure_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(levelname)s:%(name)s:%(message)s",
    )


def _parse_csv_list(raw: str, field_name: str) -> list[str]:
    values = [value.strip().lower() for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError(f"No {field_name} provided.")
    return values


def _to_utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_epoch_filter(raw: str | None) -> set[int] | None:
    """
    Parse comma-separated epoch selectors and ranges (e.g., "50,100-200,350").
    """
    if raw is None:
        return None

    selected: set[int] = set()
    for token in [part.strip() for part in raw.split(",") if part.strip()]:
        if "-" in token:
            left, right = token.split("-", 1)
            start = int(left)
            end = int(right)
            if start <= 0 or end <= 0:
                raise ValueError(f"Invalid non-positive epoch range: {token}")
            if end < start:
                raise ValueError(f"Invalid descending epoch range: {token}")
            selected.update(range(start, end + 1))
        else:
            epoch = int(token)
            if epoch <= 0:
                raise ValueError(f"Invalid non-positive epoch: {token}")
            selected.add(epoch)
    return selected


def _load_training_rows(training_csv: Path, selected_epochs: set[int] | None) -> list[dict[str, str]]:
    """
    Load checkpoint-bearing rows from the training CSV.

    Rows without a checkpoint path (for example audit-only rows) are skipped.
    """
    if not training_csv.exists():
        raise FileNotFoundError(f"Training CSV not found: {training_csv}")

    with training_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"epoch", "stage_type", "checkpoint_path"}
        if not required_columns.issubset(set(reader.fieldnames or [])):
            missing = sorted(required_columns - set(reader.fieldnames or []))
            raise ValueError(f"Training CSV missing required columns: {missing}")

        rows: list[dict[str, str]] = []
        for row_index, row in enumerate(reader, start=2):
            try:
                epoch = int(str(row["epoch"]).strip())
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid epoch value in training CSV row {row_index}: {row.get('epoch')!r}"
                ) from exc
            if selected_epochs is not None and epoch not in selected_epochs:
                continue
            checkpoint_path = str(row.get("checkpoint_path", "")).strip()
            if not checkpoint_path:
                stage_type = str(row.get("stage_type", "")).strip().lower()
                if stage_type in {"fresh_untrained_audit"}:
                    logger.info(
                        "Skipping non-checkpoint training CSV row %d (stage_type=%s, epoch=%s).",
                        row_index,
                        stage_type,
                        epoch,
                    )
                else:
                    logger.warning(
                        "Skipping training CSV row %d with empty checkpoint_path (stage_type=%s, epoch=%s).",
                        row_index,
                        stage_type or "<missing>",
                        epoch,
                    )
                continue
            rows.append(row)

    rows.sort(key=lambda item: int(item["epoch"]))
    if not rows:
        raise ValueError("No checkpoint rows selected from training CSV.")
    return rows


def _resolve_checkpoint_path(raw_path: str, training_csv_dir: Path, checkpoint_root: Path | None) -> Path:
    """
    Resolve checkpoint paths across machines.

    Resolution order:
    1) Path as written in the CSV.
    2) Relative to ``--checkpoint-root`` (if provided).
    3) Filename inside ``--checkpoint-root`` (if provided).
    4) Relative to training CSV directory.
    5) Filename inside training CSV directory.
    """
    candidate_paths: list[Path] = []
    raw = Path(raw_path)

    if raw.is_absolute():
        candidate_paths.append(raw)
    else:
        candidate_paths.append(Path.cwd() / raw)

    if checkpoint_root is not None:
        if raw.is_absolute():
            candidate_paths.append(checkpoint_root / raw.name)
        else:
            candidate_paths.append(checkpoint_root / raw)
            candidate_paths.append(checkpoint_root / raw.name)

    if raw.is_absolute():
        candidate_paths.append(training_csv_dir / raw.name)
    else:
        candidate_paths.append(training_csv_dir / raw)
        candidate_paths.append(training_csv_dir / raw.name)

    seen: set[Path] = set()
    for candidate in candidate_paths:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    checked = "\n".join(f"- {path.expanduser().resolve()}" for path in candidate_paths)
    raise FileNotFoundError(
        "Checkpoint not found after path remap attempts.\n"
        f"Original CSV path: {raw_path}\n"
        f"Tried:\n{checked}"
    )


def _write_summary(summary_csv_path: Path, run_rows: list[dict[str, object]]) -> None:
    if not run_rows:
        return

    group_key_fields = ["epoch", "stage_type", "checkpoint_path", "input_mode"]
    numeric_metric_fields = [
        "ram_bytes",
        "flash_bytes",
        "latency_ms",
        "latency_budget_ms",
        "arena_bytes",
        "energy_mj_per_inference",
        "avg_power_mw",
        "avg_current_ma",
        "bus_voltage_v",
        "idle_power_mw",
        "harness_latency_ms",
        "error_code",
    ]

    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in run_rows:
        group_key = tuple(row[field] for field in group_key_fields)
        grouped[group_key].append(row)

    summary_rows: list[dict[str, object]] = []
    for group_key in sorted(grouped.keys()):
        group_items = grouped[group_key]
        row = {field: value for field, value in zip(group_key_fields, group_key)}
        row["n_runs"] = len(group_items)

        for metric in numeric_metric_fields:
            values: list[float] = []
            for item in group_items:
                value = item.get(metric)
                if value is None:
                    continue
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(numeric):
                    values.append(numeric)
            if values:
                row[f"{metric}_mean"] = statistics.fmean(values)
                row[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            else:
                row[f"{metric}_mean"] = ""
                row[f"{metric}_std"] = ""
        summary_rows.append(row)

    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run HIL metrics over epoch-sweep checkpoints.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config YAML.")
    parser.add_argument(
        "--training-csv",
        required=True,
        help="CSV emitted by train_epoch_sweep.py containing checkpoint paths.",
    )
    parser.add_argument(
        "--csv-path",
        default=None,
        help="Output per-run HIL CSV path. Defaults to <training-csv-dir>/epoch_sweep_hil_metrics.csv",
    )
    parser.add_argument(
        "--summary-csv-path",
        default=None,
        help="Output checkpoint summary CSV path. Defaults to <training-csv-dir>/epoch_sweep_hil_summary.csv",
    )
    parser.add_argument("--runs", type=int, default=1, help="HIL runs per checkpoint and input mode.")
    parser.add_argument(
        "--input-modes",
        default="uniform",
        help="Comma-separated input modes for HIL sweeps.",
    )
    parser.add_argument("--cooldown", type=float, default=0.0, help="Cooldown seconds between runs.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable INFO-level logging for sweep progress and HIL internals.",
    )
    parser.add_argument(
        "--energy-aware",
        action="store_true",
        help="Force energy-aware sketch selection even if config disables it.",
    )
    parser.add_argument(
        "--epoch-filter",
        default=None,
        help="Optional epoch filter list/range (e.g., 50,100-200,300).",
    )
    parser.add_argument(
        "--checkpoint-root",
        default=None,
        help="Optional directory used to remap checkpoint paths when CSV paths come from another machine.",
    )
    args = parser.parse_args()
    _configure_logging(args.verbose)

    if args.runs <= 0:
        raise ValueError("--runs must be > 0")
    if args.cooldown < 0:
        raise ValueError("--cooldown must be >= 0")

    training_csv = Path(args.training_csv).resolve()
    checkpoint_root = Path(args.checkpoint_root).resolve() if args.checkpoint_root else None
    selected_epochs = _parse_epoch_filter(args.epoch_filter)
    checkpoints = _load_training_rows(training_csv, selected_epochs)
    input_modes = _parse_csv_list(args.input_modes, "input modes")

    csv_path = Path(args.csv_path).resolve() if args.csv_path else training_csv.parent / "epoch_sweep_hil_metrics.csv"
    summary_csv_path = (
        Path(args.summary_csv_path).resolve()
        if args.summary_csv_path
        else training_csv.parent / "epoch_sweep_hil_summary.csv"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)

    server = HILServer(config_path=Path(args.config))
    if args.energy_aware:
        server.config.training.energy_aware = True
        server.set_input_mode(server.config.training.get("input_mode", "uniform"))

    hyperparams = build_noise_scan_hyperparams(
        window_size=server.config.data.window_size,
        input_dim=server.training_data.inputs.shape[2],
    )

    run_rows: list[dict[str, object]] = []
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = None

        for checkpoint_row in checkpoints:
            epoch = int(checkpoint_row["epoch"])
            stage_type = str(checkpoint_row.get("stage_type", ""))
            checkpoint_path = _resolve_checkpoint_path(
                str(checkpoint_row["checkpoint_path"]),
                training_csv.parent,
                checkpoint_root,
            )

            model_variant = f"trained_epoch_{epoch}"
            logger.info("Evaluating checkpoint epoch=%s stage_type=%s path=%s", epoch, stage_type, checkpoint_path)

            for mode in input_modes:
                server.set_input_mode(mode)
                for run_idx in range(1, args.runs + 1):
                    metrics = server.determine_metrics(
                        hyperparams,
                        checkpoint_path=checkpoint_path,
                        model_variant=model_variant,
                    )
                    row = {
                        "epoch": epoch,
                        "stage_type": stage_type,
                        "checkpoint_path": str(checkpoint_path),
                        "input_mode": mode,
                        "run_index": run_idx,
                        "model_variant": model_variant,
                        "timestamp_utc": _to_utc_timestamp(),
                        **metrics,
                    }
                    if writer is None:
                        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
                        writer.writeheader()
                    writer.writerow(row)
                    handle.flush()
                    run_rows.append(row)

                    if run_idx < args.runs and args.cooldown > 0:
                        time.sleep(args.cooldown)

    _write_summary(summary_csv_path, run_rows)
    print(f"Wrote per-run HIL metrics CSV: {csv_path}")
    print(f"Wrote checkpoint summary CSV: {summary_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
