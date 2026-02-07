#!/usr/bin/env python3
"""
Analyze HIL noise scan CSV output and generate summary stats + plots.

This script groups noise scan metrics and produces:
- Summary CSV files with mean/std/median/min/max per metric.
- Boxplots per metric across groups.
- Run-index time series plots per metric.

If `model_variant` exists in the CSV, grouping is by
(`model_variant`, `input_mode`) as well as legacy `input_mode`.

Examples
--------
python analysis_scripts/hil_noise_analysis/hil_energy_noise_analysis.py --csv hil_energy_noise_scan.csv
python analysis_scripts/hil_noise_analysis/hil_energy_noise_analysis.py --csv hil_energy_noise_scan.csv --out-dir analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


def _numeric_columns(df: pd.DataFrame, exclude: Iterable[str]) -> list[str]:
    """
    Return numeric columns excluding the provided names.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data frame.
    exclude : Iterable[str]
        Column names to skip.

    Returns
    -------
    list[str]
        Numeric column names.
    """
    exclude_set = set(exclude)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    return [col for col in numeric_cols if col not in exclude_set]


def _save_boxplots(df: pd.DataFrame, metrics: list[str], out_dir: Path, group_col: str) -> None:
    """
    Save a boxplot for each metric grouped by the chosen grouping column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data frame with the grouping column.
    metrics : list[str]
        Metric columns to plot.
    out_dir : pathlib.Path
        Output directory for plots.
    group_col : str
        Column name used for grouping on the x-axis.
    """
    for metric in metrics:
        plt.figure(figsize=(8, 4))
        df.boxplot(column=metric, by=group_col)
        plt.title(f"{metric} by {group_col}")
        plt.suptitle("")
        plt.xlabel(group_col)
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(out_dir / f"boxplot_{metric}.png", dpi=150)
        plt.close()


def _save_time_series(df: pd.DataFrame, metrics: list[str], out_dir: Path, has_model_variant: bool) -> None:
    """
    Save per-metric run-index time series plots split by mode/variant.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data frame with ``input_mode`` and ``run_index`` columns.
    metrics : list[str]
        Metric columns to plot.
    out_dir : pathlib.Path
        Output directory for plots.
    has_model_variant : bool
        Whether ``model_variant`` exists in the input CSV.
    """
    if has_model_variant:
        grouped = df.groupby(["model_variant", "input_mode"])
    else:
        grouped = [((None, mode), group) for mode, group in df.groupby("input_mode")]

    for metric in metrics:
        plt.figure(figsize=(9, 4))
        for key, group in grouped:
            model_variant, mode = key
            if "run_index" in group.columns:
                group = group.sort_values("run_index")
                label = f"{model_variant}|{mode}" if has_model_variant else str(mode)
                plt.plot(group["run_index"], group[metric], marker="o", linewidth=1, label=label)
        plt.title(f"{metric} over runs")
        plt.xlabel("run_index")
        plt.ylabel(metric)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"timeseries_{metric}.png", dpi=150)
        plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze HIL noise scan CSV output.")
    parser.add_argument("--csv", required=True, help="Path to hil_energy_noise_scan.csv")
    parser.add_argument(
        "--out-dir",
        default="analysis_scripts/hil_noise_analysis/analysis_output",
        help="Output directory for plots/stats",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if "input_mode" not in df.columns:
        raise ValueError("CSV is missing required 'input_mode' column.")
    has_model_variant = "model_variant" in df.columns
    if has_model_variant:
        df["model_variant"] = df["model_variant"].astype(str)
        df["model_variant_input_mode"] = df["model_variant"] + "|" + df["input_mode"].astype(str)

    preferred_metrics = [
        "latency_ms",
        "energy_mj_per_inference",
        "avg_power_mw",
        "avg_current_ma",
        "idle_power_mw",
        "bus_voltage_v",
    ]
    excluded_metrics = {
        "run_index",
        "ram_bytes",
        "flash_bytes",
        "arena_bytes",
        "latency_budget_ms",
        "hil_enabled",
        "error_code",
    }
    metrics = [m for m in preferred_metrics if m in df.columns]
    if not metrics:
        metrics = _numeric_columns(df, exclude=excluded_metrics)
    metrics = [m for m in metrics if df[m].nunique(dropna=False) > 1]
    if not metrics:
        raise ValueError("No varying numeric metric columns found to analyze.")

    if has_model_variant:
        summary_by_variant_mode = (
            df.groupby(["model_variant", "input_mode"])[metrics]
            .agg(["mean", "std", "median", "min", "max"])
            .sort_index()
        )
        summary_by_variant_mode.to_csv(out_dir / "summary_by_model_variant_input_mode.csv")

    summary_by_input_mode = (
        df.groupby("input_mode")[metrics]
        .agg(["mean", "std", "median", "min", "max"])
        .sort_index()
    )
    summary_by_input_mode.to_csv(out_dir / "summary_by_input_mode.csv")

    group_col = "model_variant_input_mode" if has_model_variant else "input_mode"
    _save_boxplots(df, metrics, out_dir, group_col=group_col)
    if "run_index" in df.columns:
        _save_time_series(df, metrics, out_dir, has_model_variant=has_model_variant)

    print(f"Wrote summary and plots to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
