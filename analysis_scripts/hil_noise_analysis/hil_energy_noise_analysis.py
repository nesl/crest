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


def _has_untrained_and_trained(df: pd.DataFrame) -> bool:
    """Return True when both untrained and trained* model variants are present."""
    if "model_variant" not in df.columns:
        return False
    variants = df["model_variant"].astype(str).str.lower()
    has_untrained = (variants == "untrained").any()
    has_trained = variants.str.startswith("trained").any()
    return bool(has_untrained and has_trained)


def _group_col_for_subset(subset: pd.DataFrame, default_group_col: str) -> str:
    """
    Pick an x-axis grouping column for a model-variant subset.

    For a single model variant this uses input_mode; with multiple variants it
    keeps model_variant_input_mode so labels remain unique.
    """
    if "model_variant" not in subset.columns:
        return default_group_col
    if subset["model_variant"].astype(str).nunique() <= 1:
        return "input_mode"
    return "model_variant_input_mode" if "model_variant_input_mode" in subset.columns else default_group_col


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
    split_variants = _has_untrained_and_trained(df)
    if split_variants:
        variant_series = df["model_variant"].astype(str).str.lower()
        untrained_df = df.loc[variant_series == "untrained"].copy()
        trained_df = df.loc[variant_series.str.startswith("trained")].copy()

        for metric in metrics:
            fig, axes = plt.subplots(2, 1, figsize=(9, 8))
            top_group_col = _group_col_for_subset(untrained_df, group_col)
            bottom_group_col = _group_col_for_subset(trained_df, group_col)

            untrained_df.boxplot(column=metric, by=top_group_col, ax=axes[0])
            axes[0].set_title(f"untrained | {metric}")
            axes[0].set_xlabel(top_group_col)
            axes[0].set_ylabel(metric)

            trained_df.boxplot(column=metric, by=bottom_group_col, ax=axes[1])
            axes[1].set_title(f"trained* | {metric}")
            axes[1].set_xlabel(bottom_group_col)
            axes[1].set_ylabel(metric)

            fig.suptitle("")
            fig.tight_layout()
            fig.savefig(out_dir / f"boxplot_{metric}.png", dpi=150)
            plt.close(fig)
        return

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(8, 4))
        df.boxplot(column=metric, by=group_col, ax=ax)
        ax.set_title(f"{metric} by {group_col}")
        ax.set_xlabel(group_col)
        ax.set_ylabel(metric)
        fig.suptitle("")
        fig.tight_layout()
        fig.savefig(out_dir / f"boxplot_{metric}.png", dpi=150)
        plt.close(fig)


def _plot_timeseries_on_axis(ax, data: pd.DataFrame, metric: str, has_model_variant: bool) -> None:
    """Plot time series lines for one axis and one metric."""
    if has_model_variant:
        grouped = data.groupby(["model_variant", "input_mode"])
    else:
        grouped = [((None, mode), group) for mode, group in data.groupby("input_mode")]

    for key, group in grouped:
        model_variant, mode = key
        if "run_index" not in group.columns:
            continue
        group = group.sort_values("run_index")
        if has_model_variant and data["model_variant"].astype(str).nunique() > 1:
            label = f"{model_variant}|{mode}"
        else:
            label = str(mode)
        ax.plot(group["run_index"], group[metric], marker="o", linewidth=1, label=label)


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
    split_variants = has_model_variant and _has_untrained_and_trained(df)
    if split_variants:
        variant_series = df["model_variant"].astype(str).str.lower()
        untrained_df = df.loc[variant_series == "untrained"].copy()
        trained_df = df.loc[variant_series.str.startswith("trained")].copy()

        for metric in metrics:
            fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            _plot_timeseries_on_axis(axes[0], untrained_df, metric, has_model_variant=True)
            axes[0].set_title(f"untrained | {metric} over runs")
            axes[0].set_ylabel(metric)
            axes[0].legend()

            _plot_timeseries_on_axis(axes[1], trained_df, metric, has_model_variant=True)
            axes[1].set_title(f"trained* | {metric} over runs")
            axes[1].set_xlabel("run_index")
            axes[1].set_ylabel(metric)
            axes[1].legend()

            fig.tight_layout()
            fig.savefig(out_dir / f"timeseries_{metric}.png", dpi=150)
            plt.close(fig)
        return

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(9, 4))
        _plot_timeseries_on_axis(ax, df, metric, has_model_variant=has_model_variant)
        ax.set_title(f"{metric} over runs")
        ax.set_xlabel("run_index")
        ax.set_ylabel(metric)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"timeseries_{metric}.png", dpi=150)
        plt.close(fig)


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
