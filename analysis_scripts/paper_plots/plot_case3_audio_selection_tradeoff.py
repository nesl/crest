#!/usr/bin/env python
"""Plot Case 3 audio score-selection tradeoffs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd


SUCCESS_ERROR_CODE = 1
INFERENCE_BUDGET_MJ = 400.0
SCORE_ENERGY_WEIGHT = 0.10


@dataclass(frozen=True)
class NativeRun:
    """Plot inputs for one Case 3 audio NAS run.

    Parameters
    ----------
    label : str
        Board label used in titles.
    log_path : pathlib.Path
        Native NAS log CSV.
    color : str
        Matplotlib color for this run.

    Attributes
    ----------
    label : str
        Display label used in reports and plots.
    log_path : Path
        Path to the run log used for plotting.
    color : str
        Plot color used for the series.
    """

    label: str
    log_path: Path
    color: str


def numeric_series(frame: pd.DataFrame, column: str, default: float | None = None) -> pd.Series:
    """Return a numeric column with a default for missing data.

    Parameters
    ----------
    frame : pandas.DataFrame
        Source frame.
    column : str
        Column to parse.
    default : float or None, optional
        Fallback value when the column is absent.

    Returns
    -------
    pandas.Series
        Numeric series aligned to ``frame``.
    """
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series([default] * len(frame), index=frame.index, dtype="float64")


def boolean_series(frame: pd.DataFrame, column: str, default: bool) -> pd.Series:
    """Return a boolean column parsed from common CSV string encodings.

    Parameters
    ----------
    frame : pandas.DataFrame
        Source frame.
    column : str
        Column to parse.
    default : bool
        Fallback value when the column is absent.

    Returns
    -------
    pandas.Series
        Boolean series aligned to ``frame``.
    """
    if column not in frame.columns:
        return pd.Series([default] * len(frame), index=frame.index, dtype="bool")
    raw = frame[column]
    if raw.dtype == bool:
        return raw.fillna(default)
    return raw.astype(str).str.lower().isin({"true", "1", "yes"})


def valid_scored_trials(frame: pd.DataFrame) -> pd.DataFrame:
    """Filter a NAS log to successful, feasible, finite-score trials.

    Parameters
    ----------
    frame : pandas.DataFrame
        Raw NAS log frame.

    Returns
    -------
    pandas.DataFrame
        Filtered frame preserving original row indices.
    """
    error_code = numeric_series(frame, "error_code")
    pruned = boolean_series(frame, "pruned", False)
    feasible = boolean_series(frame, "feasible", True)
    score = numeric_series(frame, "score")
    macro_f1 = numeric_series(frame, "metric__macro_f1")
    energy = numeric_series(frame, "energy_mj_per_inference")
    valid = error_code.eq(SUCCESS_ERROR_CODE)
    valid &= ~pruned
    valid &= feasible
    valid &= np.isfinite(score)
    valid &= np.isfinite(macro_f1)
    valid &= np.isfinite(energy)
    valid &= score > -1000.0
    valid &= energy > 0.0
    return frame[valid].copy()


def configure_matplotlib() -> None:
    """Apply compact paper-figure styling consistent with Case 3 plots.

    Returns
    -------
    None
        Updates process-wide Matplotlib rcParams.
    """
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 400,
            "font.size": 9.5,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.5,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.grid": True,
            "grid.alpha": 0.24,
            "grid.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def selected_rows(valid: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return score-selected, highest macro-F1, and lowest-energy rows.

    Parameters
    ----------
    valid : pandas.DataFrame
        Valid scored NAS trials.

    Returns
    -------
    tuple[pandas.Series, pandas.Series, pandas.Series]
        Score-selected, highest-macro-F1, and lowest-energy rows.
    """
    score_selected = valid.loc[numeric_series(valid, "score").idxmax()]
    highest_f1 = valid.loc[numeric_series(valid, "metric__macro_f1").idxmax()]
    lowest_energy = valid.loc[numeric_series(valid, "energy_mj_per_inference").idxmin()]
    return score_selected, highest_f1, lowest_energy


def row_summary(run: NativeRun, label: str, row: pd.Series) -> dict[str, float | int | str]:
    """Summarize one highlighted trial row for CSV output.

    Parameters
    ----------
    run : NativeRun
        Run metadata.
    label : str
        Selection label.
    row : pandas.Series
        Highlighted NAS log row.

    Returns
    -------
    dict[str, float | int | str]
        CSV-ready summary row.
    """
    return {
        "board": run.label,
        "selection": label,
        "log_row_index": int(row.name),
        "score": float(row["score"]),
        "macro_f1": float(row["metric__macro_f1"]),
        "accuracy": float(row["metric__accuracy"]),
        "energy_mj_per_inference": float(row["energy_mj_per_inference"]),
        "latency_ms": float(row["latency_ms"]),
        "ram_bytes": float(row.get("ram_bytes", np.nan)),
        "flash_bytes": float(row.get("flash_bytes", np.nan)),
        "external_flash_bytes": float(row.get("external_flash_bytes", np.nan)),
    }


def plot_run(
    axis: plt.Axes,
    run: NativeRun,
    *,
    log_x: bool = False,
    color_by_trial: bool = False,
    trial_norm: Normalize | None = None,
    trial_cmap: str = "viridis",
) -> list[dict[str, float | int | str]]:
    """Plot one board's macro-F1/energy tradeoff panel.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Target axes.
    run : NativeRun
        Run metadata and log path.
    log_x : bool, default=False
        Whether to use log scaling on energy.
    color_by_trial : bool, default=False
        Whether to color points by trial index.
    trial_norm : matplotlib.colors.Normalize or None, optional
        Color normalization for trial-index coloring.
    trial_cmap : str, default="viridis"
        Colormap for trial-index coloring.

    Returns
    -------
    list[dict[str, float | int | str]]
        Highlighted-row summaries.
    """
    raw = pd.read_csv(run.log_path)
    valid = valid_scored_trials(raw)
    score_selected, highest_f1, lowest_energy = selected_rows(valid)
    energy = numeric_series(valid, "energy_mj_per_inference")
    macro_f1 = numeric_series(valid, "metric__macro_f1")

    if color_by_trial:
        trial_order = pd.Series(valid.index, index=valid.index, dtype="float64")
        mappable = axis.scatter(
            energy,
            macro_f1,
            s=20,
            c=trial_order,
            cmap=trial_cmap,
            norm=trial_norm,
            alpha=0.58,
            linewidths=0,
            label="Feasible scored trial",
        )
        axis._case3_trial_mappable = mappable
    else:
        axis.scatter(
            energy,
            macro_f1,
            s=20,
            color=run.color,
            alpha=0.32,
            linewidths=0,
            label="Feasible scored trial",
        )
    axis.axvline(
        INFERENCE_BUDGET_MJ,
        color="0.25",
        linestyle="--",
        linewidth=1.0,
        alpha=0.8,
        label="400 mJ allocation",
    )

    highlights = [
        ("Score-selected", score_selected, "*", 145, "0.15", run.color if not color_by_trial else "white"),
        ("Highest macro-F1", highest_f1, "s", 60, "0.15", "white"),
        ("Lowest energy", lowest_energy, "^", 74, "0.35", "white"),
    ]
    for label, row, marker, size, edge_color, face_color in highlights:
        axis.scatter(
            row["energy_mj_per_inference"],
            row["metric__macro_f1"],
            s=size,
            marker=marker,
            facecolor=face_color,
            edgecolor=edge_color,
            linewidth=1.35,
            zorder=4,
        )

    axis.set_title(run.label)
    axis.set_xlabel("Measured inference energy (mJ)")
    axis.set_ylabel("Validation macro-F1")
    if log_x:
        axis.set_xscale("log")
        axis.set_xlim(max(1.0, float(energy.min()) * 0.75), float(energy.max()) * 1.25)
    else:
        axis.set_xlim(0, max(850.0, float(energy.max()) * 1.06))
    axis.set_ylim(0.0, min(0.86, max(0.84, float(macro_f1.max()) + 0.025)))
    axis.grid(True, alpha=0.3)

    return [
        row_summary(run, "score_selected", score_selected),
        row_summary(run, "highest_macro_f1", highest_f1),
        row_summary(run, "lowest_energy", lowest_energy),
    ]


def plot_selection_tradeoff(
    native_runs: Sequence[NativeRun],
    output_dir: Path,
    output_stem: str,
    *,
    log_x: bool = False,
    vertical: bool = False,
    color_by_trial: bool = False,
) -> None:
    """Plot the Case 3 macro-F1/energy score-selection tradeoff.

    Parameters
    ----------
    native_runs : Sequence[NativeRun]
        Native NAS runs to plot.
    output_dir : pathlib.Path
        Destination directory.
    output_stem : str
        Output filename stem.
    log_x : bool, default=False
        Whether to use log scaling on energy.
    vertical : bool, default=False
        Whether to stack panels vertically.
    color_by_trial : bool, default=False
        Whether to color trials by NAS log row index.
    """
    if vertical:
        fig, axes = plt.subplots(len(native_runs), 1, figsize=(3.45, 5.15), sharex=True, sharey=True)
    else:
        fig, axes = plt.subplots(1, len(native_runs), figsize=(6.9, 3.05), sharey=True)
    if len(native_runs) == 1:
        axes = [axes]

    trial_norm = None
    trial_cmap = "plasma"
    if color_by_trial:
        max_trial_index = 0
        for run in native_runs:
            raw = pd.read_csv(run.log_path)
            valid = valid_scored_trials(raw)
            max_trial_index = max(max_trial_index, int(valid.index.max()))
        trial_norm = Normalize(vmin=0, vmax=max_trial_index)

    summaries: list[dict[str, float | int | str]] = []
    for index, (axis, run) in enumerate(zip(axes, native_runs)):
        summaries.extend(
            plot_run(
                axis,
                run,
                log_x=log_x,
                color_by_trial=color_by_trial,
                trial_norm=trial_norm,
                trial_cmap=trial_cmap,
            )
        )
        if vertical and index < len(native_runs) - 1:
            axis.set_xlabel("")
            axis.tick_params(axis="x", labelbottom=False)
        if not vertical and index > 0:
            axis.set_ylabel("")
            axis.tick_params(axis="y", labelleft=False)

    if vertical:
        legend_handles = [
            Line2D([0], [0], marker="*", color="0.15", markerfacecolor="0.55", linestyle="None", markersize=9, label="Score-selected"),
            Line2D([0], [0], marker="s", color="0.15", markerfacecolor="white", linestyle="None", markersize=5.5, label="Highest macro-F1"),
            Line2D([0], [0], color="0.25", linestyle="--", linewidth=1.0, label="400 mJ allocation"),
            Line2D([0], [0], marker="^", color="0.35", markerfacecolor="white", linestyle="None", markersize=6, label="Lowest energy"),
        ]
    else:
        legend_handles = [
            Line2D([0], [0], marker="o", color="0.55", linestyle="None", markersize=4, label="Feasible scored trial"),
            Line2D([0], [0], color="0.25", linestyle="--", linewidth=1.0, label="400 mJ allocation"),
            Line2D([0], [0], marker="*", color="0.15", markerfacecolor="0.55", linestyle="None", markersize=9, label="Score-selected"),
            Line2D([0], [0], marker="s", color="0.15", markerfacecolor="white", linestyle="None", markersize=5.5, label="Highest macro-F1"),
            Line2D([0], [0], marker="^", color="0.35", markerfacecolor="white", linestyle="None", markersize=6, label="Lowest energy"),
        ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965 if vertical else 0.98),
        ncols=2 if vertical else len(legend_handles),
        frameon=False,
        columnspacing=0.9,
        handlelength=1.3,
        handletextpad=0.45,
    )
    if color_by_trial:
        mappable = next(
            getattr(axis, "_case3_trial_mappable", None)
            for axis in axes
            if getattr(axis, "_case3_trial_mappable", None) is not None
        )
        colorbar = fig.colorbar(mappable, ax=axes, fraction=0.035, pad=0.025)
        colorbar.set_label("Trial index")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90 if vertical else 0.912))

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_stem = f"{output_stem}_summary"
    pd.DataFrame(summaries).to_csv(output_dir / f"{summary_stem}.csv", index=False)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{output_stem}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def plot_selection_tradeoff_v2(
    native_runs: Sequence[NativeRun],
    output_dir: Path,
    output_stem: str,
) -> None:
    """Plot a narrower side-by-side selection tradeoff clipped at 800 mJ.

    Parameters
    ----------
    native_runs : Sequence[NativeRun]
        Native NAS runs to plot.
    output_dir : pathlib.Path
        Destination directory.
    output_stem : str
        Output filename stem.
    """
    with plt.rc_context(
        {
            "font.size": 13.8,
            "axes.labelsize": 13.8,
            "axes.titlesize": 15.0,
            "axes.titlepad": 0.5,
            "axes.labelpad": 0.0,
            "legend.fontsize": 11.8,
            "xtick.labelsize": 12.6,
            "ytick.labelsize": 12.6,
            "xtick.major.pad": 0.5,
            "ytick.major.pad": 0.5,
        }
    ):
        fig, axes = plt.subplots(1, len(native_runs), figsize=(6.15, 3.05), sharey=True)
        if len(native_runs) == 1:
            axes = [axes]

        summaries: list[dict[str, float | int | str]] = []
        for index, (axis, run) in enumerate(zip(axes, native_runs)):
            summaries.extend(plot_run(axis, run))
            axis.set_xlim(0, 800)
            axis.set_xticks([0, 200, 400, 600, 800])
            axis.set_xlabel("Inference energy (mJ)")
            if index > 0:
                axis.set_ylabel("")
                axis.tick_params(axis="y", labelleft=False)

        legend_handles = [
            Line2D([0], [0], color="0.25", linestyle="--", linewidth=1.0, label="400 mJ allocation"),
            Line2D([0], [0], marker="*", color="0.15", markerfacecolor="0.55", linestyle="None", markersize=9, label="Selected"),
            Line2D([0], [0], marker="s", color="0.15", markerfacecolor="white", linestyle="None", markersize=5.5, label="Highest macro-F1"),
            Line2D([0], [0], marker="^", color="0.35", markerfacecolor="white", linestyle="None", markersize=6, label="Lowest energy"),
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.998),
            ncols=len(legend_handles),
            frameon=False,
            columnspacing=0.6,
            handlelength=1.0,
            handletextpad=0.35,
        )
        fig.tight_layout(rect=(-0.015, -0.055, 1.015, 0.94), w_pad=0.45)

        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summaries).to_csv(output_dir / f"{output_stem}_summary.csv", index=False)
        for suffix in ("png", "pdf"):
            fig.savefig(output_dir / f"{output_stem}.{suffix}")
        plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """
    parser = argparse.ArgumentParser(description="Plot Case 3 audio macro-F1/energy selection tradeoffs.")
    parser.add_argument("--portenta-log", type=Path, required=True)
    parser.add_argument("--stm-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-stem", required=True, help="Output filename stem.")
    parser.add_argument("--log-x", action="store_true", help="Use a log scale for measured inference energy.")
    parser.add_argument("--vertical", action="store_true", help="Stack board panels vertically for a single-column figure.")
    parser.add_argument("--color-by-trial", action="store_true", help="Color feasible scored trials by NAS log row index.")
    parser.add_argument("--v2", action="store_true", help="Create compact single-column side-by-side figure clipped at 800 mJ.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the plotting entry point.

    Parameters
    ----------
    argv : Sequence[str] or None, optional
        Command-line arguments excluding the executable name.

    Returns
    -------
    int
        Process exit code.
    """
    args = build_arg_parser().parse_args(argv)
    configure_matplotlib()
    native_runs = [
        NativeRun("Portenta H7 CM7", args.portenta_log.expanduser(), "#D55E00"),
        NativeRun("STM32 N657", args.stm_log.expanduser(), "#0072B2"),
    ]
    output_dir = args.output_dir.expanduser().resolve()
    if args.v2:
        plot_selection_tradeoff_v2(native_runs, output_dir, args.output_stem)
    else:
        plot_selection_tradeoff(
            native_runs,
            output_dir,
            args.output_stem,
            log_x=args.log_x,
            vertical=args.vertical,
            color_by_trial=args.color_by_trial,
        )
    print(f"Wrote Case 3 audio selection tradeoff figure to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
