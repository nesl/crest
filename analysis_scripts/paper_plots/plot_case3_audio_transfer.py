#!/usr/bin/env python
"""Plot Case 3 audio NAS progress and cross-board replay transfer."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd


SUCCESS_ERROR_CODE = 1

def find_repo_root() -> Path:
    """Return the repository root containing ``src/config``.

    Returns
    -------
    pathlib.Path
        Repository root path.
    """

    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "config").is_dir():
            return parent
    return Path(__file__).resolve().parents[2]


REPO_ROOT = find_repo_root()
CASE3_ROOT = REPO_ROOT / "models" / "crest_case_3"
DEFAULT_OUTPUT_DIR = CASE3_ROOT / "comparisons" / "case3_audio_cross_board_transfer"
DEFAULT_STM_LOG = (
    CASE3_ROOT
    / "nas_runs"
    / "UrbanSound8K_DSCNN_STM32_AUDIO_case3_2_t1"
    / "log_NAS_UrbanSound8K_DSCNN_STM32_AUDIO_case3_2_t1.csv"
)
DEFAULT_PORTENTA_LOG = (
    CASE3_ROOT
    / "nas_runs"
    / "UrbanSound8K_DSCNN_PORTENTA_M7_AUDIO_case3_1_t1"
    / "log_NAS_UrbanSound8K_DSCNN_PORTENTA_M7_AUDIO_case3_1_t1.csv"
)
DEFAULT_STM_ON_PORTENTA_REPLAY = (
    CASE3_ROOT / "replays" / "case3_2_stm32_best_on_portenta_m7" / "replay_results.csv"
)
DEFAULT_PORTENTA_ON_STM_REPLAY = (
    CASE3_ROOT / "replays" / "case3_1_portenta_best_on_stm32" / "replay_results.csv"
)


@dataclass(frozen=True)
class NativeRun:
    """Plot inputs for one native NAS run.

    Parameters
    ----------
    label : str
        Short board label used in legends.
    log_path : pathlib.Path
        Native NAS log CSV path.
    color : str
        Matplotlib color for this board.
    marker : str
        Marker used for native points from this board.
    """

    label: str
    log_path: Path
    color: str
    marker: str


@dataclass(frozen=True)
class TransferPoint:
    """One point in the cross-board transfer plot.

    Parameters
    ----------
    architecture : str
        Architecture-selection source, either STM-selected or Portenta-selected.
    measurement_board : str
        Board on which hardware energy/latency was measured.
    point_type : str
        ``"Native optimum"`` or ``"Replay"``.
    energy_mj : float
        Measured energy per inference in millijoules.
    latency_ms : float
        Measured latency in milliseconds.
    accuracy : float
        Validation accuracy associated with the selected architecture.
    macro_f1 : float
        Validation macro-F1 associated with the selected architecture.
    score : float
        NAS score associated with the selected architecture.
    row_index : int
        Source CSV row index.
    """

    architecture: str
    measurement_board: str
    point_type: str
    energy_mj: float
    latency_ms: float
    accuracy: float
    macro_f1: float
    score: float
    row_index: int


def numeric_series(frame: pd.DataFrame, column: str, default: float | None = None) -> pd.Series:
    """Return a numeric column with a default for missing data.

    Parameters
    ----------
    frame : pandas.DataFrame
        Data frame to read.
    column : str
        Column name.
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


def valid_successful_trials(frame: pd.DataFrame) -> pd.DataFrame:
    """Filter a NAS log to successful, unpruned, finite-score trials.

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
    pruned = frame.get("pruned", pd.Series([False] * len(frame), index=frame.index))
    pruned = pruned.astype(str).str.lower().isin({"true", "1", "yes"})
    score = numeric_series(frame, "score")
    valid = error_code.eq(SUCCESS_ERROR_CODE) & ~pruned & np.isfinite(score)
    valid &= score > -1000.0
    return frame[valid].copy()


def best_trial(frame: pd.DataFrame) -> pd.Series:
    """Return the highest-score valid trial from a NAS log.

    Parameters
    ----------
    frame : pandas.DataFrame
        Raw NAS log frame.

    Returns
    -------
    pandas.Series
        Best valid trial row.
    """

    valid = valid_successful_trials(frame)
    if valid.empty:
        raise ValueError("No successful finite-score trials found.")
    return valid.loc[numeric_series(valid, "score").idxmax()]


def configure_matplotlib() -> None:
    """Apply compact paper-figure styling.

    Returns
    -------
    None
        Updates Matplotlib process-wide rcParams.
    """

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 11.6,
            "axes.labelsize": 11.6,
            "axes.titlesize": 12.7,
            "legend.fontsize": 10.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.grid": True,
            "grid.alpha": 0.24,
            "grid.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_score_progress(native_runs: Sequence[NativeRun], output_dir: Path) -> None:
    """Plot score-over-trial progress for the native Case 3 searches.

    Parameters
    ----------
    native_runs : Sequence[NativeRun]
        Native NAS runs to plot.
    output_dir : pathlib.Path
        Directory for generated figure files.
    """

    fig, axes = plt.subplots(
        len(native_runs),
        1,
        figsize=(5.7, 4.9),
        sharex=True,
        sharey=True,
    )
    if len(native_runs) == 1:
        axes = [axes]

    max_trial = 1.0
    for axis, run in zip(axes, native_runs):
        raw = pd.read_csv(run.log_path)
        valid = valid_successful_trials(raw)
        valid = valid.sort_values("timestamp_unix").copy()
        x_values = np.arange(1, len(valid) + 1, dtype=float)
        scores = numeric_series(valid, "score").to_numpy(dtype=float)
        best_so_far = np.maximum.accumulate(scores)
        best_position = int(np.nanargmax(scores))
        best_score = float(scores[best_position])
        max_trial = max(max_trial, float(np.nanmax(x_values)))

        axis.scatter(
            x_values,
            scores,
            s=22,
            color=run.color,
            alpha=0.22,
            linewidths=0,
            label="Completed trial",
        )
        axis.plot(
            x_values,
            best_so_far,
            color=run.color,
            linewidth=2.2,
            label="Best so far",
        )
        axis.scatter(
            [x_values[best_position]],
            [best_score],
            s=92,
            marker="*",
            color="white",
            edgecolor=run.color,
            linewidth=1.35,
            zorder=4,
            label="Selected optimum",
        )
        axis.annotate(
            f"{best_score:.3f}",
            xy=(x_values[best_position], best_score),
            xytext=(5, 8),
            textcoords="offset points",
            color=run.color,
            fontsize=7.5,
        )
        axis.set_title(run.label)
        axis.set_ylabel("NAS score")
        axis.grid(True, alpha=0.3)
        axis.set_ylim(0.30, 0.80)

    axes[-1].set_xlabel("Completed trial")
    axes[0].set_xlim(0, max(max_trial * 1.04, 1.0))
    legend_handles = [
        Line2D([0], [0], marker="o", color="0.55", linestyle="None", markersize=4, label="Completed trial"),
        Line2D([0], [0], color="0.25", linewidth=1.7, label="Best so far"),
        Line2D(
            [0],
            [0],
            marker="*",
            color="0.25",
            markerfacecolor="white",
            linestyle="None",
            markersize=8,
            label="Selected optimum",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncols=3,
        frameon=False,
        columnspacing=1.3,
        handlelength=1.8,
        handletextpad=0.5,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))

    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"case3_score_progress_subplots.{suffix}", bbox_inches="tight")
    plt.close(fig)

def transfer_points(
    *,
    stm_log: Path,
    portenta_log: Path,
    stm_on_portenta_replay: Path,
    portenta_on_stm_replay: Path,
) -> list[TransferPoint]:
    """Build the four highlighted native/replay transfer points.

    Parameters
    ----------
    stm_log : pathlib.Path
        STM native NAS log CSV.
    portenta_log : pathlib.Path
        Portenta native NAS log CSV.
    stm_on_portenta_replay : pathlib.Path
        Replay CSV for the STM-selected architecture measured on Portenta.
    portenta_on_stm_replay : pathlib.Path
        Replay CSV for the Portenta-selected architecture measured on STM.

    Returns
    -------
    list[TransferPoint]
        Four plot-ready transfer points.
    """

    stm_frame = pd.read_csv(stm_log)
    portenta_frame = pd.read_csv(portenta_log)
    stm_best = best_trial(stm_frame)
    portenta_best = best_trial(portenta_frame)
    stm_replay = pd.read_csv(stm_on_portenta_replay).iloc[0]
    portenta_replay = pd.read_csv(portenta_on_stm_replay).iloc[0]

    return [
        TransferPoint(
            architecture="STM-selected",
            measurement_board="STM32 N657",
            point_type="Native optimum",
            energy_mj=float(stm_best["energy_mj_per_inference"]),
            latency_ms=float(stm_best["latency_ms"]),
            accuracy=float(stm_best["metric__accuracy"]),
            macro_f1=float(stm_best["metric__macro_f1"]),
            score=float(stm_best["score"]),
            row_index=int(stm_best.name),
        ),
        TransferPoint(
            architecture="STM-selected",
            measurement_board="Portenta H7",
            point_type="Replay",
            energy_mj=float(stm_replay["target__energy_mj_per_inference"]),
            latency_ms=float(stm_replay["target__latency_ms"]),
            accuracy=float(stm_replay["source__metric__accuracy"]),
            macro_f1=float(stm_replay["source__metric__macro_f1"]),
            score=float(stm_replay["source__score"]),
            row_index=int(stm_replay["source_row_index"]),
        ),
        TransferPoint(
            architecture="Portenta-selected",
            measurement_board="Portenta H7",
            point_type="Native optimum",
            energy_mj=float(portenta_best["energy_mj_per_inference"]),
            latency_ms=float(portenta_best["latency_ms"]),
            accuracy=float(portenta_best["metric__accuracy"]),
            macro_f1=float(portenta_best["metric__macro_f1"]),
            score=float(portenta_best["score"]),
            row_index=int(portenta_best.name),
        ),
        TransferPoint(
            architecture="Portenta-selected",
            measurement_board="STM32 N657",
            point_type="Replay",
            energy_mj=float(portenta_replay["target__energy_mj_per_inference"]),
            latency_ms=float(portenta_replay["target__latency_ms"]),
            accuracy=float(portenta_replay["source__metric__accuracy"]),
            macro_f1=float(portenta_replay["source__metric__macro_f1"]),
            score=float(portenta_replay["source__score"]),
            row_index=int(portenta_replay["source_row_index"]),
        ),
    ]


def point_dataframe(points: Sequence[TransferPoint]) -> pd.DataFrame:
    """Convert transfer points to a dataframe.

    Parameters
    ----------
    points : Sequence[TransferPoint]
        Transfer points.

    Returns
    -------
    pandas.DataFrame
        Tabular transfer summary.
    """

    return pd.DataFrame([point.__dict__ for point in points])


def plot_transfer_panel(
    axis: plt.Axes,
    frame: pd.DataFrame,
    x_metric: str,
    xlabel: str,
    *,
    x_scale: str = "linear",
) -> None:
    """Plot one macro-F1 transfer panel.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Target axis.
    frame : pandas.DataFrame
        Four-point transfer dataframe.
    x_metric : str
        X-axis metric column to plot.
    xlabel : str
        X-axis label.
    x_scale : {"linear", "log"}, optional
        Axis scale for the x metric.
    """

    arch_colors = {"STM-selected": "#1f77b4", "Portenta-selected": "#ff7f0e"}
    board_markers = {"STM32 N657": "o", "Portenta H7": "s"}

    for architecture, arch_frame in frame.groupby("architecture"):
        native = arch_frame[arch_frame["point_type"] == "Native optimum"]
        replay = arch_frame[arch_frame["point_type"] == "Replay"]
        if native.empty or replay.empty:
            continue
        start = native.iloc[0]
        end = replay.iloc[0]
        axis.annotate(
            "",
            xy=(end[x_metric], end["macro_f1"]),
            xytext=(start[x_metric], start["macro_f1"]),
            arrowprops={
                "arrowstyle": "->",
                "color": arch_colors[architecture],
                "linestyle": "-",
                "linewidth": 1.75,
                "alpha": 0.9,
                "shrinkA": 7,
                "shrinkB": 8,
            },
            zorder=2,
        )

    for _, row in frame.iterrows():
        marker = board_markers[row["measurement_board"]]
        color = arch_colors[row["architecture"]]
        facecolor = color if row["point_type"] == "Native optimum" else "white"
        axis.scatter(
            row[x_metric],
            row["macro_f1"],
            s=86,
            marker=marker,
            facecolor=facecolor,
            edgecolor=color,
            linewidth=1.35,
            zorder=3,
        )

    axis.set_xlabel(xlabel)
    axis.set_ylabel("Validation macro-F1")
    axis.set_xscale(x_scale)
    if x_scale == "log":
        axis.xaxis.set_major_formatter(ScalarFormatter())
        axis.xaxis.set_minor_formatter(ScalarFormatter())
    axis.set_ylim(0.752, 0.781)
    axis.grid(True, alpha=0.3)

def plot_transfer(points: Sequence[TransferPoint], output_dir: Path) -> None:
    """Plot cross-board replay quality transfer against energy and latency.

    Parameters
    ----------
    points : Sequence[TransferPoint]
        Four highlighted native/replay points.
    output_dir : pathlib.Path
        Directory for generated outputs.
    """

    frame = point_dataframe(points)
    fig, axes = plt.subplots(2, 1, figsize=(5.7, 5.8), sharex=False)
    plot_transfer_panel(
        axes[0],
        frame,
        "energy_mj",
        "Measured energy per inference (mJ)",
        x_scale="linear",
    )
    plot_transfer_panel(
        axes[1],
        frame,
        "latency_ms",
        "Measured inference latency (ms)",
    )
    axes[0].set_xlim(42, 245)
    axes[1].set_xlim(25, 245)

    legend_handles = [
        Line2D([0], [0], marker="o", color="#1f77b4", markerfacecolor="#1f77b4", linestyle="None", markersize=6, label="STM-selected arch."),
        Line2D([0], [0], marker="o", color="#ff7f0e", markerfacecolor="#ff7f0e", linestyle="None", markersize=6, label="Portenta-selected arch."),
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            markerfacecolor="black",
            linestyle="None",
            markersize=6,
            label="Measured on STM32",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="black",
            markerfacecolor="black",
            linestyle="None",
            markersize=6,
            label="Measured on Portenta",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            markerfacecolor="white",
            linestyle="None",
            markersize=6,
            label="Replay measurement",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            markerfacecolor="black",
            linestyle="None",
            markersize=6,
            label="Native optimum",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncols=3,
        frameon=False,
        columnspacing=1.1,
        handlelength=1.8,
        handletextpad=0.45,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))

    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "case3_cross_board_transfer_points.csv", index=False)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"case3_cross_board_transfer_quality_energy_latency.{suffix}", bbox_inches="tight")
    plt.close(fig)


def board_short(label: str) -> str:
    """Return a compact board label for endpoint annotations."""

    if "STM32" in label:
        return "STM32"
    if "Portenta" in label:
        return "Portenta"
    return label


def endpoint_label(point: pd.Series) -> str:
    """Return direct label for native/replay transfer endpoints."""

    board = board_short(point["measurement_board"])
    if point["point_type"] == "Native optimum":
        return f"selected\nfor {board}"
    return f"replay\non {board}"


def plot_transfer_v2(points: Sequence[TransferPoint], output_dir: Path) -> None:
    """Plot a simplified dumbbell-style cross-board replay view."""

    frame = point_dataframe(points)
    architecture_order = ["STM-selected", "Portenta-selected"]
    arch_colors = {"STM-selected": "#0072B2", "Portenta-selected": "#D55E00"}
    y_positions = {"STM-selected": 0.55, "Portenta-selected": 0.0}
    font_scale = 1.2
    axis_label_size = 11.6 * font_scale
    tick_label_size = 10.5 * font_scale
    endpoint_label_size = 8.6 * font_scale
    y_labels = []
    for architecture in architecture_order:
        arch_frame = frame[frame["architecture"] == architecture]
        macro_f1 = float(arch_frame["macro_f1"].iloc[0])
        y_labels.append(f"{architecture.replace('-selected', '')}\nselected\nF1={macro_f1:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(5.85, 2.61), sharey=True)
    panel_specs = [
        (axes[0], "energy_mj", "Energy (mJ)", (5, 270)),
        (axes[1], "latency_ms", "Latency (ms)", (5, 285)),
    ]

    for axis_index, (axis, metric, xlabel, xlim) in enumerate(panel_specs):
        axis.set_xlim(*xlim)
        axis.set_xticks([50, 100, 150, 200, 250])
        axis.tick_params(axis="both", labelsize=tick_label_size)
        for architecture in architecture_order:
            arch_frame = frame[frame["architecture"] == architecture]
            native = arch_frame[arch_frame["point_type"] == "Native optimum"].iloc[0]
            replay = arch_frame[arch_frame["point_type"] == "Replay"].iloc[0]
            y = y_positions[architecture]
            color = arch_colors[architecture]
            axis.plot(
                [native[metric], replay[metric]],
                [y, y],
                color=color,
                linewidth=2.0,
                alpha=0.82,
                zorder=2,
            )
            axis.scatter(
                native[metric],
                y,
                s=62,
                color=color,
                edgecolor="0.15",
                linewidth=1.1,
                zorder=3,
            )
            axis.scatter(
                replay[metric],
                y,
                s=62,
                facecolor="white",
                edgecolor=color,
                linewidth=1.8,
                zorder=3,
            )
            for point in (native, replay):
                label_dx = 0.0
                label_dy = 0.10
                ha = "center"
                if point["point_type"] == "Native optimum" and board_short(point["measurement_board"]) == "STM32":
                    label_dx = 6.0
                if metric == "energy_mj" and architecture == "Portenta-selected":
                    if point["point_type"] == "Native optimum":
                        label_dx = -24.0
                        label_dy = 0.18
                        ha = "center"
                    else:
                        label_dx = xlim[1] - 8.0 - float(point[metric])
                        label_dy = 0.035
                        ha = "right"
                if metric == "latency_ms" and architecture == "Portenta-selected":
                    if point["point_type"] == "Native optimum":
                        label_dx = -16.0
                        label_dy = 0.18
                        ha = "center"
                    else:
                        label_dx = xlim[1] - 8.0 - float(point[metric])
                        label_dy = 0.035
                        ha = "right"
                text_x = point[metric] + label_dx
                axis.text(
                    text_x,
                    y + label_dy,
                    endpoint_label(point),
                    ha=ha,
                    va="bottom",
                    fontsize=endpoint_label_size,
                    linespacing=1.18,
                )

        axis.set_xlabel(xlabel, fontsize=axis_label_size)
        axis.set_yticks([y_positions[architecture] for architecture in architecture_order])
        if axis_index == 0:
            axis.set_yticklabels(y_labels)
            for label in axis.get_yticklabels():
                label.set_linespacing(1.08)
        else:
            axis.tick_params(axis="y", labelleft=False)
        axis.set_ylim(-0.13, 0.82)
        axis.grid(True, axis="x", alpha=0.28)
        axis.grid(False, axis="y")

    fig.tight_layout(w_pad=0.55)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / "case3_cross_board_transfer_quality_energy_latency_v2.png",
        bbox_inches="tight",
        dpi=400,
    )
    fig.savefig(output_dir / "case3_cross_board_transfer_quality_energy_latency_v2.pdf", bbox_inches="tight")
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """

    parser = argparse.ArgumentParser(description="Plot Case 3 audio cross-board NAS transfer figures.")
    parser.add_argument("--stm-log", type=Path, default=DEFAULT_STM_LOG)
    parser.add_argument("--portenta-log", type=Path, default=DEFAULT_PORTENTA_LOG)
    parser.add_argument("--stm-on-portenta-replay", type=Path, default=DEFAULT_STM_ON_PORTENTA_REPLAY)
    parser.add_argument("--portenta-on-stm-replay", type=Path, default=DEFAULT_PORTENTA_ON_STM_REPLAY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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
        NativeRun("Portenta H7 native search", args.portenta_log, "#D55E00", "s"),
        NativeRun("STM32 N657 native search", args.stm_log, "#0072B2", "o"),
    ]
    output_dir = args.output_dir.expanduser().resolve()
    plot_score_progress(native_runs, output_dir)
    points = transfer_points(
        stm_log=args.stm_log,
        portenta_log=args.portenta_log,
        stm_on_portenta_replay=args.stm_on_portenta_replay,
        portenta_on_stm_replay=args.portenta_on_stm_replay,
    )
    plot_transfer(points, output_dir)
    plot_transfer_v2(points, output_dir)
    print(f"Wrote Case 3 transfer figures to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
