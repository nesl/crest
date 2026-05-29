#!/usr/bin/env python3
"""Render the Case Study 1 combined static-proxy replay front figure."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator
from matplotlib.ticker import LogFormatterMathtext
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from plot_combined_replay_fronts import LoadedPair  # noqa: E402
from plot_combined_replay_fronts import PairInput  # noqa: E402
from plot_combined_replay_fronts import load_pair  # noqa: E402


CREST_COLOR = "#1f77b4"
REPLAY_COLOR = "#ff7f0e"
MEMORY_REPLAY_COLOR = "#2ca02c"
OUTPUT_BASENAME = "combined_measured_energy_rmse_front_subplots_v2"
DEFAULT_OUTPUT_DIR = Path("models/replays/visualizations/combined_flops_proxy_replay_ordered_targets_logx")
DEFAULT_FONT_SCALE = 1.397
DEFAULT_LEGEND_FONT_SCALE = 1.65
DEFAULT_WSPACE_SCALE = 0.5


def repo_path(*relative_paths: str) -> Path:
    """Return the first existing repository-relative path.

    Parameters
    ----------
    *relative_paths : str
        Candidate paths relative to the repository root.

    Returns
    -------
    pathlib.Path
        Existing resolved path, or the first candidate if none exist.
    """

    for relative_path in relative_paths:
        path = (REPO_ROOT / relative_path).resolve()
        if path.exists():
            return path
    return (REPO_ROOT / relative_paths[0]).resolve()


def default_pairs() -> list[PairInput]:
    """Return the ordered Case Study 1 target comparisons.

    Returns
    -------
    list[PairInput]
        Four target comparisons in paper order.
    """

    return [
        PairInput(
            label="BLE33",
            crest_run_dir=repo_path(
                "models/OxIOD_BLE33_B2B_case1_2_t1",
                "models/CREST_case1/models/OxIOD_BLE33_B2B_case1_2_t1",
            ),
            replay_path=repo_path(
                "models/replays/data/flops_case1_1_t3_on_ble33",
                "models/CREST_case1/replays/data/flops_case1_1_t3_on_ble33",
            ),
        ),
        PairInput(
            label="Portenta M4",
            crest_run_dir=repo_path(
                "models/OxIOD_PORTENTA_M4_B2B_case1_4_t5",
                "models/CREST_case1/models/OxIOD_PORTENTA_M4_B2B_case1_4_t5",
            ),
            replay_path=repo_path(
                "models/replays/data/flops_case1_1_t3_on_portenta_m4",
                "models/CREST_case1/replays/data/flops_case1_1_t3_on_portenta_m4",
            ),
        ),
        PairInput(
            label="Portenta M7",
            crest_run_dir=repo_path(
                "models/OxIOD_PORTENTA_M7_B2B_case1_3_t1",
                "models/CREST_case1/models/OxIOD_PORTENTA_M7_B2B_case1_3_t1",
            ),
            replay_path=repo_path(
                "models/replays/data/flops_case1_1_t3_on_portenta_m7",
                "models/CREST_case1/replays/data/flops_case1_1_t3_on_portenta_m7",
            ),
        ),
        PairInput(
            label="STM32 N657",
            crest_run_dir=repo_path(
                "models/OxIOD_STM32_B2B_case1_5_t1",
                "models/CREST_case1/models/OxIOD_STM32_B2B_case1_5_t1",
            ),
            replay_path=repo_path(
                "models/replays/data/flops_case1_1_t3_on_stm32_cadenced",
                "models/CREST_case1/replays/data/flops_case1_1_t3_on_stm32_cadenced",
            ),
        ),
    ]


def default_memory_pairs() -> list[PairInput]:
    """Return the ordered Case Study 1 memory-traffic-proxy replay comparisons.

    Returns
    -------
    list[PairInput]
        Four target comparisons in the same paper order as ``default_pairs``.
    """

    default = default_pairs()
    memory_replay_paths = {
        "BLE33": repo_path(
            "models/CREST_case1/replays/data/memory_case1_1_on_ble33_v2",
            "models/replays/data/memory_case1_1_on_ble33_v2",
        ),
        "Portenta M4": repo_path(
            "models/CREST_case1/replays/data/memory_case1_1_on_portenta_m4_v2",
            "models/replays/data/memory_case1_1_on_portenta_m4_v2",
        ),
        "Portenta M7": repo_path(
            "models/CREST_case1/replays/data/memory_case1_1_on_portenta_m7_v2",
            "models/replays/data/memory_case1_1_on_portenta_m7_v2",
        ),
        "STM32 N657": repo_path(
            "models/CREST_case1/replays/data/memory_case1_1_on_stm32_b2b_v2",
            "models/replays/data/memory_case1_1_on_stm32_b2b_v2",
        ),
    }
    return [
        PairInput(
            label=pair.label,
            crest_run_dir=pair.crest_run_dir,
            replay_path=memory_replay_paths[pair.label],
        )
        for pair in default
    ]


def point_key_set(points: Sequence[Any]) -> set[str]:
    """Return payload keys for a point collection.

    Parameters
    ----------
    points : Sequence[Any]
        FrontPoint-like objects.

    Returns
    -------
    set[str]
        Payload keys.
    """

    return {point.payload_key for point in points}


def scatter_latency_points(
    ax: Any,
    points: Sequence[Any],
    *,
    color: str,
    marker: str,
    size: float,
    alpha: float,
    linewidth: float,
    zorder: int,
) -> None:
    """Scatter points with fill encoding latency feasibility.

    Parameters
    ----------
    ax : Any
        Matplotlib axes.
    points : Sequence[Any]
        FrontPoint-like objects.
    color : str
        Series color.
    marker : str
        Marker symbol.
    size : float
        Marker size.
    alpha : float
        Marker alpha.
    linewidth : float
        Marker edge width.
    zorder : int
        Matplotlib z-order.
    """

    feasible = [point for point in points if point.latency_feasible is not False]
    infeasible = [point for point in points if point.latency_feasible is False]
    if feasible:
        ax.scatter(
            [point.energy_mj for point in feasible],
            [point.rmse for point in feasible],
            s=size,
            marker=marker,
            facecolors=color,
            edgecolors=color,
            linewidths=linewidth,
            alpha=alpha,
            zorder=zorder,
        )
    if infeasible:
        ax.scatter(
            [point.energy_mj for point in infeasible],
            [point.rmse for point in infeasible],
            s=size,
            marker=marker,
            facecolors="white",
            edgecolors=color,
            linewidths=max(linewidth, 0.7),
            alpha=max(alpha, 0.9),
            zorder=zorder,
        )


def draw_front(ax: Any, points: Sequence[Any], *, color: str, marker: str, linestyle: str) -> None:
    """Draw a front line and front markers.

    Parameters
    ----------
    ax : Any
        Matplotlib axes.
    points : Sequence[Any]
        FrontPoint-like front points sorted by energy.
    color : str
        Series color.
    marker : str
        Marker symbol.
    linestyle : str
        Line style.
    """

    if not points:
        return
    ax.plot(
        [point.energy_mj for point in points],
        [point.rmse for point in points],
        color=color,
        linestyle=linestyle,
        linewidth=1.15,
        zorder=4,
    )
    scatter_latency_points(
        ax,
        points,
        color=color,
        marker=marker,
        size=24,
        alpha=0.98,
        linewidth=0.8,
        zorder=5,
    )


def write_plotted_points(
    path: Path,
    pairs: Sequence[LoadedPair],
    memory_pairs: Sequence[LoadedPair] | None = None,
) -> None:
    """Write plotted point data for the v2 figure.

    Parameters
    ----------
    path : pathlib.Path
        Output CSV path.
    pairs : Sequence[LoadedPair]
        Loaded target pairs.
    memory_pairs : Sequence[LoadedPair] or None, optional
        Loaded memory-traffic-proxy replay pairs.
    """

    memory_by_label = {pair.label: pair for pair in memory_pairs or []}
    fieldnames = [
        "target",
        "series",
        "row_index",
        "payload_key",
        "rmse",
        "energy_mj",
        "latency_ms",
        "latency_budget_ms",
        "latency_feasible",
        "is_front",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for pair in pairs:
            series_specs = [
                ("measured_energy_nas", pair.crest_points, pair.crest_front),
                ("flops_replay", pair.replay_points, pair.replay_front),
            ]
            memory_pair = memory_by_label.get(pair.label)
            if memory_pair is not None:
                series_specs.append(
                    ("memory_replay", memory_pair.replay_points, memory_pair.replay_front)
                )
            for series, points, front in series_specs:
                front_keys = point_key_set(front)
                for point in points:
                    writer.writerow(
                        {
                            "target": pair.label,
                            "series": series,
                            "row_index": point.row_index,
                            "payload_key": point.payload_key,
                            "rmse": point.rmse,
                            "energy_mj": point.energy_mj,
                            "latency_ms": "" if point.latency_ms is None else point.latency_ms,
                            "latency_budget_ms": ""
                            if point.latency_budget_ms is None
                            else point.latency_budget_ms,
                            "latency_feasible": point.latency_feasible,
                            "is_front": point.payload_key in front_keys,
                        }
                    )


def pair_summary_rows(pairs: Sequence[LoadedPair]) -> list[dict[str, Any]]:
    """Build per-target summary rows.

    Parameters
    ----------
    pairs : Sequence[LoadedPair]
        Loaded target pairs.

    Returns
    -------
    list[dict[str, Any]]
        Summary rows.
    """

    rows: list[dict[str, Any]] = []
    for pair in pairs:
        dominated = sum(float(row["energy_delta_mj"]) > 0.0 for row in pair.regret_rows)
        rows.append(
            {
                "target": pair.label,
                "measured_valid_points": len(pair.crest_points),
                "replay_valid_points": len(pair.replay_points),
                "measured_front_points": len(pair.crest_front),
                "replay_front_points": len(pair.replay_front),
                "replay_total_candidates": pair.replay_outcomes.total_candidates,
                "replay_valid_latency_feasible": sum(point.latency_feasible is True for point in pair.replay_points),
                "replay_front_latency_feasible": sum(point.latency_feasible is True for point in pair.replay_front),
                "replay_front_dominated_by_measured": dominated,
            }
        )
    return rows


def write_summary(
    path: Path,
    pairs: Sequence[LoadedPair],
    memory_pairs: Sequence[LoadedPair] | None = None,
) -> None:
    """Write a text summary for the v2 figure.

    Parameters
    ----------
    path : pathlib.Path
        Output summary path.
    pairs : Sequence[LoadedPair]
        Loaded target pairs.
    memory_pairs : Sequence[LoadedPair] or None, optional
        Loaded memory-traffic-proxy replay pairs.
    """

    rows = pair_summary_rows(pairs)
    memory_by_label = {pair.label: pair for pair in memory_pairs or []}
    total_replay_valid = sum(row["replay_valid_points"] for row in rows)
    total_replay_scheduled = sum(row["replay_total_candidates"] for row in rows)
    total_replay_feasible = sum(row["replay_valid_latency_feasible"] for row in rows)
    total_replay_front = sum(row["replay_front_points"] for row in rows)
    total_replay_front_feasible = sum(row["replay_front_latency_feasible"] for row in rows)
    total_dominated = sum(row["replay_front_dominated_by_measured"] for row in rows)
    lines = [
        "Case Study 1 combined measured-energy vs static-proxy replay front summary",
        "",
        "Inputs:",
        "  BLE33: models/OxIOD_BLE33_B2B_case1_2_t1 + models/replays/data/flops_case1_1_t3_on_ble33",
        "  Portenta M4: models/OxIOD_PORTENTA_M4_B2B_case1_4_t5 + models/replays/data/flops_case1_1_t3_on_portenta_m4",
        "  Portenta M7: models/OxIOD_PORTENTA_M7_B2B_case1_3_t1 + models/replays/data/flops_case1_1_t3_on_portenta_m7",
        "  STM32 N657: models/OxIOD_STM32_B2B_case1_5_t1 + models/replays/data/flops_case1_1_t3_on_stm32_cadenced",
        "  Memory traffic proxy replays: models/CREST_case1/replays/data/memory_case1_1_on_*_v2",
        "",
        f"FLOPs source-front candidates per target: {rows[0]['replay_total_candidates'] if rows else 0}",
        f"Scheduled replay target measurements: {total_replay_scheduled}",
        f"Total valid replay measurements: {total_replay_valid}",
        f"Total latency-feasible valid replay measurements: {total_replay_feasible}/{total_replay_valid}",
        f"Total valid replay-front points: {total_replay_front}",
        f"Total latency-feasible replay-front points: {total_replay_front_feasible}/{total_replay_front}",
        f"Replay-front points dominated by measured-energy NAS: {total_dominated}/{total_replay_front}",
        "",
    ]
    for row in rows:
        lines.append(row["target"])
        lines.append(f"  measured valid points: {row['measured_valid_points']}")
        lines.append(f"  replay valid points: {row['replay_valid_points']}")
        lines.append(f"  replay valid / scheduled: {row['replay_valid_points']}/{row['replay_total_candidates']}")
        lines.append(f"  measured front points: {row['measured_front_points']}")
        lines.append(f"  replay front points: {row['replay_front_points']}")
        lines.append(
            "  replay front latency-feasible: "
            f"{row['replay_front_latency_feasible']}/{row['replay_front_points']}"
        )
        lines.append(
            "  replay front dominated by measured-energy NAS: "
            f"{row['replay_front_dominated_by_measured']}/{row['replay_front_points']}"
        )
        memory_pair = memory_by_label.get(row["target"])
        if memory_pair is not None:
            lines.append(f"  memory traffic proxy replay valid points: {len(memory_pair.replay_points)}")
            lines.append(f"  memory traffic proxy replay front points: {len(memory_pair.replay_front)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_csv(path: Path, pairs: Sequence[LoadedPair]) -> None:
    """Write per-target summary counts as CSV.

    Parameters
    ----------
    path : pathlib.Path
        Output CSV path.
    pairs : Sequence[LoadedPair]
        Loaded target pairs.
    """

    rows = pair_summary_rows(pairs)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def subplot_shape(layout: str, pair_count: int) -> tuple[int, int, tuple[float, float], float, float]:
    """Return subplot geometry for a named layout.

    Parameters
    ----------
    layout : str
        Layout name: ``"grid"``, ``"row"``, or ``"column"``.
    pair_count : int
        Number of panels.

    Returns
    -------
    tuple[int, int, tuple[float, float], float, float]
        Rows, columns, figure size, horizontal spacing, and vertical spacing.
    """

    if layout == "row":
        return 1, pair_count, (10.6, 2.55), 0.12, 0.0
    if layout == "column":
        return pair_count, 1, (4.55, 8.45), 0.0, 0.22
    return 2, 2, (7.16, 4.65), 0.16, 0.28


def plot_case1_fronts_v2(
    pairs: Sequence[LoadedPair],
    output_dir: Path,
    basename: str,
    *,
    memory_pairs: Sequence[LoadedPair] | None = None,
    layout: str = "grid",
    font_scale: float = DEFAULT_FONT_SCALE,
    legend_font_scale: float | None = None,
    wspace_scale: float = DEFAULT_WSPACE_SCALE,
) -> tuple[Path, Path, Path, Path]:
    """Render the restyled Case Study 1 four-panel figure.

    Parameters
    ----------
    pairs : Sequence[LoadedPair]
        Loaded target pairs.
    output_dir : pathlib.Path
        Output directory.
    basename : str
        Output file basename.
    memory_pairs : Sequence[LoadedPair] or None, optional
        Loaded memory-traffic-proxy replay pairs to overlay as front lines only.
    layout : str, default="grid"
        Panel layout: ``"grid"``, ``"row"``, or ``"column"``.
    font_scale : float, default=DEFAULT_FONT_SCALE
        Multiplicative scale applied to all text sizes.
    legend_font_scale : float or None, default=None
        Multiplicative scale applied to legend text. Defaults to ``DEFAULT_LEGEND_FONT_SCALE``.
    wspace_scale : float, default=DEFAULT_WSPACE_SCALE
        Multiplier applied to the layout's horizontal subplot spacing.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path]
        PNG, PDF, plotted-points CSV, and summary paths.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    memory_by_label = {pair.label: pair for pair in memory_pairs or []}
    all_points = [point for pair in pairs for point in [*pair.crest_points, *pair.replay_points]]
    all_points.extend(point for pair in memory_by_label.values() for point in pair.replay_front)
    y_values = np.asarray([point.rmse for point in all_points], dtype=float)
    y_lower = max(0.0, float(np.nanmin(y_values)) - 0.08)
    y_upper = float(np.nanmax(y_values)) + 0.12

    if legend_font_scale is None:
        legend_font_scale = DEFAULT_LEGEND_FONT_SCALE

    with plt.rc_context(
        {
            "font.size": 7 * font_scale,
            "axes.titlesize": 8.8 * font_scale,
            "axes.labelsize": 7 * font_scale,
            "xtick.labelsize": 6 * font_scale,
            "ytick.labelsize": 6 * font_scale,
            "legend.fontsize": 6 * legend_font_scale,
            "lines.linewidth": 1.0,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        rows, cols, figsize, wspace, hspace = subplot_shape(layout, len(pairs))
        wspace *= wspace_scale
        fig, axes = plt.subplots(rows, cols, figsize=figsize, sharey=True, squeeze=False)
        axes_flat = list(axes.ravel())
        for ax, pair in zip(axes_flat, pairs):
            crest_front_keys = point_key_set(pair.crest_front)
            replay_front_keys = point_key_set(pair.replay_front)
            crest_cloud = [point for point in pair.crest_points if point.payload_key not in crest_front_keys]
            replay_cloud = [point for point in pair.replay_points if point.payload_key not in replay_front_keys]

            scatter_latency_points(
                ax,
                crest_cloud,
                color=CREST_COLOR,
                marker="o",
                size=9,
                alpha=0.24,
                linewidth=0.25,
                zorder=2,
            )
            scatter_latency_points(
                ax,
                replay_cloud,
                color=REPLAY_COLOR,
                marker="s",
                size=11,
                alpha=0.34,
                linewidth=0.35,
                zorder=2,
            )
            draw_front(ax, pair.crest_front, color=CREST_COLOR, marker="o", linestyle="-")
            draw_front(ax, pair.replay_front, color=REPLAY_COLOR, marker="s", linestyle="--")
            memory_pair = memory_by_label.get(pair.label)
            if memory_pair is not None:
                draw_front(
                    ax,
                    memory_pair.replay_front,
                    color=MEMORY_REPLAY_COLOR,
                    marker="D",
                    linestyle="-.",
                )

            ax.set_title(pair.label, pad=3)
            ax.set_xscale("log")
            ax.set_ylim(y_lower, y_upper)
            ax.grid(True, alpha=0.3, which="both")
            ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
            ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
            ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))
            ax.text(
                0.965,
                0.955,
                f"CREST front: {len(pair.crest_front)}\nReplay front: {len(pair.replay_front)}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=5.8 * font_scale,
                color="0.2",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.62, "pad": 1.6},
            )

        for ax in axes[:, 0]:
            ax.set_ylabel("Aggregate RMSE")
        for ax in axes[-1, :]:
            ax.set_xlabel("Measured energy per inference (mJ)")

        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=CREST_COLOR,
                markeredgecolor=CREST_COLOR,
                markersize=4,
                linestyle="none",
                label="Measured NAS",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markerfacecolor=REPLAY_COLOR,
                markeredgecolor=REPLAY_COLOR,
                markersize=4,
                linestyle="none",
                label="FLOPs proxy replay",
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                color="none",
                markerfacecolor=MEMORY_REPLAY_COLOR,
                markeredgecolor=MEMORY_REPLAY_COLOR,
                markersize=4,
                linestyle="none",
                label="Memory traffic proxy replay",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="0.5",
                markeredgecolor="black",
                markersize=4,
                linestyle="none",
                label="<= 200 ms",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=4,
                linestyle="none",
                label="> 200 ms",
            ),
            Line2D([0], [0], color="black", linewidth=1.0, linestyle="-", label="Native front"),
            Line2D([0], [0], color="black", linewidth=1.0, linestyle="--", label="Replay front"),
        ]
        legend_columns = 3 if layout == "column" else 7
        legend_y = 0.975 if layout == "column" else 0.978
        fig.legend(
            handles=handles,
            loc="upper center",
            ncol=legend_columns,
            bbox_to_anchor=(0.5, legend_y),
            frameon=False,
            columnspacing=0.75,
            handletextpad=0.35,
            borderaxespad=0.0,
        )
        if layout == "row":
            fig.subplots_adjust(left=0.055, right=0.995, top=0.82, bottom=0.17, wspace=wspace, hspace=hspace)
        elif layout == "column":
            fig.subplots_adjust(left=0.14, right=0.985, top=0.92, bottom=0.055, wspace=wspace, hspace=hspace)
        else:
            fig.subplots_adjust(left=0.075, right=0.992, top=0.88, bottom=0.12, wspace=wspace, hspace=hspace)

        png_path = output_dir / f"{basename}.png"
        pdf_path = output_dir / f"{basename}.pdf"
        csv_path = output_dir / f"{basename}_plotted_points.csv"
        summary_path = output_dir / f"{basename}_summary.txt"
        fig.savefig(png_path, dpi=300, facecolor="white")
        fig.savefig(pdf_path, facecolor="white", bbox_inches="tight", pad_inches=0.006)
        plt.close(fig)

    write_plotted_points(csv_path, pairs, memory_pairs)
    write_summary(summary_path, pairs, memory_pairs)
    write_summary_csv(output_dir / f"{basename}_summary.csv", pairs)
    return png_path, pdf_path, csv_path, summary_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--basename", default=OUTPUT_BASENAME, help="Output file basename.")
    parser.add_argument(
        "--layout",
        choices=("grid", "row", "column"),
        default="grid",
        help="Panel layout.",
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=DEFAULT_FONT_SCALE,
        help="Multiplicative text-size scale.",
    )
    parser.add_argument(
        "--legend-font-scale",
        type=float,
        default=None,
        help=f"Multiplicative legend text-size scale. Defaults to {DEFAULT_LEGEND_FONT_SCALE}.",
    )
    parser.add_argument(
        "--wspace-scale",
        type=float,
        default=DEFAULT_WSPACE_SCALE,
        help="Multiplier for horizontal subplot spacing. Values below 1.0 move panels closer.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the v2 renderer.

    Parameters
    ----------
    argv : Sequence[str] or None, optional
        CLI arguments excluding executable name.

    Returns
    -------
    int
        Exit code.
    """

    args = build_arg_parser().parse_args(argv)
    pairs = [load_pair(pair, feasible_only=False) for pair in default_pairs()]
    memory_pairs = [load_pair(pair, feasible_only=False) for pair in default_memory_pairs()]
    paths = plot_case1_fronts_v2(
        pairs,
        Path(args.output_dir),
        args.basename,
        memory_pairs=memory_pairs,
        layout=args.layout,
        font_scale=args.font_scale,
        legend_font_scale=args.legend_font_scale,
        wspace_scale=args.wspace_scale,
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
