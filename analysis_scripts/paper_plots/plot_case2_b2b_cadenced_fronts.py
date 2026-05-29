#!/usr/bin/env python3
"""Render the Case Study 2 B2B/cadenced cross-runtime front figure."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


B2B_COLOR = "#1f77b4"
CADENCED_COLOR = "#ff7f0e"
SUCCESS_ERROR_CODE = 1
DEFAULT_LATENCY_BUDGET_MS = 200.0
DEFAULT_WINDOW_BUDGET_MS = 2000.0
DEFAULT_OUTPUT_DIR = Path(
    "models/crest_case_2_v2/comparisons/case2_b2b_cadenced_cross_runtime_fronts"
)
DEFAULT_PLOT_NAME = "case2_b2b_cadenced_cross_runtime_fronts"
CASE3_MATCH_WIDTH_IN = 1942.0 / 300.0
CASE3_MATCH_HEIGHT_IN = 880.5 / 300.0
Y_AXIS_LIMITS = (0.3, 1.7)
Y_AXIS_TICKS = np.arange(0.4, 1.8, 0.2)


def numeric_series(frame: pd.DataFrame, column: str, default: float | None = None) -> pd.Series:
    """Return a numeric dataframe column, or a default-valued series if absent."""

    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series([default] * len(frame), index=frame.index, dtype="float64")


def pareto_mask(x_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
    """Return non-dominated membership for two minimized objectives."""

    values = np.column_stack([x_values, y_values]).astype(float)
    mask = np.ones(len(values), dtype=bool)
    for index, candidate in enumerate(values):
        dominated = np.all(values <= candidate, axis=1) & np.any(values < candidate, axis=1)
        if np.any(dominated):
            mask[index] = False
    return mask


def add_front_flags(points: pd.DataFrame) -> pd.DataFrame:
    """Annotate each policy group with panel-front membership."""

    result = points.copy()
    result["is_front"] = False
    for policy, group in result.groupby("policy", sort=False):
        clean = group[np.isfinite(group["x_value"]) & np.isfinite(group["rmse"])]
        if clean.empty:
            continue
        result.loc[clean.index, "is_front"] = pareto_mask(
            clean["x_value"].to_numpy(dtype=float),
            clean["rmse"].to_numpy(dtype=float),
        )
    return result


def load_study_points(study_points_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load native B2B and cadenced study points from the v2 comparison CSV."""

    raw = pd.read_csv(study_points_csv)
    b2b = raw[raw["study_label"].astype(str).str.contains("B2B", case=False, na=False)].copy()
    cadenced = raw[raw["study_label"].astype(str).str.contains("cadenced", case=False, na=False)].copy()
    b2b_points = pd.DataFrame(
        {
            "panel": "b2b_energy",
            "policy": "B2B NAS",
            "source": "native",
            "row_index": numeric_series(b2b, "source_row"),
            "rmse": numeric_series(b2b, "metric__rmse_total"),
            "x_value": numeric_series(b2b, "energy_objective"),
            "energy_mj_per_inference": numeric_series(b2b, "energy_objective"),
            "cadenced_energy_mj_per_window": np.nan,
            "latency_ms": numeric_series(b2b, "latency_ms"),
            "deadline_miss_count": 0.0,
            "strict_cadenced_feasible": np.nan,
        }
    )
    cadenced_points = pd.DataFrame(
        {
            "panel": "cadenced_window_energy",
            "policy": "Cadenced NAS",
            "source": "native",
            "row_index": numeric_series(cadenced, "source_row"),
            "rmse": numeric_series(cadenced, "metric__rmse_total"),
            "x_value": numeric_series(cadenced, "energy_objective"),
            "energy_mj_per_inference": np.nan,
            "cadenced_energy_mj_per_window": numeric_series(cadenced, "energy_objective"),
            "latency_ms": numeric_series(cadenced, "latency_ms"),
            "deadline_miss_count": 0.0,
            "strict_cadenced_feasible": True,
        }
    )
    return clean_points(b2b_points), clean_points(cadenced_points)


def load_cadenced_replay_on_b2b(overlay_points_csv: Path) -> pd.DataFrame:
    """Load cadenced-front candidates replayed with B2B metrics."""

    raw = pd.read_csv(overlay_points_csv)
    replay = raw[raw["series"].astype(str).str.contains("cadenced Pareto replayed B2B", na=False)].copy()
    points = pd.DataFrame(
        {
            "panel": "b2b_energy",
            "policy": "Cadenced Pareto replay",
            "source": "replay",
            "row_index": numeric_series(replay, "source_row"),
            "rmse": numeric_series(replay, "rmse"),
            "x_value": numeric_series(replay, "b2b_energy_mj"),
            "energy_mj_per_inference": numeric_series(replay, "b2b_energy_mj"),
            "cadenced_energy_mj_per_window": numeric_series(
                replay, "source__cadenced_energy_mj_per_window"
            ),
            "latency_ms": numeric_series(replay, "b2b_latency_ms"),
            "deadline_miss_count": numeric_series(
                replay, "source__cadenced_deadline_miss_count", 0.0
            ).fillna(0.0),
            "strict_cadenced_feasible": numeric_series(
                replay, "source__cadenced_deadline_miss_count", 0.0
            ).fillna(0.0).le(0.0),
        }
    )
    return clean_points(points)


def load_b2b_replay_on_cadenced(replay_csv: Path) -> pd.DataFrame:
    """Load B2B-front candidates replayed with cadenced runtime metrics."""

    raw = pd.read_csv(replay_csv)
    completed = raw["replay_status"].astype(str).eq("completed")
    target_error = numeric_series(raw, "target__error_code")
    frame = raw[completed & target_error.eq(SUCCESS_ERROR_CODE)].copy()

    latency = numeric_series(frame, "target__latency_ms")
    latency_budget = numeric_series(frame, "target__latency_budget_ms", DEFAULT_LATENCY_BUDGET_MS)
    deadline_misses = numeric_series(frame, "target__cadenced_deadline_miss_count", 0.0).fillna(0.0)
    window_latency = numeric_series(frame, "target__cadenced_window_latency_ms")
    window_energy = numeric_series(frame, "target__cadenced_energy_mj_per_window")
    strict_feasible = (
        latency.le(latency_budget.fillna(DEFAULT_LATENCY_BUDGET_MS))
        & deadline_misses.le(0.0)
        & window_latency.le(DEFAULT_WINDOW_BUDGET_MS)
        & window_energy.gt(0.0)
    )
    points = pd.DataFrame(
        {
            "panel": "cadenced_window_energy",
            "policy": "B2B Pareto replay",
            "source": "replay",
            "row_index": numeric_series(frame, "source_row_index"),
            "rmse": numeric_series(frame, "source__metric__rmse_total"),
            "x_value": window_energy,
            "energy_mj_per_inference": numeric_series(frame, "target__energy_mj_per_inference"),
            "cadenced_energy_mj_per_window": window_energy,
            "latency_ms": latency,
            "deadline_miss_count": deadline_misses,
            "cadenced_window_latency_ms": window_latency,
            "strict_cadenced_feasible": strict_feasible,
        }
    )
    return clean_points(points[strict_feasible])


def clean_points(points: pd.DataFrame) -> pd.DataFrame:
    """Keep only finite, positive plot coordinates."""

    result = points.copy()
    result["x_value"] = pd.to_numeric(result["x_value"], errors="coerce")
    result["rmse"] = pd.to_numeric(result["rmse"], errors="coerce")
    return result[np.isfinite(result["x_value"]) & np.isfinite(result["rmse"]) & result["x_value"].gt(0.0)].copy()


def draw_policy(
    ax: Any,
    points: pd.DataFrame,
    policy: str,
    *,
    color: str,
    linestyle: str,
    marker_size_scale: float,
) -> None:
    """Draw one policy's point cloud and Pareto front."""

    group = points[points["policy"] == policy].copy()
    if group.empty:
        return
    non_front = group[~group["is_front"]]
    front = group[group["is_front"]].sort_values("x_value")
    if not non_front.empty:
        ax.scatter(
            non_front["x_value"],
            non_front["rmse"],
            s=11 * marker_size_scale,
            marker="o",
            facecolors=color,
            edgecolors="none",
            alpha=0.26,
            zorder=2,
        )
    if not front.empty:
        ax.plot(
            front["x_value"],
            front["rmse"],
            color=color,
            linestyle=linestyle,
            linewidth=1.35,
            zorder=4,
        )
        ax.scatter(
            front["x_value"],
            front["rmse"],
            s=27 * marker_size_scale,
            marker="o",
            facecolors=color,
            edgecolors="black",
            linewidths=0.45,
            alpha=0.97,
            zorder=5,
        )


def summary_text(
    *,
    plotted: pd.DataFrame,
    cadenced_csv: Path,
    b2b_replay_csv: Path,
) -> str:
    """Build the summary text for the generated plot."""

    cadenced_raw = pd.read_csv(cadenced_csv)
    err = numeric_series(cadenced_raw, "error_code")
    pruned = cadenced_raw.get("pruned", pd.Series(False, index=cadenced_raw.index)).astype(str).str.lower().eq("true")
    cadenced_success = cadenced_raw[err.eq(SUCCESS_ERROR_CODE) & ~pruned].copy()
    for column in [
        "latency_ms",
        "latency_budget_ms",
        "cadenced_window_latency_ms",
        "cadenced_energy_mj_per_window",
        "cadenced_deadline_miss_count",
    ]:
        cadenced_success[column] = numeric_series(cadenced_success, column)
    label_feasible = cadenced_success["feasible"].astype(str).str.lower().eq("true")
    strict_feasible = (
        cadenced_success["latency_ms"].le(cadenced_success["latency_budget_ms"].fillna(DEFAULT_LATENCY_BUDGET_MS))
        & cadenced_success["cadenced_deadline_miss_count"].fillna(0.0).le(0.0)
        & cadenced_success["cadenced_window_latency_ms"].le(DEFAULT_WINDOW_BUDGET_MS)
        & cadenced_success["cadenced_energy_mj_per_window"].gt(0.0)
    )

    replay_raw = pd.read_csv(b2b_replay_csv)
    completed = replay_raw["replay_status"].astype(str).eq("completed")
    replay_success = replay_raw[completed & numeric_series(replay_raw, "target__error_code").eq(SUCCESS_ERROR_CODE)].copy()
    for column in [
        "target__latency_ms",
        "target__latency_budget_ms",
        "target__cadenced_window_latency_ms",
        "target__cadenced_energy_mj_per_window",
        "target__cadenced_deadline_miss_count",
    ]:
        replay_success[column] = numeric_series(replay_success, column)
    replay_strict = (
        replay_success["target__latency_ms"].le(
            replay_success["target__latency_budget_ms"].fillna(DEFAULT_LATENCY_BUDGET_MS)
        )
        & replay_success["target__cadenced_deadline_miss_count"].fillna(0.0).le(0.0)
        & replay_success["target__cadenced_window_latency_ms"].le(DEFAULT_WINDOW_BUDGET_MS)
        & replay_success["target__cadenced_energy_mj_per_window"].gt(0.0)
    )

    lines = [
        "Case Study 2 B2B/cadenced cross-runtime front summary",
        "",
        "Strict cadenced filter: latency <= budget, deadline misses == 0, window latency <= 2000 ms, window energy > 0.",
        f"Cadenced NAS successful non-pruned rows: {len(cadenced_success)}",
        f"Cadenced NAS rows labeled feasible: {int(label_feasible.sum())}",
        f"Cadenced NAS rows passing strict filter: {int(strict_feasible.sum())}",
        f"Cadenced rows labeled feasible but rejected by strict filter: {int((label_feasible & ~strict_feasible).sum())}",
        f"B2B-on-cadenced replay successful rows: {len(replay_success)}",
        f"B2B-on-cadenced replay rows passing strict filter: {int(replay_strict.sum())}",
        "",
    ]
    for (panel, policy), group in plotted.groupby(["panel", "policy"], sort=False):
        lines.append(f"{panel} / {policy}")
        lines.append(f"  plotted rows: {len(group)}")
        lines.append(f"  front rows: {int(group['is_front'].sum())}")
        lines.append(f"  x range: {group['x_value'].min():.6g} to {group['x_value'].max():.6g}")
        lines.append(f"  RMSE range: {group['rmse'].min():.6g} to {group['rmse'].max():.6g}")
    return "\n".join(lines) + "\n"


def render_figure(
    *,
    study_points_csv: Path,
    overlay_points_csv: Path,
    b2b_on_cadenced_replay_csv: Path,
    cadenced_log_csv: Path,
    output_dir: Path,
    plot_name: str,
    left_x_max: float,
    right_x_max: float,
    b2b_display_label: str,
    legend_font_scale: float,
    marker_size_scale: float,
    legend_marker_scale: float,
    axes_top: float,
) -> tuple[Path, Path, Path, Path]:
    """Render the cross-runtime figure and write sidecar data."""

    output_dir.mkdir(parents=True, exist_ok=True)
    b2b_points, cadenced_points = load_study_points(study_points_csv)
    cadenced_replay_points = load_cadenced_replay_on_b2b(overlay_points_csv)
    b2b_replay_points = load_b2b_replay_on_cadenced(b2b_on_cadenced_replay_csv)

    left_panel = add_front_flags(pd.concat([b2b_points, cadenced_replay_points], ignore_index=True))
    right_panel = add_front_flags(pd.concat([cadenced_points, b2b_replay_points], ignore_index=True))

    font_scale = 1.810
    with plt.rc_context(
        {
            "font.size": 7.8 * font_scale,
            "axes.titlesize": 8.6 * font_scale,
            "axes.labelsize": 7.8 * font_scale,
            "xtick.labelsize": 6.8 * font_scale,
            "ytick.labelsize": 6.8 * font_scale,
            "legend.fontsize": 6.9 * font_scale * legend_font_scale,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(CASE3_MATCH_WIDTH_IN, CASE3_MATCH_HEIGHT_IN),
            sharey=True,
        )
        left_ax, right_ax = axes

        draw_policy(
            left_ax,
            left_panel,
            "B2B NAS",
            color=B2B_COLOR,
            linestyle="-",
            marker_size_scale=marker_size_scale,
        )
        draw_policy(
            left_ax,
            left_panel,
            "Cadenced Pareto replay",
            color=CADENCED_COLOR,
            linestyle="--",
            marker_size_scale=marker_size_scale,
        )
        left_ax.set_xlim(0.0, left_x_max)
        left_ax.set_xlabel("Energy per inference (mJ)", labelpad=2)
        left_ax.set_ylabel("Aggregate RMSE", labelpad=2)
        left_ax.grid(True, alpha=0.3)

        draw_policy(
            right_ax,
            right_panel,
            "Cadenced NAS",
            color=CADENCED_COLOR,
            linestyle="-",
            marker_size_scale=marker_size_scale,
        )
        draw_policy(
            right_ax,
            right_panel,
            "B2B Pareto replay",
            color=B2B_COLOR,
            linestyle="--",
            marker_size_scale=marker_size_scale,
        )
        right_ax.set_xlim(161.0, right_x_max)
        if right_x_max <= 178.0:
            right_ax.set_xticks(np.arange(162.5, right_x_max, 5.0))
        right_ax.set_xlabel("Energy per window (mJ)", labelpad=2)
        right_ax.grid(True, alpha=0.3)

        left_ax.set_ylim(*Y_AXIS_LIMITS)
        left_ax.set_yticks(Y_AXIS_TICKS)

        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=B2B_COLOR,
                markeredgecolor="black",
                markersize=4.6 * marker_size_scale**0.5 * legend_marker_scale,
                linestyle="none",
                label=b2b_display_label,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=CADENCED_COLOR,
                markeredgecolor="black",
                markersize=4.6 * marker_size_scale**0.5 * legend_marker_scale,
                linestyle="none",
                label="Cadenced",
            ),
            Line2D([0], [0], color="black", linewidth=1.2, linestyle="-", label="Native"),
            Line2D([0], [0], color="black", linewidth=1.2, linestyle="--", label="Replay"),
        ]
        fig.legend(
            handles=handles,
            loc="upper center",
            ncol=4,
            bbox_to_anchor=(0.5, 0.99),
            frameon=False,
            columnspacing=0.85,
            handletextpad=0.36,
        )
        fig.subplots_adjust(left=0.102, right=0.994, bottom=0.205, top=axes_top, wspace=0.052)

        png_path = output_dir / f"{plot_name}.png"
        pdf_path = output_dir / f"{plot_name}.pdf"
        for ax in axes:
            ax.set_facecolor("white")
        fig.patch.set_facecolor("white")
        fig.savefig(png_path, dpi=300, facecolor="white")
        fig.savefig(pdf_path, facecolor="white", bbox_inches="tight", pad_inches=0.006)
        plt.close(fig)

    plotted = pd.concat([left_panel, right_panel], ignore_index=True)
    csv_path = output_dir / f"{plot_name}_plotted_points.csv"
    summary_path = output_dir / f"{plot_name}_summary.txt"
    plotted.to_csv(csv_path, index=False)
    summary_path.write_text(
        summary_text(
            plotted=plotted,
            cadenced_csv=cadenced_log_csv,
            b2b_replay_csv=b2b_on_cadenced_replay_csv,
        ),
        encoding="utf-8",
    )
    return png_path, pdf_path, csv_path, summary_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-points-csv",
        default=(
            "models/crest_case_2_v2/comparisons/case2_1_t3_vs_case2_2_t3_study_pareto/"
            "case2_study_pareto_side_by_side_plotted_points.csv"
        ),
        help="Case 2 native study points CSV.",
    )
    parser.add_argument(
        "--overlay-points-csv",
        default=(
            "models/crest_case_2_v2/comparisons/case2_1_b2b_vs_case2_2_cadenced_replay_b2b/"
            "case2_cadenced_pareto_replay_b2b_overlay_points.csv"
        ),
        help="Cadenced-Pareto-on-B2B overlay points CSV.",
    )
    parser.add_argument(
        "--b2b-on-cadenced-replay-csv",
        default=(
            "models/crest_case_2_v2/replays/data/b2b_case2_1_t3_pareto_on_stm32_cadenced_case2_2_t3/"
            "replay_results.csv"
        ),
        help="B2B-Pareto-on-cadenced replay results CSV.",
    )
    parser.add_argument(
        "--cadenced-log-csv",
        default=(
            "models/crest_case_2_v2/nas_runs/OxIOD_STM32_CADENCED_case2_2_t3/"
            "log_NAS_OxIOD_STM32_CADENCED_case2_2_t3.csv"
        ),
        help="Native cadenced NAS log CSV, used for filter accounting.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--plot-name", default=DEFAULT_PLOT_NAME, help="Output filename stem.")
    parser.add_argument("--left-x-max", type=float, default=165.0, help="Left panel x-axis upper limit.")
    parser.add_argument("--right-x-max", type=float, default=186.5, help="Right panel x-axis upper limit.")
    parser.add_argument("--b2b-display-label", default="B2B", help="Display label for B2B/continuous series.")
    parser.add_argument("--legend-font-scale", type=float, default=1.0, help="Multiplier for legend text size.")
    parser.add_argument("--marker-size-scale", type=float, default=1.0, help="Multiplier for plotted marker areas.")
    parser.add_argument("--legend-marker-scale", type=float, default=1.0, help="Multiplier for legend marker size.")
    parser.add_argument("--axes-top", type=float, default=0.875, help="Top edge of the plot area in figure coordinates.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the plotting CLI."""

    args = build_arg_parser().parse_args(argv)
    for path in render_figure(
        study_points_csv=Path(args.study_points_csv),
        overlay_points_csv=Path(args.overlay_points_csv),
        b2b_on_cadenced_replay_csv=Path(args.b2b_on_cadenced_replay_csv),
        cadenced_log_csv=Path(args.cadenced_log_csv),
        output_dir=Path(args.output_dir),
        plot_name=args.plot_name,
        left_x_max=args.left_x_max,
        right_x_max=args.right_x_max,
        b2b_display_label=args.b2b_display_label,
        legend_font_scale=args.legend_font_scale,
        marker_size_scale=args.marker_size_scale,
        legend_marker_scale=args.legend_marker_scale,
        axes_top=args.axes_top,
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
