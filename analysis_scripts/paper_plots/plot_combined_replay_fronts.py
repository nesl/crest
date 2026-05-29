#!/usr/bin/env python3
"""Plot combined replay-vs-CREST fronts for multiple HIL replay runs."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
HIL_REPLAY_DIR = REPO_ROOT / "analysis_scripts" / "hil_replay"
for import_dir in (SCRIPT_DIR, HIL_REPLAY_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from compare_replayed_pareto import (  # noqa: E402
    FrontPoint,
    SUCCESS_ERROR_CODE,
    energy_regret_rows,
    filter_latency_feasible,
    find_log_csv,
    load_crest_points,
    load_replay_points,
    pareto_front,
    parse_float,
    read_csv_rows,
    resolve_replay_results,
)


MARKERS = ("o", "s", "^", "D", "P", "X", "v", "<", ">")
COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b")
CREST_COLOR = "#1f77b4"
REPLAY_COLOR = "#ff7f0e"


@dataclass(frozen=True)
class PairInput:
    """One labeled replay comparison input.

    Parameters
    ----------
    label : str
        Human-readable label used in plot legends.
    crest_run_dir : pathlib.Path
        Measured-energy NAS run directory containing ``log_NAS_*.csv``.
    replay_path : pathlib.Path
        Replay output directory or explicit ``replay_results.csv`` path.
    placeholder : bool
        Whether this input is an empty placeholder panel.
    """

    label: str
    crest_run_dir: Path | None
    replay_path: Path | None
    placeholder: bool = False


@dataclass(frozen=True)
class LoadedPair:
    """Loaded points and fronts for one comparison pair.

    Parameters
    ----------
    label : str
        Human-readable label used in plot legends.
    crest_points : list[FrontPoint]
        Valid measured-energy NAS points.
    replay_points : list[FrontPoint]
        Valid replayed proxy points.
    crest_front : list[FrontPoint]
        Pareto front from ``crest_points``.
    replay_front : list[FrontPoint]
        Pareto front from ``replay_points``.
    regret_rows : list[dict[str, Any]]
        Energy-regret rows comparing replay front against CREST front.
    crest_front_outcomes : CandidateOutcomeSummary
        Timing outcome counts for the measured-energy NAS front.
    replay_outcomes : CandidateOutcomeSummary
        Raw replay outcome counts for all selected source-front candidates.
    placeholder : bool
        Whether this pair is an empty placeholder panel.
    """

    label: str
    crest_points: list[FrontPoint]
    replay_points: list[FrontPoint]
    crest_front: list[FrontPoint]
    replay_front: list[FrontPoint]
    regret_rows: list[dict[str, Any]]
    crest_front_outcomes: "CandidateOutcomeSummary"
    replay_outcomes: "CandidateOutcomeSummary"
    placeholder: bool = False


@dataclass(frozen=True)
class CandidateOutcomeSummary:
    """Timing and HIL outcome counts for Pareto-front candidates.

    Parameters
    ----------
    total_candidates : int
        Number of candidates included in the outcome summary.
    timing_feasible : int
        Measurements with latency within the configured budget.
    timing_infeasible : int
        Measurements with latency above the configured budget.
    hil_failed : int
        HIL measurements that reported an error before producing valid metrics.
    replay_failed : int
        Replay rows that did not complete before target measurement interpretation.
    timing_unknown : int
        Successful measurements without a usable latency or budget.
    """

    total_candidates: int
    timing_feasible: int
    timing_infeasible: int
    hil_failed: int
    replay_failed: int
    timing_unknown: int


def parse_pair(value: str) -> PairInput:
    """Parse a ``LABEL=CREST_RUN_DIR,REPLAY_DIR`` pair argument.

    Parameters
    ----------
    value : str
        Raw CLI pair argument.

    Returns
    -------
    PairInput
        Parsed and path-expanded pair input.

    Raises
    ------
    ValueError
        If the argument is malformed.
    """

    if "=" not in value:
        raise ValueError("--pair must use LABEL=CREST_RUN_DIR,REPLAY_DIR")
    label, paths = value.split("=", 1)
    label = label.strip()
    parts = [part.strip() for part in paths.split(",", 1)]
    if not label or len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError("--pair must use LABEL=CREST_RUN_DIR,REPLAY_DIR")
    return PairInput(
        label=label,
        crest_run_dir=Path(parts[0]).expanduser().resolve(),
        replay_path=Path(parts[1]).expanduser().resolve(),
    )


def parse_panel(value: str) -> PairInput:
    """Parse an ordered real or placeholder panel argument.

    Parameters
    ----------
    value : str
        Raw panel argument. ``LABEL=CREST_RUN_DIR,REPLAY_DIR`` creates a real
        panel, while ``LABEL`` creates an empty placeholder panel.

    Returns
    -------
    PairInput
        Parsed panel input.
    """

    if "=" in value:
        return parse_pair(value)
    label = value.strip()
    if not label:
        raise ValueError("--panel placeholder labels cannot be empty")
    return PairInput(label=label, crest_run_dir=None, replay_path=None, placeholder=True)


def validate_pairs(pairs: Sequence[PairInput]) -> None:
    """Validate pair labels and paths before loading data.

    Parameters
    ----------
    pairs : Sequence[PairInput]
        Pair inputs to validate.

    Raises
    ------
    ValueError
        If labels are duplicated.
    FileNotFoundError
        If any referenced directory or file is missing.
    """

    labels = [pair.label for pair in pairs]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError(f"Duplicate pair label(s): {', '.join(duplicates)}")
    for pair in pairs:
        if pair.placeholder:
            continue
        if pair.crest_run_dir is None or pair.replay_path is None:
            raise ValueError(f"Panel paths are required for non-placeholder label: {pair.label}")
        if not pair.crest_run_dir.is_dir():
            raise FileNotFoundError(f"CREST run directory does not exist: {pair.crest_run_dir}")
        if not pair.replay_path.exists():
            raise FileNotFoundError(f"Replay path does not exist: {pair.replay_path}")


def load_pair(pair: PairInput, *, feasible_only: bool) -> LoadedPair:
    """Load one replay comparison pair.

    Parameters
    ----------
    pair : PairInput
        Pair to load.
    feasible_only : bool
        Whether to keep only latency-feasible points.

    Returns
    -------
    LoadedPair
        Loaded points, Pareto fronts, and regret rows.
    """

    if pair.placeholder:
        empty_outcomes = CandidateOutcomeSummary(
            total_candidates=0,
            timing_feasible=0,
            timing_infeasible=0,
            hil_failed=0,
            replay_failed=0,
            timing_unknown=0,
        )
        return LoadedPair(
            label=pair.label,
            crest_points=[],
            replay_points=[],
            crest_front=[],
            replay_front=[],
            regret_rows=[],
            crest_front_outcomes=empty_outcomes,
            replay_outcomes=empty_outcomes,
            placeholder=True,
        )
    if pair.crest_run_dir is None or pair.replay_path is None:
        raise ValueError(f"Panel paths are required for non-placeholder label: {pair.label}")
    crest_csv = find_log_csv(pair.crest_run_dir)
    replay_csv = resolve_replay_results(pair.replay_path)
    crest_points = filter_latency_feasible(load_crest_points(crest_csv), feasible_only)
    replay_points = filter_latency_feasible(load_replay_points(replay_csv), feasible_only)
    crest_front = pareto_front(crest_points)
    replay_front = pareto_front(replay_points)
    regret_rows = energy_regret_rows(crest_front=crest_front, proxy_front=replay_front)
    crest_front_outcomes = summarize_point_outcomes(crest_front)
    replay_outcomes = summarize_replay_outcomes(replay_csv)
    return LoadedPair(
        label=pair.label,
        crest_points=crest_points,
        replay_points=replay_points,
        crest_front=crest_front,
        replay_front=replay_front,
        regret_rows=regret_rows,
        crest_front_outcomes=crest_front_outcomes,
        replay_outcomes=replay_outcomes,
        placeholder=False,
    )


def summarize_point_outcomes(points: Sequence[FrontPoint]) -> CandidateOutcomeSummary:
    """Summarize timing outcomes for already-valid measured points.

    Parameters
    ----------
    points : Sequence[FrontPoint]
        Valid measured points to summarize.

    Returns
    -------
    CandidateOutcomeSummary
        Timing feasibility counts for the provided points.
    """

    timing_feasible = 0
    timing_infeasible = 0
    timing_unknown = 0
    for point in points:
        if point.latency_feasible is True:
            timing_feasible += 1
        elif point.latency_feasible is False:
            timing_infeasible += 1
        else:
            timing_unknown += 1
    return CandidateOutcomeSummary(
        total_candidates=len(points),
        timing_feasible=timing_feasible,
        timing_infeasible=timing_infeasible,
        hil_failed=0,
        replay_failed=0,
        timing_unknown=timing_unknown,
    )


def summarize_replay_outcomes(csv_path: Path) -> CandidateOutcomeSummary:
    """Summarize HIL replay outcomes for all scheduled replay candidates.

    Parameters
    ----------
    csv_path : pathlib.Path
        Replay results CSV.

    Returns
    -------
    CandidateOutcomeSummary
        Counts for successful timing-feasible measurements, timing violations,
        HIL failures, replay failures, and rows without latency information.
    """

    timing_feasible = 0
    timing_infeasible = 0
    hil_failed = 0
    replay_failed = 0
    timing_unknown = 0
    rows = read_csv_rows(csv_path)
    for row in rows:
        if row.get("replay_status") != "completed":
            replay_failed += 1
            continue
        error_code = parse_float(row.get("target__error_code"))
        if error_code is None or int(error_code) != SUCCESS_ERROR_CODE:
            hil_failed += 1
            continue
        latency = parse_float(row.get("target__latency_ms"))
        budget = parse_float(row.get("target__latency_budget_ms"))
        if latency is None or budget is None or latency <= 0.0 or budget <= 0.0:
            timing_unknown += 1
        elif latency <= budget:
            timing_feasible += 1
        else:
            timing_infeasible += 1
    return CandidateOutcomeSummary(
        total_candidates=len(rows),
        timing_feasible=timing_feasible,
        timing_infeasible=timing_infeasible,
        hil_failed=hil_failed,
        replay_failed=replay_failed,
        timing_unknown=timing_unknown,
    )


def scatter_points_by_latency(
    ax: Any,
    points: Sequence[FrontPoint],
    *,
    color: str,
    marker: str,
    label: str,
    alpha: float,
    size: float,
    linewidth: float,
    include_legend: bool,
) -> None:
    """Scatter points with marker fill encoding latency feasibility.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    points : Sequence[FrontPoint]
        Points to draw.
    color : str
        Series color.
    marker : str
        Matplotlib marker.
    label : str
        Base legend label.
    alpha : float
        Marker transparency.
    size : float
        Marker size.
    linewidth : float
        Marker outline width.
    include_legend : bool
        Whether this scatter call should add fill-state entries to the legend.
    """

    feasible = [point for point in points if point.latency_feasible is not False]
    infeasible = [point for point in points if point.latency_feasible is False]
    feasible_label = f"{label} (<= budget)" if include_legend else "_nolegend_"
    infeasible_label = f"{label} (> budget)" if include_legend else "_nolegend_"
    if feasible:
        ax.scatter(
            [point.energy_mj for point in feasible],
            [point.rmse for point in feasible],
            s=size,
            color=color,
            marker=marker,
            edgecolors="black",
            linewidth=linewidth,
            alpha=alpha,
            label=feasible_label,
            zorder=3 if alpha >= 0.95 else 2,
        )
    if infeasible:
        ax.scatter(
            [point.energy_mj for point in infeasible],
            [point.rmse for point in infeasible],
            s=size + 12,
            facecolors="none",
            edgecolors=color,
            marker=marker,
            linewidth=max(linewidth, 1.3),
            alpha=max(alpha, 0.85),
            label=infeasible_label,
            zorder=3 if alpha >= 0.95 else 2,
        )


def scatter_policy_points_by_latency(
    ax: Any,
    points: Sequence[FrontPoint],
    *,
    color: str,
    marker: str,
    alpha: float,
    size: float,
    linewidth: float,
) -> None:
    """Scatter policy points without adding per-panel legend entries.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    points : Sequence[FrontPoint]
        Points to draw.
    color : str
        Marker color for the policy.
    marker : str
        Matplotlib marker.
    alpha : float
        Marker transparency.
    size : float
        Marker size.
    linewidth : float
        Marker outline width.
    """

    scatter_points_by_latency(
        ax,
        points,
        color=color,
        marker=marker,
        label="_nolegend_",
        alpha=alpha,
        size=size,
        linewidth=linewidth,
        include_legend=False,
    )


def plot_combined_fronts(
    pairs: Sequence[LoadedPair],
    output_path: Path,
    *,
    title: str,
    x_scale: str,
) -> None:
    """Plot measured-energy and replay fronts for multiple targets.

    Parameters
    ----------
    pairs : Sequence[LoadedPair]
        Loaded comparison pairs.
    output_path : pathlib.Path
        Destination PNG path.
    title : str
        Plot title.
    x_scale : {"linear", "log"}
        Scale used for the measured-energy x-axis.
    """

    fig, ax = plt.subplots(figsize=(11, 7))
    real_pairs = [pair for pair in pairs if not pair.placeholder]
    for index, pair in enumerate(real_pairs):
        color = COLORS[index % len(COLORS)]
        marker = MARKERS[index % len(MARKERS)]
        crest_front_keys = {point.payload_key for point in pair.crest_front}
        replay_front_keys = {point.payload_key for point in pair.replay_front}
        non_front_crest = [point for point in pair.crest_points if point.payload_key not in crest_front_keys]
        non_front_replay = [point for point in pair.replay_points if point.payload_key not in replay_front_keys]
        scatter_points_by_latency(
            ax,
            non_front_crest,
            color=color,
            marker=marker,
            label=f"{pair.label} CREST candidates",
            alpha=0.18,
            size=30,
            linewidth=0.5,
            include_legend=True,
        )
        scatter_points_by_latency(
            ax,
            non_front_replay,
            color=color,
            marker=marker,
            label=f"{pair.label} replay candidates",
            alpha=0.55,
            size=44,
            linewidth=0.8,
            include_legend=True,
        )
        if pair.crest_front:
            ax.plot(
                [point.energy_mj for point in pair.crest_front],
                [point.rmse for point in pair.crest_front],
                color=color,
                linestyle="-",
                linewidth=2.2,
                label=f"{pair.label} CREST front",
            )
            scatter_points_by_latency(
                ax,
                pair.crest_front,
                color=color,
                marker=marker,
                label=f"{pair.label} CREST front points",
                alpha=1.0,
                size=72,
                linewidth=1.0,
                include_legend=False,
            )
        if pair.replay_front:
            ax.plot(
                [point.energy_mj for point in pair.replay_front],
                [point.rmse for point in pair.replay_front],
                color=color,
                linestyle="--",
                linewidth=2.2,
                label=f"{pair.label} replay front",
            )
            scatter_points_by_latency(
                ax,
                pair.replay_front,
                color=color,
                marker=marker,
                label=f"{pair.label} replay front points",
                alpha=1.0,
                size=82,
                linewidth=1.1,
                include_legend=False,
            )
    ax.set_title(title)
    ax.set_xlabel("Measured energy per inference on target board (mJ)")
    ax.set_ylabel("Aggregate RMSE (lower is better)")
    ax.set_xscale(x_scale)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize="small", ncols=1)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_subplot_fronts(
    pairs: Sequence[LoadedPair],
    output_path: Path,
    *,
    title: str,
    x_scale: str,
    subplot_cols: int,
) -> None:
    """Plot one replay-vs-CREST front panel per target.

    Parameters
    ----------
    pairs : Sequence[LoadedPair]
        Loaded comparison pairs.
    output_path : pathlib.Path
        Destination PNG path.
    title : str
        Figure title.
    x_scale : {"linear", "log"}
        Scale used for each measured-energy x-axis.
    subplot_cols : int
        Number of subplot columns.
    """

    if not pairs:
        return
    cols = max(1, min(int(subplot_cols), len(pairs)))
    rows = int(math.ceil(len(pairs) / cols))
    fig_width = max(5.2 * cols, 7.0)
    fig_height = max(4.6 * rows, 4.8)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), squeeze=False, sharey=True)
    axes_flat = list(axes.ravel())
    for index, pair in enumerate(pairs):
        ax = axes_flat[index]
        if pair.placeholder:
            ax.set_title(pair.label)
            ax.set_axis_off()
            ax.text(
                0.5,
                0.5,
                "Replay pending",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize="large",
            )
            continue
        crest_front_keys = {point.payload_key for point in pair.crest_front}
        replay_front_keys = {point.payload_key for point in pair.replay_front}
        non_front_crest = [point for point in pair.crest_points if point.payload_key not in crest_front_keys]
        non_front_replay = [point for point in pair.replay_points if point.payload_key not in replay_front_keys]

        scatter_policy_points_by_latency(
            ax,
            non_front_crest,
            color=CREST_COLOR,
            marker="o",
            alpha=0.18,
            size=28,
            linewidth=0.45,
        )
        scatter_policy_points_by_latency(
            ax,
            non_front_replay,
            color=REPLAY_COLOR,
            marker="s",
            alpha=0.45,
            size=42,
            linewidth=0.75,
        )
        if pair.crest_front:
            ax.plot(
                [point.energy_mj for point in pair.crest_front],
                [point.rmse for point in pair.crest_front],
                color=CREST_COLOR,
                linestyle="-",
                linewidth=2.2,
            )
            scatter_policy_points_by_latency(
                ax,
                pair.crest_front,
                color=CREST_COLOR,
                marker="o",
                alpha=1.0,
                size=70,
                linewidth=1.0,
            )
        if pair.replay_front:
            ax.plot(
                [point.energy_mj for point in pair.replay_front],
                [point.rmse for point in pair.replay_front],
                color=REPLAY_COLOR,
                linestyle="--",
                linewidth=2.2,
            )
            scatter_policy_points_by_latency(
                ax,
                pair.replay_front,
                color=REPLAY_COLOR,
                marker="s",
                alpha=1.0,
                size=78,
                linewidth=1.05,
            )
        ax.set_title(pair.label)
        ax.set_xscale(x_scale)
        ax.grid(True, alpha=0.3)
        ax.text(
            0.02,
            0.03,
            f"CREST front: {len(pair.crest_front)}\nReplay front: {len(pair.replay_front)}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize="small",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 3.0},
        )
    for ax in axes_flat[len(pairs) :]:
        ax.set_axis_off()
    for ax in axes[-1, :]:
        if ax.has_data():
            ax.set_xlabel("Measured energy per inference (mJ)")
    for ax in axes[:, 0]:
        if ax.has_data():
            ax.set_ylabel("Aggregate RMSE (lower is better)")

    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], color=CREST_COLOR, linewidth=2.2, linestyle="-", label="CREST front"),
        Line2D([0], [0], color=REPLAY_COLOR, linewidth=2.2, linestyle="--", label="FLOPs replay front"),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            color="black",
            markerfacecolor="white",
            markeredgecolor="black",
            label="Open marker: > 200 ms",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            color="black",
            markerfacecolor="black",
            markeredgecolor="black",
            label="Filled marker: <= 200 ms",
        ),
    ]
    fig.suptitle(title)
    fig.legend(handles=legend_handles, loc="lower center", ncols=4, fontsize="small")
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_candidate_outcome_counts(
    pairs: Sequence[LoadedPair],
    output_path: Path,
    *,
    title: str,
) -> None:
    """Plot outcome counts for CREST fronts and replayed proxy candidates.

    Parameters
    ----------
    pairs : Sequence[LoadedPair]
        Loaded comparison pairs.
    output_path : pathlib.Path
        Destination PNG path.
    title : str
        Plot title.
    """

    categories = [
        ("timing_feasible", "Measured <= budget", "#2ca02c"),
        ("timing_infeasible", "Measured > budget", "#ff7f0e"),
        ("hil_failed", "HIL failed", "#d62728"),
        ("replay_failed", "Replay failed", "#7f7f7f"),
        ("timing_unknown", "Timing unknown", "#9467bd"),
    ]
    bar_inputs: list[tuple[str, CandidateOutcomeSummary]] = []
    for pair in pairs:
        bar_inputs.append((f"{pair.label}\nCREST front", pair.crest_front_outcomes))
        bar_inputs.append((f"{pair.label}\nFLOPs replay", pair.replay_outcomes))
    labels = [label for label, _summary in bar_inputs]
    x_positions = np.arange(len(labels), dtype=float)
    bottoms = np.zeros(len(labels), dtype=float)
    fig, ax = plt.subplots(figsize=(11, 5.8))
    for field, category_label, color in categories:
        values = np.asarray([getattr(summary, field) for _label, summary in bar_inputs], dtype=float)
        if not np.any(values):
            continue
        ax.bar(
            x_positions,
            values,
            bottom=bottoms,
            color=color,
            edgecolor="black",
            linewidth=0.6,
            label=category_label,
        )
        bottoms += values
    for index, (_label, summary) in enumerate(bar_inputs):
        ax.text(
            x_positions[index],
            bottoms[index] + 0.25,
            f"n={summary.total_candidates}",
            ha="center",
            va="bottom",
            fontsize="small",
        )
    ax.set_title(title)
    ax.set_ylabel("Pareto candidates")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(top=max(float(np.max(bottoms)) + 2.0, 1.0))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_combined_regret(
    pairs: Sequence[LoadedPair],
    output_path: Path,
    *,
    title: str,
    x_scale: str,
) -> None:
    """Plot energy regret curves for multiple replay comparisons.

    Parameters
    ----------
    pairs : Sequence[LoadedPair]
        Loaded comparison pairs.
    output_path : pathlib.Path
        Destination PNG path.
    title : str
        Plot title.
    x_scale : {"linear", "log"}
        Scale used for the replayed proxy RMSE x-axis.
    """

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
    for index, pair in enumerate(pairs):
        if not pair.regret_rows:
            continue
        color = COLORS[index % len(COLORS)]
        rows = sorted(pair.regret_rows, key=lambda row: float(row["proxy_rmse"]))
        rmse = np.asarray([float(row["proxy_rmse"]) for row in rows], dtype=float)
        percent = np.asarray([float(row["energy_percent_increase"]) for row in rows], dtype=float)
        ax.plot(rmse, percent, marker=MARKERS[index % len(MARKERS)], color=color, label=pair.label)
    ax.set_title(title)
    ax.set_xlabel("Replayed proxy candidate RMSE")
    ax.set_ylabel("Energy increase vs matched CREST front (%)")
    ax.set_xscale(x_scale)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def summarize_pair(pair: LoadedPair) -> dict[str, Any]:
    """Build a compact summary row for one loaded pair.

    Parameters
    ----------
    pair : LoadedPair
        Loaded comparison pair.

    Returns
    -------
    dict[str, Any]
        Summary row suitable for CSV output.
    """

    if pair.regret_rows:
        deltas = np.asarray([float(row["energy_delta_mj"]) for row in pair.regret_rows], dtype=float)
        ratios = np.asarray([float(row["energy_ratio"]) for row in pair.regret_rows], dtype=float)
        median_delta = float(np.median(deltas))
        max_delta = float(np.max(deltas))
        median_ratio = float(np.median(ratios))
        max_ratio = float(np.max(ratios))
    else:
        median_delta = max_delta = median_ratio = max_ratio = float("nan")
    return {
        "label": pair.label,
        "crest_valid_points": len(pair.crest_points),
        "replay_valid_points": len(pair.replay_points),
        "crest_front_points": len(pair.crest_front),
        "replay_front_points": len(pair.replay_front),
        "crest_front_timing_feasible": pair.crest_front_outcomes.timing_feasible,
        "crest_front_timing_infeasible": pair.crest_front_outcomes.timing_infeasible,
        "crest_front_timing_unknown": pair.crest_front_outcomes.timing_unknown,
        "replay_total_candidates": pair.replay_outcomes.total_candidates,
        "replay_timing_feasible": pair.replay_outcomes.timing_feasible,
        "replay_timing_infeasible": pair.replay_outcomes.timing_infeasible,
        "replay_hil_failed": pair.replay_outcomes.hil_failed,
        "replay_failed": pair.replay_outcomes.replay_failed,
        "replay_timing_unknown": pair.replay_outcomes.timing_unknown,
        "regret_rows": len(pair.regret_rows),
        "median_energy_delta_mj": median_delta,
        "max_energy_delta_mj": max_delta,
        "median_energy_ratio": median_ratio,
        "max_energy_ratio": max_ratio,
    }


def write_summary_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write combined comparison summary rows.

    Parameters
    ----------
    path : pathlib.Path
        Destination CSV path.
    rows : Sequence[Mapping[str, Any]]
        Summary rows to write.
    """

    fieldnames = [
        "label",
        "crest_valid_points",
        "replay_valid_points",
        "crest_front_points",
        "replay_front_points",
        "crest_front_timing_feasible",
        "crest_front_timing_infeasible",
        "crest_front_timing_unknown",
        "replay_total_candidates",
        "replay_timing_feasible",
        "replay_timing_infeasible",
        "replay_hil_failed",
        "replay_failed",
        "replay_timing_unknown",
        "regret_rows",
        "median_energy_delta_mj",
        "max_energy_delta_mj",
        "median_energy_ratio",
        "max_energy_ratio",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """

    parser = argparse.ArgumentParser(
        description="Plot combined replay-vs-CREST fronts for multiple HIL replay runs."
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=None,
        help="Comparison pair as LABEL=CREST_RUN_DIR,REPLAY_DIR. Repeat for multiple targets.",
    )
    parser.add_argument(
        "--panel",
        action="append",
        default=None,
        help=(
            "Ordered subplot panel. Use LABEL=CREST_RUN_DIR,REPLAY_DIR for data "
            "or LABEL for a placeholder panel. Repeat to control order."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for combined plots and summary CSV.",
    )
    parser.add_argument(
        "--fronts-stem",
        required=True,
        help="Output filename stem for the combined front plot.",
    )
    parser.add_argument(
        "--subplot-fronts-stem",
        required=True,
        help="Output filename stem for the faceted front plot.",
    )
    parser.add_argument(
        "--regret-stem",
        required=True,
        help="Output filename stem for the energy-regret plot.",
    )
    parser.add_argument(
        "--candidate-outcomes-stem",
        required=True,
        help="Output filename stem for the candidate-outcome plot.",
    )
    parser.add_argument(
        "--summary-stem",
        required=True,
        help="Output filename stem for the summary CSV.",
    )
    parser.add_argument(
        "--feasible-only",
        action="store_true",
        help="Use only latency-feasible points from each comparison.",
    )
    parser.add_argument(
        "--title",
        default="Combined FLOPs-proxy replay vs measured-energy NAS fronts",
        help="Title for the combined front plot.",
    )
    parser.add_argument(
        "--x-scale",
        choices=("linear", "log"),
        default="linear",
        help="X-axis scale for the combined plots.",
    )
    parser.add_argument(
        "--subplot-cols",
        type=int,
        default=2,
        help="Number of columns for the faceted front plot.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the combined replay plotting CLI.

    Parameters
    ----------
    argv : Sequence[str] | None, optional
        CLI arguments excluding program name. ``None`` uses ``sys.argv``.

    Returns
    -------
    int
        Exit status code.
    """

    args = build_arg_parser().parse_args(argv)
    if args.panel:
        pair_inputs = [parse_panel(value) for value in args.panel]
    elif args.pair:
        pair_inputs = [parse_pair(value) for value in args.pair]
    else:
        raise ValueError("At least one --pair or --panel is required.")
    validate_pairs(pair_inputs)
    loaded_pairs = [load_pair(pair, feasible_only=bool(args.feasible_only)) for pair in pair_inputs]
    output_dir = Path(args.output_dir).expanduser().resolve()
    suffix = "_feasible_only" if args.feasible_only else ""
    plot_combined_fronts(
        loaded_pairs,
        output_dir / f"{args.fronts_stem}{suffix}.png",
        title=args.title,
        x_scale=args.x_scale,
    )
    plot_subplot_fronts(
        loaded_pairs,
        output_dir / f"{args.subplot_fronts_stem}{suffix}.png",
        title=args.title,
        x_scale=args.x_scale,
        subplot_cols=args.subplot_cols,
    )
    plot_combined_regret(
        loaded_pairs,
        output_dir / f"{args.regret_stem}{suffix}.png",
        title="Combined FLOPs-proxy replay energy regret",
        x_scale=args.x_scale,
    )
    plot_candidate_outcome_counts(
        loaded_pairs,
        output_dir / f"{args.candidate_outcomes_stem}{suffix}.png",
        title="CREST front and FLOPs-proxy replay candidate outcomes",
    )
    summary_rows = [summarize_pair(pair) for pair in loaded_pairs]
    write_summary_csv(output_dir / f"{args.summary_stem}{suffix}.csv", summary_rows)
    for row in summary_rows:
        print(
            f"{row['label']}: CREST front={row['crest_front_points']}, "
            f"replay front={row['replay_front_points']}, regret rows={row['regret_rows']}"
        )
    print(f"Wrote combined outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
