#!/usr/bin/env python3
# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Render the CS3 proposal comparison from explicitly supplied external CSVs.

Retain every feasible observation and repeat; stars select maximum recorded
score per board and method. The 400 mJ line is a scoring allocation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, FixedLocator, NullFormatter, FuncFormatter
from PIL import Image
from matplotlib.figure import Figure

COLORS = {"Optuna": "#0072B2", "Memory": "#D55E00", "History": "#009E73"}
LABELS = {"Optuna": "TPE", "Memory": "LLM Memory", "History": "LLM History"}
BOARDS = ["STM32", "Portenta"]
TITLES = {"STM32": "STM32 N657", "Portenta": "Portenta H7 CM7"}


def load(
    input_path: Path, endpoint_path: Path, trials_per_group: int = 200
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    """Load and validate the paper figure's observation and endpoint tables.

    Parameters
    ----------
    input_path : pathlib.Path
        CSV with board, method, trial, f1, energy_mj, score, latency_ms columns.
        Each board/method pair must have the requested distinct trial count.
    endpoint_path : pathlib.Path
        CSV with study and best_individual columns; study keys follow
        ``<lowercase board>_<lowercase method>``.

    trials_per_group : int, optional
        Expected observations per board/method pair; the paper uses 200.

    Returns
    -------
    tuple
        Observation dictionaries and maximum-score rows indexed by board/method.

    Raises
    ------
    ValueError
        If observations are nonfinite, duplicated, incomplete, or disagree with
        the recorded objective or independently computed endpoint table.

    Notes
    -----
    The paper protocol has 1,200 observations across the two boards and three
    methods, with score equal to f1 minus energy_mj / 4000.
    """
    if trials_per_group <= 0:
        raise ValueError("trials_per_group must be positive")
    with input_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for r in rows:
        r["trial"] = int(r["trial"])
        for k in ("f1", "energy_mj", "score", "latency_ms"):
            r[k] = float(r[k])
        if not all(
            math.isfinite(r[k]) for k in ("f1", "energy_mj", "score", "latency_ms")
        ):
            raise ValueError("All observation metrics must be finite")
        if not 0 <= r["f1"] <= 1 or r["energy_mj"] <= 0 or r["latency_ms"] < 0:
            raise ValueError("Invalid F1, energy, or latency")
        if abs(r["score"] - (r["f1"] - r["energy_mj"] / 4000)) >= 1e-10:
            raise ValueError("Recorded score differs from F1 - energy_mj / 4000")
        if r["board"] not in BOARDS or r["method"] not in COLORS or r["trial"] < 0:
            raise ValueError("Unknown board/method or negative trial identifier")
    with endpoint_path.open(newline="") as stream:
        endpoint_rows = list(csv.DictReader(stream))
    endpoint = {r["study"]: r for r in endpoint_rows}
    if len(endpoint) != len(endpoint_rows):
        raise ValueError("Duplicate endpoint study identifiers")
    winners = {}
    for board in BOARDS:
        for method in COLORS:
            pool = [r for r in rows if r["board"] == board and r["method"] == method]
            if (
                len(pool) != trials_per_group
                or len({r["trial"] for r in pool}) != trials_per_group
            ):
                raise ValueError(
                    f"{board}/{method}: expected {trials_per_group} distinct trials"
                )
            winner = max(pool, key=lambda r: r["score"])
            key = f"{board.lower()}_{method.lower()}"
            if key not in endpoint:
                raise ValueError(f"Missing endpoint study: {key}")
            e = endpoint[key]
            if (
                not math.isfinite(float(e["best_individual"]))
                or abs(winner["score"] - float(e["best_individual"])) >= 1e-10
            ):
                raise ValueError(
                    f"{board}/{method}: selected score disagrees with endpoint"
                )
            winners[board, method] = winner
    return rows, winners


def render(
    rows: list[dict[str, Any]],
    winners: dict[tuple[str, str], dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    """Render the published logarithmic comparison without changing its styling.

    Parameters
    ----------
    rows : list of dict
        Validated feasible observations in input order, including repeats.
    winners : dict
        Maximum-score observation for each board/method pair.
    output_dir : pathlib.Path
        Explicit destination for PNG, PDF, and SVG files.

    Returns
    -------
    dict
        Physical dimensions, raster dimensions, and resolution of the figure.
    """
    # Preserve the original paper's aspect and typography, legible after
    # reduction to 3.35-inch manuscript column width.
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(6.15, 3.5), sharey=True)
        for ax, board in zip(axes, ["Portenta", "STM32"]):
            pool = [r for r in rows if r["board"] == board]
            for method, color in COLORS.items():
                p = [r for r in pool if r["method"] == method]
                ax.scatter(
                    [r["energy_mj"] for r in p],
                    [r["f1"] for r in p],
                    s=15,
                    c=color,
                    alpha=0.25,
                    linewidths=0,
                )
            for method, color in COLORS.items():
                w = winners[board, method]
                ax.scatter(
                    w["energy_mj"],
                    w["f1"],
                    s=150,
                    marker="*",
                    c=color,
                    edgecolor="#252525",
                    linewidths=1.05,
                    zorder=5,
                    clip_on=False,
                )
            if board == "Portenta":
                # Separate the labels without moving the measured selections.
                for method, position in [
                    ("Optuna", (85, 0.825)),
                    ("Memory", (320, 0.825)),
                ]:
                    winner = winners[board, method]
                    ax.annotate(
                        "TPE" if method == "Optuna" else "Memory",
                        xy=(winner["energy_mj"], winner["f1"]),
                        xytext=position,
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=COLORS[method],
                        arrowprops={
                            "arrowstyle": "-",
                            "color": COLORS[method],
                            "lw": 0.7,
                            "shrinkB": 5,
                        },
                        zorder=7,
                    )
            ax.set(title=TITLES[board], xscale="log", xlim=(3, 1500), ylim=(0, 0.86))
            ax.xaxis.set_major_locator(FixedLocator([10, 100, 1000]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_locator(
                LogLocator(base=10, subs=range(2, 10), numticks=100)
            )
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
            ax.grid(which="major", color=".78", linewidth=0.7)
            ax.grid(which="minor", axis="x", color=".9", linewidth=0.45)
            ax.axvline(400, color=".35", linestyle="--", linewidth=1.1, zorder=3)
            ax.set_axisbelow(True)
            ax.tick_params(axis="x", which="minor", length=2, width=0.5)
            ax.tick_params(pad=2, length=3)
            for s in ax.spines.values():
                s.set_linewidth(0.8)
        axes[0].set_ylabel("Validation macro-F1", labelpad=3)
        handles = [
            Line2D([], [], marker="o", ls="", color=c, markersize=6, label=LABELS[m])
            for m, c in COLORS.items()
        ]
        marks = [
            Line2D(
                [],
                [],
                marker="*",
                ls="",
                color="#333333",
                markerfacecolor=".65",
                markersize=10,
                label="Selected",
            )
        ]
        fig.legend(
            handles=handles
            + marks
            + [Line2D([], [], color=".35", ls="--", lw=1.1, label="400 mJ allocation")],
            ncol=5,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            frameon=False,
            fontsize=11,
            columnspacing=0.4,
            handletextpad=0.2,
            handlelength=0.9,
        )
        fig.supxlabel(
            "Continuous energy per inference (mJ, log scale)",
            x=0.56,
            y=0.015,
            fontsize=13,
        )
        fig.subplots_adjust(left=0.115, right=0.985, bottom=0.17, top=0.82, wspace=0.14)
        return save(fig, output_dir)


def save(fig: Figure, output_dir: Path) -> dict[str, Any]:
    """Check text bounds and write the figure in publication formats.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Fully styled figure.
    output_dir : pathlib.Path
        Explicit output directory.

    Returns
    -------
    dict
        Dimensions in inches and pixels, and PNG resolution in dots per inch.
    """
    name = "cs3_proposal_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for label in fig.findobj(match=lambda obj: isinstance(obj, plt.Text)):
        if label.get_visible() and label.get_text():
            b = label.get_window_extent(renderer)
            if (
                b.x0 < -1
                or b.y0 < -1
                or b.x1 > fig.bbox.width + 1
                or b.y1 > fig.bbox.height + 1
            ):
                raise ValueError(f"Clipped label: {label.get_text()}")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"{name}.{ext}", dpi=400, facecolor="white")
    plt.close(fig)
    with Image.open(output_dir / f"{name}.png") as im:
        if not all(abs(d - 400) < 0.1 for d in im.info["dpi"]):
            raise ValueError("PNG resolution does not match the requested 400 dpi")
        return {
            "inches": [im.width / 400, im.height / 400],
            "pixels": list(im.size),
            "dpi": im.info["dpi"],
        }


def main() -> None:
    """Parse explicit input paths and render the validated paper figure.

    Notes
    -----
    Writes the paper asset, vector versions, plotted observations,
    and score-selected winners. No studies or model evaluations are launched.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="External feasible-observation CSV."
    )
    parser.add_argument(
        "--endpoints",
        type=Path,
        required=True,
        help="External endpoint CSV used to verify selected scores.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination for cs3_proposal_comparison figure files.",
    )
    parser.add_argument(
        "--trials-per-group",
        type=int,
        default=200,
        help="Expected observations per board/method pair (default: 200).",
    )
    args = parser.parse_args()
    if args.trials_per_group <= 0:
        parser.error("--trials-per-group must be positive")
    rows, winners = load(args.input, args.endpoints, args.trials_per_group)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dimensions = render(rows, winners, args.output_dir)
    with (args.output_dir / "cs3_proposal_comparison_plotted_points.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "dimensions": dimensions,
        "observations": len(rows),
        "selected": [winners[key] for key in sorted(winners)],
    }
    (args.output_dir / "cs3_proposal_comparison_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
