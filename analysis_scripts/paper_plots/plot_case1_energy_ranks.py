#!/usr/bin/env python3
# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Render CS1 energy ranks from joined_candidates.csv.

Ranks use all pools, independently within each (period_ms, study), with average
tied ranks and smaller energy ranked first.
Requires finite energy values and unique (pool, trial) identifiers. Search labels
retain source codes continuous/cadenced in CSV sidecars; cadenced is displayed
as Periodic search. No candidate filtering beyond the stated population occurs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


BASENAME = "cs1_energy_ranks"
POPULATION = "All joined candidates, ranks within period and study"

INPUT_HELP = "Joined candidates CSV with pool, trial, study, period_ms, continuous_mj, scheduled_mj."


def load_csv(path: Path, numeric: Sequence[str], keys: Sequence[str]) -> pd.DataFrame:
    """Read and validate the scientific input table.

    Parameters
    ----------
    path : pathlib.Path
        Input CSV file.
    numeric : Sequence[str]
        Required finite numeric columns.
    keys : Sequence[str]
        Nonmissing columns that jointly identify each observation.

    Returns
    -------
    pandas.DataFrame
        Validated observations; numeric fields converted to numbers.

    Raises
    ------
    ValueError
        If required columns are missing, the table is empty, identifiers are
        missing or duplicated, or numeric fields are invalid or nonfinite.
    OSError
        If the CSV cannot be read.
    """
    data = pd.read_csv(path)
    required = set(numeric) | set(keys)
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if data.empty or data[list(keys)].isna().any().any():
        raise ValueError("Input is empty or contains missing observation identifiers")
    if data.duplicated(list(keys)).any():
        raise ValueError(f"Duplicate observation identifiers: {list(keys)}")
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="raise")
        if not np.isfinite(data[column]).all():
            raise ValueError(f"Nonfinite values in {column}")
    return data


def render(input_csv: Path, output_dir: Path) -> None:
    """Render a figure and write the exact plotted values and summary.

    Parameters
    ----------
    input_csv : pathlib.Path
        CSV with pool, trial, study, period_ms, continuous_mj and
        scheduled_mj columns.
    output_dir : pathlib.Path
        Directory for PNG, PDF, plotted-points CSV and JSON summary.
    """
    import matplotlib.pyplot as plt

    numeric = ["period_ms", "continuous_mj", "scheduled_mj"]
    data = load_csv(input_csv, numeric, ["pool", "trial", "study"])
    if data.duplicated(["pool", "trial"]).any():
        raise ValueError("Duplicate (pool, trial) candidates")
    if set(data.study) != {"continuous", "cadenced"}:
        raise ValueError("Expected study codes continuous and cadenced")
    if set(data.period_ms) != {100, 200, 500}:
        raise ValueError("Expected the measured periods 100, 200 and 500 ms")
    data["continuous_rank"] = data.groupby(["period_ms", "study"]).continuous_mj.rank(
        method="average"
    )
    data["periodic_rank"] = data.groupby(["period_ms", "study"]).scheduled_mj.rank(
        method="average"
    )
    fig, axes = plt.subplots(3, 1, figsize=(1243 / 400, 1371 / 400))
    summary = {"population": POPULATION, "groups": []}
    for ax, period in zip(axes, (100, 200, 500)):
        panel = data.loc[data.period_ms == period]
        if panel.empty:
            raise ValueError(f"No observations at {period} ms")
        maximum = 1
        for study, color, marker, label in zip(
            ("continuous", "cadenced"),
            ("#009e73", "#9467bd"),
            ("o", "^"),
            ("Continuous search", "Periodic search"),
        ):
            group = panel.loc[panel.study == study]
            if group.empty:
                raise ValueError(f"Missing {study} group at {period} ms")
            maximum = max(maximum, len(group))
            ax.scatter(
                group.continuous_rank,
                group.periodic_rank,
                s=6,
                alpha=0.65,
                color=color,
                marker=marker,
                edgecolors="none",
                label=label,
            )
            summary["groups"].append(
                {
                    "period_ms": period,
                    "study": study,
                    "n": len(group),
                    "spearman_rho": (
                        float(group.continuous_rank.corr(group.periodic_rank))
                        if group.continuous_rank.nunique() > 1
                        and group.periodic_rank.nunique() > 1
                        else None
                    ),
                }
            )
        ax.set_title(f"{period} ms", pad=3, loc="center")
        ax.grid(color="#c7cdd3", alpha=0.65, linewidth=0.55)
        ax.set_axisbelow(True)
        ax.tick_params(length=3, pad=2)
        ax.plot([1, maximum], [1, maximum], color=".4", linestyle="--", linewidth=0.7)
        ax.set_xlim(-3, maximum + 4)
        ax.set_ylim(-3, maximum + 4)
        ticks = sorted(set([1] + [x for x in (50, 100) if x < maximum] + [maximum]))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
    axes[1].set_ylabel("Periodic-energy rank", labelpad=3)
    axes[-1].set_xlabel("Continuous-energy rank", labelpad=3)
    fig.subplots_adjust(left=0.155, right=0.98, bottom=0.10, top=0.90, hspace=0.55)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.56, 1.0),
        markerscale=1.25,
        ncol=2,
        frameon=False,
        fontsize=7,
        handletextpad=0.2,
        columnspacing=0.8,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"{BASENAME}.{extension}", dpi=400, facecolor="white")
    plt.close(fig)
    data.to_csv(output_dir / f"{BASENAME}_plotted_points.csv", index=False)
    (output_dir / f"{BASENAME}_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Parse inputs and render the paper figure with audit sidecars.

    Parameters
    ----------
    argv : Sequence[str] or None, optional
        Command line arguments, excluding the executable.

    Returns
    -------
    int
        Zero on successful completion.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True, help=INPUT_HELP)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {
        "font.family": "DejaVu Sans",
        "font.size": 7,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#818b94",
        "axes.linewidth": 0.7,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
    with plt.rc_context(style):
        render(args.input_csv, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
