#!/usr/bin/env python3
# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Render CS1 model transfer error from joined_candidates.csv.

Transfer excludes the 200ms_cadenced pool, which is not held out. Error is
measured scheduled_mj minus model prediction E_pred, in mJ.
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


BASENAME = "cs1_model_transfer"
POPULATION = "Held out candidates; excludes 200ms_cadenced (not held out)"

INPUT_HELP = "Joined candidates CSV with pool, trial, study, period_ms, continuous_mj, scheduled_mj, E_pred."


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
        Source table described by the module documentation.
    output_dir : pathlib.Path
        Directory for PNG, PDF, plotted-points CSV and JSON summary.
    """
    import matplotlib.pyplot as plt

    numeric = ["period_ms", "continuous_mj", "scheduled_mj"]
    numeric.append("E_pred")
    data = load_csv(input_csv, numeric, ["pool", "trial", "study"])
    if data.duplicated(["pool", "trial"]).any():
        raise ValueError("Duplicate (pool, trial) candidates")
    if set(data.study) != {"continuous", "cadenced"}:
        raise ValueError("Expected study codes continuous and cadenced")
    if set(data.period_ms) != {100, 200, 500}:
        raise ValueError("Expected the measured periods 100, 200 and 500 ms")
    data = data.loc[data.pool != "200ms_cadenced"].copy()
    data["error_mj"] = data.scheduled_mj - data.E_pred
    fig, axes = plt.subplots(1, 3, figsize=(3.7, 2.05), sharey=True)
    summary = {"population": POPULATION, "groups": []}
    for ax, period in zip(axes, (100, 200, 500)):
        panel = data.loc[data.period_ms == period]
        if panel.empty:
            raise ValueError(f"No observations at {period} ms")
        for study, color, marker, label in zip(
            ("continuous", "cadenced"),
            ("#1f77b4", "#ff7f0e"),
            ("o", "^"),
            ("Continuous search", "Periodic search"),
        ):
            group = panel.loc[panel.study == study]
            if group.empty:
                if period == 200 and study == "cadenced":
                    continue
                raise ValueError(f"Missing {study} group at {period} ms")
            ax.scatter(
                group.scheduled_mj,
                group.error_mj,
                s=7,
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
                    "mean_error_mj": float(group.error_mj.mean()),
                    "rmse_mj": float(np.sqrt(np.mean(group.error_mj**2))),
                }
            )
        ax.set_title(f"{period} ms", fontsize=9, pad=3)
        ax.grid(color="#c7cdd3", alpha=0.65, linewidth=0.55)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7, length=3, pad=2)
        ax.axhline(0, color=".4", linestyle="--", linewidth=0.7)
        ax.xaxis.set_major_locator(plt.MaxNLocator(2))
    bound = max(16.0, float(data.error_mj.abs().max()) * 1.05)
    axes[0].set_ylim(-bound, bound)
    axes[0].set_ylabel("Error (mJ)")
    fig.supxlabel("Measured energy per release (mJ)", fontsize=8, y=0.16)
    fig.subplots_adjust(left=0.15, right=0.99, bottom=0.36, top=0.88, wspace=0.16)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.56, 0.015),
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
        "font.size": 8,
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
