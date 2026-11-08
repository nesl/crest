#!/usr/bin/env python3
# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Render CS1 clock reversal from policy_per_round.csv.

Compare trials 203 and 137 at 300 MHz against trial 203 at 800 MHz. Match
observations explicitly by round across both measured periods (100 and 500 ms).
Lines interpolate paired round differences only within this measured interval;
endpoint error bars are two-sided 95% Student-t confidence intervals. The CSV
sidecar includes paired measurements and interpolated pointwise intervals, which
are not additional measurements. Crossings are reported only inside the interval.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


INPUT_HELP = "Policy rounds CSV: candidate, clock_mhz, period_ms, round, E_stop_mj, E_inf_mj, C_s."
BASENAME = "cs1_clock_reversal"


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


def _clock_difference_records(data: pd.DataFrame) -> list[dict]:
    """Summarize paired 300-minus-800 MHz energy differences.

    Parameters
    ----------
    data : pandas.DataFrame
        Validated policy measurements with paired round identifiers.

    Returns
    -------
    list of dict
        Ordered inference and release differences with 95% Student-t intervals.

    Raises
    ------
    ValueError
        If fewer than two complete clock pairs are available.
    """
    from scipy.stats import t

    records = []
    # Continuous inference and periodic release differences quoted in the text.
    fixed = data.loc[data.candidate == "continuous_203"]
    for period, column, label in (
        (500, "E_inf_mj", "continuous_inference"),
        (100, "E_stop_mj", "100ms_release"),
        (500, "E_stop_mj", "500ms_release"),
    ):
        paired = fixed.loc[fixed.period_ms == period].pivot(
            index="round", columns="clock_mhz", values=column
        )
        if (
            not {300, 800}.issubset(paired.columns)
            or paired[[300, 800]].isna().any().any()
        ):
            raise ValueError(f"Incomplete clock pairing for {label}")
        delta = paired[300] - paired[800]
        if len(delta) < 2:
            raise ValueError(f"At least two clock pairs needed for {label}")
        mean = float(delta.mean())
        half = float(
            t.ppf(0.975, len(delta) - 1) * delta.std(ddof=1) / np.sqrt(len(delta))
        )
        records.append(
            {
                "quantity": label,
                "n": len(delta),
                "mean_300_minus_800_mj": mean,
                "ci95_low_mj": mean - half,
                "ci95_high_mj": mean + half,
            }
        )
    return records


def _paired_round_analysis(data: pd.DataFrame) -> tuple[list, list, dict]:
    """Calculate paired curves, audit records and clock summaries.

    Parameters
    ----------
    data : pandas.DataFrame
        Validated policy rounds with candidate, clock_mhz, period_ms, round,
        E_stop_mj, E_inf_mj and C_s columns.

    Returns
    -------
    tuple of list, list, dict
        Drawing curves/endpoints, ordered plotted-point records and summary.

    Raises
    ------
    ValueError
        If reference endpoints or comparison/clock round pairs are incomplete.
    """
    from scipy.stats import t

    reference = data.loc[(data.candidate == "continuous_203") & (data.clock_mhz == 800)]
    ref = (
        reference.loc[reference.period_ms.isin([100, 500])]
        .pivot(index="round", columns="period_ms", values="E_stop_mj")
        .sort_index()
    )
    if set(ref.columns) != {100, 500} or len(ref) < 2 or ref.isna().any().any():
        raise ValueError(
            "Reference needs at least two complete paired rounds at 100 and 500 ms"
        )
    curves = []
    records: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "reference": "continuous_203 at 800 MHz",
        "confidence_level": 0.95,
        "interpolation_interval_ms": [100, 500],
        "comparisons": [],
        "clock_differences": [],
    }
    n = len(ref)
    tcrit = float(t.ppf(0.975, n - 1))
    grid = np.linspace(100, 500, 400)
    for candidate, color in zip(
        ("continuous_203", "continuous_137"), ("#1f77b4", "#ff7f0e")
    ):
        alternative = data.loc[(data.candidate == candidate) & (data.clock_mhz == 300)]
        paired = (
            alternative.loc[alternative.period_ms.isin([100, 500])]
            .pivot(index="round", columns="period_ms", values="E_stop_mj")
            .sort_index()
        )
        if (
            set(paired.columns) != {100, 500}
            or not paired.index.equals(ref.index)
            or paired.isna().any().any()
        ):
            raise ValueError(
                f"{candidate}: round IDs must match reference across both periods"
            )
        differences = paired - ref
        slopes = (differences[500].to_numpy() - differences[100].to_numpy()) / 400
        intercepts = differences[100].to_numpy() - slopes * 100
        lines = intercepts[:, None] + slopes[:, None] * grid[None, :]
        means = lines.mean(axis=0)
        half = tcrit * lines.std(axis=0, ddof=1) / np.sqrt(n)
        endpoints = []
        for period in (100, 500):
            values = differences[period]
            ci = tcrit * float(values.std(ddof=1)) / np.sqrt(n)
            endpoints.append((period, values.mean(), ci))
            for round_id, value in values.items():
                records.append(
                    {
                        "kind": "measured_paired_difference",
                        "candidate": candidate,
                        "round": round_id,
                        "period_ms": period,
                        "difference_mj": value,
                    }
                )
        for period, mean, interval in zip(grid, means, half):
            records.append(
                {
                    "kind": "interpolated_mean",
                    "candidate": candidate,
                    "period_ms": period,
                    "difference_mj": mean,
                    "ci95_low_mj": mean - interval,
                    "ci95_high_mj": mean + interval,
                }
            )
        curves.append((candidate, color, grid, means, endpoints))
        slope = float(slopes.mean())
        crossing = None if slope == 0 else float(-intercepts.mean() / slope)
        summary["comparisons"].append(
            {
                "candidate": candidate,
                "clock_mhz": 300,
                "n_paired_rounds": n,
                "slope_mw": slope * 1000,
                "crossing_within_measured_interval_ms": (
                    crossing
                    if crossing is not None and 100 <= crossing <= 500
                    else None
                ),
                "mean_inference_ms": float(alternative.C_s.mean() * 1000),
            }
        )
    summary["clock_differences"] = _clock_difference_records(data)
    return curves, records, summary


def render(input_csv: Path, output_dir: Path) -> None:
    """Render a figure and write the exact plotted values and summary.

    Parameters
    ----------
    input_csv : pathlib.Path
        CSV with candidate, clock_mhz, period_ms, round, E_stop_mj,
        E_inf_mj and C_s columns.
    output_dir : pathlib.Path
        Directory for PNG, PDF, plotted-points CSV and JSON summary.
    """
    import matplotlib.pyplot as plt

    data = load_csv(
        input_csv,
        ["clock_mhz", "period_ms", "E_stop_mj", "E_inf_mj", "C_s"],
        ["candidate", "clock_mhz", "period_ms", "round"],
    )
    curves, records, summary = _paired_round_analysis(data)
    # Preserve the combined-paper grid geometry and hidden lower-row extent:
    # these axes determine the original export size even in this standalone panel.
    fig, panels = plt.subplots(2, 3, figsize=(3.33, 4.1))
    fig.subplots_adjust(left=0.16, right=0.98, wspace=0.12)
    for row, bottom, height in zip(panels, [0.61, 0.151], [0.354, 0.305]):
        for panel in row:
            pos = panel.get_position()
            panel.set_position([pos.x0, bottom, pos.width, height])
    left = panels[0, 0].get_position().x0
    right = panels[0, 2].get_position().x1
    for panel in panels[0]:
        panel.remove()
    for panel in panels[1]:
        panel.set_visible(False)
    ax = fig.add_axes([left, 0.61, right - left, 0.354])
    ax.tick_params(length=3, pad=2)
    ax.grid(color="#c7cdd3", alpha=0.65, linewidth=0.55)
    ax.set_axisbelow(True)
    ax.axhline(0, color=".4", linestyle="--", linewidth=0.7)
    for candidate, color, grid, means, endpoints in curves:
        label = f"Trial {candidate.rsplit('_', 1)[1]}, 300 MHz"
        ax.plot(grid, means, color=color, linewidth=1.4, label=label)
        for period, mean, ci in endpoints:
            ax.errorbar(
                period,
                mean,
                yerr=ci,
                fmt="o",
                color=color,
                markersize=4.2,
                capsize=2.2,
                elinewidth=0.9,
                zorder=3,
                markeredgecolor="white",
                markeredgewidth=0.5,
            )
    ax.set_xlabel("Release period (ms)", labelpad=1)
    ax.set_ylabel("Difference (mJ)", labelpad=2)
    ax.set_xlim(0, 600)
    ax.set_ylim(-4, 3)
    ax.set_yticks([-4, -2, 0, 2])
    ax.legend(
        loc="lower left",
        frameon=True,
        framealpha=1,
        facecolor="white",
        edgecolor="none",
        fontsize=7.5,
        handlelength=1.6,
        borderaxespad=0.2,
        borderpad=0.25,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(
            output_dir / f"{BASENAME}.{extension}",
            dpi=400,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.02,
        )
    plt.close(fig)
    pd.DataFrame(records).to_csv(
        output_dir / f"{BASENAME}_plotted_points.csv", index=False
    )
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
        "font.size": 8.75,
        "axes.titlesize": 9.4,
        "axes.labelsize": 8.75,
        "xtick.labelsize": 8.1,
        "ytick.labelsize": 8.1,
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
