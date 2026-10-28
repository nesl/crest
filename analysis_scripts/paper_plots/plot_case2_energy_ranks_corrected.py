#!/usr/bin/env python3
# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Render the corrected first-256-attempt CS2 rank figure.

Supply an external TSV with period_ms, study, trial,
continuous_mj_per_inference, and cadenced_mj_per_release columns.
Input identity, trial-budget, group, and finite-energy integrity are checked.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "crest-cs2-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASENAME = "case2_energy_ranks"
COLORS = {"continuous": "#1f77b4", "cadenced": "#ff7f0e"}

def load(path, attempt_budget=256):
    """Read paired energy observations and validate candidate integrity.

    Parameters
    ----------
    path : pathlib.Path
        External tab-separated candidate table.
    attempt_budget : int, optional
        Exclusive upper bound on original trial identifiers.

    Returns
    -------
    rows : list of dict
        Parsed paired energy observations in source order.
    digest : str
        SHA-256 checksum of the input bytes.
    """
    raw = path.read_bytes()
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"period_ms", "study", "trial", "continuous_mj_per_inference", "cadenced_mj_per_release"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Incomplete input header")
        rows = []
        for source in reader:
            row = {
                "period_ms": int(source["period_ms"]), "study": source["study"],
                "trial": int(source["trial"]),
                "continuous": float(source["continuous_mj_per_inference"]),
                "cadenced": float(source["cadenced_mj_per_release"]),
            }
            if (row["period_ms"] not in (100, 200, 500) or row["study"] not in COLORS or
                not 0 <= row["trial"] < attempt_budget or
                any(not math.isfinite(row[key]) or row[key] <= 0 for key in ("continuous", "cadenced"))):
                raise ValueError(f"Invalid paired trial {row}")
            rows.append(row)
    if not rows:
        raise ValueError("Expected a nonempty paired candidate table")
    identities = [(r["period_ms"], r["study"], r["trial"]) for r in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate trial within pool and period")
    if {(r["period_ms"], r["study"]) for r in rows} != {(p, s) for p in (100, 200, 500) for s in COLORS}:
        raise ValueError("All six pool-period groups are required")
    return rows, hashlib.sha256(raw).hexdigest()

def ranks(pool, key):
    """Compute ascending ranks, assigning average ranks to exact ties.

    Parameters
    ----------
    pool : list of dict
        Candidate records with unique trial identifiers.
    key : str
        Energy field by which to rank candidates.

    Returns
    -------
    dict
        Trial identifier mapped to its one-based average rank.
    """
    ordered = sorted(pool, key=lambda row: (row[key], row["trial"]))
    result = {}
    left = 0
    while left < len(ordered):
        right = left + 1
        while right < len(ordered) and ordered[right][key] == ordered[left][key]:
            right += 1
        for row in ordered[left:right]:
            result[row["trial"]] = (left + 1 + right) / 2
        left = right
    return result

def pair_stats(pool):
    """Count pairwise rank reversals, excluding exact energy ties.

    Parameters
    ----------
    pool : list of dict
        Candidate records from one search pool and period.

    Returns
    -------
    dict
        Population, reversal, comparable and tied counts, plus reversal percent.
        The percent is None when no pair is comparable.
    """
    inverted = comparable = tied = 0
    for index, first in enumerate(pool):
        for second in pool[index + 1:]:
            product = ((first["continuous"] - second["continuous"]) *
                       (first["cadenced"] - second["cadenced"]))
            if product == 0:
                tied += 1
            else:
                comparable += 1
                inverted += product < 0
    return {"n": len(pool), "reversed_pairs": inverted, "comparable_pairs": comparable,
            "tied_pairs_excluded": tied, "percent_reversed": 100 * inverted / comparable if comparable else None}

def main():
    """Parse external paths, compute ranks, and export publication figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--basename", default=BASENAME)
    parser.add_argument("--attempt-budget", type=int, default=256)
    args = parser.parse_args()
    if args.attempt_budget < 1:
        parser.error("--attempt-budget must be positive")
    rows, input_hash = load(args.input, args.attempt_budget)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats, plotted = {}, []
    style = {"font.family": "DejaVu Sans", "font.size": 10.5,
             "axes.labelsize": 10.5, "axes.titlesize": 11,
             "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
             "axes.linewidth": .7, "svg.fonttype": "none"}
    with plt.rc_context(style):
        fig, axes = plt.subplots(3, 1, figsize=(3.35, 4.25))
        for axis, period in zip(axes, (100, 200, 500)):
            maximum = max(sum(r["period_ms"] == period and r["study"] == study for r in rows) for study in COLORS)
            axis.plot([1, maximum], [1, maximum], color="#707070", ls="--", lw=.9, zorder=1)
            for study, color in COLORS.items():
                pool = [row for row in rows if row["period_ms"] == period and row["study"] == study]
                x, y = ranks(pool, "continuous"), ranks(pool, "cadenced")
                axis.scatter([x[row["trial"]] for row in pool], [y[row["trial"]] for row in pool],
                             s=10.5, c=color, alpha=.62, edgecolors="none", zorder=2)
                result = pair_stats(pool)
                stats[f"{period}/{study}"] = result
                plotted.extend({**row, "continuous_rank": x[row["trial"]],
                                "cadenced_rank": y[row["trial"]]} for row in pool)
            final_tick = max(1, round(maximum / 10) * 10)
            ticks = sorted({1, final_tick, *range(50, min(final_tick, 150), 50)})
            axis.set(xlim=(-3, maximum + 5), ylim=(-3, maximum + 5),
                     xticks=ticks, yticks=ticks)
            axis.set_title(f"{period} ms window", loc="center", pad=2)
            axis.grid(color="#d0d0d0", alpha=.6, lw=.55)
            axis.set_axisbelow(True)
            axis.tick_params(length=3, pad=2)
        axes[-1].set_xlabel("Continuous-energy rank", labelpad=2)
        fig.text(.035, .50, "Cadenced-energy rank", rotation=90,
                 va="center", ha="center", fontsize=10.5)
        handles = [Line2D([], [], ls="", marker="o", color=color, markersize=4.5,
                          label=study.capitalize() + " search") for study, color in COLORS.items()]
        fig.legend(handles=handles, ncol=2, loc="upper center", bbox_to_anchor=(.56, 1.005),
                   frameon=False, fontsize=9, columnspacing=.85, handletextpad=.3)
        fig.subplots_adjust(left=.205, right=.985, top=.905, bottom=.105, hspace=.50)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for label in fig.findobj(match=lambda item: isinstance(item, plt.Text)):
            if label.get_visible() and label.get_text():
                box = label.get_window_extent(renderer)
                if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width + 1 or box.y1 > fig.bbox.height + 1:
                    raise ValueError(f"Clipped figure label {label.get_text()}")
        for extension in ("png", "pdf", "svg"):
            fig.savefig(args.output_dir / f"{args.basename}.{extension}", dpi=400, facecolor="white")
        fig.savefig(args.output_dir / "column_size_preview.png", dpi=110, facecolor="white")
        plt.close(fig)
    with (args.output_dir / "ranked_candidates.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(plotted[0]))
        writer.writeheader()
        writer.writerows(plotted)
    metadata = {
        "input": str(args.input.resolve()), "input_sha256": input_hash,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "canvas_inches": [3.35, 4.25], "png_dpi": 400, "statistics": stats,
        "attempt_budget": args.attempt_budget,
        "method": "Within-period, within-search ranks. Exact ties get average ranks. Pair reversals exclude ties in either energy measure. Original trial attempts bounded by the supplied attempt budget.",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
