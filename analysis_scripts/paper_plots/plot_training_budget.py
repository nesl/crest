#!/usr/bin/env python3
# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Reconstruct the training-budget appendix analyses from external checkpoints.

Standalone panel styling was recovered from the local final-paper renderer.
The submitted PNGs were exported with Matplotlib 3.11.2; that version reproduces
both frozen-data panels pixel-for-pixel. Other versions may change text metrics
and constrained layout, even when the plotting instructions are identical.
The validated JSON analysis workflow was reconstructed separately. The external
frozen CSV records the 50-candidate, seed-17 paper panel; CSV provenance reports
best-validation-loss checkpoints and fixed search-time energy, but its schema
cannot independently validate those policies. No datasets are bundled.

Examples
--------
python plot_training_budget.py --input-csv /path/panel.csv --output-dir /path/figures
python plot_training_budget.py --results /path/results.json --output-dir /path/figures

CSV columns: candidate_id,budget,rmse_total,energy_mj,source.
JSON schema is documented in ``load``. Outputs use the paper asset basenames
cs2_training_budget_ranking and cs2_training_budget_fronts (PNG/PDF/SVG), plus
training_budget_summary.json with explicit input and provenance metadata.
Both panels use continued-training (budget_only) checkpoints and fixed historical
energy. No training, hardware measurements, or checkpoint selection is performed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, ScalarFormatter
from scipy.stats import spearmanr


def load(
    paths: list[Path], budgets: list[int], seed: int
) -> dict[int, dict[str, tuple[float, float]]]:
    """Validate a matched candidate panel from saved checkpoint evaluations.

    Parameters
    ----------
    paths : list of pathlib.Path
        External results.json files, each containing a list of checkpoint rows.
        Required fields are candidate_id, seed, budget, policy, status,
        energy_policy, energy_mj_per_inference, and metrics.rmse_total.
    budgets : list of int
        Required continued-training checkpoint budgets for every candidate.
    seed : int
        Seed to select; other seeds and original_policy rows are ignored.

    Returns
    -------
    dict
        Budget to candidate ID to (validation RMSE, historical energy) mapping.

    Raises
    ------
    ValueError
        If requested rows are missing, duplicated, failed, nonfinite, or use
        changing energy measurements across budgets.
    """
    points = {b: {} for b in budgets}
    candidates = set()
    for path in paths:
        rows = json.loads(path.read_text())
        if not isinstance(rows, list):
            raise ValueError("Each results file must contain a JSON list")
        for row in rows:
            if row["seed"] != seed or row["policy"] != "budget_only":
                continue
            candidate = row["candidate_id"]
            candidates.add(candidate)
            if row["budget"] not in points:
                continue
            budget = row["budget"]
            error = row["metrics"]["rmse_total"]
            energy = row["energy_mj_per_inference"]
            if (
                row["status"] != "ok"
                or row["energy_policy"] != "fixed_historical_measurement"
                or any(
                    isinstance(v, bool)
                    or not isinstance(v, (int, float))
                    or not math.isfinite(v)
                    for v in (error, energy)
                )
                or error < 0
                or energy <= 0
            ):
                raise ValueError("Invalid checkpoint evaluation or energy policy")
            if candidate in points[budget]:
                raise ValueError("Duplicated candidate/seed/budget/policy row")
            points[budget][candidate] = (error, energy)
    if len(candidates) < 2 or any(set(p) != candidates for p in points.values()):
        raise ValueError(
            "Every candidate needs every requested budget; at least two required"
        )
    for candidate in candidates:
        if len({points[b][candidate][1] for b in budgets}) != 1:
            raise ValueError("Historical energy must remain fixed across budgets")
    return points


def front(points: dict[str, tuple[float, float]]) -> list[str]:
    """Return the nondominated front, minimizing error and energy.

    Parameters
    ----------
    points : dict
        Candidate IDs mapped to (validation RMSE, historical energy).

    Returns
    -------
    list
        Front candidate IDs ordered by increasing energy, error, then ID.
        Equal objective points are retained.
    """
    return sorted(
        (
            key
            for key, value in points.items()
            if not any(
                other != key
                and all(a <= b for a, b in zip(values, value))
                and any(a < b for a, b in zip(values, value))
                for other, values in points.items()
            )
        ),
        key=lambda key: (points[key][1], points[key][0], key),
    )


def load_csv(
    path: Path, budgets: list[int]
) -> dict[int, dict[str, tuple[float, float]]]:
    """Read the externally supplied frozen paper panel.

    Parameters
    ----------
    path : pathlib.Path
        CSV with candidate_id, budget, rmse_total, energy_mj, and source columns.
        The source column records original experiment result paths; it does not
        independently verify seed, checkpoint selection, or energy policy.
    budgets : list of int
        Required budgets, matched across all candidates.

    Returns
    -------
    dict
        Budget/candidate mapping to (RMSE, historical energy).

    Raises
    ------
    ValueError
        If rows are duplicated, nonfinite, incomplete, or have changing energy.
    """
    points = {budget: {} for budget in budgets}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"candidate_id", "budget", "rmse_total", "energy_mj", "source"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV requires columns: {sorted(required)}")
        for row in reader:
            budget = int(row["budget"])
            if budget not in points:
                continue
            candidate = row["candidate_id"]
            error, energy = float(row["rmse_total"]), float(row["energy_mj"])
            if (
                not candidate
                or not row["source"]
                or candidate in points[budget]
                or not all(math.isfinite(v) for v in (error, energy))
                or error < 0
                or energy <= 0
            ):
                raise ValueError("Invalid or duplicate frozen checkpoint row")
            points[budget][candidate] = (error, energy)
    ids = set().union(*(set(panel) for panel in points.values()))
    if len(ids) < 2 or any(set(panel) != ids for panel in points.values()):
        raise ValueError(
            "Every candidate needs every requested budget; at least two required"
        )
    if any(len({points[b][cid][1] for b in budgets}) != 1 for cid in ids):
        raise ValueError("Historical energy must remain fixed across budgets")
    return points


def render(
    points: dict[int, dict[str, tuple[float, float]]],
    front_budgets: list[int],
    chosen_budget: int,
    output_dir: Path,
) -> dict:
    """Export separate paper ranking and Pareto panels at 400 dpi.

    Parameters
    ----------
    points : dict
        Matched budgets/candidates containing (validation RMSE, energy).
    front_budgets : list of int
        Budgets displayed in the Pareto panel.
    chosen_budget : int
        Highlighted search budget.
    output_dir : pathlib.Path
        Explicit directory for PNG, PDF, SVG panels and analysis JSON.

    Returns
    -------
    dict
        Spearman coefficients and nondominated candidate memberships.

    Notes
    -----
    Average ranks handle ties; constant error vectors are rejected. Both
    objectives are minimized and equal nondominated points are retained.
    The external CSV is the frozen paper panel, whereas JSON is a validated
    reconstruction workflow. Neither path performs training or HIL measurement.
    """
    budgets = sorted(points)
    reference = max(budgets)
    ids = sorted(points[reference])
    ref = [points[reference][cid][0] for cid in ids]
    rho = []
    for budget in budgets:
        errors = [points[budget][cid][0] for cid in ids]
        if len(set(errors)) < 2 or len(set(ref)) < 2:
            raise ValueError("Spearman correlation requires nonconstant errors")
        rho.append(float(spearmanr(errors, ref).statistic))
    memberships = {str(budget): front(points[budget]) for budget in front_budgets}
    output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    ):
        for name in ("ranking", "fronts"):
            fig, ax = plt.subplots(figsize=(3.5, 3.0), layout="constrained")
            ax.grid(color="0.87", linewidth=0.5)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            if name == "ranking":
                ax.plot(
                    budgets,
                    rho,
                    marker="o",
                    color="#1f77b4",
                    linewidth=1.3,
                    markersize=4,
                )
                selected = rho[budgets.index(chosen_budget)]
                ax.scatter([chosen_budget], [selected], color="#ff7f0e", s=40, zorder=3)
                for budget, value in zip(budgets, rho):
                    if budget != chosen_budget:
                        ax.annotate(
                            f"{value:.3f}",
                            (budget, value),
                            xytext=(0, -14) if budget == reference else (4, 7),
                            textcoords="offset points",
                            ha="center" if budget == reference else "left",
                            fontsize=7,
                        )
                ax.annotate(
                    f"Chosen search budget\n{chosen_budget} epochs ({selected:.3f})",
                    xy=(chosen_budget, selected),
                    xytext=(chosen_budget + 40, 0.59),
                    fontsize=7,
                    color="#cc6600",
                    arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=1),
                )
                ax.set(
                    title="Ranking agreement",
                    xlabel="Training budget (epochs)",
                    ylabel="Spearman correlation",
                    ylim=(min(0.475, min(rho) - 0.05), 1.08),
                )
                ax.set_xticks(
                    sorted(
                        set(
                            [min(budgets), chosen_budget, reference]
                            + [b for b in budgets if b >= 100]
                        )
                    )
                )
            else:
                styles = [
                    ("#999999", "s", "--"),
                    ("#ff7f0e", "o", "-"),
                    ("#1f77b4", "^", "-."),
                    ("#2ca02c", "D", "-"),
                ]
                for i, budget in enumerate(front_budgets):
                    keys = memberships[str(budget)]
                    color, marker, line = styles[i % len(styles)]
                    ax.plot(
                        [points[budget][cid][1] for cid in keys],
                        [points[budget][cid][0] for cid in keys],
                        color=color,
                        marker=marker,
                        linestyle=line,
                        linewidth=1.3,
                        markersize=4,
                        label=f"{budget} epochs",
                    )
                ax.set_xscale("log")
                ax.xaxis.set_major_locator(FixedLocator([4, 10, 25, 50, 125]))
                ax.xaxis.set_major_formatter(ScalarFormatter())
                ax.set(
                    title="Pareto fronts",
                    xlabel="Historical energy (mJ)",
                    ylabel="Validation RMSE total",
                )
                front_values = [
                    points[b][cid][0]
                    for b in front_budgets
                    for cid in memberships[str(b)]
                ]
                ax.set_ylim(
                    min(0.2, min(front_values) - 0.02),
                    max(1.82, max(front_values) + 0.05),
                )
                ax.legend(loc="upper right", frameon=False, fontsize=7)
            for extension in ("png", "pdf", "svg"):
                fig.savefig(
                    output_dir / f"cs2_training_budget_{name}.{extension}",
                    dpi=400,
                    facecolor="white",
                )
            plt.close(fig)
    return {
        "reference_budget": reference,
        "candidate_count": len(ids),
        "spearman": dict(zip(map(str, budgets), rho)),
        "fronts": memberships,
        "limitations": "Selected matched panel, one reported seed, fixed historical energy; reference budget does not demonstrate convergence.",
    }


def main() -> None:
    """Read explicit external result paths and write offline analysis figures.

    Notes
    -----
    Select exactly one input format: frozen panel CSV or checkpoint JSON files.
    Writes both paper panels and a summary recording input/provenance limits.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--results",
        nargs="+",
        type=Path,
        help="Validated external JSON checkpoint lists",
    )
    inputs.add_argument(
        "--input-csv",
        type=Path,
        help="External frozen paper panel; provenance is reported, not revalidated",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--budgets", nargs="+", type=int, default=[15, 30, 55, 100, 200, 300]
    )
    parser.add_argument(
        "--front-budgets", nargs="+", type=int, default=[15, 55, 100, 300]
    )
    parser.add_argument("--chosen-budget", type=int, default=55)
    args = parser.parse_args()
    if (
        len(set(args.budgets)) != len(args.budgets)
        or any(b <= 0 for b in args.budgets)
        or not set(args.front_budgets).issubset(args.budgets)
        or args.chosen_budget not in args.budgets
    ):
        parser.error("Use unique positive budgets containing front and chosen budgets")
    points = (
        load_csv(args.input_csv, args.budgets)
        if args.input_csv
        else load(args.results, args.budgets, args.seed)
    )
    summary = render(points, args.front_budgets, args.chosen_budget, args.output_dir)
    summary["seed"] = (
        args.seed
        if args.results
        else "17 reported by frozen-panel provenance; not encoded in CSV"
    )
    summary["input_format"] = (
        "frozen CSV" if args.input_csv else "validated checkpoint JSON"
    )
    summary["matplotlib_version"] = matplotlib.__version__
    summary["submitted_matplotlib_version"] = "3.11.2"
    summary["renderer"] = (
        "Standalone paper styling recovered from local renderer; JSON workflow reconstructed"
    )
    summary["inputs"] = [
        str(p) for p in ([args.input_csv] if args.input_csv else args.results)
    ]
    summary["policy"] = (
        "CSV reports continued-training, best-validation-loss checkpoints and fixed search-time energy; not independently verified"
        if args.input_csv
        else "budget_only; fixed_historical_measurement"
    )
    (args.output_dir / "training_budget_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
