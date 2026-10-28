#!/usr/bin/env python3
# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Reconstruct the training-budget appendix analyses from external checkpoints.

This renderer was reconstructed from the documented analyses and published
figure; the original final renderer was not recovered. It is not pixel exact.
Both panels use continued-training (budget_only) checkpoints and fixed historical
energy. No training, hardware measurements, or checkpoint selection is performed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter
from scipy.stats import spearmanr


def load(paths, budgets, seed):
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
            raise ValueError('Each results file must contain a JSON list')
        for row in rows:
            if row['seed'] != seed or row['policy'] != 'budget_only':
                continue
            candidate = row['candidate_id']
            candidates.add(candidate)
            if row['budget'] not in points:
                continue
            budget = row['budget']
            error = row['metrics']['rmse_total']
            energy = row['energy_mj_per_inference']
            if (row['status'] != 'ok' or
                    row['energy_policy'] != 'fixed_historical_measurement' or
                    any(isinstance(v, bool) or not isinstance(v, (int, float))
                        or not math.isfinite(v) for v in (error, energy)) or
                    error < 0 or energy <= 0):
                raise ValueError('Invalid checkpoint evaluation or energy policy')
            if candidate in points[budget]:
                raise ValueError('Duplicated candidate/seed/budget/policy row')
            points[budget][candidate] = (error, energy)
    if len(candidates) < 2 or any(set(p) != candidates for p in points.values()):
        raise ValueError('Every candidate needs every requested budget; at least two required')
    for candidate in candidates:
        if len({points[b][candidate][1] for b in budgets}) != 1:
            raise ValueError('Historical energy must remain fixed across budgets')
    return points


def front(points):
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
    return sorted((key for key, value in points.items() if not any(
        other != key and all(a <= b for a, b in zip(values, value))
        and any(a < b for a, b in zip(values, value))
        for other, values in points.items())),
        key=lambda key: (points[key][1], points[key][0], key))


def render(points, front_budgets, chosen_budget, output_dir):
    """Plot rank agreement and continued-training Pareto fronts.

    Parameters
    ----------
    points : dict
        Validated matched checkpoint panel indexed by budget and candidate.
    front_budgets : list of int
        Budgets shown in the Pareto panel.
    chosen_budget : int
        Search budget highlighted in the correlation panel.
    output_dir : pathlib.Path
        Explicit destination for PNG, PDF, SVG, and computed summary JSON.

    Returns
    -------
    dict
        Computed Spearman coefficients, front memberships, and protocol metadata.

    Notes
    -----
    Spearman correlation uses average ranks for ties. Constant error vectors
    are rejected because their rank correlation is undefined. No uncertainty
    estimate or population-level reliability claim is inferred.
    """
    budgets = sorted(points)
    reference = max(budgets)
    ids = sorted(points[reference])
    reference_errors = [points[reference][key][0] for key in ids]
    correlations = []
    for budget in budgets:
        errors = [points[budget][key][0] for key in ids]
        if len(set(errors)) < 2 or len(set(reference_errors)) < 2:
            raise ValueError('Spearman correlation requires nonconstant errors')
        correlations.append(float(spearmanr(errors, reference_errors).statistic))
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10.5,
                         'axes.labelsize': 10.5, 'axes.titlesize': 11,
                         'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5,
                         'axes.linewidth': .7, 'ps.fonttype': 42,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.5))
        ax = axes[0]
        ax.plot(budgets, correlations, '-o', color='#1f77b4', linewidth=1.2, markersize=4)
        selected = correlations[budgets.index(chosen_budget)]
        ax.scatter([chosen_budget], [selected], color='#ff7f0e', s=48, zorder=4)
        for budget, rho in zip(budgets, correlations):
            if budget != chosen_budget:
                ax.annotate(f'{rho:.3f}', (budget, rho), xytext=((0, -17) if budget == reference else (4, 7)),
                            ha=('center' if budget == reference else 'left'),
                            textcoords='offset points', fontsize=8.5)
        ax.annotate(f'Chosen search budget\n{chosen_budget} epochs ({selected:.3f})',
                    (chosen_budget, selected), xytext=(.30, .19),
                    textcoords='axes fraction', color='#c65f00', fontsize=9,
                    arrowprops={'arrowstyle': '->', 'color': '#ff7f0e', 'lw': .9})
        ax.set(title='(a) Ranking agreement', xlabel='Training budget (epochs)',
               ylabel='Spearman correlation')
        ax.set_xticks(sorted(set([min(budgets), chosen_budget, reference] +
                                [b for b in budgets if b >= 100])))
        ax.margins(y=.18)
        palette = ['#999999', '#ff7f0e', '#1f77b4', '#2ca02c']
        markers = ['s', 'o', '^', 'D']
        styles = ['--', '-', '-.', '-']
        memberships = {}
        for index, budget in enumerate(front_budgets):
            keys = front(points[budget])
            memberships[str(budget)] = keys
            axes[1].plot([points[budget][k][1] for k in keys],
                         [points[budget][k][0] for k in keys],
                         color=palette[index % len(palette)],
                         marker=markers[index % len(markers)],
                         linestyle=styles[index % len(styles)],
                         label=f'{budget} epochs', linewidth=1.2, markersize=3.5)
        axes[1].set(xscale='log', title='(b) Pareto fronts',
                    xlabel='Historical energy (mJ)', ylabel='Validation RMSE total')
        energy_min = min(value[1] for panel in points.values() for value in panel.values())
        energy_max = max(value[1] for panel in points.values() for value in panel.values())
        energy_ticks = [tick for tick in (1, 2, 4, 10, 25, 50, 125, 250, 500)
                        if energy_min * .95 <= tick <= energy_max * 1.05]
        axes[1].set_xticks(energy_ticks)
        axes[1].xaxis.set_major_formatter(ScalarFormatter())
        axes[1].xaxis.set_minor_formatter(NullFormatter())
        axes[1].legend(frameon=False, fontsize=8.5, handlelength=2,
                       borderaxespad=.35, labelspacing=.35)
        for ax in axes:
            ax.grid(color='#dddddd', linewidth=.4)
            ax.tick_params(length=3, pad=2)
            ax.set_axisbelow(True)
        fig.subplots_adjust(left=.085, right=.99, bottom=.17, top=.90, wspace=.32)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for label in fig.findobj(match=lambda item: isinstance(item, plt.Text)):
            if label.get_visible() and label.get_text():
                box = label.get_window_extent(renderer)
                if (box.x0 < -1 or box.y0 < -1 or
                        box.x1 > fig.bbox.width + 1 or box.y1 > fig.bbox.height + 1):
                    raise ValueError(f'Clipped figure label {label.get_text()}')
        output_dir.mkdir(parents=True, exist_ok=True)
        for extension in ('png', 'pdf', 'svg'):
            fig.savefig(output_dir / f'training_budget.{extension}', dpi=400,
                        facecolor='white')
        plt.close(fig)
    summary = {'renderer': 'reconstructed; not original or pixel exact',
               'policy': 'budget_only', 'reference_budget': reference,
               'candidate_count': len(ids), 'spearman': dict(zip(map(str, budgets), correlations)),
               'fronts': memberships,
               'limitations': 'Selected matched panel, one seed, fixed historical energy; reference budget does not demonstrate convergence.'}
    (output_dir / 'training_budget_summary.json').write_text(
        json.dumps(summary, indent=2, allow_nan=False) + '\n')
    return summary


def main():
    """Read explicit external result paths and write offline analysis figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', nargs='+', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--budgets', nargs='+', type=int, default=[15, 30, 55, 100, 200, 300])
    parser.add_argument('--front-budgets', nargs='+', type=int, default=[15, 55, 100, 300])
    parser.add_argument('--chosen-budget', type=int, default=55)
    args = parser.parse_args()
    if (len(set(args.budgets)) != len(args.budgets) or any(b <= 0 for b in args.budgets)
            or not set(args.front_budgets).issubset(args.budgets)
            or args.chosen_budget not in args.budgets):
        parser.error('Use unique positive budgets containing front and chosen budgets')
    points = load(args.results, args.budgets, args.seed)
    summary = render(points, args.front_budgets, args.chosen_budget, args.output_dir)
    summary['seed'] = args.seed
    (args.output_dir / 'training_budget_summary.json').write_text(
        json.dumps(summary, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
