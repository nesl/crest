# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Fixed-reference hypervolume reporting using Optuna's Matplotlib implementation."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from .progress_artifacts import atomic_output, as_datetime, eligible_trial


def _reference(output, study_name, directions, names, eligible, configured):
    """Resolve and persist the reference once; later observations cannot move it."""
    path = output / 'hypervolume_reference.json'
    saved = json.loads(path.read_text()) if path.exists() else {}
    signature = dict(study_name=study_name, directions=directions)
    if saved and any(saved.get(key) != value for key, value in signature.items()):
        raise ValueError('Hypervolume metadata belongs to another study or objective directions.')
    reference = saved.get('reference_point')
    source = saved.get('source')
    if configured is not None:
        proposed = [float(value) for value in configured]
        if len(proposed) != len(directions) or not all(map(math.isfinite, proposed)):
            raise ValueError('Hypervolume reference must have one finite value per objective.')
        if reference is not None and reference != proposed:
            raise ValueError('Cannot change the persisted hypervolume reference on resume.')
        reference, source = proposed, 'configured'
    elif reference is None and eligible:
        first = min(eligible, key=lambda trial: trial['number'])
        reference = [value + (1 if direction == 'MINIMIZE' else -1) * max(abs(value), 1.0) * .1
                     for value, direction in zip(first['scores'], directions)]
        source = 'first_eligible_trial_plus_10_percent_margin'
    metadata = dict(signature, objective_names=names, reference_point=reference, source=source,
                    note=('Fixed reference in original objective units. Points outside the reference '
                          'contribute zero. Auto-reference is monitoring only, not comparable between studies.'))
    with atomic_output(path, 'w') as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return reference, source


def write_hypervolume(trials, directory, study_name, *, directions, objective_names,
                      feasibility_enabled, reference_point):
    """Plot the accumulated feasible front without modifying the source study.

    Optuna computes the curve and handles objective directions. A temporary
    in-memory study keeps original trial numbers, but excludes noneligible values.
    We export the plotted series to CSV and a time view, extending it over failed
    or rejected attempts so those attempts remain visible in the search budget.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import optuna
    from optuna.trial import TrialState, create_trial
    from optuna.visualization.matplotlib import plot_hypervolume_history

    starts = [as_datetime(t['start']) for t in trials if t['start']]
    if not starts:
        return []
    origin = min(starts)
    terminal = sorted((t for t in trials if t['state'] in {'COMPLETE', 'PRUNED', 'FAIL'} and t['end']),
                      key=lambda t: t['number'])
    for trial in trials:
        values = trial.get('scores')
        if values is not None and len(values) != len(directions):
            raise ValueError('Trial values must match objective directions.')
    eligible = [t for t in terminal if eligible_trial(t, feasibility_enabled)
                and t.get('scores') is not None
                and all(v is not None and math.isfinite(v) for v in t['scores'])]
    output = Path(directory) / 'progress'
    output.mkdir(parents=True, exist_ok=True)
    reference, source = _reference(output, study_name, directions, objective_names, eligible, reference_point)

    before = set(plt.get_fignums())
    try:
        volumes = {}
        if reference is not None:
            # Only this detached view is altered. Failed placeholders preserve
            # trial indices even when the original trial is queued or rejected.
            view = optuna.create_study(directions=[d.lower() for d in directions])
            by_number = {t['number']: t for t in eligible}
            for number in range(max((t['number'] for t in trials), default=-1) + 1):
                trial = by_number.get(number)
                view.add_trial(create_trial(state=TrialState.COMPLETE, values=trial['scores'])
                               if trial is not None else create_trial(state=TrialState.FAIL))
            with plt.rc_context():
                ax = plot_hypervolume_history(view, reference)
            # Read the public Matplotlib series rather than depending on Optuna's
            # private hypervolume helpers or implementing a second algorithm.
            if ax.lines:
                numbers, values = ax.lines[0].get_data()
                volumes = dict(zip(map(int, numbers), map(float, values)))
        else:
            _, ax = plt.subplots()
        # Trial order and completion order coincide in CREST's serial runner.
        # Offline databases can contain parallel trials, so a time plot must not
        # credit a result before it completed. Reuse Optuna for that order too.
        chronological = sorted(terminal, key=lambda t: (as_datetime(t['end']), t['number']))
        time_volumes = volumes
        if reference is not None and chronological != terminal:
            time_view = optuna.create_study(directions=[d.lower() for d in directions])
            eligible_numbers = {t['number'] for t in eligible}
            for trial in chronological:
                time_view.add_trial(create_trial(state=TrialState.COMPLETE, values=trial['scores'])
                                    if trial['number'] in eligible_numbers else create_trial(state=TrialState.FAIL))
            with plt.rc_context():
                time_ax = plot_hypervolume_history(time_view, reference)
            time_volumes = {}
            if time_ax.lines:
                positions, values = time_ax.lines[0].get_data()
                time_volumes = {chronological[int(position)]['number']: float(value)
                                for position, value in zip(positions, values)}
        time_history, current = {}, 0.0
        for trial in chronological:
            current = time_volumes.get(trial['number'], current)
            time_history[trial['number']] = current
        rows, current = [], 0.0
        for trial in terminal:
            current = volumes.get(trial['number'], current)
            rows.append(dict(trial_number=trial['number'], attempt=trial['number'] + 1,
                             elapsed_hours=(as_datetime(trial['end']) - origin).total_seconds() / 3600,
                             state=trial['state'], hypervolume=current,
                             hypervolume_by_time=time_history[trial['number']]))
        with atomic_output(output / 'hypervolume_progress.csv', 'w') as stream:
            writer = csv.DictWriter(stream, fieldnames=['trial_number', 'attempt', 'elapsed_hours', 'state', 'hypervolume', 'hypervolume_by_time'])
            writer.writeheader()
            writer.writerows(rows)
        # Include noncontributing attempts in both plots, preserving flat sections.
        for key, xlabel, stem in (
            ('attempt', 'Trial attempt (1-based; all finished attempts)', 'hypervolume_history'),
            ('elapsed_hours', 'Elapsed hours since first trial started', 'hypervolume_vs_time'),
        ):
            ax.clear()
            plotted = sorted(rows, key=lambda row: (row[key], row['trial_number']))
            volume_key = 'hypervolume_by_time' if key == 'elapsed_hours' else 'hypervolume'
            ax.step([row[key] for row in plotted], [row[volume_key] for row in plotted], where='post')
            if reference is None:
                ax.text(.5, .5, 'Waiting for first eligible completed trial', ha='center', transform=ax.transAxes)
            ax.set(title=f'{study_name}\nFeasible hypervolume; {len(terminal)} finished attempts',
                   xlabel=xlabel, ylabel='Hypervolume (higher is better)')
            ax.grid(alpha=.25)
            fig = ax.figure
            fig.set_size_inches(10, 5.5)
            note = ('Configured fixed reference.' if source == 'configured' else
                    'Monitoring only: reference frozen from first eligible trial; not comparable between studies.')
            # Reuse the axes while replacing footer text for each exported view.
            for text in list(fig.texts):
                text.remove()
            fig.text(.01, .01, note, fontsize=8)
            fig.tight_layout(rect=(0, .04, 1, 1))
            for extension in ('png', 'pdf'):
                with atomic_output(output / f'{stem}.{extension}', 'wb') as stream:
                    fig.savefig(stream, format=extension, dpi=180)
        return rows
    finally:
        for number in set(plt.get_fignums()) - before:
            plt.close(number)
