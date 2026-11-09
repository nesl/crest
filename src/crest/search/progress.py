# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Live and offline score/hypervolume progress artifacts for NAS studies."""
import csv
import json
import math
from pathlib import Path
import sqlite3
from .progress_artifacts import atomic_output as _atomic_output, as_datetime as _datetime, eligible_trial


def read_trials(database, study_name):
    """Read a consistent snapshot without modifying the campaign database.

    Parameters
    ----------
    database : pathlib.Path
        Existing Optuna SQLite database.
    study_name : str
        Exact study to select.

    Returns
    -------
    list of dict
        Trials with scores, timestamps, states, and persisted user attributes.

    Raises
    ------
    ValueError
        If the study is absent or has no supported objective directions.
    sqlite3.Error
        If the database cannot be opened or its schema cannot be read.
    """
    with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True) as source:
        with sqlite3.connect(':memory:') as snapshot:
            source.backup(snapshot)
            study = snapshot.execute('SELECT study_id FROM studies WHERE study_name=?',
                                     (study_name,)).fetchone()
            if study is None:
                raise ValueError(f'Study not found: {study_name}')
            directions = snapshot.execute(
                'SELECT direction FROM study_directions WHERE study_id=? ORDER BY objective',
                (study[0],)).fetchall()
            directions = [row[0] for row in directions]
            if not directions or any(d not in {'MAXIMIZE', 'MINIMIZE'} for d in directions):
                raise ValueError('Study must have maximize/minimize objectives.')
            rows = snapshot.execute('''
                SELECT t.trial_id,t.number,t.state,t.datetime_start,t.datetime_complete
                FROM trials t
                WHERE t.study_id=? ORDER BY t.number
            ''', (study[0],)).fetchall()
            result = []
            for tid, number, state, start, end in rows:
                values = dict(snapshot.execute(
                    "SELECT objective,value FROM trial_values WHERE trial_id=?", (tid,)))
                scores = [values.get(i) for i in range(len(directions))]
                attrs = {key: json.loads(value) for key, value in snapshot.execute(
                    'SELECT key,value_json FROM trial_user_attributes WHERE trial_id=?', (tid,))}
                result.append(dict(number=number, state=state, start=start, end=end,
                                   score=scores[0] if len(scores) == 1 else None,
                                   scores=scores, directions=directions, attrs=attrs))
            return result


def trial_records(trials):
    """Adapt Optuna trials without opening a second storage connection.

    Parameters
    ----------
    trials : iterable of optuna.trial.FrozenTrial
        Current study snapshot.

    Returns
    -------
    list of dict
        Records accepted by ``write_progress``.
    """
    return [dict(number=t.number, state=t.state.name, start=t.datetime_start,
                 end=t.datetime_complete, score=t.value if len(t.values or []) <= 1 else None,
                 scores=list(t.values) if t.values is not None else None,
                 attrs=t.user_attrs) for t in trials]


def _write_objective_progress(trials, directory, study_name, warmup_count=None, *,
                              feasibility_enabled=True, direction="MAXIMIZE",
                              objective_name="Score"):
    """Render a snapshot including every finished attempt and feasible incumbent.

    Parameters
    ----------
    trials : iterable of dict
        Adapted records or records returned by ``read_trials``.
    directory : pathlib.Path
        Per-study artifact directory; files are written in ``progress``.
    study_name : str
        Title for both charts.
    warmup_count : int or None, optional
        Expected warmup size. Without it, a subsequent non-warmup attempt is
        required before declaring the warmup complete.

    feasibility_enabled : bool, optional
        Require affirmative feasibility evidence, default True. Set False only
        for a study configured without feasibility rules. Explicit rejection
        and infeasibility evidence still excludes a score.

    Returns
    -------
    list of dict
        CSV rows, including failed, pruned, and infeasible attempts. Returns
        an empty list without writing artifacts if no trial has started.

    Notes
    -----
    Writes ``score_progress.csv`` and PNG/PDF plots against elapsed time and
    trial attempt. Only finite scores from feasible COMPLETE trials contribute
    to the incumbent curve. All terminal attempts retain their positions.
    """
    # Keep plotting imports out of startup; callback failures leave a diagnostic
    # artifact and allow the search to continue.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    trials = list(trials)
    starts = [_datetime(t['start']) for t in trials if t['start']]
    if not starts:
        return []
    origin = min(starts)
    terminal = [t for t in trials if t['state'] in {'COMPLETE', 'PRUNED', 'FAIL'} and t['end']]
    terminal.sort(key=lambda t: (_datetime(t['end']), t['number']))
    warmup = [t for t in trials if t['attrs'].get('crest_proposal', {}).get('proposal_source') == 'llm_warmup']
    later = any(t['number'] > max((w['number'] for w in warmup), default=-1)
                and t['attrs'].get('crest_proposal', {}).get('proposal_source') != 'llm_warmup'
                for t in trials)
    warmup_done = bool(warmup) and all(t['end'] for t in warmup) and (
        len(warmup) == warmup_count if warmup_count is not None else later)
    boundary_hours = ((_datetime(max(warmup, key=lambda t: _datetime(t['end']))['end']) - origin)
                      .total_seconds() / 3600) if warmup_done else None
    plotted, best = [], None
    for t in terminal:
        score = t['score']
        feasible = (eligible_trial(t, feasibility_enabled)
                    and score is not None and math.isfinite(score))
        if feasible:
            choose = max if direction == "MAXIMIZE" else min
            best = choose(best, score) if best is not None else score
        plotted.append(dict(trial_number=t['number'], attempt=t['number'] + 1,
                            elapsed_hours=(_datetime(t['end']) - origin).total_seconds()/3600,
                            state=t['state'], score=score if feasible else None, best_so_far=best,
                            proposal_source=t['attrs'].get('crest_proposal', {}).get('proposal_source', 'unknown')))
    output = Path(directory) / 'progress'
    output.mkdir(parents=True, exist_ok=True)
    fields = ['trial_number', 'attempt', 'elapsed_hours', 'state', 'score', 'best_so_far', 'proposal_source']
    with _atomic_output(output / 'score_progress.csv', 'w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(plotted)
    for key, xlabel, stem in (
        ('elapsed_hours', 'Elapsed hours since first trial started', 'score_vs_time'),
        ('attempt', 'Trial attempt (1-based; failed/pruned attempts retain their positions)', 'score_vs_trial'),
    ):
        fig, ax = plt.subplots(figsize=(10, 5.5))
        try:
            for source, label, color in (('llm_warmup', 'Warmup', 'tab:gray'),
                                         ('llm', 'LLM proposal', 'tab:blue'),
                                         ('other', 'Other proposal', 'tab:orange')):
                group = [r for r in plotted if r['score'] is not None and
                         (r['proposal_source'] == source if source != 'other'
                          else r['proposal_source'] not in {'llm_warmup', 'llm'})]
                if group:
                    ax.scatter([r[key] for r in group], [r['score'] for r in group],
                               label=label, color=color, alpha=0.75, s=30)
            incumbents = sorted((r for r in plotted if r['best_so_far'] is not None), key=lambda r: r[key])
            if incumbents:
                ax.step([r[key] for r in incumbents], [r['best_so_far'] for r in incumbents],
                        where='post', label='Best eligible value', color='tab:green')
            if warmup_done:
                ax.axvline(boundary_hours if key == 'elapsed_hours' else max(t['number'] for t in warmup) + 1.5,
                           color='gray', linestyle='--', label='Warmup finished')
            count = sum(r['score'] is not None for r in plotted)
            ax.set(title=f'{study_name}\n{count} eligible values; {len(terminal)} finished attempts',
                   xlabel=xlabel, ylabel=f'{objective_name} ({"higher" if direction == "MAXIMIZE" else "lower"} is better)')
            ax.grid(alpha=0.25)
            if incumbents or warmup_done:
                ax.legend()
            note = 'Eligible completed trials only. Time includes API/retry/idle gaps.'
            fig.text(0.01, 0.01, note, fontsize=8)
            fig.tight_layout(rect=(0, 0.04, 1, 1))
            for extension in ('png', 'pdf'):
                with _atomic_output(output / f'{stem}.{extension}', 'wb') as stream:
                    fig.savefig(stream, format=extension, dpi=180)
        finally:
            plt.close(fig)
    return plotted


def write_progress(trials, directory, study_name, warmup_count=None, *,
                   feasibility_enabled=True, directions=None, objective_names=None,
                   reference_point=None):
    """Write scalar score history or multiobjective hypervolume after each trial.

    In-memory records require directions for multiobjective studies; SQLite records
    include them. Scalar records default to maximization. Multiobjective studies
    use Optuna's Matplotlib hypervolume plot with a fixed, persisted reference in
    objective order/native units. Without an explicit reference, the first eligible
    trial plus a direction-aware 10% margin is frozen for monitoring only.

    Returns CSV rows. Scalar artifact names stay unchanged; multiobjective output
    uses ``hypervolume_progress.csv``, ``hypervolume_history`` and
    ``hypervolume_vs_time`` PNG/PDF files under ``directory/progress``.
    """
    trials = list(trials)
    directions = directions or next((t['directions'] for t in trials if t.get('directions')), None)
    if directions is None:
        if any(len(t.get('scores') or []) > 1 for t in trials):
            raise ValueError('Multiobjective records require objective directions.')
        directions = ['MAXIMIZE']
    directions = [getattr(d, 'name', str(d)).upper() for d in directions]
    if not directions or any(d not in {'MAXIMIZE', 'MINIMIZE'} for d in directions):
        raise ValueError('Expected maximize/minimize objective directions.')
    names = list(objective_names) if objective_names is not None else [
        'Score' if len(directions) == 1 else f'Objective {i}' for i in range(len(directions))]
    if len(names) != len(directions):
        raise ValueError('Objective names must match the number of directions.')
    if len(directions) > 1:
        from .hypervolume import write_hypervolume
        return write_hypervolume(
            trials, directory, study_name, directions=directions, objective_names=names,
            feasibility_enabled=feasibility_enabled, reference_point=reference_point,
        )
    records = []
    for trial in trials:
        values = trial.get('scores')
        if values is not None and len(values) != 1:
            raise ValueError('Trial values must match objective directions.')
        records.append({**trial, 'score': values[0] if values else trial.get('score')})
    return _write_objective_progress(
        records, directory, study_name, warmup_count,
        feasibility_enabled=feasibility_enabled, direction=directions[0], objective_name=names[0],
    )
