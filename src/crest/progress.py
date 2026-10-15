"""Offline, reusable feasible-score progress artifacts for NAS studies."""
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import sqlite3


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
            if directions != [('MAXIMIZE',)]:
                raise ValueError('This score plot requires one maximized objective.')
            rows = snapshot.execute('''
                SELECT t.trial_id,t.number,t.state,t.datetime_start,t.datetime_complete,v.value
                FROM trials t LEFT JOIN trial_values v
                  ON v.trial_id=t.trial_id AND v.objective=0
                WHERE t.study_id=? ORDER BY t.number
            ''', (study[0],)).fetchall()
            result = []
            for tid, number, state, start, end, score in rows:
                attrs = {key: json.loads(value) for key, value in snapshot.execute(
                    'SELECT key,value_json FROM trial_user_attributes WHERE trial_id=?', (tid,))}
                result.append(dict(number=number, state=state, start=start, end=end,
                                   score=score, attrs=attrs))
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
                 attrs=t.user_attrs) for t in trials]


def _datetime(value):
    """Normalize a stored timestamp for elapsed-time arithmetic.

    Parameters
    ----------
    value : str or datetime.datetime
        ISO-formatted SQLite value or an Optuna timestamp.

    Returns
    -------
    datetime.datetime
        Parsed or original timestamp.
    """
    return datetime.fromisoformat(value) if isinstance(value, str) else value


def write_progress(trials, directory, study_name, warmup_count=None):
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

    Returns
    -------
    list of dict
        CSV rows, including failed, pruned, and infeasible attempts.
    """
    # Import plotting lazily: missing optional plotting dependencies must not
    # prevent an otherwise valid NAS run from starting.
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
        feasible = (t['state'] == 'COMPLETE' and t['attrs'].get('feasibility_status') == 'feasible'
                    and score is not None and math.isfinite(score))
        if feasible:
            best = max(best, score) if best is not None else score
        plotted.append(dict(trial_number=t['number'], attempt=t['number'] + 1,
                            elapsed_hours=(_datetime(t['end']) - origin).total_seconds()/3600,
                            state=t['state'], score=score if feasible else None, best_so_far=best,
                            proposal_source=t['attrs'].get('crest_proposal', {}).get('proposal_source', 'unknown')))
    output = Path(directory) / 'progress'
    output.mkdir(parents=True, exist_ok=True)
    fields = ['trial_number', 'attempt', 'elapsed_hours', 'state', 'score', 'best_so_far', 'proposal_source']
    with (output / 'score_progress.csv').open('w', newline='') as stream:
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
                        where='post', label='Best feasible score', color='tab:green')
            if warmup_done:
                ax.axvline(boundary_hours if key == 'elapsed_hours' else max(t['number'] for t in warmup) + 1.5,
                           color='gray', linestyle='--', label='Warmup finished')
            count = sum(r['score'] is not None for r in plotted)
            ax.set(title=f'{study_name}\n{count} feasible scores; {len(terminal)} finished attempts',
                   xlabel=xlabel, ylabel='Score (higher is better)')
            ax.grid(alpha=0.25)
            if incumbents or warmup_done:
                ax.legend()
            fig.text(0.01, 0.01, 'Only feasible completed scores shown. Time includes inter-trial API/retry/idle gaps.', fontsize=8)
            fig.tight_layout(rect=(0, 0.04, 1, 1))
            for extension in ('png', 'pdf'):
                fig.savefig(output / f'{stem}.{extension}', dpi=180)
        finally:
            plt.close(fig)
    return plotted
