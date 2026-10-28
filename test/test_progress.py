"""Offline regression coverage for feasible-score progress artifacts."""
import csv
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

import optuna
from optuna.trial import TrialState

from crest.search.progress import read_trials, trial_records, write_progress


class ProgressTests(unittest.TestCase):
    def test_all_attempts_preserve_incumbent_and_real_attempt_positions(self):
        """Failed, pruned, and infeasible attempts extend the existing incumbent."""
        origin = datetime(2026, 1, 1)
        trials = []
        for number, (state, score, feasible) in enumerate([
                ('FAIL', None, False), ('COMPLETE', .4, True),
                ('PRUNED', .9, False), ('COMPLETE', .95, False),
                ('COMPLETE', .3, True), ('COMPLETE', .7, True)]):
            trials.append(dict(number=number, state=state, score=score,
                               start=origin + timedelta(hours=number),
                               end=origin + timedelta(hours=number + .5),
                               attrs={'feasibility_status': 'feasible' if feasible else 'infeasible',
                                      'crest_proposal': {'proposal_source': 'llm_warmup' if number < 2 else 'llm'}}))
        with tempfile.TemporaryDirectory() as directory:
            rows = write_progress(trials, directory, 'offline', warmup_count=2)
            self.assertEqual([r['attempt'] for r in rows], [1, 2, 3, 4, 5, 6])
            self.assertEqual([r['best_so_far'] for r in rows], [None, .4, .4, .4, .4, .7])
            self.assertEqual(rows[-1]['elapsed_hours'], 5.5)
            self.assertEqual(rows[3]['score'], None)
            output = Path(directory) / 'progress'
            for stem in ('score_vs_time', 'score_vs_trial'):
                for extension in ('png', 'pdf'):
                    self.assertGreater((output / f'{stem}.{extension}').stat().st_size, 100)
            with (output / 'score_progress.csv').open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 6)

    def test_disabled_feasibility_includes_success_and_preserves_rejections(self):
        """Unconstrained successes count while rejected or nonfinite attempts do not."""
        trials = [dict(number=n, state=state, score=score, attrs=attrs,
                       start='2026-01-01 00:00:00', end='2026-01-01 00:30:00')
                  for n, (state, score, attrs) in enumerate([
                      ('COMPLETE', .5, {}),
                      ('COMPLETE', .6, {'feasibility_status': 'not_evaluated'}),
                      ('COMPLETE', .9, {'feasibility_status': 'infeasible'}),
                      ('COMPLETE', .95, {'pruned': True}),
                      ('PRUNED', .99, {}),
                      ('COMPLETE', float('inf'), {}),
                  ])]
        with tempfile.TemporaryDirectory() as directory:
            rows = write_progress(trials, directory, 'unconstrained', feasibility_enabled=False)
            self.assertEqual([r['score'] for r in rows], [.5, .6, None, None, None, None])
            self.assertEqual(rows[-1]['best_so_far'], .6)
            strict = write_progress(trials, directory, 'strict-default')
            self.assertTrue(all(r['score'] is None for r in strict))

    def test_no_feasible_score_still_writes_snapshot(self):
        """A failed first attempt still yields reviewable progress files."""
        trial = dict(number=0, state='FAIL', score=None, attrs={},
                     start='2026-01-01 00:00:00', end='2026-01-01 00:30:00')
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(write_progress([trial], directory, 'failed')[0]['best_so_far'])
            self.assertTrue((Path(directory) / 'progress/score_vs_time.png').exists())

    def test_warmup_boundary_requires_full_expected_sequence(self):
        """Warmup completion is marked only once its configured count finishes."""
        trial = dict(number=0, state='COMPLETE', score=.5,
                     attrs={'feasibility_status': 'feasible',
                            'crest_proposal': {'proposal_source': 'llm_warmup'}},
                     start='2026-01-01 00:00:00', end='2026-01-01 01:00:00')
        with tempfile.TemporaryDirectory() as directory:
            with patch('matplotlib.axes.Axes.axvline') as boundary:
                write_progress([trial], directory, 'partial', warmup_count=2)
                boundary.assert_not_called()
            with patch('matplotlib.axes.Axes.axvline') as boundary:
                write_progress([trial], directory, 'complete', warmup_count=1)
                self.assertEqual([call.args[0] for call in boundary.call_args_list], [1.0, 1.5])

    def test_callback_plot_failure_is_nonfatal_and_uses_current_study(self):
        """Plot exceptions cannot terminate the training or proposal loop."""
        from nas_model_client import NASModelClient
        client = NASModelClient.__new__(NASModelClient)
        study = SimpleNamespace(directions=[SimpleNamespace(name='MAXIMIZE')],
                                study_name='offline', get_trials=MagicMock(return_value=[]))
        with patch.object(client, '_artifacts_dir', return_value=Path('/unused')), \
                patch.object(client, '_feasibility_enabled', return_value=False):
            with patch('crest.search.progress.write_progress', side_effect=OSError('disk full')) as writer:
                NASModelClient._refresh_progress(client, study, None)
                writer.assert_called_once()
                self.assertFalse(writer.call_args.kwargs['feasibility_enabled'])
                study.get_trials.assert_called_once_with(deepcopy=False)

    def test_readonly_sqlite_and_frozen_trial_adapters_agree(self):
        """The standalone SQLite snapshot agrees with the live trial adapter."""
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / 'optuna.db'
            study = optuna.create_study(study_name='offline', direction='maximize',
                                        storage=f'sqlite:///{database}')
            trial = study.ask()
            trial.set_user_attr('feasibility_status', 'feasible')
            study.tell(trial, .5)
            before = database.read_bytes()
            records = read_trials(database, 'offline')
            self.assertEqual(database.read_bytes(), before)
            adapted = trial_records(study.get_trials(deepcopy=False))
            self.assertEqual(records[0]['score'], adapted[0]['score'])
            self.assertEqual(records[0]['attrs'], adapted[0]['attrs'])
            self.assertEqual(records[0]['state'], 'COMPLETE')


if __name__ == '__main__':
    unittest.main()
