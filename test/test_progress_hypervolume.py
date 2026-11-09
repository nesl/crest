# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Regression coverage for feasible multiobjective hypervolume snapshots."""
import csv
from datetime import datetime, timedelta
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import optuna
from optuna.trial import TrialState

from crest.search.progress import read_trials, trial_records, write_progress


def record(number, scores, *, state='COMPLETE', status='feasible', **attrs):
    origin = datetime(2026, 1, 1)
    return dict(number=number, state=state, scores=scores, score=None,
                start=origin + timedelta(hours=number),
                end=origin + timedelta(hours=number + .5),
                attrs={'feasibility_status': status, **attrs})


class HypervolumeProgressTests(unittest.TestCase):
    def test_exact_hypervolume_and_noncontributing_attempts(self):
        trials = [record(0, [8, 8]), record(1, [6, 9]),
                  record(2, None, state='FAIL'),
                  record(3, [1, 1], status='infeasible'),
                  record(4, [0, 0], pruned=True),
                  record(5, [0, 0], state='PRUNED'),
                  record(6, [float('nan'), 0]),
                  record(7, [0, float('inf')]),
                  record(8, [0, 0], status='not_evaluated')]
        with tempfile.TemporaryDirectory() as directory:
            rows = write_progress(trials, directory, 'multi',
                                  directions=['minimize', 'minimize'],
                                  objective_names=['Latency', 'Energy'], reference_point=[10, 10])
            self.assertEqual([r['hypervolume'] for r in rows], [4, 6, 6, 6, 6, 6, 6, 6, 6])
            self.assertEqual([r['attempt'] for r in rows], list(range(1, 10)))
            self.assertEqual(rows[-1]['elapsed_hours'], 8.5)
            output = Path(directory) / 'progress'
            with (output / 'hypervolume_progress.csv').open() as stream:
                saved = list(csv.DictReader(stream))
            self.assertEqual([float(r['hypervolume']) for r in saved], [4, 6, 6, 6, 6, 6, 6, 6, 6])
            metadata = json.loads((output / 'hypervolume_reference.json').read_text())
            self.assertEqual(metadata['reference_point'], [10, 10])
            self.assertEqual(metadata['study_name'], 'multi')
            self.assertEqual(metadata['objective_names'], ['Latency', 'Energy'])
            for stem in ('hypervolume_history', 'hypervolume_vs_time'):
                for extension in ('png', 'pdf'):
                    self.assertGreater((output / f'{stem}.{extension}').stat().st_size, 100)

    def test_parallel_completion_time_does_not_include_future_results(self):
        first = record(0, [8, 8])
        first['end'] = datetime(2026, 1, 1, 4)
        second = record(1, [6, 9])
        with tempfile.TemporaryDirectory() as directory:
            rows = write_progress([first, second], directory, 'parallel',
                                  directions=['MINIMIZE', 'MINIMIZE'], reference_point=[10, 10])
            self.assertEqual([row['trial_number'] for row in rows], [0, 1])
            self.assertEqual([row['hypervolume'] for row in rows], [4, 6])
            self.assertEqual([row['hypervolume_by_time'] for row in rows], [6, 4])

    def test_mixed_directions_transform_reference_and_values_together(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = write_progress([record(0, [8, 2]), record(1, [6, 1])],
                                  directory, 'mixed', directions=['MINIMIZE', 'MAXIMIZE'],
                                  reference_point=[10, 0])
            self.assertEqual([r['hypervolume'] for r in rows], [4, 6])
            metadata = json.loads((Path(directory) / 'progress/hypervolume_reference.json').read_text())
            self.assertEqual(metadata['reference_point'], [10, 0])

    def test_fallback_reference_uses_first_trial_number_and_survives_resume(self):
        # Completion order differs from trial order; only eligible trials anchor the reference.
        first = record(1, [8, 2])
        first['end'] = datetime(2026, 1, 1, 4)
        trials = [record(0, [100, 100], status='infeasible'), first, record(2, [6, 3])]
        with tempfile.TemporaryDirectory() as directory:
            write_progress(trials, directory, 'resume', directions=['MINIMIZE', 'MAXIMIZE'])
            metadata_path = Path(directory) / 'progress/hypervolume_reference.json'
            metadata = json.loads(metadata_path.read_text())
            self.assertEqual(metadata['reference_point'], [8.8, 1.8])
            write_progress(trials + [record(3, [1, 10])], directory, 'resume',
                           directions=['MINIMIZE', 'MAXIMIZE'])
            self.assertEqual(json.loads(metadata_path.read_text()), metadata)

    def test_no_eligible_trial_waits_before_freezing_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            trials = [record(0, None, state='FAIL')]
            rows = write_progress(trials, directory, 'waiting', directions=['MINIMIZE', 'MAXIMIZE'])
            self.assertEqual(rows[0]['hypervolume'], 0)
            metadata_path = Path(directory) / 'progress/hypervolume_reference.json'
            self.assertIsNone(json.loads(metadata_path.read_text())['reference_point'])
            write_progress(trials + [record(1, [0, 0])], directory, 'waiting',
                           directions=['MINIMIZE', 'MAXIMIZE'])
            # Zero objectives still receive a nonzero direction-aware margin.
            self.assertEqual(json.loads(metadata_path.read_text())['reference_point'], [.1, -.1])

    def test_reference_exterior_and_dominated_points_do_not_add_volume(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = write_progress([record(0, [11, 1]), record(1, [8, 8]), record(2, [9, 9])],
                                  directory, 'dominated', directions=['MINIMIZE', 'MINIMIZE'],
                                  reference_point=[10, 10])
            self.assertEqual([row['hypervolume'] for row in rows], [0, 4, 4])

    def test_configured_reference_cannot_change_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            trials = [record(0, [8, 8])]
            rows = write_progress(trials, directory, 'immutable',
                                  directions=['MINIMIZE', 'MINIMIZE'], reference_point=[10, 10])
            output = Path(directory) / 'progress'
            before = {path.name: path.read_bytes() for path in output.iterdir()}
            with self.assertRaises(ValueError):
                write_progress(trials, directory, 'immutable',
                               directions=['MINIMIZE', 'MINIMIZE'], reference_point=[20, 20])
            self.assertEqual({path.name: path.read_bytes() for path in output.iterdir()}, before)
            resumed = write_progress(trials, directory, 'immutable', directions=['MINIMIZE', 'MINIMIZE'])
            self.assertEqual(resumed, rows)

    def test_unconstrained_trials_count_but_explicit_rejections_do_not(self):
        trials = [record(0, [8, 8], status='not_evaluated'),
                  record(1, [0, 0], status='infeasible'), record(2, [0, 0], pruned=True)]
        with tempfile.TemporaryDirectory() as directory:
            rows = write_progress(trials, directory, 'unconstrained', feasibility_enabled=False,
                                  directions=['MINIMIZE', 'MINIMIZE'], reference_point=[10, 10])
            self.assertEqual([r['hypervolume'] for r in rows], [4, 4, 4])

    def test_native_optuna_callback_writes_snapshot_and_clears_failure(self):
        from nas_model_client import NASModelClient
        client = NASModelClient.__new__(NASModelClient)
        client.config = SimpleNamespace(outputs=SimpleNamespace(hypervolume_reference_point=[10, 10]))
        study = optuna.create_study(study_name='callback', directions=['minimize', 'minimize'])
        study.set_metric_names(['Latency', 'Energy'])

        def objective(trial):
            trial.set_user_attr('feasibility_status', 'feasible')
            return [8, 8]

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'progress'
            output.mkdir()
            diagnostic = output / 'error.txt'
            with patch.object(client, '_artifacts_dir', return_value=Path(directory)), \
                    patch.object(client, '_feasibility_enabled', return_value=True):
                with patch('crest.search.progress.write_progress', side_effect=OSError('disk full')):
                    study.optimize(objective, n_trials=1, callbacks=[client._refresh_progress])
                self.assertIn('disk full', diagnostic.read_text())
                study.optimize(objective, n_trials=1, callbacks=[client._refresh_progress])
            self.assertFalse(diagnostic.exists())
            with (output / 'hypervolume_progress.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([float(row['hypervolume']) for row in rows], [4, 4])
            metadata = json.loads((output / 'hypervolume_reference.json').read_text())
            self.assertEqual(metadata['objective_names'], ['Latency', 'Energy'])
            self.assertEqual(metadata['reference_point'], [10, 10])

    def test_sqlite_snapshot_matches_live_multiobjective_trials(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / 'optuna.db'
            study = optuna.create_study(study_name='multi', directions=['minimize', 'maximize'],
                                        storage=f'sqlite:///{database}')
            for values, status in [([8, 2], 'feasible'), ([6, 1], 'feasible'), ([0, 9], 'infeasible')]:
                trial = study.ask()
                trial.set_user_attr('feasibility_status', status)
                study.tell(trial, values)
            study.tell(study.ask(), state=TrialState.FAIL)
            before = database.read_bytes()
            offline = read_trials(database, 'multi')
            self.assertEqual(database.read_bytes(), before)
            live = trial_records(study.get_trials(deepcopy=False))
            self.assertEqual([t['scores'] for t in offline[:-1]], [t['scores'] for t in live[:-1]])
            self.assertTrue(all(value is None for value in offline[-1]['scores']))
            self.assertIsNone(live[-1]['scores'])
            offline_rows = write_progress(offline, Path(directory) / 'offline', 'multi', reference_point=[10, 0])
            live_rows = write_progress(live, Path(directory) / 'live', 'multi',
                                       directions=study.directions, reference_point=[10, 0])
            self.assertEqual(offline_rows, live_rows)
            self.assertEqual([r['hypervolume'] for r in live_rows], [4, 6, 6, 6])


if __name__ == '__main__':
    unittest.main()
