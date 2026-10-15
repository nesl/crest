"""Offline checks for factual late-memory scenario preparation."""
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
SPEC = importlib.util.spec_from_file_location('prepare_memory_late_smoke', ROOT / 'experiments/cs3/prepare_memory_late_smoke.py')
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


class LateScenarioTest(unittest.TestCase):
    """Check source selection, factual seed scope and fresh-path protection."""

    def test_grouped_seed_is_deterministic_and_restricted(self):
        records = [dict(number=i, value=i / 100, latency_ms=i + 1,
                        feasibility_status='feasible', params=dict(
                            num_blocks=2, base_channels=8, stride_schedule='early',
                            global_pool_type='avg')) for i in range(75)]
        findings = prepare.seed_findings(records)
        self.assertEqual(findings, prepare.seed_findings(records))
        self.assertEqual(len(findings), 4)
        self.assertIn('n=75, feasible=75; score n=75, mean=0.37000', findings[0]['observation'])
        self.assertIn('range=1.000..75.000 ms', findings[0]['observation'])
        self.assertTrue(all(f['trial_numbers'] == list(range(75)) for f in findings))
        with self.assertRaises(ValueError):
            prepare.seed_findings(records + [dict(number=75)])

    def test_missing_terminal_trial_and_existing_output_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with sqlite3.connect(root / 'optuna.db') as connection:
                connection.executescript('''
                    CREATE TABLE trials(trial_id INTEGER, number INTEGER, state TEXT);
                    CREATE TABLE trial_params(trial_id INTEGER, param_name TEXT, param_value REAL, distribution_json TEXT);
                    CREATE TABLE trial_user_attributes(trial_id INTEGER, key TEXT, value_json TEXT);
                    CREATE TABLE trial_values(trial_id INTEGER, objective INTEGER, value REAL);
                ''')
                connection.executemany('INSERT INTO trials VALUES (?, ?, ?)',
                                       [(i, i, 'RUNNING' if i == 104 else 'COMPLETE') for i in range(105)])
            request = root / 'request.json'
            request.write_text(json.dumps(dict(messages=[dict(role='user', content='{"context": {}}')])))
            output = root / 'scenario.json'
            before = (root / 'optuna.db').read_bytes()
            with self.assertRaisesRegex(ValueError, 'terminal trials 0..104'):
                prepare.prepare(root, request, output)
            self.assertFalse(output.exists())
            self.assertEqual(before, (root / 'optuna.db').read_bytes())
            output.write_text('preserve')
            with self.assertRaises(FileExistsError):
                prepare.prepare(root, request, output)
            self.assertEqual(output.read_text(), 'preserve')


if __name__ == '__main__':
    unittest.main()
