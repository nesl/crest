"""Offline migration safety and preservation checks on temporary SQLite files."""
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from crest.optimizers.llm.rationale_migration import migrate, validate_transition


class RationaleMigrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.db = self.root / 'study.db'
        self.ledger = self.root / 'ledger'
        self.ledger.mkdir()
        (self.ledger / 'spend_state.json').write_text('{"spent":"1.5"}')
        (self.ledger / 'memory').mkdir()
        (self.ledger / 'memory' / 'state.json').write_text('{"version":7}')
        self.before = {'version': 1, 'optimizer': 'llm_generator', 'config': {'model': 'test', 'candidate_schema_version': 1}, 'sampler': {'class': 'test'}}
        self.after = {**self.before, 'config': {**self.before['config'], 'require_rationale': True, 'candidate_schema_version': 2}}
        with sqlite3.connect(self.db) as conn:
            conn.executescript('CREATE TABLE studies(study_id INTEGER PRIMARY KEY,study_name TEXT); CREATE TABLE study_user_attributes(study_id INTEGER,key TEXT,value_json TEXT); CREATE TABLE trials(number INTEGER,state TEXT,study_id INTEGER);')
            conn.execute('INSERT INTO studies VALUES (1,?)', ('test',))
            conn.execute('INSERT INTO study_user_attributes VALUES (1,?,?)', ('crest_optimizer_signature', json.dumps(self.before)))
            conn.executemany('INSERT INTO trials VALUES (?,?,1)', [(0,'COMPLETE'), (1,'WAITING')])

    def run_migration(self, **kwargs):
        return migrate(self.db, self.ledger, 'test', self.before, self.after, self.root / 'backup', **kwargs)

    def test_preserves_trials_and_ledger_and_backs_up_old_identity(self):
        event = self.run_migration(confirm_stopped=True)
        self.assertEqual(event['legacy_waiting_trial_numbers'], [1])
        self.assertEqual(event['existing_terminal_trial_numbers'], [0])
        self.assertEqual(event['existing_trial_state_counts'], {'COMPLETE': 1, 'WAITING': 1})
        self.assertIn('migrated_at_utc', event)
        with sqlite3.connect(self.db) as conn:
            self.assertEqual(conn.execute('SELECT number,state FROM trials').fetchall(), [(0,'COMPLETE'),(1,'WAITING')])
            self.assertEqual(json.loads(conn.execute("SELECT value_json FROM study_user_attributes WHERE key='crest_optimizer_signature'").fetchone()[0]), self.after)
        with sqlite3.connect(self.root / 'backup' / 'study.sqlite3') as conn:
            self.assertEqual(json.loads(conn.execute('SELECT value_json FROM study_user_attributes').fetchone()[0]), self.before)
        for path in self.ledger.rglob('*'):
            if path.is_file():
                self.assertEqual(path.read_bytes(), (self.root / 'backup' / 'llm_optimizer' / path.relative_to(self.ledger)).read_bytes())
        with self.assertRaises(ValueError):
            self.run_migration(confirm_stopped=True)

    def test_requires_stopped_and_rejects_running_without_backup(self):
        with self.assertRaises(ValueError):
            self.run_migration()
        with sqlite3.connect(self.db) as conn:
            conn.execute("UPDATE trials SET state='RUNNING' WHERE number=1")
        with self.assertRaises(ValueError):
            self.run_migration(confirm_stopped=True)
        self.assertFalse((self.root / 'backup').exists())

    def test_rejects_other_identity_changes(self):
        for key in ('model', 'max_usd', 'prompt_version'):
            changed = {**self.after, 'config': {**self.after['config'], key: 'changed'}}
            with self.assertRaises(ValueError):
                validate_transition(self.before, changed)
        for value in (1, 'true', False):
            with self.assertRaises(ValueError):
                validate_transition(self.before, {**self.after, 'config': {**self.after['config'], 'require_rationale': value}})
        self.before['config']['model'] = 'wrong-source'
        with self.assertRaises(ValueError):
            self.run_migration(confirm_stopped=True)
        self.assertFalse((self.root / 'backup').exists())


if __name__ == '__main__':
    unittest.main()
