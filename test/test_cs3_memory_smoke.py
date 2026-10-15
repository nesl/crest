"""Focused offline tests for the isolated real-evidence memory smoke runner."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
spec = importlib.util.spec_from_file_location('cs3_memory_smoke', ROOT / 'experiments/cs3/memory_smoke.py')
smoke = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = smoke
spec.loader.exec_module(smoke)


class MemorySmokeTest(unittest.TestCase):
    """Verify exact evidence, bounded rejection and provider-free resume."""

    def test_saved_evidence_and_reload(self):
        """Retain source evidence byte-for-byte and reload without any calls."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = [dict(number=i, measured_score=i / 100, state='COMPLETE') for i in range(15)]
            source = root / 'source.json'
            source.write_text(json.dumps({'messages': [{'role': 'user', 'content': json.dumps(
                dict(context={'study_name': 'saved-real-evidence', 'search_space': {},
                              'memory_policy': {'lag_trials': 15}}, new_trials=records))}]}))
            with patch.dict('os.environ', {'OPENAI_API_KEY': 'offline'}), patch(
                    'socket.socket.connect', side_effect=AssertionError('Network forbidden')):
                report = smoke.run(root / 'success', source)
            self.assertTrue(report['passed'])
            self.assertEqual(report['reload_provider_requests'], 0)
            self.assertEqual(report['http_requests'], 0)
            self.assertEqual(report['transport_attempts'], 1)
            self.assertEqual(report['source_sha256'], hashlib.sha256(source.read_bytes()).hexdigest())
            generated = json.loads((root / 'success/memory/requests/000001.request.json').read_text())
            body = json.loads(generated['messages'][1]['content'])
            self.assertEqual(body['new_trials'], records)
            self.assertEqual(body['context']['memory_policy']['lag_trials'], 0)
            empty = root / 'empty.json'
            empty.write_text('{"updates": [], "retire": []}')
            with patch.dict('os.environ', {'OPENAI_API_KEY': 'offline'}), patch(
                    'socket.socket.connect', side_effect=AssertionError('Network forbidden')):
                with self.assertRaisesRegex(RuntimeError, 'nonempty summary'):
                    smoke.run(root / 'empty', source, response=empty)
            failure = json.loads((root / 'empty/report.json').read_text())
            self.assertFalse(failure['passed'])
            self.assertEqual(failure['source_sha256'], report['source_sha256'])
            self.assertEqual(failure['transport_attempts'], 1)
            invalid = root / 'invalid.json'
            invalid.write_text('{"wrong": true}')
            with patch.dict('os.environ', {'OPENAI_API_KEY': 'offline'}), patch(
                    'socket.socket.connect', side_effect=AssertionError('Network forbidden')):
                with self.assertRaises(RuntimeError):
                    smoke.run(root / 'invalid', source, response=invalid)
            failure = json.loads((root / 'invalid/report.json').read_text())
            self.assertEqual(failure['transport_attempts'], 2)
            self.assertFalse(failure['passed'])


    def test_late_scenario_holds_out_recent_trials(self):
        """Exercise seeded memory through the production lagged updater.

        Returns
        -------
        None
            Asserts exact new evidence, coverage, and provider-free reload.
        """
        # No API or real data is needed to verify the 75/15/15 boundary.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            finding = dict(id='seed', observation='Seed fixture.', conditions='Offline only.',
                           evidence='Earlier records.', exceptions='Not scientific evidence.',
                           uncertainty='Lifecycle only.', trial_numbers=[0])
            fixture = dict(context={'study_name': 'late-proxy'},
                           records=[dict(number=i, state='complete') for i in range(105)],
                           initial_findings=[finding], covered_trial_numbers=list(range(75)),
                           lag_trials=15, expected_new_trial_numbers=list(range(75, 90)),
                           provenance={'kind': 'offline fixture'})
            source = root / 'scenario.json'
            source.write_text(json.dumps(fixture))
            with patch.dict('os.environ', {'OPENAI_API_KEY': 'offline'}), patch(
                    'socket.socket.connect', side_effect=AssertionError('Network forbidden')):
                report = smoke.run(root / 'run', scenario=source)
            self.assertTrue(report['passed'])
            self.assertEqual(report['covered_trial_count'], 90)
            self.assertEqual(report['http_requests'], 0)
            self.assertEqual(report['transport_attempts'], 1)
            self.assertEqual(report['reload_provider_requests'], 0)
            saved = json.loads((root / 'run/memory/state.json').read_text())
            self.assertEqual(set(saved['covered']), {str(i) for i in range(90)})
            request = json.loads((root / 'run/memory/requests/000001.request.json').read_text())
            body = json.loads(request['messages'][1]['content'])
            self.assertEqual([r['number'] for r in body['new_trials']], list(range(75, 90)))
            self.assertEqual(body['previous_findings'], [finding])


if __name__ == '__main__':
    unittest.main()
