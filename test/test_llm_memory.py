# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Memory lifecycle, evidence retention, bounded backlog and resume tests."""
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import optuna
from optuna.trial import TrialState, create_trial

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from crest.optimizers.llm.history import build_terminal_trial_history
from crest.optimizers.llm.memory import ExperimentalMemory, MemoryConfig
from crest.optimizers.llm.prompt_builder import PromptContext, build_candidate_request
from crest.optimizers.llm.provider import FakeProvider, LLMRequest, ProviderAuthenticationError, ProviderTransportError
from crest.optimizers.llm.search_space import SearchParam, SearchSpaceDescriptor
from crest.model import _normalize_optimizer_config
from addict import Dict


def finding(**overrides):
    return dict(id='memory-risk', observation='Wider float models encountered allocation failures.',
                conditions='Float on this board, six dilation levels.',
                evidence='Trial 0 reports an allocation failure at 48 filters.',
                exceptions='No int8 model has been tested.',
                uncertainty='A filter threshold has not been isolated.', trial_numbers=[0], **overrides)


def context(study, recent=1):
    records = build_terminal_trial_history(study)
    return PromptContext(study_name=study.study_name, model_family='test',
                         descriptor=SearchSpaceDescriptor([SearchParam('width', 'int', low=1, high=64)]),
                         objective_summary='minimize error', attempted_trials=len(study.trials),
                         feasible_completed_trials=len(records), target_feasible_trials=20,
                         max_total_attempts=30, batch_size=1,
                         recent_trials=records[-recent:] if recent else ())


def add(study, state=TrialState.COMPLETE):
    study.add_trial(create_trial(state=state, value=1.0 if state == TrialState.COMPLETE else None))


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.study = optuna.create_study(study_name='memory-test')
        self.config = MemoryConfig(start_after_trials=2, interval_trials=2, max_pending_trials=4)

    def test_start_interval_and_unseen_pending_are_independent_of_recent_window(self):
        provider = FakeProvider([{'updates': [finding()], 'retire': []}, {'updates': [], 'retire': []}])
        memory = ExperimentalMemory(self.root, self.config)
        add(self.study)
        result = memory.enrich(self.study, context(self.study, 0), provider)
        self.assertEqual(len(provider.requests), 0)
        self.assertEqual([r['number'] for r in result.pending_trials], [0])
        add(self.study)
        result = memory.enrich(self.study, context(self.study), provider)
        self.assertEqual(result.knowledge_base['covered_trial_count'], 2)
        self.assertIn('six dilation levels', result.knowledge_base['summary'])
        self.assertIn('No int8', result.knowledge_base['summary'])
        add(self.study)
        result = memory.enrich(self.study, context(self.study, 0), provider)
        self.assertEqual(len(provider.requests), 1)
        self.assertEqual([r['number'] for r in result.pending_trials], [2])
        add(self.study)
        result = memory.enrich(self.study, context(self.study), provider)
        self.assertEqual(len(provider.requests), 2)
        self.assertIn('threshold has not been isolated', result.knowledge_base['summary'])
        self.assertEqual([r['number'] for r in json.loads(provider.requests[1].user_prompt)['new_trials']], [2, 3])

    def test_interval_larger_than_recent_window_keeps_gap_visible(self):
        memory = ExperimentalMemory(self.root, MemoryConfig(start_after_trials=4, interval_trials=4))
        for _ in range(3):
            add(self.study)
        result = memory.enrich(self.study, context(self.study, 1), FakeProvider([]))
        self.assertEqual([r['number'] for r in result.pending_trials], [0, 1])
        self.assertEqual([r['number'] for r in result.recent_trials], [2])
        payload = json.loads(build_candidate_request(result, prompt_version='v1').user_prompt)
        self.assertEqual(len(payload['pending_trials']), 2)

    def test_restart_preserves_memory_without_reprocessing(self):
        for _ in range(2):
            add(self.study)
        ExperimentalMemory(self.root, self.config).enrich(
            self.study, context(self.study), FakeProvider([{'updates': [finding()], 'retire': []}]))
        provider = FakeProvider([])
        restored = ExperimentalMemory(self.root, self.config)
        result = restored.enrich(self.study, context(self.study), provider)
        self.assertEqual(len(provider.requests), 0)
        self.assertEqual(result.knowledge_base['version'], 1)
        self.assertEqual(len(list((self.root/'snapshots').glob('*.json'))), 1)

    def test_invalid_summary_keeps_old_state_and_all_pending_evidence(self):
        memory = ExperimentalMemory(self.root, self.config)
        for _ in range(2):
            add(self.study)
        memory.enrich(self.study, context(self.study), FakeProvider([{'updates': [finding()], 'retire': []}]))
        before = (self.root/'state.json').read_bytes()
        for _ in range(2):
            add(self.study)
        bad = finding()
        bad['trial_numbers'] = [999]
        result = memory.enrich(self.study, context(self.study, 1), FakeProvider([{'updates': [bad], 'retire': []}] * 2))
        self.assertEqual((self.root/'state.json').read_bytes(), before)
        self.assertEqual([r['number'] for r in result.pending_trials], [2])
        self.assertEqual(result.knowledge_base['covered_trial_count'], 2)

    def test_oversized_and_incomplete_findings_are_rejected(self):
        for _ in range(2):
            add(self.study)
        for update in [{'id': 'incomplete'}, {**finding(), 'evidence': 'x'*10000}]:
            memory = ExperimentalMemory(self.root, self.config)
            result = memory.enrich(self.study, context(self.study), FakeProvider([{'updates': [update], 'retire': []}] * 2))
            self.assertEqual(result.knowledge_base['covered_trial_count'], 0)
            self.assertFalse((self.root/'state.json').exists())

    def test_contradictory_evidence_can_qualify_existing_finding(self):
        memory = ExperimentalMemory(self.root, self.config)
        for _ in range(2):
            add(self.study)
        memory.enrich(self.study, context(self.study), FakeProvider([{'updates': [finding()], 'retire': []}]))
        for _ in range(2):
            add(self.study)
        updated = finding()
        updated.update(exceptions='Trial 2 deployed 48 filters with int8.', trial_numbers=[0, 2])
        result = memory.enrich(self.study, context(self.study), FakeProvider([{'updates': [updated], 'retire': []}]))
        self.assertIn('int8', result.knowledge_base['summary'])
        self.assertIn('Float on this board', result.knowledge_base['summary'])
        self.assertIn('allocation failure', result.knowledge_base['summary'])

    def test_provider_execution_failure_propagates_and_retains_evidence(self):
        """A dead provider stops before stale memory can hide the outage."""
        memory = ExperimentalMemory(self.root, self.config)
        for _ in range(2):
            add(self.study)
        for error in (ProviderAuthenticationError('check account credits'), ProviderTransportError('outage')):
            provider = Mock()
            provider.complete_json.side_effect = error
            with self.subTest(error=error), self.assertRaises(type(error)):
                memory.enrich(self.study, context(self.study, 1), provider)
            self.assertEqual(memory.state['covered'], {})
            self.assertFalse(memory.state_path.exists())
            provider.complete_json.assert_called_once()
        events = [json.loads(line) for line in (self.root/'optimizer_events.jsonl').read_text().splitlines()]
        self.assertEqual(events[-1]['reason'], 'outage')

    def test_rejected_summary_repairs_with_reason_and_same_evidence(self):
        """Size rejection feedback allows correction without advancing coverage early."""
        for _ in range(2):
            add(self.study)
        memory = ExperimentalMemory(self.root, self.config)
        oversized = finding()
        oversized['evidence'] = 'x' * 10000
        provider = FakeProvider([
            {'updates': [oversized], 'retire': []},
            {'updates': [finding()], 'retire': []},
        ])
        result = memory.enrich(self.study, context(self.study), provider)
        first, repair = [json.loads(r.user_prompt) for r in provider.requests]
        self.assertEqual(len(provider.requests), 2)
        self.assertEqual(first['new_trials'], repair['new_trials'])
        self.assertIn('max_chars', repair['repair_feedback']['reason'])
        self.assertEqual(result.knowledge_base['covered_trial_count'], 2)
        # Both responses, including the rejected evidence, remain auditable.
        self.assertEqual(len(list((self.root/'requests').glob('*.response.json'))), 2)

    def test_incomplete_summary_is_repaired_before_committing(self):
        """A truncated but otherwise valid summary cannot commit memory."""
        for _ in range(2):
            add(self.study)
        memory = ExperimentalMemory(self.root, self.config)
        response = FakeProvider([{'updates': [finding()], 'retire': []}]).complete_json(LLMRequest('', '', 'memory-v1'))
        provider = Mock()
        provider.complete_json.side_effect = [replace(response, finish_reason='length'), response]
        result = memory.enrich(self.study, context(self.study), provider)
        self.assertEqual(provider.complete_json.call_count, 2)
        repair = json.loads(provider.complete_json.call_args_list[1].args[0].user_prompt)
        self.assertIn("finish_reason='length'", repair['repair_feedback']['reason'])
        self.assertEqual(result.knowledge_base['version'], 1)
        # Only the complete response creates a committed snapshot.
        self.assertEqual(len(list((self.root/'snapshots').glob('*.json'))), 1)

    def test_persistent_rejection_bounds_prompt_without_losing_backlog(self):
        """Rejected summaries keep all evidence uncovered and explicitly report omissions."""
        for _ in range(7):
            add(self.study)
        memory = ExperimentalMemory(self.root, self.config)
        provider = FakeProvider([{'updates': [{'id': 'incomplete'}], 'retire': []}] * 2)
        result = memory.enrich(self.study, context(self.study, 0), provider)
        self.assertEqual(len(provider.requests), 2)  # No unbounded same-round repair loop.
        self.assertEqual(len(result.pending_trials), 4)
        self.assertEqual(result.knowledge_base['pending_trial_count'], 7)
        self.assertEqual(result.knowledge_base['omitted_pending_trial_count'], 3)
        self.assertEqual(memory.state['covered'], {})
        self.assertFalse(memory.state_path.exists())
        # The next successful round drains both chunks, including previously omitted trials.
        recovery = FakeProvider([{'updates': [], 'retire': []}] * 2)
        result = memory.enrich(self.study, context(self.study, 0), recovery)
        self.assertEqual(result.knowledge_base['covered_trial_count'], 7)
        self.assertEqual(result.knowledge_base['omitted_pending_trial_count'], 0)
        self.assertEqual([r['number'] for r in json.loads(recovery.requests[1].user_prompt)['new_trials']], [4, 5, 6])

    def test_scope_or_changed_history_cannot_reuse_old_memory(self):
        for _ in range(2):
            add(self.study)
        memory = ExperimentalMemory(self.root, self.config)
        memory.enrich(self.study, context(self.study), FakeProvider([{'updates': [], 'retire': []}]))
        with self.assertRaisesRegex(ValueError, 'different context'):
            memory.enrich(self.study, replace(context(self.study), objective_summary='changed'), FakeProvider([]))
        other = optuna.create_study(study_name='memory-test')
        other.add_trial(create_trial(value=5.0))
        add(other)
        with self.assertRaisesRegex(ValueError, 'trial history'):
            ExperimentalMemory(self.root, self.config).enrich(other, context(other), FakeProvider([]))

    def test_only_terminal_trials_are_summarized_and_late_completion_is_not_lost(self):
        running = self.study.ask()
        add(self.study, TrialState.FAIL)
        add(self.study, TrialState.PRUNED)
        memory = ExperimentalMemory(self.root, self.config)
        provider = FakeProvider([{'updates': [], 'retire': []}])
        memory.enrich(self.study, context(self.study), provider)
        self.assertEqual([r['number'] for r in json.loads(provider.requests[0].user_prompt)['new_trials']], [1, 2])
        self.study.tell(running, 1.0)
        result = memory.enrich(self.study, context(self.study, 0), FakeProvider([]))
        self.assertEqual([r['number'] for r in result.pending_trials], [0])

    def test_retirement_is_explicit_and_old_snapshot_survives(self):
        memory = ExperimentalMemory(self.root, self.config)
        for _ in range(2):
            add(self.study)
        memory.enrich(self.study, context(self.study), FakeProvider([{'updates': [finding()], 'retire': []}]))
        for _ in range(2):
            add(self.study)
        memory.enrich(self.study, context(self.study), FakeProvider([{'updates': [], 'retire': [{'id': 'memory-risk', 'reason': 'Superseded by corrected allocation measurements.'}]}]))
        self.assertEqual(memory.state['findings'], [])
        old = json.loads((self.root/'snapshots/000001.json').read_text())
        self.assertEqual(old['findings'][0]['id'], 'memory-risk')

    def test_config_defaults_overrides_and_validation(self):
        base = Dict(optimizer=Dict(type='llm_generator', llm=Dict(provider='fake', responses=[{}])))
        normalized = _normalize_optimizer_config(base)
        self.assertTrue(normalized.llm.memory.enabled)
        self.assertEqual(normalized.llm.memory.interval_trials, 5)
        base.optimizer.llm.memory = Dict(enabled=False, start_after_trials=15, interval_trials=20)
        normalized = _normalize_optimizer_config(base)
        self.assertFalse(normalized.llm.memory.enabled)
        self.assertEqual(normalized.llm.memory.interval_trials, 20)
        for bad in [{'enabled': 'yes'}, {'interval_trials': 0}, {'start_after_trials': True},
                    {'max_pending_trials': 2}, {'typo': 5}]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                MemoryConfig.from_config(bad)


if __name__ == '__main__':
    unittest.main()
