# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Memory lifecycle, evidence retention, bounded backlog and resume tests."""
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import optuna
from optuna.trial import TrialState, create_trial

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from crest.optimizers.llm.history import build_terminal_trial_history
from crest.optimizers.llm.memory import ExperimentalMemory, MemoryConfig
from crest.optimizers.llm.prompt_builder import PromptContext, build_candidate_request
from crest.optimizers.llm.provider import FakeProvider
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
        result = memory.enrich(self.study, context(self.study, 1), FakeProvider([{'updates': [bad], 'retire': []}]))
        self.assertEqual((self.root/'state.json').read_bytes(), before)
        self.assertEqual([r['number'] for r in result.pending_trials], [2])
        self.assertEqual(result.knowledge_base['covered_trial_count'], 2)

    def test_oversized_and_incomplete_findings_are_rejected(self):
        for _ in range(2):
            add(self.study)
        for update in [{'id': 'incomplete'}, {**finding(), 'evidence': 'x'*10000}]:
            memory = ExperimentalMemory(self.root, self.config)
            result = memory.enrich(self.study, context(self.study), FakeProvider([{'updates': [update], 'retire': []}]))
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

    def test_failed_provider_retains_evidence_and_backlog_limit_stops_growth(self):
        memory = ExperimentalMemory(self.root, self.config)
        for _ in range(4):
            add(self.study)
        result = memory.enrich(self.study, context(self.study, 1), FakeProvider([]))
        self.assertEqual(len(result.pending_trials), 3)
        add(self.study)
        with self.assertRaisesRegex(RuntimeError, 'backlog'):
            memory.enrich(self.study, context(self.study), FakeProvider([]))

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
