# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Characterize exact proposal evidence across the neutral snapshot boundary."""
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import optuna
from optuna.trial import TrialState, create_trial

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crest.optimizers.history import build_trial_snapshot, freeze_plain_data
from crest.pipeline_types import BudgetSnapshot, SearchContext
from crest.optimizers.llm.component import LLMGeneratorOptimizer
from crest.optimizers.llm.generation import generate_llm_batch
from crest.optimizers.llm.history import (build_recent_trial_history,
    build_terminal_trial_history, build_best_trial_anchors)
from crest.optimizers.llm.ledger import LLMLedger
from crest.optimizers.llm.provider import FakeProvider
from crest.optimizers.context import disabled_semantic_context
from test.test_llm_generation import DESCRIPTOR, context
from crest.optimizers.llm.component import digest


class ProposalContractTests(unittest.TestCase):
    def test_exact_compact_payload_and_distinct_duplicate_anchor_eligibility(self):
        study = optuna.create_study(direction="maximize")
        distribution = {"width": optuna.distributions.IntDistribution(2, 8),
                        "mode": optuna.distributions.CategoricalDistribution(("small", "large"))}
        for width, state, attrs in (
            (4, TrialState.COMPLETE, {"feasibility_status": "feasible", "latency_ms": 4.5}),
            (5, TrialState.COMPLETE, {"pruned": True, "error_code": 42}),
            (6, TrialState.PRUNED, {"prune_reason": "arena"}),
            (7, TrialState.FAIL, {}),
        ):
            study.add_trial(create_trial(state=state,
                value=float(width) if state == TrialState.COMPLETE else None,
                params={"width": width, "mode": "small"}, distributions=distribution,
                user_attrs=attrs, system_attrs={"fail_reason": "build failed"} if state == TrialState.FAIL else {}))
        study.enqueue_trial({"width": 8, "mode": "large"}, user_attrs={
            "crest_proposal": {"proposal_params": {"width": 8, "mode": "large"}}})
        history = build_trial_snapshot(study)
        expected = (
            {"number": 0, "state": "complete", "params": {"width": 4, "mode": "small"},
             "value": 4.0, "directions": ["maximize"], "feasibility_status": "feasible", "latency_ms": 4.5},
            {"number": 1, "state": "complete", "params": {"width": 5, "mode": "small"},
             "value": 5.0, "directions": ["maximize"], "pruned": True},
            {"number": 2, "state": "pruned", "params": {"width": 6, "mode": "small"},
             "prune_reason": "arena", "pruned": True},
            {"number": 3, "state": "fail", "params": {"width": 7, "mode": "small"},
             "failure_reason": "build failed"},
        )
        compact = build_terminal_trial_history(history, directions=("maximize",))
        self.assertEqual(compact, expected)
        self.assertEqual(compact, build_terminal_trial_history(study))
        self.assertEqual(digest(compact), digest(expected))
        self.assertEqual(build_recent_trial_history(history, window_size=3, directions=("maximize",)),
                         build_recent_trial_history(study, window_size=3))
        self.assertEqual(build_best_trial_anchors(history, anchor_count=3, directions=("maximize",)),
                         (expected[0],))
        self.assertEqual(history[-1].params, {})
        self.assertEqual(history[-1].user_attrs["crest_proposal"]["proposal_params"]["width"], 8)
        with self.assertRaises(TypeError):
            history[-1].user_attrs["crest_proposal"]["proposal_params"]["width"] = 3
        provider = FakeProvider([{"candidates": [{"width": w, "mode": "small"} for w in range(4, 9)]}])
        with tempfile.TemporaryDirectory() as tmp:
            proposals = generate_llm_batch(history, provider, context(5), LLMLedger(Path(tmp)),
                                           prompt_version="v1", max_repair_attempts=0, rng=random.Random(0))
            self.assertEqual([p.params["width"] for p in proposals], [7, 8])
            self.assertEqual([p.provenance["batch_index"] for p in proposals], [0, 1])
            self.assertEqual(len(study.trials), 5)
            rows = [json.loads(line) for line in (Path(tmp)/"returned_candidates.jsonl").read_text().splitlines()]
            self.assertTrue(all(r["status"] == "returned_to_runner" and r["schema_version"] == 2 for r in rows))
            self.assertFalse((Path(tmp)/"accepted_candidates.jsonl").exists())

    def test_pareto_and_scalar_ties_preserve_trial_numbers(self):
        for directions, values in ((("maximize",), [(2,), (2,), (1,)]),
                                   (("minimize", "minimize"), [(0, 4), (1, 1), (4, 0), (3, 3)])):
            study = optuna.create_study(directions=directions)
            for value in values:
                study.add_trial(create_trial(values=value))
            history = build_trial_snapshot(study)
            for count in (1, 2, 5):
                self.assertEqual(build_best_trial_anchors(history, anchor_count=count, directions=directions),
                                 build_best_trial_anchors(study, anchor_count=count))

    pass

    def test_component_plain_context_batch_cap_and_disabled_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            search = SearchContext("proposal", Path(tmp), ("score",), ("maximize",), "full score JSON",
                                   freeze_plain_data({"enabled": True, "version": "v1", "hash": "h", "payload": {}}),
                                   DESCRIPTOR, "test_family", False)
            config = {"type": "llm_generator", "llm": {"provider": "fake", "responses": [
                {"candidates": [{"width": 4, "mode": "small"}]}], "batch_size": 5,
                "semantic_context": False, "memory": {"enabled": False}}}
            component = LLMGeneratorOptimizer()
            component.initialize(search, config)
            proposals = component.propose_round((), BudgetSnapshot(0, 0, 10, 12, 1))
            self.assertEqual(proposals.n_trials, 1)
            payload = json.loads(component.provider.requests[0].user_prompt)
            self.assertEqual(payload["model_family"], "test_family")
            self.assertEqual(payload["objective_summary"], "full score JSON")
            self.assertEqual(payload["trial_budget"]["batch_size"], 1)
            self.assertEqual(payload["semantic_context"], {k:v for k,v in disabled_semantic_context().items() if k != "payload"})
            self.assertEqual(component.semantic_context, disabled_semantic_context())
            with self.assertRaisesRegex(ValueError, "descriptor"):
                component.initialize(SearchContext("bad", tmp, (), (), "", {}), config)

    pass

if __name__ == "__main__":
    unittest.main()
