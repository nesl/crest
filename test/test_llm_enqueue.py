# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for generation-only LLM proposals, repair and fallback."""

import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import optuna

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crest.optimizers.history import build_trial_snapshot
from crest.pipeline_types import TrialRecord
from crest.optimizers.llm.enqueue import generate_llm_batch  # noqa: E402
from crest.optimizers.llm.ledger import LLMLedger  # noqa: E402
from crest.optimizers.llm.prompt_builder import PromptContext  # noqa: E402
from crest.optimizers.llm.provider import FakeProvider, LLMResponse, ProviderResponseError  # noqa: E402
from crest.optimizers.search_space import SearchParam, SearchSpaceDescriptor  # noqa: E402


DESCRIPTOR = SearchSpaceDescriptor(
    (
        SearchParam("width", "int", low=2, high=8),
        SearchParam("mode", "categorical", choices=("small", "large")),
    )
)


def context(batch_size: int) -> PromptContext:
    """Build one enqueue-focused prompt context."""
    return PromptContext(
        study_name="enqueue-smoke",
        model_family="test_family",
        descriptor=DESCRIPTOR,
        objective_summary="maximize width",
        attempted_trials=0,
        feasible_completed_trials=0,
        target_feasible_trials=batch_size,
        max_total_attempts=4,
        batch_size=batch_size,
    )


def objective(trial: optuna.Trial) -> float:
    """Consume the exact raw fields enqueued by the candidate generator."""
    width = trial.suggest_int("width", 2, 8)
    trial.suggest_categorical("mode", ("small", "large"))
    return float(width)


class EnqueueTests(unittest.TestCase):
    """Exercise valid, repaired, and random-fallback enqueue paths."""

    def test_fake_provider_enqueues_two_trials_for_unchanged_objective(self) -> None:
        """Generation leaves the study untouched; returned values retain execution parity."""
        study = optuna.create_study(direction="maximize")
        provider = FakeProvider(
            [
                {
                    "candidates": [
                        {"width": 4, "mode": "small"},
                        {"width": 7, "mode": "large"},
                    ]
                }
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger = LLMLedger(Path(tmpdir) / "llm_optimizer")
            accepted = generate_llm_batch(
                build_trial_snapshot(study),
                provider,
                context(2),
                ledger,
                prompt_version="v1",
                max_repair_attempts=0,
                rng=random.Random(0),
            )
            self.assertEqual(len(study.trials), 0)
            params = tuple(proposal.params for proposal in accepted)
            for proposal in accepted:
                study.enqueue_trial(dict(proposal.params))
            study.optimize(objective, n_trials=len(accepted))

            self.assertEqual(params, ({"width": 4, "mode": "small"}, {"width": 7, "mode": "large"}))
            self.assertEqual([trial.params for trial in study.trials], list(params))
            accepted_lines = (Path(tmpdir) / "llm_optimizer/returned_candidates.jsonl").read_text()
            self.assertEqual(len(accepted_lines.splitlines()), 2)

    def test_invalid_first_response_can_be_repaired(self) -> None:
        """A later provider response is tried after structured rejection."""
        study = optuna.create_study(direction="maximize")
        provider = FakeProvider(
            [
                {"candidates": [{"width": 99, "mode": "small"}]},
                {"candidates": [{"width": 6, "mode": "large"}]},
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger = LLMLedger(Path(tmpdir) / "llm_optimizer")
            accepted = generate_llm_batch(
                build_trial_snapshot(study),
                provider,
                context(1),
                ledger,
                prompt_version="v1",
                max_repair_attempts=1,
                rng=random.Random(0),
            )

            self.assertEqual(tuple(p.params for p in accepted), ({"width": 6, "mode": "large"},))
            rejected = json.loads(
                (Path(tmpdir) / "llm_optimizer/rejected_candidates.jsonl").read_text()
            )
            self.assertEqual(rejected["code"], "out_of_range")
            self.assertEqual(len(provider.requests), 2)
            repaired_payload = json.loads(provider.requests[1].user_prompt)
            self.assertIn("out_of_range", repaired_payload["repair_feedback"])
            prompt_rows = [json.loads(line) for line in (
                Path(tmpdir) / "llm_optimizer/prompt_contexts.jsonl"
            ).read_text().splitlines()]
            self.assertIsNone(prompt_rows[0]["repair_feedback"])
            self.assertEqual(prompt_rows[1]["repair_feedback"], repaired_payload["repair_feedback"])

    def test_terminal_proposals_and_waiting_fixed_params_are_deduplicated(self) -> None:
        """Partial terminal params retain identity; infrastructure and unknown FAIL retry."""
        study = optuna.create_study(direction="maximize")
        study.enqueue_trial({"width": 7, "mode": "small"})
        history = (
            TrialRecord(1, "PRUNED", {"width": 4}, None, {
                "crest_proposal": {"proposal_params": {"width": 4, "mode": "small"}}}),
            TrialRecord(2, "FAIL", {}, None, {
                "measurement_failure_class": "deterministic",
                "crest_proposal": {"proposal_params": {"width": 5, "mode": "small"}}}),
            TrialRecord(3, "COMPLETE", {"width": 6, "mode": "small"}, (0.0,), {
                "measurement_failure_class": "infrastructure"}),
            TrialRecord(4, "FAIL", {"width": 8, "mode": "small"}, None),
            TrialRecord(5, "FAIL", {"width": 6, "mode": "small"}, None, {
                "measurement_failure_class": "infrastructure", "pruned": True}),
            TrialRecord(6, "PRUNED", {"width": 6, "mode": "small"}, None, {
                "measurement_failure_class": "infrastructure"}),
            *build_trial_snapshot(study),
        )
        provider = FakeProvider([{"candidates": [
            {"width": width, "mode": "small"} for width in range(4, 9)
        ]}])
        with tempfile.TemporaryDirectory() as tmpdir:
            accepted = generate_llm_batch(
                history, provider, context(5), LLMLedger(Path(tmpdir)),
                prompt_version="v1", max_repair_attempts=0, rng=random.Random(0),
            )
            self.assertEqual([p.params["width"] for p in accepted], [6, 8])
            rejected = [json.loads(line) for line in (
                Path(tmpdir) / "rejected_candidates.jsonl"
            ).read_text().splitlines()]
            self.assertEqual([row["code"] for row in rejected], [
                "duplicate_completed", "duplicate_completed", "duplicate_queued",
            ])

    def test_random_fallback_avoids_terminal_and_queued_candidates(self) -> None:
        """Fallback uses the same terminal and queued duplicate policy as the LLM."""
        history = (
            TrialRecord(0, "PRUNED", {"width": 4, "mode": "small"}, None),
            TrialRecord(1, "FAIL", {"width": 5, "mode": "small"}, None,
                        {"measurement_failure_class": "deterministic"}),
            TrialRecord(2, "WAITING", {}, None, system_attrs={
                "fixed_params": {"width": 6, "mode": "small"}}),
        )
        choices = [{"width": width, "mode": "small"} for width in range(4, 8)]
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "crest.optimizers.llm.enqueue.sample_random_candidate", side_effect=choices
        ) as sample:
            accepted = generate_llm_batch(
                history, FakeProvider(["not-json"]), context(1), LLMLedger(Path(tmpdir)),
                prompt_version="v1", max_repair_attempts=0, rng=random.Random(0),
            )
            self.assertEqual(accepted[0].params, choices[-1])
            self.assertEqual(sample.call_count, 4)

    def test_provider_execution_errors_abort_without_content_repair_or_fallback(self) -> None:
        """Provider execution failures remain visible after their ledger event."""
        provider = Mock()
        failure = RuntimeError("provider unavailable")
        provider.complete_json.side_effect = failure
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with self.assertRaises(RuntimeError) as raised:
                generate_llm_batch(
                    (), provider, context(1), LLMLedger(root),
                    prompt_version="v1", max_repair_attempts=2, rng=random.Random(0),
                )
            self.assertIs(raised.exception, failure)
            self.assertEqual(provider.complete_json.call_count, 1)
            events = [json.loads(line) for line in (root / "optimizer_events.jsonl").read_text().splitlines()]
            self.assertEqual([row["event"] for row in events], ["provider_error"])
            self.assertFalse((root / "returned_candidates.jsonl").exists())

    def test_malformed_provider_envelope_can_be_repaired(self) -> None:
        """Malformed provider content retains the bounded content-repair path."""
        provider = Mock()
        provider.complete_json.side_effect = [
            ProviderResponseError("invalid chat response"),
            LLMResponse(json.dumps({"candidates": [{"width": 4, "mode": "small"}]}),
                        "fake", "fake", {}),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            accepted = generate_llm_batch(
                (), provider, context(1), LLMLedger(Path(tmpdir)),
                prompt_version="v1", max_repair_attempts=1, rng=random.Random(0),
            )
            self.assertEqual(accepted[0].params, {"width": 4, "mode": "small"})
            self.assertEqual(provider.complete_json.call_count, 2)

    def test_invalid_provider_output_triggers_logged_random_fallback(self) -> None:
        """Exhausted invalid responses still produce one locally valid trial."""
        study = optuna.create_study(direction="maximize")
        provider = FakeProvider(["not-json"])
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "llm_optimizer"
            accepted = generate_llm_batch(
                build_trial_snapshot(study),
                provider,
                context(1),
                LLMLedger(root),
                prompt_version="v1",
                max_repair_attempts=0,
                rng=random.Random(7),
            )

            self.assertEqual(len(accepted), 1)
            self.assertGreaterEqual(accepted[0].params["width"], 2)
            self.assertLessEqual(accepted[0].params["width"], 8)
            event = json.loads((root / "optimizer_events.jsonl").read_text())
            self.assertEqual(event["event"], "random_fallback")
            self.assertEqual(event["status"], "returned_to_runner")
            self.assertEqual(accepted[0].provenance["proposal_source"], "random_fallback")
            self.assertEqual(len(study.trials), 0)


if __name__ == "__main__":
    unittest.main()
