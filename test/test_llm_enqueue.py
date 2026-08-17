# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for validated LLM candidate enqueueing and fallback."""

import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

import optuna

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crest.optimizers.llm.enqueue import enqueue_llm_batch  # noqa: E402
from crest.optimizers.llm.ledger import LLMLedger  # noqa: E402
from crest.optimizers.llm.prompt_builder import PromptContext  # noqa: E402
from crest.optimizers.llm.provider import FakeProvider  # noqa: E402
from crest.optimizers.llm.search_space import SearchParam, SearchSpaceDescriptor  # noqa: E402


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
        """Accepted raw dictionaries flow through Optuna's normal objective path."""
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
            accepted = enqueue_llm_batch(
                study,
                provider,
                context(2),
                ledger,
                prompt_version="v1",
                max_repair_attempts=0,
                rng=random.Random(0),
            )
            study.optimize(objective, n_trials=len(accepted))

            self.assertEqual(accepted, ({"width": 4, "mode": "small"}, {"width": 7, "mode": "large"}))
            self.assertEqual([trial.params for trial in study.trials], list(accepted))
            accepted_lines = (Path(tmpdir) / "llm_optimizer/accepted_candidates.jsonl").read_text()
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
            accepted = enqueue_llm_batch(
                study,
                provider,
                context(1),
                ledger,
                prompt_version="v1",
                max_repair_attempts=1,
                rng=random.Random(0),
            )

            self.assertEqual(accepted, ({"width": 6, "mode": "large"},))
            rejected = json.loads(
                (Path(tmpdir) / "llm_optimizer/rejected_candidates.jsonl").read_text()
            )
            self.assertEqual(rejected["code"], "out_of_range")
            self.assertEqual(len(provider.requests), 2)

    def test_invalid_provider_output_triggers_logged_random_fallback(self) -> None:
        """Exhausted invalid responses still produce one locally valid trial."""
        study = optuna.create_study(direction="maximize")
        provider = FakeProvider(["not-json"])
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "llm_optimizer"
            accepted = enqueue_llm_batch(
                study,
                provider,
                context(1),
                LLMLedger(root),
                prompt_version="v1",
                max_repair_attempts=0,
                rng=random.Random(7),
            )

            self.assertEqual(len(accepted), 1)
            self.assertGreaterEqual(accepted[0]["width"], 2)
            self.assertLessEqual(accepted[0]["width"], 8)
            event = json.loads((root / "optimizer_events.jsonl").read_text())
            self.assertEqual(event["event"], "random_fallback")


if __name__ == "__main__":
    unittest.main()
