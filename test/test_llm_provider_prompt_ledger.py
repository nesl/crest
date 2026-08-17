# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the fake provider, prompt policy, and filesystem ledger."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crest.optimizers.llm.ledger import LLMLedger  # noqa: E402
from crest.optimizers.llm.phase_policy import resolve_phase  # noqa: E402
from crest.optimizers.llm.prompt_builder import (  # noqa: E402
    PromptContext,
    build_candidate_request,
)
from crest.optimizers.llm.provider import FakeProvider  # noqa: E402
from crest.optimizers.llm.schemas import CandidateBatch, CandidateRejection  # noqa: E402
from crest.optimizers.llm.search_space import SearchParam, SearchSpaceDescriptor  # noqa: E402


DESCRIPTOR = SearchSpaceDescriptor(
    (
        SearchParam("width", "int", low=2, high=8),
        SearchParam("mode", "categorical", choices=("small", "large")),
    )
)


def prompt_context(*, attempted: int = 3, max_attempts: int = 20) -> PromptContext:
    """Build one representative bounded prompt context."""
    return PromptContext(
        study_name="demo",
        model_family="odom_tcn",
        descriptor=DESCRIPTOR,
        objective_summary="maximize score subject to device feasibility",
        attempted_trials=attempted,
        feasible_completed_trials=2,
        target_feasible_trials=10,
        max_total_attempts=max_attempts,
        batch_size=2,
        task_context={"type": "regression"},
        device_context={"name": "TEST_BOARD"},
        runtime_context={"hil": False},
    )


class PhasePolicyTests(unittest.TestCase):
    """Validate deterministic phase thresholds and remaining budget."""

    def test_phase_thresholds(self) -> None:
        """The 35/50/15 policy uses cumulative boundaries 0.35 and 0.85."""
        self.assertEqual(resolve_phase(0, 100).name, "exploration")
        self.assertEqual(resolve_phase(34, 100).name, "exploration")
        self.assertEqual(resolve_phase(35, 100).name, "specification")
        self.assertEqual(resolve_phase(84, 100).name, "specification")
        self.assertEqual(resolve_phase(85, 100).name, "finalization")
        self.assertEqual(resolve_phase(120, 100).remaining, 0)


class ProviderPromptLedgerTests(unittest.TestCase):
    """Exercise the network-free provider-to-envelope and ledger path."""

    def test_fake_provider_response_parses_as_candidate_batch(self) -> None:
        """Configured fake JSON drives the same Pydantic envelope as a live provider."""
        provider = FakeProvider(
            [{"candidates": [{"width": 4, "mode": "small"}]}],
            model="fixture-model",
        )
        request = build_candidate_request(prompt_context(), prompt_version="v1")

        response = provider.complete_json(request)
        parsed = CandidateBatch.model_validate_json(response.content)

        self.assertEqual(parsed.candidates[0]["width"], 4)
        self.assertEqual(response.provider, "fake")
        self.assertEqual(response.model, "fixture-model")
        self.assertEqual(provider.requests, [request])

    def test_prompt_contains_schema_budget_phase_and_output_contract(self) -> None:
        """Every request exposes raw keys and deterministic trial-budget state."""
        request = build_candidate_request(
            prompt_context(attempted=8, max_attempts=20),
            prompt_version="v1",
        )
        payload = json.loads(request.user_prompt)

        self.assertEqual(payload["study_name"], "demo")
        self.assertEqual(payload["search_space"]["width"], {"kind": "int", "low": 2, "high": 8})
        self.assertEqual(payload["trial_budget"]["attempted"], 8)
        self.assertEqual(payload["trial_budget"]["remaining_attempts"], 12)
        self.assertEqual(payload["trial_budget"]["phase"], "specification")
        self.assertEqual(payload["output_contract"]["exact_keys"], ["width", "mode"])
        self.assertEqual(request.metadata["phase"], "specification")

    def test_ledger_writes_exchange_context_and_candidate_decisions(self) -> None:
        """The minimum reproducibility files are written in stable formats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger = LLMLedger(Path(tmpdir) / "llm_optimizer")
            context = prompt_context()
            request = build_candidate_request(context, prompt_version="v1")
            response = FakeProvider(
                [{"candidates": [{"width": 4, "mode": "small"}]}]
            ).complete_json(request)
            rejection = CandidateRejection(
                index=1,
                code="out_of_range",
                message="width too large",
                candidate={"width": 99, "mode": "small"},
            )

            ledger.write_exchange(1, request, response)
            ledger.record_prompt_context(context.as_prompt_payload())
            ledger.record_accepted({"request_id": 1, "candidate": {"width": 4, "mode": "small"}})
            ledger.record_rejected(rejection)
            ledger.record_event({"event": "batch_complete", "accepted": 1, "rejected": 1})

            root = Path(tmpdir) / "llm_optimizer"
            request_payload = json.loads((root / "requests/000001.request.json").read_text())
            response_payload = json.loads((root / "requests/000001.response.json").read_text())
            rejected_payload = json.loads((root / "rejected_candidates.jsonl").read_text())

            self.assertEqual(request_payload["prompt_version"], "v1")
            self.assertEqual(response_payload["provider"], "fake")
            self.assertEqual(rejected_payload["code"], "out_of_range")
            for filename in (
                "accepted_candidates.jsonl",
                "rejected_candidates.jsonl",
                "optimizer_events.jsonl",
                "prompt_contexts.jsonl",
            ):
                self.assertTrue((root / filename).is_file())


if __name__ == "__main__":
    unittest.main()
