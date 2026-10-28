# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Offline architecture rationale contract and provenance regression tests."""

import random
from dataclasses import replace

import pytest

from crest.optimizers.llm.enqueue import generate_llm_batch
from crest.optimizers.llm.ledger import LLMLedger
from crest.optimizers.llm.provider import FakeProvider
from crest.optimizers.llm.schemas import CandidateBatch, validate_candidate_batch
from test.test_llm_enqueue import DESCRIPTOR, context


@pytest.mark.parametrize("candidate", [
    {"width": 4, "mode": "small"},
    {"params": {"width": 4, "mode": "small"}},
    {"params": {"width": 4, "mode": "small"}, "rationale": "  "},
    {"params": {"width": 4, "mode": "small"}, "rationale": 42},
    {"params": {"width": 4, "mode": "small"}, "rationale": "x" * 2001},
    {"params": {"width": 4, "mode": "small"}, "rationale": "useful", "extra": 1},
])
def test_required_rationale_rejects_malformed_envelopes(candidate):
    """Malformed explanations trigger repair rather than enter trial parameters."""
    result = validate_candidate_batch(CandidateBatch(candidates=[candidate]), DESCRIPTOR,
                                      batch_size=1, require_rationale=True)
    assert not result.accepted
    assert result.rejected[0].code == "invalid_rationale_envelope"


def test_rationale_survives_partial_rejection_and_ledger(tmp_path):
    """Keep each accepted explanation attached after earlier rejection."""
    provider = FakeProvider([{"candidates": [
        {"params": {"width": 99, "mode": "small"}, "rationale": "invalid design"},
        {"params": {"width": 4, "mode": "small"}, "rationale": " Test a compact design. "},
    ]}])
    prompt = replace(context(2), require_rationale=True)
    ledger = LLMLedger(tmp_path)
    proposals = generate_llm_batch((), provider, prompt, ledger, prompt_version="v1",
                                  max_repair_attempts=0, rng=random.Random(0))
    assert len(proposals) == 1
    assert proposals[0].params == {"width": 4, "mode": "small"}
    assert proposals[0].provenance["rationale"] == "Test a compact design."
    assert "Test a compact design." in (tmp_path / "returned_candidates.jsonl").read_text()
    contract = prompt.as_prompt_payload()["output_contract"]
    assert set(contract["format"]["candidates"][0]) == {"params", "rationale"}


def test_repair_can_supply_missing_rationale(tmp_path):
    """Missing rationale is repairable and fallback does not fabricate one."""
    provider = FakeProvider([
        {"candidates": [{"width": 4, "mode": "small"}]},
        {"candidates": [{"params": {"width": 4, "mode": "small"},
                          "rationale": "Explore smaller memory demand."}]},
    ])
    proposals = generate_llm_batch((), provider, replace(context(1), require_rationale=True),
                                  LLMLedger(tmp_path), prompt_version="v1",
                                  max_repair_attempts=1, rng=random.Random(0))
    assert proposals[0].provenance["rationale"] == "Explore smaller memory demand."


def test_fallback_has_no_invented_rationale(tmp_path):
    """Random fallback remains explicitly attributed without model explanation."""
    proposals = generate_llm_batch((), FakeProvider([{"candidates": [{"width": 4}]}]),
                                  replace(context(1), require_rationale=True), LLMLedger(tmp_path),
                                  prompt_version="v1", max_repair_attempts=0, rng=random.Random(0))
    assert proposals[0].provenance["proposal_source"] == "random_fallback"
    assert "rationale" not in proposals[0].provenance


def test_rationale_identity_is_opt_in():
    """Version only the enabled contract, preserving existing study signatures."""
    from crest.optimizers.llm.component import LLMGeneratorOptimizer
    from crest.optimizers.llm.config import normalize_llm_config

    config = {"type": "llm_generator", "llm": {"provider": "fake", "responses": [{}]}}
    legacy = LLMGeneratorOptimizer().identity_config(config)
    assert "require_rationale" not in legacy
    assert legacy["candidate_schema_version"] == 1
    config["llm"]["require_rationale"] = True
    active = LLMGeneratorOptimizer().identity_config(config)
    assert active["require_rationale"] is True
    assert active["candidate_schema_version"] == 2
    config["llm"]["require_rationale"] = "yes"
    with pytest.raises(ValueError, match="require_rationale"):
        normalize_llm_config(config)


def test_duplicate_identity_uses_parameters_only():
    """Different explanations cannot bypass duplicate architecture rejection."""
    batch = CandidateBatch(candidates=[
        {"params": {"width": 4, "mode": "small"}, "rationale": "First hypothesis."},
        {"params": {"width": 4, "mode": "small"}, "rationale": "Another hypothesis."},
    ])
    result = validate_candidate_batch(batch, DESCRIPTOR, batch_size=2, require_rationale=True)
    assert result.accepted == ({"width": 4, "mode": "small"},)
    assert result.rationales == ("First hypothesis.",)
    assert result.rejected[0].code == "duplicate_batch"
