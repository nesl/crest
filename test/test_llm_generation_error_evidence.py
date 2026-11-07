# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Provider failures remain auditable after generation writes ledger artifacts."""

import io
import json
import random

import pytest

from crest.optimizers.llm.generation import generate_llm_batch
from crest.optimizers.llm.ledger import LLMLedger
from crest.optimizers.llm.prompting.prompt_builder import PromptContext
from crest.optimizers.llm.providers.provider import build_provider
from crest.optimizers.llm.spending import SpendTracker
from crest.optimizers.search_space import SearchParam, SearchSpaceDescriptor


@pytest.mark.parametrize("raw", [
    {"status": "failed", "output": []},
    {"status": "completed", "output": []},
])
def test_responses_failure_evidence_survives_generation_ledger(tmp_path, monkeypatch, raw):
    """Paid invalid envelopes retain exact wire, usage, and settlement on disk."""
    monkeypatch.setenv("CREST_LEDGER_TEST_KEY", "offline-credential-never-persist")
    settings = dict(
        provider="openai_responses", model="offline-model",
        base_url="https://example.invalid/v1", api_key_env="CREST_LEDGER_TEST_KEY",
        temperature=None, max_output_tokens=100, max_input_tokens=100000,
        output_token_parameter="max_output_tokens",
        max_usd="1", input_usd_per_million_tokens="1",
        output_usd_per_million_tokens="2", pricing_source="offline test prices",
    )
    tracker = SpendTracker(tmp_path, settings)
    provider = build_provider(settings, spend_tracker=tracker)
    raw = {**raw, "model": "offline-model", "usage": {
        "input_tokens": 30, "output_tokens": 5,
        "input_tokens_details": {"cached_tokens": 10, "cache_write_tokens": 2},
    }}
    requests = []

    def transport(request, timeout):
        requests.append(json.loads(request.data))
        return io.BytesIO(json.dumps(raw).encode())

    provider._urlopen = transport
    context = PromptContext(
        study_name="error-ledger", model_family="test",
        descriptor=SearchSpaceDescriptor((SearchParam("width", "int", low=2, high=8),)),
        objective_summary="maximize width", attempted_trials=0,
        feasible_completed_trials=0, target_feasible_trials=1,
        max_total_attempts=4, batch_size=1,
    )
    proposals = generate_llm_batch(
        (), provider, context, LLMLedger(tmp_path), prompt_version="v1",
        max_repair_attempts=0, rng=random.Random(0),
    )
    assert proposals[0].provenance["proposal_source"] == "random_fallback"
    saved = json.loads((tmp_path / "requests/000001.response.json").read_text())
    assert saved["error_type"] == "ProviderResponseError"
    assert saved["raw_response"] == raw
    assert saved["wire_payload"] == requests[0]
    assert saved["usage"]["prompt_tokens"] == 30
    assert saved["usage"]["cached_tokens"] == 10
    assert saved["usage"]["cache_write_tokens"] == 2
    assert saved["spending"]["charged_usd"] == "0.00004"
    assert saved["response_hash"]
    for artifact in tmp_path.rglob("*.json*"):
        assert "offline-credential-never-persist" not in artifact.read_text()
        assert "Authorization" not in artifact.read_text()


def test_error_ledger_excludes_unapproved_exception_attributes(tmp_path):
    """Persist only explicit provider evidence, even on execution failures."""
    error = RuntimeError("provider unavailable")
    error.wire_payload = {"model": "offline-model", "input": "safe prompt"}
    error.raw_response = {"error": {"code": "unavailable"}}
    error.usage = None
    error.spending = {"charged_usd": "0.01"}
    error.headers = {"Authorization": "Bearer must-not-be-written"}
    LLMLedger(tmp_path).write_provider_error(1, error)
    artifact = (tmp_path / "requests/000001.response.json").read_text()
    saved = json.loads(artifact)
    assert saved["wire_payload"] == error.wire_payload
    assert saved["raw_response"] == error.raw_response
    assert saved["usage"] is None
    assert saved["spending"] == error.spending
    assert saved["error_type"] == "RuntimeError"
    assert saved["error"] == "provider unavailable"
    assert "headers" not in saved
    assert "must-not-be-written" not in artifact
