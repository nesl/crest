"""Next-study integration and an immutable real-campaign prompt regression."""
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest

from crest.optimizers.llm.config import normalize_llm_config
from crest.optimizers.llm.prompt_builder import build_candidate_request, enrich_full_history
from crest.optimizers.llm.component import LLMGeneratorOptimizer
from crest.pipeline_types import SearchContext, BudgetSnapshot, TrialRecord

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("next_study_smoke", ROOT / "experiments/cs3/next_study_smoke.py")
SMOKE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SMOKE)


def test_saved_current_campaign_prompt_is_byte_identical(tmp_path):
    # Rebuild the actual audit prompt, not a new expected output from edited code.
    saved = json.loads((ROOT / "test/fixtures/cs3_legacy_request_000088.json").read_text())
    payload = json.loads(saved["messages"][1]["content"])
    context_path = tmp_path / "context.jsonl"
    context_path.write_text(json.dumps(payload) + "\n")
    budget = payload["trial_budget"]
    context = replace(SMOKE.load_context(context_path), full_history_mode=False,
        attempted_trials=budget["attempted"], feasible_completed_trials=budget["feasible_completed"],
        batch_size=budget["batch_size"], recent_trials=tuple(payload["recent_trials"]),
        pending_trials=tuple(payload["pending_trials"]), anchors=tuple(payload["anchors"]),
        knowledge_base=payload["knowledge_base"])
    request = build_candidate_request(context, prompt_version=saved["prompt_version"])
    assert request.messages() == saved["messages"]


def test_new_policy_is_explicit_and_identity_separated():
    # Defaults must not add new-policy keys to old study identities.
    base = {"type": "llm_generator", "llm": {"provider": "fake", "responses": [{"candidates": []}]}}
    optimizer = LLMGeneratorOptimizer()
    old = optimizer.identity_config(base)
    assert "history_mode" not in old and "max_refill_attempts" not in old
    new = {"type": "llm_generator", "llm": {**base["llm"], "history_mode": "full",
            "require_rationale": True, "memory": {"enabled": False}}}
    identity = optimizer.identity_config(new)
    assert identity["history_mode"] == "full"
    assert identity["candidate_schema_version"] == 3
    assert identity["max_refill_attempts"] == 2
    new["llm"]["memory"]["enabled"] = True
    with pytest.raises(ValueError, match="memory.enabled=false"):
        normalize_llm_config(new)


def test_full_context_orders_evidence_and_omits_memory(tmp_path):
    # The shared enrichment path emits all evidence without summary calls.
    saved = json.loads((ROOT / "test/fixtures/cs3_legacy_request_000088.json").read_text())
    path = tmp_path / "context.jsonl"
    path.write_text(saved["messages"][1]["content"] + "\n")
    context = enrich_full_history(SMOKE.load_context(path), ())
    request = build_candidate_request(context, prompt_version="full-history-v1")
    payload = json.loads(request.user_prompt)
    assert list(payload) == ["history", "current_request"]
    assert context.allocation == {"local": 0, "exploration": 5}
    assert len(context.exploration_targets) == 5
    assert "knowledge_base" not in request.user_prompt
    assert "MCU" in request.system_prompt or "device_context" in request.system_prompt
    assert "columns" in payload["current_request"]["coverage"]["cells"]


def test_component_runs_full_policy_with_fake_provider(tmp_path):
    # Exercise normalized config -> initialize -> shared prompt -> real validation.
    saved = json.loads((ROOT / "test/fixtures/cs3_legacy_request_000088.json").read_text())
    path = tmp_path / "context.jsonl"
    path.write_text(saved["messages"][1]["content"] + "\n")
    base = SMOKE.load_context(path)
    enriched = enrich_full_history(base, ())
    candidates = []
    for target in enriched.exploration_targets:
        params = {name: param.choices[0] for name, param in base.descriptor.items()}
        params.update(num_blocks=target["num_blocks"], stride_schedule=target["aliases"][0])
        candidates.append({"params": params, "rationale": "Probe the assigned unobserved stride cell.",
                           "intent": "exploration", "parent_trial": None})
    config = {"type": "llm_generator", "llm": {"provider": "fake",
        "responses": [{"candidates": candidates}], "history_mode": "full",
        "require_rationale": True, "memory": {"enabled": False}, "batch_size": 5}}
    optimizer = LLMGeneratorOptimizer()
    optimizer.identity_config(config)
    optimizer.initialize(SearchContext(base.study_name, tmp_path / "campaign", ("score",),
        ("maximize",), base.objective_summary, base.semantic_context, base.descriptor,
        "audio_dscnn", True), config)
    result = optimizer.propose_round((), BudgetSnapshot(0, 0, 200, 300, 5))
    assert len(result.candidates) == 5
    assert optimizer.memory is None
    assert len(optimizer.provider.requests) == 1
    assert all(c.provenance["intent"] == "exploration" for c in result.candidates)


def test_300_columnar_trials_with_pending_and_repair_fit_guard(tmp_path, monkeypatch):
    # Synthetic expansion validates size only; it is not additional study evidence.
    from crest.optimizers.llm.provider import build_provider
    saved = json.loads((ROOT / "test/fixtures/cs3_legacy_request_000088.json").read_text())
    payload = json.loads(saved["messages"][1]["content"])
    path = tmp_path / "context.jsonl"
    path.write_text(json.dumps(payload) + "\n")
    seeds = payload["recent_trials"]
    history = []
    for number in range(302):
        row = seeds[number % len(seeds)]
        attrs = {key: value for key, value in row.items() if key not in {"number", "params", "value", "state"}}
        state, values = ("COMPLETE", (row["value"],)) if number < 300 else ("WAITING", None)
        if number < 300 and number % 43 == 0:
            attrs.update(measurement_failure_class="infrastructure",
                         failure_reason="Saved-size fixture: measurement server connection failed.")
            state, values = "FAIL", None
        history.append(TrialRecord(number, state, row["params"], values, attrs))
    context = enrich_full_history(replace(SMOKE.load_context(path), attempted_trials=302,
                                          feasible_completed_trials=293), history)
    request = build_candidate_request(context, prompt_version="full-history-v1",
        repair_feedback=json.dumps({"reason": "Replace rejected entries only.", "details": "x" * 2600}))
    monkeypatch.setenv("CREST_TEST_KEY", "offline-fixture")
    provider = build_provider({"provider": "openai_responses", "base_url": "https://api.openai.com/v1",
        "model": "gpt-6-luna", "api_key_env": "CREST_TEST_KEY", "temperature": None,
        "reasoning_effort": "medium", "max_output_tokens": 8192})
    wire = provider.build_http_request(request)
    assert len(wire.data) + 1024 <= 100000
    assert len(context.full_history["rows"]) == 300
    assert len(context.full_history["pending"]) == 2
