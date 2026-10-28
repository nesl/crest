"""Synthetic full-history integration regressions through public runtime APIs."""
import json
from dataclasses import replace

import pytest

from crest.optimizers.llm.config import normalize_llm_config
from crest.optimizers.llm.prompt_builder import build_candidate_request, enrich_full_history
from crest.optimizers.llm.component import LLMGeneratorOptimizer
from crest.pipeline_types import SearchContext, BudgetSnapshot, TrialRecord

def synthetic_context(batch_size=5):
    """Build public audio search declarations with invented task/device context.

    Parameters
    ----------
    batch_size : int, optional
        Number of proposals requested from the synthetic context.

    Returns
    -------
    PromptContext
        Full-history context without recorded measurements or private inputs.
    """
    from crest.model_families.audio_dscnn.decode import AUDIO_DSCNN_SEARCH_CHOICES
    from crest.optimizers.search_space import SearchParam, SearchSpaceDescriptor
    from crest.optimizers.llm.prompt_builder import PromptContext
    descriptor = SearchSpaceDescriptor(tuple(
        SearchParam(name, "categorical", choices=choices)
        for name, choices in AUDIO_DSCNN_SEARCH_CHOICES.items()))
    return PromptContext(study_name="synthetic-audio", model_family="audio_dscnn",
        descriptor=descriptor, objective_summary="maximize score subject to device feasibility",
        attempted_trials=0, feasible_completed_trials=0, target_feasible_trials=200,
        max_total_attempts=305, batch_size=batch_size,
        task_context={"task": "synthetic audio classification"},
        device_context={"device": "synthetic MCU"}, require_rationale=True,
        full_history_mode=True)


def synthetic_rows():
    """Return invented architecture observations, without historical study data.

    Returns
    -------
    list[dict]
        Five synthetic trials using legal public architecture choices.
    """
    base = synthetic_context()
    raw = {name: param.choices[0] for name, param in base.descriptor.items()}
    return [{"number": i, "params": {**raw, "base_channels": width},
             "value": 0.5 + i / 100, "state": "COMPLETE", "feasible": True,
             "feasibility_status": "feasible"}
            for i, width in enumerate(base.descriptor["base_channels"].choices[:5])]


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


def test_full_context_orders_evidence_and_omits_memory(tmp_path, monkeypatch):
    # The shared enrichment path emits all evidence without summary calls.
    context = enrich_full_history(synthetic_context(), ())
    request = build_candidate_request(context, prompt_version="full-history-v1")
    payload = json.loads(request.user_prompt)
    assert list(payload) == ["history", "current_request"]
    assert context.allocation == {"local": 0, "exploration": 5}
    assert len(context.exploration_targets) == 5
    assert "knowledge_base" not in request.user_prompt
    assert "MCU" in request.system_prompt or "device_context" in request.system_prompt
    # Full-history exploration optimizes the score, not implicit model smallness.
    assert "not model smallness" in request.system_prompt
    assert "does not require a small or cheap model" in request.system_prompt
    assert "columns" in payload["current_request"]["coverage"]["cells"]
    # Responses JSON mode checks input itself, not just top-level instructions.
    from crest.optimizers.llm.provider import build_provider
    monkeypatch.setenv("CREST_TEST_KEY", "offline")
    provider = build_provider({"provider": "openai_responses", "base_url": "https://api.openai.com/v1",
        "model": "gpt-6-luna", "api_key_env": "CREST_TEST_KEY", "json_response_mode": True})
    for feedback in (None, "Replace the duplicate candidate"):
        wire = json.loads(provider.build_http_request(build_candidate_request(context,
            prompt_version="test", repair_feedback=feedback)).data)
        assert wire["text"]["format"]["type"] == "json_object"
        assert "JSON" in json.loads(wire["input"])["current_request"]["response_instruction"]


def test_unknown_failure_retry_is_enforced_by_real_generation(tmp_path):
    # Unknown failures get one retry, then decoded dedup blocks aliases operationally.
    import random
    from crest.optimizers.llm.enqueue import generate_llm_batch
    from crest.optimizers.llm.provider import FakeProvider
    from crest.optimizers.llm.ledger import LLMLedger
    base = synthetic_context(1)
    raw = {name: param.choices[0] for name, param in base.descriptor.items()}
    raw.update(num_blocks=2, stride_schedule="light")
    alias = {**raw, "stride_schedule": "late"}
    novel = {**raw, "activation": "relu6" if raw["activation"] == "relu" else "relu"}
    history = [TrialRecord(0, "PRUNED", raw, None,
                          {"feasible": False, "feasibility_status": "not_evaluated"}),
               TrialRecord(1, "FAIL", alias, None, {})]
    def candidate(params):
        """Wrap architecture parameters for the fake provider.

        Parameters
        ----------
        params : dict
            Raw architecture.

        Returns
        -------
        dict
            Exploration proposal envelope.
        """
        return {"params": params, "intent": "exploration", "parent_trial": None, "rationale": "Test this design."}
    for count in (1, 2):
        ctx = replace(enrich_full_history(base, history[:count]), exploration_targets=())
        provider = FakeProvider([{"candidates": [candidate(alias)]}, {"candidates": [candidate(novel)]}])
        result = generate_llm_batch(tuple(history[:count]), provider, ctx, LLMLedger(tmp_path / str(count)),
            prompt_version="test", max_repair_attempts=1, max_refill_attempts=2, rng=random.Random(0))
        assert result[0].params == (alias if count == 1 else novel)
        assert len(provider.requests) == count


def test_component_runs_full_policy_with_fake_provider(tmp_path):
    # Exercise normalized config -> initialize -> shared prompt -> real validation.
    base = synthetic_context()
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
    seeds = synthetic_rows()
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
    context = enrich_full_history(replace(synthetic_context(), attempted_trials=302,
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


def test_real_component_300_rows_bounded_refill_and_malformed_repair(tmp_path, monkeypatch):
    from crest.optimizers.llm.full_history import decoded_fingerprint
    from crest.optimizers.llm.provider import build_provider
    # Exercise actual enrichment, real decoding, local parents, and worst-case
    # duplicate feedback rather than injecting a hand-written feedback string.
    base = synthetic_context()
    seeds = synthetic_rows()
    history = []
    for number in range(300):
        seed = seeds[number % len(seeds)]
        attrs = {"task_metrics": {"macro_f1": 0.712345678901234},
                 "latency_ms": 13.1234567890123, "energy_mj_per_inference": 0.123456789012345,
                 "feasibility_status": "feasible" if number < 2 else "infeasible",
                 "feasible": number < 2,
                 "feasibility_reason": "Latency exceeds deployment constraint" if number >= 2 else ""}
        history.append(TrialRecord(number, "COMPLETE", seed["params"], (0.712345678901234,), attrs))
    enriched = enrich_full_history(replace(base, attempted_trials=300, feasible_completed_trials=2), history)
    assert enriched.eligible_parents
    parent = enriched.eligible_parents[0]
    # Choose a valid unused graph through descriptor choices and the real decoder.
    used = {decoded_fingerprint(t.params) for t in history}
    local = dict(parent["params"])
    for name in ("dropout_rate", "activation", "dense_units", "base_channels"):
        for value in base.descriptor[name].choices:
            candidate = {**local, name: value}
            if decoded_fingerprint(candidate) not in used:
                local = candidate
                break
        if decoded_fingerprint(local) not in used:
            break
    assert decoded_fingerprint(local) not in used
    duplicate = {"params": history[0].params, "intent": "exploration", "rationale": "r" * 2000}
    accepted = {"params": local, "intent": "local", "parent_trial": parent["trial_number"],
                "rationale": "r" * 2000, "intended_change": "Probe one decoded graph change."}
    responses = [{"candidates": [accepted] + [duplicate] * 5},
                 {"candidates": ["malformed" * 2000] * 20},
                 {"candidates": [duplicate] * 5}, {"candidates": [duplicate] * 5}]
    config = {"type": "llm_generator", "llm": {"provider": "fake", "responses": responses,
        "history_mode": "full", "require_rationale": True, "memory": {"enabled": False}, "batch_size": 5}}
    optimizer = LLMGeneratorOptimizer()
    optimizer.initialize(SearchContext(base.study_name, tmp_path / "campaign", ("score",),
        ("maximize",), base.objective_summary, base.semantic_context, base.descriptor,
        "audio_dscnn", True), config)
    result = optimizer.propose_round(tuple(history), BudgetSnapshot(300, 2, 200, 305, 5))
    assert len(result.candidates) == 1 and result.candidates[0].params == local
    assert result.candidates[0].provenance["parent_trial"] == parent["trial_number"]
    requests = optimizer.provider.requests
    assert len(requests) == 4
    monkeypatch.setenv("CREST_TEST_KEY", "offline")
    wire_provider = build_provider({"provider": "openai_responses", "base_url": "https://api.openai.com/v1",
        "model": "gpt-6-luna", "api_key_env": "CREST_TEST_KEY", "temperature": None,
        "reasoning_effort": "medium", "max_output_tokens": 8192})
    bounds = []
    for request in requests:
        # Use the same compact secret-free wire body as the spending guard.
        wire = wire_provider.build_http_request(request)
        wire_payload = json.loads(wire.data)
        bound = len(json.dumps(wire_payload, separators=(",", ":")).encode()) + 1024
        bounds.append(bound)
        assert bound <= 100000
    assert enriched.full_history["failure_reason_encoding"] == "dictionary_index"
    reasons = enriched.full_history["failure_reasons"]
    reason_index = enriched.full_history["columns"].index("failure_reason")
    for row in enriched.full_history["rows"][2:]:
        assert reasons[row[reason_index]] == "Latency exceeds deployment constraint"
    for request in requests[1:]:
        feedback = json.loads(json.loads(request.user_prompt)["current_request"]["repair_feedback"])
        assert len(json.dumps(feedback)) < 4000
        assert "rationale" not in feedback["retained_candidates"][0]
        assert feedback["rejections"][0]["code"] == "duplicate_raw"
    repair_feedback = json.loads(json.loads(requests[2].user_prompt)["current_request"]["repair_feedback"])
    assert "repair" in repair_feedback
    assert len(enriched.full_history["rows"]) == 300


def test_coverage_completion_transition_labels_python_policy(tmp_path):
    from crest.optimizers.llm.full_history import candidate_cell
    from crest.optimizers.search_space import SearchParam, SearchSpaceDescriptor
    from crest.optimizers.llm.enqueue import generate_llm_batch
    from crest.optimizers.llm.provider import FakeProvider
    from crest.optimizers.llm.ledger import LLMLedger
    import random
    # Restrict to one real decoded cell: a success plus explicit infeasibility
    # completes coverage; pending reservations alone never complete it.
    base = synthetic_context()
    raw = {name: param.choices[0] for name, param in base.descriptor.items()}
    descriptor = SearchSpaceDescriptor(tuple(SearchParam(name, "categorical", choices=(raw[name],))
        if name in {"num_blocks", "stride_schedule"} else param for name, param in base.descriptor.items()))
    base = replace(base, descriptor=descriptor, batch_size=2, attempted_trials=0,
                   feasible_completed_trials=0)
    altered = {**raw, "activation": base.descriptor["activation"].choices[-1]}
    history = (TrialRecord(0, "COMPLETE", raw, (0.7,), {"feasibility_status": "feasible"}),
               TrialRecord(1, "PRUNED", altered, None, {"feasibility_status": "infeasible"}))
    before = enrich_full_history(base, history[:1])
    assert not before.coverage["goal_complete"] and len(before.exploration_targets) == 1
    after = enrich_full_history(base, history)
    assert after.coverage["goal_complete"] and not after.exploration_targets
    candidate = {**raw, "dropout_rate": base.descriptor["dropout_rate"].choices[-1]}
    provider = FakeProvider([{"candidates": [{"params": candidate, "intent": "exploration", "rationale": "Explore after coverage."}]}])
    result = generate_llm_batch(history, provider, replace(after, allocation={"exploration": 1}, batch_size=1),
        LLMLedger(tmp_path / "ledger"), prompt_version="test", max_repair_attempts=0,
        max_refill_attempts=0, rng=random.Random(0))
    assert result[0].provenance["exploration_policy"] == "post_coverage"
    pending = (TrialRecord(2, "WAITING", raw, None), TrialRecord(3, "RUNNING", altered, None))
    with pytest.raises(ValueError, match="already pending"):
        enrich_full_history(replace(base, batch_size=1), pending)
