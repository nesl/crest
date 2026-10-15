# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Portable hardware-free smoke checks against a miniature saved SQLite study."""
import argparse
import builtins
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
from unittest.mock import Mock

import pytest

from crest.audio_dscnn_decode import AUDIO_DSCNN_SEARCH_CHOICES
from crest.optimizers.llm.provider import SpendLimitExceeded

SCRIPT = Path(__file__).resolve().parents[1] / "experiments/cs3/next_study_smoke.py"
spec = importlib.util.spec_from_file_location("cs3_next_study_smoke", SCRIPT)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


@pytest.fixture
def saved(tmp_path):
    """Create a minimal real Optuna-shaped snapshot and semantic context.

    Parameters
    ----------
    tmp_path : Path
        Test artifact directory.

    Returns
    -------
    argparse.Namespace
        Arguments defaulting to offline fixture mode.
    """
    db = tmp_path / "saved.db"
    with sqlite3.connect(db) as c:
        c.executescript("""
            CREATE TABLE studies(study_id INTEGER,study_name TEXT);
            CREATE TABLE trials(trial_id INTEGER,number INTEGER,study_id INTEGER,state TEXT);
            CREATE TABLE trial_params(trial_id INTEGER,param_name TEXT,param_value REAL,distribution_json TEXT);
            CREATE TABLE trial_values(trial_id INTEGER,objective INTEGER,value REAL,value_type TEXT);
            CREATE TABLE trial_user_attributes(trial_id INTEGER,key TEXT,value_json TEXT);
            CREATE TABLE trial_system_attributes(trial_id INTEGER,key TEXT,value_json TEXT);
            INSERT INTO studies VALUES(1,'saved-study');
            INSERT INTO trials VALUES(10,0,1,'COMPLETE');
            INSERT INTO trials VALUES(11,1,1,'FAIL');
            INSERT INTO trial_values VALUES(10,0,0.7,'FINITE');
            INSERT INTO trial_user_attributes VALUES(10,'feasible','true');
            INSERT INTO trial_user_attributes VALUES(10,'feasibility_status','"feasible"');
        """)
        for name, choices in AUDIO_DSCNN_SEARCH_CHOICES.items():
            c.execute("INSERT INTO trial_params VALUES(10,?,?,?)", (name, 0,
                json.dumps({"name": "CategoricalDistribution", "attributes": {"choices": list(choices)}})))
    context = tmp_path / "prompt_contexts.jsonl"
    payload = {"study_name": "saved-study", "model_family": "audio_dscnn",
        "objective_summary": "maximize measured score", "search_space": {
            name: {"kind": "categorical", "choices": list(choices), "description": "saved semantics"}
            for name, choices in AUDIO_DSCNN_SEARCH_CHOICES.items()},
        "trial_budget": {"target_feasible": 200, "max_total_attempts": 300},
        "semantic_context": {"enabled": True, "version": "saved-v1", "hash": "saved-hash", "guidance": ["measured evidence first"]},
        "device_context": {"mcu": "saved-real-device", "ram_bytes": 123456},
        "task_context": {"metric": "macro_f1"}}
    context.write_text(json.dumps(payload) + "\n")
    return argparse.Namespace(history_db=db, prompt_context=context, output=tmp_path / "out",
        trials=None, batch_size=5, render_only=False, run_paid=False)


def test_render_never_calls_network_or_imports_training(saved, monkeypatch):
    # Purpose: rendering must remain safe without credentials or hardware runtime.
    saved.render_only = True
    original_import = builtins.__import__
    def no_training(name, *args, **kwargs):
        if name.split(".")[0] in {"tensorflow", "keras", "serial"}:
            raise AssertionError("Training/hardware import forbidden")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_training)
    monkeypatch.setattr(smoke, "generate_llm_batch", Mock(side_effect=AssertionError("generation forbidden")))
    before = hashlib.sha256(saved.history_db.read_bytes()).hexdigest()
    report = smoke.run(saved)
    assert report["status"] == "rendered"
    assert report["content_calls"] == 0
    assert report["input_guard_bound"] < 120000
    assert report["spending"]["charged_usd"] == "0"
    assert hashlib.sha256(saved.history_db.read_bytes()).hexdigest() == before
    wire = json.loads((saved.output / "wire_payload.json").read_text())
    assert wire["prompt_cache_options"] == {"mode": "explicit"}
    assert "saved-real-device" in wire["instructions"]
    assert "knowledge_base" not in wire["instructions"]


def test_offline_uses_production_policy_and_responses_wire(saved):
    # Purpose: generation must validate real quotas/targets and retain rationale/diffs.
    before = saved.history_db.read_bytes()
    report = smoke.run(saved)
    assert report["status"] == "passed"
    assert report["accepted_count"] == 5
    assert not report["fallback"]
    assert report["content_calls"] == 1
    assert report["cache_reads"] == report["cache_writes"] == 0
    assert saved.history_db.read_bytes() == before
    local = [p for p in report["accepted"] if p["provenance"]["intent"] == "local"]
    assert len(local) == report["allocation"]["local"]
    assert all(p["provenance"]["rationale"] for p in report["accepted"])
    assert all(p["provenance"]["parent_differences"]["decoded"] for p in local)
    response = json.loads(next((saved.output / "requests").glob("*.response.json")).read_text())
    assert response["raw_response"]["status"] == "completed"
    assert response["metadata"]["wire_payload"]["prompt_cache_options"] == {"mode": "explicit"}


def test_readonly_prefix_and_categorical_external_types(saved):
    # Purpose: saved parameter aliases/types survive direct read-only reconstruction.
    history = smoke.read_history(saved.history_db, "saved-study", 1)
    assert len(history) == 1
    assert history[0].number == 0
    assert isinstance(history[0].params["norm_flag"], bool)
    assert history[0].values == (0.7,)
    with pytest.raises(TypeError):
        history[0].params["num_blocks"] = 99
    with pytest.raises(sqlite3.OperationalError):
        smoke.read_history(saved.history_db.parent / "absent.db", "saved-study")


def test_hard_four_call_bound():
    # Purpose: even unexpected generation behavior cannot exceed authorized calls.
    delegate = Mock()
    bounded = smoke.LimitedProvider(delegate)
    for _ in range(4):
        bounded.complete_json("fixture")
    with pytest.raises(SpendLimitExceeded):
        bounded.complete_json("fifth")
    assert delegate.complete_json.call_count == 4


def test_existing_output_is_never_overwritten(saved):
    # Purpose: repeat commands cannot replace previous review artifacts.
    saved.output.mkdir()
    with pytest.raises(FileExistsError):
        smoke.run(saved)


def test_guard_exhaustion_is_reported_without_calls(saved, monkeypatch):
    # Purpose: over-limit history/context produces a review artifact, never a request.
    payload = json.loads(saved.prompt_context.read_text())
    payload["task_context"]["large_evidence"] = "x" * 120000
    saved.prompt_context.write_text(json.dumps(payload) + "\n")
    monkeypatch.setattr(smoke, "generate_llm_batch", Mock(side_effect=AssertionError("generation forbidden")))
    report = smoke.run(saved)
    assert report["status"] == "inconclusive"
    assert "120000 guard" in report["error"]
    assert report["content_calls"] == 0
    assert (saved.output / "report.json").exists()


@pytest.mark.parametrize("source", ["llm", "random_fallback"])
def test_partial_and_fallback_batches_are_inconclusive(saved, monkeypatch, source):
    # Purpose: a smoke cannot call fallback or a partially filled batch a success.
    from crest.pipeline_types import CandidateProposal
    monkeypatch.setattr(smoke, "generate_llm_batch", lambda *a, **k: (
        CandidateProposal({}, {"proposal_source": source}),))
    report = smoke.run(saved)
    assert report["status"] == "inconclusive"
    assert report["accepted_count"] == 1
    assert report["fallback"] == (source == "random_fallback")


def test_cache_probe_offline_never_claims_confirmation(saved):
    # Purpose: simulated transport must not establish a real API cache hit.
    saved.test_cache = True
    report = smoke.run(saved)
    assert report["status"] == "inconclusive"
    assert report["cache_result"] == "simulated_only"
    assert not report["cache_confirmed"]
    assert report["content_calls"] == report["http_attempts"] == 2
    assert report["http_attempt_limit"] == 4
    responses = [json.loads(p.read_text()) for p in sorted((saved.output / "requests").glob("*.response.json"))]
    wires = [r["metadata"]["wire_payload"] for r in responses]
    assert wires[0] == json.loads((saved.output / "wire_payload.json").read_text())
    assert "For this cache diagnostic only" in wires[0]["input"][0]["content"][0]["text"]
    assert wires[0]["input"][0] == wires[1]["input"][0]
    assert wires[0]["input"][1] != wires[1]["input"][1]
    assert wires[0]["prompt_cache_key"] == wires[1]["prompt_cache_key"]
    assert wires[0]["prompt_cache_options"] == {"mode": "explicit", "ttl": "30m"}
    assert wires[0]["input"][0]["content"][0]["prompt_cache_breakpoint"] == {"mode": "explicit"}


@pytest.mark.parametrize("cached_tokens,expected", [(0, "inconclusive"), (1280, "passed")])
def test_paid_cache_probe_requires_second_response_api_hit(saved, monkeypatch, cached_tokens, expected):
    # Purpose: only API-reported reads on the second request confirm caching.
    import io
    saved.test_cache = True
    saved.run_paid = True
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    original_build = smoke.build_provider
    calls = []
    def build(*args, **kwargs):
        provider = original_build(*args, **kwargs)
        def transport(request, timeout=None):
            raw = json.loads(smoke.cache_fixture_transport(request).read())
            raw["usage"]["input_tokens_details"]["cached_tokens"] = cached_tokens if calls else 0
            calls.append(request)
            return io.BytesIO(json.dumps(raw).encode())
        provider._urlopen = transport
        return provider
    monkeypatch.setattr(smoke, "build_provider", build)
    report = smoke.run(saved)
    assert len(calls) == 2
    assert report["status"] == expected
    assert report["cache_confirmed"] == (cached_tokens > 0)


def test_missing_cache_writes_remain_unknown_with_confirmed_reads(saved, monkeypatch):
    # Purpose: a real read can confirm reuse while absent write evidence stays unknown.
    import io
    saved.test_cache = True
    saved.run_paid = True
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    original_build = smoke.build_provider

    def build(*args, **kwargs):
        provider = original_build(*args, **kwargs)

        def transport(request, timeout=None):
            raw = json.loads(smoke.cache_fixture_transport(request).read())
            raw["usage"]["input_tokens_details"] = {"cached_tokens": 1280}
            return io.BytesIO(json.dumps(raw).encode())

        provider._urlopen = transport
        return provider

    monkeypatch.setattr(smoke, "build_provider", build)
    report = smoke.run(saved)
    assert report["status"] == "passed"
    assert report["cache_confirmed"]
    assert report["cache_writes"] is None
    assert report["cache_reads"] == 2560
    assert report["cache_evidence_status"] == "incomplete"
    assert not report["no_cache_writes_confirmed"]


def test_input_guard_reservation_preserves_five_cent_cap(saved, monkeypatch):
    from decimal import Decimal
    original = smoke.SpendTracker
    trackers = []
    def capture(*args, **kwargs):
        tracker = original(*args, **kwargs)
        trackers.append(tracker)
        return tracker
    monkeypatch.setattr(smoke, "SpendTracker", capture)
    saved.render_only = True
    smoke.run(saved)
    assert trackers[0].max_input_tokens == 120000
    assert trackers[0].max_usd == Decimal("0.05")
    assert trackers[0].reservation_usd == Decimal("0.019096")
