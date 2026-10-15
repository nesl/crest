#!/usr/bin/env python3
# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Hardware-free next-study render/offline smoke; paid requests require --run-paid."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import random
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from crest.optimizer_history import freeze_plain_data, to_plain
from crest.pipeline_types import TrialRecord
from crest.search_space import SearchParam, SearchSpaceDescriptor
from crest.optimizers.llm.enqueue import generate_llm_batch, sample_random_candidate
from crest.optimizers.llm.full_history import decoded_fingerprint, reconstruct_params, outcome_class
from crest.optimizers.llm.ledger import LLMLedger
from crest.optimizers.llm.prompt_builder import PromptContext, build_candidate_request, enrich_full_history
from crest.optimizers.llm.provider import build_provider, LLMRequest, ProviderError, SpendLimitExceeded
from crest.optimizers.llm.spending import SpendTracker


def read_history(path, study_name, count=None):
    """Read a historical prefix through SQLite's enforced read-only connection.

    Parameters
    ----------
    path : Path
        Saved Optuna SQLite database.
    study_name : str
        Exact saved study identity.
    count : int or None
        Number of earliest trials, including nonterminal trials.

    Returns
    -------
    tuple[TrialRecord, ...]
        Immutable evidence ordered by trial number.
    """
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        found = connection.execute("SELECT study_id FROM studies WHERE study_name=?", (study_name,)).fetchone()
        if found is None:
            raise ValueError(f"Study {study_name!r} is absent from saved database")
        trials = connection.execute("SELECT trial_id,number,state FROM trials WHERE study_id=? ORDER BY number", found).fetchall()
        if count is not None:
            trials = trials[:count]
        records = []
        for trial_id, number, state in trials:
            params = {}
            for name, value, distribution in connection.execute(
                    "SELECT param_name,param_value,distribution_json FROM trial_params WHERE trial_id=?", (trial_id,)):
                distribution = json.loads(distribution)
                kind, attributes = distribution["name"], distribution["attributes"]
                if kind == "CategoricalDistribution":
                    value = attributes["choices"][int(value)]
                elif kind.startswith("Int"):
                    value = int(value)
                params[name] = value
            attrs = []
            for table in ("trial_user_attributes", "trial_system_attributes"):
                attrs.append(dict((key, json.loads(value)) for key, value in connection.execute(
                    f"SELECT key,value_json FROM {table} WHERE trial_id=?", (trial_id,))))
            values = connection.execute("SELECT value,value_type FROM trial_values WHERE trial_id=? ORDER BY objective", (trial_id,)).fetchall()
            values = tuple(value if kind == "FINITE" else None for value, kind in values) if values else None
            records.append(TrialRecord(number, state, freeze_plain_data(params), values,
                                       freeze_plain_data(attrs[0]), freeze_plain_data(attrs[1])))
    return tuple(records)


def load_context(path, batch_size=5):
    """Recover real semantic MCU/task context and raw declarations from the ledger.

    Parameters
    ----------
    path : Path
        Saved candidate prompt_contexts.jsonl.
    batch_size : int
        Requested proposal count.

    Returns
    -------
    PromptContext
        Production context awaiting historical enrichment.
    """
    saved = None
    for line in Path(path).read_text().splitlines():
        candidate = json.loads(line)
        if "search_space" in candidate or "fixed_context" in candidate:
            saved = candidate
    if saved is None:
        raise ValueError("No candidate context found in prompt-context ledger")
    fixed = saved.get("fixed_context", saved)
    budget = saved.get("current_request", saved)["trial_budget"]
    descriptor = SearchSpaceDescriptor(tuple(SearchParam(name=name, kind=spec["kind"],
        low=spec.get("low"), high=spec.get("high"),
        choices=tuple(spec["choices"]) if "choices" in spec else None)
        for name in fixed.get("output_contract", {}).get("exact_keys", list(fixed["search_space"]))
        for spec in [fixed["search_space"][name]]))
    metadata = fixed.get("semantic_context", {})
    semantic = {}
    if metadata:
        semantic = {**metadata, "payload": {
            **{key: fixed[key] for key in ("dataset_context", "task_context", "model_context", "device_context", "runtime_context") if key in fixed},
            "guidance": metadata.get("guidance", []),
            "parameter_semantics": {name: {key: value for key, value in spec.items()
                if key not in {"kind", "low", "high", "choices"}}
                for name, spec in fixed["search_space"].items()},
        }}
    return PromptContext(study_name=fixed["study_name"], model_family=fixed["model_family"],
        descriptor=descriptor, objective_summary=fixed["objective_summary"],
        attempted_trials=0, feasible_completed_trials=0,
        target_feasible_trials=budget["target_feasible"], max_total_attempts=budget["max_total_attempts"],
        batch_size=batch_size, task_context=fixed.get("task_context", {}),
        device_context=fixed.get("device_context", {}), runtime_context=fixed.get("runtime_context", {}),
        semantic_context=semantic, require_rationale=True, full_history_mode=True)


class LimitedProvider:
    """Bound actual content calls independently of repair/refill policy."""

    def __init__(self, provider):
        """Store a provider.

        Parameters
        ----------
        provider : object
            Production Responses provider with retries disabled.
        """
        self.provider, self.calls = provider, 0

    def complete_json(self, request):
        """Call the production provider at most four times.

        Parameters
        ----------
        request : LLMRequest
            Production ordered prompt.

        Returns
        -------
        LLMResponse
            Normalized production response.
        """
        if self.calls >= 4:
            raise SpendLimitExceeded("Smoke content-call cap of four reached")
        self.calls += 1
        return self.provider.complete_json(request)


def fixture_transport(context, history):
    """Generate deterministic policy-compliant fixtures behind Responses transport.

    Parameters
    ----------
    context : PromptContext
        Enriched production prompt context.
    history : tuple
        Saved evidence used for duplicate exclusion.

    Returns
    -------
    callable
        Local HTTP replacement returning raw Responses envelopes.
    """
    rng = random.Random(20261015)
    seen = {decoded_fingerprint(reconstruct_params(t, context.descriptor)) for t in history}
    def respond(http_request, timeout=None):
        """Return one local wire response.

        Parameters
        ----------
        http_request : urllib.request.Request
            Exact production wire request.
        timeout : float or None
            Ignored local transport timeout.

        Returns
        -------
        io.BytesIO
            JSON envelope consumable by the production provider.
        """
        wire = json.loads(http_request.data)
        dynamic = json.loads(wire["input"])["current_request"]
        entries = []
        targets = iter(dynamic["exploration_targets"])
        parents = iter(context.eligible_parents)
        for intent, count in dynamic["allocation"].items():
            for _ in range(count):
                parent = next(parents, context.eligible_parents[0] if context.eligible_parents else None) if intent == "local" else None
                target = next(targets, None) if intent == "exploration" else None
                for attempt in range(10000):
                    params = sample_random_candidate(context.descriptor, rng=rng)
                    if parent:
                        key = rng.choice(list(context.descriptor))
                        params = {**parent["params"], key: params[key]}
                    if target:
                        params["num_blocks"] = target["num_blocks"]
                        params["stride_schedule"] = target["aliases"][0]
                    identity = decoded_fingerprint(params)
                    if identity is not None and identity not in seen:
                        seen.add(identity)
                        break
                else:
                    raise RuntimeError("Offline fixture could not find a novel network")
                entries.append({"params": params, "intent": intent,
                    "parent_trial": parent["trial_number"] if parent else None,
                    "rationale": "Offline fixture: test a novel permitted training architecture; no scientific recommendation."})
        raw = {"model": "gpt-6-luna", "status": "completed", "service_tier": "default",
            "output": [{"type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": json.dumps({"candidates": entries})}]}],
            "usage": {"input_tokens": max(1, len(http_request.data) // 4), "output_tokens": 500,
                      "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0}}}
        return io.BytesIO(json.dumps(raw).encode())
    return respond


def check_cache(calls, ledger, fixed_prefix, cache_key):
    """Probe a repeated fixed prefix using shared limits and the production ledger.

    Parameters
    ----------
    calls : LimitedProvider
        Provider sharing this invocation's spend and call limits.
    ledger : LLMLedger
        Durable request and response artifacts.
    fixed_prefix : str
        Production fixed context, reused verbatim.
    cache_key : str
        Stable per-context routing key, following ranker replay.

    Returns
    -------
    list[dict]
        Observed usage; zero reads remain inconclusive.
    """
    usages = []
    for index in range(2):
        request = LLMRequest(fixed_prefix,
            f'Return a JSON object containing only "cache_probe": {index}.',
            "cs3-cache-probe-v1", {"cache_probe": index},
            cache_fixed_prefix=True, prompt_cache_key=cache_key)
        request_id = ledger.next_request_id()
        ledger.write_request(request_id, request)
        try:
            response = calls.complete_json(request)
        except ProviderError as exc:
            ledger.write_response(request_id, {"error": type(exc).__name__, "message": str(exc),
                "raw_response": getattr(exc, "raw_response", None),
                "usage": getattr(exc, "usage", None),
                "wire_payload": getattr(exc, "wire_payload", None)})
            raise
        ledger.write_response(request_id, response)
        usages.append(response.usage or {})
    return usages


def cache_fixture_transport(http_request, timeout=None):
    """Return simulated cache usage, without claiming a real cache confirmation.

    Parameters
    ----------
    http_request : urllib.request.Request
        Production explicit-breakpoint request.
    timeout : float or None
        Unused transport timeout.

    Returns
    -------
    io.BytesIO
        Minimal completed Responses envelope.
    """
    raw = {"status": "completed", "output": [{"type": "message", "role": "assistant", "status": "completed",
        "content": [{"type": "output_text", "text": '{"cache_probe":true}'}]}],
        "usage": {"input_tokens": len(http_request.data) // 4, "output_tokens": 8,
            "input_tokens_details": {"cached_tokens": 0}}}
    return io.BytesIO(json.dumps(raw).encode())


def run(args):
    """Render or exercise the production generation path without study mutation.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed source paths, rendering mode, and explicit paid opt-in.

    Returns
    -------
    dict
        Complete status, accounting, coverage, and accepted proposal report.
    """
    test_cache = getattr(args, "test_cache", False)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    context = load_context(args.prompt_context, args.batch_size)
    history = read_history(args.history_db, context.study_name, args.trials)
    from dataclasses import replace
    context = enrich_full_history(replace(context, attempted_trials=len(history),
        feasible_completed_trials=sum(outcome_class(t) == "successful" for t in history)), history)
    ledger = LLMLedger(output)
    request = build_candidate_request(context, prompt_version="cs3-full-history-v1")
    (output / "system_prompt.txt").write_text(request.system_prompt)
    (output / "user_prompt.txt").write_text(request.user_prompt)
    ledger.record_prompt_context(context.as_prompt_payload())
    settings = dict(provider="openai_responses", model="gpt-6-luna", base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY" if args.run_paid else "CREST_SMOKE_OFFLINE_KEY",
        temperature=None, reasoning_effort="medium", service_tier="default", json_response_mode=True,
        max_transport_retries=0, max_output_tokens=8192, max_input_tokens=120000, max_usd="0.05",
        input_usd_per_million_tokens="0.125", output_usd_per_million_tokens="0.50",
        pricing_source="Configured conservative smoke prices; verify before any paid call",
        output_token_parameter="max_output_tokens")
    tracker = SpendTracker(output, settings)
    provider = build_provider(settings, spend_tracker=tracker)
    if not args.run_paid:
        os.environ["CREST_SMOKE_OFFLINE_KEY"] = "offline-fixture"
        provider._urlopen = cache_fixture_transport if test_cache else fixture_transport(context, history)
    if test_cache:
        request = LLMRequest(request.system_prompt + "\n\nFor this cache diagnostic only, return the small "
            "JSON object requested by the user instead of generating candidates.",
            'Return a JSON object containing only "cache_probe": 0.',
            "cs3-cache-probe-v1", cache_fixed_prefix=True)
    cache_key = "cs3-cache-probe-" + hashlib.sha256(request.system_prompt.encode()).hexdigest()[:16]
    if test_cache:
        from dataclasses import replace
        request = replace(request, prompt_cache_key=cache_key)
    # Reserve and release nothing in render-only mode, but apply the actual guard.
    wire = provider.build_http_request(request)
    (output / "wire_payload.json").write_bytes(wire.data)
    input_bound = len(json.dumps(json.loads(wire.data), separators=(",", ":"), allow_nan=False).encode()) + 1024
    calls = LimitedProvider(provider)
    proposals, failure = (), None
    cache_usages = []
    if input_bound > settings["max_input_tokens"]:
        failure = f"Conservative serialized input bound {input_bound} exceeds 120000 guard"
    if not args.render_only and not failure:
        try:
            if test_cache:
                cache_usages = check_cache(calls, ledger, request.system_prompt, cache_key)
            else:
                proposals = generate_llm_batch(history, calls, context, ledger,
                    prompt_version="cs3-full-history-v1", max_repair_attempts=1,
                    max_refill_attempts=2, rng=random.Random(20261015))
        except ProviderError as exc:
            failure = f"{type(exc).__name__}: {exc}"
    accepted = [to_plain(p) for p in proposals]
    retained_before_error = []
    if failure:
        for line in (output / "prompt_contexts.jsonl").read_text().splitlines():
            retained_before_error = json.loads(line).get("retained_candidates", retained_before_error)
    fallback = any(p.provenance.get("proposal_source") == "random_fallback" for p in proposals)
    filled = len(proposals) == context.batch_size and not fallback
    report = {"mode": "render" if args.render_only else "paid" if args.run_paid else "offline_fixture",
        "status": "rendered" if args.render_only and not failure else "passed" if filled and not failure else "inconclusive",
        "history_db": str(Path(args.history_db).resolve()), "history_trials": len(history),
        "history_source": "actual_saved_prefix",
        "prompt_context": str(Path(args.prompt_context).resolve()), "http_attempt_limit": 4,
        "content_calls": calls.calls, "input_guard_bound": input_bound, "requested": context.batch_size, "accepted_count": len(proposals),
        "fill_rate": len(proposals) / context.batch_size, "fallback": fallback, "error": failure,
        "accepted": accepted, "retained_before_error": retained_before_error, "coverage": context.coverage, "allocation": context.allocation,
        "spending": json.loads((output / "spend_state.json").read_text()),
        "fixture_usage_is_simulated": not args.run_paid,
        "cache_policy": "explicit_no_breakpoints", "cache_reads": 0, "cache_writes": 0}
    if test_cache:
        confirmed = bool(args.run_paid and len(cache_usages) == 2
            and (cache_usages[1].get("cached_tokens") or 0) > 0 and not failure)
        report.update(status="rendered" if args.render_only and not failure else "passed" if confirmed else "inconclusive",
            cache_policy="explicit_fixed_prefix_probe", cache_confirmed=confirmed,
            cache_probe_usage=cache_usages,
            cache_result="confirmed" if confirmed else "rendered" if args.render_only else
                "simulated_only" if not args.run_paid else "no_cache_hit_observed")
    report["http_attempts"] = len(report["spending"]["reservations"])
    report["conservative_per_attempt_reservation_usd"] = str(tracker.reservation_usd)
    report["cache_savings_assumed"] = False
    counters_complete = {"cache_reads": True, "cache_writes": True}
    for response_path in (output / "requests").glob("*.response.json"):
        usage = json.loads(response_path.read_text()).get("usage") or {}
        for total, counter in (("cache_reads", "cached_tokens"), ("cache_writes", "cache_write_tokens")):
            value = usage.get(counter)
            if value is None:
                counters_complete[total] = False
            else:
                report[total] += value
    report["cache_counter_completeness"] = counters_complete
    for total, complete in counters_complete.items():
        if not complete:
            report[total] = None
    report["no_cache_writes_confirmed"] = bool(counters_complete["cache_writes"] and
        report["http_attempts"] > 0 and report["cache_writes"] == 0)
    report["cache_evidence_status"] = "complete" if all(counters_complete.values()) and report["http_attempts"] else "incomplete"
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main(argv=None):
    """Run the explicit smoke command.

    Parameters
    ----------
    argv : list[str] or None
        CLI arguments, defaulting to the process arguments.

    Returns
    -------
    int
        Zero for rendered/full batch; two for an inconclusive smoke.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-db", type=Path, required=True)
    parser.add_argument("--prompt-context", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int)
    parser.add_argument("--batch-size", type=int, choices=range(1, 6), default=5)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--render-only", action="store_true")
    mode.add_argument("--run-paid", action="store_true")
    parser.add_argument("--test-cache", action="store_true",
        help="Separate two-request cache probe; --run-paid required for real confirmation")
    args = parser.parse_args(argv)
    if args.trials is not None and args.trials < 0:
        parser.error("--trials must be nonnegative")
    report = run(args)
    print(json.dumps({key: report[key] for key in ("mode", "status", "history_trials", "accepted_count", "error")}, indent=2))
    return 0 if report["status"] in {"passed", "rendered"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
