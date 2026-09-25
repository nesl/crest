#!/usr/bin/env python3
# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Count saved CREST prompts via OpenAI, then project search token costs.

Standard library only. No training, model generation, or hardware access.
Without --count-api or --reuse-counts, only prepare the counting payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import urllib.error
import urllib.request

ENDPOINT = "https://api.openai.com/v1/responses/input_tokens"


def finite_number(value, name, *, minimum=0, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(value) or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside the allowed range")
    return value


def prepare_request(record, model):
    """Convert a text-only CREST ledger request or preserve a native count body."""
    if not isinstance(record, dict):
        raise ValueError("Request JSON must be an object")
    if "messages" not in record:
        if "input" not in record:
            raise ValueError("Expected CREST messages or a native Responses input")
        if record.get("model", model) != model:
            raise ValueError("Native request model differs from --model")
        return {**record, "model": model}, "native_responses_request"
    # Reject transport options that would otherwise be silently dropped.
    extras = set(record) - {"messages", "prompt_version", "metadata", "request_hash", "model"}
    if extras:
        raise ValueError(f"Unsupported Chat request fields: {sorted(extras)}; supply a native Responses count body")
    messages = record["messages"]
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list")
    for message in messages:
        if (not isinstance(message, dict) or set(message) != {"role", "content"}
                or message["role"] not in {"system", "developer", "user", "assistant"}
                or not isinstance(message["content"], str)):
            raise ValueError("CREST conversion accepts only role/content text messages")
    return {"model": model, "input": messages}, "crest_chat_messages_as_responses_input"


def count_input_tokens(payload, api_key, *, timeout=60, opener=None):
    """Call only the input-count endpoint; never send a generation request."""
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for --count-api")
    request = urllib.request.Request(
        ENDPOINT, data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        with (opener or urllib.request.urlopen)(request, timeout=timeout) as response:
            result = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        # Avoid saving arbitrary server bodies or credentials in reports/errors.
        raise ValueError(f"Token-count API returned HTTP {exc.code}; check model, payload, and credentials") from None
    except urllib.error.URLError:
        raise ValueError("Could not reach the token-count API") from None
    count = result.get("input_tokens") if isinstance(result, dict) else None
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("Token-count API response has no valid input_tokens")
    return count


def project(counts, *, trials, batch_size, acceptance_rate=1.0,
            extra_calls_per_round=0.0, repair_extra_input_tokens=0,
            output_tokens_per_call=None, input_price=None, output_price=None):
    """Scenario arithmetic; trial budget means attempted evaluations, not feasible trials."""
    if not counts:
        raise ValueError("At least one counted prompt is required")
    for count in counts:
        finite_number(count, "input token count")
    for name, value in (("trials", trials), ("batch_size", batch_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    finite_number(acceptance_rate, "acceptance_rate", maximum=1)
    if acceptance_rate == 0:
        raise ValueError("acceptance_rate must be greater than zero")
    finite_number(extra_calls_per_round, "extra_calls_per_round")
    finite_number(repair_extra_input_tokens, "repair_extra_input_tokens")
    for name, value in (("output_tokens_per_call", output_tokens_per_call),
                        ("input_price", input_price), ("output_price", output_price)):
        if value is not None:
            finite_number(value, name)
    rounds = math.ceil(trials / (batch_size * acceptance_rate))
    extra_calls = rounds * extra_calls_per_round
    calls = rounds + extra_calls
    output = None if output_tokens_per_call is None else calls * output_tokens_per_call
    scenarios = []
    for name, prompt_tokens in (("sample_min", min(counts)),
                                ("sample_mean", statistics.mean(counts)),
                                ("sample_max", max(counts))):
        input_tokens = calls * prompt_tokens + extra_calls * repair_extra_input_tokens
        input_cost = None if input_price is None else input_tokens * input_price / 1_000_000
        output_cost = None if output is None or output_price is None else output * output_price / 1_000_000
        scenarios.append({
            "scenario": name, "input_tokens_per_base_call": prompt_tokens,
            "projected_input_tokens": input_tokens, "projected_output_tokens": output,
            "input_cost_usd": input_cost, "output_cost_usd": output_cost,
            "total_cost_usd": None if input_cost is None or output_cost is None else input_cost + output_cost,
        })
    return {"attempted_evaluations": trials, "generation_rounds": rounds,
            "expected_api_calls": calls, "scenarios": scenarios}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--request", type=Path, action="append", help="Saved request JSON; repeat for cold/warm/repair prompts")
    source.add_argument("--reuse-counts", type=Path, help="Reproject a previously counted report without API calls")
    p.add_argument("--model", required=True, help="Exact OpenAI model ID, not an OpenRouter-prefixed ID")
    p.add_argument("--count-api", action="store_true", help="Send prompt payloads to the OpenAI input-token endpoint")
    p.add_argument("--trials", type=int, action="append", help="Attempted trial budget; repeat; default: 150 and 250")
    p.add_argument("--batch-size", type=int, default=5)
    p.add_argument("--acceptance-rate", type=float, default=1.0, help="Fraction of requested candidates accepted per generation round")
    p.add_argument("--extra-calls-per-round", type=float, default=0, help="Mean repair calls per round, e.g. 0.1")
    p.add_argument("--repair-extra-input-tokens", type=float, default=0)
    p.add_argument("--output-tokens-per-call", type=float, help="Assumed billed output, including reasoning; not predicted by count API")
    p.add_argument("--input-usd-per-million", type=float)
    p.add_argument("--output-usd-per-million", type=float)
    p.add_argument("--price-source", help="Pricing URL/date or description, saved as provenance")
    p.add_argument("--report", type=Path, help="Write JSON here instead of stdout")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        assumptions = dict(batch_size=args.batch_size, acceptance_rate=args.acceptance_rate,
                           extra_calls_per_round=args.extra_calls_per_round,
                           repair_extra_input_tokens=args.repair_extra_input_tokens,
                           output_tokens_per_call=args.output_tokens_per_call,
                           input_price=args.input_usd_per_million, output_price=args.output_usd_per_million)
        budgets = args.trials or [150, 250]
        # Validate every scenario before making any network requests.
        for budget in budgets:
            project([0], trials=budget, **assumptions)
        if args.reuse_counts:
            if args.count_api:
                raise ValueError("--reuse-counts cannot be combined with --count-api")
            old = json.loads(args.reuse_counts.read_text())
            if old.get("model") != args.model:
                raise ValueError("Saved counts use a different model")
            records = old["requests"]
            if not records or any(type(r.get("input_tokens")) is not int or r["input_tokens"] < 0 for r in records):
                raise ValueError("Saved report does not contain valid API counts")
        else:
            records = []
            for path in args.request:
                raw = path.read_bytes()
                payload, conversion = prepare_request(json.loads(raw), args.model)
                records.append({"source": str(path.resolve()), "source_sha256": hashlib.sha256(raw).hexdigest(),
                                "conversion": conversion, "count_request": payload, "input_tokens": None})
            if args.count_api:
                for record in records:
                    record["input_tokens"] = count_input_tokens(record["count_request"], os.environ.get("OPENAI_API_KEY"))
        counts = [r["input_tokens"] for r in records if r["input_tokens"] is not None]
        report = {
            "schema_version": 1, "model": args.model, "endpoint": ENDPOINT,
            "mode": "reused_counts" if args.reuse_counts else "api_count" if args.count_api else "dry_run",
            "assumptions": assumptions, "price_source": args.price_source, "requests": records,
            "projections": [project(counts, trials=n, **assumptions) for n in budgets] if counts else [],
            "limitations": [
                "Converted CREST chat messages are counted as Responses input; this is not an exact OpenRouter/Chat Completions bill. Transport JSON mode is not included in that conversion.",
                "Sample min/mean/max are scenarios, not confidence bounds; the mean weights each supplied prompt equally. Include representative populated-history prompts.",
                "Output/reasoning tokens and retry/acceptance assumptions are supplied by the user, not measured or predicted by this endpoint.",
                "Uses full-batch rounds, including the final partial batch. No cache discounts, provider fees, taxes, training costs, or HIL costs are included.",
                "Acceptance means candidates enqueued, not hardware feasibility. Frequent random fallback requires its own measured scenario.",
            ],
        }
        rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(rendered)
            print(f"Wrote {args.report}", file=sys.stderr)
        else:
            print(rendered, end="")
    except (ValueError, OSError, KeyError, TypeError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
