# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Resolve a reviewed tariff for an exact provider endpoint, model, and tier.

Only the selected entry is retained in study accounting. File location and
unrelated catalog entries do not affect resume compatibility.
"""

import json
import math
from datetime import date
from pathlib import Path

RATE_FIELDS = ("input_usd_per_million_tokens", "output_usd_per_million_tokens")


def resolve_pricing(config):
    """Load one matching USD tariff, failing closed on missing or ambiguous rates."""
    supplied = config.get("pricing_file")
    if supplied is not None and (not isinstance(supplied, str) or not supplied.strip()):
        raise ValueError("optimizer.llm.pricing_file must be a nonempty path.")
    path = Path(supplied).expanduser() if supplied else Path(__file__).with_name("pricing.json")
    try:
        catalog = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read pricing catalog: {path}") from exc
    if (not isinstance(catalog, dict) or type(catalog.get("schema_version")) is not int
            or catalog["schema_version"] != 1 or not isinstance(catalog.get("rates"), list)):
        raise ValueError("Pricing catalog requires schema_version: 1 and a rates list.")
    selector = {
        "provider": config["provider"], "model": config["model"],
        "base_url": config["base_url"].rstrip("/"),
        "service_tier": config.get("service_tier") or "default",
    }
    matches = []
    for entry in catalog["rates"]:
        if not isinstance(entry, dict):
            raise ValueError("Each pricing entry must be an object.")
        if all(entry.get(k) == v for k, v in selector.items()):
            matches.append(entry)
    if len(matches) != 1:
        raise ValueError(f"Expected one pricing entry for {selector}; found {len(matches)}. "
                         "Provide a matching optimizer.llm.pricing_file.")
    entry = matches[0]
    if entry.get("currency") != "USD":
        raise ValueError("Pricing currency must be USD.")
    for key in RATE_FIELDS:
        value = entry.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise ValueError(f"Pricing {key} must be finite and positive.")
    if not isinstance(entry.get("source"), str) or not entry["source"].strip():
        raise ValueError("Pricing source must be nonempty.")
    try:
        date.fromisoformat(entry["verified_on"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Pricing verified_on must be an ISO date.") from exc
    # Flat accounting reserves using an upper-bound tariff: cache writes may
    # exceed ordinary input rates, and the configured input ceiling determines
    # whether long-context prices can apply. Keep raw tariff evidence as well.
    effective = {k: entry[k] for k in RATE_FIELDS}
    cache_write = entry.get("cache_write_usd_per_million_tokens")
    if cache_write is not None:
        if isinstance(cache_write, bool) or not isinstance(cache_write, (int, float)) or not math.isfinite(cache_write) or cache_write <= 0:
            raise ValueError("Cache-write price must be finite and positive.")
        effective[RATE_FIELDS[0]] = max(effective[RATE_FIELDS[0]], cache_write)
    long_context = entry.get("long_context")
    if long_context is not None:
        if not isinstance(long_context, dict) or type(long_context.get("input_token_threshold")) is not int or long_context["input_token_threshold"] < 1:
            raise ValueError("Long-context threshold must be a positive integer.")
        for key in ("input_multiplier", "output_multiplier"):
            value = long_context.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 1:
                raise ValueError("Long-context multipliers must be finite and at least one.")
        limit = config.get("max_input_tokens")
        if type(limit) is not int or limit < 1:
            raise ValueError("Tiered pricing requires a positive max_input_tokens.")
        if limit > long_context["input_token_threshold"]:
            effective[RATE_FIELDS[0]] *= long_context["input_multiplier"]
            effective[RATE_FIELDS[1]] *= long_context["output_multiplier"]
    snapshot = {**selector, "currency": "USD", **effective,
            "source": entry["source"], "verified_on": entry["verified_on"]}
    if cache_write is not None or long_context is not None:
        snapshot["tariff"] = {k: entry[k] for k in (*RATE_FIELDS, "cache_write_usd_per_million_tokens", "long_context") if k in entry}
    return snapshot
