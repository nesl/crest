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
    # A copy of the selected rates, not the whole catalog or its machine-local path.
    return {**selector, "currency": "USD", **{k: entry[k] for k in RATE_FIELDS},
            "source": entry["source"], "verified_on": entry["verified_on"]}
