# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""ESP32 classifier-only serial evidence validation."""

from __future__ import annotations

import math
import re
import statistics
from typing import Any, Sequence


_READY_RE = re.compile(
    r"^ML_READY model_sha256=([0-9a-f]{64}) fixture_sha256=([0-9a-f]{64}) "
    r"arena_used=(\d+) arena_reserved=(\d+) fixtures=(\d+) warmups=(\d+) "
    r"repetitions=(\d+) aggregation=(mean|median)$"
)
_SAMPLE_RE = re.compile(r"^ML_SAMPLE fixture=(\d+) id=(\S+) repetition=(\d+) invoke_us=(\d+)$")
_OUTPUT_RE = re.compile(r"^ML_OUTPUT fixture=(\d+) id=(\S+) values=(-?\d+(?:,-?\d+)*)$")
_VALID_RE = re.compile(
    r"^MEASUREMENT_VALID model_sha256=([0-9a-f]{64}) fixture_sha256=([0-9a-f]{64}) samples=(\d+)$"
)


def validate_measurement_log(
    *,
    lines: Sequence[str],
    manifest: dict[str, Any],
    latency_s: float | None,
    psram_bytes: int,
) -> tuple[dict[str, object], dict[str, object]]:
    """Validate identities, ordered samples, outputs, arena, and aggregate."""

    ids = [str(value) for value in manifest.get("fixture_ids", [])]
    wire_ids = [str(value) for value in manifest.get("fixture_wire_ids", ids)]
    fixture_count = int(manifest.get("fixture_count", 0))
    repetitions = int(manifest.get("measured_runs_per_fixture", 0))
    warmups = int(manifest.get("warmup_runs_per_fixture", 0))
    aggregation = str(manifest.get("latency_aggregation", ""))
    model_sha = str(manifest.get("tflite_sha256", ""))
    fixture_sha = str(manifest.get("fixture_sha256", ""))
    output_elements = int(manifest.get("output_elements", 0))
    expected_samples = fixture_count * repetitions
    ready = None
    marker = None
    samples: list[dict[str, object]] = []
    outputs: list[dict[str, object]] = []
    for raw in lines:
        line = raw.strip()
        ready_match = _READY_RE.match(line)
        if ready_match:
            ready = ready_match.groups()
        sample_match = _SAMPLE_RE.match(line)
        if sample_match:
            samples.append({
                "fixture": int(sample_match.group(1)),
                "id": sample_match.group(2),
                "repetition": int(sample_match.group(3)),
                "invoke_us": int(sample_match.group(4)),
            })
        output_match = _OUTPUT_RE.match(line)
        if output_match:
            values = [int(value) for value in output_match.group(3).split(",")]
            outputs.append({
                "fixture": int(output_match.group(1)),
                "id": output_match.group(2),
                "values": values,
            })
        valid_match = _VALID_RE.match(line)
        if valid_match:
            marker = valid_match.groups()
    reasons: list[str] = []
    if (
        fixture_count <= 0 or repetitions <= 0 or warmups < 0
        or len(ids) != fixture_count or len(wire_ids) != fixture_count
    ):
        reasons.append("manifest_workload")
    if output_elements <= 0:
        reasons.append("manifest_output")
    if ready is None:
        reasons.append("missing_ready")
        arena_used = arena_reserved = 0
    else:
        (ready_model, ready_fixture, used, reserved, count, ready_warmups,
         ready_repetitions, ready_aggregation) = ready
        arena_used, arena_reserved = int(used), int(reserved)
        if (ready_model, ready_fixture) != (model_sha, fixture_sha): reasons.append("ready_identity")
        if int(count) != fixture_count: reasons.append("ready_fixture_count")
        if int(ready_warmups) != warmups or int(ready_repetitions) != repetitions: reasons.append("ready_runs")
        if ready_aggregation != aggregation: reasons.append("ready_aggregation")
        if not (0 < arena_used <= arena_reserved <= psram_bytes): reasons.append("arena_accounting")
    expected_order = [
        (fixture, wire_ids[fixture], repetition)
        for fixture in range(fixture_count)
        for repetition in range(repetitions)
    ] if len(ids) == fixture_count else []
    actual_order = [(int(row["fixture"]), str(row["id"]), int(row["repetition"])) for row in samples]
    timings = [int(row["invoke_us"]) for row in samples]
    if actual_order != expected_order: reasons.append("sample_order")
    if len(timings) != expected_samples or any(value <= 0 for value in timings): reasons.append("sample_timings")
    if len(outputs) != fixture_count: reasons.append("output_count")
    else:
        for index, output in enumerate(outputs):
            values = output["values"]
            expected_id = wire_ids[index] if index < len(wire_ids) else None
            if (output["fixture"], output["id"]) != (index, expected_id): reasons.append("output_order")
            if len(values) != output_elements or any(value < -128 or value > 127 for value in values):
                reasons.append("output_values")
    if marker != (model_sha, fixture_sha, str(expected_samples)): reasons.append("valid_marker")
    computed_latency_s = None
    if timings and len(timings) == expected_samples:
        aggregate_us = statistics.median(timings) if aggregation == "median" else statistics.mean(timings)
        computed_latency_s = float(aggregate_us) / 1_000_000.0
        if latency_s is None or not math.isfinite(latency_s) or abs(latency_s - computed_latency_s) > 1e-6:
            reasons.append("aggregate")
    else:
        reasons.append("aggregate")
    valid = not reasons
    evidence = {
        "measurement_instrumentation_valid": valid,
        "measurement_model_sha256": model_sha,
        "measurement_fixture_sha256": fixture_sha,
        "measurement_sample_count": expected_samples if valid else 0,
        "measurement_warmup_runs_per_fixture": warmups,
        "measurement_runs_per_fixture": repetitions,
        "measurement_latency_aggregation": aggregation,
    }
    diagnostics = {
        "valid": valid,
        "reasons": sorted(set(reasons)),
        "arena_used_bytes": arena_used,
        "arena_reserved_bytes": arena_reserved,
        "samples": samples,
        "outputs": outputs,
        "computed_latency_s": computed_latency_s,
    }
    return evidence, diagnostics
