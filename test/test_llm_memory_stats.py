# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Computed summary evidence uses only each new chunk and finite samples."""
import math

from crest.optimizers.llm.memory import build_computed_statistics


def test_computed_statistics_samples_groups_and_scope():
    records = [
        {"number": 80, "state": "complete", "params": {"pool": "light5", "fixed": 2},
         "value": .6, "latency_ms": 10., "feasibility_status": "feasible",
         "task_metrics": {"macro_f1": .7, "loss": 2}},
        {"number": 81, "state": "complete", "params": {"pool": "light5", "fixed": 2},
         "value": .6442928, "energy_mj_per_inference": .4},
        {"number": 82, "state": "fail", "params": {"pool": "other", "fixed": 2},
         "value": -100., "latency_ms": math.inf, "energy_mj_per_inference": None,
         "task_metrics": {"macro_f1": math.nan}},
    ]
    stats = build_computed_statistics(records)
    assert stats["new_trial_numbers"] == [80, 81, 82]
    assert stats["scope"] == "new_trials_only"
    assert stats["overall"]["state_counts"] == {"complete": 2, "fail": 1}
    assert stats["overall"]["feasible_count"] == 1
    assert stats["overall"]["metrics"]["latency_ms"] == {"n": 1, "mean": 10., "min": 10., "max": 10.}
    assert stats["overall"]["metrics"]["task_metrics.macro_f1"]["n"] == 1
    assert "task_metrics.loss" not in stats["overall"]["metrics"]
    assert "fixed" not in stats["by_parameter"]
    group = stats["by_parameter"]["pool"][0]
    assert group["value"] == "light5"
    assert group["metrics"]["recorded_objective_value"]["mean"] == .6221464
    assert stats["by_parameter"]["pool"][1]["metrics"]["recorded_objective_value"]["min"] == -100.


def test_computed_statistics_empty_and_missing_values():
    assert build_computed_statistics([])["overall"] == {
        "count": 0, "state_counts": {}, "feasible_count": 0, "metrics": {}}
    stats = build_computed_statistics([
        {"number": 1, "state": "pruned", "params": {"blocks": 1}, "value": True},
        {"number": 2, "state": "fail", "params": {}, "value": None},
    ])
    assert stats["overall"]["metrics"] == {}
    assert stats["by_parameter"] == {}
