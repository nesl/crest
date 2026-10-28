"""Full-history evidence preserves aliases, failures, and distinct training graphs."""
from dataclasses import replace

from crest.pipeline_types import TrialRecord
from crest.optimizers.llm.full_history import (
    RAW_FIELDS, assign_coverage_targets, build_coverage, build_full_history,
    candidate_cell, decoded_architecture, decoded_fingerprint, nearest_prior_distances,
    parameter_differences, reconstruct_params,
)


def params(**changes):
    result = dict(zip(RAW_FIELDS, (16, 3, 3, 3, "balanced", 2.0, 128, 1, 1.0, 0.1, True, 32, "avg", "relu")))
    return {**result, **changes}


def trial(number, state="COMPLETE", raw=None, **attrs):
    return TrialRecord(number, state, params() if raw is None else raw, (0.75,) if state == "COMPLETE" else None, attrs)


def test_decoded_aliases_and_training_graph_distinctions():
    # Caps erase raw width differences; dropout still changes training identity.
    capped = params(base_channels=24, pointwise_scale=1.5, channel_growth=2.0, max_channels=64)
    assert decoded_architecture(capped)["widths"] == [36, 64, 64]
    one = params(base_channels=4, pointwise_scale=1.5, num_blocks=2, channel_growth=1.0, stride_schedule="light")
    twin = params(base_channels=8, pointwise_scale=0.75, num_blocks=2, channel_growth=1.0, stride_schedule="late")
    assert decoded_fingerprint(one) == decoded_fingerprint(twin)
    for field, value in (("dropout_rate", 0.2), ("activation", "relu6"), ("norm_flag", False),
                         ("kernel_time", 5), ("depth_multiplier", 2), ("dense_units", 16),
                         ("global_pool_type", "max")):
        assert decoded_fingerprint(one) != decoded_fingerprint({**one, field: value})
    assert decoded_fingerprint({"num_blocks": 2}) is None


def test_columnar_partial_reconstruction_and_failure_visibility():
    # Proposal metadata fills gaps while actual sampled values remain authoritative.
    partial = trial(200, "FAIL", {"base_channels": 8}, crest_proposal={"proposal_params": params()}, failure_reason="lost worker")
    pending = trial(201, "WAITING", {})
    pending = replace(pending, system_attrs={"fixed_params": params()})
    table = build_full_history([pending, partial, trial(3, task_metrics={"macro_f1": 0.7}, latency_ms=-1)])
    assert [row[0] for row in table["rows"]] == [3, 200]
    row = dict(zip(table["columns"], table["rows"][1]))
    assert row["base_channels"] == 8 and row["num_blocks"] == 3
    assert row["failure_class"] == "unclassified" and table["failure_reasons"][row["failure_reason"]] == "lost worker"
    assert row["score"] is None and row["macro_f1"] is None
    assert table["pending"][0][0] == 201
    assert dict(zip(table["columns"], table["rows"][0]))["latency_ms"] is None
    assert reconstruct_params(pending) == params()


def test_coverage_resolves_aliases_and_counts_informative_union():
    # Repeated aliases and infrastructure retries must not inflate compulsory coverage.
    one = params(base_channels=4, pointwise_scale=1.5, num_blocks=2, channel_growth=1.0, stride_schedule="light")
    twin = params(base_channels=8, pointwise_scale=0.75, num_blocks=2, channel_growth=1.0, stride_schedule="late")
    evidence = [trial(0, raw=one), trial(1, raw=twin),
                trial(2, "FAIL", params(**{**one, "dropout_rate": 0.2}), measurement_failure_class="infrastructure"),
                trial(3, "FAIL", params(**{**one, "dropout_rate": 0.3})),
                trial(4, "FAIL", params(**{**one, "activation": "relu6"}), candidate_failure_class="deterministic")]
    coverage = build_coverage(evidence)
    assert len(coverage["cells"]) == 29
    cell = next(c for c in coverage["cells"] if c["cell_id"] == candidate_cell(one))
    assert cell["aliases"] == ["light", "late"]
    assert cell["informative"] == 2 and cell["successful"] == 1
    assert cell["infrastructure"] == 1 and cell["unclassified"] == 1
    assert not coverage["goal_complete"]
    targets = assign_coverage_targets(coverage, 5)
    assert len({t["cell_id"] for t in targets}) == 5
    assert cell["cell_id"] not in {t["cell_id"] for t in targets}


def test_coverage_ties_are_stable_without_shallow_first_order():
    # Tie handling must not turn breadth exploration into a shallow-first sweep.
    coverage = build_coverage([])
    targets = assign_coverage_targets(coverage, 5)
    reversed_cells = {**coverage, "cells": list(reversed(coverage["cells"]))}
    assert targets == assign_coverage_targets(reversed_cells, 5)
    assert len({t["num_blocks"] for t in targets}) > 1
    assert [t["num_blocks"] for t in targets] != sorted(t["num_blocks"] for t in targets)


def test_finite_target_reserves_pending_and_supports_restricted_space():
    # A single restricted cell stops targeting after two reservations/outcomes.
    from crest.optimizers.search_space import SearchParam
    descriptor = {"num_blocks": SearchParam(name="num_blocks", kind="categorical", choices=(2,)),
                  "stride_schedule": SearchParam(name="stride_schedule", kind="categorical", choices=("light", "late"))}
    coverage = build_coverage([trial(0, "WAITING", params(num_blocks=2, stride_schedule="light"))], descriptor)
    assert len(coverage["cells"]) == 1
    assert len(assign_coverage_targets(coverage, 5)) == 1
    assert not coverage["goal_complete"]
    complete = build_coverage([trial(0, raw=params(num_blocks=2, stride_schedule="light")),
                              trial(1, "PRUNED", params(num_blocks=2, stride_schedule="late", activation="relu6"), feasibility_status="infeasible")], descriptor)
    assert complete["goal_complete"]
    assert assign_coverage_targets(complete, 5) == []


def test_parent_diff_and_independent_nearest_representations():
    # A raw alias has nonzero provenance distance but zero decoded graph distance.
    parent = params(num_blocks=2, stride_schedule="light")
    alias = {**parent, "stride_schedule": "late"}
    diff = parameter_differences(alias, parent)
    assert diff == {"raw": {"stride_schedule": ["light", "late"]}, "decoded": {}}
    distances = nearest_prior_distances(alias, [trial(7, raw=parent)])
    assert distances["raw"] == {"trial": 7, "distance": 1}
    assert distances["decoded"] == {"trial": 7, "distance": 0}


def test_unknown_identity_never_satisfies_target_and_failures_do_not_become_scores():
    # Partial cell metadata remains visible, but cannot count as a unique network.
    partial = trial(0, "FAIL", {"num_blocks": 2, "stride_schedule": "light"})
    coverage = build_coverage([partial])
    cell = next(c for c in coverage["cells"] if c["cell_id"] == candidate_cell(partial.params))
    assert cell["unknown_decoded_trials"] == 1
    assert cell["informative"] == 0
    failed = replace(trial(1, "FAIL", measurement_failure_class="infrastructure"), values=(-1.0,))
    # Objective values may be negative; only measurement sentinels lose their value.
    table = build_full_history([failed])
    assert dict(zip(table["columns"], table["rows"][0]))["score"] == -1.0


def test_successful_rows_have_no_failure_while_real_errors_remain_visible():
    # A success label is not a failure reason; genuine errors must survive compacting.
    good = trial(0, error_code_label="HIL_MASTER_SUCCESS")
    bad = trial(1, error_code_label="HIL_ERROR", failure_reason="device error")
    table = build_full_history([good, bad, trial(2, "FAIL")])
    rows = [dict(zip(table["columns"], row)) for row in table["rows"]]
    assert rows[0]["failure_class"] is None and rows[0]["failure_reason"] is None
    assert rows[1]["failure_class"] == "unclassified" and table["failure_reasons"][rows[1]["failure_reason"]] == "device error"
    assert rows[2]["failure_class"] == "unclassified"


def test_penalty_outcomes_reasons_and_real_sentinel_readings():
    from crest.optimizers.llm.full_history import outcome_class
    # Distinguish the actual penalty convention from confirmed constraint evidence.
    penalty = dict(feasible=False, feasibility_status="not_evaluated", pruned=True,
        latency_ms=10000.0, energy_mj_per_inference=10000.0, prune_reason="HIL fatal error")
    cases = [trial(0, "PRUNED", **penalty), trial(1, "FAIL", **penalty,
                 measurement_failure_class="infrastructure"),
             trial(2, "PRUNED", **penalty, candidate_failure_class="deterministic"),
             trial(3, "PRUNED", feasibility_status="infeasible", feasible=False,
                 feasibility_reason="latency exceeds limit", error_code_label="HIL_MASTER_SUCCESS",
                 latency_ms=10000.0),
             trial(4, "WAITING", **penalty),
             trial(5, latency_ms=10000.0, energy_mj_per_inference=10000.0)]
    assert [outcome_class(t) for t in cases] == ["unclassified", "infrastructure", "deterministic",
                                               "infeasible", "pending", "successful"]
    table = build_full_history(cases)
    rows = {r[0]: dict(zip(table["columns"], r)) for r in table["rows"] + table["pending"]}
    for number in (0, 1, 2):
        assert rows[number]["latency_ms"] is None and rows[number]["energy_mj_per_inference"] is None
    assert rows[3]["failure_class"] == "infeasible"
    assert table["failure_reasons"][rows[3]["failure_reason"]] == "latency exceeds limit"
    assert rows[3]["latency_ms"] == 10000
    assert rows[4]["failure_class"] is None and rows[4]["failure_reason"] is None
    assert rows[5]["latency_ms"] == 10000 and rows[5]["energy_mj_per_inference"] == 10000


def test_reason_dictionary_grows_only_when_a_new_error_is_observed():
    # New errors append; repeated errors reuse an ID without rewriting earlier rows.
    evidence = [trial(0, "FAIL", failure_reason="x error"),
                trial(1, "FAIL", failure_reason="x error"),
                trial(2, "FAIL", failure_reason="y error"),
                trial(3, "FAIL", failure_reason="x error")]
    previous = []
    for count, expected in ((1, ["x error"]), (2, ["x error"]),
                            (3, ["x error", "y error"]), (4, ["x error", "y error"])):
        table = build_full_history(evidence[:count])
        assert table["failure_reasons"] == expected
        assert table["rows"][:len(previous)] == previous
        previous = table["rows"]
    index = table["columns"].index("failure_reason")
    assert [row[index] for row in table["rows"]] == [0, 0, 1, 0]
    assert "failure_reasons" not in build_full_history([trial(0)])


def test_unknown_retry_budget_is_decoded_and_not_coverage():
    # Two unknown failures block another attempt; aliases count, infrastructure does not.
    from crest.optimizers.llm.full_history import exhausted_unknown_retries
    first = trial(0, "PRUNED", params(num_blocks=2, stride_schedule="light"),
                  feasible=False, feasibility_status="not_evaluated", prune_reason="HIL fatal error")
    second = trial(1, "FAIL", params(num_blocks=2, stride_schedule="late"))
    assert not exhausted_unknown_retries([first])
    assert exhausted_unknown_retries([first, second]) == {decoded_fingerprint(first.params)}
    assert build_full_history([first, second])["retry_exhausted_trials"] == [0, 1]
    assert all(c["informative"] == 0 for c in build_coverage([first, second])["cells"])
    infra = trial(2, "FAIL", first.params, measurement_failure_class="infrastructure")
    assert not exhausted_unknown_retries([first, infra])
