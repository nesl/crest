"""Offline audit aggregates raw repeats and accidental decoded aliases equally."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import math

from crest.pipeline_types import TrialRecord

spec = importlib.util.spec_from_file_location("cs3_search_audit", Path(__file__).resolve().parents[1] / "analysis_scripts/analyze_cs3_search.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def raw(**changes):
    params = dict(base_channels=16, num_blocks=2, kernel_time=3, kernel_freq=3,
                  stride_schedule="light", channel_growth=2.0, max_channels=128,
                  depth_multiplier=1, pointwise_scale=1.0, dropout_rate=0.1,
                  norm_flag=True, dense_units=32, global_pool_type="avg", activation="relu")
    return {**params, **changes}


def record(number, value, params=None, state="COMPLETE", **attrs):
    return TrialRecord(number, state, raw() if params is None else params,
        (value,) if value is not None else None,
        {"task_metrics": {"macro_f1": value}, "feasibility_status": "feasible", **attrs})


def test_alias_and_exact_repeats_share_one_network_mean():
    # Every repeat contributes equally, including aliases with distinct raw schedules.
    history = [record(0, 0.9), record(1, 0.5, raw(stride_schedule="late")),
               record(2, 0.7), record(3, 0.8, raw(activation="relu6")),
               record(4, 1.0, state="FAIL", measurement_failure_class="infrastructure")]
    rows, networks, pooled = audit.aggregate_prefixes(history, {0: 2, 1: 3})
    latest = [r for r in rows if r["metric"] == "macro_f1"][-1]
    assert latest["best_network_mean"] == 0.8
    assert latest["best_individual_observed"] == 0.9
    assert latest["top10_mean"] == latest["top10_median"] == 0.75
    assert latest["valid_observations"] == 4 and latest["unique_scored_networks"] == 2
    assert latest["repeat_groups"] == 1 and latest["repeated_observations"] == 3
    assert latest["evaluation_wall_seconds_proxy"] == 5
    assert pooled["decoded_network"]["measurements"] == 3
    assert math.isclose(pooled["decoded_network"]["macro_f1_sd"], 0.2)
    assert latest["best_network_observation_count"] == 1
    assert latest["top10_single_observation_networks"] == 1
    assert latest["coverage_informative_networks"] == 2
    assert latest["coverage_successful_networks"] == 2
    assert latest["coverage_cells_meeting_goal"] == 1
    assert latest["coverage_goal_slots"] == 58
    assert pooled["measurements"] == 2 and pooled["degrees_of_freedom"] == 1
    assert math.isclose(pooled["macro_f1_sd"], math.sqrt(0.02))
    repeated = [n for n in networks if n["metric"] == "macro_f1" and n["count"] == 3]
    assert len(repeated) == 1 and repeated[0]["mean"] == 0.7


def test_missing_and_partial_never_get_fabricated_scores():
    # Unknown identities and failure penalties remain outside score ranking.
    history = [record(0, None, state="PRUNED"), record(1, 0.8, {"num_blocks": 2}),
               record(2, -1, state="FAIL"), record(3, 0.6)]
    rows, groups, pooled = audit.aggregate_prefixes(history)
    latest = rows[-1]
    assert latest["valid_observations"] == 1 and latest["best_network_mean"] == 0.6
    assert pooled["macro_f1_sd"] is None


def test_readonly_database_audit_writes_csv_and_provenance(tmp_path):
    # A minimal saved Optuna database exercises SQL decoding and CLI artifacts.
    import hashlib
    import json
    import sqlite3
    database = tmp_path / "history.db"
    with sqlite3.connect(database) as db:
        db.executescript('''
        CREATE TABLE studies(study_id INTEGER, study_name TEXT);
        CREATE TABLE trials(trial_id INTEGER, number INTEGER, state TEXT, study_id INTEGER, datetime_start TEXT, datetime_complete TEXT);
        CREATE TABLE trial_params(trial_id INTEGER, param_name TEXT, param_value REAL, distribution_json TEXT);
        CREATE TABLE trial_user_attributes(trial_id INTEGER, key TEXT, value_json TEXT);
        CREATE TABLE trial_system_attributes(trial_id INTEGER, key TEXT, value_json TEXT);
        CREATE TABLE trial_values(trial_id INTEGER, value REAL, value_type TEXT, objective INTEGER);
        INSERT INTO studies VALUES(1,'fixture');
        INSERT INTO trials VALUES(1,0,'COMPLETE',1,'2026-10-01 00:00:00','2026-10-01 00:00:02');
        INSERT INTO trial_values VALUES(1,0.7,'FINITE',0);
        INSERT INTO trial_user_attributes VALUES(1,'task_metrics','{"macro_f1":0.7}');
        INSERT INTO trial_user_attributes VALUES(1,'feasibility_status','"feasible"');
        ''')
        for key, value in raw().items():
            distribution = json.dumps({"name": "CategoricalDistribution", "attributes": {"choices": [value]}})
            db.execute("INSERT INTO trial_params VALUES(1,?,?,?)", (key, 0, distribution))
        # Trial 1 completes first even though trial 0 was allocated earlier.
        db.execute("INSERT INTO trials VALUES(2,1,'COMPLETE',1,'2026-10-01 00:00:00','2026-10-01 00:00:01')")
        db.execute("INSERT INTO trial_values VALUES(2,0.5,'FINITE',0)")
        db.execute("INSERT INTO trial_user_attributes VALUES(2,'task_metrics','{\"macro_f1\":0.5}')")
        db.execute("INSERT INTO trial_user_attributes VALUES(2,'feasibility_status','\"feasible\"')")
        db.execute("INSERT INTO trial_params SELECT 2,param_name,param_value,distribution_json FROM trial_params WHERE trial_id=1")
    history, meta = audit.read_history(database)
    rows, _, _ = audit.aggregate_prefixes(history, meta["durations_seconds"], completion_order=meta["completion_order"])
    scores = [r for r in rows if r["metric"] == "score"]
    assert [r["trial_number"] for r in scores] == [1, 0]
    assert [r["best_network_mean"] for r in scores] == [0.5, 0.6]
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    output = tmp_path / "audit"
    report = audit.run(SimpleNamespace(history_db=database, study_name=None, output=output))
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before == report["input"]["sha256"]
    assert report["input"]["study_name"] == "fixture"
    assert (output / "audit.json").exists() and (output / "trajectories.csv").exists()
    assert len(report["coverage"]["cells"]) == 29
    assert len(report["networks"]) == 2
    assert report["code_sha256"]


def test_later_repeat_changes_only_later_network_estimates():
    # A low repeat lowers the current best mean without rewriting its earlier peak.
    first = record(94, 0.9)
    later = record(125, 0.5, raw(stride_schedule="late"))
    before, _, _ = audit.aggregate_prefixes([first])
    after, _, _ = audit.aggregate_prefixes([first, later])
    assert after[:2] == before
    scores = [r for r in after if r["metric"] == "score"]
    assert [r["best_network_mean"] for r in scores] == [0.9, 0.7]
    assert [r["best_individual_observed"] for r in scores] == [0.9, 0.9]


def test_completion_order_prevents_future_result_leakage():
    # Concurrent/resumed records must follow availability rather than trial number.
    history = [record(0, 0.9), record(1, 0.5)]
    rows, _, _ = audit.aggregate_prefixes(history, completion_order={1: 0, 0: 1})
    scores = [r for r in rows if r["metric"] == "score"]
    assert [r["trial_number"] for r in scores] == [1, 0]
    assert scores[0]["best_network_mean"] == 0.5
    assert scores[1]["best_network_mean"] == 0.7


def test_time_axes_keep_failures_and_gaps_visible():
    # Completed-only time excludes failures; all-attempt time and elapsed expose them.
    history = [record(0, 0.8), record(1, None, state="FAIL"),
               record(2, None, state="PRUNED"), record(3, 0.6)]
    rows, _, _ = audit.aggregate_prefixes(history, {0: 10, 1: 20, 2: 30, 3: 40},
        elapsed_seconds={0: 10, 1: 50, 2: 90, 3: 150})
    final = rows[-1]
    assert final["completed_trial_wall_seconds_proxy"] == 50
    assert final["all_attempt_wall_seconds_proxy"] == 100
    assert final["elapsed_wall_seconds"] == 150
    assert final["missing_duration_count"] == 0


def test_api_overhead_includes_memory_and_flags_missing_timings(tmp_path):
    # Missing/error responses cannot masquerade as zero-cost API time.
    import json
    for name, latency in (("requests/1", 1000), ("memory/requests/1", 2000), ("requests/2", None)):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(".request.json").write_text("{}")
        if latency is not None:
            path.with_suffix(".response.json").write_text(json.dumps({"latency_ms": latency}))
    report = audit.api_overhead(tmp_path)
    assert report["recorded_api_wall_seconds"] == 3
    assert report["missing_timing_count"] == 1
    assert not report["timing_complete"]
    assert audit.api_overhead(None)["recorded_api_wall_seconds"] is None


def test_duration_completeness_is_separate_for_each_axis():
    # Missing failed timing leaves completed timing complete; missing COMPLETE does not.
    rows, _, _ = audit.aggregate_prefixes([record(0, .8), record(1, None, state="FAIL"), record(2, .7)], {0: 10})
    assert rows[3]["completed_trial_duration_complete"]
    assert rows[3]["all_attempt_duration_is_lower_bound"]
    assert rows[-1]["completed_trial_missing_duration_count"] == 1
    assert rows[-1]["completed_trial_duration_is_lower_bound"]
    assert not rows[-1]["completed_trial_duration_complete"]


def test_equal_means_rank_by_decoded_id():
    # Reversing arrival order cannot change which equal-mean network wins a tie.
    history = [record(0, .8), record(1, .8, raw(activation="relu6"))]
    rows, networks, _ = audit.aggregate_prefixes(history)
    assert rows[-1]["best_network_id"] == min(n["decoded_id"] for n in networks)


def test_saved_proposal_denominators_exclude_malformed_and_fallback(tmp_path):
    # Rejected proposals are observable only in the saved candidate ledger.
    import json
    def stream(name, values):
        (tmp_path / name).write_text("".join(json.dumps(v) + "\n" for v in values))
    stream("returned_candidates.jsonl", [{"proposal_source": "llm"}, {"proposal_source": "random_fallback"}])
    stream("rejected_candidates.jsonl", [{"code": "duplicate_raw"}, {"code": "duplicate_decoded"}, {"code": "quota_overflow"}, {"code": "malformed_response"}])
    stream("optimizer_events.jsonl", [{"refill_calls": 2}])
    report = audit.proposal_audit(tmp_path)
    assert report["observed_candidate_decisions"] == 4
    assert report["duplicate_rejection_rate"] == .5
    assert report["raw_duplicate_rejection_rate"] == .25
    assert report["decoded_duplicate_rejection_rate"] == .25
    assert report["malformed_calls"] == report["fallback_candidates"] == 1
    assert report["refill_attempts"] == 2
    assert not report["missing_data"]
    stream("optimizer_events.jsonl", [{"event": "provider_error"}])
    incomplete = audit.proposal_audit(tmp_path)
    assert incomplete["missing_data"] and incomplete["provider_error_rounds"] == 1
    assert incomplete["duplicate_rejection_rate"] is None
    assert incomplete["observed_duplicate_rejection_rate"] == .5
    assert incomplete["refill_data_missing"] and incomplete["refill_attempts"] is None
    (tmp_path / "rejected_candidates.jsonl").unlink()
    assert audit.proposal_audit(tmp_path)["missing_data"]
    assert audit.proposal_audit(tmp_path)["duplicate_rejection_rate"] is None
    assert audit.proposal_audit(None)["missing_data"]


def test_progressive_coverage_counts_informative_outcomes_separately():
    # A confirmed infeasible network adds evidence but no successful observation.
    history = [record(0, .8), record(1, None, raw(activation="relu6"), state="PRUNED", feasibility_status="infeasible")]
    rows, _, _ = audit.aggregate_prefixes(history)
    assert rows[0]["coverage_informative_networks"] == 1
    assert rows[-1]["coverage_informative_networks"] == 2
    assert rows[-1]["coverage_successful_networks"] == 1
    assert rows[-1]["coverage_informative_goal_slots_filled"] == 2
    assert rows[-1]["coverage_cells_meeting_goal"] == 1
    assert not rows[-1]["coverage_goal_complete"]
