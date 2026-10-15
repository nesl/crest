#!/usr/bin/env python3
"""Read-only CS3 network-mean trajectories and repeat-variability audit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from crest.pipeline_types import TrialRecord
from crest.optimizers.llm.full_history import (
    DECODED_SCHEMA_VERSION, RAW_FIELDS, build_coverage, decoded_fingerprint,
    outcome_class, reconstruct_params,
)

METRIC_POLICY = {
    "version": "cs3-search-audit-v3", "identity": DECODED_SCHEMA_VERSION,
    "repeat_aggregation": "arithmetic mean per decoded network using only observations available at each completion prefix; later repeats never revise earlier points",
    "curve_shape": "best network mean and top-network statistics may decrease; historical best individual observation is monotonic",
    "ranking": "descending network mean, then lexicographic decoded ID for ties; include all networks when fewer than ten",
    "variance": "separate exact-raw and decoded-network pooled macro-F1 squared deviations / sum(n_group-1); repeat groups require n >= 2",
    "failure_policy": "failed/infeasible/pruned/infra attempts add no fabricated score",
    "comparability": "one study per report; verify comparable training/data settings before cross-study comparison",
    "time": "completed_trial_wall_seconds_proxy sums COMPLETE durations; all_attempt_wall_seconds_proxy includes FAIL/PRUNED; missing duration counts and separate completeness/lower-bound flags accompany both observed sums; neither is measured compute-active time or elapsed time",
    "coverage": "two informative distinct networks per resolved block-count/stride cell is breadth over 29 cells (at least 58 networks), not coverage of the full architecture space",
    "allocation": "2/3 -> 3/2 -> 4/1 is a fixed hypothesis, not an inferred optimal response to a plateau; no adaptive plateau rule",
}


def read_history(path: Path, study_name: str | None = None) -> tuple[tuple[TrialRecord, ...], dict]:
    """Read immutable trials and elapsed durations through SQLite read-only mode.

    Parameters
    ----------
    path : Path
        Saved Optuna database.
    study_name : str or None
        Exact study; required when the database contains multiple studies.

    Returns
    -------
    tuple
        Trials and metadata containing completion order and per-trial durations.
    """
    from datetime import datetime
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")
        studies = db.execute("SELECT study_id,study_name FROM studies").fetchall()
        selected = [s for s in studies if study_name is None or s[1] == study_name]
        if len(selected) != 1:
            raise ValueError("Select exactly one study with --study-name")
        study_id, name = selected[0]
        result, durations, starts, completions = [], {}, {}, {}
        for tid, number, state, start, end in db.execute(
                "SELECT trial_id,number,state,datetime_start,datetime_complete FROM trials WHERE study_id=? ORDER BY number", (study_id,)):
            params = {}
            for key, value, distribution in db.execute(
                    "SELECT param_name,param_value,distribution_json FROM trial_params WHERE trial_id=?", (tid,)):
                distribution = json.loads(distribution)
                if distribution["name"] == "CategoricalDistribution":
                    value = distribution["attributes"]["choices"][int(value)]
                elif distribution["name"].startswith("Int"):
                    value = int(value)
                params[key] = value
            attrs = [dict((k, json.loads(v)) for k, v in db.execute(
                f"SELECT key,value_json FROM {table} WHERE trial_id=?", (tid,)))
                for table in ("trial_user_attributes", "trial_system_attributes")]
            values = tuple(v if kind == "FINITE" else None for v, kind in db.execute(
                "SELECT value,value_type FROM trial_values WHERE trial_id=? ORDER BY objective", (tid,))) or None
            result.append(TrialRecord(number, state, params, values, attrs[0], attrs[1]))
            if start and end:
                durations[number] = max(0.0, (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds())
            if start:
                starts[number] = datetime.fromisoformat(start).timestamp()
            if end:
                completions[number] = datetime.fromisoformat(end).timestamp()
    terminal = [t for t in result if t.state.upper() in {"COMPLETE", "FAIL", "PRUNED"}]
    if any(t.number not in completions for t in terminal):
        raise ValueError("Terminal trials need completion timestamps for an as-observed trajectory")
    origin = min(starts.values()) if starts else None
    order = {t.number: index for index, t in enumerate(sorted(terminal, key=lambda t: (completions[t.number], t.number)))}
    return tuple(result), {"study_name": name, "durations_seconds": durations,
        "completion_order": order, "elapsed_seconds": {
            n: max(0.0, end - origin) for n, end in completions.items()} if origin is not None else {}}


def _finite(value):
    """Check that a metric is a finite real value rather than a boolean.

    Parameters
    ----------
    value : object
        Persisted metric value.

    Returns
    -------
    bool
        Whether the value can participate in numeric aggregation.
    """
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def aggregate_prefixes(history, durations=None, *, completion_order=None, elapsed_seconds=None):
    """Apply the same decoded repeat rule to all prefixes and both score metrics.

    Parameters
    ----------
    history : iterable of TrialRecord
        Comparable training/data observations from one study.
    durations : dict or None
        Trial-number to terminal evaluation-wall-duration proxy.
    completion_order : dict or None
        Observed completion order from the database; absent for serial fixtures.
    elapsed_seconds : dict or None
        Wall-clock time from first trial start through each completion.

    Returns
    -------
    tuple
        Prefix metric rows, final per-network observations, and pooled raw F1
        dispersion with separately labeled decoded-network dispersion.
    """
    durations = durations or {}
    groups = {metric: {} for metric in ("score", "macro_f1")}
    raw_f1 = {}
    rows, elapsed, completed_time, feasible, attempts = [], 0.0, 0.0, 0, 0
    missing_durations, complete_count, missing_completed_durations = 0, 0, 0
    observed = []
    coverage_finished_at = None
    for trial in sorted(history, key=lambda t: (completion_order.get(t.number, float("inf"))
                                               if completion_order is not None else t.number)):
        if trial.state.upper() not in {"COMPLETE", "FAIL", "PRUNED"}:
            continue
        attempts += 1
        elapsed += durations.get(trial.number, 0.0)
        missing_durations += trial.number not in durations
        if trial.state.upper() == "COMPLETE":
            completed_time += durations.get(trial.number, 0.0)
            complete_count += 1
            missing_completed_durations += trial.number not in durations
        observed.append(trial)
        coverage = build_coverage(observed)
        coverage_done = coverage["goal_complete"]
        cells, target = coverage["cells"], coverage["target_per_cell"]
        if coverage_done and coverage_finished_at is None:
            coverage_finished_at = attempts
        success = outcome_class(trial) == "successful"
        feasible += int(success and (trial.user_attrs.get("feasibility_status") == "feasible" or trial.user_attrs.get("feasible") is True))
        params = reconstruct_params(trial)
        identity = decoded_fingerprint(params)
        macro = (trial.user_attrs.get("task_metrics") or {}).get("macro_f1")
        score = trial.values[0] if trial.values and len(trial.values) == 1 else None
        if success and identity:
            for metric, value in (("score", score), ("macro_f1", macro)):
                if _finite(value) and (metric != "macro_f1" or 0 <= value <= 1):
                    groups[metric].setdefault(identity, []).append((trial.number, float(value)))
            if _finite(macro) and 0 <= macro <= 1:
                raw_key = json.dumps({k: params[k] for k in RAW_FIELDS}, sort_keys=True, separators=(",", ":"))
                raw_f1.setdefault(raw_key, []).append(float(macro))
        for metric, networks in groups.items():
            ranked = sorted(networks, key=lambda identity: (-statistics.mean(v for _, v in networks[identity]), identity))
            means = [statistics.mean(v for _, v in networks[identity]) for identity in ranked]
            top = means[:10]
            observations = [value for group in networks.values() for _, value in group]
            rows.append({"trial_number": trial.number, "attempts": attempts, "feasible_completions": feasible,
                "completion_prefix": attempts, "completed_trials": complete_count,
                "evaluation_wall_seconds_proxy": elapsed,
                "all_attempt_wall_seconds_proxy": elapsed,
                "completed_trial_wall_seconds_proxy": completed_time,
                "missing_duration_count": missing_durations,
                "all_attempt_duration_complete": missing_durations == 0,
                "all_attempt_duration_is_lower_bound": missing_durations > 0,
                "completed_trial_missing_duration_count": missing_completed_durations,
                "completed_trial_duration_complete": missing_completed_durations == 0,
                "completed_trial_duration_is_lower_bound": missing_completed_durations > 0,
                "elapsed_wall_seconds": (elapsed_seconds or {}).get(trial.number),
                "coverage_goal_complete": coverage_done,
                "coverage_cell_count": len(cells), "coverage_target_per_cell": target,
                "coverage_cells_meeting_goal": sum(c["informative"] >= target for c in cells),
                "coverage_informative_networks": sum(c["informative"] for c in cells),
                "coverage_successful_networks": sum(c["successful"] for c in cells),
                "coverage_goal_slots": len(cells) * target,
                "coverage_informative_goal_slots_filled": sum(min(c["informative"], target) for c in cells),
                "coverage_first_complete_prefix": coverage_finished_at,
                "metric": metric, "unique_scored_networks": len(means),
                "top_count": len(top),
                "best_network_id": ranked[0] if ranked else None,
                "best_network_observation_count": len(networks[ranked[0]]) if ranked else None,
                "top10_single_observation_networks": sum(len(networks[i]) == 1 for i in ranked[:10]), "best_network_mean": max(means) if means else None,
                "top10_mean": statistics.mean(top) if top else None,
                "top10_median": statistics.median(top) if top else None,
                "best_individual_observed": max(observations) if observations else None,
                "valid_observations": len(observations),
                "repeat_groups": sum(len(v) > 1 for v in networks.values()),
                "repeated_observations": sum(len(v) for v in networks.values() if len(v) > 1)})
    networks = [{"metric": metric, "decoded_id": identity, "observations": [{"trial_number": n, "value": v} for n, v in values],
                 "count": len(values), "mean": statistics.mean(v for _, v in values),
                 "sample_sd": statistics.stdev(v for _, v in values) if len(values) > 1 else None}
                for metric, by_id in groups.items() for identity, values in sorted(by_id.items())]
    pooled = pooled_dispersion(raw_f1.values())
    # Preserve the original return keys for callers using the raw estimate.
    pooled["decoded_network"] = pooled_dispersion(
        [[v for _, v in values] for values in groups["macro_f1"].values()])
    return rows, networks, pooled


def pooled_dispersion(groups):
    """Compute pooled sample dispersion.

    Parameters
    ----------
    groups : iterable of sequences of float
        Observations grouped by a single declared identity rule.

    Returns
    -------
    dict
        Repeat sample counts, squared deviations, and pooled SD.
    """
    repeats = [v for v in groups if len(v) > 1]
    denominator = sum(len(v) - 1 for v in repeats)
    sum_squares = sum(sum((x - statistics.mean(values)) ** 2 for x in values) for values in repeats)
    return {"repeat_groups": len(repeats), "measurements": sum(map(len, repeats)),
            "degrees_of_freedom": denominator, "sum_squares": sum_squares,
            "macro_f1_sd": math.sqrt(sum_squares / denominator) if denominator else None}


def proposal_audit(ledger_dir):
    """Summarize saved proposal decisions without inferring rejects from trials.

    Parameters
    ----------
    ledger_dir : Path or None
        Candidate ledger root, excluding nested memory ledgers.

    Returns
    -------
    dict
        Mutually exclusive rejection categories and explicit observed denominator.
        Missing streams make population rates unknown; observed rates remain
        labeled conditional on the saved decision records.
    """
    if ledger_dir is None:
        return {"missing_data": True, "reason": "No saved LLM ledger supplied",
                "duplicate_rejection_rate": None}
    root = Path(ledger_dir)
    def records(name):
        path = root / name
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    returned = records("returned_candidates.jsonl")
    legacy = not (root / "returned_candidates.jsonl").exists()
    if legacy:
        returned = records("accepted_candidates.jsonl")
    rejected = records("rejected_candidates.jsonl")
    malformed_codes = {"malformed_response", "invalid_json", "schema_validation"}
    malformed = [r for r in rejected if r.get("code") in malformed_codes]
    candidate_rejects = [r for r in rejected if r.get("code") not in malformed_codes]
    fallback = sum(r.get("proposal_source") == "random_fallback" for r in returned)
    accepted = len(returned) - fallback
    denominator = accepted + len(candidate_rejects)
    raw = sum(r.get("code") in {"duplicate_raw", "duplicate", "duplicate_completed", "duplicate_queued", "duplicate_batch"} for r in candidate_rejects)
    decoded = sum(r.get("code") == "duplicate_decoded" for r in candidate_rejects)
    missing = legacy or not (root / "rejected_candidates.jsonl").exists()
    events = records("optimizer_events.jsonl")
    refill_known = any("refill_calls" in e for e in events)
    provider_errors = sum(e.get("event") == "provider_error" for e in events)
    missing = missing or bool(provider_errors) or not (root / "optimizer_events.jsonl").exists()
    return {"missing_data": missing, "provider_error_rounds": provider_errors,
            "legacy_acceptance_stream": legacy,
            "population_completeness": "saved decisions only; retained candidates lost before return are not observable; provider errors or missing streams make the population rate unknown",
            "denominator_definition": "saved LLM returned candidates plus candidate-level rejection records; excludes malformed calls and random fallback; not evaluated trials or all raw response entries",
            "observed_candidate_decisions": denominator, "llm_returned_candidates": accepted,
            "raw_duplicate_rejections": raw, "decoded_duplicate_rejections": decoded,
            "other_candidate_rejections": len(candidate_rejects) - raw - decoded,
            "malformed_calls": len(malformed), "fallback_candidates": fallback,
            "refill_attempts": sum(e.get("refill_calls", 0) for e in events) if refill_known else None,
            "refill_definition": "sum of explicitly recorded refill_calls on batch lifecycle events; unknown if no such counters exist",
            "refill_data_missing": not refill_known,
            "observed_duplicate_rejection_rate": (raw + decoded) / denominator if denominator else None,
            "duplicate_rejection_rate": (raw + decoded) / denominator if denominator and not missing else None,
            "raw_duplicate_rejection_rate": raw / denominator if denominator and not missing else None,
            "decoded_duplicate_rejection_rate": decoded / denominator if denominator and not missing else None}


def api_overhead(ledger_dir):
    """Report recorded request wall time separately from trial durations.

    Parameters
    ----------
    ledger_dir : Path or None
        Saved LLM ledger, never a live service connection.

    Returns
    -------
    dict
        Recorded response latency sum and missing timings; no invented zero
        overhead when the ledger or failed-request timing is unavailable.
    """
    if ledger_dir is None:
        return {"recorded_api_wall_seconds": None, "timing_complete": False,
                "reason": "No saved LLM ledger supplied"}
    requests = list(Path(ledger_dir).rglob("*.request.json"))
    total, missing = 0.0, 0
    for request in requests:
        response = request.with_name(request.name.replace(".request.json", ".response.json"))
        value = json.loads(response.read_text()).get("latency_ms") if response.exists() else None
        if _finite(value) and value >= 0:
            total += value / 1000
        else:
            missing += 1
    return {"recorded_api_wall_seconds": total if requests else None,
            "ledger_path": str(Path(ledger_dir).resolve()),
            "request_count": len(requests), "missing_timing_count": missing,
            "timing_complete": bool(requests) and missing == 0,
            "definition": "sum across all saved candidate and memory requests in the supplied ledger, including retries/backoff when recorded; excludes untimed client processing and unresolved calls; ledger capture may differ from DB capture"}


def run(args):
    """Save immutable-input provenance, metric definitions, CSV and JSON evidence.

    Parameters
    ----------
    args : argparse.Namespace
        Input saved database, optional study, and fresh output directory.

    Returns
    -------
    dict
        Complete frozen audit report.
    """
    path = Path(args.history_db).resolve()
    history, meta = read_history(path, args.study_name)
    rows, networks, pooled = aggregate_prefixes(history, meta.pop("durations_seconds"),
        completion_order=meta.pop("completion_order"), elapsed_seconds=meta.pop("elapsed_seconds"))
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except subprocess.SubprocessError:
        revision = None
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {"input": {"path": str(path), "sha256": digest, **meta}, "code_revision": revision,
              "code_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (Path(__file__).resolve(), ROOT / "src/crest/optimizers/llm/full_history.py", ROOT / "src/crest/audio_dscnn_decode.py")},
              "input_sidecars": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (Path(str(path) + "-wal"),) if p.exists()},
              "metric_definitions": METRIC_POLICY, "terminal_prefixes": len(rows) // 2,
              "networks": networks, "pooled_within_raw_f1": {k: v for k, v in pooled.items() if k != "decoded_network"},
              "pooled_within_decoded_f1": pooled["decoded_network"],
              "proposal_audit": proposal_audit(getattr(args, "ledger_dir", None)), "coverage": build_coverage(history),
              "api_overhead": api_overhead(getattr(args, "ledger_dir", None)),
              "coverage_first_complete_prefix": next((r["completion_prefix"] for r in rows if r["coverage_goal_complete"]), None)}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (output / "trajectories.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["trial_number", "metric"])
        writer.writeheader()
        writer.writerows(rows)
    return report


def main(argv=None):
    """Parse the offline audit command.

    Parameters
    ----------
    argv : list of str or None
        Explicit arguments or process command line.

    Returns
    -------
    int
        Zero on successful read-only analysis.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-db", type=Path, required=True)
    parser.add_argument("--study-name")
    parser.add_argument("--ledger-dir", type=Path, help="Optional saved LLM ledger for recorded API wall-time totals")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run(args)
    print(json.dumps({"input": report["input"], "pooled_within_raw_f1": report["pooled_within_raw_f1"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
