# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Audit configured periodic campaigns from read-only Optuna and measurement files.

The JSON manifest supplies ``studies`` with name, db, period_ms, attempt_limit,
paired_mode (raw or historical), raw_roots, and optional exclude_paired,
source_csv/replay_csv/replay_raw_roots. Paths are explicit; no campaign discovery
or manuscript constants are built in. Historical telemetry is preserved as
recorded, without repairing timer wrap. Run with MANIFEST OUTPUT_DIRECTORY.
"""

from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3


def positive(value):
    """Return whether a numeric scalar is positive and finite.

    Returns
    -------
    bool
        False for booleans, missing values, and nonfinite numbers.
    """
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value > 0
    )


def cadence_reasons(active_ms, window_ms, misses, period_ms, runs, tolerance_ms):
    """Return independent recorded cadence eligibility failures.

    Parameters
    ----------
    active_ms, window_ms, misses : float
        Recorded active latency, aggregate window and deadline count.
    period_ms : float
        Release interval.
    runs : int
        Number of releases in the measurement window.
    tolerance_ms : float
        Allowed aggregate window excess due to representation precision.
    """
    reasons = []
    if not positive(active_ms) or active_ms > period_ms:
        reasons.append("cadence_active_latency")
    if not positive(window_ms) or window_ms > runs * period_ms + tolerance_ms:
        reasons.append("cadence_window")
    if misses != 0:
        reasons.append("cadence_deadline_misses")
    return reasons


def outcome(state, attrs):
    """Classify completed source outcomes under the recorded policy.

    Returns
    -------
    str
        Measured outcomes require completed state, success code and energy.
    """
    if (
        state == "COMPLETE"
        and attrs.get("error_code") == 1
        and positive(attrs.get("energy_mj_per_inference"))
    ):
        if attrs.get("feasibility_status") == "feasible":
            return "measured_passed"
        if attrs.get("feasibility_status") == "infeasible":
            return "measured_violation"
    return "not_deployed_or_incomplete"


def audit(manifest):
    """Return rowwise source outcomes, paired decisions and input hashes.

    Parameters
    ----------
    manifest : dict
        Explicit campaign configuration described in the module docstring.

    Returns
    -------
    dict
        Summary, rows, and SHA-256 hashes of all consumed evidence files.
    """
    hashes = {}

    def consume(path):
        """Record a consumed file digest.

        Returns
        -------
        pathlib.Path
            Resolved evidence path."""
        path = Path(path).resolve()
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def records_index(roots):
        """Index record basenames.

        Returns
        -------
        dict
            All matching paths, retaining ambiguity for rejection."""
        index = {}
        for root in roots:
            for path in Path(root).rglob("*.json"):
                index.setdefault(path.name, []).append(path)
        return index

    def raw_pair(paths, index, cfg):
        """Derive pair eligibility from phase records and timing.

        Returns
        -------
        tuple
            Independent rejection reasons and consumed evidence paths."""
        reasons, evidence, phases = [], [], {}
        for remote in paths:
            matches = index.get(Path(remote).name, [])
            if len(matches) != 1:
                reasons.append("raw_record_missing_or_ambiguous")
                continue
            path = consume(matches[0])
            raw = json.loads(path.read_text())
            evidence.append(str(path))
            phase = raw["request"]["phase"]
            if phase in phases:
                reasons.append("duplicate_phase")
            phases[phase] = raw
        if set(phases) != {"back_to_back", "cadenced"}:
            reasons.append("paired_phases_missing")
        for phase, raw in phases.items():
            acceptance = raw.get("acceptance", {})
            metrics = raw.get("result", {}).get("power_metrics", {})
            if acceptance.get("instrumentation_valid") is not True or not all(
                acceptance.get("checks", {}).values()
            ):
                reasons.append(phase + "_instrumentation")
            if acceptance.get("deployment_eligible") is not True:
                reasons.append(phase + "_deployment")
            if not positive(acceptance.get("energy_mj_per_release")):
                reasons.append(phase + "_energy")
            active = (
                metrics.get(
                    "timer_per_inference_s", raw.get("result", {}).get("latency_s", -1)
                )
                * 1000
            )
            if not positive(active) or active > cfg["period_ms"]:
                reasons.append(phase + "_active_latency")
            if phase == "cadenced":
                reasons.extend(
                    cadence_reasons(
                        active,
                        metrics.get("timer_per_window_s", -1) * 1000,
                        metrics.get("deadline_miss_count"),
                        cfg["period_ms"],
                        cfg.get("runs", 10),
                        cfg.get("window_tolerance_ms", 0.001),
                    )
                )
        return sorted(set(reasons)), evidence

    rows = []
    for cfg in manifest["studies"]:
        db = consume(cfg["db"])
        connection = sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)
        columns = {row[1] for row in connection.execute("PRAGMA table_info(trials)")}
        study_id = cfg.get("study_id")
        if "study_id" in columns:
            study_ids = [
                row[0]
                for row in connection.execute("SELECT DISTINCT study_id FROM trials")
            ]
            if study_id is None:
                if len(study_ids) != 1:
                    connection.close()
                    raise ValueError(
                        "Configure study_id for a database with multiple studies"
                    )
                study_id = study_ids[0]
            if study_id not in study_ids:
                connection.close()
                raise ValueError("Configured study_id is absent")
            trials = connection.execute(
                "SELECT trial_id,number,state FROM trials WHERE study_id=? ORDER BY number",
                (study_id,),
            ).fetchall()
        elif study_id is not None:
            connection.close()
            raise ValueError("Database has no study_id column")
        else:
            trials = connection.execute(
                "SELECT trial_id,number,state FROM trials ORDER BY number"
            ).fetchall()
        attrs = {
            tid: {
                key: json.loads(value)
                for key, value in connection.execute(
                    "SELECT key,value_json FROM trial_user_attributes WHERE trial_id=?",
                    (tid,),
                )
            }
            for tid, _, _ in trials
        }
        connection.close()
        source_indices = {}
        if cfg.get("source_csv"):
            exported = list(csv.DictReader(consume(cfg["source_csv"]).open()))
            complete = [
                (tid, number) for tid, number, state in trials if state == "COMPLETE"
            ]
            if len(exported) != len(complete):
                raise ValueError("Source CSV does not match completed trial population")
            for index, ((tid, number), source) in enumerate(zip(complete, exported)):
                for key in ("energy_mj_per_inference", "latency_ms", "error_code"):
                    if float(source[key]) != float(attrs[tid][key]):
                        raise ValueError("Source CSV scalar mismatch")
                if source["feasible"].lower() != str(attrs[tid]["feasible"]).lower():
                    raise ValueError("Source CSV feasibility mismatch")
                source_indices[number] = index
        replay = {}
        if cfg.get("replay_csv"):
            for entry in csv.DictReader(consume(cfg["replay_csv"]).open()):
                index = int(entry["source_row_index"])
                if index in replay:
                    raise ValueError("Duplicate replay source row")
                replay[index] = entry
        raw_index = records_index(cfg.get("raw_roots", []))
        replay_index = records_index(cfg.get("replay_raw_roots", []))
        for tid, number, state in trials:
            if number >= cfg["attempt_limit"]:
                continue
            a = attrs[tid]
            category = outcome(state, a)
            reasons, evidence = [], []
            if category != "measured_passed":
                reasons.append("source_" + category)
                if category == "measured_violation" and cfg["paired_mode"] == "raw":
                    _, evidence = raw_pair(
                        a.get("measurement_record_paths", []), raw_index, cfg
                    )
                    if not evidence:
                        raise ValueError("Measured violation has no raw evidence")
                    if any(
                        json.loads(Path(p).read_text())
                        .get("acceptance", {})
                        .get("instrumentation_valid")
                        is not True
                        for p in evidence
                    ):
                        raise ValueError(
                            "Measured violation instrumentation is invalid"
                        )
            elif cfg["paired_mode"] == "historical":
                reasons.extend(
                    cadence_reasons(
                        a.get("cadenced_active_inference_latency_ms"),
                        a.get("cadenced_window_latency_ms"),
                        a.get("cadenced_deadline_miss_count"),
                        cfg["period_ms"],
                        cfg.get("runs", 10),
                        cfg.get("window_tolerance_ms", 0.001),
                    )
                )
                if not all(
                    positive(a.get(key))
                    for key in (
                        "energy_mj_per_inference",
                        "cadenced_energy_mj_per_window",
                    )
                ):
                    reasons.append("paired_energy")
                if (
                    not positive(a.get("clock_hz"))
                    or a.get("clock_hz") != a.get("cadenced_clock_hz")
                    or a.get("clock_hz") != a.get("cpu_clock_mhz_requested", -1) * 1e6
                ):
                    reasons.append("paired_clock")
            elif cfg["paired_mode"] == "raw":
                reasons, evidence = raw_pair(
                    a.get("measurement_record_paths", []), raw_index, cfg
                )
            elif cfg["paired_mode"] == "replay":
                entry = replay.get(source_indices.get(number))
                if entry is None or entry.get("replay_status") != "completed":
                    reasons.append("replay_missing_or_incomplete")
                else:
                    reasons, evidence = raw_pair(
                        json.loads(entry["target__measurement_record_paths"]),
                        replay_index,
                        cfg,
                    )
            else:
                raise ValueError("Unknown paired_mode")
            eligible = not reasons
            final_reasons = list(reasons)
            if cfg.get("exclude_paired"):
                final_reasons.append("configured_population_exclusion")
            rows.append(
                dict(
                    case_study=manifest.get("case_study", "CS1"),
                    campaign=cfg["name"],
                    study=cfg["name"],
                    period_ms=cfg["period_ms"],
                    trial=number,
                    state=state,
                    source_csv_row=source_indices.get(number),
                    outcome=category,
                    source_rule=a.get("feasibility_rule"),
                    source_status=a.get("feasibility_status"),
                    error_code=a.get("error_code"),
                    instrumentation_reasons=a.get(
                        "measurement_instrumentation_reasons"
                    ),
                    prune_reason=a.get("prune_reason"),
                    source_latency_ms=a.get("latency_ms"),
                    source_constraint=a.get("feasibility_constraints"),
                    cadence_active_ms=a.get("cadenced_active_inference_latency_ms"),
                    cadence_window_ms=a.get("cadenced_window_latency_ms"),
                    cadence_misses=a.get("cadenced_deadline_miss_count"),
                    campaign_passed=category == "measured_passed",
                    paired_eligible=eligible,
                    paired_included=not final_reasons,
                    model_eligible=not final_reasons,
                    eligibility_reasons=reasons,
                    paired_reasons=final_reasons,
                    raw_evidence=evidence,
                    evidence_mode=cfg["paired_mode"],
                )
            )
    summary = dict(
        outcomes=dict(Counter(r["outcome"] for r in rows)),
        total=len(rows),
        paired_eligible=sum(r["paired_eligible"] for r in rows),
        paired_included=sum(r["paired_included"] for r in rows),
        source_passed_pair_rejections=dict(
            Counter(
                reason
                for r in rows
                if r["outcome"] == "measured_passed" and not r["paired_eligible"]
                for reason in r["eligibility_reasons"]
            )
        ),
        studies={
            cfg["name"]: dict(
                outcomes=dict(
                    Counter(r["outcome"] for r in rows if r["study"] == cfg["name"])
                ),
                paired_eligible=sum(
                    r["paired_eligible"] for r in rows if r["study"] == cfg["name"]
                ),
                paired_included=sum(
                    r["paired_included"] for r in rows if r["study"] == cfg["name"]
                ),
            )
            for cfg in manifest["studies"]
        },
    )
    return dict(summary=summary, rows=rows, sources_sha256=hashes)


def main():
    """Write a configured audit report and rowwise decisions.

    Returns
    -------
    None
        Audit artifacts are written to the requested output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = audit(json.loads(args.manifest.read_text()))
    result["sources_sha256"][str(args.manifest.resolve())] = hashlib.sha256(
        args.manifest.read_bytes()
    ).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in result.items():
        (args.output / (name + ".json")).write_text(json.dumps(value, indent=2) + "\n")
    with (args.output / "trial_outcomes.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["rows"][0]))
        writer.writeheader()
        writer.writerows(result["rows"])
    (args.output / "REPORT.md").write_text(
        "# Periodic campaign audit\n\n"
        + "Source outcomes follow recorded configured feasibility, not retrospective cadence eligibility. "
        "Each attempt consumes budget before filtering. Historical timing is used as recorded; no wrap repair is inferred. "
        "Paired eligibility is independently derived from phase records or historical cadence telemetry. "
        "Configured population exclusions are applied after eligibility.\n\n```json\n"
        + json.dumps(result["summary"], indent=2)
        + "\n```\n\nSee trial_outcomes.csv for every inclusion decision and sources_sha256.json for evidence hashes.\n"
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
