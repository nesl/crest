# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Audit final legacy trial records without counting replays or internal retries.

The passed convention is historical recorded eligibility: COMPLETE, success,
nonpruned, and no explicit false feasibility flag. Missing feasibility is
reported separately; it does not establish modern instrumentation validity.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import sqlite3


def classify(record, *, host_only=False):
    """Return disjoint outcome and evidence for one stored final record.

    Parameters
    ----------
    record : dict
        Final state and attributes.
    host_only : bool, optional
        Identify synthetic host objective success.

    Returns
    -------
    tuple of str
        Outcome and supporting evidence.
    """
    attrs = record.get("attrs", {})
    label = attrs.get("error_code_label")
    reason = attrs.get("prune_reason", "")
    if record.get("state") != "COMPLETE":
        return "not_deployed_or_incomplete", "state_" + str(
            record.get("state", "missing")
        )
    if label in {"HIL_MASTER_ARENA_EXHAUSTED", "HIL_MASTER_FLASH_OVERFLOW"}:
        return "not_deployed_or_incomplete", "flash_or_allocation_failure"
    if reason == "TFLite evaluation failed":
        return "not_deployed_or_incomplete", "task_evaluation_failed"
    if label != "HIL_MASTER_SUCCESS":
        return "not_deployed_or_incomplete", label or "missing_success_evidence"
    if attrs.get("pruned"):
        return "not_deployed_or_incomplete", "pruned_without_complete_candidate"
    if attrs.get("feasible") is False:
        return "measured_violation", "explicit_infeasible_success"
    return "measured_passed", (
        "host_objective_completed" if host_only else "recorded_operational_eligibility"
    )


def read_sqlite(path, *, study_id=None, trial_min=0, trial_max=None):
    """Read Optuna trial rows read-only within inclusive/exclusive bounds.

    Parameters
    ----------
    path : path-like
        Optuna SQLite database.
    study_id : int, optional
        Required when the database has multiple studies.
    trial_min, trial_max : int, optional
        Inclusive lower and exclusive upper trial-number bounds.

    Returns
    -------
    list of dict
        Final records with original state, attributes, and objectives.
    """
    with sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True) as conn:
        studies = [x[0] for x in conn.execute("SELECT study_id FROM studies")]
        if study_id is None:
            if len(studies) != 1:
                raise ValueError(
                    "study_id is required for a database without exactly one study"
                )
            study_id = studies[0]
        if study_id not in studies:
            raise ValueError("study_id is absent from database")
        rows = []
        for tid, number, state in conn.execute(
            "SELECT trial_id,number,state FROM trials WHERE study_id=? ORDER BY number",
            (study_id,),
        ):
            if number < trial_min or (trial_max is not None and number >= trial_max):
                continue
            attrs = {
                k: json.loads(v)
                for k, v in conn.execute(
                    "SELECT key,value_json FROM trial_user_attributes WHERE trial_id=?",
                    (tid,),
                )
            }
            values = [
                v
                for v, in conn.execute(
                    "SELECT value FROM trial_values WHERE trial_id=? ORDER BY objective",
                    (tid,),
                )
            ]
            rows.append(dict(trial=number, state=state, attrs=attrs, values=values))
        return rows


def read_export(path, campaign, *, trial_min=0, trial_max=None):
    """Read a saved JSON or gzip/base64 JSON campaign export.

    Parameters
    ----------
    path : path-like
        Saved export file.
    campaign : str
        Campaign key for a multi-campaign export.
    trial_min, trial_max : int, optional
        Inclusive lower and exclusive upper trial bounds.

    Returns
    -------
    list of dict
        Selected final source records.
    """
    path = Path(path)
    data = path.read_bytes()
    if path.name.endswith(".gz.b64"):
        data = gzip.decompress(base64.b64decode(data))
    elif path.suffix == ".gz":
        data = gzip.decompress(data)
    decoded = json.loads(data)
    records = (
        decoded["records"] if "records" in decoded else decoded[campaign]["records"]
    )
    return [
        r
        for r in records
        if r["trial"] >= trial_min and (trial_max is None or r["trial"] < trial_max)
    ]


def audit_campaign(records, *, campaign, host_only=False, case_study="CS2"):
    """Return summary and one evidence row per unique trial identity.

    Parameters
    ----------
    records : iterable of dict
        Final records from one campaign.
    campaign, case_study : str
        Explicit identity labels.
    host_only : bool, optional
        Report host objectives separately from native measurements.

    Returns
    -------
    tuple
        Summary mapping and evidence rows.
    """
    rows = []
    seen = set()
    for record in records:
        number = record["trial"]
        if number in seen:
            raise ValueError(f"duplicate trial number in campaign {campaign}: {number}")
        seen.add(number)
        outcome, evidence = classify(record, host_only=host_only)
        attrs = record.get("attrs", {})
        rows.append(
            dict(
                case_study=case_study,
                campaign=campaign,
                trial=number,
                state=record.get("state"),
                outcome=outcome,
                evidence=evidence,
                error_code_label=attrs.get("error_code_label"),
                pruned=attrs.get("pruned"),
                feasible=attrs.get("feasible"),
                feasibility_present="feasible" in attrs,
                host_only=host_only,
            )
        )
    counts = Counter(r["outcome"] for r in rows)
    summary = dict(
        campaign=campaign,
        host_only=host_only,
        attempts=len(rows),
        **{
            k: counts[k]
            for k in (
                "measured_passed",
                "measured_violation",
                "not_deployed_or_incomplete",
            )
        },
        states=dict(Counter(r["state"] for r in rows)),
        evidence=dict(Counter(r["evidence"] for r in rows)),
        passed_feasibility_missing=sum(
            r["outcome"] == "measured_passed" and not r["feasibility_present"]
            for r in rows
        ),
        instrumentation_failures=None,
        recovered_retries=None,
    )
    return summary, rows


def main(argv=None):
    """Audit explicit configuration and write summary JSON and native CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="JSON list: campaign, db or export+export_campaign, study_id, trial_min, trial_max, host_only",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    summaries, all_rows = [], []
    for spec in json.loads(args.config.read_text()):
        bounds = {k: spec[k] for k in ("trial_min", "trial_max") if k in spec}
        if "db" in spec:
            records = read_sqlite(spec["db"], study_id=spec.get("study_id"), **bounds)
        else:
            records = read_export(spec["export"], spec["export_campaign"], **bounds)
        summary, rows = audit_campaign(
            records,
            campaign=spec["campaign"],
            host_only=spec.get("host_only", False),
            case_study=spec.get("case_study", "CS2"),
        )
        summaries.append(summary)
        all_rows.extend(rows)
    native = [s for s in summaries if not s["host_only"]]
    result = dict(
        campaigns=summaries,
        native_totals={
            k: sum(s[k] for s in native)
            for k in (
                "attempts",
                "measured_passed",
                "measured_violation",
                "not_deployed_or_incomplete",
            )
        },
        scope="Final native search records only; host objectives reported separately. Replays and internal retries excluded.",
        eligibility="Historical recorded eligibility; absent feasibility flags do not prove independently validated feasibility.",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.output_dir / "trial_outcomes.csv").open("w", newline="") as stream:
        fields = (
            list(all_rows[0])
            if all_rows
            else ["campaign", "trial", "state", "outcome", "evidence"]
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(r for r in all_rows if not r["host_only"])
    print(json.dumps(result["native_totals"]))


if __name__ == "__main__":
    main()
