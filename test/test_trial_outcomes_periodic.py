# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Synthetic regression tests for source policy versus paired eligibility."""

import json
import sqlite3
from analysis_scripts.trial_outcomes.periodic import audit, cadence_reasons, outcome


def test_source_policy_is_distinct_from_cadence():
    attrs = {
        "error_code": 1,
        "feasibility_status": "feasible",
        "energy_mj_per_inference": 1,
    }
    assert outcome("COMPLETE", attrs) == "measured_passed"
    assert cadence_reasons(198, 2148, 10, 200, 10, 0.001) == [
        "cadence_window",
        "cadence_deadline_misses",
    ]
    assert cadence_reasons(201, 2000, 0, 200, 10, 0.001) == ["cadence_active_latency"]
    assert not cadence_reasons(200, 2000.001, 0, 200, 10, 0.001)
    assert cadence_reasons(None, None, None, 200, 10, 0.001)


def test_attempt_budget_and_population_exclusion(tmp_path):
    db = tmp_path / "study.db"
    connection = sqlite3.connect(db)
    connection.executescript(
        "CREATE TABLE trials(trial_id INTEGER,number INTEGER,state TEXT); CREATE TABLE trial_user_attributes(trial_id INTEGER,key TEXT,value_json TEXT);"
    )
    good = dict(
        error_code=1,
        feasibility_status="feasible",
        cadenced_active_inference_latency_ms=10,
        cadenced_window_latency_ms=500,
        cadenced_deadline_miss_count=0,
        energy_mj_per_inference=1,
        cadenced_energy_mj_per_window=2,
        clock_hz=1e6,
        cadenced_clock_hz=1e6,
        cpu_clock_mhz_requested=1,
    )
    for number, state, attrs in [
        (0, "FAIL", {}),
        (1, "COMPLETE", good),
        (2, "COMPLETE", good),
    ]:
        connection.execute("INSERT INTO trials VALUES(?,?,?)", (number, number, state))
        for key, value in attrs.items():
            connection.execute(
                "INSERT INTO trial_user_attributes VALUES(?,?,?)",
                (number, key, json.dumps(value)),
            )
    connection.commit()
    connection.close()
    cfg = dict(
        name="synthetic",
        db=str(db),
        period_ms=50,
        attempt_limit=2,
        paired_mode="historical",
        exclude_paired=True,
    )
    result = audit({"case_study": "example", "studies": [cfg]})
    assert result["summary"]["total"] == 2
    assert result["summary"]["paired_eligible"] == 1
    assert result["summary"]["paired_included"] == 0
    row = result["rows"][1]
    assert row["trial"] == 1 and row["case_study"] == "example"
    assert (
        row["campaign_passed"] and row["paired_eligible"] and not row["model_eligible"]
    )
    assert row["paired_reasons"] == ["configured_population_exclusion"]
    assert len(result["sources_sha256"]) == 1


def test_complete_csv_rows_do_not_replace_attempt_identity(tmp_path):
    import csv

    db = tmp_path / "study.db"
    connection = sqlite3.connect(db)
    connection.executescript(
        "CREATE TABLE trials(trial_id INTEGER,number INTEGER,state TEXT); CREATE TABLE trial_user_attributes(trial_id INTEGER,key TEXT,value_json TEXT);"
    )
    good = dict(
        error_code=1,
        feasibility_status="feasible",
        feasible=True,
        energy_mj_per_inference=1,
        latency_ms=5,
    )
    for number, state in [(0, "FAIL"), (1, "COMPLETE"), (2, "COMPLETE")]:
        connection.execute("INSERT INTO trials VALUES(?,?,?)", (number, number, state))
        if state == "COMPLETE":
            for key, value in good.items():
                connection.execute(
                    "INSERT INTO trial_user_attributes VALUES(?,?,?)",
                    (number, key, json.dumps(value)),
                )
    connection.commit()
    connection.close()
    source = tmp_path / "source.csv"
    with source.open("w") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "error_code",
                "energy_mj_per_inference",
                "latency_ms",
                "feasible",
            ],
        )
        writer.writeheader()
        for _ in range(2):
            writer.writerow({key: good[key] for key in writer.fieldnames})
    result = audit(
        {
            "studies": [
                dict(
                    name="synthetic",
                    db=str(db),
                    source_csv=str(source),
                    period_ms=50,
                    attempt_limit=2,
                    paired_mode="historical",
                )
            ]
        }
    )
    assert [(row["trial"], row["source_csv_row"]) for row in result["rows"]] == [
        (0, None),
        (1, 0),
    ]
    assert not result["rows"][1]["paired_eligible"]
    assert "cadence_deadline_misses" in result["rows"][1]["paired_reasons"]


def test_stale_success_attributes_are_incomplete():
    from analysis_scripts.trial_outcomes.periodic import positive

    attrs = dict(error_code=1, feasibility_status="feasible", energy_mj_per_inference=1)
    assert outcome("RUNNING", attrs) == "not_deployed_or_incomplete"
    assert outcome("FAIL", attrs) == "not_deployed_or_incomplete"
    assert not positive(True)
    assert (
        outcome("COMPLETE", dict(attrs, energy_mj_per_inference=-1))
        == "not_deployed_or_incomplete"
    )


def test_multistudy_requires_explicit_identity(tmp_path):
    import pytest

    db = tmp_path / "multiple.db"
    connection = sqlite3.connect(db)
    connection.executescript(
        'CREATE TABLE trials(trial_id INTEGER,study_id INTEGER,number INTEGER,state TEXT); CREATE TABLE trial_user_attributes(trial_id INTEGER,key TEXT,value_json TEXT); INSERT INTO trials VALUES(1,3,0,"FAIL"); INSERT INTO trials VALUES(2,4,0,"FAIL");'
    )
    connection.close()
    cfg = dict(
        name="synthetic",
        db=str(db),
        period_ms=50,
        attempt_limit=2,
        paired_mode="historical",
    )
    with pytest.raises(ValueError, match="study_id"):
        audit({"studies": [cfg]})
    cfg["study_id"] = 4
    result = audit({"studies": [cfg]})
    assert result["summary"]["total"] == 1
