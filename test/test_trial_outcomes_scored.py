# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Synthetic scored outcome coverage without optimizer or provider dependencies."""

import json
import sqlite3

import pytest

from analysis_scripts.trial_outcomes.scored import audit


def database(tmp_path):
    path = tmp_path / "archive.db"
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
        create table studies(study_id integer, study_name text);
        insert into studies values(1,'synthetic');
        create table trials(trial_id integer, number integer, state text, datetime_start text, datetime_complete text, study_id integer);
        create table trial_user_attributes(trial_id integer, key text, value_json text);
        create table trial_values(trial_id integer, objective integer, value real);
        create table trial_params(trial_id integer, param_name text, param_value real, distribution_json text);
        """
        )
        cases = [
            ("RUNNING", {}),
            ("WAITING", {}),
            ("FAIL", {}),
            ("PRUNED", {}),
            (
                "COMPLETE",
                {
                    "feasibility_status": "infeasible",
                    "energy_mj_per_inference": 1,
                    "latency_ms": 2,
                },
            ),
            (
                "COMPLETE",
                {
                    "feasibility_status": "infeasible",
                    "energy_mj_per_inference": 1,
                    "latency_ms": 2,
                    "measurement_instrumentation_valid": False,
                },
            ),
            (
                "COMPLETE",
                {
                    "feasibility_status": "infeasible",
                    "energy_mj_per_inference": 0,
                    "latency_ms": 2,
                },
            ),
            (
                "COMPLETE",
                {"feasible": True, "measurement_record_paths": ["retry1", "retry2"]},
            ),
            ("FAIL", {}),
            ("COMPLETE", {"feasible": True}),
        ]
        for n, (state, attrs) in enumerate(cases):
            conn.execute(
                "insert into trials values(?,?,?,?,?,1)",
                (n, n, state, "2026-01-01T00:00:00", f"2026-01-01T00:00:{n+1:02}"),
            )
            conn.execute("insert into trial_values values(?,0,1)", (n,))
            conn.executemany(
                "insert into trial_user_attributes values(?,?,?)",
                [(n, k, json.dumps(v)) for k, v in attrs.items()],
            )
    return path


def test_terminal_prefix_and_measured_evidence(tmp_path):
    result = audit(
        {"campaigns": [{"name": "example", "database": str(database(tmp_path))}]}, 1
    )
    assert result["terminal_attempts"] == 6
    assert result["totals"] == {
        "measured_passed": 1,
        "measured_violation": 1,
        "not_deployed_or_incomplete": 4,
    }
    assert result["campaigns"][0]["endpoint_reached"]
    assert result["trials"][-1]["trial"] == 7
    assert "measurement_record_paths" not in result["trials"][-1]
    assert "attributes_json" not in result["trials"][-1]


def test_arbitrary_and_unreached_endpoint(tmp_path):
    config = {"campaigns": [{"name": "example", "database": str(database(tmp_path))}]}
    assert audit(config, 2)["totals"]["measured_passed"] == 2
    result = audit(config, 3)
    assert result["terminal_attempts"] == 8
    assert not result["campaigns"][0]["endpoint_reached"]
    with pytest.raises(ValueError, match="positive integer"):
        audit(config, 0)


def test_contradictory_feasible_evidence_keeps_recorded_endpoint(tmp_path):
    path = database(tmp_path)
    with sqlite3.connect(path) as conn:
        conn.execute(
            "insert into trial_user_attributes values(7,?,?)",
            ("measurement_instrumentation_valid", "false"),
        )
    result = audit({"campaigns": [{"name": "example", "database": str(path)}]}, 1)
    assert result["campaigns"][0]["endpoint_reached"]
    assert result["trials"][-1]["trial"] == 7
    assert result["trials"][-1]["outcome"] == "not_deployed_or_incomplete"


@pytest.mark.parametrize(
    "field,value", [("hil_error_code", 3), ("energy_mj_per_inference", -1)]
)
def test_contradictory_feasible_failure(tmp_path, field, value):
    path = database(tmp_path)
    with sqlite3.connect(path) as conn:
        conn.execute(
            "insert into trial_user_attributes values(7,?,?)",
            (field, json.dumps(value)),
        )
    result = audit({"campaigns": [{"name": "example", "database": str(path)}]}, 1)
    assert result["trials"][-1]["outcome"] == "not_deployed_or_incomplete"


def test_direct_cli_help_outside_repository(tmp_path):
    from pathlib import Path
    import subprocess
    import sys

    script = (
        Path(__file__).resolve().parents[1]
        / "analysis_scripts/trial_outcomes/scored.py"
    )
    process = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr
    assert "--endpoint" in process.stdout
