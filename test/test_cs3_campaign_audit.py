# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Synthetic campaign audit tests avoid providers and private archives."""

import importlib.util
import json
from pathlib import Path
import sqlite3
import pytest

spec = importlib.util.spec_from_file_location(
    "campaign",
    Path(__file__).resolve().parents[1] / "analysis_scripts/cs3_audit/campaign.py",
)
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


def test_effort_endpoint_and_missingness():
    rows = [
        dict(
            number=i,
            state=state,
            feasible=valid,
            duration_seconds=duration,
            proposal_source=source,
        )
        for i, (state, valid, duration, source) in enumerate(
            [
                ("FAIL", False, 10, "random_fallback"),
                ("COMPLETE", True, 20, "warmup"),
                ("PRUNED", False, None, "llm"),
                ("COMPLETE", True, 30, "llm"),
                ("COMPLETE", True, 99, "llm"),
            ]
        )
    ]
    result = campaign.effort(rows, 2)
    assert result["terminal_attempts"] == 4
    assert result["complete_trial_hours"] == 50 / 3600
    assert result["all_terminal_trial_hours"] == 60 / 3600
    assert result["missing_duration_count"] == 1
    assert result["proposal_sources"]["random_fallback"] == 1
    assert not campaign.effort(rows, 4)["endpoint_reached"]
    with pytest.raises(ValueError):
        campaign.effort(rows, 0)


def test_saved_usage_includes_summary_without_imputation(tmp_path):
    (tmp_path / "memory").mkdir()
    (tmp_path / "proposal.response.json").write_text(
        json.dumps(
            {"usage": {"prompt_tokens": 20, "completion_tokens": 3}, "latency_ms": 12}
        )
    )
    (tmp_path / "memory/summary.response.json").write_text(
        json.dumps({"usage": {"prompt_tokens": 9}, "latency_ms": None})
    )
    result, files = campaign.saved_usage(tmp_path)
    assert len(files) == 2
    assert result["input_tokens_observed_sum"] == 29
    assert result["output_tokens_observed_sum"] == 3
    assert result["output_tokens_missing_or_invalid_records"] == 1
    assert result["latency_ms_missing_or_invalid_records"] == 1


def test_read_only_sqlite_completion_order(tmp_path):
    path = tmp_path / "archive.db"
    with sqlite3.connect(path) as db:
        db.executescript(
            'create table studies(study_id integer,study_name text); insert into studies values(1,"synthetic"); create table trials(trial_id integer,number integer,state text,datetime_start text,datetime_complete text,study_id integer); create table trial_user_attributes(trial_id integer,key text,value_json text); create table trial_values(trial_id integer,objective integer,value real); create table trial_params(trial_id integer,param_name text,param_value real,distribution_json text);'
        )
        db.executemany(
            "insert into trials values(?,?,?,?,?,1)",
            [
                (1, 0, "COMPLETE", "2020-01-01T00:00:00", "2020-01-01T00:00:20"),
                (2, 1, "COMPLETE", "2020-01-01T00:00:00", "2020-01-01T00:00:10"),
            ],
        )
        db.executemany(
            "insert into trial_user_attributes values(?,?,?)",
            [(1, "feasible", "true"), (2, "feasible", "true")],
        )
        db.executemany("insert into trial_values values(?,0,?)", [(1, 0.2), (2, 0.4)])
    before = path.read_bytes()
    rows = campaign.read_trials(path)
    assert [r["number"] for r in rows] == [1, 0]
    assert campaign.effort(rows, 1)["last_trial_number"] == 1
    assert path.read_bytes() == before


def test_usage_mapping_and_invalid_fields(tmp_path):
    (tmp_path / "saved.response.json").write_text(
        json.dumps({"tokens": {"in": 11, "out": -1}, "timing": True})
    )
    result, _ = campaign.saved_usage(
        tmp_path,
        {
            "input_tokens": "tokens.in",
            "output_tokens": "tokens.out",
            "latency_ms": "timing",
        },
    )
    assert result["input_tokens_observed_sum"] == 11
    assert result["output_tokens_observed_sum"] == 0
    assert result["output_tokens_missing_or_invalid_records"] == 1
    assert result["latency_ms_missing_or_invalid_records"] == 1


def test_active_response_root_excludes_migration_backup(tmp_path):
    active = tmp_path / "llm_optimizer"
    backup = tmp_path / "rationale_contract_backup_v1" / "llm_optimizer"
    for root in (active, backup):
        (root / "requests").mkdir(parents=True)
        (root / "requests/proposal.response.json").write_text(
            json.dumps(
                {"usage": {"prompt_tokens": 7, "completion_tokens": 2}, "latency_ms": 4}
            )
        )
    result, files = campaign.saved_usage(active)
    assert result["saved_response_records"] == 1
    assert result["input_tokens_observed_sum"] == 7
    assert len(files) == 1
    assert files[0].is_relative_to(active)


def test_fail_pruned_and_missing_duration_remain_distinct():
    rows = [
        dict(
            number=i,
            state=state,
            feasible=valid,
            duration_seconds=duration,
            proposal_source="unknown",
        )
        for i, (state, valid, duration) in enumerate(
            [
                ("FAIL", False, 11),
                ("PRUNED", False, 13),
                ("COMPLETE", False, None),
                ("COMPLETE", True, 17),
            ]
        )
    ]
    result = campaign.effort(rows, 1)
    assert result["terminal_attempts"] == 4
    assert result["all_terminal_trial_hours"] == 41 / 3600
    assert result["complete_trial_hours"] == 17 / 3600
    assert result["missing_duration_count"] == 1
    assert result["feasible"] == 1
