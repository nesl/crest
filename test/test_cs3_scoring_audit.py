# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Synthetic checks for exact score envelopes and CSV filtering."""

import csv
import importlib.util
import sys
from fractions import Fraction as F
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / "analysis_scripts/cs3_audit/scoring.py"
spec = importlib.util.spec_from_file_location("cs3_scoring", MODULE)
s = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = s
spec.loader.exec_module(s)


def candidate(trial, q, e, method="method"):
    return s.Candidate("board", method, str(trial), F(q), F(e))


def test_exact_crossing_and_tie_endpoint():
    a, b = candidate(10, "1", "400"), candidate(2, "0.9", "0")
    intervals = s.envelope([a, b])
    assert [(r[0], r[1], r[2], r[3]) for r in intervals] == [
        (F(0), F("0.1"), True, True),
        (F("0.1"), F(1), False, True),
    ]
    assert s.select([a, b], F("0.1")) == a
    assert s.select([a, b], F("0.100000000000001")) == b


def test_numeric_trial_tie_and_dominated_parallel_line():
    cs = [candidate(10, "0.9", "1"), candidate(2, "0.9", "1"), candidate(1, "0.8", "1")]
    assert len(s.envelope(cs)) == 1
    assert s.envelope(cs)[0][4].trial_id == "2"


def test_single_point_winner_retained():
    a, b = candidate(1, "1", "400"), candidate(2, "0.9", "0")
    intervals = s.envelope([a, b], F("0.1"), F(1))
    assert intervals[0][:4] == (F("0.1"), F("0.1"), True, True)


def test_pool_merging_and_written_exact_endpoints(tmp_path):
    pools = {
        ("board", "a"): [candidate(1, "1", "400", "a"), candidate(2, "0.9", "0", "a")],
        ("board", "b"): [candidate(1, "0.8", "0", "b")],
    }
    result = s.audit(pools, tmp_path)
    assert len(result["board_pool_intervals"]) == 1
    assert result["board_pool_intervals"][0]["hi_exact"] == "1"
    assert (tmp_path / "selection_intervals.csv").exists()


def test_cli_filters_and_endpoint(tmp_path):
    path = tmp_path / "input.csv"
    path.write_text(
        "b,m,id,q,e,status,ok,score,lat\nx,y,0,1,1,FAIL,true,1,1\nx,y,1,0.9,1,COMPLETE,true,nan,1\nx,y,2,0.8,1,COMPLETE,true,1,3000\nx,y,3,0.7,1,COMPLETE,true,1,1\nx,y,4,1,0,COMPLETE,true,1,1\n"
    )
    out = tmp_path / "out"
    assert (
        s.main(
            [
                "--input",
                f"a,b={path}",
                "--board-col",
                "b",
                "--method-col",
                "m",
                "--quality-col",
                "q",
                "--energy-col",
                "e",
                "--trial-id-col",
                "id",
                "--status-col",
                "status",
                "--feasible-col",
                "ok",
                "--source-score-col",
                "score",
                "--latency-col",
                "lat",
                "--feasible-limit",
                "1",
                "--output-dir",
                str(out),
            ]
        )
        == 0
    )
    rows = list(csv.DictReader((out / "baseline_stability.csv").open()))
    assert rows[0]["trial_id"] == "3"
    assert rows[0]["board"] == "x"
    assert rows[0]["candidate_count"] == "1"


def test_canonical_observations_cli(tmp_path):
    """Read endpoint-filtered canonical CSVs without raw campaign columns."""
    path = tmp_path / "observations.csv"
    path.write_text(
        "board,method,trial,f1,energy_mj,score,latency_ms\nboard,optuna,2,0.9,40,0.89,50\n"
    )
    out = tmp_path / "out"
    assert s.main(["--observations", str(path), "--output-dir", str(out)]) == 0
    rows = list(csv.DictReader((out / "baseline_stability.csv").open()))
    assert rows[0]["method"] == "optuna"
    assert rows[0]["lo_exact"] == "0"
    assert rows[0]["hi_exact"] == "1"
