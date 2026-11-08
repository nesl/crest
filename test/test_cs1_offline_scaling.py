# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Numerical checks for portable CS1 scaling and direct-session analyses."""

import os
from pathlib import Path
import sys
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "analysis_scripts/cs1"
sys.path.insert(0, str(SCRIPTS))
import scaling_periods
import direct_session


def test_interval_known_values():
    """Check the interval against a known five-observation t calculation.

    Returns
    -------
    None
        Assertions validate the interval and invalid sample handling.
    """
    result = scaling_periods.ci([1, 2, 3, 4, 5])
    assert result["mean"] == 3
    assert result["ci95_lo"] == pytest.approx(1.0367568385)
    with pytest.raises(ValueError):
        scaling_periods.ci([1])


def test_saved_archive(tmp_path):
    """Recompute the submitted manuscript's selected numerical claims.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary output directory supplied by pytest.

    Returns
    -------
    None
        Verifies scaling, optima, savings, reversal, and waiting slopes.
    """
    archive = os.environ.get("CREST_CS1_ARCHIVE_ROOT")
    if not archive:
        pytest.skip("Set CREST_CS1_ARCHIVE_ROOT to the retained research archive")
    scaling_periods.run(Path(archive), tmp_path / "scaling")
    direct_session.run(Path(archive), tmp_path / "direct")
    extents = pd.read_csv(tmp_path / "scaling/scaling_extents.csv").query(
        "shape == 'continuous'"
    )
    assert extents.max_minus_min_cycles_pct.tolist() == pytest.approx(
        [0.8593481989, 0.9431148601]
    )
    opt = pd.read_csv(tmp_path / "scaling/single_architecture_optima.csv")
    assert opt.minimum_latency_frequency_mhz.eq(800).all()
    assert (
        opt.query("shape == 'w10'").minimum_measured_energy_frequency_mhz.eq(300).all()
    )
    gaps = pd.read_csv(tmp_path / "scaling/period_gap_contrasts.csv")
    for policy in ("stop", "awake"):
        g = gaps[gaps.policy.eq(policy)].set_index("period_ms")
        assert g.loc["100", "mean"] > 0
        assert g.loc["500", "mean"] < 0
    contrasts = pd.read_csv(tmp_path / "direct/cross_candidate_contrasts.csv")
    savings = contrasts.query(
        "a == 'continuous_203@300' and b == 'continuous_203@800' and policy == 'stop'"
    ).sort_values("period_ms")
    assert (-savings["mean"]).tolist() == pytest.approx([0.258, 1.582], abs=0.0005)
    reversal = contrasts.query(
        "a == 'continuous_137@300' and b == 'continuous_203@800' and policy == 'stop'"
    ).sort_values("period_ms")
    assert reversal["mean"].tolist() == pytest.approx([1.87, -3.25], abs=0.005)
    slopes = (
        pd.read_csv(tmp_path / "direct/waiting_slopes.csv")
        .query("policy == 'stop'")
        .set_index(["candidate", "clock_mhz"])
    )
    assert slopes.loc[("continuous_203", 800), "P_mw_mean"] == pytest.approx(
        803.78, abs=0.005
    )
    assert slopes.loc[("continuous_203", 300), "P_mw_mean"] == pytest.approx(
        800.48, abs=0.005
    )
    assert slopes.loc[("continuous_137", 300), "P_mw_mean"] == pytest.approx(
        791.04, abs=0.005
    )
    contrast = (
        pd.read_csv(tmp_path / "direct/waiting_slope_contrast.csv")
        .query(
            "a == 'continuous_203@300' and b == 'continuous_137@300' and policy == 'stop'"
        )
        .iloc[0]
    )
    assert contrast[["mean", "ci95_lo", "ci95_hi"]].tolist() == pytest.approx(
        [9.43, 7.47, 11.39], abs=0.005
    )


def test_generic_two_round_session(tmp_path):
    """Verify unfamiliar cells and periods work without the retained archive.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory supplied by pytest.

    Returns
    -------
    None
        Checks generic effective slopes, pairing, and frequency baselines.
    """
    records = []
    for rnd in (8, 12):
        for candidate, clock, slope in [("alpha", 50, 2), ("beta", 75, 3)]:
            common = dict(
                round=rnd,
                candidate=candidate,
                clock_mhz=clock,
                status="accepted",
                C_dwt_s=0.01,
                dwt_cycles_per_inference=0.01 * clock * 1e6,
            )
            records.append(
                dict(
                    common,
                    shape="continuous",
                    period_ms=200,
                    policy="awake",
                    energy_mj_per_inference=1,
                )
            )
            for period in (200, 800):
                records.append(
                    dict(
                        common,
                        shape="w10",
                        period_ms=period,
                        policy="stop",
                        energy_mj_per_inference=1 + (period / 1000 - 0.01) * slope,
                        deadline_miss_count=0,
                        stop_entries=10,
                        timer_per_window_s=period / 100,
                    )
                )
    path = tmp_path / "raw.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    direct_session.run(
        output_dir=tmp_path / "generic", measurements=path, periods=(200, 800)
    )
    slopes = pd.read_csv(tmp_path / "generic/waiting_slopes.csv")
    assert slopes.P_mw_mean.tolist() == pytest.approx([2, 3])
    contrast = pd.read_csv(tmp_path / "generic/cross_candidate_contrasts.csv")
    assert set(contrast.n) == {2}
    probe = pd.DataFrame(
        [
            dict(
                round=rnd,
                clock_mhz=clock,
                shape="continuous",
                status="accepted",
                C_dwt_s=1 / clock,
                dwt_cycles_per_inference=1e6,
                energy_mj_per_inference=clock,
            )
            for rnd in (8, 12)
            for clock in (50, 75)
        ]
    )
    probe_path = tmp_path / "probe.csv"
    probe.to_csv(probe_path, index=False)
    scaling_periods.analyze_probes([(200, probe_path)], tmp_path / "probe_out")
    ext = pd.read_csv(tmp_path / "probe_out/scaling_extents.csv")
    assert ext.max_minus_min_cycles_pct.iloc[0] == pytest.approx(0)


def test_reproduction_uses_external_pairing_files(tmp_path, monkeypatch):
    """Pass caller-owned pairings through the CLI to the two analysis commands."""
    import reproduce_paper
    from subprocess import CompletedProcess

    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(reproduce_paper.subprocess, "run", run)
    policy = tmp_path / "external-policy.json"
    repeat = tmp_path / "external-repeat.json"
    policy.write_text('[{"left": "policy_a", "right": "policy_b"}]')
    repeat.write_text('[{"left": "repeat_a", "right": "repeat_b"}]')
    monkeypatch.setattr(sys, "argv", [
        "reproduce_paper.py", "--archive-root", str(tmp_path / "archive"),
        "--output-dir", str(tmp_path / "results"),
        "--policy-pairs", str(policy), "--repeat-pairs", str(repeat),
    ])
    reproduce_paper.main()
    commands = {Path(command[1]).stem: command for command in calls}
    for name, path in (("policy_replay", policy), ("repeats", repeat)):
        command = commands[name]
        assert command[command.index("--pairs") + 1] == str(path.resolve())
    assert len(calls) == 9
