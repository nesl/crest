#!/usr/bin/env python3
# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Reproduce direct-session paired energy contrasts and effective waiting slopes."""
import argparse
from itertools import permutations
import hashlib
import json
from pathlib import Path
import pandas as pd
import numpy as np

try:
    from .scaling_periods import ci
except ImportError:
    from scaling_periods import ci


def run(
    archive_root=None,
    output_dir=None,
    *,
    measurements=None,
    periods=(100, 500),
    probe_100=None,
    probe_500=None,
):
    """Export paired contrasts, fitted slopes, and source hashes.

    Parameters
    ----------
    archive_root : pathlib.Path
        Research archive containing the final_slot_v1 session and probes.
    output_dir : pathlib.Path
        Destination for derived CSV and report files.

    measurements : pathlib.Path, optional
        Explicit raw records; overrides archive mapping.
    periods : tuple of float
        Two periods in milliseconds for effective slope fitting.
    probe_100, probe_500 : pathlib.Path, optional
        Earlier waiting_power_per_round tables for descriptive comparison.

    Returns
    -------
    None
        Writes outputs without changing the source archive.
    """
    if len(periods) != 2 or periods[0] >= periods[1]:
        raise ValueError("Provide two increasing periods")
    t0, t1 = periods
    root = Path(archive_root).resolve() if archive_root else None
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    base = root / "analysis_results/cs2_sleep_ratio_model_20261027" if root else None
    if measurements is None:
        if base is None:
            raise ValueError("Provide --measurements or --archive-root")
        measurements = base / "final_slot_v1/raw_measurements.csv"
    inputs = [Path(measurements).resolve()]
    raw = pd.read_csv(inputs[0])
    if not raw.status.eq("accepted").all():
        raise ValueError("Only accepted records may be analyzed")
    keys = ["round", "candidate", "clock_mhz", "shape", "period_ms", "policy"]
    if raw.duplicated(keys).any():
        raise ValueError("Duplicate records or incomplete rounds")
    raw = raw[raw["shape"].eq("continuous") | raw.period_ms.isin(periods)].copy()
    windows = raw[raw["shape"].eq("w10")].copy()
    policies = sorted(windows.policy.unique())
    if not (
        windows.deadline_miss_count.eq(0).all()
        and np.array_equal(
            windows.stop_entries, np.where(windows.policy.eq("stop"), 10, 9)
        )
        and np.allclose(windows.timer_per_window_s, windows.period_ms / 100)
    ):
        raise ValueError("Invalid deadline, STOP-entry, or timer evidence")
    r = (
        windows.pivot(
            index=["round", "candidate", "clock_mhz", "period_ms"],
            columns="policy",
            values="energy_mj_per_inference",
        )
        .rename(columns={"awake": "E_awake_mj", "stop": "E_stop_mj"})
        .reset_index()
    )
    r["cell"] = r.candidate + "@" + r.clock_mhz.astype(str)
    pairs = list(permutations(sorted(r.cell.unique()), 2))
    contrasts = []
    for a, b in pairs:
        for period in periods:
            for policy in policies:
                x = (
                    r[(r.cell == a) & (r.period_ms == period)]
                    .set_index("round")
                    .sort_index()[f"E_{policy}_mj"]
                )
                y = (
                    r[(r.cell == b) & (r.period_ms == period)]
                    .set_index("round")
                    .sort_index()[f"E_{policy}_mj"]
                )
                if not x.index.equals(y.index) or len(x) < 2:
                    raise ValueError("Missing or unmatched round blocks")
                contrasts.append(
                    dict(a=a, b=b, period_ms=period, policy=policy, **ci(x - y))
                )
    pd.DataFrame(contrasts).to_csv(out / "cross_candidate_contrasts.csv", index=False)
    previous = []
    for period, path in ((100, probe_100), (500, probe_500)):
        if path is None:
            continue
        path = Path(path).resolve()
        inputs.append(path)
        p = pd.read_csv(path)
        for cell, current in r[r.period_ms.eq(period)].groupby("cell"):
            clock = current.clock_mhz.iloc[0]
            earlier = (p[p.clock_mhz.eq(clock)].W10_mj / 10).mean()
            previous.append(
                dict(
                    cell=cell,
                    period_ms=period,
                    clock_mhz=clock,
                    earlier_probe_mj=earlier,
                    this_run_awake_mj=current.E_awake_mj.mean(),
                    shift_mj=current.E_awake_mj.mean() - earlier,
                )
            )
    pd.DataFrame(previous).to_csv(out / "cross_session_awake.csv", index=False)
    fitted = []
    for (candidate, clock), config in raw.groupby(["candidate", "clock_mhz"]):
        for policy in policies:
            for rnd, group in config.groupby("round"):
                continuous = group[group["shape"].eq("continuous")]
                p = group[group["shape"].eq("w10") & group.policy.eq(policy)].set_index(
                    "period_ms"
                )
                if len(continuous) != 1 or set(p.index) != set(periods):
                    raise ValueError("Incomplete direct-session factorial")
                ec = continuous.energy_mj_per_inference.iloc[0]
                x = p.index.to_numpy() / 1000 - p.C_dwt_s
                y = p.energy_mj_per_inference - ec
                slope = (y.loc[t1] - y.loc[t0]) / (x.loc[t1] - x.loc[t0])
                fitted.append(
                    dict(
                        candidate=candidate,
                        clock_mhz=clock,
                        policy=policy,
                        round=rnd,
                        P_mw=slope,
                        E_tr_mj=y.loc[t0] - x.loc[t0] * slope,
                        C_ms=1000 * p.C_dwt_s.mean(),
                        E_cont_mj=ec,
                    )
                )
    fit = pd.DataFrame(fitted)
    fit.to_csv(out / "waiting_slopes_per_round.csv", index=False)
    summary = (
        fit.groupby(["candidate", "clock_mhz", "policy"])
        .agg(
            P_mw_mean=("P_mw", "mean"),
            P_mw_sd=("P_mw", "std"),
            E_tr_mj_mean=("E_tr_mj", "mean"),
            E_tr_mj_sd=("E_tr_mj", "std"),
        )
        .reset_index()
    )
    summary.to_csv(out / "waiting_slopes.csv", index=False)
    fit["cell"] = fit.candidate + "@" + fit.clock_mhz.astype(str)
    windows["cell"] = windows.candidate + "@" + windows.clock_mhz.astype(str)
    decomposition, slope_contrasts = [], []
    for policy in policies:
        pivot = fit[fit.policy.eq(policy)].pivot(
            index="round", columns="cell", values="P_mw"
        )
        for a, b in pairs:
            slope_contrasts.append(
                dict(a=a, b=b, policy=policy, **ci(pivot[a] - pivot[b]))
            )
            for rnd in pivot.index:
                cfg = fit[fit["round"].eq(rnd) & fit.policy.eq(policy)].set_index(
                    "cell"
                )
                hi, lo = cfg.loc[a], cfg.loc[b]
                for period in periods:
                    w = windows[
                        windows["round"].eq(rnd)
                        & windows.policy.eq(policy)
                        & windows.period_ms.eq(period)
                    ].set_index("cell")
                    active = hi.E_cont_mj - lo.E_cont_mj
                    shared = (w.loc[b, "C_dwt_s"] - w.loc[a, "C_dwt_s"]) * lo.P_mw
                    slope = (period / 1000 - w.loc[a, "C_dwt_s"]) * (hi.P_mw - lo.P_mw)
                    correction = hi.E_tr_mj - lo.E_tr_mj
                    total = (
                        w.loc[a, "energy_mj_per_inference"]
                        - w.loc[b, "energy_mj_per_inference"]
                    )
                    if not np.isclose(active + shared + slope + correction, total):
                        raise ValueError("Accounting decomposition does not close")
                    decomposition.append(
                        dict(
                            a=a,
                            b=b,
                            policy=policy,
                            round=rnd,
                            period_ms=period,
                            continuous_mj=active,
                            shared_wait_mj=shared,
                            slope_difference_mj=slope,
                            correction_difference_mj=correction,
                            measured_total_mj=total,
                        )
                    )
    dec = pd.DataFrame(decomposition)
    dec.to_csv(out / "accounting_decomposition_per_round.csv", index=False)
    dec.groupby(["a", "b", "policy", "period_ms"]).mean().drop(columns="round").to_csv(
        out / "accounting_decomposition.csv"
    )
    pd.DataFrame(slope_contrasts).to_csv(
        out / "waiting_slope_contrast.csv", index=False
    )
    (out / "input_hashes.json").write_text(
        json.dumps(
            [
                dict(
                    path=str(p),
                    sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                )
                for p in inputs
            ],
            indent=2,
        )
        + "\n"
    )
    (out / "REPORT.md").write_text(
        """# Direct-session energy accounting

Contrasts are first configuration minus second, paired within observed round blocks. Negative values indicate savings by the first named cell. Waiting slopes fit
E_release - E_cont = (T - C) P + correction separately per configuration, policy,
and round using measured C at each period. Summaries report mean and sample SD;
contrasts report nominal unadjusted 95% Student t repeatability intervals.

These are effective two-period slopes, not independent rail-power measurements.
Fits do not isolate a causal phase or establish architecture preservation. Earlier
probe comparisons cross campaigns/firmware and are descriptive, never paired.
"""
    )


def main():
    """Parse explicit input archive and output directory paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path)
    parser.add_argument("--measurements", type=Path)
    parser.add_argument("--probe-100", type=Path)
    parser.add_argument("--probe-500", type=Path)
    parser.add_argument("--periods", type=float, nargs=2, default=(100, 500))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.archive_root,
        args.output_dir,
        measurements=args.measurements,
        periods=args.periods,
        probe_100=args.probe_100,
        probe_500=args.probe_500,
    )


if __name__ == "__main__":
    main()
