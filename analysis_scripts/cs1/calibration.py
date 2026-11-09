# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Rebuild the submitted CS1 waiting calibration and frozen candidate predictions."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def probe_windows(raw):
    """Recover paired window energies and continuous DWT duration.

    Parameters
    ----------
    raw : pandas.DataFrame
        Accepted probe records for one period, with one row per round/clock/shape.

    Returns
    -------
    pandas.DataFrame
        Continuous energy (mJ), active time (s), and one/ten-release energy (mJ).

    Raises
    ------
    ValueError
        If records are incomplete, duplicated, or fail recorded acceptance checks.
    """
    keys = ["round", "clock_mhz", "shape"]
    if raw.empty or raw.duplicated(keys).any() or not raw.status.eq("accepted").all():
        raise ValueError("Expected unique accepted probe measurements.")
    expected = {
        "continuous": ("back_to_back", 10, -1),
        "w1": ("cadenced", 1, 0),
        "w10": ("cadenced", 10, 9),
    }
    for shape, (phase, runs, stops) in expected.items():
        group = raw[raw["shape"].eq(shape)]
        if (
            group.empty
            or not group.phase.eq(phase).all()
            or not group.runs.eq(runs).all()
        ):
            raise ValueError(f"Invalid probe phase/run count for {shape}.")
        if (
            shape != "continuous"
            and not (
                group.stop_entries.eq(stops) & group.deadline_miss_count.eq(0)
            ).all()
        ):
            raise ValueError(f"Invalid STOP/deadline evidence for {shape}.")
    continuous = raw[raw["shape"].eq("continuous")].set_index(["round", "clock_mhz"])
    result = pd.DataFrame(
        {
            "E_inf_mj": continuous.energy_mj_per_inference,
            "C_s": continuous.dwt_cycles_per_inference / continuous.clock_hz,
        }
    )
    for shape, runs in [("w1", 1), ("w10", 10)]:
        window = raw[raw["shape"].eq(shape)].set_index(["round", "clock_mhz"])
        if set(window.index) != set(result.index):
            raise ValueError("Probe rounds/clock settings are not matched.")
        result[f"W{runs}_mj"] = runs * window.energy_mj_per_inference
    if not np.isfinite(result.to_numpy()).all() or not result.C_s.gt(0).all():
        raise ValueError("Invalid probe energy or duration.")
    return result.reset_index()


def joint_fit(long_period, short_period, periods=(0.5, 0.1)):
    """Fit busy/STOP slopes and intercepts by matched round and clock.

    Parameters
    ----------
    long_period : pandas.DataFrame
        Window summaries for the longer period.
    short_period : pandas.DataFrame
        Matching window summaries for the shorter period.
    periods : tuple of float, optional
        Corresponding periods in seconds.

    Returns
    -------
    pandas.DataFrame
        Per-round fitted powers in mW and corrections in mJ.
    """
    merged = long_period.merge(
        short_period,
        on=["round", "clock_mhz"],
        suffixes=("_5", "_1"),
        validate="one_to_one",
    )
    if len(merged) != len(long_period) or len(merged) != len(short_period):
        raise ValueError("The two probe sessions must have matching round/clock keys.")
    a5, a1 = periods[0] - merged.C_s_5, periods[1] - merged.C_s_1
    if periods[0] <= periods[1] or not ((a5 > 0) & (a1 > 0) & (a5 != a1)).all():
        raise ValueError(
            "Require distinct increasing available-wait times and feasible periods."
        )
    y5 = merged.W1_mj_5 - merged.E_inf_mj_5
    y1 = merged.W1_mj_1 - merged.E_inf_mj_1
    busy = (y5 - y1) / (a5 - a1)
    d5 = (merged.W10_mj_5 - 10 * merged.E_inf_mj_5 - y5) / 9
    d1 = (merged.W10_mj_1 - 10 * merged.E_inf_mj_1 - y1) / 9
    stop = (d5 - d1) / (a5 - a1)
    return pd.DataFrame(
        {
            "round": merged["round"],
            "clock_mhz": merged.clock_mhz,
            "P_busy_mw": busy,
            "E0_mj": y1 - a1 * busy,
            "P_stop_mw": stop,
            "K_mj": d1 - a1 * stop,
            f"E_inf_{periods[0]*1000:g}_mj": merged.E_inf_mj_5,
            f"E_inf_{periods[1]*1000:g}_mj": merged.E_inf_mj_1,
            f"C_{periods[0]*1000:g}_ms": merged.C_s_5 * 1000,
            f"C_{periods[1]*1000:g}_ms": merged.C_s_1 * 1000,
        }
    )


def calibrate(raw_long, raw_short, periods=(0.5, 0.1), fit_decimals=None):
    """Fit waiting calibration from two-period, one/ten-release probes.

    Parameters
    ----------
    raw_long : pandas.DataFrame
        Raw longer-period probe table.
    raw_short : pandas.DataFrame
        Raw shorter-period probe table.
    periods : tuple of float, optional
        Probe periods in seconds.
    fit_decimals : int, optional
        Round fitted summaries before reuse; use 3 for submitted-paper parity.

    Returns
    -------
    rounds : pandas.DataFrame
        Per-round joint fits.
    calibration : pandas.DataFrame
        Clock-indexed coefficients and active-power statistics.

    Notes
    -----
    The submitted analysis read a joint-fit CSV rounded to three decimals before
    combining coefficients. Retaining that step reproduces its frozen predictors.
    """
    rounds = joint_fit(probe_windows(raw_long), probe_windows(raw_short), periods)
    calibration = rounds.groupby("clock_mhz").agg(["mean", "std"]).drop(columns="round")
    if fit_decimals is not None:
        calibration = calibration.round(fit_decimals)
    calibration.columns = [f"{column}_{stat}" for column, stat in calibration.columns]
    raw = pd.concat([raw_long, raw_short], ignore_index=True)
    active = raw[raw.phase.eq("back_to_back")].copy()
    active["active_mw"] = active.energy_mj_per_inference / active.harness_latency_s
    stats = active.groupby("clock_mhz").active_mw.agg(["mean", "std", "count"])
    calibration["active_mw"] = stats["mean"]
    calibration["wait_mw"] = (
        0.9 * calibration.P_stop_mw_mean + 0.1 * calibration.P_busy_mw_mean
    )
    calibration["offset_mj"] = 0.9 * calibration.K_mj_mean + calibration.E0_mj_mean / 10
    return rounds, calibration.join(stats.add_prefix("active_"))


def candidate_predictions(candidates, latencies, calibration, exclude_pools=()):
    """Apply probe coefficients to a candidate bank.

    Parameters
    ----------
    candidates : pandas.DataFrame
        Saved joined candidate bank, including measured energy and quality.
    latencies : pandas.DataFrame
        Continuous harness durations keyed by pool and trial.
    calibration : pandas.DataFrame
        Clock-indexed active/waiting powers and correction.
    exclude_pools : sequence of str, optional
        Pools with incompatible measurement protocols to exclude explicitly.

    Returns
    -------
    pandas.DataFrame
        Candidate bank with frozen continuous and scheduled predictions.
    """
    data = candidates[~candidates.pool.isin(exclude_pools)].copy()
    data = data.merge(
        latencies[["pool", "trial", "continuous_latency_ms"]],
        on=["pool", "trial"],
        how="left",
        validate="one_to_one",
    )
    data["latency_s"] = data.continuous_latency_ms / 1000
    if data.empty or not np.allclose(data.latency_s, data.C_s, atol=1e-12, rtol=0):
        raise ValueError("Expected a nonempty bank with matching harness durations.")
    for column in ["active_mw", "wait_mw", "offset_mj"]:
        data[column] = data.clock_mhz.map(calibration[column])
    if data[["active_mw", "wait_mw", "offset_mj"]].isna().any().any():
        raise ValueError("A candidate clock is missing calibration.")
    data["continuous_hat_mj"] = data.active_mw * data.latency_s
    data["latency_model_mj"] = (
        data.T_s * data.wait_mw
        + data.latency_s * (data.active_mw - data.wait_mw)
        + data.offset_mj
    )
    rebuilt = data.continuous_mj + (data.T_s - data.C_s) * data.wait_mw + data.offset_mj
    if "E_pred" in data and not np.allclose(rebuilt, data.E_pred):
        raise ValueError(
            "Rebuilt measured-continuous predictions differ from the frozen bank."
        )
    data["E_pred"] = rebuilt
    return data


def main():
    """Parse explicit input paths and write calibration/prediction CSVs."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["probe-long", "probe-short", "candidates", "latencies", "output-dir"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--periods-ms", nargs=2, type=float, default=[500, 100])
    parser.add_argument("--fit-decimals", type=int)
    parser.add_argument("--exclude-pool", action="append", default=[])
    args = parser.parse_args()
    paths = {
        key: value
        for key, value in vars(args).items()
        if isinstance(value, Path) and key != "output_dir"
    }
    tables = {key: pd.read_csv(value) for key, value in paths.items()}
    rounds, calibration = calibrate(
        tables["probe_long"],
        tables["probe_short"],
        tuple(t / 1000 for t in args.periods_ms),
        args.fit_decimals,
    )
    predictions = candidate_predictions(
        tables["candidates"], tables["latencies"], calibration, args.exclude_pool
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rounds.to_csv(args.output_dir / "probe_joint_fit_per_round.csv", index=False)
    calibration.to_csv(args.output_dir / "calibration.csv")
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    provenance = {
        key: {
            "name": value.name,
            "sha256": hashlib.sha256(value.read_bytes()).hexdigest(),
        }
        for key, value in paths.items()
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(calibration[["active_mw", "wait_mw", "offset_mj"]].round(4).to_string())


if __name__ == "__main__":
    main()
