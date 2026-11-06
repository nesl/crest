# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Check regenerated CS1 results against numbers printed in the submitted paper."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def verify(results):
    """Check paper-specific counts and displayed numerical precision.

    Parameters
    ----------
    results : pathlib.Path
        Output directory produced by ``reproduce_paper.py``.

    Returns
    -------
    pandas.DataFrame
        Named checks with actual values, expected values, tolerances and status.
    """
    checks = []

    def check(name, actual, expected, digits=2):
        """Record one paper value at its displayed precision.

        Parameters
        ----------
        name : str
            Check identifier.
        actual, expected : float
            Computed and printed values.
        digits : int or None, optional
            Decimal places, or None for an exact count.

        Returns
        -------
        None
            Appends the result to the enclosing check list.
        """
        tolerance = 0 if digits is None else 0.5 * 10 ** (-digits) + 1e-10
        checks.append(
            dict(
                check=name,
                actual=float(actual),
                expected=expected,
                tolerance=tolerance,
                passed=bool(
                    np.isfinite(actual) and abs(actual - expected) <= tolerance
                ),
            )
        )

    same, transfer = (
        "same_pool_architecture_holdout",
        "leave_pool_out_architecture_disjoint",
    )
    manifest = pd.read_csv(results / "prediction/architecture_manifest.csv")
    check("observations", len(manifest), 756, None)
    check("graph identities", manifest.arch_id.nunique(), 470, None)
    metrics = pd.read_csv(results / "prediction/metrics.csv").set_index(
        ["protocol", "model", "target"]
    )
    for protocol, model, target, expected in [
        (same, "affine", "continuous", 1.38),
        (same, "affine", "scheduled", 3.92),
        (same, "measured_continuous", "scheduled", 3.48),
        (same, "scheduled_direct_affine", "scheduled", 3.11),
        (transfer, "affine", "continuous", 1.80),
        (transfer, "affine", "scheduled", 4.30),
    ]:
        row = metrics.loc[(protocol, model, target)]
        check(f"{protocol}/{model}/{target} RMSE", row.rmse_mj, expected)
    check(
        "heldout predictions",
        metrics.loc[(same, "affine", "continuous")].n_predictions,
        7502,
        None,
    )
    check(
        "heldout covered rows",
        metrics.loc[(same, "affine", "continuous")].n_rows,
        753,
        None,
    )
    quant = pd.read_csv(results / "prediction/metrics_by_quantization.csv")
    quant = quant[
        (quant.protocol == same)
        & (quant.target == "scheduled")
        & (quant.quantization == "int8_ptq")
    ].set_index("model")
    for model, expected in [
        ("proportional", 4.10),
        ("affine", 4.04),
        ("measured_continuous", 3.59),
        ("scheduled_direct_affine", 3.18),
    ]:
        check(f"int8/{model} RMSE", quant.loc[model].rmse_mj, expected)
    selection = pd.read_csv(results / "prediction/selection_summary.csv").set_index(
        ["protocol", "model"]
    )
    for model, expected in [
        ("affine", 1.72),
        ("proportional", 1.71),
        ("measured_continuous", 1.64),
        ("scheduled_direct_affine", 1.73),
    ]:
        check(f"{model} regret", selection.loc[(same, model), "mean"], expected)
        check(f"{model} selections", selection.loc[(same, model), "count"], 622, None)
    raw = pd.read_csv(results / "selection/selection_summary.csv").set_index(
        ["pr", "m"]
    )
    for model, expected in [("raw_continuous", 2.64), ("raw_latency", 2.77)]:
        check(f"{model} regret", raw.loc[(same, model), "mean"], expected)
    for model, expected in [
        ("affine", 2.93),
        ("measured_continuous", 2.61),
        ("raw_continuous", 4.13),
    ]:
        check(f"transfer {model} regret", raw.loc[(transfer, model), "mean"], expected)
        check(
            f"transfer {model} selections",
            raw.loc[(transfer, model), "count"],
            15,
            None,
        )
    calibration = pd.read_csv(results / "calibration/calibration.csv").set_index(
        "clock_mhz"
    )
    for clock, active, wait, offset in [
        (200, 920.70, 791.89, 0.3708),
        (300, 957.53, 792.10, 0.4593),
        (400, 997.03, 801.74, 0.4428),
        (600, 1074.17, 802.97, 0.5655),
        (800, 1190.60, 809.51, 0.7818),
    ]:
        for column, expected, digits in [
            ("active_mw", active, 2),
            ("wait_mw", wait, 2),
            ("offset_mj", offset, 4),
        ]:
            check(f"{clock}/{column}", calibration.loc[clock, column], expected, digits)
    paired = pd.read_csv(results / "transfer/paired_summary.csv").set_index(
        ["pair", "metric"]
    )
    for pair, observed, residual, timing in [
        ("replication_400MHz", -6.778, -6.60, 0.13),
        ("random3_400MHz", -3.623, -3.85, -0.19),
    ]:
        for metric, expected, digits in [
            ("observed", observed, 3),
            ("meas_frozen_residual", residual, 2),
            ("lat_scheduled_plugin", timing, 2),
        ]:
            check(
                f"{pair}/{metric}", paired.loc[(pair, metric), "mean"], expected, digits
            )
    policy = json.loads((results / "policy/paired_analysis.json").read_text())
    for pair, values in [
        ("110_minus_cadenced_123", [-6.46, -0.21, 6.25]),
        ("228_minus_cadenced_243", [-3.73, -5.81, -2.09]),
    ]:
        for suffix, expected in zip(["awake", "stop", "policy_interaction"], values):
            key = f"cadenced_{pair}_{suffix}"
            check(key, policy[key]["mean_mj"], expected)
    repeat = pd.read_csv(results / "repeats/paired_summary.csv").set_index(
        ["pair", "metric"]
    )
    for pair, values in [("203--114", [2.49, -0.29]), ("228--131", [0.28, 0.25])]:
        for metric, expected in zip(["scheduled_mj", "estimated_stop_mj"], values):
            check(
                f"repeat {pair}/{metric}",
                repeat.loc[(pair, metric), "mean_mj"],
                expected,
            )
    direct = pd.read_csv(
        results / "direct_session/cross_candidate_contrasts.csv"
    ).set_index(["a", "b", "period_ms", "policy"])
    for a, b, period, expected in [
        ("continuous_203@300", "continuous_203@800", 100, -0.26),
        ("continuous_203@300", "continuous_203@800", 500, -1.58),
        ("continuous_137@300", "continuous_203@800", 100, 1.87),
        ("continuous_137@300", "continuous_203@800", 500, -3.25),
    ]:
        check(
            f"{a}/{b}/{period} STOP difference",
            direct.loc[(a, b, period, "stop"), "mean"],
            expected,
        )
    slopes = pd.read_csv(results / "direct_session/waiting_slopes.csv").set_index(
        ["candidate", "clock_mhz", "policy"]
    )
    for candidate, clock, expected in [
        ("continuous_203", 800, 803.78),
        ("continuous_203", 300, 800.48),
        ("continuous_137", 300, 791.04),
    ]:
        check(
            f"{candidate}/{clock} waiting slope",
            slopes.loc[(candidate, clock, "stop"), "P_mw_mean"],
            expected,
        )
    return pd.DataFrame(checks)


def main():
    """Write all check results and fail if any submitted value differs.

    Returns
    -------
    None
        Writes ``paper_checks.csv`` and prints a one-line summary.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    checks = verify(args.results)
    checks.to_csv(args.results / "paper_checks.csv", index=False)
    failures = checks[~checks.passed]
    if len(failures):
        raise SystemExit(failures.to_string(index=False))
    print(f"Paper checks: {len(checks)}/{len(checks)} passed.")


if __name__ == "__main__":
    main()
