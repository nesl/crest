# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Reproduce paired search-pool repeats and conditional STOP estimates."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t


def summarize(values):
    """Compute paired-round uncertainty.

    Parameters
    ----------
    values : array_like
        Per-round energy differences, in mJ.

    Returns
    -------
    dict
        Count, mean, and Student t 95 percent interval bounds.
    """
    values = np.asarray(values, float)
    half = t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return dict(
        n=len(values),
        mean_mj=values.mean(),
        ci95_lo=values.mean() - half,
        ci95_hi=values.mean() + half,
    )


def main(argv=None):
    """Analyze explicit repeat observations and frozen joint coefficients.

    Parameters
    ----------
    argv : list of str, optional
        CLI arguments, defaulting to process arguments.

    Returns
    -------
    None
        Writes paired contrasts, summaries, report, and input provenance.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeated-observations", required=True, type=Path)
    parser.add_argument(
        "--pairs",
        required=True,
        type=Path,
        help="JSON list of {left, right} configuration IDs",
    )
    parser.add_argument("--joint-fit", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    records = pd.read_csv(args.repeated_observations)
    fit = pd.read_csv(args.joint_fit).set_index("clock_mhz")
    if records.duplicated(["round", "config"]).any():
        raise ValueError("Duplicate round/configuration observations")
    clock = records.clock_hz / 1e6
    pb = clock.map(fit.P_busy_mw_mean)
    ps = clock.map(fit.P_stop_mw_mean)
    k = clock.map(fit.K_mj_mean)
    if pb.isna().any() or ps.isna().any() or k.isna().any():
        raise ValueError("Missing frozen coefficients")
    records["estimated_stop_mj"] = records.scheduled_mj - 0.1 * (
        (records.T_s - records.C_s) * (pb - ps) - k
    )
    rounds, summaries = [], []
    for pair_spec in json.loads(args.pairs.read_text()):
        left, right = pair_spec["left"], pair_spec["right"]
        for metric in ["continuous_mj", "scheduled_mj", "estimated_stop_mj"]:
            wide = records[records.config.isin([left, right])].pivot(
                index="round", columns="config", values=metric
            )
            if len(wide) < 2 or wide.isna().any().any():
                raise ValueError("Expected at least two complete paired rounds")
            differences = wide[left] - wide[right]
            pair = (
                left.removeprefix("continuous_")
                + "--"
                + right.removeprefix("continuous_")
            )
            summaries.append(dict(pair=pair, metric=metric, **summarize(differences)))
            rounds.extend(
                dict(pair=pair, metric=metric, round=r, difference_mj=v)
                for r, v in differences.items()
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rounds).to_csv(args.output_dir / "paired_rounds.csv", index=False)
    result = pd.DataFrame(summaries)
    result.to_csv(args.output_dir / "paired_summary.csv", index=False)
    provenance = {
        "inputs": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [args.repeated_observations, args.joint_fit, args.pairs]
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    report = [
        "# Search-pool paired repeats",
        "",
        "Paired rounds; first configuration minus second, mJ per release. "
        "Intervals are paired Student t 95%. STOP-final values are model-derived estimates "
        "conditional on frozen joint-fit coefficients; intervals exclude coefficient uncertainty. "
        "Correction: E_STOP = E_scheduled - 0.1[(T-C)(P_busy-P_stop)-K].",
        "",
        "| Pair | Metric | Mean | 95% interval |",
        "|---|---|---:|---|",
    ]
    report.extend(
        f"| {r.pair} | {r.metric} | {r.mean_mj:+.6f} | [{r.ci95_lo:+.6f}, {r.ci95_hi:+.6f}] |"
        for r in result.itertuples()
    )
    (args.output_dir / "REPORT.md").write_text("\n".join(report) + "\n")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
