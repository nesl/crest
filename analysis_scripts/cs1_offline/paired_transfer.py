"""Frozen independent-calibration transfer contrasts; offline, no hardware access."""

from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import t


def ci(x):
    """Compute a paired-round Student t interval.

    Parameters
    ----------
    x : array_like
        Round-level contrasts.

    Returns
    -------
    dict
        Mean and two-sided 95 percent interval bounds.
    """
    x = np.asarray(x, float)
    m = x.mean()
    h = t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return dict(mean=m, lo=m - h, hi=m + h)


def holm_adjust(pvalues):
    """Adjust a pre-specified family of p-values using Holm's method.

    Parameters
    ----------
    pvalues : array_like
        Unadjusted two-sided p-values.

    Returns
    -------
    numpy.ndarray
        Adjusted p-values in input order.
    """
    pvalues = np.asarray(pvalues, float)
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    adjusted[order] = np.minimum(
        1,
        np.maximum.accumulate(
            (len(pvalues) - np.arange(len(pvalues))) * pvalues[order]
        ),
    )
    return adjusted


def main(argv=None):
    """Run frozen transfer calculations from explicit portable inputs.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments; defaults to process arguments.

    Returns
    -------
    None
        Writes paired rounds, intervals, sensitivity, and provenance.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--energy-records", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--holm-cohort", help="Cohort defining one pre-specified comparison family"
    )
    args = parser.parse_args(argv)
    RAW, CAL, OUT = args.energy_records, args.calibration, args.output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    raw = json.loads(RAW.read_text())
    cal = pd.read_csv(CAL).set_index("clock_mhz")
    rows = []
    summaries = []
    sensitivity = []
    for cohort, pairs in raw["pairs"].items():
        records = raw["results"][cohort]
        for pair in pairs:

            def candidate(name, phase):
                """Select accepted ten-run records keyed by round.

                Parameters
                ----------
                name : str
                    Candidate identifier.
                phase : str
                    Measurement phase.

                Returns
                -------
                dict
                    Round identifiers mapped to measurement records.
                """
                rr = [
                    r for r in records if r["candidate"] == name and r["phase"] == phase
                ]
                if not all(r["status"] == "accepted" and r["runs"] == 10 for r in rr):
                    raise ValueError("Expected accepted ten-run measurements")
                if len({r["round"] for r in rr}) != len(rr):
                    raise ValueError("Duplicate candidate/phase/round records")
                return {r["round"]: r for r in rr}

            lc = candidate(pair["left"], "back_to_back")
            rc = candidate(pair["right"], "back_to_back")
            ls = candidate(pair["left"], "cadenced")
            rs = candidate(pair["right"], "cadenced")
            if not lc.keys() == rc.keys() == ls.keys() == rs.keys():
                raise ValueError("Unmatched round keys across candidates or phases")
            if len(lc) < 2:
                raise ValueError("At least two complete paired rounds required")
            k = cal.loc[pair["clock_mhz"]]
            pw = k.wait_mw
            pa = k.active_mw
            pr = []
            for rnd in sorted(lc):
                a, b, c, d = [x[rnd] for x in [lc, rc, ls, rs]]
                energy = lambda r: r["power_metrics"]["energy_mj_per_inference"]
                dwt = (
                    lambda r: r["power_metrics"]["dwt_cycles_per_inference"]
                    / r["power_metrics"]["clock_hz"]
                )
                dc = a["latency_s"] - b["latency_s"]
                dd = dwt(a) - dwt(b)
                ds = dwt(c) - dwt(d)
                de = energy(a) - energy(b)
                obs = energy(c) - energy(d)
                q = dict(
                    cohort=cohort,
                    pair=pair["pair"],
                    round=rnd,
                    clock_mhz=pair["clock_mhz"],
                    observed=obs,
                    delta_Econt=de,
                    delta_C_harness_s=dc,
                    delta_C_dwt_s=dd,
                    delta_S_dwt_s=ds,
                    meas_frozen=de - pw * dc,
                    lat_frozen=(pa - pw) * dc,
                    meas_dwt_cont=de - pw * dd,
                    lat_dwt_cont=(pa - pw) * dd,
                    lat_scheduled_plugin=(pa - pw) * ds,
                    meas_scheduled_plugin=de + pa * (ds - dd) - pw * ds,
                )
                for model in [
                    "meas_frozen",
                    "lat_frozen",
                    "meas_dwt_cont",
                    "lat_dwt_cont",
                    "lat_scheduled_plugin",
                    "meas_scheduled_plugin",
                ]:
                    q[model + "_residual"] = obs - q[model]
                rows.append(q)
                pr.append(q)
            frame = pd.DataFrame(pr)
            for metric in [
                x for x in frame if x not in ["cohort", "pair", "round", "clock_mhz"]
            ]:
                summaries.append(
                    dict(
                        cohort=cohort,
                        pair=pair["pair"],
                        n=len(frame),
                        metric=metric,
                        **ci(frame[metric]),
                    )
                )
            # Illustrative fixed broad positive net-active slopes, plus frozen value. No fit to test pairs.
            for slope in [50, 100, 200, 400, 800, 1200, 1600, pa - pw]:
                pred = slope * frame.delta_S_dwt_s
                sensitivity.append(
                    dict(
                        cohort=cohort,
                        pair=pair["pair"],
                        kind="scheduled_net_slope",
                        value_mw=slope,
                        prediction_mj=pred.mean(),
                        **{
                            "residual_" + k: v
                            for k, v in ci(frame.observed - pred).items()
                        },
                    )
                )
            # Waiting stress range includes frozen value; broad range is a stress test, not a CI.
            for wait in [0, 400, 600, 800, 1000, 1200, pw * 0.95, pw * 1.05, pw]:
                pred = frame.delta_Econt - wait * frame.delta_C_harness_s
                sensitivity.append(
                    dict(
                        cohort=cohort,
                        pair=pair["pair"],
                        kind="continuous_wait",
                        value_mw=wait,
                        prediction_mj=pred.mean(),
                        **{
                            "residual_" + k: v
                            for k, v in ci(frame.observed - pred).items()
                        },
                    )
                )
    panel = pd.DataFrame(rows)
    tests = []
    for pair, frame in panel[panel.cohort == args.holm_cohort].groupby(
        "pair", sort=False
    ):
        values = frame.observed.to_numpy()
        se = values.std(ddof=1) / np.sqrt(len(values))
        pvalue = (
            2 * t.sf(abs(values.mean()) / se, len(values) - 1)
            if se
            else float(values.mean() == 0)
        )
        tests.append(dict(pair=pair, n=len(values), p_two_sided=pvalue))
    tests = pd.DataFrame(tests, columns=["pair", "n", "p_two_sided"])
    tests["p_holm"] = holm_adjust(tests.p_two_sided)
    tests["resolved_holm_005"] = tests.p_holm < 0.05
    tests.to_csv(OUT / "random_panel_holm.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "paired_rounds.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT / "paired_summary.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(OUT / "sensitivity.csv", index=False)
    pd.DataFrame(cal).to_csv(OUT / "frozen_calibration.csv")
    provenance = {
        "inputs": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [RAW, CAL]
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "interval": "Unadjusted paired-round Student t 95%; conditional on frozen calibration",
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    s = pd.DataFrame(summaries)
    metrics = ["observed", "meas_frozen", "meas_frozen_residual", "delta_S_dwt_s"]
    report = [
        "# Frozen paired transfer",
        "",
        "Energy is mJ per release, left minus right. Each round contrasts ten-inference means. "
        "Intervals are unadjusted paired-round Student t 95%, conditional on frozen calibration. "
        "Calibration uncertainty, systematic errors, and cross-round drift are excluded. "
        "Scheduled timing plug-ins and slope grids are exploratory sensitivity analyses.",
        "",
        "| Cohort | Pair | Metric | Mean | 95% CI |",
        "|---|---|---|---:|---|",
    ]
    for row in s[s.metric.isin(metrics)].itertuples():
        report.append(
            f"| {row.cohort} | {row.pair} | {row.metric} | {row.mean:+.6f} | [{row.lo:+.6f}, {row.hi:+.6f}] |"
        )
    report.extend(
        [
            "",
            f"Holm adjustment across the random-panel observed release contrasts resolves {int(tests.resolved_holm_005.sum())} at alpha=0.05; see random_panel_holm.csv. Displayed intervals remain unadjusted.",
        ]
    )
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n")
    print(f"Wrote {len(rows)} paired rounds and {len(summaries)} summaries to {OUT}")


if __name__ == "__main__":
    main()
