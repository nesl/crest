#!/usr/bin/env python3
"""Reproduce nominal-frequency scaling and paired period contrasts offline."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import t


def ci(v):
    """Compute an unadjusted Student t interval.

    Parameters
    ----------
    v : array_like
        Independent paired-round contrasts.

    Returns
    -------
    dict
        Count, mean, and nominal 95 percent interval endpoints.
    """
    a = np.asarray(v, dtype=float)
    n = len(a)
    if n < 2 or not np.isfinite(a).all():
        raise ValueError("At least two finite observations are required")
    mean = a.mean()
    h = t.ppf(0.975, n - 1) * a.std(ddof=1) / np.sqrt(n)
    return dict(n=n, mean=mean, ci95_lo=mean - h, ci95_hi=mean + h)


def run(archive_root, output_dir):
    """Validate provenance and export scaling and paired period summaries.

    Parameters
    ----------
    archive_root : pathlib.Path
        Research archive containing analysis_results and outputs.
    output_dir : pathlib.Path
        Destination for CSV summaries, report, and input hashes.

    Returns
    -------
    None
        Writes derived outputs only.
    """
    ROOT = Path(archive_root).resolve()
    OUT = Path(output_dir).resolve()
    SRC = ROOT / "analysis_results/cs2_sleep_ratio_model_20261027"
    AUDIT = ROOT / "outputs/calibration_direct_audit_20261101/remote"
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = []

    def read(p):
        """Read and register a CSV input.

        Parameters
        ----------
        p : pathlib.Path
            Input path.

        Returns
        -------
        pandas.DataFrame
            Parsed records.
        """
        inputs.append(p)
        return pd.read_csv(p)

    def js(p):
        """Read and register a JSON input.

        Parameters
        ----------
        p : pathlib.Path
            Input path.

        Returns
        -------
        object
            Decoded JSON document.
        """
        inputs.append(p)
        return json.loads(p.read_text())

    meta = js(AUDIT / "metadata.json")
    ident = js(AUDIT / "artifact_identity.json")
    identity137 = js(
        ROOT
        / "analysis_results/offline_model_validation_20261103/scaling/trial137_remote_identity.json"
    )
    assert len(identity137) == 6
    assert {r["actual_weights_sha256"] for r in identity137} == {
        "e8b8ad837aedd7311cd788019337be42a2bdf47aed8e75171b21955c6a4e3c06"
    }
    runner = meta["direct"]["study.json"]["firmware_git_state"]["file_sha256"][
        "sketches/stm32/crest_stm32_lrun/Appli/Src/crest_dut_runner.c"
    ]
    assert all(r["runner_sha256"] == runner for r in identity137)
    assert all(
        r["last_programmed_weights_sha256"] == r["actual_weights_sha256"]
        for r in identity137
        if "/variants/" in r["project"]
    )
    for name, v in meta["direct"]["artifacts/variants.json"].items():
        if v["shape"] == "w10":
            assert (
                v["image_protocol"]["protocol"]
                == meta["direct"]["study.json"]["expected_protocols"][v["policy"]]
            )
            assert v["device_options"]["latency_budget_ms"] == v["period_ms"]
            assert v["device_options"]["final_slot_policy"] == v["policy"]

    # Verify retained actual-file hashes, separately from application binaries (which
    # must differ across clock/period/policy builds).
    weight_rows = []
    for campaign, entries in ident.items():
        for path, val in entries.items():
            if isinstance(val, dict) and "actual_weights_sha256" in val:
                weight_rows.append(
                    dict(
                        campaign=campaign,
                        path=path,
                        sha256=val["actual_weights_sha256"],
                    )
                )
    weights = pd.DataFrame(weight_rows)
    assert weights.sha256.nunique() == 1
    assert (
        weights.sha256.iloc[0]
        == "2cf9b35315e07f38e8e0e238a8b128c3fdb20dfca8488a7ee24de422009d1706"
    )
    weights.to_csv(OUT / "verified_trial203_hashes.csv", index=False)
    assert (
        meta["cal500"]["study.json"]["architecture"]
        == meta["cal100"]["study.json"]["architecture"]
    )
    assert (
        meta["cal500"]["study.json"]["firmware_git_state"]["file_sha256"]
        == meta["cal100"]["study.json"]["firmware_git_state"]["file_sha256"]
    )
    all_sweeps = []
    stats = []
    extents = []
    opt = []
    for period, folder in [(500, "probe_v2"), (100, "probe_100ms_v1")]:
        d = read(SRC / folder / "raw_measurements.csv")
        assert len(d) == 75 and d.status.eq("accepted").all()
        assert not d.duplicated(["round", "clock_mhz", "shape"]).any()
        assert np.allclose(d.C_dwt_s, d.dwt_cycles_per_inference / (d.clock_mhz * 1e6))
        d["period_ms"] = period
        d["effective_cycles"] = d.clock_mhz * 1e6 * d.C_dwt_s
        all_sweeps.append(d)
        for shape, g in d.groupby("shape"):
            mean = g.groupby("clock_mhz")[
                ["effective_cycles", "C_dwt_s", "energy_mj_per_inference"]
            ].mean()
            base = g[g.clock_mhz.eq(200)].set_index("round").effective_cycles
            for f, h in g.groupby("clock_mhz"):
                z = h.set_index("round").effective_cycles / base - 1
                stats.append(
                    dict(
                        period_ms=period,
                        shape=shape,
                        clock_mhz=f,
                        cycles_mean=h.effective_cycles.mean(),
                        latency_ms_mean=1000 * h.C_dwt_s.mean(),
                        energy_mj_mean=h.energy_mj_per_inference.mean(),
                        **{"cycle_ratio_minus1_" + k: v for k, v in ci(z).items()}
                    )
                )
            extents.append(
                dict(
                    period_ms=period,
                    shape=shape,
                    max_minus_min_cycles_pct=100
                    * (mean.effective_cycles.max() / mean.effective_cycles.min() - 1),
                )
            )
            opt.append(
                dict(
                    period_ms=period,
                    shape=shape,
                    minimum_latency_frequency_mhz=int(mean.C_dwt_s.idxmin()),
                    minimum_measured_energy_frequency_mhz=int(
                        mean.energy_mj_per_inference.idxmin()
                    ),
                )
            )
    sweeps = pd.concat(all_sweeps, ignore_index=True)
    pd.DataFrame(stats).to_csv(OUT / "frequency_scaling.csv", index=False)
    pd.DataFrame(extents).to_csv(OUT / "scaling_extents.csv", index=False)
    pd.DataFrame(opt).to_csv(OUT / "single_architecture_optima.csv", index=False)
    # Direct within-session design has one fixed-frequency architecture pair, across
    # both periods and both policies. Trial137 hash was not retained by prior hash
    # audit; a targeted read-only retrieval now closes that actual-hash gap.
    direct = read(SRC / "final_slot_v1/raw_measurements.csv")
    assert len(direct) == 75 and direct.status.eq("accepted").all()
    w = direct[direct["shape"].eq("w10")]
    assert w.deadline_miss_count.eq(0).all()
    assert (w.stop_entries == np.where(w.policy.eq("stop"), 10, 9)).all()
    assert np.allclose(w.timer_per_window_s, w.period_ms / 100)
    rows = []
    period_diffs = []
    for policy, g in w[w.clock_mhz.eq(300)].groupby("policy"):
        p = g.pivot(
            index=["round", "period_ms"],
            columns="candidate",
            values="energy_mj_per_inference",
        )
        gaps = (p.continuous_137 - p.continuous_203).unstack("period_ms")
        for period in [100, 500]:
            rows.append(
                dict(
                    policy=policy,
                    period_ms=period,
                    contrast="137minus203_mJ",
                    **ci(gaps[period])
                )
            )
        delta = gaps[500] - gaps[100]
        rows.append(
            dict(
                policy=policy,
                period_ms="500minus100",
                contrast="gap_change_mJ",
                **ci(delta)
            )
        )
        period_diffs.append(
            pd.DataFrame(
                dict(
                    policy=policy,
                    round=delta.index,
                    gap100_mJ=gaps[100],
                    gap500_mJ=gaps[500],
                    gap_change_mJ=delta,
                )
            )
        )
    pd.DataFrame(rows).to_csv(OUT / "period_gap_contrasts.csv", index=False)
    pd.concat(period_diffs).to_csv(OUT / "period_gap_by_round.csv", index=False)
    # Same trial203 at two frequencies: this is NOT the fixed-frequency pair-gap
    # invariance proposition; frequency-dependent waiting power allows a T slope.
    clock_rows = []
    for policy, g in w[w.candidate.eq("continuous_203")].groupby("policy"):
        p = g.pivot(
            index=["round", "period_ms"],
            columns="clock_mhz",
            values="energy_mj_per_inference",
        )
        gaps = (p[800] - p[300]).unstack("period_ms")
        for period in [100, 500]:
            clock_rows.append(
                dict(
                    policy=policy,
                    period_ms=period,
                    contrast="800minus300_mJ",
                    **ci(gaps[period])
                )
            )
        clock_rows.append(
            dict(
                policy=policy,
                period_ms="slope",
                contrast="clock_gap_slope_mW",
                **ci((gaps[500] - gaps[100]) / 0.4)
            )
        )
    pd.DataFrame(clock_rows).to_csv(
        OUT / "same_artifact_clock_period_contrasts.csv", index=False
    )
    # Repeatability intervals are unadjusted and do not cover drift or provenance.
    # Pairing uses the interleaved direct session's round blocks. Calibration ratios
    # pair clocks within rounds, never pair separate period-campaign rounds.
    manifest = [
        dict(
            path=str(p.relative_to(ROOT)),
            sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        )
        for p in inputs
    ]
    (OUT / "input_hashes.json").write_text(json.dumps(manifest, indent=2) + "\n")

    report = """# Scaling and period contrasts

Frequency ratios pair clocks within rounds; separate calibration campaigns are never paired.
CSV outputs give effective-cycle ranges, measured frequency minima, fixed-frequency
137-minus-203 period contrasts, and same-artifact clock contrasts. Positive gaps
mean the first named configuration uses more energy. All intervals are nominal,
unadjusted five-round Student t repeatability intervals.

Only trial 203 spans frequencies. These data cannot test latency-curve crossings,
a change in optimal architecture, or architecture preservation. Nominal DWT clock
scaling is not independent physical-clock calibration. Period campaigns differ in
time; direct-session period comparisons are interleaved. Artifact hashes support
retained provenance, not source-to-binary certification or a causal mechanism.
"""
    report += (
        "\nEffective-cycle ranges (%):\n\n"
        + pd.DataFrame(extents).to_string(index=False)
        + "\n\nMeasured frequency minima:\n\n"
        + pd.DataFrame(opt).to_string(index=False)
        + "\n\nFixed-frequency period contrasts (mJ):\n\n"
        + pd.DataFrame(rows).to_string(index=False)
        + "\n"
    )
    (OUT / "REPORT.md").write_text(report)


def analyze_probes(probes, output_dir):
    """Analyze arbitrary accepted nominal-frequency probe campaigns.

    Parameters
    ----------
    probes : sequence of tuple
        Period in milliseconds and raw-measurement CSV path per campaign.
    output_dir : pathlib.Path
        Destination for scaling CSVs and input hashes.

    Returns
    -------
    None
        Writes scaling ratios against each shape's lowest observed frequency.
    """
    OUT = Path(output_dir).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    all_sweeps = []
    stats = []
    extents = []
    opt = []
    for period, path in probes:
        d = pd.read_csv(path)
        if not d.status.eq("accepted").all():
            raise ValueError("Only accepted records may be analyzed")
        assert not d.duplicated(["round", "clock_mhz", "shape"]).any()
        assert np.allclose(d.C_dwt_s, d.dwt_cycles_per_inference / (d.clock_mhz * 1e6))
        d["period_ms"] = period
        d["effective_cycles"] = d.clock_mhz * 1e6 * d.C_dwt_s
        all_sweeps.append(d)
        for shape, g in d.groupby("shape"):
            mean = g.groupby("clock_mhz")[
                ["effective_cycles", "C_dwt_s", "energy_mj_per_inference"]
            ].mean()
            base = (
                g[g.clock_mhz.eq(g.clock_mhz.min())].set_index("round").effective_cycles
            )
            for f, h in g.groupby("clock_mhz"):
                observed = h.set_index("round").effective_cycles
                if set(observed.index) != set(base.index):
                    raise ValueError("Unmatched frequency round blocks")
                z = observed / base - 1
                stats.append(
                    dict(
                        period_ms=period,
                        shape=shape,
                        clock_mhz=f,
                        cycles_mean=h.effective_cycles.mean(),
                        latency_ms_mean=1000 * h.C_dwt_s.mean(),
                        energy_mj_mean=h.energy_mj_per_inference.mean(),
                        **{"cycle_ratio_minus1_" + k: v for k, v in ci(z).items()}
                    )
                )
            extents.append(
                dict(
                    period_ms=period,
                    shape=shape,
                    max_minus_min_cycles_pct=100
                    * (mean.effective_cycles.max() / mean.effective_cycles.min() - 1),
                )
            )
            opt.append(
                dict(
                    period_ms=period,
                    shape=shape,
                    minimum_latency_frequency_mhz=int(mean.C_dwt_s.idxmin()),
                    minimum_measured_energy_frequency_mhz=int(
                        mean.energy_mj_per_inference.idxmin()
                    ),
                )
            )
    sweeps = pd.concat(all_sweeps, ignore_index=True)
    pd.DataFrame(stats).to_csv(OUT / "frequency_scaling.csv", index=False)
    pd.DataFrame(extents).to_csv(OUT / "scaling_extents.csv", index=False)
    pd.DataFrame(opt).to_csv(OUT / "single_architecture_optima.csv", index=False)

    (OUT / "input_hashes.json").write_text(
        json.dumps(
            [
                dict(
                    path=str(Path(path).resolve()),
                    sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                )
                for _, path in probes
            ],
            indent=2,
        )
        + "\n"
    )
    (OUT / "REPORT.md").write_text(
        "Nominal-clock frequency scaling; lowest measured frequency is each shape baseline.\n"
        "Intervals pair observed round blocks and describe repeatability. Cross-campaign comparisons are not paired.\n"
        "Measured minima do not establish architecture preservation or latency-curve crossings.\n"
    )


def main():
    """Parse explicit archive and output paths and run the audit.

    Returns
    -------
    None
        Writes requested analysis outputs.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-root", type=Path, help="Optional retained-archive provenance audit"
    )
    parser.add_argument("--probe-100", type=Path)
    parser.add_argument("--probe-500", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    probes = [
        (period, path)
        for period, path in ((100, args.probe_100), (500, args.probe_500))
        if path is not None
    ]
    if probes:
        analyze_probes(probes, args.output_dir)
    elif args.archive_root:
        run(args.archive_root, args.output_dir)
    else:
        parser.error("Provide --probe-100/--probe-500 or --archive-root")


if __name__ == "__main__":
    main()
