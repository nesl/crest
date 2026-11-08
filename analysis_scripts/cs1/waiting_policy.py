# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Reproduce frozen final-interval policy contrasts without hardware access."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
from scipy.stats import t


def summary(values):
    """Summarize paired round differences with a Student t interval.

    Parameters
    ----------
    values : sequence of float
        Paired round energy differences in mJ; at least two are needed.

    Returns
    -------
    dict
        Count, mean, nominal 95% Student t bounds, and original differences.
    """
    n = len(values)
    mean = statistics.mean(values)
    half = t.ppf(0.975, n - 1) * statistics.stdev(values) / n**0.5
    return dict(
        n=n,
        mean_mj=mean,
        ci95_lo=mean - half,
        ci95_hi=mean + half,
        differences_mj=values,
    )


def energy(record):
    """Extract measured energy per inference.

    Parameters
    ----------
    record : dict
        Measurement record containing nested ``power_metrics``.

    Returns
    -------
    float
        Recorded energy per inference in mJ.
    """
    return record["power_metrics"]["energy_mj_per_inference"]


def main(argv=None):
    """Analyze frozen policy inputs and write paired summaries.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Command-line arguments; None reads the process arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--energy-records", required=True, type=Path)
    parser.add_argument(
        "--pairs",
        required=True,
        type=Path,
        help="JSON list of {left, right} candidate IDs",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    records = json.loads(args.energy_records.read_text())
    if not records or not all(
        r["status"] == "accepted" and r["runs"] == 10 for r in records
    ):
        raise ValueError("Expected accepted ten-run records")
    rows = {(r["round"], r["candidate"], r["shape"], r["policy"]): r for r in records}
    if len(rows) != len(records):
        raise ValueError("Duplicate round/candidate/shape/policy records")
    out = {}
    rounds = sorted({r["round"] for r in records if r["shape"] == "w10"})
    if len(rounds) < 2:
        raise ValueError("At least two complete paired rounds required")
    for pair_spec in json.loads(args.pairs.read_text()):
        left, right = pair_spec["left"], pair_spec["right"]
        for policy in ["awake", "stop"]:
            out[f"{left}_minus_{right}_{policy}"] = summary(
                [
                    energy(rows[r, left, "w10", policy])
                    - energy(rows[r, right, "w10", policy])
                    for r in rounds
                ]
            )
        out[f"{left}_minus_{right}_policy_interaction"] = summary(
            [
                (
                    energy(rows[r, left, "w10", "stop"])
                    - energy(rows[r, right, "w10", "stop"])
                )
                - (
                    energy(rows[r, left, "w10", "awake"])
                    - energy(rows[r, right, "w10", "awake"])
                )
                for r in rounds
            ]
        )
    for candidate in sorted({r["candidate"] for r in records if r["shape"] == "w10"}):
        out[f"{candidate}_stop_minus_awake"] = summary(
            [
                energy(rows[r, candidate, "w10", "stop"])
                - energy(rows[r, candidate, "w10", "awake"])
                for r in rounds
            ]
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "paired_analysis.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )
    provenance = {
        "inputs": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [args.energy_records, args.pairs]
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    report = [
        "# Final-interval policy replay",
        "",
        "Accepted ten-inference measurements paired by round. "
        "Energy differences are mJ per release. Intervals are paired Student t 95% across rounds. "
        "This is one session; intervals exclude systematic errors and cross-round drift.",
        "",
        "| Contrast | Mean | 95% CI |",
        "|---|---:|---|",
    ]
    for name, result in out.items():
        report.append(
            f"| {name} | {result['mean_mj']:+.6f} | [{result['ci95_lo']:+.6f}, {result['ci95_hi']:+.6f}] |"
        )
    (args.output_dir / "REPORT.md").write_text("\n".join(report) + "\n")
    print(f"Wrote {len(out)} paired policy summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
