#!/usr/bin/env python3
# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Exact upper envelopes of quality - weight * energy / scale.

CSV decimals are interpreted exactly. Intervals include explicit endpoint flags;
point-only winners are retained. Ties use quality descending, energy ascending,
numeric trial ID ascending, then method and source order. This complements the
published coarse score_sensitivity sweep rather than replacing its reports.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from fractions import Fraction as F
from pathlib import Path


@dataclass(frozen=True)
class Candidate:
    """A feasible observation with exact quality and energy values.

    Parameters
    ----------
    board, method, trial_id : str
        Observation group and trial identifier.
    quality, energy : Fraction
        Validation quality and measured inference energy.
    order : int, optional
        Source order used to resolve otherwise identical observations.
    """

    board: str
    method: str
    trial_id: str
    quality: F
    energy: F
    order: int = 0

    def tie_key(self):
        """Return deterministic ordering for score ties.

        Returns
        -------
        tuple
            Quality, energy, identifier, method, and source ordering.
        """
        try:
            trial = (0, F(self.trial_id))
        except ValueError:
            trial = (1, self.trial_id)
        return (-self.quality, self.energy, trial, self.method, self.order)


def select(candidates, weight, scale=F(400)):
    """Select the exact score maximum with deterministic ties.

    Parameters
    ----------
    candidates : sequence of Candidate
        Nonempty observation pool.
    weight : Fraction
        Tradeoff weight.
    scale : Fraction, optional
        Positive energy reference scale.

    Returns
    -------
    Candidate
        Highest scoring observation.
    """
    return min(
        candidates,
        key=lambda c: (-(c.quality - weight * c.energy / scale), c.tie_key()),
    )


def envelope(candidates, lower=F(0), upper=F(1), scale=F(400)):
    """Intersect pairwise score inequalities to obtain exact winning intervals.

    Parameters
    ----------
    candidates : sequence of Candidate
        Nonempty pool of observations.
    lower : Fraction, optional
        Inclusive lower weight bound, strictly below upper.
    upper : Fraction, optional
        Inclusive upper weight bound.
    scale : Fraction, optional
        Positive reference energy scale.

    Returns
    -------
    list of tuple
        Sorted (lower, upper, lower_closed, upper_closed, candidate) intervals.
        Complexity is quadratic in the number of candidates.
    """
    if not candidates or scale <= 0 or lower >= upper:
        raise ValueError(
            "nonempty candidates, positive scale and lower < upper required"
        )
    result = []
    for a in candidates:
        lo, hi = lower, upper
        possible = True
        for b in candidates:
            dq, de = a.quality - b.quality, (a.energy - b.energy) / scale
            if de == 0:
                if dq < 0 or (dq == 0 and a.tie_key() > b.tie_key()):
                    possible = False
                    break
            elif de > 0:
                hi = min(hi, dq / de)
            else:
                lo = max(lo, dq / de)
            if lo > hi:
                possible = False
                break
        if possible:
            lc = select(candidates, lo, scale) is a
            hc = select(candidates, hi, scale) is a
            if lo < hi or (lc and hc):
                result.append((lo, hi, lc, hc, a))
    return sorted(result, key=lambda row: (row[0], row[1], row[4].tie_key()))


def load_inputs(args):
    """Load CSVs, filtering before an optional per-pool feasible limit.

    Parameters
    ----------
    args : argparse.Namespace
        Input paths, column mappings, accepted values, and filtering limits.

    Returns
    -------
    dict
        Board/method keys mapped to eligible candidate lists in CSV order.
    """
    pools = {}
    for spec in args.input:
        label, path = spec.split("=", 1)
        board, method = label.split(",", 1)
        with Path(path).open(newline="") as stream:
            for row in csv.DictReader(stream):
                if (
                    args.status_col
                    and row[args.status_col].casefold() not in args.status_values
                ):
                    continue
                if (
                    args.feasible_col
                    and row[args.feasible_col].casefold() not in args.feasible_values
                ):
                    continue
                try:
                    q, e = F(row[args.quality_col]), F(row[args.energy_col])
                    if args.source_score_col:
                        F(row[args.source_score_col])
                    if args.latency_col and F(row[args.latency_col]) > args.max_latency:
                        continue
                except (ValueError, ZeroDivisionError):
                    continue
                if e < 0:
                    continue
                b = row[args.board_col] if args.board_col else board
                m = row[args.method_col] if args.method_col else method
                pool = pools.setdefault((b, m), [])
                if args.feasible_limit and len(pool) >= args.feasible_limit:
                    continue
                pool.append(Candidate(b, m, row[args.trial_id_col], q, e, len(pool)))
    if not pools or any(not rows for rows in pools.values()):
        raise ValueError("no eligible candidates")
    return pools


def interval_row(scope, interval):
    """Serialize exact endpoints alongside display values and inclusion flags.

    Parameters
    ----------
    scope : str
        Report scope label.
    interval : tuple
        Envelope interval and its winning observation.

    Returns
    -------
    dict
        CSV-compatible interval record.
    """
    lo, hi, lc, hc, c = interval
    return dict(
        scope=scope,
        board=c.board,
        method=c.method,
        trial_id=c.trial_id,
        weight_lo=float(lo),
        weight_hi=float(hi),
        lo_exact=str(lo),
        hi_exact=str(hi),
        lo_closed=lc,
        hi_closed=hc,
        quality=float(c.quality),
        energy=float(c.energy),
    )


def audit(pools, output_dir, weight=F("0.1"), lower=F(0), upper=F(1), scale=F(400)):
    """Write pool envelopes, board winning-pool ranges, and baseline stability.

    Parameters
    ----------
    pools : dict
        Board/method keys mapped to candidate lists.
    output_dir : Path
        Destination for CSV tables and JSON summary.
    weight : Fraction, optional
        Baseline weight within the closed domain.
    lower : Fraction, optional
        Inclusive lower weight bound.
    upper : Fraction, optional
        Inclusive upper weight bound.
    scale : Fraction, optional
        Positive energy reference scale.

    Returns
    -------
    dict
        Summary containing baseline selections and winning-pool intervals.
    """
    if not lower <= weight <= upper:
        raise ValueError("baseline weight must be inside interval")
    rows, baseline = [], []
    for (board, method), candidates in pools.items():
        intervals = envelope(candidates, lower, upper, scale)
        rows.extend(interval_row("selection", s) for s in intervals)
        winner = select(candidates, weight, scale)
        stable = next(s for s in intervals if s[4] is winner)
        baseline.append(
            interval_row("baseline", stable)
            | {"weight": float(weight), "candidate_count": len(candidates)}
        )
    board_rows = []
    for board in sorted({b for b, _ in pools}):
        candidates = [c for (b, _), cs in pools.items() if b == board for c in cs]
        intervals = envelope(candidates, lower, upper, scale)
        merged = []
        for lo, hi, lc, hc, c in intervals:
            if (
                merged
                and merged[-1][4].method == c.method
                and merged[-1][1] == lo
                and (merged[-1][3] or lc)
            ):
                old = merged[-1]
                merged[-1] = (old[0], hi, old[2], hc, old[4])
            else:
                merged.append((lo, hi, lc, hc, c))
        for interval in merged:
            row = interval_row("board_pool", interval)
            for key in ("trial_id", "quality", "energy"):
                row.pop(key)
            board_rows.append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in [
        ("selection_intervals", rows),
        ("baseline_stability", baseline),
        ("board_pool_intervals", board_rows),
    ]:
        with (output_dir / (name + ".csv")).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    summary = dict(
        formula="quality - weight * energy / scale",
        scale=str(scale),
        weight=str(weight),
        domain=[str(lower), str(upper)],
        arithmetic="exact CSV decimals",
        baseline=baseline,
        board_pool_intervals=board_rows,
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv=None):
    """Run explicit-input interval analysis.

    Parameters
    ----------
    argv : sequence of str, optional
        CLI arguments; defaults to process arguments.

    Returns
    -------
    int
        Zero on successful report generation.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", action="append", metavar="BOARD,METHOD=CSV")
    p.add_argument(
        "--observations", type=Path, help="Endpoint-filtered canonical campaign CSV"
    )
    p.add_argument("--output-dir", type=Path, required=True)
    for name, default in [
        ("quality-col", "user_attrs_metric__macro_f1"),
        ("energy-col", "user_attrs_energy_mj_per_inference"),
        ("trial-id-col", "number"),
        ("status-col", "state"),
        ("feasible-col", "user_attrs_feasible"),
        ("source-score-col", "value_score"),
        ("latency-col", "user_attrs_latency_ms"),
        ("board-col", None),
        ("method-col", None),
    ]:
        p.add_argument(
            "--" + name,
            default=default,
            help="CSV column; empty string disables optional filter",
        )
    p.add_argument("--status-values", nargs="+", default=["complete"])
    p.add_argument("--feasible-values", nargs="+", default=["true"])
    p.add_argument(
        "--feasible-limit",
        type=int,
        default=0,
        help="First N eligible CSV rows per board/method; 0 uses all",
    )
    for name, default in [
        ("energy-scale", "400"),
        ("weight", "0.1"),
        ("lower", "0"),
        ("upper", "1"),
        ("max-latency", "2000"),
    ]:
        p.add_argument("--" + name, type=F, default=F(default))
    args = p.parse_args(argv)
    if args.observations:
        if args.input:
            p.error("use either --observations or --input")
        args.input = [f",={args.observations}"]
        args.board_col, args.method_col = "board", "method"
        args.quality_col, args.energy_col, args.trial_id_col = (
            "f1",
            "energy_mj",
            "trial",
        )
        args.source_score_col = "score"
        args.status_col = args.feasible_col = args.latency_col = ""
    if not args.input:
        p.error("--observations or --input required")
    if args.feasible_limit < 0:
        p.error("--feasible-limit must be nonnegative")
    args.status_values = [s.casefold() for s in args.status_values]
    args.feasible_values = [s.casefold() for s in args.feasible_values]
    audit(
        load_inputs(args),
        args.output_dir,
        args.weight,
        args.lower,
        args.upper,
        args.energy_scale,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
