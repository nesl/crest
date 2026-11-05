"""Report operational totals, matched trained candidates, and RMSE discrepancies.

All four conditions are required per board for a matched comparison. Differences
are flags for investigation, never automatic proof that implementations differ.
"""

import argparse
import json
import math
from pathlib import Path
from collections import defaultdict

CONDITIONS = {"continuous", "cadenced", "flops", "memory"}


def compare(reports, *, atol=1e-5, rtol=1e-3, requests=None):
    """Compare matched completed candidates and preserve operational totals.

    Parameters
    ----------
    reports : sequence of dict
        Results schema 2 reports, at most one condition per board and timing mode.
    atol, rtol : float
        Absolute and relative RMSE tolerances.
    requests : sequence of dict, optional
        Analyzer request records used to associate matched HIL costs.

    Returns
    -------
    list of dict
        Quality checks and operational totals grouped by board and timing mode.
    """
    boards = defaultdict(dict)
    for report in reports:
        if report.get("schema_version") != 2:
            raise ValueError("Expected results schema 2")
        run = report["run"]
        key = (run["board"], report["timing_enabled"])
        if run["condition"] in boards[key]:
            raise ValueError(
                "Duplicate condition; compare one matched repetition at a time"
            )
        boards[key][run["condition"]] = report
    output = []
    for (board, timing), conditions in boards.items():
        identities = {
            (r["run"]["manifest_sha256"], r["run"]["seed"]) for r in conditions.values()
        }
        if len(identities) != 1:
            raise ValueError("Cannot match different manifests/seeds")
        missing = sorted(CONDITIONS - set(conditions))
        trained = {
            c: {
                t["candidate_id"]: t
                for t in r["trials"]
                if t["trained"] and t["state"] == "COMPLETE"
            }
            for c, r in conditions.items()
        }
        matched = (
            sorted(set.intersection(*(set(t) for t in trained.values())))
            if not missing
            else []
        )
        quality = []
        for candidate in matched:
            rmse = {
                c: trained[c][candidate]["user_attrs"]
                .get("task_metrics", {})
                .get("rmse_total")
                for c in CONDITIONS
            }
            finite = all(
                isinstance(v, (int, float)) and math.isfinite(v) for v in rmse.values()
            )
            reference = rmse.get("flops")
            agrees = finite and all(
                math.isclose(v, reference, abs_tol=atol, rel_tol=rtol)
                for v in rmse.values()
            )
            quality.append(
                {
                    "candidate_id": candidate,
                    "rmse": rmse,
                    "rmse_matches_tolerance": agrees,
                    "missing_rmse": not finite,
                }
            )
        output.append(
            {
                "board": board,
                "timing_enabled": timing,
                "missing_conditions": missing,
                "all_runs_complete": not missing
                and all(r["status"] == "complete" for r in conditions.values()),
                "matched_trained_candidates": matched,
                "quality_checks": quality,
                "rmse_tolerance": {"absolute": atol, "relative": rtol},
                "operational": {
                    c: {
                        "status": r["status"],
                        "counts": r["counts"],
                        "elapsed_s": r["elapsed_s"],
                        "all_attempt_duration_s": sum(
                            t["duration_s"] or 0 for t in r["trials"]
                        ),
                        "first_candidate": [
                            t for t in r["trials"] if t["first_candidate"]
                        ],
                        "matched_trained_duration_s": sum(
                            trained[c][i]["duration_s"] or 0 for i in matched
                        ),
                        "hil_costs_on_matched_trained_candidates": (
                            None
                            if requests is None
                            else [
                                q
                                for q in requests
                                if q["run_id"] == r["run"]["run_id"]
                                and q.get("candidate_id") in matched
                            ]
                        ),
                    }
                    for c, r in conditions.items()
                },
            }
        )
    return output


def main():
    """Write a quality comparison from local results and optional request evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--requests",
        type=Path,
        help="requests.json from analyze.py for hardware-stage cost comparisons",
    )
    parser.add_argument("--rmse-atol", type=float, default=1e-5)
    parser.add_argument("--rmse-rtol", type=float, default=1e-3)
    args = parser.parse_args()
    output = compare(
        [json.loads(p.read_text()) for p in args.results],
        atol=args.rmse_atol,
        rtol=args.rmse_rtol,
        requests=(
            None if args.requests is None else json.loads(args.requests.read_text())
        ),
    )
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
