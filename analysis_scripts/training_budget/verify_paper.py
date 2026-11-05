"""Check training-budget outputs against the submitted appendix (no bundled data)."""

import argparse
import csv
import json
from pathlib import Path


def verify(summary):
    """Compare reproduced values with printed counts and rounding precision.

    Parameters
    ----------
    summary : dict
        Output from ``analyze.py``.

    Returns
    -------
    list of dict
        Named checks, including actual values and pass/fail status.
    """
    checks = []
    seed = summary["seeds"]["17"]
    expected = [
        ("configurations", summary["candidate_count"], 50, 0),
        ("float configurations", summary["quantization_counts"].get("float", 0), 22, 0),
        (
            "int8 configurations",
            summary["quantization_counts"].get("int8_ptq", 0),
            28,
            0,
        ),
        ("reference budget", summary["reference_budget"], 300, 0),
        ("continued reference", summary["reference_policy"] == "budget_only", True, 0),
        ("55-epoch policies agree", seed["policies_identical"]["55"], True, 0),
    ]
    for budget, rho in [
        (15, 0.554),
        (55, 0.842),
        (100, 0.926),
        (200, 0.953),
        (300, 1.0),
    ]:
        matches = [
            r
            for r in seed["comparisons"]
            if r["budget"] == budget and r["policy"] == "budget_only"
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one continued-training comparison at budget {budget}."
            )
        expected.append(
            (f"Spearman at {budget}", matches[0]["spearman_rho"], rho, 0.0005)
        )
    for budget, hours in [(55, 4.8), (300, 24.4)]:
        expected.append(
            (
                f"training hours at {budget}",
                seed["cumulative_training_hours"][str(budget)],
                hours,
                0.05,
            )
        )
    for name, actual, target, tolerance in expected:
        checks.append(
            dict(
                check=name,
                actual=actual,
                expected=target,
                tolerance=tolerance,
                passed=actual is not None and abs(actual - target) <= tolerance,
            )
        )
    return checks


def main():
    """Write a local check table and return failure for any mismatch.

    Returns
    -------
    None
        Writes ``paper_checks.csv`` beside the supplied summary.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    checks = verify(json.loads(args.summary.read_text()))
    with args.summary.with_name("paper_checks.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(checks[0]))
        writer.writeheader()
        writer.writerows(checks)
    failed = [r["check"] for r in checks if not r["passed"]]
    if failed:
        raise SystemExit("Failed: " + ", ".join(failed))
    print(f"Submitted-paper checks: {len(checks)}/{len(checks)} passed.")


if __name__ == "__main__":
    main()
