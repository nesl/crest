# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Check the submitted appendix totals against independently generated CSVs."""

import argparse
import csv
from pathlib import Path

from combine import combine

EXPECTED = {
    "measured_passed": {"CS1": 915, "CS2": 624, "CS3": 1200, "Total": 2739},
    "measured_violation": {"CS1": 564, "CS2": 0, "CS3": 2, "Total": 566},
    "not_deployed_or_incomplete": {"CS1": 57, "CS2": 74, "CS3": 85, "Total": 216},
    "Total": {"CS1": 1536, "CS2": 698, "CS3": 1287, "Total": 3521},
}


def verify(tables):
    """Validate appendix counts and independent periodic population flags.

    Parameters
    ----------
    tables : sequence of list of dict
        Generated final native trial tables, including periodic flags.

    Returns
    -------
    dict
        Confirmed campaign and population totals.

    Raises
    ------
    ValueError
        If observed totals or periodic inclusion relations differ.
    """
    counts = combine(tables)
    for row in counts:
        expected = dict(outcome=row["outcome"], **EXPECTED[row["outcome"]])
        if row != expected:
            raise ValueError(f"Appendix count mismatch: {row!r}; expected {expected!r}")
    periodic = [r for table in tables for r in table if r["case_study"] == "CS1"]
    truth = lambda value: str(value).lower() in ("true", "1")
    passed = sum(r["outcome"] == "measured_passed" for r in periodic)
    eligible = sum(truth(r["paired_eligible"]) for r in periodic)
    included = sum(truth(r["paired_included"]) for r in periodic)
    excluded = sum(
        truth(r["paired_eligible"]) and not truth(r["paired_included"])
        for r in periodic
    )
    if (passed, eligible, included, excluded) != (915, 905, 756, 149):
        raise ValueError("Periodic population count mismatch")
    for row in periodic:
        if truth(row["paired_included"]) and not truth(row["paired_eligible"]):
            raise ValueError("Paired inclusion lacks eligibility")
        if truth(row["paired_eligible"]) and row["outcome"] != "measured_passed":
            raise ValueError("Paired eligibility lacks campaign pass")
        if truth(row["model_eligible"]) != truth(row["paired_included"]):
            raise ValueError("Model population differs from paired population")
    return dict(
        attempts=3521,
        campaign_passed=passed,
        paired_eligible=eligible,
        paired_included=included,
        population_excluded=excluded,
    )


def main(argv=None):
    """Read explicit CSV inputs and print verified appendix totals.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, nargs="+", type=Path)
    args = parser.parse_args(argv)
    tables = []
    for path in args.input:
        with path.open(newline="") as stream:
            tables.append(list(csv.DictReader(stream)))
    print(verify(tables))


if __name__ == "__main__":
    main()
