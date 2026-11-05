"""Combine disjoint final-trial outcome tables without counting retries twice."""

import argparse
from collections import Counter
import csv
from pathlib import Path

OUTCOMES = ("measured_passed", "measured_violation", "not_deployed_or_incomplete")


def combine(tables):
    """Count unique final trials by study and mutually exclusive outcome.

    Parameters
    ----------
    tables : sequence of iterable of dict
        Rows with case_study, campaign, trial and outcome fields. Only native
        HIL rows should be supplied; host-only work and replay are separate.

    Returns
    -------
    list of dict
        Wide outcome counts with per-study and overall totals.

    Raises
    ------
    ValueError
        If identities are missing, repeated, or assigned an unknown outcome.
    """
    seen = set()
    counts = Counter()
    studies = set()
    for table in tables:
        for row in table:
            key = tuple(
                str(row.get(k, "")) for k in ("case_study", "campaign", "trial")
            )
            if any(not v for v in key) or key in seen:
                raise ValueError("Missing or duplicate final-trial identity")
            if row["outcome"] not in OUTCOMES:
                raise ValueError("Unknown final-trial outcome")
            if row.get("scope", "hil") != "hil" or str(
                row.get("host_only", False)
            ).lower() not in ("false", "0", ""):
                raise ValueError(
                    "Supply native HIL tables only; exclude proxy/replay records"
                )
            seen.add(key)
            studies.add(key[0])
            counts[key[0], row["outcome"]] += 1
    if not seen:
        raise ValueError("No trial outcomes supplied")
    labels = sorted(studies)
    if "Total" in labels:
        raise ValueError("Total is reserved for the aggregate column")
    rows = []
    for outcome in OUTCOMES:
        row = dict(
            outcome=outcome, **{study: counts[study, outcome] for study in labels}
        )
        row["Total"] = sum(row[study] for study in labels)
        rows.append(row)
    total = dict(
        outcome="Total",
        **{
            study: sum(counts[study, outcome] for outcome in OUTCOMES)
            for study in labels
        },
    )
    total["Total"] = len(seen)
    rows.append(total)
    return rows


def main():
    """Write one aggregate table from explicit outcome CSV inputs.

    Returns
    -------
    None
        Writes ``outcome_counts.csv`` in the external output directory.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    tables = []
    for path in args.input:
        with path.open(newline="") as stream:
            tables.append(list(csv.DictReader(stream)))
    rows = combine(tables)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "outcome_counts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Counted {rows[-1]["Total"]} unique final trials.')


if __name__ == "__main__":
    main()
