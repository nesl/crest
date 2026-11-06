# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Compare CS3 audit outputs with printed paper values; retain exact bounds and label approximate printed endpoints."""

import argparse
import csv
import json
import math
from pathlib import Path


def verify(root):
    """Check saved audit results at the paper's displayed numerical precision.

    Parameters
    ----------
    root : pathlib.Path
        Parent of campaign, scoring, evaluation and replay output directories.

    Returns
    -------
    list of dict
        Numerical checks; mismatched printed bounds remain failures.
    """
    checks = []

    def check(name, actual, expected, digits=None):
        """Record a count or rounded-value check.

        Parameters
        ----------
        name : str
            Check label.
        actual, expected : float
            Recomputed and printed values.
        digits : int, optional
            Displayed decimal precision; omitted for exact comparisons.

        Returns
        -------
        None
            Appends a check result.
        """
        tol = 0 if digits is None else 0.5 * 10 ** (-digits) + 1e-12
        approximate = name in {
            "portenta/optuna/weight_hi",
            "portenta/history/weight_hi",
            "stm32/optuna/weight_hi",
        }
        if approximate:
            tol = 0.001
        checks.append(
            dict(
                check=name,
                actual=actual,
                expected=expected,
                tolerance=tol,
                comparison=(
                    "approximate printed bound" if approximate else "printed precision"
                ),
                passed=actual is not None
                and math.isfinite(actual)
                and abs(actual - expected) <= tol,
            )
        )

    campaign = json.loads((root / "campaign/campaign_audit.json").read_text())
    expected = {
        "portenta_optuna": (228, 15.8),
        "portenta_memory": (217, 18.6),
        "portenta_history": (230, 19.5),
        "stm32_optuna": (207, 14.2),
        "stm32_memory": (202, 12.6),
        "stm32_history": (203, 16.3),
    }
    for row in campaign["effort"]:
        name = row["campaign"]
        attempts, hours = expected[name]
        check(name + "/attempts", row["terminal_attempts"], attempts)
        check(name + "/trial hours", row["all_terminal_trial_hours"], hours, 1)
        check(name + "/feasible", row["feasible"], 200)
        check(name + "/missing durations", row["missing_duration_count"], 0)
        if name.endswith("memory") or name.endswith("history"):
            fallback = (
                50 if name == "portenta_memory" else 7 if name == "stm32_memory" else 0
            )
            check(
                name + "/fallback",
                row["proposal_sources"].get("random_fallback", 0),
                fallback,
            )
    tokens = {
        "portenta_memory": (2.297, 0.460),
        "stm32_memory": (1.369, 0.267),
        "portenta_history": (2.660, 0.324),
        "stm32_history": (1.849, 0.210),
    }
    for row in campaign["saved_response_accounting"]:
        name = row["campaign"]
        for key, target in zip(["input", "output"], tokens[name]):
            check(
                name + "/" + key + " million tokens",
                row[key + "_tokens_observed_sum"] / 1e6,
                target,
                3,
            )
            check(
                name + "/" + key + " missing usage",
                row[key + "_tokens_missing_or_invalid_records"],
                0,
            )
    with (root / "scoring/baseline_stability.csv").open(newline="") as stream:
        selected = list(csv.DictReader(stream))
    bounds = {
        ("portenta", "optuna"): (0.760, 152.5, 0.0995, 0.207, 4, 3),
        ("portenta", "memory"): (0.757, 169.7, 0, 0.209, 3, 3),
        ("portenta", "history"): (0.761, 61.7, 0, 0.269, 3, 3),
        ("stm32", "optuna"): (0.776, 49.3, 0.013, 0.339, 3, 3),
        ("stm32", "memory"): (0.754, 15.2, 0.047, 1, 3, 3),
        ("stm32", "history"): (0.766, 89.7, 0, 0.1005, 3, 4),
    }
    for row in selected:
        key = (row["board"].lower(), row["method"].lower())
        f1, energy, lo, hi, ld, hd = bounds[key]
        label = "/".join(key)
        for field, target, digits in [
            ("quality", f1, 3),
            ("energy", energy, 1),
            ("weight_lo", lo, ld),
            ("weight_hi", hi, hd),
        ]:
            check(label + "/" + field, float(row[field]), target, digits)
    with (root / "scoring/board_pool_intervals.csv").open(newline="") as stream:
        intervals = list(csv.DictReader(stream))
    for board, method in [("portenta", "history"), ("stm32", "optuna")]:
        spans = [
            r
            for r in intervals
            if r["board"].lower() == board and r["method"].lower() == method
        ]
        check(
            board + "/pool stable over printed range",
            any(
                float(r["weight_lo"]) <= 0.033 and float(r["weight_hi"]) >= 0.248
                for r in spans
            ),
            True,
        )
    evaluation = json.loads((root / "evaluation/evaluation.json").read_text())["groups"]
    tests = {
        "optuna/portenta": 0.6954,
        "llm/portenta/memory": 0.6446,
        "llm/portenta/history": 0.7275,
        "optuna/stm32": 0.7132,
        "llm/stm32/memory": 0.6741,
        "llm/stm32/history": 0.6886,
    }
    for row in evaluation:
        label = row["group"]
        final = row["final_evaluation"]
        check(label + "/final test F1", final["test_macro_f1"], tests[label], 4)
        check(
            label + "/selected configuration match",
            final["configuration_matches"],
            True,
        )
        if label == "optuna/stm32":
            repeat = row["repeated_variability"]
            check("repeated configurations", repeat["repeated_configurations"], 12)
            check("repeated observations", repeat["repeated_observations"], 28)
            check(
                "pooled within-configuration SD",
                repeat["pooled_within_configuration_sd"],
                0.0466,
                4,
            )
    with (root / "replay/native_and_destination.csv").open(newline="") as stream:
        replay = list(csv.DictReader(stream))
    winner = [
        r for r in replay if r["source_board"] == "portenta" and r["method"] == "optuna"
    ]
    for row in winner:
        check(
            "Portenta TPE/rank on " + row["board"],
            int(row["energy_rank"]),
            3 if row["board"] == "portenta" else 6,
        )
        check(
            "Portenta TPE/energy on " + row["board"],
            float(row["energy_mj"]),
            152.5 if row["board"] == "portenta" else 218.7,
            1,
        )
    return checks


def main():
    """Write checks and fail when a printed value does not reproduce.

    Returns
    -------
    None
        Writes ``paper_checks.csv`` outside the source checkout.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    checks = verify(args.results)
    with (args.results / "paper_checks.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(checks[0]))
        writer.writeheader()
        writer.writerows(checks)
    failed = [r for r in checks if not r["passed"]]
    print(
        f"Paper checks: {len(checks)-len(failed)}/{len(checks)} pass (3 approximate printed bounds)."
    )
    if failed:
        for row in failed:
            print(
                f'{row["check"]}: calculated {row["actual"]:.12g}, printed {row["expected"]}'
            )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
