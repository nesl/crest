"""Check regenerated timing tables against the submitted evaluation-cost appendix."""

import argparse
import csv
import json
import math
from pathlib import Path


def verify(directory):
    """Check the two submitted panels at their printed precision.

    Parameters
    ----------
    directory : pathlib.Path
        STM32 tables with Portenta tables in ``portenta_m7/``.

    Returns
    -------
    list of dict
        Check names, numerical values and pass/fail status.
    """
    checks = []
    for board, folder, expected in [
        ("stm32", directory, [5.77, 5.13, 0.52, 0.13, 1.34, 2.03]),
        (
            "portenta_m7",
            directory / "portenta_m7",
            [5.58, 4.97, 0.51, 0.10, 2.87, 3.27],
        ),
    ]:
        data = json.loads((folder / "breakdown.json").read_text())
        arena = json.loads((folder / "arena_search.json").read_text())["conditions"]
        conditions = data["conditions"]
        n = sum(v["trained_candidates"] for v in conditions.values())
        category = {
            key: sum(v["categories_s"].get(key, 0) for v in conditions.values())
            / n
            / 60
            for key in (
                "training_fit",
                "post_fit_evaluation",
                "loading",
                "model_and_proxy_setup",
                "other_startup_bookkeeping",
            )
        }
        setup = sum(
            category[k]
            for k in ("loading", "model_and_proxy_setup", "other_startup_bookkeeping")
        )
        training, evaluation = category["training_fit"], category["post_fit_evaluation"]
        actual = [setup + training + evaluation, training, evaluation, setup] + [
            conditions[c]["categories_s"]["hil_request"]
            / conditions[c]["trained_candidates"]
            / 60
            for c in ("continuous", "cadenced")
        ]
        values = [
            (name, value, target, 0.005)
            for name, value, target in zip(
                [
                    "host minutes",
                    "training minutes",
                    "evaluation minutes",
                    "setup minutes",
                    "continuous HIL minutes",
                    "periodic HIL minutes",
                ],
                actual,
                expected,
            )
        ]
        values.extend(
            [
                ("evaluations", n, 40, 0),
                ("matched requests", data["validation"]["matched_requests"], 20, 0),
                ("linked workers", data["validation"]["linked_workers"], 40, 0),
                ("orphan parents", data["validation"]["orphan_parents"], 0, 0),
                ("worker issues", data["validation"]["worker_issues"], 0, 0),
            ]
        )
        for c, v in conditions.items():
            values.extend(
                [
                    (c + " candidates", v["trained_candidates"], 10, 0),
                    (c + " training epochs", v["epochs"], 550, 0),
                    (
                        c + " accounting closure",
                        sum(v["categories_s"].values()) - v["client_process_s"],
                        0,
                        1e-5,
                    ),
                ]
            )
        for c in ("continuous", "cadenced"):
            if board == "portenta_m7":
                values.extend(
                    [
                        (c + " base attempts", arena[c]["base_attempts"], 53, 0),
                        (c + " failed probes", arena[c]["failed_attempts"], 14, 0),
                        (
                            c + " successful probes",
                            arena[c]["successful_base_attempts"],
                            39,
                            0,
                        ),
                        (
                            c + " timeout seconds per candidate",
                            arena[c]["failed_probe_measurement_s"] / 10,
                            50.5,
                            0.05,
                        ),
                        (
                            c + " followup attempts",
                            arena[c]["followup_attempts"],
                            10 if c == "cadenced" else 0,
                            0,
                        ),
                    ]
                )
            else:
                detail = conditions[c]["hil_detail_s"]
                values.extend(
                    [
                        (
                            c + " phase attempts",
                            arena[c]["phase_attempts"],
                            20 if c == "cadenced" else 10,
                            0,
                        ),
                        (c + " failed probes", arena[c]["failed_attempts"], 0, 0),
                        (
                            c + " firmware build seconds",
                            detail["firmware_build_signing"] / 10,
                            8.0 if c == "cadenced" else 4.8,
                            0.05,
                        ),
                        (
                            c + " export/preparation seconds",
                            detail["export_stedgeai_analysis_codegen"] / 10,
                            35,
                            1,
                        ),
                    ]
                )
        for name, value, target, tolerance in values:
            checks.append(
                dict(
                    check=f"{board}/{name}",
                    actual=value,
                    expected=target,
                    tolerance=tolerance,
                    passed=math.isfinite(value)
                    and abs(value - target) <= tolerance + 1e-10,
                )
            )
    return checks


def main():
    """Write numerical checks beside the supplied external timing tables.

    Returns
    -------
    None
        Writes ``paper_checks.csv`` and fails on mismatches.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    checks = verify(args.data_dir)
    with (args.data_dir / "paper_checks.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(checks[0]))
        writer.writeheader()
        writer.writerows(checks)
    failures = [row["check"] for row in checks if not row["passed"]]
    if failures:
        raise SystemExit("Failed: " + ", ".join(failures))
    print(f"Submitted-paper checks: {len(checks)}/{len(checks)} passed.")


if __name__ == "__main__":
    main()
