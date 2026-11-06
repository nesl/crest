# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Audit scored Optuna campaigns from explicit, read-only archive paths.

Only terminal trials through the requested feasible completion are counted. A
measured violation requires COMPLETE, explicit infeasibility, and positive finite
energy and latency. Failed deployments and invalid instrumentation remain
unsuccessful evaluations, irrespective of a stored feasibility label.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis_scripts.cs3_audit.campaign import read_trials


def positive(value):
    """Return whether a measurement is finite and strictly positive.

    Parameters
    ----------
    value : object
        Measurement to validate.

    Returns
    -------
    bool
        Whether the measurement is positive and finite.
    """
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def audit(config, endpoint=200):
    """Classify terminal trial outcomes without exposing raw attributes.

    Parameters
    ----------
    config : dict
        Campaigns with unique name, explicit database, optional study_id and
        attribute mapping. Global mapping may also override attribute names.
    endpoint : int, optional
        Recorded feasible COMPLETE target; defaults to 200. Contradictory
        measurement evidence affects outcome classification, not prefix selection.

    Returns
    -------
    dict
        Campaign summaries, selected trial evidence, totals and source hashes.
    """
    if isinstance(endpoint, bool) or not isinstance(endpoint, int) or endpoint < 1:
        raise ValueError("endpoint must be a positive integer")
    names = [c["name"] for c in config["campaigns"]]
    if not names or any(not n for n in names) or len(names) != len(set(names)):
        raise ValueError("Campaign names must be nonempty and unique")
    rows, summaries, sources = [], [], []
    for campaign in config["campaigns"]:
        database = Path(campaign["database"]).resolve()
        before = hashlib.sha256(database.read_bytes()).hexdigest()
        mapping = {**config.get("mapping", {}), **campaign.get("mapping", {})}
        key = lambda name: mapping.get(name, name)
        trials = read_trials(database, mapping, campaign.get("study_id"))
        feasible = 0
        selected = []
        with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
            for trial in trials:
                params = (trial["number"], campaign.get("study_id"))
                # read_trials already enforces a unique study when unspecified.
                query = "select trial_id from trials where number=?"
                if params[1] is not None:
                    query += " and study_id=?"
                tid = connection.execute(
                    query, params if params[1] is not None else params[:1]
                ).fetchone()[0]
                attrs = {
                    k: json.loads(v)
                    for k, v in connection.execute(
                        "select key,value_json from trial_user_attributes where trial_id=?",
                        (tid,),
                    )
                }
                instrument = attrs.get(key("measurement_instrumentation_valid"))
                status = attrs.get(key("feasibility_status"))
                energy = attrs.get(key("energy_mj_per_inference"))
                latency = attrs.get(
                    key("latency_ms"), attrs.get(key("latency_ms_per_inference"))
                )
                feasible += int(trial["feasible"])
                code = attrs.get(key("hil_error_code"))
                invalid_measurement = any(
                    v is not None and not positive(v) for v in (energy, latency)
                )
                failed_hil = code in (2, 3)
                if (
                    trial["feasible"]
                    and instrument is not False
                    and not failed_hil
                    and not invalid_measurement
                ):
                    outcome, category, evidence = (
                        "measured_passed",
                        "valid_feasible",
                        "COMPLETE with feasible status and finite score",
                    )
                elif instrument is False:
                    outcome, category, evidence = (
                        "not_deployed_or_incomplete",
                        "instrumentation_failure",
                        "Instrumentation explicitly invalid",
                    )
                elif (
                    trial["state"] == "COMPLETE"
                    and status == "infeasible"
                    and not failed_hil
                    and positive(energy)
                    and positive(latency)
                ):
                    outcome, category, evidence = (
                        "measured_violation",
                        "valid_infeasible",
                        "COMPLETE with explicit infeasibility and positive finite energy and latency",
                    )
                elif (
                    attrs.get(key("hil_error_code")) == 2
                    and attrs.get(key("prune_reason")) == "HIL arena exhausted"
                ):
                    outcome, category, evidence = (
                        "not_deployed_or_incomplete",
                        "build_flash_allocation_failure",
                        "HIL arena exhausted",
                    )
                else:
                    outcome, category, evidence = (
                        "not_deployed_or_incomplete",
                        "other_unknown",
                        "No positively measured constraint violation established",
                    )
                selected.append(
                    dict(
                        case_study=campaign.get(
                            "case_study", config.get("case_study", "CS3")
                        ),
                        campaign=campaign["name"],
                        trial=trial["number"],
                        state=trial["state"],
                        end=trial["end"],
                        outcome=outcome,
                        category=category,
                        evidence=evidence,
                        score=trial["score"],
                        energy_mj=energy,
                        latency_ms=latency,
                        instrumentation_valid=instrument,
                    )
                )
                if feasible >= endpoint:
                    break
        after = hashlib.sha256(database.read_bytes()).hexdigest()
        if before != after:
            raise ValueError("Archive changed during audit")
        rows.extend(selected)
        summaries.append(
            dict(
                campaign=campaign["name"],
                endpoint=endpoint,
                endpoint_reached=feasible >= endpoint,
                terminal_attempts=len(selected),
                **dict(Counter(r["outcome"] for r in selected)),
            )
        )
        sources.append(
            dict(campaign=campaign["name"], source_name=database.name, sha256=before)
        )
    return dict(
        campaigns=summaries,
        trials=rows,
        totals=dict(Counter(r["outcome"] for r in rows)),
        terminal_attempts=len(rows),
        sources=sources,
        qualifications=[
            "Terminal COMPLETE, FAIL and PRUNED records only; RUNNING and WAITING are excluded.",
            "Measurements and retries within one trial never add trial outcomes.",
            "Unknown evaluator failures are not inferred to be measured violations.",
            "Prefix selection uses recorded feasible COMPLETE trials with finite scores; contradictory measurement evidence may reduce measured passes within that prefix.",
        ],
    )


def main(argv=None):
    """Write audit JSON from explicit configuration paths.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--endpoint", default=200, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result = audit(json.loads(args.config.read_text()), args.endpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if result["trials"]:
        with (args.output_dir / "trial_outcomes.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=result["trials"][0])
            writer.writeheader()
            writer.writerows(result["trials"])
    (args.output_dir / "scored_outcomes.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
