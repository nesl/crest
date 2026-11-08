# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Audit exact-parameter repeats and saved final evaluations without training.

Use ``--group LABEL DATABASE TEST_METRICS_JSON`` repeatedly. Each database must
contain one study, or select it with ``--study LABEL STUDY_NAME``. Optional
``--evidence LABEL PATH`` files are hashed as supporting provenance; their
presence alone does not establish the executed checkpoint/fold policy.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np


def source(path):
    """Return a source identity without modifying the file.

    Parameters
    ----------
    path : path-like
        File to hash.

    Returns
    -------
    dict
        Absolute path and SHA256 digest.
    """
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read_trials(database, study_name=None, endpoint=None):
    """Read COMPLETE feasible single-objective observations via SQLite read-only.

    Parameters
    ----------
    database : path-like
        Explicit Optuna database.
    study_name : str, optional
        Required for multiple studies.
    endpoint : int, optional
        First N feasible completions; None retains all.

    Returns
    -------
    tuple
        Study name and completion-ordered observation dictionaries.
    """
    if endpoint is not None and endpoint < 1:
        raise ValueError("endpoint must be positive")
    uri = Path(database).resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        studies = connection.execute(
            "SELECT study_id, study_name FROM studies"
        ).fetchall()
        matches = [s for s in studies if study_name is None or s[1] == study_name]
        if len(matches) != 1:
            raise ValueError("Select exactly one database study explicitly")
        study_id, study_name = matches[0]
        rows = []
        for tid, number, completed in connection.execute(
            "SELECT trial_id, number, datetime_complete FROM trials WHERE study_id=? AND state='COMPLETE'",
            (study_id,),
        ):
            attrs = {
                k: json.loads(v)
                for k, v in connection.execute(
                    "SELECT key, value_json FROM trial_user_attributes WHERE trial_id=?",
                    (tid,),
                )
            }
            if attrs.get("feasible") is not True:
                continue
            values = connection.execute(
                "SELECT objective,value FROM trial_values WHERE trial_id=?", (tid,)
            ).fetchall()
            if len(values) != 1 or values[0][0] != 0 or not np.isfinite(values[0][1]):
                raise ValueError("Expected one finite objective per included trial")
            # Exact parameter identity uses only these known integer schemas;
            # do not broaden it to the search-history reader's Int* rule.
            params = {}
            for name, value, distribution in connection.execute(
                "SELECT param_name,param_value,distribution_json FROM trial_params WHERE trial_id=?",
                (tid,),
            ):
                spec = json.loads(distribution)
                if spec["name"] == "CategoricalDistribution":
                    value = spec["attributes"]["choices"][int(value)]
                elif spec["name"] in (
                    "IntDistribution",
                    "IntUniformDistribution",
                    "IntLogUniformDistribution",
                ):
                    value = int(value)
                params[name] = value
            if not params:
                raise ValueError("Included trial has no parameter identity")
            f1 = float(attrs["metric__macro_f1"])
            if not np.isfinite(f1):
                raise ValueError("Validation macro-F1 must be finite")
            if completed is None:
                raise ValueError("Feasible trial lacks completion time")
            rows.append(
                dict(
                    trial=number,
                    completed=completed,
                    params=params,
                    attrs=attrs,
                    score=float(values[0][1]),
                    validation_macro_f1=f1,
                )
            )
    if not rows:
        raise ValueError("No COMPLETE feasible observations")
    rows.sort(key=lambda r: (datetime.fromisoformat(r["completed"]), r["trial"]))
    if endpoint is not None:
        if len(rows) < endpoint:
            raise ValueError(
                "Requested endpoint exceeds available feasible observations"
            )
        rows = rows[:endpoint]
    return study_name, rows


def repeated_variability(rows):
    """Pool within exact-parameter validation-F1 residuals with sum(n-1) df.

    Parameters
    ----------
    rows : list of dict
        Observations with parameters and validation macro-F1.

    Returns
    -------
    dict
        Repeated groups, residual sums, degrees of freedom and pooled SD.
    """
    groups = defaultdict(list)
    for row in rows:
        groups[json.dumps(row["params"], sort_keys=True, separators=(",", ":"))].append(
            row
        )
    repeated = []
    for identity, observations in sorted(groups.items()):
        if len(observations) < 2:
            continue
        values = np.asarray(
            [r["validation_macro_f1"] for r in observations], dtype=float
        )
        residual_ss = float(np.sum((values - np.mean(values)) ** 2))
        repeated.append(
            dict(
                parameters=json.loads(identity),
                trials=[r["trial"] for r in observations],
                observations=len(values),
                mean_validation_macro_f1=float(np.mean(values)),
                residual_sum_squares=residual_ss,
                degrees_of_freedom=len(values) - 1,
            )
        )
    df = sum(g["degrees_of_freedom"] for g in repeated)
    ss = sum(g["residual_sum_squares"] for g in repeated)
    return dict(
        identity="exact decoded Optuna parameter dictionary; not resolved-network equivalence",
        repeated_configurations=len(repeated),
        repeated_observations=sum(g["observations"] for g in repeated),
        residual_sum_squares=ss,
        degrees_of_freedom=df,
        pooled_within_configuration_sd=float(np.sqrt(ss / df)) if df else None,
        groups=repeated,
    )


def final_evaluation(study_name, rows, metrics):
    """Associate saved final metrics with the maximum individual search score.

    Parameters
    ----------
    study_name : str
        Selected study identity.
    rows : list of dict
        Eligible search observations.
    metrics : dict
        Saved final test metrics and hyperparameters.

    Returns
    -------
    dict
        Selection association, saved test metrics and provenance limits.
    """
    for key in ("macro_f1", "accuracy"):
        value = metrics.get(key)
        if value is None and key == "accuracy":
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
        ):
            raise ValueError(f"Saved final {key} must be finite")
    winner = min(rows, key=lambda r: (-r["score"], r["trial"]))
    hp = metrics.get("hyperparameters", {})
    mismatches = {
        k: {"final": v, "selected": winner["params"].get(k)}
        for k, v in hp.items()
        if k not in winner["params"] or winner["params"][k] != v
    }
    missing = sorted(set(winner["params"]) - set(hp))
    study_matches = metrics.get("study_name") == study_name
    associated = bool(hp) and study_matches and not mismatches and not missing
    result = dict(
        selection_policy="maximum saved individual objective; lowest trial number breaks ties",
        selected_trial=winner["trial"],
        selected_score=winner["score"],
        selected_validation_macro_f1=winner["validation_macro_f1"],
        study_name_matches=study_matches,
        configuration_matches=associated,
        parameter_mismatches=mismatches,
        missing_final_parameters=missing,
        test_macro_f1=metrics.get("macro_f1"),
        accuracy=metrics.get("accuracy"),
        evaluation_backend=metrics.get("evaluation_backend"),
        quantization_mode=metrics.get("quantization_mode"),
        checkpoint_path_recorded=metrics.get("checkpoint_path"),
        tflite_path_recorded=metrics.get("tflite_path"),
        fold_policy="Not established by saved metric schema unless explicitly recorded",
        checkpoint_policy="Not established by checkpoint filename or metric values",
        explicitly_recorded_policy={
            k: v
            for k, v in metrics.items()
            if "fold" in k.lower()
            or "policy" in k.lower()
            or k in ("monitor", "restore_best_weights")
        },
    )
    return result


def audit_group(
    label, database, metrics_path, study_name=None, evidence=(), endpoint=None
):
    """Produce reproducible source hashes, repeat statistics, and final association.

    Parameters
    ----------
    label : str
        User-selected group label.
    database, metrics_path : path-like
        Explicit database and saved final metrics inputs.
    study_name : str, optional
        Explicit study selector.
    evidence : iterable of path-like, optional
        Additional sources to hash.
    endpoint : int, optional
        First N feasible completions; None retains all.

    Returns
    -------
    dict
        Group audit with source identities and analysis results.
    """
    study, rows = read_trials(database, study_name, endpoint)
    metrics = json.loads(Path(metrics_path).read_text())
    return dict(
        group=label,
        study_name=study,
        observations=len(rows),
        endpoint=endpoint,
        observation_order="completion timestamp then trial number; no endpoint filter by default",
        sources=[source(database), source(metrics_path)],
        supporting_evidence=[source(p) for p in evidence],
        repeated_variability=repeated_variability(rows),
        final_evaluation=final_evaluation(study, rows, metrics),
        limits=[
            "Single selected-model test evaluation; no independent campaign replication.",
            "Within-configuration variation does not isolate training randomness from other changes.",
            "Saved artifact association does not prove checkpoint execution or physical deployment.",
        ],
    )


def main(argv=None):
    """Write audits for explicitly supplied groups.

    Parameters
    ----------
    argv : list of str, optional
        CLI arguments; None uses process arguments.

    Returns
    -------
    list of dict
        Group reports also saved to evaluation.json.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        nargs=3,
        action="append",
        required=True,
        metavar=("LABEL", "DATABASE", "METRICS"),
    )
    parser.add_argument(
        "--study", nargs=2, action="append", default=[], metavar=("LABEL", "STUDY")
    )
    parser.add_argument(
        "--evidence", nargs=2, action="append", default=[], metavar=("LABEL", "PATH")
    )
    parser.add_argument("--out", "--output-dir", type=Path, required=True)
    parser.add_argument(
        "--endpoint",
        type=int,
        default=None,
        help="First N feasible observations in completion order; default: all",
    )
    args = parser.parse_args(argv)
    labels = [g[0] for g in args.group]
    if len(set(labels)) != len(labels):
        parser.error("Group labels must be unique")
    if any(k not in labels for k, _ in args.study + args.evidence):
        parser.error("Study/evidence labels must name an input group")
    studies = dict(args.study)
    reports = [
        audit_group(
            label,
            db,
            metrics,
            studies.get(label),
            [p for k, p in args.evidence if k == label],
            args.endpoint,
        )
        for label, db, metrics in args.group
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "evaluation.json").write_text(
        json.dumps({"schema_version": 1, "groups": reports}, indent=2, allow_nan=False)
        + "\n"
    )
    return reports


if __name__ == "__main__":
    main()
