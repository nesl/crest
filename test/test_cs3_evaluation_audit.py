"""Synthetic checks for repeat pooling and final-model association."""

import importlib.util
import json
from pathlib import Path
import sqlite3

import pytest

SPEC = importlib.util.spec_from_file_location(
    "cs3_evaluation",
    Path(__file__).resolve().parents[1] / "analysis_scripts/cs3_audit/evaluation.py",
)
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def test_pooled_residual_degrees_of_freedom_and_exact_identity():
    rows = [
        dict(trial=i, params={"x": x}, validation_macro_f1=y)
        for i, (x, y) in enumerate(
            [("a", 0.2), ("a", 0.4), ("b", 0.4), ("b", 0.8), ("b", 0.6), ("c", 1.0)]
        )
    ]
    result = evaluation.repeated_variability(rows)
    assert result["repeated_configurations"] == 2
    assert result["repeated_observations"] == 5
    assert result["degrees_of_freedom"] == 3
    assert result["pooled_within_configuration_sd"] == pytest.approx((0.10 / 3) ** 0.5)
    assert (
        evaluation.repeated_variability(rows[-1:])["pooled_within_configuration_sd"]
        is None
    )


def database(path):
    with sqlite3.connect(path) as c:
        c.executescript(
            """
        CREATE TABLE studies (study_id INTEGER, study_name TEXT);
        CREATE TABLE trials (trial_id INTEGER, study_id INTEGER, number INTEGER, state TEXT, datetime_complete TEXT);
        CREATE TABLE trial_values (trial_id INTEGER, objective INTEGER, value REAL);
        CREATE TABLE trial_params (trial_id INTEGER, param_name TEXT, param_value REAL, distribution_json TEXT);
        CREATE TABLE trial_user_attributes (trial_id INTEGER, key TEXT, value_json TEXT);
        INSERT INTO studies VALUES (1,'example');
        """
        )
        distribution = json.dumps(
            {"name": "CategoricalDistribution", "attributes": {"choices": ["a", "b"]}}
        )
        for i, (state, feasible, value, f1, choice) in enumerate(
            [
                ("COMPLETE", True, 0.7, 0.8, 0),
                ("COMPLETE", True, 0.6, 0.95, 1),
                ("COMPLETE", False, 0.99, 0.99, 1),
                ("PRUNED", True, 1.0, 1.0, 1),
            ]
        ):
            c.execute(
                "INSERT INTO trials VALUES (?,1,?,?,?)",
                (i, i, state, f"2026-01-01T00:00:0{3-i}"),
            )
            c.execute("INSERT INTO trial_values VALUES (?,0,?)", (i, value))
            c.execute(
                "INSERT INTO trial_params VALUES (?,'mode',?,?)",
                (i, choice, distribution),
            )
            for key, val in {"feasible": feasible, "metric__macro_f1": f1}.items():
                c.execute(
                    "INSERT INTO trial_user_attributes VALUES (?,?,?)",
                    (i, key, json.dumps(val)),
                )


def test_sqlite_selection_is_individual_score_and_not_best_f1(tmp_path):
    db = tmp_path / "runs.db"
    database(db)
    before = db.read_bytes()
    study, rows = evaluation.read_trials(db)
    assert [r["trial"] for r in rows] == [1, 0]
    assert rows[1]["params"] == {"mode": "a"}
    metrics = dict(
        study_name=study,
        hyperparameters={"mode": "a"},
        macro_f1=0.51,
        evaluation_backend="tflite",
        quantization_mode="int8_ptq",
    )
    result = evaluation.final_evaluation(study, rows, metrics)
    assert result["selected_trial"] == 0
    assert result["configuration_matches"]
    assert result["test_macro_f1"] == 0.51
    assert "Not established" in result["checkpoint_policy"]
    metrics["hyperparameters"] = {"mode": "b"}
    assert not evaluation.final_evaluation(study, rows, metrics)[
        "configuration_matches"
    ]
    metrics["hyperparameters"] = {}
    assert not evaluation.final_evaluation(study, rows, metrics)[
        "configuration_matches"
    ]
    assert db.read_bytes() == before


def test_multiple_studies_require_explicit_selection(tmp_path):
    db = tmp_path / "runs.db"
    database(db)
    with sqlite3.connect(db) as c:
        c.execute("INSERT INTO studies VALUES (2,'other')")
    with pytest.raises(ValueError, match="exactly one"):
        evaluation.read_trials(db)
    assert evaluation.read_trials(db, "example")[0] == "example"


def test_endpoint_uses_completion_order_and_rejects_invalid_metrics(tmp_path):
    db = tmp_path / "runs.db"
    database(db)
    study, rows = evaluation.read_trials(db, endpoint=1)
    assert [row["trial"] for row in rows] == [1]
    metrics = dict(study_name=study, hyperparameters={"mode": "a"}, macro_f1=0.5)
    assert not evaluation.final_evaluation(study, rows, metrics)[
        "configuration_matches"
    ]
    for value in (float("nan"), float("inf"), None, True):
        metrics["macro_f1"] = value
        with pytest.raises(ValueError, match="must be finite"):
            evaluation.final_evaluation(study, rows, metrics)
    with pytest.raises(ValueError, match="exceeds"):
        evaluation.read_trials(db, endpoint=3)
    with pytest.raises(ValueError, match="positive"):
        evaluation.read_trials(db, endpoint=0)
