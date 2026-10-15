"""Check prose against configured score arithmetic and constraint boundaries."""
import json
from pathlib import Path

import pytest

from crest.optimizers.llm.objective_renderer import render_objective


def fixture():
    """Load the saved campaign scoring inputs.

    Returns
    -------
    tuple
        Scoring definition and original semantic context.
    """
    saved = json.loads((Path(__file__).parent / "fixtures/cs3_legacy_request_000088.json").read_text())
    context = json.loads(saved["messages"][1]["content"])
    return json.loads(context["objective_summary"]), context


def test_score_reference_is_computed_from_context():
    # Changing the budget must change the equation and marginal tradeoff.
    config, context = fixture()
    context["runtime_context"]["latency_budget_ms"] = 2000
    text = render_objective(config, context)
    assert "energy_mj_per_inference / 400" in text
    assert "40 mJ decrease" in text
    assert "0.01 unit increase in macro_f1" in text
    context["runtime_context"]["latency_budget_ms"] = 1000
    assert "energy_mj_per_inference / 200" in render_objective(config, context)


@pytest.mark.parametrize("condition,operator", [("gt", "≤"), ("gte", "<"), ("lt", "≥"), ("lte", ">")])
def test_constraint_complements_failure_condition(condition, operator):
    # Inclusive rejection must produce an exclusive permitted limit and vice versa.
    config, context = fixture()
    context["task_context"]["feasibility"]["rules"] = [{"condition": condition,
        "metric": "latency_ms", "reference": {"type": "literal", "value": 25}}]
    assert f"latency_ms {operator} 25 ms" in render_objective(config, context)


def test_unknown_objective_falls_back_without_guessing():
    # A nonlinear term or missing normalization budget must not get invented prose.
    config, context = fixture()
    config["params"]["terms"][0]["type"] = "target"
    assert render_objective(config, context) is None
    config, context = fixture()
    context["runtime_context"].pop("latency_budget_ms")
    assert render_objective(config, context) is None


@pytest.mark.parametrize("metric,weight,reference,equation,change", [
    ("accuracy", 2, None, "2 × accuracy", "0.005 unit increase in accuracy"),
    ("loss", -1, None, "− loss", "0.01 unit decrease in loss"),
    ("latency_ms", -0.5, None, "− 0.5 × latency_ms", "0.02 ms decrease in latency_ms"),
    ("latency_ms", 0.5, 100, "0.5 × (latency_ms / 100)", "2 ms increase in latency_ms"),
    ("ram_bytes", -2, 1024, "− 2 × (ram_bytes / 1024)", "5.12 bytes decrease in ram_bytes"),
    ("custom_quality", 0.25, None, "0.25 × custom_quality", "0.04 unit increase in custom_quality"),
])
def test_metric_weights_and_normalization(metric, weight, reference, equation, change):
    # Expected tradeoffs are independently calculated, including an unusual reward for latency.
    config, context = fixture()
    term = {"type": "weighted", "metric": metric, "weight": weight}
    if reference is not None:
        term.update(type="normalized-weighted", reference={"type": "literal", "value": reference})
    config["params"]["terms"] = [term]
    text = render_objective(config, context)
    assert f"Score = {equation}." in text
    assert f"{change} raises score by 0.01." in text
    assert "Energy units" not in text


def test_mixed_terms_keep_signs_and_separate_tradeoffs():
    # A multi-metric objective must preserve the complete sum and each marginal effect.
    config, context = fixture()
    config["params"]["terms"] = [
        {"type": "weighted", "metric": "accuracy", "weight": 2},
        {"type": "normalized-weighted", "metric": "latency_ms", "weight": -0.2,
         "reference": {"type": "literal", "value": 100}},
        {"type": "weighted", "metric": "loss", "weight": -0.5},
    ]
    text = render_objective(config, context)
    assert "Score = 2 × accuracy − 0.2 × (latency_ms / 100) − 0.5 × loss." in text
    for change in ("0.005 unit increase in accuracy", "5 ms decrease in latency_ms", "0.02 unit decrease in loss"):
        assert change in text


@pytest.mark.parametrize("second_weight,expected", [(-1, "0.01 unit increase in accuracy"), (-3, "0.01 unit decrease in accuracy"), (-2, None)])
def test_repeated_metric_uses_net_effect(second_weight, expected):
    # Multiple terms for one metric can reinforce, reverse, or cancel its effect.
    config, context = fixture()
    config["params"]["terms"] = [
        {"type": "weighted", "metric": "accuracy", "weight": 2},
        {"type": "normalized-weighted", "metric": "accuracy", "weight": second_weight * 10,
         "reference": {"type": "literal", "value": 10}},
    ]
    text = render_objective(config, context)
    if expected:
        assert expected in text
        assert text.count("raises score by 0.01") == 1
    else:
        assert "raises score" not in text


def test_zero_weight_has_no_improvement_claim():
    # A disabled term remains visible but cannot promise a score improvement.
    config, context = fixture()
    config["params"]["terms"] = [{"type": "weighted", "metric": "accuracy", "weight": 0}]
    text = render_objective(config, context)
    assert "Score = 0 × accuracy." in text
    assert "raises score" not in text


def test_unknown_constraint_keeps_symbolic_limit():
    # Unavailable configuration values must not become guessed numerical limits.
    config, context = fixture()
    context["task_context"]["feasibility"]["rules"] = [{"condition": "gt", "metric": "latency_ms",
        "reference": {"type": "metric", "metric": "custom_limit"}}]
    assert "latency_ms ≤ custom_limit (unresolved here) ms" in render_objective(config, context)


@pytest.mark.parametrize("reference", [0, -1, float("nan"), float("inf")])
def test_invalid_normalization_falls_back(reference):
    # Invalid denominators must never produce authoritative-looking prose.
    config, context = fixture()
    config["params"]["terms"] = [{"type": "normalized-weighted", "metric": "accuracy", "weight": 1,
        "reference": {"type": "literal", "value": reference}}]
    assert render_objective(config, context) is None
