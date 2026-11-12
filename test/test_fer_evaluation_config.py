# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Static contract checks for the resolution-neutral FER/HCI fragment."""

from pathlib import Path

from addict import Dict
import yaml

from crest.component_selection import (
    RUNTIME_TRAINING_BATCH_SIZE_KEY,
    resolve_component_selection,
    resolve_training_batch_size,
)
from crest.model import evaluate_feasibility_rules, evaluate_score_config


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "src" / "config"


def _load(name: str) -> dict:
    """Load one repository YAML fragment."""
    return yaml.safe_load((CONFIG_ROOT / name).read_text(encoding="utf-8"))


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge mapping keys in the documented composition order."""
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def test_fer_esp32_evaluation_fragment_is_resolution_and_budget_neutral() -> None:
    """Resolution, campaign budgets, outputs, and duplicate batch knobs stay external."""
    config = _load("fer_esp32_evaluation.yaml")

    assert "dataset" not in config
    assert "task" not in config
    assert "model" not in config
    assert "outputs" not in config
    assert "batch_size" not in config["training"]
    for budget_key in (
        "nas_epochs",
        "model_epochs",
        "nas_trials",
        "nas_multiobjective_population_size",
        "max_total_trials",
    ):
        assert budget_key not in config["training"]
    for resolution in (64, 96):
        component = _load(f"fer_components_{resolution}.yaml")
        assert component["training"] == {"batch_size": 64}
        composed = _deep_merge(component, config)
        assert composed["dataset"]["params"]["image_size"] == resolution
        assert composed["training"]["batch_size"] == 64
        assert composed["training"]["quantization"]["mode"] == "int8_ptq"


def test_composed_fer_policy_resolves_batch_and_preserves_boundary_objectives() -> None:
    """Composition retains batch 64 and treats only latency above 3000 as infeasible."""
    component = _load("fer_components_64.yaml")
    policy = _load("fer_esp32_evaluation.yaml")
    config = Dict(_deep_merge(component, policy))

    selection = resolve_component_selection(config)
    assert resolve_training_batch_size(config.training) == 64
    assert selection["task_config"][RUNTIME_TRAINING_BATCH_SIZE_KEY] == 64
    assert config.device.serial_port.startswith("/dev/serial/by-id/")
    assert config.nas.prune.rules == []
    assert "latency_budget_ms" not in config.device

    for latency_ms, expected_feasible, expected_constraint in (
        (2999.0, True, -1.0),
        (3000.0, True, 0.0),
        (3001.0, False, 1.0),
    ):
        metrics = {"macro_f1": 0.72, "latency_ms": latency_ms}
        feasibility = evaluate_feasibility_rules(
            metrics=metrics,
            hyperparams=Dict(flops=1),
            score_config=config.nas.score,
            feasibility_config=config.nas.feasibility,
            task_nonnegative_metric_names={"macro_f1"},
        )
        score = evaluate_score_config(
            metrics=metrics,
            hyperparams=Dict(flops=1),
            score_config=config.nas.score,
            task_nonnegative_metric_names={"macro_f1"},
        )

        assert feasibility.feasible is expected_feasible
        assert feasibility.constraints == [expected_constraint]
        assert score.objective_names == ["macro_f1", "latency_ms"]
        assert score.objective_values == [0.72, latency_ms]
        assert score.objective_directions == ["maximize", "minimize"]
