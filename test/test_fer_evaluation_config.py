# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Static contract checks for the resolution-neutral FER/HCI fragment."""

from pathlib import Path

import yaml


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


def test_fer_esp32_evaluation_fragment_matches_reviewed_hci_policy() -> None:
    """The fragment keeps the latency ceiling solely in constrained feasibility."""
    config = _load("fer_esp32_evaluation.yaml")
    device = config["device"]
    esp32 = device["esp32"]
    training = config["training"]
    nas = config["nas"]
    network = config["network"]

    assert device["name"] == "FREENOVE_ESP32_S3_WROOM"
    assert device["serial_port"].startswith("/dev/serial/by-id/")
    assert device["runtime_mode"] == "back_to_back"
    assert device["measured_inference_runs"] == 10
    assert device["serial_timeout_s"] == 1200.0
    assert device["dut_ready_timeout_s"] == 30.0
    assert esp32["arena_sizes_kb"] == [2048]
    assert esp32["warmup_runs"] == 1
    assert esp32["latency_aggregation"] == "median"
    assert esp32["build_timeout_s"] == 900.0
    assert esp32["upload_timeout_s"] == 120.0
    assert training["energy_aware"] is False
    assert training["quantization"] == {
        "mode": "int8_ptq",
        "search": False,
        "choices": ["int8_ptq"],
    }
    assert nas["hil_first"] is False
    assert nas["prune"]["rules"] == []
    assert nas["score"]["type"] == "multi-objective"
    assert nas["score"]["params"]["objectives"] == [
        {"metric": "macro_f1", "direction": "maximize"},
        {"metric": "latency_ms", "direction": "minimize"},
    ]
    assert nas["feasibility"]["rules"] == [
        {
            "rule": "classifier_latency_ceiling",
            "metric": "latency_ms",
            "condition": "gt",
            "reference": {"type": "literal", "value": 3000.0},
            "reason": "Classifier inference exceeds the provisional 3000 ms ceiling",
        }
    ]
    assert "latency_budget_ms" not in device
    assert network["recv_timeout_sec"] == 2400
    assert network["send_timeout_sec"] == 30
    assert network["hil_probe_timeout_sec"] == 5.0
