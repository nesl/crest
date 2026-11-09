# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Compatibility checks for the public interface package."""

import importlib
import subprocess
import sys
from types import SimpleNamespace

import pytest

from crest import interfaces


@pytest.mark.parametrize(
    ("module", "names"),
    [
        ("dataset", ("DatasetABC",)),
        ("task", ("TaskABC",)),
        ("model_family", ("ModelFamilyABC",)),
        ("optimizer", ("OptimizerABC", "OptimizerBudgetExhausted")),
    ],
)
def test_package_reexports_original_class_objects(module, names):
    direct = importlib.import_module(f"crest.interfaces.{module}")
    for name in names:
        assert getattr(interfaces, name) is getattr(direct, name)


@pytest.mark.parametrize(
    ("name", "required"),
    [
        ("DatasetABC", {"load"}),
        (
            "TaskABC",
            {"build_target_spec", "metric_contract", "compile_model", "build_fit_plan",
             "evaluate_predictions", "evaluate"},
        ),
        ("ModelFamilyABC", {"sample_hparams", "build_model"}),
        ("OptimizerABC", {"propose_round"}),
    ],
)
def test_abstract_requirements_are_preserved(name, required):
    base = getattr(interfaces, name)
    assert base.__abstractmethods__ == required
    with pytest.raises(TypeError):
        base()


def _implementation(base):
    return type("Implementation", (base,), {
        method: lambda *args, **kwargs: None for method in base.__abstractmethods__
    })()


def test_default_dataset_and_task_hooks():
    dataset = _implementation(interfaces.DatasetABC)
    task = _implementation(interfaces.TaskABC)
    calibration = object()
    assert dataset.name == task.name == "Implementation"
    assert dataset.validate_config(object()) is None
    assert dataset.make_calibration_data(SimpleNamespace(calibration=calibration), {}) is calibration
    assert dataset.make_calibration_data(SimpleNamespace(calibration=None), {}) is None
    assert task.validate_config(object()) is None
    assert task.validate_model_outputs(object(), object()) is None
    assert task.history_component_keys(object()) == []
    assert task.generate_closeout_artifacts(None, None, None, None, output_dir=None) == {}


def test_default_model_family_and_optimizer_hooks():
    family = _implementation(interfaces.ModelFamilyABC)
    optimizer = _implementation(interfaces.OptimizerABC)
    assert family.name == optimizer.name == "Implementation"
    assert family.validate_config({}) is None
    assert family.validate_hparams({}, None, {}) is None
    raw = {"width": 8}
    decoded = family.decode_trial_hparams(raw, None, {})
    assert decoded == raw and decoded is not raw
    assert family.default_seed_trial(None, {}) is None
    assert family.custom_objects() == {}
    assert family.supports_tflite() is True
    with pytest.raises(NotImplementedError):
        family.trial_search_space(None, {})
    assert optimizer.requires_search_space is True
    assert optimizer.requires_semantic_context({}) is True
    assert optimizer.validate_config({}) is None
    assert optimizer.identity_config({}) == {}
    assert optimizer.normalize_stored_identity(raw, {}) is raw
    assert optimizer.initialize(None, {}) is None
    assert issubclass(interfaces.OptimizerBudgetExhausted, RuntimeError)


def test_importing_interfaces_does_not_load_ml_runtimes():
    result = subprocess.run(
        [sys.executable, "-c", """
import sys
import crest.interfaces
import crest.interfaces.dataset
import crest.interfaces.task
import crest.interfaces.model_family
import crest.interfaces.optimizer
for name in ('tensorflow', 'keras', 'torch', 'optuna'):
    assert not any(key == name or key.startswith(name + '.') for key in sys.modules), name
"""],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
