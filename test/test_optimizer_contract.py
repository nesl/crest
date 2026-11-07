# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Foundational proposal contracts and plugin configuration integration."""

from dataclasses import FrozenInstanceError
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest

from addict import Dict
import yaml

from crest.builtin_components import ensure_optimizer_components_registered
from crest.component_selection import resolve_optimizer_selection
from crest.interfaces import OptimizerABC
from crest.model import _normalize_optimizer_config, load_config
from crest.optimizers.optuna import OptunaOptimizer
from crest.pipeline_types import BudgetSnapshot, CandidateProposal, ExplicitRound, NativeRound
from crest.registry import ComponentRegistry, optimizer_registry
from crest.optimizers.search_space import SearchParam, SearchSpaceDescriptor, candidate_error, validate_candidate

ROOT = Path(__file__).resolve().parents[1]


class ThirdOptimizer(OptimizerABC):
    """A configurable proposer needs no Optuna APIs or runner changes."""

    def validate_config(self, config):
        if type(config.get("width")) is not int:
            raise ValueError("optimizer.width must be an integer")

    def identity_config(self, config):
        return {"width": config.width}

    def initialize(self, context, config):
        self.width = config.width

    def propose_round(self, history, budget):
        return ExplicitRound((CandidateProposal({"width": self.width}),))


class OptimizerContractTests(unittest.TestCase):
    def test_registry_unknown_duplicate_and_idempotent_builtins(self):
        registry = ComponentRegistry("optimizer")
        registry.register("third", ThirdOptimizer)
        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register("third", ThirdOptimizer)
        with self.assertRaisesRegex(KeyError, "Unknown optimizer"):
            registry.get("missing")
        with self.assertRaisesRegex(ValueError, "non-empty"):
            registry.register(" ", ThirdOptimizer)
        ensure_optimizer_components_registered()
        before = dict(optimizer_registry._items)
        ensure_optimizer_components_registered()
        self.assertEqual(before, optimizer_registry._items)
        self.assertIs(optimizer_registry.get("optuna"), OptunaOptimizer)

    def test_native_contract_imports_without_execution_dependencies(self):
        script = """
import importlib.abc
import sys
class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in {'optuna', 'tensorflow'}:
            raise RuntimeError('execution dependency imported: ' + fullname)
sys.meta_path.insert(0, BlockRuntime())
from crest.optimizers.optuna import OptunaOptimizer
from crest.pipeline_types import BudgetSnapshot
assert OptunaOptimizer().propose_round((), BudgetSnapshot(0, 0, 4, 8, 4)).n_trials == 4
"""
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    pass

    pass

    pass

    def test_custom_registration_case_is_not_aliased(self):
        name = "CaseSensitivePlugin"
        optimizer_registry.register(name, ThirdOptimizer)
        self.addCleanup(optimizer_registry._items.pop, name)
        with self.assertRaisesRegex(ValueError, "Unknown optimizer"):
            _normalize_optimizer_config(Dict(optimizer={"type": name.lower(), "width": 7}))

    def test_payloads_are_passive_frozen_and_explicit_count_is_derived(self):
        native = NativeRound(3)
        with self.assertRaises(FrozenInstanceError):
            native.n_trials = 4
        explicit = ExplicitRound((CandidateProposal({"width": 3}), CandidateProposal({"width": 3})))
        self.assertEqual(explicit.n_trials, 2)
        with self.assertRaises(TypeError):
            ExplicitRound(explicit.candidates, n_trials=8)
        # Invalid round counts remain transport data until runner validation.
        self.assertEqual(NativeRound(0).n_trials, 0)

    pass

    pass


    def test_missing_family_descriptor_differs_from_invalid_declaration(self):
        from types import SimpleNamespace
        from crest.optimizers.search_space import build_search_space_descriptor
        with self.assertRaises(NotImplementedError):
            build_search_space_descriptor(SimpleNamespace(name="legacy"), None, {}, {}, collect_compile_metrics=False)
        with self.assertRaises(ValueError):
            build_search_space_descriptor(SimpleNamespace(trial_search_space="invalid"), None, {}, {}, collect_compile_metrics=False)
        with self.assertRaises(ValueError):
            SearchParam("choice", "categorical", choices=({"invalid": True},))
        choices = ["first", "second"]
        parameter = SearchParam("choice", "categorical", choices=choices)
        choices.append("third")
        self.assertEqual(parameter.choices, ("first", "second"))


    pass


if __name__ == "__main__":
    unittest.main()
