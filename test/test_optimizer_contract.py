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
from crest.search_space import SearchParam, SearchSpaceDescriptor, candidate_error, validate_candidate
from crest.optimizers.llm.schemas import CandidateBatch, validate_candidate_batch

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

    def test_builtin_registration_does_not_import_datasets_or_build_provider(self):
        script = """
import importlib.abc
import sys
class BlockDatasets(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith('crest.datasets'):
            raise RuntimeError('dataset imported: ' + fullname)
sys.meta_path.insert(0, BlockDatasets())
from unittest.mock import patch
from crest.builtin_components import ensure_optimizer_components_registered
with patch('crest.optimizers.llm.provider.build_provider', side_effect=RuntimeError('provider called')):
    ensure_optimizer_components_registered()
    ensure_optimizer_components_registered()
"""
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_third_plugin_registered_before_load_config_keeps_custom_fields(self):
        name = "MixedCaseContractPlugin"
        optimizer_registry.register(name, ThirdOptimizer)
        self.addCleanup(optimizer_registry._items.pop, name)
        config = yaml.safe_load((ROOT / "src/config/nas_config_flops_rmse.yaml").read_text())
        config["optimizer"] = {"type": name, "width": 7, "custom_payload": {"seed": 3}}
        with tempfile.TemporaryDirectory() as temp:
            config["outputs"]["models_dir"] = str(Path(temp) / "models")
            config["outputs"]["candidate_dir"] = str(Path(temp) / "candidate")
            path = Path(temp) / "config.yaml"
            path.write_text(yaml.safe_dump(config))
            loaded = load_config(path)
        self.assertEqual(loaded.optimizer.type, name)
        self.assertEqual(loaded.optimizer.width, 7)
        self.assertEqual(loaded.optimizer.custom_payload.seed, 3)
        self.assertNotIn("llm", loaded.optimizer)
        optimizer = optimizer_registry.get(name)()
        optimizer.initialize(None, loaded.optimizer)
        self.assertEqual(optimizer.identity_config(loaded.optimizer), {"width": 7})
        self.assertEqual(optimizer.propose_round((), BudgetSnapshot(0, 0, 1, 1, 1)).candidates[0].params, {"width": 7})
        with self.assertRaisesRegex(ValueError, "width"):
            _normalize_optimizer_config(Dict(optimizer={"type": name, "width": "7"}))

    def test_builtin_case_aliases_and_exact_registration_precedence(self):
        for name in ("optuna", "llm_generator"):
            with self.subTest(name=name):
                resolved_name, component_cls = resolve_optimizer_selection(" " + name.upper() + " ")
                self.assertEqual(resolved_name, name)
                self.assertIs(component_cls, optimizer_registry.get(name))
                exact_name = name.upper()
                optimizer_registry.register(exact_name, ThirdOptimizer)
                try:
                    config = _normalize_optimizer_config(Dict(optimizer={"type": exact_name, "width": 7}))
                    self.assertEqual(config.type, exact_name)
                    self.assertNotIn("llm", config)
                    self.assertEqual(resolve_optimizer_selection(exact_name), (exact_name, ThirdOptimizer))
                finally:
                    optimizer_registry._items.pop(exact_name)

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

    def test_shared_legality_matches_llm_rejection_without_duplicate_policy(self):
        descriptor = SearchSpaceDescriptor((SearchParam("width", "int", low=2, high=8),))
        for candidate in ({}, {"width": True}, {"width": 10}, {"width": 3, "extra": 0}):
            with self.subTest(candidate=candidate):
                shared = candidate_error(candidate, descriptor)
                llm = validate_candidate_batch(CandidateBatch(candidates=[candidate]), descriptor, batch_size=1)
                self.assertEqual(shared, (llm.rejected[0].code, llm.rejected[0].message))
                with self.assertRaises(ValueError):
                    validate_candidate(candidate, descriptor)
        validate_candidate({"width": 3}, descriptor)
        validate_candidate({"width": 3}, descriptor)
        llm = validate_candidate_batch(CandidateBatch(candidates=[{"width": 3}, {"width": 3}]), descriptor, batch_size=2)
        self.assertEqual(llm.rejected[0].code, "duplicate_batch")

    def test_neutral_and_compatibility_search_declarations_are_identical(self):
        from crest.optimizers.llm.search_space import SearchParam as CompatibilityParam
        self.assertIs(CompatibilityParam, SearchParam)
        with self.assertRaisesRegex(ValueError, "finite bounds"):
            SearchParam("width", "float", low=0.0, high=float("inf"))
        with self.assertRaisesRegex(ValueError, "SearchParam"):
            SearchSpaceDescriptor(("invalid",))


    def test_missing_family_descriptor_differs_from_invalid_declaration(self):
        from types import SimpleNamespace
        from crest.search_space import build_search_space_descriptor
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


    def test_provider_endpoint_rejects_embedded_credentials_or_routing_suffixes(self):
        for endpoint in (
            "https://user:password@example.test/v1",
            "https://example.test/v1?api_key=secret",
            "https://example.test/v1#fragment",
        ):
            with self.subTest(endpoint=endpoint), self.assertRaisesRegex(ValueError, "clean endpoint"):
                _normalize_optimizer_config(Dict(optimizer={
                    "type": "llm_generator", "llm": {
                        "provider": "openai_compatible", "model": "example-model",
                        "api_key_env": "EXAMPLE_API_KEY", "base_url": endpoint,
                    },
                }))


if __name__ == "__main__":
    unittest.main()
