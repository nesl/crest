"""Proposal setup builds only the context requested by its component."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from crest.interfaces import ModelFamilyABC
from crest.optimizer_history import freeze_plain_data
from crest.optimizers.llm.component import LLMGeneratorOptimizer
from crest.optimizers.optuna import OptunaOptimizer
from crest.pipeline_types import BudgetSnapshot
from crest.search_space import SearchParam
from crest.semantic_context import build_semantic_context, disabled_semantic_context
from test.proposal_plugin import ThirdProposer
from test.test_nas_model_client import _build_test_client


class LegacyFamily(ModelFamilyABC):
    def sample_hparams(self, trial, ctx, config):
        return {"width": trial.suggest_int("width", 2, 8)}

    def build_model(self, hparams, ctx, config):
        raise AssertionError("setup must not build a model")


class BrokenDescriptorFamily(LegacyFamily):
    def trial_search_space(self, ctx, config):
        raise NotImplementedError("internal descriptor failure")


@pytest.fixture
def client(tmp_path):
    client = _build_test_client(tmp_path)
    client.config.training.train = False
    client.config.device.hil = False
    client.config.device.compile_when_hil_disabled = "false"
    client.model_family.trial_search_space = MagicMock(
        return_value=[SearchParam("width", "int", low=2, high=8)]
    )
    return client


def llm_config(enabled=True):
    return {"type": "llm_generator", "llm": {
        "provider": "fake", "semantic_context": enabled,
        "responses": [{"candidates": [{"width": 3}]}, {"candidates": [{"width": 4}]}],
        "memory": {"enabled": False},
    }}


@pytest.mark.parametrize("family", [LegacyFamily(), BrokenDescriptorFamily(), SimpleNamespace()])
def test_native_setup_ignores_absent_and_broken_descriptor_hooks(client, family):
    client.model_family = family
    optimizer = OptunaOptimizer()
    optimizer.initialize = MagicMock()
    client._classify_nas_metric_dependencies = MagicMock(side_effect=AssertionError("unused dependencies"))
    with patch("nas_model_client.build_search_space_descriptor", side_effect=ValueError("bad descriptor")) as descriptor, \
         patch("nas_model_client.build_semantic_context", side_effect=ValueError("bad semantics")) as semantics:
        assert client._initialize_optimizer(optimizer, {"type": "optuna"}) is None
    descriptor.assert_not_called()
    semantics.assert_not_called()
    client._classify_nas_metric_dependencies.assert_not_called()
    context = optimizer.initialize.call_args.args[0]
    assert context.search_space is None
    assert context.semantic_context == {}


@pytest.mark.parametrize("family", [LegacyFamily(), SimpleNamespace()])
@pytest.mark.parametrize("optimizer", [LLMGeneratorOptimizer(), ThirdProposer()])
def test_explicit_setup_reports_absent_hook_before_provider_or_evaluation(client, family, optimizer):
    client.model_family = family
    optimizer.initialize = MagicMock()
    client._hil_request = MagicMock()
    client.objective = MagicMock()
    with patch("crest.optimizers.llm.component.build_provider") as provider, \
         patch("nas_model_client.build_search_space_descriptor") as descriptor, \
         pytest.raises(ValueError, match="requires a declared trial_search_space descriptor"):
        client._initialize_optimizer(optimizer, llm_config())
    provider.assert_not_called()
    descriptor.assert_not_called()
    optimizer.initialize.assert_not_called()
    client._hil_request.assert_not_called()
    client.objective.assert_not_called()


@pytest.mark.parametrize("optimizer", [LLMGeneratorOptimizer(), ThirdProposer()])
def test_overridden_descriptor_not_implemented_is_an_internal_error(client, optimizer):
    client.model_family = BrokenDescriptorFamily()
    optimizer.initialize = MagicMock()
    with patch("crest.optimizers.llm.component.build_provider") as provider, \
         pytest.raises(NotImplementedError, match="internal descriptor failure"):
        client._initialize_optimizer(optimizer, llm_config())
    optimizer.initialize.assert_not_called()
    provider.assert_not_called()


def test_llm_disabled_semantics_skip_builder_and_keep_exact_disabled_prompt(client):
    optimizer = LLMGeneratorOptimizer()
    config = llm_config(False)
    assert optimizer.requires_semantic_context(config) is False
    with patch("nas_model_client.build_semantic_context", side_effect=ValueError("unused semantics")) as builder:
        descriptor = client._initialize_optimizer(optimizer, config)
        optimizer.propose_round((), BudgetSnapshot(0, 0, 2, 2, 1))
    builder.assert_not_called()
    assert descriptor is optimizer.context.search_space
    assert optimizer.semantic_context == disabled_semantic_context()
    payload = json.loads(optimizer.provider.requests[0].user_prompt)
    assert payload["semantic_context"] == {
        key: value for key, value in disabled_semantic_context().items() if key != "payload"
    }
    assert all(key not in payload for key in ("dataset_context", "task_context", "model_context", "device_context", "runtime_context"))


def test_llm_enabled_semantics_build_once_and_preserve_payload_and_hash(client):
    optimizer = LLMGeneratorOptimizer()
    config = llm_config()
    default_config = llm_config()
    del default_config["llm"]["semantic_context"]
    assert optimizer.requires_semantic_context(default_config) is True
    with patch("nas_model_client.build_semantic_context", wraps=build_semantic_context) as builder:
        client._initialize_optimizer(optimizer, config)
        optimizer.propose_round((), BudgetSnapshot(0, 0, 2, 2, 1))
        optimizer.propose_round((), BudgetSnapshot(1, 1, 2, 2, 1))
    builder.assert_called_once()
    expected = build_semantic_context(**builder.call_args.kwargs)
    assert optimizer.context.semantic_context == freeze_plain_data(expected)
    assert optimizer.semantic_context == expected
    assert optimizer.context.objective_summary == json.dumps(client.config.nas.score, sort_keys=True, default=str)
    for request in optimizer.provider.requests:
        payload = json.loads(request.user_prompt)
        assert payload["semantic_context"] == {
            "enabled": True, "version": expected["version"], "hash": expected["hash"],
            "guidance": expected["payload"]["guidance"],
        }
        for field in ("dataset_context", "task_context", "model_context", "device_context", "runtime_context"):
            assert payload.get(field) == expected["payload"].get(field)


def test_third_proposer_retains_descriptor_and_semantics_by_default(client):
    optimizer = ThirdProposer()
    optimizer.initialize = MagicMock()
    assert optimizer.requires_search_space is True
    assert optimizer.requires_semantic_context({"type": "third"}) is True
    with patch("nas_model_client.build_semantic_context", wraps=build_semantic_context) as builder:
        descriptor = client._initialize_optimizer(optimizer, {"type": "third", "setting": 7})
    builder.assert_called_once()
    context = optimizer.initialize.call_args.args[0]
    assert context.search_space is descriptor
    assert context.semantic_context["enabled"] is True
    assert all(value is not client and value is not client.dataset_bundle for value in vars(context).values())


def test_instance_descriptor_override_is_respected_on_legacy_family(client):
    client.model_family = LegacyFamily()
    client.model_family.trial_search_space = MagicMock(
        return_value=[SearchParam("width", "int", low=2, high=8)]
    )
    optimizer = LLMGeneratorOptimizer()
    client._initialize_optimizer(optimizer, llm_config(False))
    client.model_family.trial_search_space.assert_called_once()
    assert tuple(optimizer.context.search_space) == ("width",)
