# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Proposal setup builds only the context requested by its component."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from crest.interfaces import ModelFamilyABC
from crest.optimizers.history import freeze_plain_data
from crest.optimizers.optuna import OptunaOptimizer
from crest.pipeline_types import BudgetSnapshot
from crest.optimizers.search_space import SearchParam
from crest.optimizers.context import build_semantic_context, disabled_semantic_context
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



