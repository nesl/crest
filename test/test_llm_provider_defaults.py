# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Keep normalized and directly constructed provider defaults consistent."""

import pytest

from crest.optimizers.llm.config import normalize_llm_config
from crest.optimizers.llm.providers.provider import build_provider


@pytest.mark.parametrize(
    ("provider_name", "expected"),
    [("openrouter", True), ("openai_responses", True), ("openai_compatible", False)],
)
@pytest.mark.parametrize("override", [None, True, False])
def test_json_mode_matches_for_normalized_and_direct_config(provider_name, expected, override):
    """Both entry points preserve provider defaults and explicit boolean overrides."""
    config = {
        "provider": provider_name,
        "model": "offline-model",
        "base_url": "https://example.invalid/v1",
        "api_key_env": "CREST_TEST_DEFAULTS_KEY",
    }
    if override is not None:
        config["json_response_mode"] = override
        expected = override

    normalized = normalize_llm_config({"llm": config}).llm

    assert normalized.json_response_mode is expected
    assert build_provider(config).json_response_mode is expected
    assert build_provider(normalized).json_response_mode is expected
