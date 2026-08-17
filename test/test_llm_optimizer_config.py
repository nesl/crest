# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for optimizer selection normalization."""

import sys
import unittest
from pathlib import Path

from addict import Dict

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crest.model import _normalize_optimizer_config  # noqa: E402


class OptimizerConfigTests(unittest.TestCase):
    """Validate the opt-in boundary and compact LLM defaults."""

    def test_missing_optimizer_defaults_to_existing_optuna_path(self) -> None:
        """Legacy configurations retain Optuna behavior without edits."""
        optimizer = _normalize_optimizer_config(Dict())

        self.assertEqual(optimizer.type, "optuna")

    def test_llm_generator_config_is_normalized(self) -> None:
        """The fake provider path receives deterministic MVP defaults."""
        optimizer = _normalize_optimizer_config(
            Dict(
                optimizer=Dict(
                    type="llm_generator",
                    llm=Dict(
                        provider="fake",
                        responses=[{"candidates": [{"width": 4}]}],
                    ),
                )
            )
        )

        self.assertEqual(optimizer.type, "llm_generator")
        self.assertEqual(optimizer.llm.batch_size, 5)
        self.assertEqual(optimizer.llm.max_repair_attempts, 1)
        self.assertEqual(optimizer.llm.prompt_version, "v1")
        self.assertEqual(optimizer.llm.random_seed, 0)
        self.assertEqual(optimizer.llm.recent_trial_window, 10)

    def test_invalid_optimizer_configs_are_rejected(self) -> None:
        """Malformed selection and fake-provider shapes fail during config load."""
        invalid = [
            Dict(optimizer="llm_generator"),
            Dict(optimizer=Dict(type="unknown")),
            Dict(optimizer=Dict(type="llm_generator", llm=Dict())),
            Dict(optimizer=Dict(type="llm_generator", llm=Dict(provider="fake", responses=[]))),
            Dict(
                optimizer=Dict(
                    type="llm_generator",
                    llm=Dict(provider="fake", responses=[{}], batch_size=0),
                )
            ),
            Dict(
                optimizer=Dict(
                    type="llm_generator",
                    llm=Dict(provider="fake", responses=[{}], recent_trial_window=-1),
                )
            ),
        ]
        for config in invalid:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    _normalize_optimizer_config(config)

    def test_openrouter_defaults_and_live_fields_are_normalized(self) -> None:
        """OpenRouter receives its standard URL and validated provider settings."""
        optimizer = _normalize_optimizer_config(
            Dict(
                optimizer=Dict(
                    type="llm_generator",
                    llm=Dict(
                        provider="openrouter",
                        model="openai/test-model",
                        temperature=0.2,
                        timeout_s=15,
                        extra_headers=Dict(**{"X-Title": "CREST"}),
                    ),
                )
            )
        )

        self.assertEqual(optimizer.llm.base_url, "https://openrouter.ai/api/v1")
        self.assertEqual(optimizer.llm.api_key_env, "OPENROUTER_API_KEY")
        self.assertEqual(optimizer.llm.temperature, 0.2)
        self.assertEqual(optimizer.llm.timeout_s, 15.0)
        self.assertEqual(optimizer.llm.extra_headers["X-Title"], "CREST")

    def test_openai_compatible_requires_explicit_api_key_environment_variable(self) -> None:
        """Generic endpoints must not silently inherit OpenRouter credentials."""
        base_llm = Dict(
            provider="openai_compatible",
            base_url="https://llm.example/v1",
            model="example-model",
        )
        with self.assertRaisesRegex(ValueError, "require it explicitly"):
            _normalize_optimizer_config(
                Dict(optimizer=Dict(type="llm_generator", llm=base_llm))
            )

        base_llm.api_key_env = "GENERIC_LLM_API_KEY"
        optimizer = _normalize_optimizer_config(
            Dict(optimizer=Dict(type="llm_generator", llm=base_llm))
        )

        self.assertEqual(optimizer.llm.api_key_env, "GENERIC_LLM_API_KEY")
        self.assertEqual(optimizer.llm.base_url, "https://llm.example/v1")


if __name__ == "__main__":
    unittest.main()
