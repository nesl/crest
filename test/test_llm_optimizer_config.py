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
from crest.optimizers.llm.component import LLMGeneratorOptimizer  # noqa: E402


class OptimizerConfigTests(unittest.TestCase):
    """Validate the opt-in boundary and compact LLM defaults."""

    def test_missing_optimizer_defaults_to_existing_optuna_path(self) -> None:
        """Legacy configurations retain Optuna behavior without edits."""
        for config in (Dict(), Dict(optimizer=None)):
            with self.subTest(config=config):
                optimizer = _normalize_optimizer_config(config)
                self.assertEqual(optimizer.type, "optuna")
                self.assertEqual(optimizer.recover_orphaned_trials, [])

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
        self.assertEqual(optimizer.llm.anchor_count, 5)
        self.assertTrue(optimizer.llm.semantic_context)

    pass

    def test_anchor_count_accepts_disable_and_rejects_nonnegative_integer_violations(self) -> None:
        """Best-model evidence can be omitted or bounded explicitly."""
        for count in [0, 1, 9]:
            with self.subTest(count=count):
                optimizer = _normalize_optimizer_config(Dict(optimizer=Dict(
                    type="llm_generator", llm=Dict(provider="fake", responses=[{}], anchor_count=count)
                )))
                self.assertEqual(optimizer.llm.anchor_count, count)
        for count in [-1, True, False, 1.0, "5", None]:
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "anchor_count"):
                _normalize_optimizer_config(Dict(optimizer=Dict(
                    type="llm_generator", llm=Dict(provider="fake", responses=[{}], anchor_count=count)
                )))

    def test_semantic_context_can_be_disabled_for_ablation(self) -> None:
        """The semantic ablation flag is retained as a validated boolean."""
        optimizer = _normalize_optimizer_config(
            Dict(
                optimizer=Dict(
                    type="llm_generator",
                    llm=Dict(provider="fake", responses=[{}], semantic_context=False),
                )
            )
        )

        self.assertFalse(optimizer.llm.semantic_context)

    pass

    pass

    pass

    pass


if __name__ == "__main__":
    unittest.main()
