# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""LLM proposal preparation behind CREST's shared optimizer contract."""
from __future__ import annotations

import random
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from ...interfaces import OptimizerABC
from ...pipeline_types import BudgetSnapshot, ExplicitRound, SearchContext, TrialRecord
from ...semantic_context import SEMANTIC_CONTEXT_VERSION, disabled_semantic_context
from .config import normalize_llm_config
from .enqueue import generate_llm_batch
from .history import build_best_trial_anchors, build_recent_trial_history
from .ledger import LLMLedger
from .memory import ExperimentalMemory, MemoryConfig, digest
from .prompt_builder import PromptContext
from .provider import build_provider


def _plain(value):
    """Detach immutable context mappings for JSON prompt serialization."""
    from collections.abc import Mapping
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


class LLMGeneratorOptimizer(OptimizerABC):
    """Own prompts, provider requests, derived memory, repair and fallback."""

    @property
    def name(self) -> str:
        return "llm_generator"

    def validate_config(self, config) -> None:
        normalize_llm_config(config)

    def requires_semantic_context(self, config) -> bool:
        return normalize_llm_config(config)["llm"]["semantic_context"]

    def identity_config(self, config) -> dict:
        llm = normalize_llm_config(config)["llm"]
        identity = {key: _plain(llm[key]) for key in (
            "provider", "prompt_version", "batch_size", "max_repair_attempts",
            "random_seed", "recent_trial_window", "anchor_count", "semantic_context", "memory",
        )}
        identity.update({
            "model": llm.get("model", "fake-model"),
            "candidate_schema_version": 1,
            "semantic_context_version": SEMANTIC_CONTEXT_VERSION,
            "memory_schema_version": 1,
            "memory_prompt_version": "memory-v1",
            "phase_policy_version": 1,
            "fallback_policy": {"type": "unique_random", "max_attempts": 100},
        })
        if llm["provider"] == "fake":
            # Only configured responses affect this provider, never credentials.
            identity["fixture_responses_hash"] = digest(llm["responses"])
        else:
            endpoint = urlsplit(llm["base_url"])
            host = endpoint.hostname or ""
            if endpoint.port is not None:
                host += f":{endpoint.port}"
            identity.update({key: llm[key] for key in (
                "temperature", "timeout_s", "json_response_mode",
            )})
            # Credentials and optional auth headers are not study identity.
            identity["base_url"] = urlunsplit((endpoint.scheme, host, endpoint.path, "", ""))
        return identity

    def initialize(self, context: SearchContext, config) -> None:
        llm = normalize_llm_config(config)["llm"]
        if context.search_space is None:
            raise ValueError("llm_generator requires a declared trial_search_space descriptor.")
        self.context = context
        self.config = llm
        self.provider = build_provider(llm)
        self.ledger = LLMLedger(Path(context.artifact_dir) / "llm_optimizer")
        self.rng = random.Random(llm["random_seed"])
        memory_config = MemoryConfig.from_config(llm["memory"])
        self.memory = (ExperimentalMemory(self.ledger.root / "memory", memory_config)
                       if memory_config.enabled else None)
        self.semantic_context = _plain(context.semantic_context)
        if not llm["semantic_context"]:
            self.semantic_context = disabled_semantic_context()

    def propose_round(self, history: tuple[TrialRecord, ...],
                      budget: BudgetSnapshot) -> ExplicitRound:
        if budget.permitted_round_size < 1:
            raise ValueError("LLM proposal requires a positive permitted round size.")
        context = self.context
        llm = self.config
        batch_size = min(budget.permitted_round_size, llm["batch_size"])
        prompt = PromptContext(
            study_name=context.study_name,
            model_family=context.model_family_name,
            descriptor=context.search_space,
            objective_summary=context.objective_summary,
            attempted_trials=budget.attempted_count,
            feasible_completed_trials=budget.completed_target_count,
            target_feasible_trials=budget.target,
            max_total_attempts=budget.total_cap,
            batch_size=batch_size,
            recent_trials=build_recent_trial_history(
                history, window_size=llm["recent_trial_window"],
                directions=context.objective_directions,
            ),
            anchors=build_best_trial_anchors(
                history, anchor_count=llm["anchor_count"],
                feasibility_enabled=context.feasibility_enabled,
                directions=context.objective_directions,
            ),
            semantic_context=self.semantic_context,
        )
        if self.memory is not None:
            prompt = self.memory.enrich(history, prompt, self.provider,
                                        directions=context.objective_directions)
        return ExplicitRound(generate_llm_batch(
            history, self.provider, prompt, self.ledger,
            prompt_version=llm["prompt_version"],
            max_repair_attempts=llm["max_repair_attempts"], rng=self.rng,
        ))
