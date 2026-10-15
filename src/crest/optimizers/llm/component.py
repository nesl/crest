# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""LLM proposal preparation behind CREST's shared optimizer contract."""
from __future__ import annotations

import os
import json
import random
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from ...interfaces import OptimizerABC
from ...optimizer_history import to_plain as _plain
from ...pipeline_types import BudgetSnapshot, CandidateProposal, ExplicitRound, SearchContext, TrialRecord
from ...search_space import validate_candidate
from ...semantic_context import SEMANTIC_CONTEXT_VERSION, disabled_semantic_context
from .config import normalize_llm_config
from .enqueue import generate_llm_batch
from .history import build_best_trial_anchors, build_recent_trial_history
from .ledger import LLMLedger
from .memory import MEMORY_PROMPT_VERSION, ExperimentalMemory, MemoryConfig, digest
from .prompt_builder import PromptContext, enrich_full_history
from .provider import build_provider


def _load_warmup(llm):
    """Read ordered fixed candidates without invoking the provider.

    Parameters
    ----------
    llm : Mapping
        Normalized LLM configuration. ``load_config`` resolves paths relative
        to the YAML directory; direct callers supply their own resolved paths.

    Returns
    -------
    list of dict
        Raw candidate parameters, or an empty list when warm-up is disabled.

    Raises
    ------
    ValueError
        If the file is not a nonempty JSON list of parameter dictionaries.
    OSError
        If the configured file cannot be read.
    """
    filename = llm.get("warmup_params_file")
    if filename is None:
        return []
    candidates = json.loads(Path(filename).expanduser().read_text())
    if not isinstance(candidates, list) or not candidates or not all(isinstance(p, dict) for p in candidates):
        raise ValueError("warmup_params_file must contain a nonempty JSON list of parameter dictionaries.")
    return candidates


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
        self._warmup_params = _load_warmup(llm)
        identity = {key: _plain(llm[key]) for key in (
            "provider", "prompt_version", "batch_size", "max_repair_attempts",
            "random_seed", "recent_trial_window", "anchor_count", "semantic_context", "memory",
        )}
        identity.update({
            "model": llm.get("model", "fake-model"),
            "candidate_schema_version": 1,
            "semantic_context_version": SEMANTIC_CONTEXT_VERSION,
            "memory_schema_version": 1,
            "memory_prompt_version": MEMORY_PROMPT_VERSION,
            "phase_policy_version": 2,
            "fallback_policy": {"type": "unique_random", "max_attempts": 100},
        })
        if llm["require_rationale"]:
            identity["require_rationale"] = True
            identity["candidate_schema_version"] = 2
        if llm.get("history_mode") == "full":
            identity.update(history_mode="full", candidate_schema_version=3,
                            phase_policy_version=3, decoded_fingerprint_version=1,
                            coverage_policy_version=2, coverage_target_per_cell=2,
                            feedback_policy_version=2,
                            max_refill_attempts=llm["max_refill_attempts"])
        if llm.get("max_output_tokens") is not None:
            identity["max_output_tokens"] = llm["max_output_tokens"]
            identity["output_token_parameter"] = llm["output_token_parameter"]
        if self._warmup_params:
            identity["warmup_params_hash"] = digest(self._warmup_params)
        if llm["provider"] == "fake":
            # Only configured responses affect this provider, never credentials.
            identity["fixture_responses_hash"] = digest(llm["responses"])
        else:
            endpoint = urlsplit(llm["base_url"])
            host = endpoint.hostname or ""
            if endpoint.port is not None:
                host += f":{endpoint.port}"
            identity.update({key: llm[key] for key in (
                "temperature", "json_response_mode",
            )})
            # Add opt-in request controls only when set, preserving old studies.
            for name in ("reasoning_effort", "service_tier"):
                if llm.get(name) is not None:
                    identity[name] = llm[name]
            # Credentials and optional auth headers are not study identity.
            identity["base_url"] = urlunsplit((endpoint.scheme, host, endpoint.path, "", ""))
        return identity

    def normalize_stored_identity(self, stored, active):
        """Alias only the built-in version-1 transport-timeout signature.

        Parameters
        ----------
        stored : Any
            Persisted complete optimizer signature.
        active : dict
            Current complete optimizer signature.

        Returns
        -------
        Any
            Signature with a recognized legacy timeout removed, otherwise the
            original value. All other keys and values remain strict.
        """
        old = stored.get("config") if isinstance(stored, dict) else None
        if (type(self) is LLMGeneratorOptimizer
                and isinstance(stored, dict)
                and stored.get("version") == 1
                and stored.get("optimizer") == "llm_generator"
                and isinstance(old, dict)
                and old.get("provider") in {"openrouter", "openai_compatible"}
                and set(old) == set(active["config"]) | {"timeout_s"}
                and isinstance(old["timeout_s"], (int, float))
                and not isinstance(old["timeout_s"], bool)
                and 0.001 <= old["timeout_s"] < float("inf")):
            return {**stored, "config": {key: value for key, value in old.items()
                                         if key != "timeout_s"}}
        return stored

    def initialize(self, context: SearchContext, config) -> None:
        llm = normalize_llm_config(config)["llm"]
        if context.search_space is None:
            raise ValueError("llm_generator requires a declared trial_search_space descriptor.")
        # Reuse the exact file contents hashed during identity checking.
        self.warmup_params = getattr(self, "_warmup_params", None)
        if self.warmup_params is None:
            self.warmup_params = _load_warmup(llm)
        for params in self.warmup_params:
            validate_candidate(params, context.search_space)
        self.warmup_hash = digest(self.warmup_params)
        if llm["provider"] != "fake" and not os.environ.get(llm["api_key_env"]):
            raise RuntimeError(
                f"Required API key environment variable '{llm['api_key_env']}' is not set."
            )
        self.context = context
        if llm.get("history_mode") == "full" and context.model_family_name != "audio_dscnn":
            raise ValueError("Full-history decoded policy currently supports audio_dscnn only.")
        self.config = llm
        self.ledger = LLMLedger(Path(context.artifact_dir) / "llm_optimizer")
        spend_tracker = None
        if llm.get("max_usd") is not None:
            from .spending import SpendTracker
            spend_tracker = SpendTracker(self.ledger.root, llm)
        elif (self.ledger.root / "spend_state.json").exists():
            raise ValueError("This campaign has spend accounting; retain max_usd when resuming.")
        self.provider = (build_provider(llm, spend_tracker=spend_tracker)
                         if spend_tracker is not None else build_provider(llm))
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
        if self.warmup_params:
            ordered = sorted(history, key=lambda trial: trial.number)
            for index, trial in enumerate(ordered[:len(self.warmup_params)]):
                provenance = trial.user_attrs.get("crest_proposal", {})
                if (provenance.get("proposal_source") != "llm_warmup"
                        or provenance.get("warmup_index") != index
                        or provenance.get("warmup_params_hash") != self.warmup_hash
                        or _plain(provenance.get("proposal_params")) != self.warmup_params[index]):
                    raise RuntimeError("Warm-up history does not match its fixed candidate sequence.")
                if trial.state not in {"COMPLETE", "PRUNED", "FAIL"}:
                    raise RuntimeError("Warm-up has unfinished trials; resolve them before proposing more work.")
            if len(ordered) < len(self.warmup_params):
                index = len(ordered)
                return ExplicitRound((CandidateProposal(dict(self.warmup_params[index]), {
                    "proposal_source": "llm_warmup", "warmup_index": index,
                    "warmup_params_hash": self.warmup_hash,
                }),))
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
            require_rationale=llm["require_rationale"],
        )
        if llm.get("history_mode") == "full":
            prompt = enrich_full_history(prompt, history)
            self.ledger.record_event({"event": "full_history_policy", "phase": prompt.phase().name,
                "attempted": budget.attempted_count, "allocation": prompt.allocation,
                "coverage_goal_complete": prompt.coverage["goal_complete"],
                "exploration_targets": list(prompt.exploration_targets)})
        elif self.memory is not None:
            prompt = self.memory.enrich(history, prompt, self.provider,
                                        directions=context.objective_directions)
        options = ({"max_refill_attempts": llm["max_refill_attempts"]}
                   if llm.get("history_mode") == "full" else {})
        return ExplicitRound(generate_llm_batch(
            history, self.provider, prompt, self.ledger,
            prompt_version=llm["prompt_version"],
            max_repair_attempts=llm["max_repair_attempts"], rng=self.rng, **options,
        ))
