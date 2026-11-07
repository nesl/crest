# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Public building blocks for CREST's LLM candidate proposer.

Re-export the provider protocol, prompt inputs, validation records, ledger, and
proposal helpers for callers that need to exercise generation independently.
The runner-facing optimizer lives in ``component``; importing this package does
not register a separate execution loop or evaluate proposed architectures."""

from .ledger import LLMLedger
from .history import build_best_trial_anchors, build_recent_trial_history
from .phase_policy import PhaseState, resolve_phase
from .prompt_builder import PromptContext, build_candidate_request
from .provider import (
    FakeProvider,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    build_provider,
)
from ..search_space import SearchParam, SearchSpaceDescriptor, build_search_space_descriptor
from ..context import SEMANTIC_CONTEXT_VERSION, build_semantic_context
from .schemas import (
    CandidateBatch,
    CandidateRejection,
    CandidateValidationResult,
    validate_candidate_batch,
)

__all__ = [
    "CandidateBatch",
    "CandidateRejection",
    "CandidateValidationResult",
    "FakeProvider",
    "LLMLedger",
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "PhaseState",
    "PromptContext",
    "SearchParam",
    "SearchSpaceDescriptor",
    "SEMANTIC_CONTEXT_VERSION",
    "build_search_space_descriptor",
    "build_candidate_request",
    "build_provider",
    "build_best_trial_anchors",
    "build_recent_trial_history",
    "build_semantic_context",
    "resolve_phase",
    "generate_llm_batch",
    "sample_random_candidate",
    "validate_candidate_batch",
]
from .enqueue import generate_llm_batch, sample_random_candidate
