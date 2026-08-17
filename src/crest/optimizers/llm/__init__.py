# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""LLM candidate-generator foundations."""

from .ledger import LLMLedger
from .phase_policy import PhaseState, resolve_phase
from .prompt_builder import PromptContext, build_candidate_request
from .provider import FakeProvider, LLMProvider, LLMRequest, LLMResponse, build_provider
from .search_space import SearchParam, SearchSpaceDescriptor, build_search_space_descriptor
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
    "build_search_space_descriptor",
    "build_candidate_request",
    "build_provider",
    "resolve_phase",
    "enqueue_llm_batch",
    "sample_random_candidate",
    "validate_candidate_batch",
]
from .enqueue import enqueue_llm_batch, sample_random_candidate
