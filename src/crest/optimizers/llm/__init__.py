# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""LLM candidate-generator foundations."""

from .search_space import SearchParam, SearchSpaceDescriptor, build_search_space_descriptor

__all__ = ["SearchParam", "SearchSpaceDescriptor", "build_search_space_descriptor"]
