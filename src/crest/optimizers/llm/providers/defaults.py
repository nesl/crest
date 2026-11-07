# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Shared defaults for configuration normalization and provider construction."""


def default_json_response_mode(provider_name: str) -> bool:
    """Enable structured JSON responses for providers with a known JSON API."""
    return provider_name in {"openrouter", "openai_responses"}
