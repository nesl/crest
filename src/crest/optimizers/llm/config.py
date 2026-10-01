# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Provider-local defaults and validation, shared by config loading and setup."""

import math
from urllib.parse import urlsplit

from addict import Dict


def normalize_llm_config(optimizer_config) -> Dict:
    """Normalize the optimizer block without resolving credentials or clients."""
    if not isinstance(optimizer_config, (dict, Dict)):
        raise ValueError("optimizer must be a mapping when provided.")
    optimizer = Dict(optimizer_config)
    raw_llm = optimizer.get("llm", None)
    if not isinstance(raw_llm, (dict, Dict)):
        raise ValueError("optimizer.llm must be a mapping for optimizer.type=llm_generator.")
    llm = Dict(raw_llm)
    provider = str(llm.get("provider", "")).strip().lower()
    if provider not in {"fake", "openrouter", "openai_compatible"}:
        raise ValueError(
            "optimizer.llm.provider must be one of: fake, openrouter, openai_compatible."
        )
    llm.provider = provider

    for field_name, default, minimum in (
        ("batch_size", 5, 1),
        ("max_repair_attempts", 1, 0),
        ("random_seed", 0, 0),
        ("recent_trial_window", 10, 0),
        ("anchor_count", 5, 0),
    ):
        raw_value = llm.get(field_name, default)
        if isinstance(raw_value, bool) or not isinstance(raw_value, int) or raw_value < minimum:
            raise ValueError(f"optimizer.llm.{field_name} must be an integer >= {minimum}.")
        llm[field_name] = raw_value

    raw_semantic_context = llm.get("semantic_context", True)
    if not isinstance(raw_semantic_context, bool):
        raise ValueError("optimizer.llm.semantic_context must be a boolean.")
    llm.semantic_context = raw_semantic_context
    from .memory import MemoryConfig
    from dataclasses import asdict
    raw_memory = llm.get("memory", {})
    if not isinstance(raw_memory, (dict, Dict)):
        raise ValueError("optimizer.llm.memory must be a mapping")
    llm.memory = Dict(asdict(MemoryConfig.from_config(raw_memory)))

    prompt_version = str(llm.get("prompt_version", "v1")).strip()
    if not prompt_version:
        raise ValueError("optimizer.llm.prompt_version must be a non-empty string.")
    llm.prompt_version = prompt_version
    if provider == "fake":
        responses = llm.get("responses", None)
        if not isinstance(responses, list) or not responses:
            raise ValueError("optimizer.llm.responses must be a non-empty list for provider=fake.")
    else:
        default_base_url = "https://openrouter.ai/api/v1" if provider == "openrouter" else None
        base_url = llm.get("base_url", default_base_url)
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("optimizer.llm.base_url must be a non-empty URL string.")
        llm.base_url = base_url.strip().rstrip("/")
        endpoint = urlsplit(llm.base_url)
        if endpoint.username is not None or endpoint.password is not None or "?" in llm.base_url or "#" in llm.base_url:
            raise ValueError(
                "optimizer.llm.base_url must be a clean endpoint without userinfo, "
                "query, or fragment; configure credentials through api_key_env "
                "or extra_headers."
            )
        model = llm.get("model", None)
        if not isinstance(model, str) or not model.strip():
            raise ValueError("optimizer.llm.model must be a non-empty string.")
        llm.model = model.strip()
        default_api_key_env = "OPENROUTER_API_KEY" if provider == "openrouter" else None
        api_key_env = llm.get("api_key_env", default_api_key_env)
        if not isinstance(api_key_env, str) or not api_key_env.strip():
            raise ValueError(
                "optimizer.llm.api_key_env must be a non-empty string; "
                "generic openai_compatible providers require it explicitly."
            )
        llm.api_key_env = api_key_env.strip()
        raw_json_response_mode = llm.get(
            "json_response_mode",
            provider == "openrouter",
        )
        if not isinstance(raw_json_response_mode, bool):
            raise ValueError("optimizer.llm.json_response_mode must be a boolean.")
        llm.json_response_mode = raw_json_response_mode
        for field_name, default, minimum in (
            ("temperature", 0.4, 0.0),
            ("timeout_s", 60.0, 0.001),
        ):
            raw_value = llm.get(field_name, default)
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise ValueError(f"optimizer.llm.{field_name} must be a number >= {minimum}.")
            normalized_value = float(raw_value)
            if not math.isfinite(normalized_value) or normalized_value < minimum:
                raise ValueError(f"optimizer.llm.{field_name} must be a number >= {minimum}.")
            llm[field_name] = normalized_value
        extra_headers = llm.get("extra_headers", {})
        if not isinstance(extra_headers, (dict, Dict)) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in extra_headers.items()
        ):
            raise ValueError("optimizer.llm.extra_headers must map strings to strings.")
        llm.extra_headers = Dict(extra_headers)
    optimizer.llm = llm
    return optimizer
