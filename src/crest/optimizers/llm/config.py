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

    # Omitted limits preserve existing campaigns and their optimizer identity.
    output_limit = llm.get("max_output_tokens")
    if output_limit is not None:
        if type(output_limit) is not int or output_limit < 1:
            raise ValueError("optimizer.llm.max_output_tokens must be a positive integer.")
        output_parameter = llm.get("output_token_parameter", "max_completion_tokens")
        if not isinstance(output_parameter, str) or output_parameter not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError("optimizer.llm.output_token_parameter must be max_tokens or max_completion_tokens.")
        llm.output_token_parameter = output_parameter
    elif llm.get("output_token_parameter") is not None:
        raise ValueError("optimizer.llm.output_token_parameter requires max_output_tokens.")

    max_usd = llm.get("max_usd")
    if max_usd is not None:
        if provider == "fake":
            raise ValueError("optimizer.llm.max_usd requires a paid provider with token usage accounting.")
        if output_limit is None:
            raise ValueError("optimizer.llm.max_usd requires max_output_tokens.")
        for name in ("max_usd", "input_usd_per_million_tokens", "output_usd_per_million_tokens"):
            value = llm.get(name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0 or (name != "max_usd" and value == 0)):
                raise ValueError(f"optimizer.llm.{name} must be finite and nonnegative (token prices must be positive).")
        if type(llm.get("max_input_tokens")) is not int or llm.max_input_tokens < 1:
            raise ValueError("optimizer.llm.max_input_tokens must be a positive integer with max_usd.")
        if not isinstance(llm.get("pricing_source"), str) or not llm.pricing_source.strip():
            raise ValueError("optimizer.llm.pricing_source must document the configured token prices.")

    warmup_file = llm.get("warmup_params_file")
    if warmup_file is not None and (not isinstance(warmup_file, str) or not warmup_file.strip()):
        raise ValueError("optimizer.llm.warmup_params_file must be a non-empty path or null.")

    require_rationale = llm.get("require_rationale", False)
    if not isinstance(require_rationale, bool):
        raise ValueError("optimizer.llm.require_rationale must be a boolean.")
    llm.require_rationale = require_rationale

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
        raw_retries = llm.get("max_transport_retries", 5)
        if type(raw_retries) is not int or raw_retries < 0:
            raise ValueError("optimizer.llm.max_transport_retries must be an integer >= 0.")
        llm.max_transport_retries = raw_retries
        for field_name, allowed in (
            ("reasoning_effort", {"none", "minimal", "low", "medium", "high", "xhigh", "max"}),
            ("service_tier", {"auto", "default", "flex", "scale", "priority", "fast"}),
        ):
            value = llm.get(field_name)
            if value is not None and (not isinstance(value, str) or value not in allowed):
                raise ValueError(f"optimizer.llm.{field_name} must be one of {sorted(allowed)} or null.")
        for field_name, default, minimum in (
            ("retry_backoff_s", 5.0, 0.0),
            ("retry_backoff_cap_s", 60.0, 0.0),
            ("temperature", 0.4, 0.0),
            ("timeout_s", 60.0, 0.001),
        ):
            raw_value = llm.get(field_name, default)
            # Reasoning models may require temperature to be absent from requests.
            if field_name == "temperature" and raw_value is None:
                llm[field_name] = None
                continue
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
