# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Build deterministic, bounded semantic context for LLM candidate prompts."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .contexts.data_task import _dataset_context, _task_context
from .contexts.device import _device_context
from .contexts.runtime import _runtime_context
from .contexts.stm32 import _stm32_context
from .contexts.values import _cfg_get

from .search_space import SearchSpaceDescriptor

SEMANTIC_CONTEXT_VERSION = "v3"
SEMANTIC_CONTEXT_GUIDANCE = (
    "Parameter effects are architectural priors, not measured facts or guaranteed outcomes.",
    "Actual CREST and HIL observations take precedence over semantic priors.",
    "Candidates must still contain the exact raw Optuna parameter names and values.",
    "Registered RAM/flash capacities are not necessarily free tensor-arena or application budgets; use explicit feasibility rules and measured allocation failures.",
    "FLOPs and parameter counts are imperfect latency/energy proxies: memory traffic, placement, kernels, and clock also matter; measured evidence takes precedence.",
    "Greater model capacity does not guarantee greater accuracy, especially under short training budgets.",
)

def _parameter_semantics(descriptor: SearchSpaceDescriptor) -> dict[str, Any]:
    """Extract optional parameter meaning from the active declarations.

    Parameters
    ----------
    descriptor : SearchSpaceDescriptor
        Active declarations with optional descriptions, units, and typical effects.

    Returns
    -------
    dict
        Semantic metadata by raw name, excluding distribution bounds.
    """
    semantics: dict[str, Any] = {}
    for name in descriptor:
        param = descriptor[name]
        entry: dict[str, Any] = {}
        if param.description:
            entry["description"] = param.description
        if param.units:
            entry["units"] = param.units
        if param.typical_effects:
            entry["typical_effects"] = list(param.typical_effects)
        if entry:
            semantics[name] = entry
    return semantics


def _semantic_hash(payload: dict[str, Any]) -> str:
    """Hash the canonical normalized semantic payload.

    Parameters
    ----------
    payload : dict
        Finite JSON-serializable normalized semantic payload.

    Returns
    -------
    str
        SHA-256 hexadecimal digest of canonical JSON.

    Raises
    ------
    ValueError
        If payload contains nonfinite numbers.
    TypeError
        If payload contains a value that cannot be JSON encoded.
    """
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def disabled_semantic_context() -> dict[str, Any]:
    """Build a versioned disabled semantic-context record.

    Returns
    -------
    dict
        Disabled flag, current version, empty payload, and hash of that payload.
    """
    return {"enabled": False, "version": SEMANTIC_CONTEXT_VERSION,
            "hash": _semantic_hash({}), "payload": {}}


def build_semantic_context(
    *,
    enabled: bool,
    dataset_name: str,
    dataset_config: Any,
    dataset_bundle: Any,
    model_build_context: Any,
    task_name: str,
    target_spec: Any,
    metric_contract: Any,
    score_config: Any,
    feasibility_config: Any,
    model_family_name: str,
    descriptor: SearchSpaceDescriptor,
    device_config: Any,
    training_config: Any,
    collect_compile_metrics: bool,
) -> dict[str, Any]:
    """Build deterministic versioned semantics for candidate prompts and ledgers.

    Parameters
    ----------
    enabled : bool
        Whether to extract semantic facts; false produces an empty payload.
    dataset_name : str
        Registered dataset name used for built-in modality hints.
    dataset_config : Any
        Dataset configuration providing fallback scalar metadata.
    dataset_bundle : Any
        Loaded bundle providing input properties and metadata.
    model_build_context : Any
        Resolved input properties and preferred dataset metadata.
    task_name : str
        Registered task identifier.
    target_spec : Any
        Resolved task type, outputs, and target metadata.
    metric_contract : Any
        Available, primary, and training-only metric names.
    score_config : Any
        Scalar score terms or multi-objective output specification.
    feasibility_config : Any
        Explicit feasibility rules and training policy.
    model_family_name : str
        Registered model-family identifier.
    descriptor : SearchSpaceDescriptor
        Active raw parameter declarations and optional semantics.
    device_config : Any
        Resolved device, memory, HIL, and runtime options.
    training_config : Any
        Training, quantization, and optional deterministic seed settings.
    collect_compile_metrics : bool
        Runner-resolved compile metric policy.

    Returns
    -------
    dict
        Enabled flag, schema version, canonical payload hash, and whitelisted facts.

    Raises
    ------
    ValueError
        If device options or a payload value are invalid.
    TypeError
        If a payload value cannot be encoded as JSON.
    """
    if not enabled:
        payload: dict[str, Any] = {}
    else:
        payload = {
            "guidance": list(SEMANTIC_CONTEXT_GUIDANCE),
            "dataset_context": _dataset_context(
                dataset_name,
                dataset_config,
                dataset_bundle,
                model_build_context,
            ),
            "task_context": _task_context(
                task_name,
                target_spec,
                metric_contract,
                score_config,
                feasibility_config,
            ),
            "model_context": {"family": model_family_name},
            "device_context": _device_context(
                device_config,
                training_config,
                descriptor,
            ),
            "runtime_context": _runtime_context(
                device_config,
                training_config,
                descriptor,
                collect_compile_metrics=collect_compile_metrics,
            ),
            "parameter_semantics": _parameter_semantics(descriptor),
        }
    if enabled:
        device_facts, runtime_facts = _stm32_context(device_config)
        payload["device_context"].update(device_facts)
        payload["runtime_context"].update(runtime_facts)
        # Seeded studies explicitly describe their actual training policy;
        # legacy prompts remain byte-identical when the option is absent.
        seed = _cfg_get(training_config, "random_seed", None)
        if seed is not None:
            payload["runtime_context"]["training_random_seed"] = seed
            payload["runtime_context"]["training_deterministic_ops"] = bool(
                _cfg_get(training_config, "deterministic_ops", False))
            payload["runtime_context"]["nas_max_epochs"] = _cfg_get(training_config, "nas_epochs")
            payload["guidance"][-1] = (
                "Both insufficient and excessive model capacity can reduce accuracy; "
                "use measured outcomes and the stated training budget rather than "
                "assuming smaller or larger models are better.")
    return {
        "enabled": bool(enabled),
        "version": SEMANTIC_CONTEXT_VERSION,
        "hash": _semantic_hash(payload),
        "payload": payload,
    }


__all__ = [
    "SEMANTIC_CONTEXT_GUIDANCE",
    "SEMANTIC_CONTEXT_VERSION",
    "build_semantic_context",
    "disabled_semantic_context",
]
