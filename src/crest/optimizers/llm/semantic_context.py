# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Build deterministic, bounded semantic context for LLM candidate prompts."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from crest.devices import get_device_spec

from .search_space import SearchSpaceDescriptor

SEMANTIC_CONTEXT_VERSION = "v1"
SEMANTIC_CONTEXT_GUIDANCE = (
    "Parameter effects are architectural priors, not measured facts or guaranteed outcomes.",
    "Actual CREST and HIL observations take precedence over semantic priors.",
    "Candidates must still contain the exact raw Optuna parameter names and values.",
)

_BUILTIN_MODALITIES = {
    "oxiod": "inertial_measurement_unit",
    "urbansound8k_mel": "log_mel_audio",
}

_DATASET_METADATA_FIELDS = {
    "sampling_rate_hz": "sampling_rate_hz",
    "sample_rate_hz": "sampling_rate_hz",
    "window_size": "window_size",
    "stride": "stride",
    "input_dim": "input_features",
    "feature_kind": "feature_kind",
    "mel_bins": "mel_bins",
    "window_ms": "feature_window_ms",
    "hop_ms": "feature_hop_ms",
    "clip_duration_s": "clip_duration_s",
    "expected_frames": "feature_frames",
    "batch_period_ms": "batch_period_ms",
}


def _cfg_get(container: Any, key: str, default: Any = None) -> Any:
    """Read one mapping-style or attribute-style field."""
    if container is None:
        return default
    getter = getattr(container, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(container, key, default)


def _json_scalar(value: Any) -> Any | None:
    """Return a finite JSON scalar, including NumPy-like scalar values."""
    if hasattr(value, "item") and callable(value.item):
        try:
            value = value.item()
        except (TypeError, ValueError):
            return None
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else None
    return None


def _json_shape(value: Any) -> list[Any] | None:
    """Normalize a small shape-like sequence without retaining data payloads."""
    if not isinstance(value, (list, tuple)):
        return None
    normalized = [_json_scalar(item) for item in value]
    if any(item is None and original is not None for item, original in zip(normalized, value)):
        return None
    return normalized


def _parameter_semantics(descriptor: SearchSpaceDescriptor) -> dict[str, Any]:
    """Extract only optional semantics; legal bounds remain in the descriptor."""
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


def _dataset_context(
    dataset_name: str,
    dataset_config: Any,
    dataset_bundle: Any,
    model_build_context: Any,
) -> dict[str, Any]:
    """Build a compact dataset and input-feature description."""
    context: dict[str, Any] = {"name": dataset_name}
    modality = _BUILTIN_MODALITIES.get(dataset_name)
    if modality:
        context["modality"] = modality

    input_shape = _json_shape(_cfg_get(model_build_context, "input_shape"))
    if input_shape is None:
        input_shape = _json_shape(_cfg_get(dataset_bundle, "input_shape"))
    if input_shape is not None:
        context["input_shape"] = input_shape
    input_dtype = _json_scalar(_cfg_get(model_build_context, "input_dtype"))
    if input_dtype is None:
        input_dtype = _json_scalar(_cfg_get(dataset_bundle, "input_dtype"))
    if input_dtype not in (None, ""):
        context["input_dtype"] = input_dtype

    metadata = _cfg_get(dataset_bundle, "metadata", {})
    build_metadata = _cfg_get(model_build_context, "dataset_metadata", {})
    for source_key, output_key in _DATASET_METADATA_FIELDS.items():
        value = _json_scalar(_cfg_get(build_metadata, source_key))
        if value is None:
            value = _json_scalar(_cfg_get(metadata, source_key))
        if value is None:
            value = _json_scalar(_cfg_get(dataset_config, source_key))
        if value is not None and output_key not in context:
            context[output_key] = value
    return context


def _task_context(
    task_name: str,
    target_spec: Any,
    metric_contract: Any,
    score_config: Any,
    feasibility_config: Any,
) -> dict[str, Any]:
    """Build task, output, objective, and feasibility semantics."""
    context: dict[str, Any] = {"name": task_name}
    task_type = _json_scalar(_cfg_get(target_spec, "task_type"))
    if task_type not in (None, ""):
        context["type"] = task_type

    output_names = list(_cfg_get(target_spec, "output_names", []) or [])
    output_shapes = list(_cfg_get(target_spec, "output_shapes", []) or [])
    outputs = []
    for index, name in enumerate(output_names):
        output: dict[str, Any] = {"name": str(name)}
        if index < len(output_shapes):
            shape = _json_shape(output_shapes[index])
            if shape is not None:
                output["shape"] = shape
        outputs.append(output)
    if outputs:
        context["outputs"] = outputs

    target_metadata = _cfg_get(target_spec, "metadata", {})
    num_classes = _json_scalar(_cfg_get(target_metadata, "num_classes"))
    if num_classes is not None:
        context["num_classes"] = num_classes

    metric_groups = {}
    for field_name, output_name in (
        ("available_metric_names", "available"),
        ("primary_metric_names", "primary"),
        ("training_only_metric_names", "training_only"),
    ):
        values = _cfg_get(metric_contract, field_name, set()) or set()
        if values:
            metric_groups[output_name] = sorted(str(value) for value in values)
    if metric_groups:
        context["task_metrics"] = metric_groups

    score_type = str(_cfg_get(score_config, "type", "")).strip().lower()
    score_params = _cfg_get(score_config, "params", {})
    objective: dict[str, Any] = {}
    if score_type:
        objective["score_type"] = score_type
    if score_type == "multi-objective":
        objective["study_outputs"] = [
            {
                "metric": str(_cfg_get(item, "metric")),
                "direction": str(_cfg_get(item, "direction")),
            }
            for item in (_cfg_get(score_params, "objectives", []) or [])
        ]
    else:
        objective["study_outputs"] = [{"metric": "score", "direction": "maximize"}]
        components = []
        for term in _cfg_get(score_params, "terms", []) or []:
            component = {
                "metric": str(_cfg_get(term, "metric")),
                "term_type": str(_cfg_get(term, "type")),
            }
            weight = _json_scalar(_cfg_get(term, "weight"))
            if weight is not None:
                component["weight"] = weight
            components.append(component)
        if components:
            objective["score_components"] = components
    context["objective"] = objective

    rules = []
    for rule in _cfg_get(feasibility_config, "rules", []) or []:
        reference = _cfg_get(rule, "reference", {})
        ref_type = str(_cfg_get(reference, "type", "")).strip()
        normalized_reference: dict[str, Any] = {"type": ref_type}
        if ref_type == "literal":
            value = _json_scalar(_cfg_get(reference, "value"))
            if value is not None:
                normalized_reference["value"] = value
        elif ref_type == "metric":
            normalized_reference["metric"] = str(_cfg_get(reference, "metric"))
        normalized_rule = {
            "rule": str(_cfg_get(rule, "rule")),
            "metric": str(_cfg_get(rule, "metric")),
            "condition": str(_cfg_get(rule, "condition")),
            "reference": normalized_reference,
        }
        reason = str(_cfg_get(rule, "reason", "")).strip()
        if reason:
            normalized_rule["reason"] = reason
        rules.append(normalized_rule)
    if rules:
        context["feasibility"] = {
            "train_if_infeasible": bool(
                _cfg_get(feasibility_config, "train_if_infeasible", False)
            ),
            "rules": rules,
        }
    return context


def _device_capacities(device_name: str, device_config: Any) -> dict[str, int]:
    """Read registered capacity metadata without constructing a device backend."""
    try:
        if device_name == "PORTENTA_H7":
            from crest.microcontrollers.arduino_portenta_h7 import (
                build_portenta_h7_spec,
                resolve_portenta_h7_options,
            )

            portenta = _cfg_get(device_config, "portenta", {})
            options = resolve_portenta_h7_options(
                {
                    key: _cfg_get(portenta, key)
                    for key in ("target_core", "split", "security")
                    if _cfg_get(portenta, key) not in (None, "")
                }
            )
            spec = build_portenta_h7_spec(options=options)
        else:
            spec = get_device_spec(device_name)
    except (ImportError, TypeError, ValueError):
        return {}
    return {
        "ram_capacity_bytes": int(spec.max_ram_bytes),
        "flash_capacity_bytes": int(spec.max_flash_bytes),
    }


def _device_context(
    device_config: Any,
    training_config: Any,
    descriptor: SearchSpaceDescriptor,
) -> dict[str, Any]:
    """Build device resource and legal deployment-choice semantics."""
    device_name = str(_cfg_get(device_config, "name", "")).strip().upper()
    context: dict[str, Any] = {"name": device_name}
    context.update(_device_capacities(device_name, device_config))

    clocks = _cfg_get(device_config, "cpu_clock_mhz_options")
    if isinstance(clocks, (list, tuple)) and clocks:
        normalized_clocks = [_json_scalar(clock) for clock in clocks]
        if all(clock is not None for clock in normalized_clocks):
            context["cpu_clock_mhz_choices"] = normalized_clocks

    quantization = _cfg_get(training_config, "quantization", {})
    configured_mode = _json_scalar(_cfg_get(quantization, "mode"))
    if configured_mode not in (None, ""):
        context["configured_quantization_mode"] = configured_mode
    if "quantization_mode" in descriptor:
        context["quantization_choices"] = list(
            descriptor["quantization_mode"].choices or ()
        )

    stm32 = _cfg_get(device_config, "stm32", {})
    weight_storage_mode = _json_scalar(_cfg_get(stm32, "weight_storage_mode"))
    if weight_storage_mode not in (None, ""):
        context["weight_storage_mode"] = weight_storage_mode
    return context


def _runtime_context(
    device_config: Any,
    training_config: Any,
    descriptor: SearchSpaceDescriptor,
    *,
    collect_compile_metrics: bool,
) -> dict[str, Any]:
    """Build compact flags that affect how candidates are evaluated."""
    context = {
        "hil_enabled": bool(_cfg_get(device_config, "hil", False)),
        "training_enabled": bool(_cfg_get(training_config, "train", False)),
        "compile_metrics_enabled": bool(collect_compile_metrics),
        "quantization_search_enabled": "quantization_mode" in descriptor,
        "cpu_clock_search_enabled": "cpu_clock_mhz_index" in descriptor,
    }
    compile_policy = _json_scalar(_cfg_get(device_config, "compile_when_hil_disabled"))
    if compile_policy is not None:
        context["compile_when_hil_disabled"] = compile_policy
    return context


def _semantic_hash(payload: dict[str, Any]) -> str:
    """Hash the exact canonical normalized semantic payload."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
    """Return one versioned semantic record suitable for prompt and ledger use."""
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
]
