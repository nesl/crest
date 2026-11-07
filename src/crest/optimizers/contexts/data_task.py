# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Dataset, target, score, and feasibility facts for proposal context."""
from __future__ import annotations

from typing import Any
from .values import _cfg_get, _json_scalar, _json_shape

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


def _dataset_context(
    dataset_name: str,
    dataset_config: Any,
    dataset_bundle: Any,
    model_build_context: Any,
) -> dict[str, Any]:
    """Build a compact dataset and input-feature description.

    Parameters
    ----------
    dataset_name : str
        Registered dataset name used for built-in modality hints.
    dataset_config : Any
        Dataset configuration providing fallback scalar metadata.
    dataset_bundle : Any
        Loaded bundle providing input properties and metadata.
    model_build_context : Any
        Resolved input properties and preferred dataset metadata.

    Returns
    -------
    dict
        Whitelisted dataset identity, input shape, dtype, and feature facts.
    """
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
    """Build task, output, objective, and feasibility semantics.

    Parameters
    ----------
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

    Returns
    -------
    dict
        Task outputs, metric semantics, objective, and feasibility rules.
    """
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


