"""Exact columnar evidence and pure decoded-network coverage for audio DS-CNN."""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping
from itertools import zip_longest
from typing import Any

from ...errors import HIL_MASTER_SUCCESS
from ...audio_dscnn_decode import AUDIO_DSCNN_SEARCH_CHOICES, _pointwise_filters, resolve_stride_schedule
from .history import MEASUREMENT_FIELDS, _available_measurement, _state_name
from ...optimizer_history import to_plain

DECODED_SCHEMA_VERSION = "audio-dscnn-training-v1"
RAW_FIELDS = tuple(AUDIO_DSCNN_SEARCH_CHOICES)
GRAPH_FIELDS = ("kernel_time", "kernel_freq", "depth_multiplier", "norm_flag", "activation",
                "global_pool_type", "dense_units", "dropout_rate")
HISTORY_COLUMNS = ("trial_number", *RAW_FIELDS, "widths", "decoded_id", "state", "feasibility",
                   "failure_class", "failure_reason", "score", "macro_f1", *MEASUREMENT_FIELDS)
OUTCOME_CLASSES = ("successful", "deterministic", "infeasible", "infrastructure", "unclassified", "pending")


def _trials(history: Any) -> list[Any]:
    """Order detached or study-owned trial evidence chronologically.

    Parameters
    ----------
    history : Any
        Study or iterable of trials.

    Returns
    -------
    list
        Trials ordered by stable trial number.
    """
    return sorted(getattr(history, "trials", history), key=lambda trial: trial.number)


def reconstruct_params(trial: Any, names: Any = None) -> dict[str, Any]:
    """Recover known raw parameters with sampled values taking precedence.

    Parameters
    ----------
    trial : Any
        Detached trial or Optuna trial with proposal and fixed-parameter metadata.
    names : iterable of str, optional
        Fields to retain; defaults to the fourteen audio architecture fields.

    Returns
    -------
    dict
        Known values only; missing values are never synthesized.
    """
    attrs = getattr(trial, "user_attrs", {}) or {}
    provenance = attrs.get("crest_proposal", {})
    proposed = provenance.get("proposal_params", {}) if isinstance(provenance, Mapping) else {}
    fixed = (getattr(trial, "system_attrs", {}) or {}).get("fixed_params", {})
    result = {}
    for source in (proposed, fixed, getattr(trial, "params", {})):
        if isinstance(source, Mapping):
            result.update(source)
    return {name: result[name] for name in (RAW_FIELDS if names is None else names) if name in result}


def decoded_architecture(params: Mapping[str, Any]) -> dict[str, Any] | None:
    """Decode all training graph settings or return unknown for partial input.

    Parameters
    ----------
    params : Mapping
        Raw audio DS-CNN architecture parameters.

    Returns
    -------
    dict or None
        Versioned graph signature, including training dropout and resolved strides.
    """
    if any(name not in params for name in RAW_FIELDS):
        return None
    try:
        for name, choices in AUDIO_DSCNN_SEARCH_CHOICES.items():
            value = params[name]
            if name == "norm_flag":
                if not isinstance(value, (bool, int)) or value not in (0, 1):
                    return None
            elif isinstance(value, bool) or value not in choices:
                return None
        result = {"schema_version": DECODED_SCHEMA_VERSION,
                  "widths": [_pointwise_filters(params, i) for i in range(params["num_blocks"])],
                  "strides": [list(s) for s in resolve_stride_schedule(params["stride_schedule"], params["num_blocks"])],
                  **{name: params[name] for name in GRAPH_FIELDS}}
        result["norm_flag"] = bool(result["norm_flag"])
        result["dropout_rate"] = float(result["dropout_rate"])
        return result
    except (ValueError, TypeError, OverflowError):
        return None


def decoded_fingerprint(params: Mapping[str, Any]) -> str | None:
    """Return a stable complete training-network identifier, or unknown.

    Parameters
    ----------
    params : Mapping
        Raw audio architecture.

    Returns
    -------
    str or None
        SHA-256 signature with an explicit schema version prefix.
    """
    decoded = decoded_architecture(params)
    if decoded is None:
        return None
    data = json.dumps(decoded, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return DECODED_SCHEMA_VERSION + ":" + hashlib.sha256(data.encode()).hexdigest()[:24]


def candidate_cell(params: Mapping[str, Any]) -> str | None:
    """Resolve a depth/stride cell using only its required known parameters.

    Parameters
    ----------
    params : Mapping
        Known raw architecture fields; only depth and schedule are required.

    Returns
    -------
    str or None
        Stable resolved cell identifier, or unknown.
    """
    try:
        blocks = params["num_blocks"]
        strides = resolve_stride_schedule(params["stride_schedule"], blocks)
        return str(blocks) + ":" + ",".join(f"{a}x{b}" for a, b in strides)
    except (KeyError, TypeError, ValueError):
        return None


def failure_class(trial: Any) -> str:
    """Keep missing classifications explicit rather than inferring determinism.

    Parameters
    ----------
    trial : Any
        Trial with persisted failure metadata.

    Returns
    -------
    str
        Persisted failure classification, otherwise unclassified.
    """
    attrs = getattr(trial, "user_attrs", {}) or {}
    return str(attrs.get("measurement_failure_class") or attrs.get("candidate_failure_class") or "unclassified")


def outcome_class(trial: Any) -> str:
    """Classify informative evidence without turning unclassified failures into it.

    Parameters
    ----------
    trial : Any
        Trial carrying state, measurements and failure metadata.

    Returns
    -------
    str
        One of the six explicit outcome classes.
    """
    state = _state_name(trial).upper()
    attrs = getattr(trial, "user_attrs", {}) or {}
    if state in {"RUNNING", "WAITING"}:
        return "pending"
    classification = failure_class(trial)
    if classification == "infrastructure":
        return "infrastructure"
    if attrs.get("feasibility_status") == "infeasible":
        return "infeasible"
    if classification == "deterministic":
        return "deterministic"
    errors = any(attrs.get(k) not in (None, "", HIL_MASTER_SUCCESS, "HIL_MASTER_SUCCESS")
                 for k in ("error_code", "hil_error_code", "error_code_label"))
    values = getattr(trial, "values", None)
    if (state == "COMPLETE" and not errors and not attrs.get("pruned")
            and values and any(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values)):
        return "successful"
    return "unclassified"


def outcome_reason(trial: Any) -> Any:
    """Select useful failure evidence without presenting success as a failure.

    Parameters
    ----------
    trial : Any
        Trial carrying persisted outcome attributes.

    Returns
    -------
    Any
        Failure reason, or None for successful and pending work.
    """
    if outcome_class(trial) in {"successful", "pending"}:
        return None
    attrs = getattr(trial, "user_attrs", {}) or {}
    for value in (attrs.get("feasibility_reason"), attrs.get("failure_reason"),
                  attrs.get("prune_reason"), attrs.get("error_code_label"),
                  (getattr(trial, "system_attrs", {}) or {}).get("fail_reason")):
        if value not in (None, "", HIL_MASTER_SUCCESS, "HIL_MASTER_SUCCESS"):
            return value
    return None


def observed_measurement(trial: Any, name: str) -> Any:
    """Hide ambiguous penalty defaults while preserving supported real readings.

    Parameters
    ----------
    trial : Any
        Persisted measurement and feasibility evidence.
    name : str
        Measurement attribute name.

    Returns
    -------
    Any
        Available reading, or None when missing or a possible penalty sentinel.
    """
    attrs = getattr(trial, "user_attrs", {}) or {}
    value = attrs.get(name)
    if not _available_measurement(value):
        return None
    # Old producers set absent timing/energy to 10000 on the penalty path.
    # Explicit evaluated feasibility or successful completion supports a real
    # reading; older failed records without that evidence remain unknown.
    if (name in {"latency_ms", "energy_mj_per_inference"} and value == 10000
            and attrs.get("feasibility_status") not in {"feasible", "infeasible"}
            and outcome_class(trial) != "successful"):
        return None
    return value


def _compact_metric(value: Any) -> Any:
    """Serialize exact integral metrics compactly without changing raw parameters."""
    return int(value) if isinstance(value, float) and math.isfinite(value) and value.is_integer() else value


def exhausted_unknown_retries(history: Any) -> set[str]:
    """Find decoded networks with two unclassified terminal failures.

    Parameters
    ----------
    history : Any
        Saved trial evidence; infrastructure failures are excluded.

    Returns
    -------
    set[str]
        Identities blocked after the original attempt and one retry. This
        operational limit does not make these failures architecture evidence.
    """
    counts = defaultdict(int)
    for trial in _trials(history):
        if (_state_name(trial).upper() in {"COMPLETE", "FAIL", "PRUNED"}
                and outcome_class(trial) == "unclassified"):
            identity = decoded_fingerprint(reconstruct_params(trial))
            if identity is not None:
                counts[identity] += 1
    return {identity for identity, count in counts.items() if count >= 2}


def build_full_history(history: Any, *, directions: tuple[str, ...] = ()) -> dict[str, Any]:
    """Serialize every terminal trial chronologically, keeping pending work separate.

    Parameters
    ----------
    history : Any
        Study snapshot or iterable of trials.
    directions : tuple of str
        Objective directions retained for interpreting score values.

    Returns
    -------
    dict
        Shared columns, terminal row arrays, pending row arrays and directions.
    """
    rows, pending = [], []
    for trial in _trials(history):
        attrs = trial.user_attrs
        params = reconstruct_params(trial)
        decoded = decoded_architecture(params)
        values = getattr(trial, "values", None)
        score = values[0] if values and len(values) == 1 and isinstance(values[0], (int, float)) and math.isfinite(values[0]) else None
        metrics = attrs.get("task_metrics", {})
        macro = metrics.get("macro_f1") if isinstance(metrics, Mapping) else None
        reason = outcome_reason(trial)
        state = _state_name(trial).lower()
        outcome = outcome_class(trial)
        row = [trial.number, *[params.get(name) for name in RAW_FIELDS],
               decoded["widths"] if decoded else None,
               decoded_fingerprint(params).rsplit(":", 1)[-1] if decoded else None, state,
               attrs.get("feasibility_status", attrs.get("feasible")),
               None if outcome in {"successful", "pending"} else outcome, reason,
               _compact_metric(score), _compact_metric(macro) if _available_measurement(macro) else None,
               *[_compact_metric(observed_measurement(trial, name)) for name in MEASUREMENT_FIELDS]]
        if state in {"complete", "fail", "pruned"}:
            rows.append(row)
        elif state in {"running", "waiting"}:
            pending.append(row)
    table = {"columns": list(HISTORY_COLUMNS), "rows": rows, "pending": pending,
             "directions": list(directions), "decoded_schema": DECODED_SCHEMA_VERSION}
    exhausted = exhausted_unknown_retries(history)
    table["retry_exhausted_trials"] = [trial.number for trial in _trials(history)
        if decoded_fingerprint(reconstruct_params(trial)) in exhausted]
    _intern_failure_reasons(table)
    return to_plain(table, nonfinite_to_none=True)


def _intern_failure_reasons(table: dict[str, Any]) -> None:
    """Assign IDs to observed explanations in first-appearance order.

    Parameters
    ----------
    table : dict
        Columnar history containing only explanations observed so far.

    Returns
    -------
    None
        Integer references replace reason strings with an explicit encoding key.
    """
    index = table["columns"].index("failure_reason")
    rows = table["rows"] + table["pending"]
    values = [row[index] for row in rows]
    if any(value is not None and not isinstance(value, str) for value in values):
        return
    reasons = list(dict.fromkeys(value for value in values if value is not None))
    if not reasons:
        return
    mapping = {reason: number for number, reason in enumerate(reasons)}
    encoded = [mapping[value] if value is not None else None for value in values]
    metadata = {"failure_reason_encoding": "dictionary_index", "failure_reasons": reasons}
    for row, value in zip(rows, encoded):
        row[index] = value
    table.update(metadata)


def _choices(descriptor: Any, name: str) -> tuple[Any, ...]:
    """Resolve restricted categorical choices with authoritative defaults.

    Parameters
    ----------
    descriptor : Any
        Active search descriptor, or None for the full audio search space.
    name : str
        Architecture parameter name.

    Returns
    -------
    tuple
        Active categorical values in declaration order.
    """
    if descriptor is None or name not in descriptor:
        return AUDIO_DSCNN_SEARCH_CHOICES[name]
    param = descriptor[name]
    choices = getattr(param, "choices", None)
    return tuple(choices) if choices else AUDIO_DSCNN_SEARCH_CHOICES[name]


def build_coverage(history: Any, descriptor: Any = None) -> dict[str, Any]:
    """Count distinct decoded outcomes in every supported resolved stride cell.

    Parameters
    ----------
    history : Any
        Study or ordered trial snapshot; warmup and pending work are included.
    descriptor : Any, optional
        Restricted active search descriptor for block counts and schedule aliases.

    Returns
    -------
    dict
        Cell counts, observed width signatures and depth/first-width counts.
        Informative coverage is the union of successful, deterministic and
        confirmed infeasible network IDs, never their sum.
    """
    cells = {}
    for blocks in sorted(_choices(descriptor, "num_blocks")):
        for alias in _choices(descriptor, "stride_schedule"):
            params = {"num_blocks": blocks, "stride_schedule": alias}
            key = candidate_cell(params)
            if key not in cells:
                cells[key] = {"cell_id": key, "num_blocks": blocks,
                              "strides": [list(s) for s in resolve_stride_schedule(alias, blocks)], "aliases": []}
            cells[key]["aliases"].append(alias)
    unknown = {key: 0 for key in cells}
    identities = {key: {kind: set() for kind in OUTCOME_CLASSES} for key in cells}
    widths, first_widths = defaultdict(set), defaultdict(set)
    for trial in _trials(history):
        params = reconstruct_params(trial)
        key, identity = candidate_cell(params), decoded_fingerprint(params)
        if key not in identities:
            continue
        if identity is None:
            unknown[key] += 1
            continue
        kind = outcome_class(trial)
        identities[key][kind].add(identity)
        decoded = decoded_architecture(params)
        widths[tuple(decoded["widths"])].add(identity)
        first_widths[(len(decoded["widths"]), decoded["widths"][0])].add(identity)
    result = []
    for key, cell in cells.items():
        groups = identities[key]
        informative = set.union(*(groups[k] for k in ("successful", "deterministic", "infeasible")))
        # Pending retries of an informative network reserve no additional evidence.
        pending = groups["pending"] - informative
        result.append({**cell, **{k: len(v) for k, v in groups.items()},
                       "informative": len(informative), "pending": len(pending),
                       "unknown_decoded_trials": unknown[key]})
    return {"cells": result, "target_per_cell": 2,
            "goal_complete": all(c["informative"] >= 2 for c in result),
            "width_signatures": [{"widths": list(k), "unique_networks": len(v)} for k, v in sorted(widths.items())],
            "depth_first_width": [{"num_blocks": k[0], "first_width": k[1], "unique_networks": len(v)}
                                  for k, v in sorted(first_widths.items())]}


def assign_coverage_targets(coverage: Mapping[str, Any], count: int) -> list[dict[str, Any]]:
    """Assign least-covered cells, reserving pending and earlier assigned slots.

    Parameters
    ----------
    coverage : Mapping
        Result of ``build_coverage``.
    count : int
        Number of exploration slots to target.

    Returns
    -------
    list of dict
        At most ``count`` target cells. Ties use a stable hash rather than
        shallow-first ordering; fully reserved cells are skipped.
    """
    cells = coverage["cells"]
    reserved = {c["cell_id"]: c["informative"] + c["pending"] for c in cells}
    result = []
    for _ in range(count):
        available = [c for c in cells if reserved[c["cell_id"]] < coverage.get("target_per_cell", 2)]
        if not available:
            break
        cell = min(available, key=lambda c: (reserved[c["cell_id"]],
                   hashlib.sha256(c["cell_id"].encode()).hexdigest()))
        result.append(dict(cell))
        reserved[cell["cell_id"]] += 1
    return result


def build_eligible_parents(history: Any) -> tuple[dict[str, Any], ...]:
    """Return known feasible successful trials with decoded parent identities.

    Parameters
    ----------
    history : Any
        Study or iterable of trial records.

    Returns
    -------
    tuple of dict
        Feasible successful trial references in chronological order.
    """
    result = []
    for trial in _trials(history):
        attrs = trial.user_attrs
        if outcome_class(trial) != "successful" or not (attrs.get("feasibility_status") == "feasible" or attrs.get("feasible") is True):
            continue
        params = reconstruct_params(trial)
        identity = decoded_fingerprint(params)
        if identity:
            result.append({"trial_number": trial.number, "decoded_id": identity, "params": params,
                           "score": trial.values[0] if trial.values and len(trial.values) == 1 else None})
    return tuple(result)


def parameter_differences(candidate: Mapping[str, Any], parent: Mapping[str, Any]) -> dict[str, Any]:
    """Compute exact raw and decoded before/after differences from a parent.

    Parameters
    ----------
    candidate : Mapping
        Proposed raw architecture.
    parent : Mapping
        Historical parent raw architecture.

    Returns
    -------
    dict
        Raw and decoded differing fields with before/after values.
    """
    def diff(a: Mapping, b: Mapping) -> dict:
        return {k: [b.get(k), a.get(k)] for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}
    decoded, parent_decoded = decoded_architecture(candidate), decoded_architecture(parent)
    return {"raw": diff(candidate, parent), "decoded": diff(decoded, parent_decoded)
            if decoded is not None and parent_decoded is not None else None}


def _decoded_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> int:
    # Freeze variable-length handling: missing width/stride blocks each count once.
    return (sum(a != b for a, b in zip_longest(left["widths"], right["widths"]))
            + sum(a != b for a, b in zip_longest(left["strides"], right["strides"]))
            + sum(left[k] != right[k] for k in GRAPH_FIELDS))


def nearest_prior_distances(candidate: Mapping[str, Any], history: Any) -> dict[str, Any]:
    """Report raw Hamming and decoded structural nearest distances independently.

    Parameters
    ----------
    candidate : Mapping
        Complete proposed architecture.
    history : Any
        Earlier trials only; caller controls the evidence cutoff.

    Returns
    -------
    dict
        Nearest trial/distance per representation, or None when unknown.
        Decoded distance counts differing width blocks, stride blocks, and each
        other graph field; absent trailing blocks count as differences.
    """
    raw, decoded = [], []
    graph = decoded_architecture(candidate)
    for trial in _trials(history):
        prior = reconstruct_params(trial)
        if all(k in prior and k in candidate for k in RAW_FIELDS):
            raw.append((sum(candidate[k] != prior[k] for k in RAW_FIELDS), trial.number))
        prior_graph = decoded_architecture(prior)
        if graph is not None and prior_graph is not None:
            decoded.append((_decoded_distance(graph, prior_graph), trial.number))
    def nearest(items: list) -> dict | None:
        if not items:
            return None
        distance, number = min(items)
        return {"trial": number, "distance": distance}
    return {"raw": nearest(raw), "decoded": nearest(decoded)}
