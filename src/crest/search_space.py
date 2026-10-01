# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Declarative descriptions of raw Optuna trial parameters."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any, Literal

from .pipeline_types import ModelBuildContext

SearchParamKind = Literal["int", "float", "categorical"]


@dataclass(frozen=True)
class SearchParam:
    """Describe one raw parameter accepted by an Optuna trial.

    Integer and float parameters use inclusive ``low`` and ``high`` bounds.
    Categorical parameters use ordered ``choices``. These names and values are
    the persisted Optuna surface, before any model-family decoding.
    """

    name: str
    kind: SearchParamKind
    low: int | float | None = None
    high: int | float | None = None
    choices: tuple[Any, ...] | None = None
    description: str = ""
    units: str | None = None
    typical_effects: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject malformed descriptor declarations."""
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Search parameter names must be non-empty strings.")
        if not isinstance(self.description, str):
            raise ValueError(f"Search parameter '{self.name}' description must be a string.")
        if self.units is not None and not isinstance(self.units, str):
            raise ValueError(f"Search parameter '{self.name}' units must be a string or None.")
        if not isinstance(self.typical_effects, tuple) or not all(
            isinstance(effect, str) for effect in self.typical_effects
        ):
            raise ValueError(
                f"Search parameter '{self.name}' typical_effects must be a tuple of strings."
            )
        if self.kind == "categorical":
            if self.low is not None or self.high is not None:
                raise ValueError(f"Categorical parameter '{self.name}' cannot define bounds.")
            if not isinstance(self.choices, (tuple, list)) or len(self.choices) == 0:
                raise ValueError(f"Categorical parameter '{self.name}' requires choices.")
            if any(
                choice is not None and type(choice) not in (str, int, float, bool)
                or isinstance(choice, float) and not math.isfinite(choice)
                for choice in self.choices
            ):
                raise ValueError(
                    f"Categorical parameter '{self.name}' requires finite scalar choices."
                )
            object.__setattr__(self, "choices", tuple(self.choices))
            return
        if self.kind not in {"int", "float"}:
            raise ValueError(f"Unsupported search parameter kind: {self.kind!r}.")
        if self.choices is not None:
            raise ValueError(f"Numeric parameter '{self.name}' cannot define categorical choices.")
        if self.kind == "int":
            bounds_are_valid = all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in (self.low, self.high)
            )
        else:
            bounds_are_valid = all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in (self.low, self.high)
            )
        if not bounds_are_valid:
            raise ValueError(f"{self.kind} parameter '{self.name}' requires numeric bounds.")
        if not all(math.isfinite(value) for value in (self.low, self.high)):
            raise ValueError(f"Search parameter '{self.name}' requires finite bounds.")
        if self.low > self.high:
            raise ValueError(f"Search parameter '{self.name}' has low greater than high.")

    def suggest(self, trial: Any) -> Any:
        """Sample this parameter through an Optuna-compatible trial object."""
        if self.kind == "int":
            return trial.suggest_int(self.name, self.low, self.high)
        if self.kind == "float":
            return trial.suggest_float(self.name, self.low, self.high)
        return trial.suggest_categorical(self.name, self.choices)


@dataclass(frozen=True)
class SearchSpaceDescriptor(Mapping[str, SearchParam]):
    """Ordered mapping of raw trial parameter names to declarations."""

    params: tuple[SearchParam, ...]

    def __post_init__(self) -> None:
        """Require each raw Optuna parameter name exactly once."""
        if not isinstance(self.params, (tuple, list)) or not all(
            isinstance(param, SearchParam) for param in self.params
        ):
            raise ValueError("Search-space params must contain SearchParam declarations.")
        object.__setattr__(self, "params", tuple(self.params))
        names = [param.name for param in self.params]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Duplicate search parameter names: {', '.join(duplicates)}.")

    def __getitem__(self, name: str) -> SearchParam:
        """Return the declaration for ``name``."""
        for param in self.params:
            if param.name == name:
                return param
        raise KeyError(name)

    def __iter__(self) -> Iterator[str]:
        """Iterate raw parameter names in sampling order."""
        return (param.name for param in self.params)

    def __len__(self) -> int:
        """Return the number of raw trial parameters."""
        return len(self.params)


def _cfg_get(container: Any, key: str, default: Any = None) -> Any:
    """Read a field from a mapping-like or namespace-like config object."""
    if container is None:
        return default
    getter = getattr(container, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(container, key, default)


def build_search_space_descriptor(
    model_family: Any,
    ctx: ModelBuildContext,
    model_config: Any,
    config: Any,
    *,
    collect_compile_metrics: bool,
) -> SearchSpaceDescriptor:
    """Build the active raw Optuna search surface for one CREST run.

    ``collect_compile_metrics`` is supplied by the NAS runner because that
    decision depends on its score, prune, and HIL policy. It controls both the
    deployment-path quantization search and CPU-clock sampling exactly as it
    does in ``NASModelClient.objective``.
    """
    hook = getattr(model_family, "trial_search_space", None)
    if hook is None:
        raise NotImplementedError(
            f"Model family '{getattr(model_family, 'name', type(model_family).__name__)}' "
            "does not expose a trial search-space descriptor."
        )
    if not callable(hook):
        raise ValueError("Model-family trial_search_space must be callable when declared.")
    params = list(hook(ctx, model_config))

    training = _cfg_get(config, "training")
    quantization = _cfg_get(training, "quantization")
    uses_quantized_deployment_path = collect_compile_metrics or bool(
        _cfg_get(training, "train", False)
    )
    if uses_quantized_deployment_path and bool(_cfg_get(quantization, "search", False)):
        params.append(
            SearchParam(
                name="quantization_mode",
                kind="categorical",
                choices=tuple(_cfg_get(quantization, "choices", ())),
                description="Controls the numeric representation used for deployment export.",
                typical_effects=(
                    "The selected representation typically affects model size, arithmetic, and calibration requirements.",
                ),
            )
        )

    device = _cfg_get(config, "device")
    cpu_clock_mhz_options: Sequence[int] | None = _cfg_get(
        device,
        "cpu_clock_mhz_options",
        None,
    )
    if collect_compile_metrics and cpu_clock_mhz_options is not None:
        params.append(
            SearchParam(
                name="cpu_clock_mhz_index",
                kind="int",
                low=0,
                high=len(cpu_clock_mhz_options) - 1,
                description="Selects an index into the configured target-MCU CPU clock choices.",
                typical_effects=(
                    "The selected clock typically affects measured latency, power, and energy tradeoffs.",
                ),
            )
        )

    return SearchSpaceDescriptor(tuple(params))


def _categorical_match(value: Any, choices: tuple[Any, ...]) -> bool:
    """Match categorical values without Python's bool/int equality ambiguity."""
    return any(type(value) is type(choice) and value == choice for choice in choices)


def value_error(param: SearchParam, value: Any) -> tuple[str, str] | None:
    """Return a structured code/message when ``value`` violates ``param``."""
    if param.kind == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            return "type_error", f"'{param.name}' must be an integer."
        if value < param.low or value > param.high:
            return "out_of_range", f"'{param.name}' must be between {param.low} and {param.high}."
        return None

    if param.kind == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return "type_error", f"'{param.name}' must be a finite number."
        if not math.isfinite(float(value)):
            return "non_finite_float", f"'{param.name}' must be finite."
        if value < param.low or value > param.high:
            return "out_of_range", f"'{param.name}' must be between {param.low} and {param.high}."
        return None

    choices = param.choices or ()
    if isinstance(value, float) and not math.isfinite(value):
        return "non_finite_float", f"'{param.name}' must be finite."
    if not _categorical_match(value, choices):
        return "invalid_categorical", f"'{param.name}' must be one of {choices!r}."
    return None


def candidate_error(
    candidate: Mapping[str, Any],
    descriptor: SearchSpaceDescriptor,
) -> tuple[str, str] | None:
    """Check the exact active raw fields without applying duplicate policy."""
    if not isinstance(candidate, Mapping):
        return "type_error", "Candidate parameters must be a mapping."
    if not all(isinstance(key, str) for key in candidate):
        return "unknown_keys", "Candidate parameter keys must be strings."
    expected_keys = set(descriptor)
    actual_keys = set(candidate)
    missing = sorted(expected_keys - actual_keys)
    if missing:
        return "missing_keys", f"Missing required keys: {', '.join(missing)}."
    unknown = sorted(actual_keys - expected_keys)
    if unknown:
        return "unknown_keys", f"Unknown keys: {', '.join(unknown)}."
    for name in descriptor:
        error = value_error(descriptor[name], candidate[name])
        if error is not None:
            return error
    return None


def validate_candidate(
    candidate: Mapping[str, Any],
    descriptor: SearchSpaceDescriptor,
) -> None:
    """Reject illegal explicit parameters before the runner queues any batch.

    Deliberate repeated experiments remain legal. Proposal components own any
    duplicate filtering, repair, and fallback policy.
    """
    error = candidate_error(candidate, descriptor)
    if error is not None:
        raise ValueError(f"{error[0]}: {error[1]}")
