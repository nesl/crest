# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Declarative descriptions of raw Optuna trial parameters."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from ...pipeline_types import ModelBuildContext

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

    def __post_init__(self) -> None:
        """Reject malformed descriptor declarations."""
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Search parameter names must be non-empty strings.")
        if self.kind == "categorical":
            if self.low is not None or self.high is not None:
                raise ValueError(f"Categorical parameter '{self.name}' cannot define bounds.")
            if self.choices is None or len(self.choices) == 0:
                raise ValueError(f"Categorical parameter '{self.name}' requires choices.")
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
    params = list(model_family.trial_search_space(ctx, model_config))

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
            )
        )

    return SearchSpaceDescriptor(tuple(params))
