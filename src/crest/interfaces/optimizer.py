# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Candidate proposal and optimizer lifecycle contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..pipeline_types import (
    BudgetSnapshot, ExplicitRound, NativeRound, SearchContext, TrialRecord,
)


class OptimizerBudgetExhausted(RuntimeError):
    """Stop proposing when an optimizer-owned resource budget cannot fund a round."""

class OptimizerABC(ABC):
    """Choose search rounds within CREST's shared Optuna execution lifecycle.

    Components use zero-argument construction and receive normalized optimizer
    configuration. They never evaluate, enqueue, or mutate a Study or Trial.
    Stateful components derive evidence from fresh history on every call.

    Attributes
    ----------
    requires_search_space : bool
        Whether initialization needs a declarative raw parameter descriptor.
        Defaults to True; native sampler components can opt out.
    """

    requires_search_space: bool = True

    def requires_semantic_context(self, config: Any) -> bool:
        """Declare whether initialization needs the stable semantic payload.

        Parameters
        ----------
        config : Any
            Normalized component configuration; unused by the default hook.

        Returns
        -------
        bool
            True by default. Components may override this to skip semantic
            context construction for configurations that do not need it.
        """
        del config
        return True

    @property
    def name(self) -> str:
        """Return the default component display name.

        Returns
        -------
        str
            Implementing class name. Components may provide a stable override.
        """
        return type(self).__name__

    def validate_config(self, config: Any) -> None:
        """Validate component settings without credentials or runtime access.

        Parameters
        ----------
        config : Any
            Normalized optimizer configuration. The default accepts it.

        Raises
        ------
        ValueError
            Implementations should raise this for invalid component settings.
        """
        del config

    def identity_config(self, config: Any) -> dict[str, Any]:
        """Return deterministic JSON-safe, non-secret proposal settings.

        Configurable components must override this hook for settings that
        affect their proposals. Paths, secrets, and run budgets are excluded.

        Parameters
        ----------
        config : Any
            Normalized optimizer configuration.

        Returns
        -------
        dict[str, Any]
            Settings used in strict resume-identity comparison. The default
            returns an empty mapping for a component without proposal settings.
        """
        del config
        return {}

    def normalize_stored_identity(self, stored: Any, active: dict[str, Any]) -> Any:
        """Apply component-owned compatibility aliases before strict comparison.

        Parameters
        ----------
        stored : Any
            Persisted optimizer signature, including its version and name.
        active : dict[str, Any]
            Current complete signature used for strict equality afterward.

        Returns
        -------
        Any
            Comparable signature. The default leaves persisted data unchanged.
        """
        del active
        return stored

    def initialize(self, context: SearchContext, config: Any) -> None:
        """Initialize component-local runtime state.

        Parameters
        ----------
        context : SearchContext
            Detached run metadata and requested search-space information.
        config : Any
            Validated optimizer configuration.

        Notes
        -----
        The default hook is a no-op. Implementations may create their runtime
        clients here, after configuration and campaign identity validation.
        """
        del context, config

    @abstractmethod
    def propose_round(
        self,
        history: tuple[TrialRecord, ...],
        budget: BudgetSnapshot,
    ) -> NativeRound | ExplicitRound:
        """Propose an explicit batch or native sampling round.

        Parameters
        ----------
        history : tuple[TrialRecord, ...]
            Fresh detached trial evidence, including reserved and failed trials.
        budget : BudgetSnapshot
            Current attempt reservations and permitted round size.

        Returns
        -------
        NativeRound or ExplicitRound
            Requested work for the shared runner to validate and execute.
            The component must not enqueue or evaluate trials itself.

        Raises
        ------
        OptimizerBudgetExhausted
            If a component-owned resource budget cannot fund another round.
        """
