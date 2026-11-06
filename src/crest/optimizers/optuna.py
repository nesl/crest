# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Native proposal component; sampling stays inside the existing objective."""

from ..interfaces import OptimizerABC
from ..pipeline_types import BudgetSnapshot, NativeRound, TrialRecord


class OptunaOptimizer(OptimizerABC):
    """Request native sampler rounds without depending on Optuna objects."""

    requires_search_space = False

    def requires_semantic_context(self, config) -> bool:
        """Report whether native sampling needs semantic prompt context.

        Parameters
        ----------
        config : Any
            Optimizer configuration; unused by native sampling.

        Returns
        -------
        bool
            Always ``False``.
        """
        del config
        return False

    def propose_round(
        self, history: tuple[TrialRecord, ...], budget: BudgetSnapshot,
    ) -> NativeRound:
        """Request the number of native trials permitted by the runner.

        Parameters
        ----------
        history : tuple of TrialRecord
            Prior evidence; unused by the native sampler.
        budget : BudgetSnapshot
            Runner budget containing the permitted round size.

        Returns
        -------
        NativeRound
            Request for native objective-side sampling.
        """
        del history
        return NativeRound(budget.permitted_round_size)
