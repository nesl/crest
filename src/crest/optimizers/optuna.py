# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Native proposal component; sampling stays inside the existing objective."""

from ..interfaces import OptimizerABC
from ..pipeline_types import BudgetSnapshot, NativeRound, TrialRecord


class OptunaOptimizer(OptimizerABC):
    """Request native sampler rounds without depending on Optuna objects."""

    requires_search_space = False

    def requires_semantic_context(self, config) -> bool:
        del config
        return False

    def propose_round(
        self, history: tuple[TrialRecord, ...], budget: BudgetSnapshot,
    ) -> NativeRound:
        del history
        return NativeRound(budget.permitted_round_size)
