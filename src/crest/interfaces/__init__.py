# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Shared component contracts; package imports preserve the original public API.

Concrete implementations belong to datasets, tasks, model_families, and optimizers.
These contracts import no ML runtime until a model-dependent default hook is called.
"""

from .dataset import DatasetABC
from .task import TaskABC
from .model_family import ModelFamilyABC
from .optimizer import OptimizerABC, OptimizerBudgetExhausted

__all__ = [
    "DatasetABC",
    "TaskABC",
    "ModelFamilyABC",
    "OptimizerABC",
    "OptimizerBudgetExhausted",
]
