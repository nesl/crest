# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Concrete task implementations for the modular CREST pipeline."""

from .odometry_regression import OdometryRegressionTask
from .sound_classification import SoundClassificationTask

__all__ = ["OdometryRegressionTask", "SoundClassificationTask"]
