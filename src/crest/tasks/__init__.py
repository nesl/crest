"""Concrete task implementations for the modular CREST pipeline."""

from .odometry_regression import OdometryRegressionTask
from .sound_classification import SoundClassificationTask

__all__ = ["OdometryRegressionTask", "SoundClassificationTask"]
