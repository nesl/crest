# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Shared test helpers used across CREST unit and integration suites."""

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf  # type: ignore[attr-defined]

ROOT_DIR = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT_DIR / "test"
SRC_DIR = ROOT_DIR / "src"
SKETCH_SOURCE_DIR = TEST_DIR / "odom_tcn"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crest.microcontrollers.arduino_base import ARDUINO_CLI_BIN  # noqa: E402


def _cli_exists() -> bool:
    """Determine whether the configured Arduino CLI binary is callable.

    Returns
    -------
    bool
        ``True`` when the configured Arduino CLI exists and is executable, or
        when it can be resolved on ``PATH``.
    """
    cli_path = Path(ARDUINO_CLI_BIN)
    if cli_path.exists() and os.access(cli_path, os.X_OK):
        return True
    return shutil.which(ARDUINO_CLI_BIN) is not None


class TinyModelMixin:
    """Provide a small model + dataset so converter tests stay fast."""

    @classmethod
    def setUpClass(cls):
        """Create one deterministic tiny model and training batch per test class."""
        super().setUpClass()
        tf.random.set_seed(1234)
        np.random.seed(1234)
        cls.model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(4,), name="input"),
                tf.keras.layers.Dense(4, activation="relu"),
                tf.keras.layers.Dense(2, activation="linear"),
            ]
        )
        cls.model.compile(optimizer="adam", loss="mse")
        cls.train_x = np.random.rand(16, 4).astype(np.float32)
        cls.train_y = np.random.rand(16, 2).astype(np.float32)
