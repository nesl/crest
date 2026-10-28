"""Offline checks for the explicit training randomness policy."""
import random
from unittest.mock import patch

import numpy as np
import pytest
import tensorflow as tf

from crest.search.training_randomness import reset_training_randomness, training_randomness_policy


@pytest.mark.parametrize("seed", [True, -1, 2**32, 1.5, "0"])
def test_invalid_training_seed_is_rejected(seed):
    # Invalid values must fail before any campaign starts.
    with pytest.raises(ValueError):
        training_randomness_policy({"random_seed": seed})


def test_legacy_policy_does_not_touch_rng():
    # Existing unseeded runs must not acquire an implicit seed.
    with patch.object(tf.keras.utils, "set_random_seed") as setter:
        assert reset_training_randomness({}) is None
        setter.assert_not_called()
    with pytest.raises(ValueError):
        training_randomness_policy({"deterministic_ops": True})


def test_seed_resets_initialization_and_random_streams():
    # Repeated evaluations reset all three RNGs despite intervening draws.
    policy = {"random_seed": 0, "deterministic_ops": True}
    with patch.object(tf.config.experimental, "enable_op_determinism") as enable:
        record = reset_training_randomness(policy)
        first = (random.random(), np.random.rand(), tf.random.uniform((4,)).numpy())
        layer1 = tf.keras.layers.Dense(3)
        layer1(tf.ones((1, 2)))
        random.random(); np.random.rand(); tf.random.uniform((100,))
        assert reset_training_randomness(policy) == record
        second = (random.random(), np.random.rand(), tf.random.uniform((4,)).numpy())
        layer2 = tf.keras.layers.Dense(3)
        layer2(tf.ones((1, 2)))
        assert first[:2] == second[:2]
        np.testing.assert_array_equal(first[2], second[2])
        for a, b in zip(layer1.get_weights(), layer2.get_weights()):
            np.testing.assert_array_equal(a, b)
        assert record["seed"] == 0
        assert enable.call_count == 2
