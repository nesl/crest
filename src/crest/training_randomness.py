"""Opt-in, recorded training randomness policy shared by all optimizers."""

from .component_selection import cfg_get


def training_randomness_policy(training):
    """Validate the configured per-evaluation training seed.

    Parameters
    ----------
    training : mapping or object
        Training configuration; absent seed retains legacy behavior.

    Returns
    -------
    dict or None
        Serializable policy, or None when seeding is disabled.
    """
    seed = cfg_get(training, "random_seed", None)
    deterministic = cfg_get(training, "deterministic_ops", False)
    if not isinstance(deterministic, bool):
        raise ValueError("training.deterministic_ops must be a boolean.")
    if seed is None:
        if deterministic:
            raise ValueError("training.deterministic_ops requires training.random_seed.")
        return None
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("training.random_seed must be an integer in [0, 2**32).")
    return {"seed": seed, "deterministic_ops": deterministic,
            "policy": "fixed_per_evaluation_v1"}


def reset_training_randomness(training):
    """Reset Python, NumPy and TensorFlow before each model evaluation.

    Parameters
    ----------
    training : mapping or object
        Opt-in training seed and deterministic-op settings.

    Returns
    -------
    dict or None
        Policy with the TensorFlow version used, or None for legacy behavior.

    Notes
    -----
    The same seed is reused across architectures and final fits, independent of
    trial order and optimizer. Deterministic operations are process-wide and
    require a compatible, unchanged software/hardware environment.
    """
    policy = training_randomness_policy(training)
    if policy is None:
        return None
    import tensorflow as tf
    tf.keras.utils.set_random_seed(policy["seed"])
    if policy["deterministic_ops"]:
        tf.config.experimental.enable_op_determinism()
    return {**policy, "tensorflow_version": tf.__version__}
