# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Audio DS-CNN model family for cached log-mel classification inputs."""

from __future__ import annotations

from typing import Any, Mapping

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    DepthwiseConv2D,
    Dropout,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    Input,
    ReLU,
    Reshape,
)
from ..interfaces import ModelFamilyABC
from ..model_metrics import count_flops_keras
from ..pipeline_types import ModelBuildContext, TargetSpec

BASE_CHANNELS_CHOICES = (4, 8, 12, 16, 20, 24, 32)
NUM_BLOCKS_CHOICES = (2, 3, 4, 5, 6)
KERNEL_TIME_CHOICES = (3, 5)
KERNEL_FREQ_CHOICES = (3, 5)
STRIDE_SCHEDULE_CHOICES = ("light", "balanced", "aggressive", "early_time", "early_freq", "late")
CHANNEL_GROWTH_CHOICES = (1.0, 1.5, 2.0)
MAX_CHANNELS_CHOICES = (64, 96, 128)
DEPTH_MULTIPLIER_CHOICES = (1, 2)
POINTWISE_SCALE_CHOICES = (0.75, 1.0, 1.25, 1.5)
DROPOUT_RATE_CHOICES = (0.0, 0.1, 0.2, 0.3)
NORM_FLAG_CHOICES = (True, False)
DENSE_UNITS_CHOICES = (0, 16, 32, 64)
GLOBAL_POOL_TYPE_CHOICES = ("avg", "max")
ACTIVATION_CHOICES = ("relu", "relu6")

AUDIO_DSCNN_SEARCH_CHOICES: dict[str, tuple[Any, ...]] = {
    "base_channels": BASE_CHANNELS_CHOICES,
    "num_blocks": NUM_BLOCKS_CHOICES,
    "kernel_time": KERNEL_TIME_CHOICES,
    "kernel_freq": KERNEL_FREQ_CHOICES,
    "stride_schedule": STRIDE_SCHEDULE_CHOICES,
    "channel_growth": CHANNEL_GROWTH_CHOICES,
    "max_channels": MAX_CHANNELS_CHOICES,
    "depth_multiplier": DEPTH_MULTIPLIER_CHOICES,
    "pointwise_scale": POINTWISE_SCALE_CHOICES,
    "dropout_rate": DROPOUT_RATE_CHOICES,
    "norm_flag": NORM_FLAG_CHOICES,
    "dense_units": DENSE_UNITS_CHOICES,
    "global_pool_type": GLOBAL_POOL_TYPE_CHOICES,
    "activation": ACTIVATION_CHOICES,
}

INTEGER_CATEGORICAL_FIELDS = frozenset(
    {
        "base_channels",
        "num_blocks",
        "kernel_time",
        "kernel_freq",
        "max_channels",
        "depth_multiplier",
        "dense_units",
    }
)
BOOLEAN_FIELDS = frozenset({"norm_flag"})
FLOAT_CATEGORICAL_FIELDS = frozenset({"channel_growth", "pointwise_scale", "dropout_rate"})
STRING_CATEGORICAL_FIELDS = frozenset({"global_pool_type", "stride_schedule", "activation"})

DEFAULT_AUDIO_DSCNN_SEED: dict[str, Any] = {
    "base_channels": 16,
    "num_blocks": 3,
    "kernel_time": 3,
    "kernel_freq": 3,
    "stride_schedule": "balanced",
    "channel_growth": 2.0,
    "max_channels": 128,
    "depth_multiplier": 1,
    "pointwise_scale": 1.0,
    "dropout_rate": 0.1,
    "norm_flag": True,
    "dense_units": 32,
    "global_pool_type": "avg",
    "activation": "relu",
}


def _as_mapping(value: Any, *, section_name: str) -> dict[str, Any]:
    """Convert a config section into a plain dictionary.

    Parameters
    ----------
    value : Any
        Dict-like or namespace-like section to normalize.
    section_name : str
        Human-readable section label used in error messages.

    Returns
    -------
    dict[str, Any]
        Plain dictionary preserving the caller-provided key order.

    Raises
    ------
    ValueError
        If ``value`` cannot be interpreted as a mapping section.
    """
    if isinstance(value, Mapping):
        return dict(value.items())
    if hasattr(value, "items"):
        return dict(value.items())
    raise ValueError(f"audio_dscnn config section '{section_name}' must be a mapping.")


def _config_mapping(config: Any) -> dict[str, Any]:
    """Validate and normalize the top-level audio model config.

    Parameters
    ----------
    config : Any
        Model-family config returned by component selection.

    Returns
    -------
    dict[str, Any]
        Plain top-level config mapping.

    Raises
    ------
    ValueError
        If the config does not match the component-selection shape.
    """
    mapping = _as_mapping(config, section_name="model")
    required = {"family", "params", "search"}
    missing = sorted(required - set(mapping))
    if missing:
        raise ValueError(f"audio_dscnn model config missing keys: {', '.join(missing)}.")
    if mapping["family"] != "audio_dscnn":
        raise ValueError("audio_dscnn model config requires family='audio_dscnn'.")
    return mapping


def _section(config: Any, section_name: str) -> dict[str, Any]:
    """Read one required model config section from the normalized config.

    Parameters
    ----------
    config : Any
        Model-family config returned by component selection.
    section_name : str
        Section name to extract.

    Returns
    -------
    dict[str, Any]
        Plain section dictionary.
    """
    mapping = _config_mapping(config)
    return _as_mapping(mapping[section_name], section_name=section_name)


def _validate_exact_choice(name: str, value: Any, choices: tuple[Any, ...]) -> None:
    """Validate one hyperparameter value against its exact allowed set.

    Parameters
    ----------
    name : str
        Hyperparameter name.
    value : Any
        Candidate value to validate.
    choices : tuple[Any, ...]
        Exact allowed values.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the value has an invalid type or is outside the allowed set.
    """
    if name in INTEGER_CATEGORICAL_FIELDS:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"audio_dscnn '{name}' must be one of {choices}.")
    elif name in BOOLEAN_FIELDS:
        if not isinstance(value, bool):
            raise ValueError(f"audio_dscnn '{name}' must be a real boolean.")
    elif name in FLOAT_CATEGORICAL_FIELDS:
        _normalize_float_choice(name, value, choices)
        return
    elif name in STRING_CATEGORICAL_FIELDS:
        if not isinstance(value, str):
            raise ValueError(f"audio_dscnn '{name}' must be one of {choices}.")
    if value not in choices:
        raise ValueError(f"audio_dscnn '{name}' must be one of {choices}; got {value!r}.")


def _normalize_float_choice(name: str, value: Any, choices: tuple[Any, ...]) -> float:
    """Normalize one float categorical value and validate its allowed value.

    Parameters
    ----------
    name : str
        Hyperparameter name.
    value : Any
        Candidate categorical value.
    choices : tuple[Any, ...]
        Allowed normalized float choices.

    Returns
    -------
    float
        Normalized float value.

    Raises
    ------
    ValueError
        If the value is not numeric or is outside the allowed set.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"audio_dscnn '{name}' must be one of {choices}.")
    normalized = float(value)
    if normalized not in choices:
        raise ValueError(f"audio_dscnn '{name}' must be one of {choices}; got {value!r}.")
    return normalized


def _normalize_search_values(name: str, values: list[Any] | tuple[Any, ...]) -> tuple[Any, ...]:
    """Normalize ordered search override values for one hyperparameter.

    Parameters
    ----------
    name : str
        Hyperparameter name.
    values : list[Any] | tuple[Any, ...]
        Ordered caller-provided categorical values.

    Returns
    -------
    tuple[Any, ...]
        Normalized values preserving caller order.
    """
    choices = AUDIO_DSCNN_SEARCH_CHOICES[name]
    if name not in FLOAT_CATEGORICAL_FIELDS:
        return tuple(values)
    return tuple(_normalize_float_choice(name, value, choices) for value in values)


def _normalize_bool_from_storage(name: str, value: Any) -> bool:
    """Normalize one persisted boolean trial value.

    Parameters
    ----------
    name : str
        Hyperparameter name.
    value : Any
        Raw persisted value.

    Returns
    -------
    bool
        Normalized Python boolean.

    Raises
    ------
    ValueError
        If the value is not a boolean or integer 0/1 storage artifact.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise ValueError(f"audio_dscnn '{name}' must decode to a real boolean.")


def resolve_stride_schedule(name: str, num_blocks: int) -> tuple[tuple[int, int], ...]:
    """Resolve a named stride schedule for a variable number of DS-CNN blocks.

    Parameters
    ----------
    name : str
        Schedule name from ``STRIDE_SCHEDULE_CHOICES``.
    num_blocks : int
        Number of repeated DS-CNN blocks in the model.

    Returns
    -------
    tuple[tuple[int, int], ...]
        Per-block ``(time_stride, freq_stride)`` pairs.

    Raises
    ------
    ValueError
        If the schedule name or block count is invalid.
    """
    if name not in STRIDE_SCHEDULE_CHOICES:
        raise ValueError(f"audio_dscnn stride_schedule must be one of {STRIDE_SCHEDULE_CHOICES}.")
    if isinstance(num_blocks, bool) or not isinstance(num_blocks, int) or num_blocks <= 0:
        raise ValueError("audio_dscnn num_blocks must be a positive integer.")

    strides = [(1, 1) for _ in range(num_blocks)]

    def set_when_present(index: int, stride: tuple[int, int]) -> None:
        """Assign one stride when its zero-based block index exists.

        Parameters
        ----------
        index : int
            Block index used to derive convolution naming and placement.
        stride : tuple[int, int]
            Temporal stride applied by the convolution block.
        """
        if 0 <= index < num_blocks:
            strides[index] = stride

    if name == "light":
        set_when_present(1, (2, 2))
    elif name == "balanced":
        set_when_present(0, (2, 2))
        set_when_present(2, (2, 2))
    elif name == "aggressive":
        set_when_present(0, (2, 2))
        set_when_present(1, (2, 2))
        set_when_present(2, (2, 2))
    elif name == "early_time":
        set_when_present(0, (2, 1))
        set_when_present(1, (2, 1))
    elif name == "early_freq":
        set_when_present(0, (1, 2))
        set_when_present(1, (1, 2))
    elif name == "late":
        set_when_present(num_blocks - 1, (2, 2))
        if num_blocks >= 4:
            set_when_present(num_blocks - 3, (2, 2))

    return tuple(strides)


def _hidden_activation(name: str, activation: str) -> ReLU:
    """Create a named hidden activation layer for the audio DS-CNN.

    Parameters
    ----------
    name : str
        Keras layer name to assign.
    activation : str
        Hidden activation choice, either ``relu`` or ``relu6``.

    Returns
    -------
    tensorflow.keras.layers.ReLU
        Configured ReLU layer.

    Raises
    ------
    ValueError
        If existing validation or execution checks fail.
    """
    if activation == "relu":
        return ReLU(name=name)
    if activation == "relu6":
        return ReLU(max_value=6.0, name=name)
    raise ValueError(f"audio_dscnn activation must be one of {ACTIVATION_CHOICES}.")


def _pointwise_filters(hparams: dict[str, Any], block_index: int) -> int:
    """Compute the pointwise output width for one DS-CNN block.

    Parameters
    ----------
    hparams : dict[str, Any]
        Validated audio DS-CNN hyperparameters.
    block_index : int
        Zero-based block index.

    Returns
    -------
    int
        Clamped pointwise ``Conv2D`` filter count.
    """
    nominal = round(hparams["base_channels"] * (hparams["channel_growth"] ** block_index))
    filters = round(nominal * hparams["pointwise_scale"])
    return min(hparams["max_channels"], max(1, filters))



class AudioDSCNNFamily(ModelFamilyABC):
    """Explicit depthwise-plus-pointwise CNN family for log-mel audio classification."""

    @property
    def name(self) -> str:
        """Return the stable registry-facing model-family name.

        Returns
        -------
        str
            Stable model-family identifier.
        """
        return "audio_dscnn"

    def validate_config(self, model_config: Any) -> None:
        """Validate audio DS-CNN model-family config.

        Parameters
        ----------
        model_config : Any
            Config subtree returned by component selection.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If family parameters or search overrides are unsupported.
        """
        params = _section(model_config, "params")
        unknown_params = sorted(set(params) - {"export_variant"})
        if unknown_params:
            raise ValueError(
                f"audio_dscnn model.params contains unsupported keys: {', '.join(unknown_params)}."
            )
        if "export_variant" in params:
            export_variant = params["export_variant"]
            if not isinstance(export_variant, str) or not export_variant.strip():
                raise ValueError("audio_dscnn model.params.export_variant must be a non-empty string.")

        search = _section(model_config, "search")
        unknown_search = sorted(set(search) - set(AUDIO_DSCNN_SEARCH_CHOICES))
        if unknown_search:
            raise ValueError(
                f"audio_dscnn model.search contains unsupported keys: {', '.join(unknown_search)}."
            )
        for name, values in search.items():
            if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
                raise ValueError(f"audio_dscnn model.search.{name} must be a non-empty list.")
            if not values:
                raise ValueError(f"audio_dscnn model.search.{name} must not be empty.")
            for value in values:
                _validate_exact_choice(name, value, AUDIO_DSCNN_SEARCH_CHOICES[name])
            normalized_values = _normalize_search_values(name, values)
            if len(set(normalized_values)) != len(normalized_values):
                raise ValueError(f"audio_dscnn model.search.{name} must not contain duplicates.")

    def _choices_for(self, name: str, config: Any) -> tuple[Any, ...]:
        """Resolve search choices for one hyperparameter.

        Parameters
        ----------
        name : str
            Hyperparameter name.
        config : Any
            Model-family config that may contain ordered search overrides.

        Returns
        -------
        tuple[Any, ...]
            Ordered categorical choices used for Optuna sampling.
        """
        search = _section(config, "search")
        if name not in search:
            return AUDIO_DSCNN_SEARCH_CHOICES[name]
        return _normalize_search_values(name, search[name])

    def sample_hparams(
        self,
        trial: Any,
        ctx: ModelBuildContext,
        config: Any,
    ) -> dict[str, Any]:
        """Sample the audio DS-CNN categorical search surface.

        Parameters
        ----------
        trial : Any
            Trial-like object exposing Optuna-compatible sampling methods.
        ctx : ModelBuildContext
            Build-time context. Used only to validate the target contract.
        config : Any
            Model-family config with optional ``model.search`` overrides.

        Returns
        -------
        dict[str, Any]
            Sampled family hyperparameters.
        """
        self.validate_config(config)
        self._validate_target_spec(ctx.target_spec)
        return {
            name: trial.suggest_categorical(name, self._choices_for(name, config))
            for name in AUDIO_DSCNN_SEARCH_CHOICES
        }

    def decode_trial_hparams(
        self,
        raw_params: dict[str, Any],
        ctx: ModelBuildContext,
        config: Any,
    ) -> dict[str, Any]:
        """Decode persisted Optuna trial parameters for audio DS-CNN.

        Parameters
        ----------
        raw_params : dict[str, Any]
            Raw persisted Optuna parameter mapping.
        ctx : ModelBuildContext
            Build-time context used for validation.
        config : Any
            Model-family config used for validation.

        Returns
        -------
        dict[str, Any]
            Normalized hyperparameters accepted by :meth:`build_model`.
        """
        decoded = dict(raw_params)
        for name in BOOLEAN_FIELDS:
            if name in decoded:
                decoded[name] = _normalize_bool_from_storage(name, decoded[name])
        for name in FLOAT_CATEGORICAL_FIELDS:
            if name in decoded:
                decoded[name] = _normalize_float_choice(name, decoded[name], AUDIO_DSCNN_SEARCH_CHOICES[name])

        self.validate_hparams(decoded, ctx, config)
        return decoded

    def default_seed_trial(
        self,
        ctx: ModelBuildContext,
        config: Any,
    ) -> dict[str, Any] | None:
        """Return the legacy audio DS-CNN seed payload.

        The main NAS driver no longer enqueues this payload for fresh studies.

        Parameters
        ----------
        ctx : ModelBuildContext
            Build-time context. Unused for the current seed payload.
        config : Any
            Model-family config. Unused for the current seed payload.

        Returns
        -------
        dict[str, Any] | None
            Raw trial parameters matching the audio sampling surface.
        """
        del ctx, config
        return dict(DEFAULT_AUDIO_DSCNN_SEED)

    def validate_hparams(
        self,
        hparams: dict[str, Any],
        ctx: ModelBuildContext,
        config: Any,
    ) -> None:
        """Validate one sampled audio DS-CNN hyperparameter payload.

        Parameters
        ----------
        hparams : dict[str, Any]
            Sampled or decoded model-family hyperparameters.
        ctx : ModelBuildContext
            Build-time context containing input and target contracts.
        config : Any
            Model-family config. Unused beyond preserving the ABC signature.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If hyperparameters or build context are invalid.
        """
        del config
        expected = set(AUDIO_DSCNN_SEARCH_CHOICES)
        actual = set(hparams)
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing:
            raise ValueError(f"audio_dscnn hyperparameters missing keys: {', '.join(missing)}.")
        if extra:
            raise ValueError(f"audio_dscnn hyperparameters contain extra keys: {', '.join(extra)}.")

        for name, choices in AUDIO_DSCNN_SEARCH_CHOICES.items():
            _validate_exact_choice(name, hparams[name], choices)
        self._validate_input_shape(ctx.input_shape)
        self._validate_target_spec(ctx.target_spec)

    def build_model(
        self,
        hparams: dict[str, Any],
        ctx: ModelBuildContext,
        config: Any,
    ) -> Model:
        """Build an uncompiled audio DS-CNN Keras model.

        Parameters
        ----------
        hparams : dict[str, Any]
            Validated audio DS-CNN hyperparameters.
        ctx : ModelBuildContext
            Build-time context containing the logical log-mel input shape.
        config : Any
            Model-family config.

        Returns
        -------
        tensorflow.keras.Model
            Uncompiled logits model with one ``class_logits`` output.
        """
        self.validate_hparams(hparams, ctx, config)
        frames, mel_bins = self._validate_input_shape(ctx.input_shape)

        inputs = Input(shape=(frames, mel_bins))
        x = Reshape((frames, mel_bins, 1))(inputs)
        strides = resolve_stride_schedule(hparams["stride_schedule"], hparams["num_blocks"])
        for block_index, stride in enumerate(strides):
            block_number = block_index + 1
            x = DepthwiseConv2D(
                kernel_size=(hparams["kernel_time"], hparams["kernel_freq"]),
                strides=stride,
                padding="same",
                depth_multiplier=hparams["depth_multiplier"],
                use_bias=not hparams["norm_flag"],
                name=f"dscnn_block_{block_number}_depthwise_conv",
            )(x)
            if hparams["norm_flag"]:
                x = BatchNormalization(name=f"dscnn_block_{block_number}_depthwise_bn")(x)
            x = _hidden_activation(
                f"dscnn_block_{block_number}_depthwise_activation",
                hparams["activation"],
            )(x)
            x = Conv2D(
                filters=_pointwise_filters(hparams, block_index),
                kernel_size=(1, 1),
                padding="same",
                use_bias=not hparams["norm_flag"],
                name=f"dscnn_block_{block_number}_pointwise_conv",
            )(x)
            if hparams["norm_flag"]:
                x = BatchNormalization(name=f"dscnn_block_{block_number}_pointwise_bn")(x)
            x = _hidden_activation(
                f"dscnn_block_{block_number}_pointwise_activation",
                hparams["activation"],
            )(x)
            if hparams["dropout_rate"] > 0.0:
                x = Dropout(hparams["dropout_rate"], name=f"dscnn_block_{block_number}_dropout")(x)

        if hparams["global_pool_type"] == "avg":
            x = GlobalAveragePooling2D(name="global_avg_pool")(x)
        else:
            x = GlobalMaxPooling2D(name="global_max_pool")(x)

        if hparams["dense_units"] > 0:
            x = Dense(hparams["dense_units"], activation=None, name="dense_hidden")(x)
            x = _hidden_activation("dense_hidden_activation", hparams["activation"])(x)
            if hparams["dropout_rate"] > 0.0:
                x = Dropout(hparams["dropout_rate"], name="dense_hidden_dropout")(x)

        outputs = Dense(10, name="class_logits")(x)
        return Model(inputs=inputs, outputs=outputs)

    def count_flops(
        self,
        model: tf.keras.Model,
        ctx: ModelBuildContext,
        config: Any,
    ) -> int:
        """Estimate FLOPs for one audio DS-CNN model.

        Parameters
        ----------
        model : tensorflow.keras.Model
            Built model to profile.
        ctx : ModelBuildContext
            Build-time context containing the logical input shape.
        config : Any
            Model-family config. Unused for FLOP counting.

        Returns
        -------
        int
            Estimated forward-pass FLOP count.
        """
        del config
        input_shape = self._validate_input_shape(ctx.input_shape)
        return count_flops_keras(model, input_shape)

    @staticmethod
    def _validate_input_shape(input_shape: tuple[int, ...] | None) -> tuple[int, int]:
        """Validate and normalize the logical audio input shape.

        Parameters
        ----------
        input_shape : tuple[int, ...] | None
            Candidate ``(frames, mel_bins)`` shape.

        Returns
        -------
        tuple[int, int]
            Normalized two-dimensional input shape.

        Raises
        ------
        ValueError
            If the shape is missing, not 2D, or contains non-positive values.
        """
        if input_shape is None or len(input_shape) != 2:
            raise ValueError("AudioDSCNNFamily requires a 2D input shape: (frames, mel_bins).")
        if any(isinstance(dim, bool) or not isinstance(dim, int) for dim in input_shape):
            raise ValueError("AudioDSCNNFamily input dimensions must be integer values.")
        frames, mel_bins = input_shape
        if frames <= 0 or mel_bins <= 0:
            raise ValueError("AudioDSCNNFamily input dimensions must be positive.")
        return frames, mel_bins

    @staticmethod
    def _validate_target_spec(target_spec: TargetSpec | None) -> None:
        """Validate the audio classification target contract.

        Parameters
        ----------
        target_spec : TargetSpec | None
            Task-owned target/output contract.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the target contract is missing or not the Phase 3 audio
            classification logits contract.
        """
        if not isinstance(target_spec, TargetSpec):
            raise ValueError("AudioDSCNNFamily requires a classification TargetSpec.")
        if target_spec.task_type != "classification":
            raise ValueError("AudioDSCNNFamily requires task_type='classification'.")
        if target_spec.output_names != ["class_logits"]:
            raise ValueError("AudioDSCNNFamily requires output_names=['class_logits'].")
        if target_spec.output_shapes != [(10,)]:
            raise ValueError("AudioDSCNNFamily requires output_shapes=[(10,)].")
        metadata = target_spec.metadata or {}
        if metadata.get("num_classes") != 10:
            raise ValueError("AudioDSCNNFamily requires metadata['num_classes'] == 10.")
        if metadata.get("label_encoding") != "class_index":
            raise ValueError("AudioDSCNNFamily requires metadata['label_encoding'] == 'class_index'.")
        if metadata.get("from_logits") is not True:
            raise ValueError("AudioDSCNNFamily requires metadata['from_logits'] is True.")
