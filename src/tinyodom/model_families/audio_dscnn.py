"""Audio DS-CNN model family for cached log-mel classification inputs."""

from __future__ import annotations

from typing import Any, Mapping

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    Input,
    ReLU,
    Reshape,
    SeparableConv2D,
)
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from ..interfaces import ModelFamilyABC
from ..pipeline_types import ModelBuildContext, TargetSpec

BASE_CHANNELS_CHOICES = (8, 12, 16, 24, 32)
NUM_BLOCKS_CHOICES = (2, 3, 4)
KERNEL_TIME_CHOICES = (3, 5)
KERNEL_FREQ_CHOICES = (3, 5)
DEPTHWISE_SEPARABLE_CHOICES = (True, False)
DROPOUT_RATE_CHOICES = (0.0, 0.1, 0.2, 0.3)
NORM_FLAG_CHOICES = (True, False)
DENSE_UNITS_CHOICES = (0, 16, 32, 64)
GLOBAL_POOL_TYPE_CHOICES = ("avg", "max")
STRIDE_SCHEDULE_CHOICES = ("light", "balanced", "aggressive")

STRIDE_SCHEDULES: dict[str, tuple[tuple[int, int], ...]] = {
    "light": ((1, 1), (2, 2), (1, 1), (1, 1)),
    "balanced": ((2, 2), (1, 1), (2, 2), (1, 1)),
    "aggressive": ((2, 2), (2, 2), (2, 2), (1, 1)),
}

AUDIO_DSCNN_SEARCH_CHOICES: dict[str, tuple[Any, ...]] = {
    "base_channels": BASE_CHANNELS_CHOICES,
    "num_blocks": NUM_BLOCKS_CHOICES,
    "kernel_time": KERNEL_TIME_CHOICES,
    "kernel_freq": KERNEL_FREQ_CHOICES,
    "depthwise_separable": DEPTHWISE_SEPARABLE_CHOICES,
    "dropout_rate": DROPOUT_RATE_CHOICES,
    "norm_flag": NORM_FLAG_CHOICES,
    "dense_units": DENSE_UNITS_CHOICES,
    "global_pool_type": GLOBAL_POOL_TYPE_CHOICES,
    "stride_schedule": STRIDE_SCHEDULE_CHOICES,
}

INTEGER_CATEGORICAL_FIELDS = frozenset(
    {"base_channels", "num_blocks", "kernel_time", "kernel_freq", "dense_units"}
)
BOOLEAN_FIELDS = frozenset({"depthwise_separable", "norm_flag"})
FLOAT_CATEGORICAL_FIELDS = frozenset({"dropout_rate"})
STRING_CATEGORICAL_FIELDS = frozenset({"global_pool_type", "stride_schedule"})

DEFAULT_AUDIO_DSCNN_SEED: dict[str, Any] = {
    "base_channels": 16,
    "num_blocks": 3,
    "kernel_time": 3,
    "kernel_freq": 3,
    "depthwise_separable": True,
    "stride_schedule": "balanced",
    "dropout_rate": 0.1,
    "norm_flag": True,
    "dense_units": 32,
    "global_pool_type": "avg",
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
        if isinstance(value, bool) or not isinstance(value, float):
            raise ValueError(f"audio_dscnn '{name}' must be one of {choices}.")
    elif name in STRING_CATEGORICAL_FIELDS:
        if not isinstance(value, str):
            raise ValueError(f"audio_dscnn '{name}' must be one of {choices}.")
    if value not in choices:
        raise ValueError(f"audio_dscnn '{name}' must be one of {choices}; got {value!r}.")


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


def _count_flops(model: tf.keras.Model, input_shape: tuple[int, int]) -> int:
    """Estimate DS-CNN FLOPs from a frozen TensorFlow graph.

    Parameters
    ----------
    model : tensorflow.keras.Model
        Built Keras model to profile.
    input_shape : tuple[int, int]
        Logical ``(frames, mel_bins)`` input shape.

    Returns
    -------
    int
        Forward-pass FLOP count for batch size 1.
    """

    concrete = tf.function(model).get_concrete_function(
        tf.TensorSpec([1, *input_shape], tf.float32)
    )
    frozen = convert_variables_to_constants_v2(concrete)
    graph_def = frozen.graph.as_graph_def()
    with tf.Graph().as_default() as graph:
        tf.compat.v1.import_graph_def(graph_def, name="")
        options = (
            tf.compat.v1.profiler.ProfileOptionBuilder(
                tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            )
            .with_empty_output()
            .build()
        )
        flops = tf.compat.v1.profiler.profile(graph, options=options)
    return int(flops.total_float_ops)


class AudioDSCNNFamily(ModelFamilyABC):
    """Depthwise-separable CNN family for log-mel audio classification."""

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
            if len(set(values)) != len(values):
                raise ValueError(f"audio_dscnn model.search.{name} must not contain duplicates.")
            for value in values:
                _validate_exact_choice(name, value, AUDIO_DSCNN_SEARCH_CHOICES[name])

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
        return tuple(search[name])

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
        if "dropout_rate" in decoded:
            value = decoded["dropout_rate"]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("audio_dscnn 'dropout_rate' must decode to an allowed float.")
            decoded["dropout_rate"] = float(value)

        self.validate_hparams(decoded, ctx, config)
        return decoded

    def default_seed_trial(
        self,
        ctx: ModelBuildContext,
        config: Any,
    ) -> dict[str, Any] | None:
        """Return the built-in audio DS-CNN NAS seed trial.

        Parameters
        ----------
        ctx : ModelBuildContext
            Build-time context. Unused for the seed payload.
        config : Any
            Model-family config. Unused for the seed payload.

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
        strides = STRIDE_SCHEDULES[hparams["stride_schedule"]][: hparams["num_blocks"]]
        for block_index, stride in enumerate(strides):
            filters = min(hparams["base_channels"] * (2**block_index), 128)
            conv_cls = SeparableConv2D if hparams["depthwise_separable"] else Conv2D
            x = conv_cls(
                filters=filters,
                kernel_size=(hparams["kernel_time"], hparams["kernel_freq"]),
                strides=stride,
                padding="same",
                use_bias=not hparams["norm_flag"],
                name=f"dscnn_block_{block_index + 1}_conv",
            )(x)
            if hparams["norm_flag"]:
                x = BatchNormalization(name=f"dscnn_block_{block_index + 1}_bn")(x)
            x = ReLU(name=f"dscnn_block_{block_index + 1}_relu")(x)
            if hparams["dropout_rate"] > 0.0:
                x = Dropout(hparams["dropout_rate"], name=f"dscnn_block_{block_index + 1}_dropout")(x)

        if hparams["global_pool_type"] == "avg":
            x = GlobalAveragePooling2D(name="global_avg_pool")(x)
        else:
            x = GlobalMaxPooling2D(name="global_max_pool")(x)

        if hparams["dense_units"] > 0:
            x = Dense(hparams["dense_units"], activation="relu", name="dense_hidden")(x)
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
        return _count_flops(model, input_shape)

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
