# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Searchable FER Mini-Xception family with an MIT-derived learned topology."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import tensorflow as tf

from ...interfaces import ModelFamilyABC
from ...model_metrics import count_flops_keras
from ...optimizers.search_space import SearchParam
from ...pipeline_types import ModelBuildContext, TargetSpec
from .decode import (
    DEFAULT_FER_MINI_XCEPTION_SEED,
    FER_MINI_XCEPTION_SEARCH_CHOICES,
    decode_attention_indices,
    decode_block_strides,
    decode_block_widths,
    decoded_architecture_identity,
    validate_raw_hparams,
)
from .semantics import FER_MINI_XCEPTION_SEARCH_SEMANTICS

COMPONENT_NAME = "fer_mini_xception"


def _as_mapping(value: Any, *, section_name: str) -> dict[str, Any]:
    """Normalize one mapping-like config section."""
    if isinstance(value, Mapping):
        return dict(value.items())
    items = getattr(value, "items", None)
    if callable(items):
        return dict(items())
    raise ValueError(f"FER Mini-Xception config section '{section_name}' must be a mapping.")


def _sections(config: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return normalized fixed params and search overrides."""
    mapping = _as_mapping(config, section_name="model")
    if set(mapping) != {"family", "params", "search"}:
        raise ValueError("FER Mini-Xception model config requires family, params, and search sections.")
    if mapping["family"] != COMPONENT_NAME:
        raise ValueError(f"FER Mini-Xception model config requires family='{COMPONENT_NAME}'.")
    return (
        _as_mapping(mapping["params"], section_name="params"),
        _as_mapping(mapping["search"], section_name="search"),
    )


def _positive_integer(value: Any, *, field_name: str) -> int:
    """Validate one positive integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return int(value)


def _fixed_settings(config: Any) -> dict[str, Any]:
    """Resolve fixed learned-topology and regularization settings."""
    params, _ = _sections(config)
    unknown = sorted(
        set(params)
        - {
            "export_variant",
            "stem_convolutions",
            "separable_convolutions_per_block",
            "se_reduction_ratio",
            "l2",
            "avoid_duplicate_architectures",
        }
    )
    if unknown:
        raise ValueError(f"FER Mini-Xception model.params has unsupported keys: {', '.join(unknown)}.")
    if not isinstance(params.get("avoid_duplicate_architectures", False), bool):
        raise ValueError("avoid_duplicate_architectures must be a boolean.")
    export_variant = params.get("export_variant")
    if not isinstance(export_variant, str) or not export_variant.strip():
        raise ValueError("FER Mini-Xception model.params.export_variant must be nonempty.")
    stem_convolutions = _positive_integer(
        params.get("stem_convolutions", 2), field_name="model.params.stem_convolutions"
    )
    separable = _positive_integer(
        params.get("separable_convolutions_per_block", 2),
        field_name="model.params.separable_convolutions_per_block",
    )
    if stem_convolutions != 2 or separable != 2:
        raise ValueError("FER Mini-Xception currently preserves exactly two stem and two block convolutions.")
    reduction = _positive_integer(
        params.get("se_reduction_ratio", 8), field_name="model.params.se_reduction_ratio"
    )
    raw_l2 = params.get("l2", 0.001)
    if isinstance(raw_l2, bool) or not isinstance(raw_l2, (int, float)):
        raise ValueError("model.params.l2 must be a nonnegative finite number.")
    l2_value = float(raw_l2)
    if not math.isfinite(l2_value) or l2_value < 0.0:
        raise ValueError("model.params.l2 must be a nonnegative finite number.")
    return {
        "stem_convolutions": stem_convolutions,
        "separable_convolutions_per_block": separable,
        "se_reduction_ratio": reduction,
        "l2": l2_value,
    }


def _se_attention(x: Any, *, reduction_ratio: int, prefix: str) -> Any:
    """Apply the reference squeeze-and-excitation channel gate."""
    layers = tf.keras.layers
    channels = int(x.shape[-1])
    pool = (int(x.shape[1]), int(x.shape[2]))
    pooled = layers.AveragePooling2D(pool_size=pool, name=prefix + "_avg")(x)
    gates = layers.Conv2D(
        max(1, channels // reduction_ratio),
        1,
        activation="relu",
        name=prefix + "_reduce",
    )(pooled)
    gates = layers.Conv2D(channels, 1, name=prefix + "_expand")(gates)
    gates = layers.Activation("sigmoid", name=prefix + "_channel_gate")(gates)
    return layers.Multiply(name=prefix + "_channel_multiply")([x, gates])


class FERMiniXceptionFamily(ModelFamilyABC):
    """Variable-depth five-class residual separable CNN family."""

    @property
    def name(self) -> str:
        """Return the stable registry key."""
        return COMPONENT_NAME

    def validate_config(self, model_config: Any) -> None:
        """Validate fixed settings and exact categorical search overrides."""
        _fixed_settings(model_config)
        _, search = _sections(model_config)
        unknown = sorted(set(search) - set(FER_MINI_XCEPTION_SEARCH_CHOICES) - {"resolution"})
        if unknown:
            raise ValueError(f"FER Mini-Xception model.search has unsupported keys: {', '.join(unknown)}.")
        for name, values in search.items():
            if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)) or not values:
                raise ValueError(f"model.search.{name} must be a nonempty list.")
            normalized: list[Any] = []
            for value in values:
                candidate = dict(DEFAULT_FER_MINI_XCEPTION_SEED)
                candidate[name] = value
                normalized.append(validate_raw_hparams(candidate)[name])
            if len(set(normalized)) != len(normalized):
                raise ValueError(f"model.search.{name} must not contain duplicates.")

    def _choices_for(self, name: str, config: Any) -> tuple[Any, ...]:
        """Resolve one default or overridden ordered categorical surface."""
        _, search = _sections(config)
        if name not in search:
            return FER_MINI_XCEPTION_SEARCH_CHOICES[name]
        values = []
        for value in search[name]:
            candidate = dict(DEFAULT_FER_MINI_XCEPTION_SEED)
            candidate[name] = value
            values.append(validate_raw_hparams(candidate)[name])
        return tuple(values)

    def trial_search_space(self, ctx: ModelBuildContext, config: Any) -> list[SearchParam]:
        """Describe the exact raw categorical search surface."""
        self.validate_config(config)
        self._validate_context(ctx)
        return [
            SearchParam(
                name,
                "categorical",
                choices=self._choices_for(name, config),
                **({"description": "Square input image resolution.", "units": "pixels"} if name == "resolution" else FER_MINI_XCEPTION_SEARCH_SEMANTICS[name]),
            )
            for name in (*FER_MINI_XCEPTION_SEARCH_CHOICES, *(["resolution"] if "resolution" in _sections(config)[1] else []))
        ]

    def sample_hparams(self, trial: Any, ctx: ModelBuildContext, config: Any) -> dict[str, Any]:
        """Sample one complete raw architecture payload."""
        return {param.name: param.suggest(trial) for param in self.trial_search_space(ctx, config)}

    def decode_trial_hparams(
        self,
        raw_params: dict[str, Any],
        ctx: ModelBuildContext,
        config: Any,
    ) -> dict[str, Any]:
        """Normalize persisted categorical values for model reconstruction."""
        decoded = validate_raw_hparams(raw_params)
        self.validate_hparams(decoded, ctx, config)
        return decoded

    def default_seed_trial(self, ctx: ModelBuildContext, config: Any) -> dict[str, Any] | None:
        """Return the selected 96-pixel SE architecture's raw parameters."""
        self.validate_config(config)
        self._validate_context(ctx)
        seed = dict(DEFAULT_FER_MINI_XCEPTION_SEED)
        if "resolution" in _sections(config)[1]:
            seed["resolution"] = 96 if 96 in self._choices_for("resolution", config) else self._choices_for("resolution", config)[0]
        return seed

    def validate_hparams(
        self,
        hparams: dict[str, Any],
        ctx: ModelBuildContext,
        config: Any,
    ) -> None:
        """Validate raw categories, logical input shape, and probability target."""
        validate_raw_hparams(hparams)
        self.validate_config(config)
        self._validate_context(ctx)
        if "resolution" in hparams and tuple(ctx.input_shape) != (hparams["resolution"],) * 2:
            raise ValueError("FER resolution must match the active dataset context.")

    def architecture_identity(
        self,
        hparams: dict[str, Any],
        config: Any,
    ) -> dict[str, Any]:
        """Return canonical evidence based on the decoded graph structure."""
        return decoded_architecture_identity(hparams, **_fixed_settings(config))

    def build_model(
        self,
        hparams: dict[str, Any],
        ctx: ModelBuildContext,
        config: Any,
    ) -> tf.keras.Model:
        """Build an uncompiled probability model with logical ``H,W`` input."""
        self.validate_hparams(hparams, ctx, config)
        values = validate_raw_hparams(hparams)
        settings = _fixed_settings(config)
        height, width = self._validate_input_shape(ctx.input_shape)
        block_widths = decode_block_widths(values)
        block_strides = decode_block_strides(values)
        attention_indices = set(decode_attention_indices(values))
        layers = tf.keras.layers
        regularizer = tf.keras.regularizers.l2(settings["l2"])

        inputs = layers.Input((height, width), name="face_gray")
        x = layers.Reshape((height, width, 1), name="face_gray_channel")(inputs)
        for index in range(settings["stem_convolutions"]):
            x = layers.Conv2D(
                values["stem_channels"],
                3,
                use_bias=False,
                kernel_regularizer=regularizer,
                name=f"stem_conv{index}",
            )(x)
            x = layers.BatchNormalization(name=f"stem_bn{index}")(x)
            x = layers.ReLU(name=f"stem_relu{index}")(x)

        for index, (channels, stride) in enumerate(zip(block_widths, block_strides)):
            prefix = f"block{index + 1}"
            residual = layers.Conv2D(
                channels,
                1,
                strides=stride,
                padding="same",
                use_bias=False,
                name=prefix + "_projection",
            )(x)
            residual = layers.BatchNormalization(name=prefix + "_projection_bn")(residual)
            x = layers.SeparableConv2D(
                channels,
                values["kernel_size"],
                padding="same",
                use_bias=False,
                depthwise_regularizer=regularizer,
                pointwise_regularizer=regularizer,
                name=prefix + "_sep1",
            )(x)
            x = layers.BatchNormalization(name=prefix + "_bn1")(x)
            x = layers.ReLU(name=prefix + "_relu")(x)
            x = layers.SeparableConv2D(
                channels,
                values["kernel_size"],
                padding="same",
                use_bias=False,
                depthwise_regularizer=regularizer,
                pointwise_regularizer=regularizer,
                name=prefix + "_sep2",
            )(x)
            x = layers.BatchNormalization(name=prefix + "_bn2")(x)
            x = layers.MaxPooling2D(
                3,
                strides=stride,
                padding="same",
                name=prefix + "_pool",
            )(x)
            x = layers.Add(name=prefix + "_add")([x, residual])
            if index in attention_indices:
                x = _se_attention(
                    x,
                    reduction_ratio=settings["se_reduction_ratio"],
                    prefix=prefix + "_attention",
                )

        x = layers.Conv2D(5, 3, padding="same", name="class_maps")(x)
        x = layers.AveragePooling2D(
            pool_size=(int(x.shape[1]), int(x.shape[2])), name="global_pool"
        )(x)
        x = layers.Flatten(name="flatten")(x)
        outputs = layers.Softmax(name="probabilities")(x)
        return tf.keras.Model(inputs, outputs, name="fer_mini_xception")

    def count_flops(
        self,
        model: tf.keras.Model,
        ctx: ModelBuildContext,
        config: Any,
    ) -> int:
        """Estimate forward-pass FLOPs using CREST's existing helper."""
        del config
        return count_flops_keras(model, self._validate_input_shape(ctx.input_shape))

    @staticmethod
    def _validate_input_shape(input_shape: tuple[int, ...] | None) -> tuple[int, int]:
        """Validate a supported two-dimensional logical image shape."""
        if input_shape is None or len(input_shape) != 2:
            raise ValueError("FER Mini-Xception requires logical input shape (height, width).")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in input_shape):
            raise ValueError("FER Mini-Xception input dimensions must be integers.")
        height, width = input_shape
        if height != width or height not in {64, 96}:
            raise ValueError("FER Mini-Xception supports square 64- or 96-pixel inputs.")
        return height, width

    @staticmethod
    def _validate_target_spec(target_spec: TargetSpec | None) -> None:
        """Validate the five-class probability target contract."""
        if not isinstance(target_spec, TargetSpec) or target_spec.task_type != "classification":
            raise ValueError("FER Mini-Xception requires a classification TargetSpec.")
        if target_spec.output_names != ["probabilities"] or target_spec.output_shapes != [(5,)]:
            raise ValueError("FER Mini-Xception requires one five-class probabilities output.")
        metadata = target_spec.metadata or {}
        if metadata.get("num_classes") != 5 or metadata.get("from_logits") is not False:
            raise ValueError("FER Mini-Xception requires five-class probability metadata.")

    @classmethod
    def _validate_context(cls, ctx: ModelBuildContext) -> None:
        """Validate logical image and task context together."""
        cls._validate_input_shape(ctx.input_shape)
        cls._validate_target_spec(ctx.target_spec)
