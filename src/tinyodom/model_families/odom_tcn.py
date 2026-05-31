"""Odom TCN family implementation and family-owned helpers."""

from __future__ import annotations

import itertools
import logging
from pathlib import Path
from collections import deque
from typing import Any

import numpy as np
import tensorflow as tf
from tcn import TCN
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Input, MaxPooling1D, Reshape
from ..interfaces import ModelFamilyABC
from ..model_metrics import (
    StaticMemoryEstimate,
    count_flops_keras,
    dtype_bytes_for_quantization,
    estimate_static_memory_keras,
    layer_tensor_elements,
    tensor_shape_elements,
    unique_weight_bytes,
)
from ..pipeline_types import ModelBuildContext

logger = logging.getLogger(__name__)

MIN_TCN_LAYERS = 3
MAX_TCN_LAYERS = 8
DILATION_POOL = [1, 2, 4, 8, 16, 32, 64, 128, 256]
DILATION_CANDIDATES = [
    list(combo)
    for layer_count in range(MIN_TCN_LAYERS, MAX_TCN_LAYERS + 1)
    for combo in itertools.combinations(DILATION_POOL, layer_count)
]
DROP_RATE_CHOICES = [0.0, 0.1, 0.2, 0.3, 0.4]


def build_tinyodom_model(hyperparams: dict[str, Any]) -> Model:
    """Build the Odom TCN architecture from plain hyperparameters.

    Parameters
    ----------
    hyperparams : dict[str, Any]
        Hyperparameter mapping containing the Odom TCN build fields.

    Returns
    -------
    tensorflow.keras.Model
        Uncompiled two-head velocity model.
    """
    inputs = Input(shape=(int(hyperparams["timesteps"]), int(hyperparams["input_dim"])))
    features = TCN(
        nb_filters=int(hyperparams["nb_filters"]),
        kernel_size=int(hyperparams["kernel_size"]),
        dilations=list(hyperparams["dilations"]),
        dropout_rate=float(hyperparams["dropout_rate"]),
        use_skip_connections=bool(hyperparams["use_skip_connections"]),
        use_batch_norm=bool(hyperparams["norm_flag"]),
    )(inputs)

    # Preserve the original TinyODOM post-processing stack so the family keeps
    # the same symbolic output contract while owning the implementation locally.
    features = Reshape((int(hyperparams["nb_filters"]), 1))(features)
    features = MaxPooling1D(pool_size=2)(features)
    features = Flatten()(features)
    features = Dense(32, activation="linear", name="pre")(features)
    vel_x = Dense(1, activation="linear", name="velx")(features)
    vel_y = Dense(1, activation="linear", name="vely")(features)
    return Model(inputs=[inputs], outputs=[vel_x, vel_y])


def _iter_layers(model: tf.keras.Model) -> list[tf.keras.layers.Layer]:
    """Return all reachable layers from a Keras model graph.

    Parameters
    ----------
    model : tensorflow.keras.Model
        Model to inspect.

    Returns
    -------
    list[tensorflow.keras.layers.Layer]
        Reachable layers in breadth-first traversal order.
    """
    queue = deque([model])
    seen_nodes: set[int] = set()
    layers: list[tf.keras.layers.Layer] = []
    while queue:
        node = queue.popleft()
        node_id = id(node)
        if node_id in seen_nodes:
            continue
        seen_nodes.add(node_id)
        if isinstance(node, tf.keras.layers.Layer):
            layers.append(node)
        for child in getattr(node, "layers", []) or []:
            queue.append(child)
    return layers


def apply_combined_perturbation(
    model: tf.keras.Model,
    seed: int = 1337,
) -> tuple[int, int]:
    """Apply the TinyODOM deterministic BN/bias perturbation pass.

    Parameters
    ----------
    model : tensorflow.keras.Model
        Model whose BN statistics and non-BN biases are perturbed in place.
    seed : int, optional
        Random seed used to keep perturbation deterministic.

    Returns
    -------
    tuple[int, int]
        Number of BN layers touched and number of non-BN bias layers touched.
    """
    rng = np.random.default_rng(int(seed))
    bn_touched = 0
    bias_touched = 0
    for layer in _iter_layers(model):
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            if layer.gamma is not None:
                layer.gamma.assign(rng.uniform(0.7, 1.3, size=layer.gamma.shape).astype(np.float32))
            if layer.beta is not None:
                layer.beta.assign(rng.uniform(-0.3, 0.3, size=layer.beta.shape).astype(np.float32))
            if layer.moving_mean is not None:
                layer.moving_mean.assign(
                    rng.uniform(-0.5, 0.5, size=layer.moving_mean.shape).astype(np.float32)
                )
            if layer.moving_variance is not None:
                layer.moving_variance.assign(
                    rng.uniform(0.3, 2.0, size=layer.moving_variance.shape).astype(np.float32)
                )
            bn_touched += 1
            continue

        bias = getattr(layer, "bias", None)
        if bias is None:
            continue
        bias.assign(rng.uniform(-0.3, 0.3, size=bias.shape).astype(np.float32))
        bias_touched += 1
    return bn_touched, bias_touched


def _conv_output_channels(layer: tf.keras.layers.Layer) -> int | None:
    """Infer output channel count for a Conv1D-like TCN child layer.

    Parameters
    ----------
    layer : tensorflow.keras.layers.Layer
        Candidate TCN child layer.

    Returns
    -------
    int | None
        Output channel count when available from layer attributes or weights.
    """
    filters = getattr(layer, "filters", None)
    if filters is not None:
        return int(filters)
    weights = getattr(layer, "weights", []) or []
    if weights:
        shape = list(weights[0].shape)
        if shape:
            return int(shape[-1])
    return None


def _activation_bytes(elements: int | None, dtype_bytes: int) -> tuple[int, int]:
    """Return activation bytes and warning increment for an element count.

    Parameters
    ----------
    elements : int | None
        Activation element count.
    dtype_bytes : int
        Deployment dtype width.

    Returns
    -------
    tuple[int, int]
        Bytes and warning increment.
    """
    if elements is None:
        return 0, 1
    return int(elements) * int(dtype_bytes), 0


def _estimate_odom_tcn_static_memory(
    model: tf.keras.Model,
    *,
    quantization_mode: str,
) -> StaticMemoryEstimate:
    """Estimate static tensor traffic for OdomTCN including TCN internals.

    Parameters
    ----------
    model : tensorflow.keras.Model
        Built OdomTCN model.
    quantization_mode : str
        Deployment quantization mode used to choose scalar byte width.

    Returns
    -------
    StaticMemoryEstimate
        Static memory proxy estimate for batch size 1.
    """
    dtype_bytes = dtype_bytes_for_quantization(quantization_mode)
    seen_weights: set[int] = set()
    weight_bytes = 0
    activation_bytes = 0
    warning_count = 0

    def add_operation(
        layer: tf.keras.layers.Layer | None,
        *,
        input_elements: int | None,
        output_elements: int | None,
        inferred_shape: bool = False,
    ) -> None:
        """Accumulate one static memory operation.

        Parameters
        ----------
        layer : tensorflow.keras.layers.Layer | None
            Layer whose unique weights should be counted.
        input_elements : int | None
            Input activation element count.
        output_elements : int | None
            Output activation element count.
        inferred_shape : bool, optional
            Whether the activation shape came from architecture-aware inference.
        """
        nonlocal weight_bytes, activation_bytes, warning_count
        input_bytes, input_warning = _activation_bytes(input_elements, dtype_bytes)
        output_bytes, output_warning = _activation_bytes(output_elements, dtype_bytes)
        activation_bytes += input_bytes + output_bytes
        warning_count += input_warning + output_warning
        if inferred_shape:
            warning_count += 1
        if layer is not None:
            weight_bytes += unique_weight_bytes(
                layer,
                dtype_bytes=dtype_bytes,
                seen_weights=seen_weights,
            )

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        if type(layer).__name__ != "TCN":
            add_operation(
                layer,
                input_elements=layer_tensor_elements(layer, "input"),
                output_elements=layer_tensor_elements(layer, "output"),
            )
            continue

        input_shape = getattr(getattr(layer, "input", None), "shape", None)
        output_elements = layer_tensor_elements(layer, "output")
        residual_blocks = list(getattr(layer, "residual_blocks", []) or [])
        if input_shape is None or len(input_shape) < 3 or not residual_blocks:
            generic = estimate_static_memory_keras(model, quantization_mode=quantization_mode)
            return StaticMemoryEstimate(
                weight_bytes=generic.weight_bytes,
                activation_bytes=generic.activation_bytes,
                memory_traffic_bytes=generic.memory_traffic_bytes,
                dtype_bytes=generic.dtype_bytes,
                warning_count=generic.warning_count + 1,
            )

        timesteps = int(input_shape[1])
        current_channels = int(input_shape[2])
        for block in residual_blocks:
            residual_channels = current_channels
            block_layers = list(getattr(block, "_layers", []) or getattr(block, "layers", []) or [])
            if not block_layers:
                warning_count += 1
                continue
            for child in block_layers:
                layer_type = type(child).__name__
                if layer_type == "Conv1D":
                    output_channels = _conv_output_channels(child)
                    if output_channels is None:
                        input_elements = None
                        output_child_elements = None
                    else:
                        input_channels = (
                            residual_channels
                            if str(child.name).startswith("matching_")
                            else current_channels
                        )
                        input_elements = timesteps * input_channels
                        output_child_elements = timesteps * output_channels
                        if not str(child.name).startswith("matching_"):
                            current_channels = output_channels
                elif layer_type == "Lambda" and str(child.name).startswith("matching_"):
                    input_elements = timesteps * residual_channels
                    output_child_elements = timesteps * residual_channels
                else:
                    input_elements = timesteps * current_channels
                    output_child_elements = timesteps * current_channels
                add_operation(
                    child,
                    input_elements=input_elements,
                    output_elements=output_child_elements,
                    inferred_shape=True,
                )
            current_channels = _conv_output_channels(block_layers[0]) or current_channels

        add_operation(
            None,
            input_elements=timesteps * current_channels,
            output_elements=output_elements,
            inferred_shape=True,
        )

    for weight in model.weights:
        key = id(weight)
        if key in seen_weights:
            continue
        seen_weights.add(key)
        elements = tensor_shape_elements(weight)
        if elements is None:
            warning_count += 1
            continue
        weight_bytes += elements * dtype_bytes

    return StaticMemoryEstimate(
        weight_bytes=int(weight_bytes),
        activation_bytes=int(activation_bytes),
        memory_traffic_bytes=int(weight_bytes + activation_bytes),
        dtype_bytes=int(dtype_bytes),
        warning_count=int(warning_count),
    )



class OdomTCNFamily(ModelFamilyABC):
    """Describe tCN model family matching the current TinyODOM architecture surface.

    The family exposes the stable ``"odom_tcn"`` name, owns the TinyODOM
    search surface, model builder, FLOP estimation, persisted-trial decoding,
    and export-model materialization path.
    """

    @property
    def name(self) -> str:
        """Return the stable registry-facing model-family name.

        Returns
        -------
        str
            Stable model-family identifier.
        """
        return "odom_tcn"

    def sample_hparams(
        self,
        trial: Any,
        ctx: ModelBuildContext,
        config: Any,
    ) -> dict[str, Any]:
        """Sample the current TinyODOM NAS hyperparameter surface.

        Parameters
        ----------
        trial : Any
            Trial-like object exposing Optuna-compatible sampling methods.
        ctx : ModelBuildContext
            Build-time context. Unused in the current search surface.
        config : Any
            Model-family configuration subtree. Unused in Phase 2.

        Returns
        -------
        dict[str, Any]
            Sampled family hyperparameters without runner-owned fields.
        """
        del ctx, config
        dilations_index = trial.suggest_int("dilations_index", 0, len(DILATION_CANDIDATES) - 1)
        return {
            "nb_filters": trial.suggest_int("nb_filters", 2, 63),
            "kernel_size": trial.suggest_int("kernel_size", 2, 15),
            "dropout_rate": trial.suggest_categorical("dropout_rate", DROP_RATE_CHOICES),
            "use_skip_connections": trial.suggest_categorical("use_skip_connections", [True, False]),
            "norm_flag": trial.suggest_categorical("norm_flag", [True, False]),
            "dilations": DILATION_CANDIDATES[dilations_index],
        }

    def decode_trial_hparams(
        self,
        raw_params: dict[str, Any],
        ctx: ModelBuildContext,
        config: Any,
    ) -> dict[str, Any]:
        """Decode persisted Optuna trial parameters for the TinyODOM family.

        Parameters
        ----------
        raw_params : dict[str, Any]
            Raw persisted Optuna parameter mapping.
        ctx : ModelBuildContext
            Build-time context. Unused for the current decode logic.
        config : Any
            Model-family configuration subtree. Unused for the current decode logic.

        Returns
        -------
        dict[str, Any]
            Build-time TinyODOM hyperparameters with ``dilations`` expanded.
        """
        del ctx, config
        decoded = dict(raw_params)
        if "dilations_index" in decoded and "dilations" not in decoded:
            decoded["dilations"] = DILATION_CANDIDATES[int(decoded["dilations_index"])]
        decoded.pop("dilations_index", None)
        return decoded

    def default_seed_trial(
        self,
        ctx: ModelBuildContext,
        config: Any,
    ) -> dict[str, Any] | None:
        """Return the legacy baseline TinyODOM seed payload.

        The main NAS driver no longer enqueues this payload for fresh studies.

        Parameters
        ----------
        ctx : ModelBuildContext
            Build-time context. Unused for the current seed payload.
        config : Any
            Model-family configuration subtree. Unused for the current seed payload.

        Returns
        -------
        dict[str, Any] | None
            Raw trial parameters matching the NAS sampling surface.
        """
        del ctx, config
        return {
            "nb_filters": 10,
            "kernel_size": 12,
            "dropout_rate": 0.0,
            "use_skip_connections": False,
            "norm_flag": True,
            "dilations_index": 107,
        }

    def build_model(
        self,
        hparams: dict[str, Any],
        ctx: ModelBuildContext,
        config: Any,
    ) -> Any:
        """Build the current Odom TCN model.

        Parameters
        ----------
        hparams : dict[str, Any]
            Sampled model-family hyperparameters.
        ctx : ModelBuildContext
            Build-time context containing the input shape.
        config : Any
            Model-family configuration subtree.

        Returns
        -------
        Any
            Uncompiled Keras model returned by the family-owned builder.

        Raises
        ------
        ValueError
            If ``ctx.input_shape`` does not contain the legacy timestep and
            channel dimensions required by the current model builder.
        """
        del config
        self.validate_hparams(hparams, ctx, None)
        if ctx.input_shape is None or len(ctx.input_shape) < 2:
            raise ValueError("OdomTCNFamily requires a 2D input shape: (timesteps, input_dim).")

        build_hparams = {
            **hparams,
            "timesteps": int(ctx.input_shape[0]),
            "input_dim": int(ctx.input_shape[1]),
        }
        return build_tinyodom_model(build_hparams)

    def validate_hparams(
        self,
        hparams: dict[str, Any],
        ctx: ModelBuildContext,
        config: Any,
    ) -> None:
        """Validate required hyperparameters for the current TCN family.

        Parameters
        ----------
        hparams : dict[str, Any]
            Sampled model-family hyperparameters.
        ctx : ModelBuildContext
            Build-time context. Unused in the validation checks.
        config : Any
            Model-family configuration subtree. Unused in Phase 2.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If required hyperparameters are missing.
        """
        del ctx, config
        required = (
            "nb_filters",
            "kernel_size",
            "dropout_rate",
            "use_skip_connections",
            "norm_flag",
            "dilations",
        )
        missing = [key for key in required if key not in hparams]
        if missing:
            raise ValueError(
                f"OdomTCNFamily requires hyperparameters: {', '.join(missing)}."
            )

    def custom_objects(self) -> dict[str, Any]:
        """Return Keras custom objects required by the legacy TCN family.

        Returns
        -------
        dict[str, Any]
            Loader mapping for custom Keras layers.
        """
        return {"TCN": TCN}

    def estimate_static_memory(
        self,
        model: tf.keras.Model,
        ctx: ModelBuildContext,
        config: Any,
        *,
        quantization_mode: str,
    ) -> StaticMemoryEstimate:
        """Estimate static memory traffic for Odom TCN models.

        Parameters
        ----------
        model : tensorflow.keras.Model
            Built Odom TCN model.
        ctx : ModelBuildContext
            Build-time context. Unused because the built model carries the
            concrete TCN input shape.
        config : Any
            Model-family configuration subtree. Unused for this estimate.
        quantization_mode : str
            Deployment quantization mode used to choose scalar byte width.

        Returns
        -------
        StaticMemoryEstimate
            Static tensor memory traffic estimate for batch size 1.
        """
        del ctx, config
        return _estimate_odom_tcn_static_memory(
            model,
            quantization_mode=quantization_mode,
        )

    def count_flops(
        self,
        model: tf.keras.Model,
        ctx: ModelBuildContext,
        config: Any,
    ) -> int:
        """Estimate FLOPs for one Odom TCN model.

        Parameters
        ----------
        model : tensorflow.keras.Model
            Built model to profile.
        ctx : ModelBuildContext
            Build-time context containing the logical input shape.
        config : Any
            Model-family configuration subtree.

        Returns
        -------
        int
            Estimated forward-pass FLOP count.

        Raises
        ------
        ValueError
            If the model build context does not expose a usable logical input shape.
        """
        del config
        if ctx.input_shape is None or len(ctx.input_shape) < 2:
            raise ValueError("OdomTCNFamily requires a 2D input shape: (timesteps, input_dim).")
        return count_flops_keras(model, (int(ctx.input_shape[0]), int(ctx.input_shape[1])))

    def materialize_export_model(
        self,
        hparams: dict[str, Any],
        ctx: ModelBuildContext,
        config: Any,
        *,
        model_variant: str,
        checkpoint_path: str | Path | None = None,
    ) -> Any:
        """Materialize one TinyODOM export model variant.

        Parameters
        ----------
        hparams : dict[str, Any]
            Normalized model-family hyperparameters.
        ctx : ModelBuildContext
            Normalized build-time context.
        config : Any
            Model-family configuration subtree.
        model_variant : str
            Requested export variant name.
        checkpoint_path : str | None, optional
            Checkpoint path required by trained variants.

        Returns
        -------
        Any
            Keras model ready for export preparation. The current
            ``"approx_trained"``, ``"representative"``, and
            ``"bn_full_plus_non_bn_bias_perturbed"`` variants all reuse the
            deterministic perturbation path on a freshly built model; other
            variants defer to :class:`ModelFamilyABC`.
        """
        normalized_variant = str(model_variant).strip().lower()
        if normalized_variant in {
            "approx_trained",
            "representative",
            "bn_full_plus_non_bn_bias_perturbed",
        }:
            model = self.build_model(hparams, ctx, config)
            bn_touched, bias_touched = apply_combined_perturbation(model=model, seed=1337)
            logger.info(
                "Materialized TinyODOM export variant '%s' with deterministic perturbation "
                "(bn_layers=%s, non_bn_bias_layers=%s)",
                model_variant,
                bn_touched,
                bias_touched,
            )
            return model
        return super().materialize_export_model(
            hparams,
            ctx,
            config,
            model_variant=model_variant,
            checkpoint_path=checkpoint_path,
        )
