# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Architecture sampling, model construction, and export contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..pipeline_types import ModelBuildContext

if TYPE_CHECKING:
    import tensorflow as tf

    from ..model_metrics import StaticMemoryEstimate
    from ..optimizers.search_space import SearchParam


class ModelFamilyABC(ABC):
    """Abstract model-family contract for sampling and model construction.

    Implementations must provide hyperparameter sampling and model
    construction. The base class also exposes optional helper hooks for
    validation, model loading, export-model materialization, custom object
    registration, FLOP counting, and export-capability reporting. By default,
    :attr:`name` returns the implementing class name.
    """

    @property
    def name(self) -> str:
        """Return a human-readable model-family identifier.

        Returns
        -------
        str
            Default identifier derived from the implementing class name.
        """
        return type(self).__name__

    @abstractmethod
    def sample_hparams(
        self,
        trial: Any,
        ctx: ModelBuildContext,
        config: Any,
    ) -> dict[str, Any]:
        """Sample normalized hyperparameters for one trial.

        Parameters
        ----------
        trial : Any
            Trial-like sampling object.
        ctx : ModelBuildContext
            Normalized build-time context.
        config : Any
            Model-family configuration subtree.

        Returns
        -------
        dict[str, Any]
            Normalized hyperparameter dictionary produced by the model-family
            implementation.
        """
    def trial_search_space(
        self,
        ctx: ModelBuildContext,
        config: Any,
    ) -> list[SearchParam]:
        """Describe the raw Optuna parameters sampled by this family.

        Model families that support declarative candidate generation override
        this hook. It is intentionally non-abstract so existing third-party
        families remain instantiable until they opt into the descriptor seam.

        Parameters
        ----------
        ctx : ModelBuildContext
            Normalized model-build context.
        config : Any
            Model-family configuration subtree.

        Returns
        -------
        list[SearchParam]
            Ordered raw parameters sampled by the family.

        Raises
        ------
        NotImplementedError
            If the family does not implement declarative candidate generation.
        """
        del ctx, config
        raise NotImplementedError(
            f"Model family '{self.name}' does not expose a trial search-space descriptor."
        )
    @abstractmethod
    def build_model(
        self,
        hparams: dict[str, Any],
        ctx: ModelBuildContext,
        config: Any,
    ) -> tf.keras.Model:
        """Construct an uncompiled model for one hyperparameter sample.

        Parameters
        ----------
        hparams : dict[str, Any]
            Normalized hyperparameter dictionary.
        ctx : ModelBuildContext
            Normalized build-time context.
        config : Any
            Model-family configuration subtree.

        Returns
        -------
        tf.keras.Model
            Model instance constructed for the sampled hyperparameters and
            build context.
        """
    def validate_config(self, model_config: Any) -> None:
        """Validate model-family configuration.

        Parameters
        ----------
        model_config : Any
            Model-family configuration subtree.

        Returns
        -------
        None
            The default implementation intentionally performs no validation and
            accepts any config object.
        """
        del model_config

    def validate_hparams(
        self,
        hparams: dict[str, Any],
        ctx: ModelBuildContext,
        config: Any,
    ) -> None:
        """Validate one sampled hyperparameter dictionary.

        Parameters
        ----------
        hparams : dict[str, Any]
            Normalized hyperparameter dictionary.
        ctx : ModelBuildContext
            Normalized build-time context.
        config : Any
            Model-family configuration subtree.

        Returns
        -------
        None
            The default implementation intentionally performs no
            hyperparameter checks.
        """
        del hparams, ctx, config

    def decode_trial_hparams(
        self,
        raw_params: dict[str, Any],
        ctx: ModelBuildContext,
        config: Any,
    ) -> dict[str, Any]:
        """Decode persisted trial parameters into build-time hyperparameters.

        Parameters
        ----------
        raw_params : dict[str, Any]
            Raw parameter mapping persisted by the NAS runner.
        ctx : ModelBuildContext
            Normalized build-time context.
        config : Any
            Model-family configuration subtree.

        Returns
        -------
        dict[str, Any]
            Model-family hyperparameters accepted by :meth:`build_model`.

        Notes
        -----
        The default implementation returns ``raw_params`` unchanged so
        families only need to override this hook when their persisted trial
        shape differs from their build-time hyperparameter shape.
        """
        del ctx, config
        return dict(raw_params)

    def default_seed_trial(
        self,
        ctx: ModelBuildContext,
        config: Any,
    ) -> dict[str, Any] | None:
        """Return a legacy default seed trial payload.

        The main NAS driver does not enqueue this hook for fresh studies.
        It remains on the interface for compatibility with older callers and
        persisted model-family implementations.

        Parameters
        ----------
        ctx : ModelBuildContext
            Normalized build-time context.
        config : Any
            Model-family configuration subtree.

        Returns
        -------
        dict[str, Any] | None
            Raw persisted-trial parameter mapping, or ``None`` when the family
            does not define a legacy seed payload.
        """
        del ctx, config
        return None

    def load_model(
        self,
        path: str | Path,
        ctx: ModelBuildContext,
        config: Any,
    ) -> tf.keras.Model:
        """Load a persisted model for this family.

        Parameters
        ----------
        path : str | Path
            Path to the persisted model artifact.
        ctx : ModelBuildContext
            Normalized build-time context.
        config : Any
            Model-family configuration subtree.

        Returns
        -------
        tf.keras.Model
            Loaded model instance using generic Keras loading semantics. The
            default implementation ignores ``ctx`` and ``config`` and passes
            :meth:`custom_objects` through to the Keras loader.
        """
        del ctx, config

        from tensorflow.keras.models import load_model as keras_load_model

        return keras_load_model(str(path), custom_objects=self.custom_objects())

    def materialize_export_model(
        self,
        hparams: dict[str, Any],
        ctx: ModelBuildContext,
        config: Any,
        *,
        model_variant: str,
        checkpoint_path: str | Path | None = None,
    ) -> tf.keras.Model:
        """Materialize the model variant used for export-oriented workflows.

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
        checkpoint_path : str | Path | None, optional
            Checkpoint path required by variants that load persisted models.

        Returns
        -------
        tf.keras.Model
            Model instance ready for export preparation. The default
            implementation builds a fresh model for ``"untrained"`` and routes
            variants whose normalized names start with ``"trained"`` through
            :meth:`load_model`.

        Raises
        ------
        FileNotFoundError
            If a trained variant points at a checkpoint path that does not
            exist on disk.
        ValueError
            If ``model_variant`` is unsupported or if a trained variant is
            requested without ``checkpoint_path``.
        """
        normalized_variant = str(model_variant).strip().lower()
        if normalized_variant == "untrained":
            return self.build_model(hparams, ctx, config)
        if normalized_variant.startswith("trained"):
            if checkpoint_path is None:
                raise ValueError(
                    f"model_variant '{model_variant}' requires checkpoint_path to be provided."
                )
            checkpoint = Path(checkpoint_path)
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"Checkpoint file not found for model_variant '{model_variant}': {checkpoint}"
                )
            return self.load_model(checkpoint, ctx, config)
        raise ValueError(
            f"Unsupported model_variant '{model_variant}'. "
            "Use 'untrained' or a variant that starts with 'trained'."
        )

    def custom_objects(self) -> dict[str, Any]:
        """Return custom objects required for model loading.

        Returns
        -------
        dict[str, Any]
            Mapping of custom object names to runtime symbols. The default
            implementation returns an empty dictionary.
        """
        return {}

    def estimate_static_memory(
        self,
        model: tf.keras.Model,
        ctx: ModelBuildContext,
        config: Any,
        *,
        quantization_mode: str,
    ) -> StaticMemoryEstimate:
        """Estimate static tensor memory traffic for one built model.

        Parameters
        ----------
        model : tensorflow.keras.Model
            Built model to inspect.
        ctx : ModelBuildContext
            Normalized build-time context. The default implementation does not
            need this value, but model families may use it for custom layers.
        config : Any
            Model-family configuration subtree. The default implementation does
            not need this value.
        quantization_mode : str
            Deployment quantization mode used to choose scalar byte width.

        Returns
        -------
        StaticMemoryEstimate
            Static proxy estimate for batch size 1.
        """
        del ctx, config
        from ..model_metrics import estimate_static_memory_keras

        return estimate_static_memory_keras(
            model,
            quantization_mode=quantization_mode,
        )

    def count_flops(
        self,
        model: tf.keras.Model,
        ctx: ModelBuildContext,
        config: Any,
    ) -> int:
        """Estimate model FLOPs for one built model.

        Parameters
        ----------
        model : tf.keras.Model
            Built model to profile.
        ctx : ModelBuildContext
            Normalized build-time context containing the logical input shape.
        config : Any
            Model-family configuration subtree. The default implementation does
            not need this value.

        Returns
        -------
        int
            Static graph FLOP proxy for a batch-size-1 forward pass.

        Raises
        ------
        ValueError
            If ``ctx.input_shape`` is missing or empty.
        """
        del config
        if ctx.input_shape is None or len(ctx.input_shape) == 0:
            raise ValueError("ModelFamilyABC requires a non-empty input shape to count FLOPs.")
        from ..model_metrics import count_flops_keras

        return count_flops_keras(
            model,
            tuple(int(dim) for dim in ctx.input_shape),
        )

    def supports_tflite(self) -> bool:
        """Return whether the family is intended to support TFLite export.

        Returns
        -------
        bool
            ``True`` by default. Families may override this when export is not
            supported.
        """
        return True
