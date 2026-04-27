"""Current TinyODOM TCN model family exposed through ``ModelFamilyABC``."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from addict import Dict
from tcn import TCN

from ..interfaces import ModelFamilyABC
from ..pipeline_types import ModelBuildContext
from ..model import (
    DILATION_CANDIDATES,
    DROP_RATE_CHOICES,
    apply_combined_perturbation,
    build_tinyodom_model,
)

logger = logging.getLogger(__name__)


class TinyOdomTCNFamily(ModelFamilyABC):
    """TCN model family matching the current TinyODOM architecture surface."""

    @property
    def name(self) -> str:
        """Return the stable registry-facing model-family name.

        Returns
        -------
        str
            Stable model-family identifier.
        """

        return "tinyodom_tcn"

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

    def build_model(
        self,
        hparams: dict[str, Any],
        ctx: ModelBuildContext,
        config: Any,
    ) -> Any:
        """Build the current TinyODOM TCN model through the legacy builder.

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
            Uncompiled Keras model returned by the legacy builder.

        Raises
        ------
        ValueError
            If ``ctx.input_shape`` does not contain the legacy timestep and
            channel dimensions required by the current model builder.
        """

        del config
        self.validate_hparams(hparams, ctx, None)
        if ctx.input_shape is None or len(ctx.input_shape) < 2:
            raise ValueError("TinyOdomTCNFamily requires a 2D input shape: (timesteps, input_dim).")

        legacy_payload = {
            **hparams,
            "timesteps": int(ctx.input_shape[0]),
            "input_dim": int(ctx.input_shape[1]),
        }
        # The legacy model builder still expects addict.Dict-style attribute
        # access, so Phase 2 bridges the new plain-dict contract here.
        return build_tinyodom_model(Dict(legacy_payload))

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
                f"TinyOdomTCNFamily requires hyperparameters: {', '.join(missing)}."
            )

    def custom_objects(self) -> dict[str, Any]:
        """Return Keras custom objects required by the legacy TCN family.

        Returns
        -------
        dict[str, Any]
            Loader mapping for custom Keras layers.
        """

        return {"TCN": TCN}

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
            Keras model ready for export preparation.
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
