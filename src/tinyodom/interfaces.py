from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .pipeline_types import (
    DataSplit,
    DatasetBundle,
    EvaluationResult,
    FitPlan,
    ModelBuildContext,
    TargetSpec,
    TaskMetricContract,
)

if TYPE_CHECKING:
    import tensorflow as tf


class DatasetABC(ABC):
    """Abstract dataset contract for modular pipeline implementations.

    Implementations must provide :meth:`load` and may optionally override the
    default hooks for config validation and calibration-data selection. By
    default, :attr:`name` returns the implementing class name,
    :meth:`validate_config` performs no validation, and
    :meth:`make_calibration_data` forwards ``bundle.calibration`` unchanged.
    """

    @property
    def name(self) -> str:
        """Return a human-readable dataset identifier.

        Returns
        -------
        str
            Default identifier derived from the implementing class name.
        """

        return type(self).__name__

    @abstractmethod
    def load(self, dataset_config: Any) -> DatasetBundle:
        """Load, preprocess, and normalize a dataset bundle.

        Parameters
        ----------
        dataset_config : Any
            Dataset-local configuration subtree.

        Returns
        -------
        DatasetBundle
            Normalized dataset package ready for downstream pipeline stages.
            The returned bundle is also the source object consumed by later
            dataset hooks, including the default calibration-data passthrough.
        """

    def validate_config(self, dataset_config: Any) -> None:
        """Validate dataset-local configuration.

        Parameters
        ----------
        dataset_config : Any
            Dataset-local configuration subtree.

        Returns
        -------
        None
            The default implementation intentionally performs no validation and
            accepts any config object.
        """

        del dataset_config

    def make_calibration_data(
        self,
        bundle: DatasetBundle,
        dataset_config: Any,
    ) -> DataSplit | None:
        """Return calibration data used by export-time representative sampling.

        Parameters
        ----------
        bundle : DatasetBundle
            Normalized dataset package produced by :meth:`load`.
        dataset_config : Any
            Dataset-local configuration subtree.

        Returns
        -------
        DataSplit | None
            Calibration split when one is available. The default implementation
            ignores ``dataset_config``, forwards ``bundle.calibration``
            unchanged, and may therefore return ``None``.
        """

        del dataset_config
        return bundle.calibration


class TaskABC(ABC):
    """Abstract task contract for training, evaluation, and metrics.

    Implementations define the task-owned target contract, metric declaration,
    compile behavior, fit wiring, and evaluation logic. By default,
    :attr:`name` returns the implementing class name,
    :meth:`validate_config` performs no validation, and
    :meth:`validate_model_outputs` performs no output checks.
    """

    @property
    def name(self) -> str:
        """Return a human-readable task identifier.

        Returns
        -------
        str
            Default identifier derived from the implementing class name.
        """

        return type(self).__name__

    @abstractmethod
    def build_target_spec(
        self,
        bundle: DatasetBundle,
        task_config: Any,
    ) -> TargetSpec:
        """Build the concrete target/output contract for one task.

        Parameters
        ----------
        bundle : DatasetBundle
            Normalized dataset package.
        task_config : Any
            Task-local configuration subtree.

        Returns
        -------
        TargetSpec
            Concrete target and output contract for model construction and
            validation. This is the task-owned contract returned by the task
            implementation for downstream task logic.
        """

    @abstractmethod
    def metric_contract(
        self,
        target_spec: TargetSpec,
        task_config: Any,
    ) -> TaskMetricContract:
        """Describe task-defined metric availability.

        Parameters
        ----------
        target_spec : TargetSpec
            Task-owned target specification.
        task_config : Any
            Task-local configuration subtree.

        Returns
        -------
        TaskMetricContract
            Task-defined metric declaration consumed by orchestration code.
        """

    @abstractmethod
    def compile_model(
        self,
        model: tf.keras.Model,
        task_config: Any,
        target_spec: TargetSpec,
    ) -> None:
        """Apply task-owned compile semantics to a model.

        Parameters
        ----------
        model : tf.keras.Model
            Uncompiled model returned by a model family.
        task_config : Any
            Task-local configuration subtree.
        target_spec : TargetSpec
            Task-owned target specification.

        Returns
        -------
        None
            The task implementation applies compile configuration to the passed
            model using ``task_config`` and ``target_spec``.
        """

    @abstractmethod
    def make_fit_plan(
        self,
        bundle: DatasetBundle,
        task_config: Any,
        target_spec: TargetSpec,
    ) -> FitPlan:
        """Build task-owned ``model.fit(...)`` wiring.

        Parameters
        ----------
        bundle : DatasetBundle
            Normalized dataset package.
        task_config : Any
            Task-local configuration subtree.
        target_spec : TargetSpec
            Task-owned target specification.

        Returns
        -------
        FitPlan
            Task-owned fit wiring excluding runner-injected schedule policy.
        """

    @abstractmethod
    def evaluate(
        self,
        model: tf.keras.Model,
        split: DataSplit,
        task_config: Any,
        target_spec: TargetSpec,
    ) -> EvaluationResult:
        """Evaluate a model on one split and return structured results.

        Parameters
        ----------
        model : tf.keras.Model
            Compiled or already-trained model.
        split : DataSplit
            Dataset split to evaluate.
        task_config : Any
            Task-local configuration subtree.
        target_spec : TargetSpec
            Task-owned target specification.

        Returns
        -------
        EvaluationResult
            Structured evaluation output.
        """

    def validate_config(self, task_config: Any) -> None:
        """Validate task-local configuration.

        Parameters
        ----------
        task_config : Any
            Task-local configuration subtree.

        Returns
        -------
        None
            The default implementation intentionally performs no validation and
            accepts any config object.
        """

        del task_config

    def validate_model_outputs(
        self,
        model: tf.keras.Model,
        target_spec: TargetSpec,
    ) -> None:
        """Validate model outputs against the task contract.

        Parameters
        ----------
        model : tf.keras.Model
            Built model to validate.
        target_spec : TargetSpec
            Task-owned target specification.

        Returns
        -------
        None
            The default implementation intentionally performs no checks and
            exists as an override hook for task-specific output validation.
        """

        del model, target_spec


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
            Normalized build-time context.
        config : Any
            Model-family configuration subtree.

        Returns
        -------
        int
            Model FLOP count when implemented by a concrete family or later
            shared utility.

        Raises
        ------
        NotImplementedError
            Raised by the Phase 1 default implementation because FLOP counting
            has not yet been generalized into the abstraction layer.
        """

        del model, ctx, config
        raise NotImplementedError("FLOP counting is not implemented by the Phase 1 default.")

    def supports_tflite(self) -> bool:
        """Return whether the family is intended to support TFLite export.

        Returns
        -------
        bool
            ``True`` by default. Families may override this when export is not
            supported.
        """

        return True
