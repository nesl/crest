"""Odometry regression task adapter for the modular TinyODOM pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from ..interfaces import TaskABC
from ..pipeline_types import DataSplit, DatasetBundle, EvaluationResult, FitPlan, TargetSpec, TaskMetricContract


class OdometryRegressionTask(TaskABC):
    """Two-head velocity-regression task matching the current TinyODOM behavior."""

    def __init__(self, checkpoint_path: Path, early_stopping_patience: int = 40) -> None:
        """Initialize the task-owned training callback configuration.

        Parameters
        ----------
        checkpoint_path : Path
            Path where the best model checkpoint should be written.
        early_stopping_patience : int, optional
            Patience used by the task-owned early stopping callback.
        """

        self.checkpoint_path = Path(checkpoint_path)
        self.early_stopping_patience = int(early_stopping_patience)

    @property
    def name(self) -> str:
        """Return the stable registry-facing task name.

        Returns
        -------
        str
            Stable task identifier.
        """

        return "odometry_regression"

    def build_target_spec(
        self,
        bundle: DatasetBundle,
        task_config: Any,
    ) -> TargetSpec:
        """Return the current TinyODOM two-output regression contract.

        Parameters
        ----------
        bundle : DatasetBundle
            Dataset bundle associated with this task.
        task_config : Any
            Task-local configuration subtree.

        Returns
        -------
        TargetSpec
            Current TinyODOM output contract.
        """

        del task_config
        return TargetSpec(
            task_type="regression",
            output_names=["velx", "vely"],
            output_shapes=[(1,), (1,)],
            metadata={"input_shape": bundle.input_shape},
        )

    def metric_contract(
        self,
        target_spec: TargetSpec,
        task_config: Any,
    ) -> TaskMetricContract:
        """Describe the RMSE metrics produced by this task.

        Parameters
        ----------
        target_spec : TargetSpec
            Task-owned target specification.
        task_config : Any
            Task-local configuration subtree.

        Returns
        -------
        TaskMetricContract
            Metric declaration for the current odometry task.
        """

        del target_spec, task_config
        return TaskMetricContract(
            available_metric_names={"rmse_vel_x", "rmse_vel_y", "rmse_total"},
            training_only_metric_names={"rmse_vel_x", "rmse_vel_y", "rmse_total"},
            nonnegative_metric_names={"rmse_vel_x", "rmse_vel_y", "rmse_total"},
            primary_metric_names={"rmse_total"},
        )

    def compile_model(
        self,
        model: Any,
        task_config: Any,
        target_spec: TargetSpec,
    ) -> None:
        """Compile a model using the current TinyODOM regression losses.

        Parameters
        ----------
        model : Any
            Uncompiled Keras model instance.
        task_config : Any
            Task-local configuration subtree.
        target_spec : TargetSpec
            Task-owned target specification.

        Returns
        -------
        None
        """

        del task_config, target_spec
        model.compile(loss={"velx": "mse", "vely": "mse"}, optimizer=Adam())

    def make_fit_plan(
        self,
        bundle: DatasetBundle,
        task_config: Any,
        target_spec: TargetSpec,
    ) -> FitPlan:
        """Build the current TinyODOM ``model.fit(...)`` wiring.

        Parameters
        ----------
        bundle : DatasetBundle
            Dataset bundle containing train and validation splits.
        task_config : Any
            Task-local configuration subtree.
        target_spec : TargetSpec
            Task-owned target specification.

        Returns
        -------
        FitPlan
            Task-owned fit wiring and callbacks.

        Raises
        ------
        ValueError
            If the dataset bundle does not contain a validation split.
        """

        del task_config, target_spec
        if bundle.val is None:
            raise ValueError("OdometryRegressionTask requires a validation split.")

        checkpoint = ModelCheckpoint(
            filepath=str(self.checkpoint_path),
            monitor="val_loss",
            mode="min",
            verbose=1,
            save_best_only=True,
        )
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=self.early_stopping_patience,
            mode="min",
            verbose=1,
            restore_best_weights=True,
        )

        return FitPlan(
            fit_kwargs={
                "x": bundle.train.inputs,
                "y": [bundle.train.targets["velx"], bundle.train.targets["vely"]],
                "validation_data": (
                    bundle.val.inputs,
                    [bundle.val.targets["velx"], bundle.val.targets["vely"]],
                ),
                "shuffle": True,
            },
            callbacks=[checkpoint, early_stop],
            monitor_metric="val_loss",
        )

    def evaluate(
        self,
        model: Any,
        split: DataSplit,
        task_config: Any,
        target_spec: TargetSpec,
    ) -> EvaluationResult:
        """Evaluate one split using the legacy TinyODOM RMSE semantics.

        Parameters
        ----------
        model : Any
            Compiled or already-trained model.
        split : DataSplit
            Split to evaluate.
        task_config : Any
            Task-local configuration subtree.
        target_spec : TargetSpec
            Task-owned target specification.

        Returns
        -------
        EvaluationResult
            Flat RMSE metrics and raw predictions.
        """

        del task_config, target_spec
        predictions = model.predict(split.inputs)
        # Preserve the legacy output-index contract from the current TinyODOM
        # model: predictions[0] is velx and predictions[1] is vely.
        rmse_vel_x = mean_squared_error(split.targets["velx"], predictions[0], squared=False)
        rmse_vel_y = mean_squared_error(split.targets["vely"], predictions[1], squared=False)
        return EvaluationResult(
            metrics={
                "rmse_vel_x": float(rmse_vel_x),
                "rmse_vel_y": float(rmse_vel_y),
                "rmse_total": float(rmse_vel_x + rmse_vel_y),
            },
            predictions=predictions,
        )
