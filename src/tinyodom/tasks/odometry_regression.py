"""Odometry regression task adapter for the modular TinyODOM pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from ..interfaces import TaskABC
from ..pipeline_types import DataSplit, DatasetBundle, EvaluationResult, FitPlan, TargetSpec, TaskMetricContract


class OdometryRegressionTask(TaskABC):
    """Two-head velocity-regression task matching the current TinyODOM behavior.

    This task exposes the stable ``"odometry_regression"`` name, a fixed
    two-output regression contract for ``velx`` and ``vely``, task-owned fit
    wiring and callbacks, and RMSE-based evaluation that preserves the legacy
    prediction ordering used by the current model stack.
    """

    def __init__(
        self,
        *,
        checkpoint_path: Path,
        early_stopping_patience: int = 40,
    ) -> None:
        """Initialize the task-owned training callback configuration.

        Parameters
        ----------
        checkpoint_path : Path
            Path stored for the task-owned ``ModelCheckpoint`` callback.
        early_stopping_patience : int, optional
            Patience used by the task-owned early stopping callback. The value
            is normalized to ``int`` during initialization.
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
            Current TinyODOM output contract: ``task_type="regression"``,
            output heads ``["velx", "vely"]``, and metadata carrying the
            bundle input shape.
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
            Metric declaration for the current odometry task, with
            ``rmse_total`` exposed as the primary metric.
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
            The current implementation compiles the model with per-head MSE
            losses for ``velx`` and ``vely`` and an ``Adam`` optimizer.
            ``task_config`` and ``target_spec`` are not currently consulted.
        """

        del task_config, target_spec
        model.compile(loss={"velx": "mse", "vely": "mse"}, optimizer=Adam())

    @staticmethod
    def _stack_target_pair(
        train_targets: dict[str, Any],
        val_targets: dict[str, Any] | None,
    ) -> list[Any]:
        """Build ordered velocity targets for one fit plan.

        Parameters
        ----------
        train_targets : dict[str, Any]
            Training target mapping.
        val_targets : dict[str, Any] | None
            Optional validation target mapping that should be concatenated into
            training targets.

        Returns
        -------
        list[Any]
            Ordered ``[velx, vely]`` target payload matching the current model
            head order.
        """

        if val_targets is None:
            return [train_targets["velx"], train_targets["vely"]]
        return [
            np.concatenate([train_targets["velx"], val_targets["velx"]], axis=0),
            np.concatenate([train_targets["vely"], val_targets["vely"]], axis=0),
        ]

    def build_fit_plan(
        self,
        bundle: DatasetBundle,
        task_config: Any,
        target_spec: TargetSpec,
        *,
        mode: Literal["search", "final"],
        combine_train_val: bool,
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
        mode : {"search", "final"}
            Runner phase requesting the fit plan.
        combine_train_val : bool
            Whether the validation split should be folded into the training
            payload.

        Returns
        -------
        FitPlan
            Task-owned fit wiring and callbacks. The current plan uses
            ``bundle.train.inputs`` as ``x`` in search mode, preserves the
            legacy ``[velx, vely]`` target ordering, and switches to
            task-owned ``"loss"`` monitoring when validation data is folded
            into training.

        Raises
        ------
        ValueError
            If ``mode`` is invalid or the dataset bundle does not contain a
            validation split when one is required.
        """

        del task_config, target_spec
        if mode not in {"search", "final"}:
            raise ValueError("OdometryRegressionTask build_fit_plan mode must be 'search' or 'final'.")

        if combine_train_val:
            # Final-fit runs intentionally fold validation into training, so
            # the task must switch monitoring to training loss and rebuild the
            # merged payload itself instead of relying on orchestration code.
            train_inputs = bundle.train.inputs
            if bundle.val is not None:
                train_inputs = np.concatenate([bundle.train.inputs, bundle.val.inputs], axis=0)
                train_targets_ordered = self._stack_target_pair(bundle.train.targets, bundle.val.targets)
            else:
                train_targets_ordered = self._stack_target_pair(bundle.train.targets, None)

            checkpoint = ModelCheckpoint(
                filepath=str(self.checkpoint_path),
                monitor="loss",
                mode="min",
                verbose=1,
                save_best_only=True,
            )
            early_stop = EarlyStopping(
                monitor="loss",
                patience=self.early_stopping_patience,
                mode="min",
                verbose=1,
                restore_best_weights=True,
            )
            return FitPlan(
                fit_kwargs={
                    "x": train_inputs,
                    "y": train_targets_ordered,
                    "shuffle": True,
                },
                callbacks=[checkpoint, early_stop],
                monitor_metric="loss",
            )

        if bundle.val is None:
            raise ValueError("OdometryRegressionTask requires a validation split when combine_train_val=False.")

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

    def history_component_keys(
        self,
        target_spec: TargetSpec,
    ) -> list[tuple[str, str]]:
        """Return the per-output loss keys emitted by the odometry model.

        Parameters
        ----------
        target_spec : TargetSpec
            Task-owned target specification.

        Returns
        -------
        list[tuple[str, str]]
            Ordered training/validation loss-key pairs for each velocity head.
        """

        del target_spec
        return [("velx_loss", "val_velx_loss"), ("vely_loss", "val_vely_loss")]

    def generate_closeout_artifacts(
        self,
        model: Any,
        dataset_bundle: DatasetBundle,
        task_config: Any,
        target_spec: TargetSpec,
        *,
        output_dir: Path,
    ) -> dict[str, Any]:
        """Return task-owned closeout artifacts for odometry runs.

        Parameters
        ----------
        model : Any
            Final trained model.
        dataset_bundle : DatasetBundle
            Dataset bundle backing the run.
        task_config : Any
            Task-local configuration subtree.
        target_spec : TargetSpec
            Task-owned target specification.
        output_dir : Path
            Directory reserved for task-owned artifacts.

        Returns
        -------
        dict[str, Any]
            Trajectory metrics and written plot paths for the built-in
            odometry task.
        """

        del task_config, target_spec
        test_split = dataset_bundle.test
        if test_split is None:
            return {}
        if not isinstance(test_split.targets, dict):
            raise ValueError(
                "OdometryRegressionTask closeout requires mapping-style test targets with 'velx' and 'vely'."
            )
        if "velx" not in test_split.targets or "vely" not in test_split.targets:
            raise ValueError(
                "OdometryRegressionTask closeout requires test targets 'velx' and 'vely'."
            )
        required_metadata = {"size_of_each", "x0", "y0"}
        missing_metadata = sorted(required_metadata - set(test_split.metadata))
        if missing_metadata:
            raise ValueError(
                "OdometryRegressionTask closeout requires odometry trajectory metadata: "
                + ", ".join(missing_metadata)
            )
        required_dataset_metadata = {"stride", "window_size", "sampling_rate_hz"}
        missing_dataset_metadata = sorted(
            key
            for key in required_dataset_metadata
            if test_split.metadata.get(key, dataset_bundle.metadata.get(key)) is None
        )
        if missing_dataset_metadata:
            raise ValueError(
                "OdometryRegressionTask closeout requires dataset metadata: "
                + ", ".join(missing_dataset_metadata)
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        stride = float(
            test_split.metadata.get(
                "stride",
                dataset_bundle.metadata.get("stride"),
            )
        )
        window_size = float(
            test_split.metadata.get(
                "window_size",
                dataset_bundle.metadata.get("window_size"),
            )
        )
        sampling_rate_hz = float(
            test_split.metadata.get(
                "sampling_rate_hz",
                dataset_bundle.metadata.get("sampling_rate_hz"),
            )
        )

        preds = model.predict(test_split.inputs)
        pred_velx = np.asarray(preds[0]).reshape(-1)
        pred_vely = np.asarray(preds[1]).reshape(-1)
        gt_velx = np.asarray(test_split.targets["velx"]).reshape(-1)
        gt_vely = np.asarray(test_split.targets["vely"]).reshape(-1)
        samples_per_window = max((window_size - stride) / stride, 1.0)

        def integrate_track(
            vx: np.ndarray,
            vy: np.ndarray,
            start_x: float,
            start_y: float,
        ) -> tuple[np.ndarray, np.ndarray]:
            """Integrate one trajectory from velocity samples."""

            xs: list[float] = []
            ys: list[float] = []
            x = float(start_x)
            y = float(start_y)
            for dx, dy in zip(vx, vy):
                x += float(dx) / samples_per_window
                y += float(dy) / samples_per_window
                xs.append(x)
                ys.append(y)
            return np.array(xs), np.array(ys)

        ate_per_traj: list[float] = []
        rte_per_traj: list[float] = []
        plot_paths: list[str] = []
        idx_start = 0
        sizes = list(test_split.metadata["size_of_each"])
        start_x_values = list(test_split.metadata["x0"])
        start_y_values = list(test_split.metadata["y0"])

        for idx, length in enumerate(sizes):
            idx_end = idx_start + int(length)
            gt_x, gt_y = integrate_track(
                gt_velx[idx_start:idx_end],
                gt_vely[idx_start:idx_end],
                float(start_x_values[idx]),
                float(start_y_values[idx]),
            )
            pred_x, pred_y = integrate_track(
                pred_velx[idx_start:idx_end],
                pred_vely[idx_start:idx_end],
                float(start_x_values[idx]),
                float(start_y_values[idx]),
            )

            # Preserve the legacy trajectory summary: ATE is mean pointwise
            # distance, while RTE uses fixed ~60 second segments.
            point_errors = np.sqrt((gt_x - pred_x) ** 2 + (gt_y - pred_y) ** 2)
            ate = float(np.mean(point_errors))
            ate_per_traj.append(ate)

            segment = max(int(60 * max(sampling_rate_hz / stride, 1.0)), 1)
            rte_segments: list[float] = []
            for start in range(0, len(gt_x) - segment, segment):
                end = start + segment - 1
                dx_gt = gt_x[end] - gt_x[start]
                dy_gt = gt_y[end] - gt_y[start]
                dx_pred = pred_x[end] - pred_x[start]
                dy_pred = pred_y[end] - pred_y[start]
                rte_segments.append(float(np.sqrt((dx_gt - dx_pred) ** 2 + (dy_gt - dy_pred) ** 2)))
            rte = float(np.median(rte_segments)) if rte_segments else float("nan")
            rte_per_traj.append(rte)

            fig, ax = plt.subplots()
            ax.plot(gt_x, gt_y, label="ground truth", linewidth=2)
            ax.plot(pred_x, pred_y, label="predicted", linewidth=2, linestyle="--")
            ax.set_xlabel("X position")
            ax.set_ylabel("Y position")
            ax.set_title(f"Trajectory {idx} (ATE={ate:.3f}, RTE={rte:.3f})")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.5)
            fig.tight_layout()
            plot_path = output_dir / f"trajectory_{idx}.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            plot_paths.append(str(plot_path))
            idx_start = idx_end

        finite_rte = [float(value) for value in rte_per_traj if np.isfinite(value)]
        metrics = {
            "ate_mean": float(np.mean(ate_per_traj)),
            "ate_median": float(np.median(ate_per_traj)),
            "ate_per_traj": ate_per_traj,
            "rte_median": float(np.median(finite_rte)) if finite_rte else float("nan"),
            "rte_per_traj": rte_per_traj,
            "plots": plot_paths,
        }
        metrics_path = output_dir / "trajectory_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        metrics["trajectory_metrics_path"] = str(metrics_path)
        return metrics

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
            Flat RMSE metrics and raw predictions. The current implementation
            uses ``model.predict(split.inputs)`` directly, interprets
            ``predictions[0]`` as ``velx`` and ``predictions[1]`` as ``vely``,
            and computes ``rmse_total`` as the sum of the two per-head RMSE
            values.
        """

        predictions = model.predict(split.inputs)
        return self.evaluate_predictions(predictions, split, task_config, target_spec)

    def evaluate_predictions(
        self,
        predictions: Any,
        split: DataSplit,
        task_config: Any,
        target_spec: TargetSpec,
    ) -> EvaluationResult:
        """Evaluate odometry RMSE metrics from normalized predictions.

        Parameters
        ----------
        predictions : Any
            Prediction payload whose first two entries are ``velx`` and
            ``vely`` outputs.
        split : DataSplit
            Split containing mapping-style ``velx`` and ``vely`` targets.
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
