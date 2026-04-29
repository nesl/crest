"""Tests for the Phase 2 modular interfaces, registry, and payload types."""

import ast
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, sentinel, patch

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tinyodom.interfaces import DatasetABC, ModelFamilyABC, TaskABC  # noqa: E402
from tinyodom.pipeline_types import (  # noqa: E402
    DataSplit,
    DatasetBundle,
    EvaluationResult,
    FitPlan,
    ModelBuildContext,
    TargetSpec,
    TaskMetricContract,
)
from tinyodom.registry import ComponentRegistry  # noqa: E402


class DummyDataset(DatasetABC):
    """Minimal dataset implementation used for scaffolding tests."""

    def load(self, dataset_config):
        split = DataSplit(inputs=[1.0], targets=[0.0])
        return DatasetBundle(train=split, metadata={"source": dataset_config})


class DummyTask(TaskABC):
    """Minimal task implementation used for scaffolding tests."""

    def build_target_spec(self, bundle, task_config):
        """Return a tiny target spec while preserving the supplied task config."""
        del bundle
        return TargetSpec(
            task_type="dummy",
            output_names=["output"],
            output_shapes=[(1,)],
            metadata={"task_config": task_config},
        )

    def metric_contract(self, target_spec, task_config):
        """Expose a single training metric for default ABC behavior tests."""
        del target_spec, task_config
        return TaskMetricContract(
            available_metric_names={"loss"},
            training_only_metric_names={"loss"},
            primary_metric_names={"loss"},
        )

    def compile_model(self, model, task_config, target_spec):
        """Accept the compile hook without mutating the supplied model."""
        del model, task_config, target_spec

    def make_fit_plan(self, bundle, task_config, target_spec):
        """Return a minimal fit plan with one callback and monitor metric."""
        del bundle, task_config, target_spec
        return FitPlan(
            fit_kwargs={"x": [1.0], "y": [0.0]},
            callbacks=[sentinel.callback],
            monitor_metric="loss",
        )

    def evaluate(self, model, split, task_config, target_spec):
        """Return a trivial evaluation payload for ABC default wiring tests."""
        del model, split, task_config, target_spec
        return EvaluationResult(metrics={"loss": 0.0})


class DummyModelFamily(ModelFamilyABC):
    """Minimal model-family implementation used for scaffolding tests."""

    def sample_hparams(self, trial, ctx, config):
        """Return one deterministic hyperparameter payload."""
        del trial, ctx, config
        return {"width": 4}

    def build_model(self, hparams, ctx, config):
        """Return the sentinel model used by the scaffolding tests."""
        del hparams, ctx, config
        return sentinel.model


class PipelineTypesTests(unittest.TestCase):
    """Validate the generic shared payload dataclasses."""

    def test_pipeline_types_instantiate_with_minimal_payloads(self) -> None:
        # Verify that pipeline types instantiate with minimal payloads.
        split = DataSplit(inputs=[1.0], targets=[0.0])
        bundle = DatasetBundle(train=split, input_shape=(1,), input_dtype="float32")
        target_spec = TargetSpec(
            task_type="classification",
            output_names=["logits"],
            output_shapes=[(10,)],
        )
        ctx = ModelBuildContext(
            input_shape=(1,),
            input_dtype="float32",
            target_spec=target_spec,
        )
        fit_plan = FitPlan(fit_kwargs={"x": [1.0], "y": [0.0]})
        evaluation = EvaluationResult(metrics={"accuracy": 1.0})
        contract = TaskMetricContract(available_metric_names={"accuracy"})

        self.assertIs(bundle.train, split)
        self.assertEqual(ctx.target_spec.task_type, "classification")
        self.assertIn("x", fit_plan.fit_kwargs)
        self.assertEqual(evaluation.metrics["accuracy"], 1.0)
        self.assertEqual(contract.available_metric_names, {"accuracy"})

    def test_payload_dataclasses_disable_value_equality(self) -> None:
        # Verify that payload dataclasses disable value equality.
        self.assertFalse(DataSplit.__dataclass_params__.eq)
        self.assertFalse(DatasetBundle.__dataclass_params__.eq)
        self.assertFalse(TargetSpec.__dataclass_params__.eq)
        self.assertFalse(ModelBuildContext.__dataclass_params__.eq)
        self.assertFalse(FitPlan.__dataclass_params__.eq)
        self.assertFalse(EvaluationResult.__dataclass_params__.eq)
        self.assertFalse(TaskMetricContract.__dataclass_params__.eq)


class RegistryTests(unittest.TestCase):
    """Validate explicit registration and lookup behavior."""

    def test_registry_registers_and_returns_classes(self) -> None:
        # Verify that registry registers and returns classes.
        registry = ComponentRegistry[object]("component")
        registry.register("dummy", DummyDataset)

        self.assertEqual(registry["dummy"], DummyDataset)
        self.assertEqual(registry.names(), ("dummy",))
        self.assertIn("dummy", registry)

    def test_registry_rejects_duplicate_names(self) -> None:
        # Verify that registry rejects duplicate names.
        registry = ComponentRegistry[object]("component")
        registry.register("dummy", DummyDataset)

        with self.assertRaises(ValueError):
            registry.register("dummy", DummyTask)

    def test_registry_rejects_empty_or_whitespace_names(self) -> None:
        # Verify that registry rejects empty or whitespace names.
        registry = ComponentRegistry[object]("component")

        with self.assertRaises(ValueError):
            registry.register("", DummyDataset)
        with self.assertRaises(ValueError):
            registry.register("   ", DummyDataset)

    def test_registry_rejects_missing_names(self) -> None:
        # Verify that registry rejects missing names.
        registry = ComponentRegistry[object]("component")

        with self.assertRaises(KeyError):
            _ = registry["missing"]


class InterfaceDefaultsTests(unittest.TestCase):
    """Validate the generic ABC defaults defined in Phase 1."""

    def setUp(self) -> None:
        self.dataset = DummyDataset()
        self.task = DummyTask()
        self.model_family = DummyModelFamily()
        self.split = DataSplit(inputs=[1.0], targets=[0.0])
        self.bundle = DatasetBundle(train=self.split, calibration=self.split)
        self.target_spec = TargetSpec(
            task_type="dummy",
            output_names=["output"],
            output_shapes=[(1,)],
        )
        self.ctx = ModelBuildContext(
            input_shape=(1,),
            input_dtype="float32",
            target_spec=self.target_spec,
        )

    def test_default_name_property_uses_class_name(self) -> None:
        # Verify that default name property uses class name.
        self.assertEqual(self.dataset.name, "DummyDataset")
        self.assertEqual(self.task.name, "DummyTask")
        self.assertEqual(self.model_family.name, "DummyModelFamily")

    def test_dataset_default_methods_are_noops_or_passthroughs(self) -> None:
        # Verify that dataset default methods are noops or passthroughs.
        self.dataset.validate_config({"ok": True})
        self.assertIs(self.dataset.make_calibration_data(self.bundle, {}), self.split)

    def test_task_default_methods_are_noops(self) -> None:
        # Verify that task default methods are noops.
        self.task.validate_config({"ok": True})
        self.task.validate_model_outputs(sentinel.model, self.target_spec)

    def test_model_family_default_helpers_are_generic(self) -> None:
        # Verify that model family default helpers are generic.
        self.model_family.validate_config({"ok": True})
        self.model_family.validate_hparams({"width": 4}, self.ctx, {"ok": True})
        self.assertEqual(self.model_family.custom_objects(), {})
        self.assertTrue(self.model_family.supports_tflite())

    def test_default_load_model_uses_generic_keras_loader(self) -> None:
        # Verify that default load model uses generic keras loader.
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.keras"
            model_path.write_text("placeholder")
            tensorflow_module = types.ModuleType("tensorflow")
            keras_module = types.ModuleType("tensorflow.keras")
            models_module = types.ModuleType("tensorflow.keras.models")
            load_mock = MagicMock(return_value=sentinel.loaded_model)
            models_module.load_model = load_mock
            keras_module.models = models_module
            tensorflow_module.keras = keras_module

            with patch.dict(
                sys.modules,
                {
                    "tensorflow": tensorflow_module,
                    "tensorflow.keras": keras_module,
                    "tensorflow.keras.models": models_module,
                },
            ):
                loaded_model = self.model_family.load_model(model_path, self.ctx, {})

        load_mock.assert_called_once_with(str(model_path), custom_objects={})
        self.assertIs(loaded_model, sentinel.loaded_model)

    def test_default_materialize_export_model_routes_trained_variant_through_load_model(self) -> None:
        # Verify that default materialize export model routes trained variant through load model.
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "trained.keras"
            checkpoint_path.write_text("placeholder")

            with patch.object(self.model_family, "load_model", return_value=sentinel.loaded_model) as load_mock:
                loaded_model = self.model_family.materialize_export_model(
                    {"width": 4},
                    self.ctx,
                    {},
                    model_variant="trained",
                    checkpoint_path=checkpoint_path,
                )

        load_mock.assert_called_once_with(checkpoint_path, self.ctx, {})
        self.assertIs(loaded_model, sentinel.loaded_model)

    def test_default_materialize_export_model_rejects_missing_trained_checkpoint(self) -> None:
        # Verify that default materialize export model rejects missing trained checkpoint.
        missing_checkpoint = Path("/tmp/does-not-exist-trained.keras")

        with self.assertRaises(FileNotFoundError):
            self.model_family.materialize_export_model(
                {"width": 4},
                self.ctx,
                {},
                model_variant="trained",
                checkpoint_path=missing_checkpoint,
            )

    def test_default_count_flops_raises_not_implemented(self) -> None:
        # Verify that default count flops raises not implemented.
        with self.assertRaises(NotImplementedError):
            self.model_family.count_flops(sentinel.model, self.ctx, {})


class PurityTests(unittest.TestCase):
    """Ensure new Phase 1 modules stay decoupled from concrete runtime code."""

    def test_new_modules_do_not_import_forbidden_runtime_modules(self) -> None:
        # Verify that new modules do not import forbidden runtime modules.
        forbidden_modules = {
            "tinyodom.data",
            "tinyodom.model",
            "nas_model_client",
            "hil_server",
        }
        module_paths = [
            SRC_DIR / "tinyodom" / "interfaces.py",
            SRC_DIR / "tinyodom" / "pipeline_types.py",
            SRC_DIR / "tinyodom" / "registry.py",
        ]

        for module_path in module_paths:
            with self.subTest(module=module_path.name):
                tree = ast.parse(module_path.read_text(), filename=str(module_path))
                imported_names: set[str] = set()

                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imported_names.add(alias.name)
                    elif isinstance(node, ast.ImportFrom) and node.module is not None:
                        imported_names.add(node.module)

                self.assertTrue(imported_names.isdisjoint(forbidden_modules))
