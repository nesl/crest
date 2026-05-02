"""Tests for the audio desktop smoke utility."""

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from addict import Dict

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
SCRIPT_DIR = ROOT_DIR / "analysis_scripts" / "audio_desktop_smoke"
for path in (SRC_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_audio_desktop_smoke import run_smoke, slice_bundle  # noqa: E402
from tinyodom.pipeline_types import DataSplit, DatasetBundle, EvaluationResult  # noqa: E402


class AudioDesktopSmokeTests(unittest.TestCase):
    """Validate the hardware-free audio desktop smoke workflow."""

    def _bundle(self) -> DatasetBundle:
        """Build a small row-aligned dataset bundle for smoke tests."""

        train = DataSplit(
            inputs=np.zeros((5, 201, 64), dtype=np.float32),
            targets=np.arange(5, dtype=np.int64),
            sample_weights=np.ones((5,), dtype=np.float32),
            metadata={"clip_id": ["a", "b", "c", "d", "e"], "split_name": "train"},
        )
        val = DataSplit(
            inputs=np.ones((4, 201, 64), dtype=np.float32),
            targets=np.arange(4, dtype=np.int64),
            metadata={"clip_id": ["v0", "v1", "v2", "v3"]},
        )
        return DatasetBundle(
            train=train,
            val=val,
            input_shape=(201, 64),
            input_dtype="float32",
            metadata={"dataset": "urbansound8k"},
        )

    def test_slice_bundle_limits_row_aligned_payloads(self) -> None:
        """Dataset slicing should preserve metadata while truncating rows."""

        sliced = slice_bundle(
            self._bundle(),
            max_train_examples=3,
            max_val_examples=2,
        )

        self.assertEqual(sliced.train.inputs.shape[0], 3)
        self.assertEqual(sliced.train.targets.tolist(), [0, 1, 2])
        self.assertEqual(sliced.train.sample_weights.tolist(), [1.0, 1.0, 1.0])
        self.assertEqual(sliced.train.metadata["clip_id"], ["a", "b", "c"])
        self.assertEqual(sliced.train.metadata["split_name"], "train")
        self.assertEqual(sliced.val.inputs.shape[0], 2)

    def test_run_smoke_writes_metrics_and_uses_output_checkpoint(self) -> None:
        """The smoke runner should wire checkpoint and metrics paths locally."""

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "smoke"
            config_path = Path(tmpdir) / "config.yaml"
            config = Dict(device=Dict(name="TEST_DEVICE"))
            bundle = self._bundle()
            model = MagicMock()
            model.fit.return_value = SimpleNamespace(history={"loss": [1.0]})
            model.save.side_effect = lambda path: Path(path).write_text("checkpoint", encoding="utf-8")
            model_family = MagicMock()
            model_family.default_seed_trial.return_value = {"base_channels": 8}
            model_family.decode_trial_hparams.return_value = {"base_channels": 8}
            model_family.build_model.return_value = model
            task = MagicMock()
            task.build_fit_plan.return_value = SimpleNamespace(
                fit_kwargs={"x": bundle.train.inputs, "y": bundle.train.targets},
                callbacks=[],
            )
            task.evaluate.return_value = EvaluationResult(metrics={"accuracy": 0.5})
            full_loaded = SimpleNamespace(
                dataset=object(),
                bundle=bundle,
                model_family=model_family,
                selection={"model_config": Dict(), "task_config": Dict()},
                model_build_context=object(),
                task=task,
                target_spec=object(),
            )
            sliced_loaded = SimpleNamespace(
                dataset=full_loaded.dataset,
                bundle=bundle,
                model_family=model_family,
                selection=full_loaded.selection,
                model_build_context=full_loaded.model_build_context,
                task=task,
                target_spec=full_loaded.target_spec,
            )
            args = argparse.Namespace(
                config=str(config_path),
                epochs=1,
                max_train_examples=3,
                max_val_examples=2,
                batch_size=2,
                output_dir=str(output_dir),
            )

            with patch("run_audio_desktop_smoke.ensure_builtin_components_registered"), patch(
                "run_audio_desktop_smoke.load_config",
                return_value=config,
            ), patch(
                "run_audio_desktop_smoke.bootstrap_pipeline",
                side_effect=[full_loaded, sliced_loaded],
            ) as bootstrap_mock:
                result = run_smoke(args)

            checkpoint_path = output_dir.resolve() / "audio_desktop_smoke.keras"
            self.assertEqual(bootstrap_mock.call_args_list[0].kwargs["checkpoint_path"], checkpoint_path)
            self.assertEqual(bootstrap_mock.call_args_list[1].kwargs["checkpoint_path"], checkpoint_path)
            self.assertEqual(result["checkpoint_path"], str(checkpoint_path))
            self.assertEqual(result["metrics"]["accuracy"], 0.5)
            self.assertTrue(checkpoint_path.is_file())
            history_path = Path(result["history_path"])
            self.assertTrue(history_path.is_file())
            self.assertEqual(json.loads(history_path.read_text(encoding="utf-8")), {"loss": [1.0]})
            metrics_path = Path(result["metrics_path"])
            self.assertTrue(metrics_path.is_file())
            json.loads(metrics_path.read_text(encoding="utf-8"))

    def test_run_smoke_missing_cache_points_to_prepare_target(self) -> None:
        """Missing cache failures should tell users how to prepare audio data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(
                config="missing.yaml",
                epochs=1,
                max_train_examples=3,
                max_val_examples=2,
                batch_size=2,
                output_dir=str(Path(tmpdir) / "unused"),
            )

            with patch("run_audio_desktop_smoke.ensure_builtin_components_registered"), patch(
                "run_audio_desktop_smoke.load_config",
                return_value=Dict(),
            ), patch(
                "run_audio_desktop_smoke.bootstrap_pipeline",
                side_effect=FileNotFoundError("cache missing"),
            ):
                with self.assertRaisesRegex(FileNotFoundError, "make prepare-audio-dataset"):
                    run_smoke(args)


if __name__ == "__main__":
    unittest.main()
