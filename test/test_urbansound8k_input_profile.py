"""Tests for the UrbanSound8K HIL input-profile exporter."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT_DIR / "analysis_scripts" / "hil_noise_analysis" / "urbansound8k_input_profile.py"
SPEC = importlib.util.spec_from_file_location("urbansound8k_input_profile", SCRIPT_PATH)
urbansound8k_input_profile = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = urbansound8k_input_profile
SPEC.loader.exec_module(urbansound8k_input_profile)


class UrbanSound8KInputProfileTests(unittest.TestCase):
    """Validate cached log-mel profile generation."""

    def _write_split(self, cache_dir: Path, split: str, inputs: np.ndarray) -> None:
        """Write one minimal UrbanSound8K cache split.

        Parameters
        ----------
        cache_dir : pathlib.Path
            Cache directory receiving the split.
        split : str
            Split filename stem.
        inputs : np.ndarray
            Input tensor to store.

        Returns
        -------
        None
        """

        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(cache_dir / f"{split}.npz", inputs=inputs)

    def test_reads_cache_split_and_writes_header_constants(self) -> None:
        """The exporter should load cached tensors and emit the sketch contract."""

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            inputs = np.arange(3 * 4 * 2, dtype=np.float32).reshape(3, 4, 2)
            self._write_split(cache_dir, "train", inputs)

            loaded = urbansound8k_input_profile.load_cached_inputs(cache_dir, "train")
            profile = urbansound8k_input_profile.build_profile(
                loaded,
                real_window_count=2,
                seed=7,
            )
            header_path = Path(tmpdir) / "urbansound8k_input_data.h"
            urbansound8k_input_profile.write_header(header_path, profile)

            header = header_path.read_text(encoding="utf-8")
            self.assertIn("static const int kInputWindowSize = 4;", header)
            self.assertIn("static const int kInputChannels = 2;", header)
            self.assertIn("static const int kRealWindowCount = 2;", header)
            self.assertIn("static const float kChannelMeans[kInputChannels]", header)
            self.assertIn("static const uint8_t kChannelIsBinary[kInputChannels]", header)
            self.assertIn("static const float kRealWindows[", header)

    def test_real_window_sampling_is_deterministic(self) -> None:
        """A fixed seed should produce identical sampled windows."""

        inputs = np.arange(5 * 3 * 2, dtype=np.float32).reshape(5, 3, 2)
        first = urbansound8k_input_profile.build_profile(inputs, real_window_count=3, seed=1337)
        second = urbansound8k_input_profile.build_profile(inputs, real_window_count=3, seed=1337)

        np.testing.assert_array_equal(first.real_windows, second.real_windows)

    def test_rejects_invalid_split_and_empty_cache(self) -> None:
        """Invalid split names and empty cached inputs should fail clearly."""

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            self._write_split(cache_dir, "train", np.empty((0, 4, 2), dtype=np.float32))

            with self.assertRaisesRegex(ValueError, "Unknown UrbanSound8K split"):
                urbansound8k_input_profile.load_cached_inputs(cache_dir, "bad")
            with self.assertRaisesRegex(ValueError, "empty"):
                urbansound8k_input_profile.load_cached_inputs(cache_dir, "train")

    def test_resolves_cache_dir_from_config(self) -> None:
        """The CLI config contract should read dataset.params.cache_dir."""

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                "dataset:\n  params:\n    cache_dir: data/urbansound8k/cache\n",
                encoding="utf-8",
            )

            self.assertEqual(
                urbansound8k_input_profile.resolve_cache_dir(config_path),
                Path("data/urbansound8k/cache"),
            )


if __name__ == "__main__":
    unittest.main()
