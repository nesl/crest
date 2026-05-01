"""Tests for the UrbanSound8K cached log-mel dataset adapter."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from addict import Dict

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tinyodom.datasets.urbansound8k_common import (  # noqa: E402
    BATCH_PERIOD_MS,
    CALIBRATION_EXAMPLES,
    CACHE_VERSION,
    CENTER,
    CLASS_NAMES,
    CLIP_DURATION_S,
    COMPONENT_NAME,
    CROP_SEED,
    DATASET_NAME,
    EXPECTED_FRAMES,
    FEATURE_DTYPE,
    FEATURE_KIND,
    FOLD_SPLIT,
    HOP_LENGTH_SAMPLES,
    HOP_MS,
    LABEL_ENCODING,
    LOG_FLOOR,
    MEL_BINS,
    N_FFT,
    NORMALIZATION_EPSILON,
    POWER,
    SAMPLE_RATE_HZ,
    TARGET_NUM_SAMPLES,
    TRAIN_CROP_VARIANTS,
    WINDOW_LENGTH_SAMPLES,
    WINDOW_MS,
)
from tinyodom.datasets.urbansound8k_mel import UrbanSound8KMelDataset  # noqa: E402


def _metadata(batch_period_ms: int = BATCH_PERIOD_MS) -> dict[str, object]:
    """Build valid cache metadata for dataset tests.

    Parameters
    ----------
    batch_period_ms : int, optional
        Batch period stored in the metadata fixture.

    Returns
    -------
    dict[str, object]
        Metadata matching the Phase 1 cache contract.
    """

    return {
        "schema_version": 1,
        "dataset_name": DATASET_NAME,
        "component_name": COMPONENT_NAME,
        "cache_version": CACHE_VERSION,
        "feature_kind": FEATURE_KIND,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "clip_duration_s": CLIP_DURATION_S,
        "target_num_samples": TARGET_NUM_SAMPLES,
        "mel_bins": MEL_BINS,
        "window_ms": WINDOW_MS,
        "window_length_samples": WINDOW_LENGTH_SAMPLES,
        "hop_ms": HOP_MS,
        "hop_length_samples": HOP_LENGTH_SAMPLES,
        "n_fft": N_FFT,
        "center": CENTER,
        "power": POWER,
        "log_floor": LOG_FLOOR,
        "log_scale": "natural_log",
        "expected_frames": EXPECTED_FRAMES,
        "feature_dtype": FEATURE_DTYPE,
        "normalization": "per_mel_bin_standardization",
        "normalization_mean_shape": [MEL_BINS],
        "normalization_std_shape": [MEL_BINS],
        "normalization_mean": [0.0] * MEL_BINS,
        "normalization_std": [1.0] * MEL_BINS,
        "normalization_epsilon": NORMALIZATION_EPSILON,
        "fold_split": {key: list(value) for key, value in FOLD_SPLIT.items()},
        "batch_period_ms": batch_period_ms,
        "train_crop_variants": TRAIN_CROP_VARIANTS,
        "crop_seed": CROP_SEED,
        "calibration_examples": CALIBRATION_EXAMPLES,
        "class_names": list(CLASS_NAMES),
    }


def _split_payload(split_name: str, *, labels: np.ndarray | None = None, folds: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Build one valid split payload for dataset tests.

    Parameters
    ----------
    split_name : str
        Split name used to select valid fold IDs.
    labels : numpy.ndarray | None, optional
        Optional label override.
    folds : numpy.ndarray | None, optional
        Optional fold override.

    Returns
    -------
    dict[str, numpy.ndarray]
        `.npz` payload matching the cache schema.
    """

    expected_folds = FOLD_SPLIT["train"] if split_name == "calibration" else FOLD_SPLIT[split_name]
    row_count = 2
    return {
        "inputs": np.ones((row_count, EXPECTED_FRAMES, MEL_BINS), dtype=np.float32),
        "labels": np.asarray([0, 1], dtype=np.int64) if labels is None else labels,
        "folds": np.asarray([expected_folds[0], expected_folds[0]], dtype=np.int64) if folds is None else folds,
        "clip_ids": np.asarray([f"{split_name}_a", f"{split_name}_b"], dtype=str),
        "crop_start_sample": np.asarray([0, 10], dtype=np.int64),
        "crop_variant_index": np.asarray([0, 1 if split_name == "train" else 0], dtype=np.int64),
        "pad_before_sample": np.asarray([0, 0], dtype=np.int64),
        "pad_after_sample": np.asarray([0, 0], dtype=np.int64),
    }


def _write_cache(cache_dir: Path, *, metadata: dict[str, object] | None = None) -> None:
    """Write a complete valid cache fixture.

    Parameters
    ----------
    cache_dir : pathlib.Path
        Destination cache directory.
    metadata : dict[str, object] | None, optional
        Optional metadata override.

    Returns
    -------
    None
        Writes metadata and split `.npz` files.
    """

    cache_dir.mkdir(parents=True)
    (cache_dir / "metadata.json").write_text(
        json.dumps(_metadata() if metadata is None else metadata),
        encoding="utf-8",
    )
    for split_name in ("train", "val", "test", "calibration"):
        np.savez_compressed(cache_dir / f"{split_name}.npz", **_split_payload(split_name))


class UrbanSound8KMelDatasetTests(unittest.TestCase):
    """Validate cached UrbanSound8K loader behavior."""

    def test_loads_valid_cache_and_metadata(self) -> None:
        """A valid fixture should load into the audio DatasetBundle contract.

        Returns
        -------
        None
            Asserts loaded bundle fields and metadata.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / CACHE_VERSION
            _write_cache(cache_dir)
            dataset = UrbanSound8KMelDataset()
            bundle = dataset.load(Dict(cache_dir=str(cache_dir), batch_period_ms=BATCH_PERIOD_MS))

        self.assertEqual(dataset.name, COMPONENT_NAME)
        self.assertEqual(bundle.input_shape, (EXPECTED_FRAMES, MEL_BINS))
        self.assertEqual(bundle.input_dtype, FEATURE_DTYPE)
        self.assertEqual(bundle.metadata["dataset_name"], DATASET_NAME)
        self.assertEqual(bundle.metadata["component_name"], COMPONENT_NAME)
        self.assertEqual(bundle.metadata["class_names"], list(CLASS_NAMES))
        self.assertEqual(bundle.metadata["num_classes"], len(CLASS_NAMES))
        self.assertEqual(bundle.metadata["batch_period_ms"], float(BATCH_PERIOD_MS))
        self.assertEqual(bundle.metadata["label_encoding"], LABEL_ENCODING)
        self.assertIs(bundle.calibration, dataset.make_calibration_data(bundle, Dict()))

    def test_rejects_cache_version_param(self) -> None:
        """Phase 3 should reject the dead `cache_version` config field.

        Returns
        -------
        None
            Asserts validation fails before loading metadata.
        """

        dataset = UrbanSound8KMelDataset()

        with self.assertRaisesRegex(ValueError, "cache_version"):
            dataset.validate_config(
                Dict(cache_dir="/tmp/cache", batch_period_ms=BATCH_PERIOD_MS, cache_version=CACHE_VERSION)
            )

    def test_missing_split_file_fails(self) -> None:
        """Missing required split files should fail clearly.

        Returns
        -------
        None
            Asserts missing `.npz` files are rejected.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / CACHE_VERSION
            _write_cache(cache_dir)
            (cache_dir / "test.npz").unlink()
            with self.assertRaises(FileNotFoundError):
                UrbanSound8KMelDataset().load(
                    Dict(cache_dir=str(cache_dir), batch_period_ms=BATCH_PERIOD_MS)
                )

    def test_missing_required_array_fails(self) -> None:
        """Split files missing required arrays should fail.

        Returns
        -------
        None
            Asserts array schema validation is enforced.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / CACHE_VERSION
            _write_cache(cache_dir)
            payload = _split_payload("train")
            payload.pop("labels")
            np.savez_compressed(cache_dir / "train.npz", **payload)
            with self.assertRaisesRegex(ValueError, "missing arrays"):
                UrbanSound8KMelDataset().load(
                    Dict(cache_dir=str(cache_dir), batch_period_ms=BATCH_PERIOD_MS)
                )

    def test_invalid_input_dtype_or_shape_fails(self) -> None:
        """Invalid feature arrays should fail shape and dtype validation.

        Returns
        -------
        None
            Asserts feature tensor validation is enforced.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / CACHE_VERSION
            _write_cache(cache_dir)
            payload = _split_payload("train")
            payload["inputs"] = np.ones((2, EXPECTED_FRAMES, MEL_BINS), dtype=np.float64)
            np.savez_compressed(cache_dir / "train.npz", **payload)
            with self.assertRaisesRegex(ValueError, "inputs"):
                UrbanSound8KMelDataset().load(
                    Dict(cache_dir=str(cache_dir), batch_period_ms=BATCH_PERIOD_MS)
                )

    def test_class_name_mismatch_fails(self) -> None:
        """Metadata class order drift should be rejected.

        Returns
        -------
        None
            Asserts class mapping is locked to the shared constant.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / CACHE_VERSION
            metadata = _metadata()
            metadata["class_names"] = list(reversed(CLASS_NAMES))
            _write_cache(cache_dir, metadata=metadata)
            with self.assertRaisesRegex(ValueError, "class_names"):
                UrbanSound8KMelDataset().load(
                    Dict(cache_dir=str(cache_dir), batch_period_ms=BATCH_PERIOD_MS)
                )

    def test_calibration_examples_mismatch_fails(self) -> None:
        """Metadata calibration count drift should be rejected.

        Returns
        -------
        None
            Asserts the loader enforces the fixed cache calibration contract.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / CACHE_VERSION
            metadata = _metadata()
            metadata["calibration_examples"] = CALIBRATION_EXAMPLES + 1
            _write_cache(cache_dir, metadata=metadata)
            with self.assertRaisesRegex(ValueError, "calibration_examples"):
                UrbanSound8KMelDataset().load(
                    Dict(cache_dir=str(cache_dir), batch_period_ms=BATCH_PERIOD_MS)
                )

    def test_invalid_labels_fail(self) -> None:
        """Labels outside the UrbanSound8K class range should fail.

        Returns
        -------
        None
            Asserts class-index label validation is enforced.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / CACHE_VERSION
            _write_cache(cache_dir)
            np.savez_compressed(
                cache_dir / "train.npz",
                **_split_payload("train", labels=np.asarray([0, len(CLASS_NAMES)], dtype=np.int64)),
            )
            with self.assertRaisesRegex(ValueError, "labels"):
                UrbanSound8KMelDataset().load(
                    Dict(cache_dir=str(cache_dir), batch_period_ms=BATCH_PERIOD_MS)
                )

    def test_split_fold_violation_fails(self) -> None:
        """Split files containing unexpected folds should fail.

        Returns
        -------
        None
            Asserts split ownership is preserved from the cache contract.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / CACHE_VERSION
            _write_cache(cache_dir)
            np.savez_compressed(
                cache_dir / "val.npz",
                **_split_payload("val", folds=np.asarray([1, 1], dtype=np.int64)),
            )
            with self.assertRaisesRegex(ValueError, "folds"):
                UrbanSound8KMelDataset().load(
                    Dict(cache_dir=str(cache_dir), batch_period_ms=BATCH_PERIOD_MS)
                )

    def test_batch_period_mismatch_fails(self) -> None:
        """Config and cache batch periods must match.

        Returns
        -------
        None
            Asserts config cadence remains the source of truth.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / CACHE_VERSION
            _write_cache(cache_dir)
            with self.assertRaisesRegex(ValueError, "batch_period_ms"):
                UrbanSound8KMelDataset().load(
                    Dict(cache_dir=str(cache_dir), batch_period_ms=BATCH_PERIOD_MS + 1)
                )


if __name__ == "__main__":
    unittest.main()
