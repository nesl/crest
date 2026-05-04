"""Unit tests for the UrbanSound8K preparation workflow."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREPARE_PATH = ROOT / "data" / "dataset_download_and_splits" / "urbansound8k" / "prepare_urbansound8k.py"
PREPARE_SPEC = importlib.util.spec_from_file_location("prepare_urbansound8k", PREPARE_PATH)
prepare_urbansound8k = importlib.util.module_from_spec(PREPARE_SPEC)
assert PREPARE_SPEC.loader is not None
sys.modules[PREPARE_SPEC.name] = prepare_urbansound8k
PREPARE_SPEC.loader.exec_module(prepare_urbansound8k)


def _fake_feature_fn(waveform: np.ndarray, _sample_rate: int) -> np.ndarray:
    """Return deterministic feature rows keyed by waveform contents.

    Parameters
    ----------
    waveform : numpy.ndarray
        Fixed-length waveform passed by the preparation code.
    _sample_rate : int
        Ignored sample rate parameter.

    Returns
    -------
    numpy.ndarray
        Feature array with the audio contract shape.
    """

    base = float(np.mean(waveform, dtype=np.float64))
    mel_offsets = np.arange(prepare_urbansound8k.MEL_BINS, dtype=np.float32) / 100.0
    frame_offsets = np.arange(prepare_urbansound8k.EXPECTED_FRAMES, dtype=np.float32)[:, None] / 1000.0
    return (base + frame_offsets + mel_offsets).astype(np.float32)


def _constant_feature_fn(_waveform: np.ndarray, _sample_rate: int) -> np.ndarray:
    """Return constant features for zero-standard-deviation tests.

    Parameters
    ----------
    _waveform : numpy.ndarray
        Ignored waveform input.
    _sample_rate : int
        Ignored sample rate input.

    Returns
    -------
    numpy.ndarray
        Constant feature array with the audio contract shape.
    """

    return np.ones(
        (prepare_urbansound8k.EXPECTED_FRAMES, prepare_urbansound8k.MEL_BINS),
        dtype=np.float32,
    )


def _record(clip_id: str, fold: int, class_id: int, length: int = 40000) -> object:
    """Create a deterministic clip record for cache tests.

    Parameters
    ----------
    clip_id : str
        Clip identifier.
    fold : int
        UrbanSound8K fold number.
    class_id : int
        UrbanSound8K class ID.
    length : int, optional
        Number of waveform samples.

    Returns
    -------
    object
        `ClipRecord` instance from the preparation module.
    """

    waveform = np.full(length, fill_value=class_id + fold / 100.0, dtype=np.float32)
    return prepare_urbansound8k.ClipRecord(
        clip_id=clip_id,
        fold=fold,
        class_id=class_id,
        class_label=prepare_urbansound8k.CLASS_NAMES[class_id],
        waveform=waveform,
        sample_rate=prepare_urbansound8k.SAMPLE_RATE_HZ,
    )


class UrbanSound8KPrepareTests(unittest.TestCase):
    """Validate deterministic UrbanSound8K cache preparation helpers."""

    def test_config_from_args_uses_phase_defaults(self) -> None:
        """config_from_args should preserve the Phase 1 CLI defaults."""

        args = SimpleNamespace(
            raw_root="data/urbansound8k/raw",
            cache_root="data/urbansound8k/cache",
            cache_version=prepare_urbansound8k.CACHE_VERSION,
            batch_period_ms=2000,
            calibration_examples=100,
            train_crop_variants=3,
            crop_seed=1337,
            fold_rotation=False,
            fold_rotation_test_folds="1,2,3",
        )

        config = prepare_urbansound8k.config_from_args(args)

        self.assertEqual(config.cache_version, prepare_urbansound8k.CACHE_VERSION)
        self.assertEqual(config.batch_period_ms, 2000)
        self.assertEqual(config.train_crop_variants, 3)
        self.assertEqual(config.crop_seed, 1337)

    def test_parse_test_folds_validates_unique_urban_sound_folds(self) -> None:
        """Fold-list parsing should reject malformed rotation requests."""

        self.assertEqual(prepare_urbansound8k.parse_test_folds("1,2,10"), (1, 2, 10))
        with self.assertRaisesRegex(ValueError, "unique"):
            prepare_urbansound8k.parse_test_folds("1,1")
        with self.assertRaisesRegex(ValueError, "1 to 10"):
            prepare_urbansound8k.parse_test_folds("11")

    def test_class_names_and_fold_split_match_contract(self) -> None:
        """Shared constants should freeze class order and NAS fold split."""

        self.assertEqual(prepare_urbansound8k.CLASS_NAMES[0], "air_conditioner")
        self.assertEqual(prepare_urbansound8k.CLASS_NAMES[-1], "street_music")
        self.assertEqual(prepare_urbansound8k.FOLD_SPLIT["train"], (1, 2, 3, 4, 5, 6, 7, 8))
        self.assertEqual(prepare_urbansound8k.FOLD_SPLIT["val"], (9,))
        self.assertEqual(prepare_urbansound8k.FOLD_SPLIT["test"], (10,))

    def test_crop_or_pad_records_padding_for_all_lengths(self) -> None:
        """crop_or_pad_waveform should emit uniform padding metadata fields."""

        long_wave = np.arange(40000, dtype=np.float32)
        exact_wave = np.arange(prepare_urbansound8k.TARGET_NUM_SAMPLES, dtype=np.float32)
        short_wave = np.arange(100, dtype=np.float32)

        long_crop = prepare_urbansound8k.crop_or_pad_waveform(long_wave, 10)
        exact_crop = prepare_urbansound8k.crop_or_pad_waveform(exact_wave, 0)
        short_crop = prepare_urbansound8k.crop_or_pad_waveform(short_wave, 0)

        self.assertEqual(long_crop.crop_start_sample, 10)
        self.assertEqual(long_crop.pad_before_sample, 0)
        self.assertEqual(long_crop.pad_after_sample, 0)
        self.assertEqual(exact_crop.pad_before_sample, 0)
        self.assertEqual(exact_crop.pad_after_sample, 0)
        self.assertEqual(short_crop.waveform.shape[0], prepare_urbansound8k.TARGET_NUM_SAMPLES)
        self.assertEqual(short_crop.crop_start_sample, 0)
        self.assertEqual(short_crop.pad_before_sample + short_crop.pad_after_sample, prepare_urbansound8k.TARGET_NUM_SAMPLES - 100)

    def test_to_mono_float32_averages_2d_waveforms(self) -> None:
        """to_mono_float32 should average multichannel waveforms."""

        waveform = np.asarray([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)

        mono = prepare_urbansound8k.to_mono_float32(waveform)

        self.assertEqual(mono.dtype, np.float32)
        self.assertTrue(np.allclose(mono, [2.0, 3.0]))

    def test_stable_clip_seed_uses_sha256_not_python_hash(self) -> None:
        """stable_clip_seed should use hashlib.sha256 for process-stable seeds."""

        source = inspect.getsource(prepare_urbansound8k.stable_clip_seed)

        self.assertIn("hashlib.sha256", source)
        self.assertNotIn("hash(", source)
        self.assertEqual(
            prepare_urbansound8k.stable_clip_seed(1337, "clip-a"),
            prepare_urbansound8k.stable_clip_seed(1337, "clip-a"),
        )

    def test_train_crop_starts_are_order_independent(self) -> None:
        """Train crop starts should depend on clip ID, not iteration order."""

        config = prepare_urbansound8k.PrepConfig(
            raw_root=Path("raw"),
            cache_root=Path("cache"),
        )

        starts_a = prepare_urbansound8k.train_crop_starts("clip-a", 40000, config)
        starts_b = prepare_urbansound8k.train_crop_starts("clip-b", 40000, config)

        self.assertEqual(starts_a, prepare_urbansound8k.train_crop_starts("clip-a", 40000, config))
        self.assertEqual(starts_b, prepare_urbansound8k.train_crop_starts("clip-b", 40000, config))
        self.assertEqual(len(starts_a), config.train_crop_variants)

    def test_compute_log_mel_features_uses_natural_log_floor(self) -> None:
        """compute_log_mel_features should use natural log, not dB scaling."""

        mel_power = np.ones(
            (prepare_urbansound8k.MEL_BINS, prepare_urbansound8k.EXPECTED_FRAMES),
            dtype=np.float32,
        )
        mel_power[0, 0] = 0.0
        fake_librosa = SimpleNamespace(
            feature=SimpleNamespace(melspectrogram=Mock(return_value=mel_power))
        )

        with patch.object(prepare_urbansound8k, "require_librosa", return_value=fake_librosa):
            features = prepare_urbansound8k.compute_log_mel_features(
                np.ones(prepare_urbansound8k.TARGET_NUM_SAMPLES, dtype=np.float32),
                prepare_urbansound8k.SAMPLE_RATE_HZ,
            )

        self.assertEqual(features.shape, (prepare_urbansound8k.EXPECTED_FRAMES, prepare_urbansound8k.MEL_BINS))
        self.assertEqual(features.dtype, np.float32)
        self.assertAlmostEqual(float(features[0, 0]), float(np.log(prepare_urbansound8k.LOG_FLOOR)), places=5)
        self.assertAlmostEqual(float(features[0, 1]), 0.0)

    def test_normalization_uses_expanded_train_rows_and_epsilon(self) -> None:
        """Normalization should use all train variants and handle zero std."""

        train_inputs = np.stack(
            [
                np.full((prepare_urbansound8k.EXPECTED_FRAMES, prepare_urbansound8k.MEL_BINS), 1.0, dtype=np.float32),
                np.full((prepare_urbansound8k.EXPECTED_FRAMES, prepare_urbansound8k.MEL_BINS), 3.0, dtype=np.float32),
            ],
            axis=0,
        )

        mean, std = prepare_urbansound8k.compute_normalization_stats(train_inputs)
        normalized = prepare_urbansound8k.normalize_features(_constant_feature_fn(np.array([]), 0), mean, np.zeros_like(std))

        self.assertEqual(mean.shape, (prepare_urbansound8k.MEL_BINS,))
        self.assertTrue(np.allclose(mean, 2.0))
        self.assertTrue(np.all(np.isfinite(normalized)))

    def test_select_calibration_records_round_robin_and_exhausts(self) -> None:
        """Calibration selection should round-robin classes and stop at exhaustion."""

        records = [
            _record("c0-a", 1, 0),
            _record("c0-b", 1, 0),
            _record("c1-a", 1, 1),
        ]

        selected = prepare_urbansound8k.select_calibration_records(records, limit=100)

        self.assertEqual([item.clip_id for item in selected], ["c0-a", "c1-a", "c0-b"])
        self.assertEqual(len({item.clip_id for item in selected}), len(selected))

    def test_build_split_payload_writes_required_arrays(self) -> None:
        """Split payloads should include all arrays with uniform lengths."""

        config = prepare_urbansound8k.PrepConfig(raw_root=Path("raw"), cache_root=Path("cache"))
        payload = prepare_urbansound8k.build_split_payload(
            [_record("short", 9, 0, length=100), _record("long", 9, 1, length=40000)],
            "val",
            config,
            feature_fn=_fake_feature_fn,
        )

        self.assertEqual(set(prepare_urbansound8k.REQUIRED_CACHE_ARRAYS), set(payload))
        self.assertEqual(payload["inputs"].shape[1:], (prepare_urbansound8k.EXPECTED_FRAMES, prepare_urbansound8k.MEL_BINS))
        self.assertEqual(payload["crop_variant_index"].tolist(), [0, 0])
        long_index = payload["clip_ids"].tolist().index("long")
        short_index = payload["clip_ids"].tolist().index("short")
        self.assertEqual(payload["pad_before_sample"][long_index], 0)
        self.assertEqual(payload["pad_after_sample"][long_index], 0)
        self.assertGreater(payload["pad_before_sample"][short_index], 0)
        self.assertGreater(payload["pad_after_sample"][short_index], 0)

    def test_build_cache_reuses_valid_cache_and_rejects_metadata_mismatch(self) -> None:
        """Cache generation should reuse valid caches and reject stale metadata."""

        with self.subTest("reuse and reject"):
            with self.temporary_config() as config:
                records = [_record("train", 1, 0), _record("val", 9, 1), _record("test", 10, 2)]
                cache_dir = prepare_urbansound8k.build_cache_from_records(records, config, feature_fn=_fake_feature_fn)
                reused_dir = prepare_urbansound8k.build_cache_from_records(records, config, feature_fn=_fake_feature_fn)
                metadata_path = cache_dir / "metadata.json"
                metadata = prepare_urbansound8k.json.loads(metadata_path.read_text())
                metadata["n_fft"] = 1024
                metadata_path.write_text(prepare_urbansound8k.json.dumps(metadata), encoding="utf-8")

                self.assertEqual(reused_dir, cache_dir)
                with self.assertRaisesRegex(ValueError, "n_fft"):
                    prepare_urbansound8k.build_cache_from_records(records, config, feature_fn=_fake_feature_fn)

    def test_build_fold_rotation_caches_write_train_only_splits(self) -> None:
        """Fold rotation should write per-fold schema-2 cache directories."""

        with self.temporary_config() as config:
            records = [
                _record(f"fold-{fold}", fold, fold % len(prepare_urbansound8k.CLASS_NAMES))
                for fold in prepare_urbansound8k.ALL_FOLDS
            ]

            fold_dirs = prepare_urbansound8k.build_fold_rotation_caches(
                records,
                config,
                test_folds=(1, 10),
                feature_fn=_fake_feature_fn,
            )

            self.assertEqual([path.name for path in fold_dirs], ["fold_01", "fold_10"])
            metadata = prepare_urbansound8k.json.loads((fold_dirs[0] / "metadata.json").read_text())
            self.assertEqual(metadata["schema_version"], prepare_urbansound8k.CACHE_SCHEMA_VERSION)
            self.assertEqual(metadata["evaluation_protocol"], "fold_rotation")
            self.assertEqual(metadata["rotation_fold_index"], 1)
            self.assertEqual(metadata["fold_split"]["test"], [1])
            self.assertEqual(metadata["fold_split"]["val"], [2])
            with np.load(fold_dirs[0] / "calibration.npz", allow_pickle=False) as calibration:
                self.assertTrue(set(calibration["folds"].tolist()).issubset(set(metadata["fold_split"]["train"])))

    def test_validate_cache_rejects_each_required_metadata_mismatch(self) -> None:
        """validate_cache should reject all contract metadata mismatches."""

        mutable_fields = [
            field
            for field in prepare_urbansound8k.REQUIRED_METADATA_FIELDS
            if field not in {"schema_version", "normalization_mean", "normalization_std"}
        ]
        for field in mutable_fields:
            with self.subTest(field=field):
                with self.temporary_config() as config:
                    records = [_record("train", 1, 0), _record("val", 9, 1), _record("test", 10, 2)]
                    cache_dir = prepare_urbansound8k.build_cache_from_records(records, config, feature_fn=_fake_feature_fn)
                    metadata_path = cache_dir / "metadata.json"
                    metadata = prepare_urbansound8k.json.loads(metadata_path.read_text())
                    original = metadata[field]
                    metadata[field] = "__wrong__" if not isinstance(original, str) else f"{original}_wrong"
                    metadata_path.write_text(prepare_urbansound8k.json.dumps(metadata), encoding="utf-8")

                    with self.assertRaisesRegex(ValueError, field):
                        prepare_urbansound8k.validate_cache(cache_dir, config)

    def test_validate_cache_rejects_malformed_split_arrays(self) -> None:
        """validate_cache should reject bad dtypes, folds, and calibration duplicates."""

        with self.temporary_config() as config:
            records = [_record("train", 1, 0), _record("val", 9, 1), _record("test", 10, 2)]
            cache_dir = prepare_urbansound8k.build_cache_from_records(records, config, feature_fn=_fake_feature_fn)

            train_path = cache_dir / "train.npz"
            with np.load(train_path, allow_pickle=False) as split_data:
                payload = {key: split_data[key] for key in split_data.files}
            payload["labels"] = payload["labels"].astype(np.float32)
            np.savez_compressed(train_path, **payload)
            with self.assertRaisesRegex(ValueError, "labels"):
                prepare_urbansound8k.validate_cache(cache_dir, config)

        with self.temporary_config() as config:
            records = [_record("train", 1, 0), _record("val", 9, 1), _record("test", 10, 2)]
            cache_dir = prepare_urbansound8k.build_cache_from_records(records, config, feature_fn=_fake_feature_fn)
            val_path = cache_dir / "val.npz"
            with np.load(val_path, allow_pickle=False) as split_data:
                payload = {key: split_data[key] for key in split_data.files}
            payload["folds"] = np.asarray([1], dtype=np.int64)
            np.savez_compressed(val_path, **payload)
            with self.assertRaisesRegex(ValueError, "folds"):
                prepare_urbansound8k.validate_cache(cache_dir, config)

        with self.temporary_config() as config:
            records = [_record("train-a", 1, 0), _record("train-b", 1, 1), _record("val", 9, 1), _record("test", 10, 2)]
            cache_dir = prepare_urbansound8k.build_cache_from_records(records, config, feature_fn=_fake_feature_fn)
            calibration_path = cache_dir / "calibration.npz"
            with np.load(calibration_path, allow_pickle=False) as split_data:
                payload = {key: split_data[key] for key in split_data.files}
            payload["clip_ids"] = np.asarray(["dup", "dup"], dtype=str)
            np.savez_compressed(calibration_path, **payload)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                prepare_urbansound8k.validate_cache(cache_dir, config)

    def test_force_overwrites_conflicting_cache(self) -> None:
        """The force flag should rebuild a conflicting cache."""

        with self.temporary_config() as config:
            records = [_record("train", 1, 0), _record("val", 9, 1), _record("test", 10, 2)]
            cache_dir = prepare_urbansound8k.build_cache_from_records(records, config, feature_fn=_fake_feature_fn)
            metadata_path = cache_dir / "metadata.json"
            metadata = prepare_urbansound8k.json.loads(metadata_path.read_text())
            metadata["n_fft"] = 1024
            metadata_path.write_text(prepare_urbansound8k.json.dumps(metadata), encoding="utf-8")

            rebuilt_dir = prepare_urbansound8k.build_cache_from_records(
                records,
                config,
                force=True,
                feature_fn=_fake_feature_fn,
            )
            rebuilt_metadata = prepare_urbansound8k.json.loads(metadata_path.read_text())

            self.assertEqual(rebuilt_dir, cache_dir)
            self.assertEqual(rebuilt_metadata["n_fft"], prepare_urbansound8k.N_FFT)

    def test_schema_one_cache_rebuilds_without_force(self) -> None:
        """Schema upgrades should rebuild stale caches without manual deletion."""

        with self.temporary_config() as config:
            records = [_record("train", 1, 0), _record("val", 9, 1), _record("test", 10, 2)]
            cache_dir = prepare_urbansound8k.build_cache_from_records(records, config, feature_fn=_fake_feature_fn)
            metadata_path = cache_dir / "metadata.json"
            metadata = prepare_urbansound8k.json.loads(metadata_path.read_text())
            metadata["schema_version"] = 1
            metadata.pop("evaluation_protocol")
            metadata.pop("normalization_scope")
            metadata.pop("label_encoding")
            metadata_path.write_text(prepare_urbansound8k.json.dumps(metadata), encoding="utf-8")

            rebuilt_dir = prepare_urbansound8k.build_cache_from_records(records, config, feature_fn=_fake_feature_fn)
            rebuilt_metadata = prepare_urbansound8k.json.loads(metadata_path.read_text())

            self.assertEqual(rebuilt_dir, cache_dir)
            self.assertEqual(rebuilt_metadata["schema_version"], prepare_urbansound8k.CACHE_SCHEMA_VERSION)
            self.assertEqual(rebuilt_metadata["evaluation_protocol"], "fixed_split")

    def test_download_requires_license_acceptance_before_soundata_import(self) -> None:
        """Download mode should reject missing license acceptance before download."""

        config = prepare_urbansound8k.PrepConfig(raw_root=Path("raw"), cache_root=Path("cache"))

        with patch.object(prepare_urbansound8k, "require_soundata") as require_soundata:
            with self.assertRaisesRegex(ValueError, "--accept-license"):
                prepare_urbansound8k.initialize_soundata_dataset(
                    config,
                    download=True,
                    accept_license=False,
                )

        require_soundata.assert_not_called()

    def test_initialize_soundata_dataset_downloads_and_validates(self) -> None:
        """initialize_soundata_dataset should call mocked download and validate."""

        import urllib

        config = prepare_urbansound8k.PrepConfig(raw_root=Path("raw"), cache_root=Path("cache"))
        fake_dataset = Mock()
        fake_dataset.validate.return_value = ([], [])
        fake_soundata = SimpleNamespace(initialize=Mock(return_value=fake_dataset))

        with patch.object(prepare_urbansound8k, "require_soundata", return_value=fake_soundata):
            returned = prepare_urbansound8k.initialize_soundata_dataset(
                config,
                download=True,
                accept_license=True,
            )

        self.assertIs(returned, fake_dataset)
        self.assertTrue(hasattr(urllib, "request"))
        fake_soundata.initialize.assert_called_once_with("urbansound8k", data_home=str(config.raw_root), version="1.0")
        fake_dataset.download.assert_called_once()
        fake_dataset.validate.assert_called_once_with(verbose=False)

    def test_initialize_soundata_dataset_accepts_empty_nested_validation_dicts(self) -> None:
        """initialize_soundata_dataset should treat empty soundata validation dicts as clean."""

        config = prepare_urbansound8k.PrepConfig(raw_root=Path("raw"), cache_root=Path("cache"))
        fake_dataset = Mock()
        fake_dataset.validate.return_value = ({"metadata": {}, "clips": {}}, {"metadata": {}, "clips": {}})
        fake_soundata = SimpleNamespace(initialize=Mock(return_value=fake_dataset))

        with patch.object(prepare_urbansound8k, "require_soundata", return_value=fake_soundata):
            returned = prepare_urbansound8k.initialize_soundata_dataset(
                config,
                download=False,
                accept_license=False,
            )

        self.assertIs(returned, fake_dataset)

    def test_initialize_soundata_dataset_reports_nested_validation_issues(self) -> None:
        """initialize_soundata_dataset should count nested validation failures."""

        config = prepare_urbansound8k.PrepConfig(raw_root=Path("raw"), cache_root=Path("cache"))
        fake_dataset = Mock()
        fake_dataset.validate.return_value = (
            {"metadata": {"missing.csv": "missing"}, "clips": {}},
            {"metadata": {}, "clips": {"bad.wav": "checksum"}},
        )
        fake_soundata = SimpleNamespace(initialize=Mock(return_value=fake_dataset))

        with patch.object(prepare_urbansound8k, "require_soundata", return_value=fake_soundata):
            with self.assertRaisesRegex(RuntimeError, "Missing files: 1; invalid files: 1"):
                prepare_urbansound8k.initialize_soundata_dataset(
                    config,
                    download=False,
                    accept_license=False,
                )

    def test_main_uses_mocked_soundata_path(self) -> None:
        """main should initialize soundata, convert clips, and build a cache."""

        args = SimpleNamespace(
            raw_root="raw",
            cache_root="cache",
            cache_version=prepare_urbansound8k.CACHE_VERSION,
            batch_period_ms=2000,
            calibration_examples=100,
            train_crop_variants=3,
            crop_seed=1337,
            download=False,
            accept_license=False,
            force=False,
        )
        fake_dataset = object()

        with patch.object(prepare_urbansound8k, "parse_args", return_value=args), patch.object(
            prepare_urbansound8k, "initialize_soundata_dataset", return_value=fake_dataset
        ) as init_dataset, patch.object(
            prepare_urbansound8k, "records_from_soundata", return_value=[]
        ) as records_from_dataset, patch.object(
            prepare_urbansound8k, "build_cache_from_records", return_value=Path("cache")
        ) as build_cache:
            prepare_urbansound8k.main()

        init_dataset.assert_called_once()
        records_from_dataset.assert_called_once_with(fake_dataset)
        build_cache.assert_called_once()

    def temporary_config(self):
        """Return a context manager yielding a temp-backed `PrepConfig`.

        Returns
        -------
        contextlib.AbstractContextManager
            Context manager yielding a preparation config.
        """

        import contextlib
        import tempfile

        @contextlib.contextmanager
        def _manager():
            """Yield a temporary preparation config."""

            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                yield prepare_urbansound8k.PrepConfig(
                    raw_root=root / "raw",
                    cache_root=root / "cache",
                )

        return _manager()


if __name__ == "__main__":
    unittest.main()
