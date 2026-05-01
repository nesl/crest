#!/usr/bin/env python3
"""Prepare UrbanSound8K raw audio and deterministic log-mel feature caches."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable, Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tinyodom.datasets.urbansound8k_common import (  # noqa: E402
    BATCH_PERIOD_MS,
    CACHE_VERSION,
    CALIBRATION_EXAMPLES,
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
    REQUIRED_CACHE_ARRAYS,
    REQUIRED_METADATA_FIELDS,
    SAMPLE_RATE_HZ,
    TARGET_NUM_SAMPLES,
    TRAIN_CROP_VARIANTS,
    WINDOW_LENGTH_SAMPLES,
    WINDOW_MS,
)

FeatureFn = Callable[[np.ndarray, int], np.ndarray]


@dataclass(frozen=True)
class PrepConfig:
    """Configuration for UrbanSound8K cache generation.

    Parameters
    ----------
    raw_root : pathlib.Path
        Directory containing or receiving the raw soundata UrbanSound8K data.
    cache_root : pathlib.Path
        Directory containing versioned generated feature caches.
    cache_version : str
        Cache version directory name.
    batch_period_ms : int
        Logical arrival period for one cached feature batch.
    calibration_examples : int
        Maximum number of calibration examples to emit.
    train_crop_variants : int
        Number of deterministic crop variants generated for each train clip.
    crop_seed : int
        Global seed mixed with stable clip IDs for train crop generation.
    """

    raw_root: Path
    cache_root: Path
    cache_version: str = CACHE_VERSION
    batch_period_ms: int = BATCH_PERIOD_MS
    calibration_examples: int = CALIBRATION_EXAMPLES
    train_crop_variants: int = TRAIN_CROP_VARIANTS
    crop_seed: int = CROP_SEED

    @property
    def cache_dir(self) -> Path:
        """Return the concrete versioned cache directory.

        Returns
        -------
        pathlib.Path
            Directory where split `.npz` files and `metadata.json` are stored.
        """

        return self.cache_root / self.cache_version


@dataclass(frozen=True)
class ClipRecord:
    """Minimal UrbanSound8K clip payload used by cache generation.

    Parameters
    ----------
    clip_id : str
        Stable soundata clip identifier.
    fold : int
        UrbanSound8K fold number from 1 to 10.
    class_id : int
        Integer class label from 0 to 9.
    class_label : str
        Human-readable class label matching `CLASS_NAMES[class_id]`.
    waveform : numpy.ndarray
        Mono or multichannel waveform loaded from soundata.
    sample_rate : int
        Sample rate of `waveform`.
    """

    clip_id: str
    fold: int
    class_id: int
    class_label: str
    waveform: np.ndarray
    sample_rate: int


@dataclass(frozen=True)
class CropResult:
    """One fixed-length crop plus its metadata.

    Parameters
    ----------
    waveform : numpy.ndarray
        Cropped or padded waveform with exactly `TARGET_NUM_SAMPLES` samples.
    crop_start_sample : int
        Source waveform start index before padding.
    pad_before_sample : int
        Number of zero samples prepended.
    pad_after_sample : int
        Number of zero samples appended.
    """

    waveform: np.ndarray
    crop_start_sample: int
    pad_before_sample: int
    pad_after_sample: int


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for UrbanSound8K preparation.

    Returns
    -------
    argparse.Namespace
        Parsed preparation options.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default="data/urbansound8k/raw")
    parser.add_argument("--cache-root", default="data/urbansound8k/cache")
    parser.add_argument("--cache-version", default=CACHE_VERSION)
    parser.add_argument("--batch-period-ms", type=int, default=BATCH_PERIOD_MS)
    parser.add_argument("--calibration-examples", type=int, default=CALIBRATION_EXAMPLES)
    parser.add_argument("--train-crop-variants", type=int, default=TRAIN_CROP_VARIANTS)
    parser.add_argument("--crop-seed", type=int, default=CROP_SEED)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--accept-license", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> PrepConfig:
    """Build a typed preparation config from parsed CLI options.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line options from `parse_args`.

    Returns
    -------
    PrepConfig
        Normalized configuration with resolved paths.
    """

    return PrepConfig(
        raw_root=Path(args.raw_root),
        cache_root=Path(args.cache_root),
        cache_version=str(args.cache_version),
        batch_period_ms=int(args.batch_period_ms),
        calibration_examples=int(args.calibration_examples),
        train_crop_variants=int(args.train_crop_variants),
        crop_seed=int(args.crop_seed),
    )


def stable_clip_seed(crop_seed: int, clip_id: str) -> int:
    """Derive a process-independent crop seed for one clip.

    Parameters
    ----------
    crop_seed : int
        Global crop seed.
    clip_id : str
        Stable clip identifier.

    Returns
    -------
    int
        Seed suitable for `numpy.random.default_rng`.
    """

    # Python's built-in hash function is randomized per process, so use SHA-256
    # to keep crop starts stable across machines and reruns.
    digest_value = int.from_bytes(hashlib.sha256(clip_id.encode("utf-8")).digest()[:8], "little")
    return int((int(crop_seed) + digest_value) % (2**63 - 1))


def to_mono_float32(waveform: np.ndarray) -> np.ndarray:
    """Convert a waveform to mono float32 samples.

    Parameters
    ----------
    waveform : numpy.ndarray
        Input waveform, optionally multichannel.

    Returns
    -------
    numpy.ndarray
        One-dimensional float32 waveform.
    """

    arr = np.asarray(waveform, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return np.mean(arr, axis=-1, dtype=np.float32)
    raise ValueError(f"Expected 1D or 2D waveform, got shape {arr.shape}.")


def require_librosa() -> Any:
    """Import librosa lazily so tests can import this module without it.

    Returns
    -------
    module
        Imported `librosa` module.
    """

    try:
        import librosa  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "librosa is required to build UrbanSound8K feature caches. "
            "Update the tinyodomex environment from environment.yml."
        ) from exc
    return librosa


def resample_audio(waveform: np.ndarray, sample_rate: int, target_rate: int = SAMPLE_RATE_HZ) -> np.ndarray:
    """Resample audio to the target rate when needed.

    Parameters
    ----------
    waveform : numpy.ndarray
        Mono waveform.
    sample_rate : int
        Current sample rate.
    target_rate : int, optional
        Desired sample rate.

    Returns
    -------
    numpy.ndarray
        Resampled mono float32 waveform.
    """

    mono = to_mono_float32(waveform)
    if int(sample_rate) == int(target_rate):
        return mono
    librosa = require_librosa()
    return np.asarray(librosa.resample(y=mono, orig_sr=int(sample_rate), target_sr=int(target_rate)), dtype=np.float32)


def crop_or_pad_waveform(waveform: np.ndarray, start_sample: int, target_samples: int = TARGET_NUM_SAMPLES) -> CropResult:
    """Crop or symmetrically pad one waveform to the fixed input length.

    Parameters
    ----------
    waveform : numpy.ndarray
        Mono waveform.
    start_sample : int
        Requested crop start for long clips. Ignored for short clips.
    target_samples : int, optional
        Desired output length.

    Returns
    -------
    CropResult
        Fixed-length waveform plus crop and padding metadata.
    """

    mono = to_mono_float32(waveform)
    if mono.shape[0] >= target_samples:
        start = max(0, min(int(start_sample), mono.shape[0] - target_samples))
        return CropResult(
            waveform=mono[start:start + target_samples].astype(np.float32, copy=False),
            crop_start_sample=start,
            pad_before_sample=0,
            pad_after_sample=0,
        )

    missing = target_samples - mono.shape[0]
    pad_before = missing // 2
    pad_after = missing - pad_before
    padded = np.pad(mono, (pad_before, pad_after), mode="constant")
    return CropResult(
        waveform=padded.astype(np.float32, copy=False),
        crop_start_sample=0,
        pad_before_sample=int(pad_before),
        pad_after_sample=int(pad_after),
    )


def center_crop_start(num_samples: int, target_samples: int = TARGET_NUM_SAMPLES) -> int:
    """Return the center-crop start for a source length.

    Parameters
    ----------
    num_samples : int
        Source waveform length.
    target_samples : int, optional
        Desired crop length.

    Returns
    -------
    int
        Center-crop start, or zero for short clips.
    """

    if int(num_samples) <= int(target_samples):
        return 0
    return (int(num_samples) - int(target_samples)) // 2


def train_crop_starts(clip_id: str, num_samples: int, config: PrepConfig) -> list[int]:
    """Return deterministic train crop starts for one clip.

    Parameters
    ----------
    clip_id : str
        Stable clip identifier.
    num_samples : int
        Source waveform length after resampling.
    config : PrepConfig
        Preparation configuration.

    Returns
    -------
    list[int]
        Crop starts, one per configured train crop variant.
    """

    if int(num_samples) <= TARGET_NUM_SAMPLES:
        return [0 for _ in range(config.train_crop_variants)]
    rng = np.random.default_rng(stable_clip_seed(config.crop_seed, clip_id))
    starts = rng.integers(0, int(num_samples) - TARGET_NUM_SAMPLES + 1, size=config.train_crop_variants)
    return [int(value) for value in starts]


def compute_log_mel_features(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """Compute fixed-shape natural-log mel power features.

    Parameters
    ----------
    waveform : numpy.ndarray
        Fixed-length mono waveform.
    sample_rate : int
        Waveform sample rate.

    Returns
    -------
    numpy.ndarray
        Float32 feature array with shape `(201, 64)`.
    """

    librosa = require_librosa()
    mel_power = librosa.feature.melspectrogram(
        y=np.asarray(waveform, dtype=np.float32),
        sr=int(sample_rate),
        n_fft=N_FFT,
        hop_length=HOP_LENGTH_SAMPLES,
        win_length=WINDOW_LENGTH_SAMPLES,
        n_mels=MEL_BINS,
        center=CENTER,
        power=POWER,
    )
    log_mel = np.log(np.maximum(np.asarray(mel_power, dtype=np.float32), LOG_FLOOR))
    features = log_mel.T.astype(np.float32, copy=False)
    if features.shape != (EXPECTED_FRAMES, MEL_BINS):
        raise ValueError(f"Expected log-mel shape {(EXPECTED_FRAMES, MEL_BINS)}, got {features.shape}.")
    return features


def normalize_features(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply per-mel-bin standardization to feature rows.

    Parameters
    ----------
    features : numpy.ndarray
        Feature array whose last dimension is mel bins.
    mean : numpy.ndarray
        Per-mel-bin mean with shape `(64,)`.
    std : numpy.ndarray
        Per-mel-bin standard deviation with shape `(64,)`.

    Returns
    -------
    numpy.ndarray
        Standardized float32 features.
    """

    normalized = (np.asarray(features, dtype=np.float32) - mean) / (std + NORMALIZATION_EPSILON)
    return normalized.astype(np.float32, copy=False)


def compute_normalization_stats(train_inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-mel-bin normalization statistics from train rows.

    Parameters
    ----------
    train_inputs : numpy.ndarray
        Unnormalized expanded training rows with shape `(N, 201, 64)`.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Mean and standard deviation arrays with shape `(64,)`.
    """

    mean = np.mean(train_inputs, axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = np.std(train_inputs, axis=(0, 1), dtype=np.float64).astype(np.float32)
    return mean, std


def split_records(records: Sequence[ClipRecord]) -> dict[str, list[ClipRecord]]:
    """Partition clip records by the fixed UrbanSound8K fold split.

    Parameters
    ----------
    records : Sequence[ClipRecord]
        Clip records to partition.

    Returns
    -------
    dict[str, list[ClipRecord]]
        Mapping for train, validation, and test records.
    """

    split_by_name = {name: [] for name in ("train", "val", "test")}
    fold_to_split = {
        fold: split_name
        for split_name, folds in FOLD_SPLIT.items()
        for fold in folds
    }
    for record in sorted(records, key=lambda item: item.clip_id):
        split_name = fold_to_split.get(int(record.fold))
        if split_name is not None:
            split_by_name[split_name].append(record)
    return split_by_name


def select_calibration_records(records: Sequence[ClipRecord], limit: int) -> list[ClipRecord]:
    """Select deterministic class-balanced calibration records.

    Parameters
    ----------
    records : Sequence[ClipRecord]
        Base training records, before crop expansion.
    limit : int
        Maximum number of records to return.

    Returns
    -------
    list[ClipRecord]
        Calibration records without duplicate clip IDs.
    """

    buckets: dict[int, list[ClipRecord]] = {class_id: [] for class_id in range(len(CLASS_NAMES))}
    for record in records:
        buckets[int(record.class_id)].append(record)
    for class_id, bucket in buckets.items():
        buckets[class_id] = sorted(bucket, key=lambda item: (item.class_id, item.fold, item.clip_id))

    selected: list[ClipRecord] = []
    seen_clip_ids: set[str] = set()
    while len(selected) < int(limit):
        added = False
        # Walk one item per class bucket each round so the representative set
        # stays class-balanced even when some classes have fewer clips.
        for class_id in range(len(CLASS_NAMES)):
            bucket = buckets[class_id]
            while bucket and bucket[0].clip_id in seen_clip_ids:
                bucket.pop(0)
            if not bucket:
                continue
            record = bucket.pop(0)
            selected.append(record)
            seen_clip_ids.add(record.clip_id)
            added = True
            if len(selected) >= int(limit):
                break
        if not added:
            break
    return selected


def build_split_payload(
    records: Sequence[ClipRecord],
    split_name: str,
    config: PrepConfig,
    feature_fn: FeatureFn = compute_log_mel_features,
    normalization: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Build one split payload in the cache `.npz` schema.

    Parameters
    ----------
    records : Sequence[ClipRecord]
        Clip records included in this split.
    split_name : str
        Name of the split being built.
    config : PrepConfig
        Preparation configuration.
    feature_fn : callable, optional
        Feature extraction function used for tests or production.
    normalization : tuple[numpy.ndarray, numpy.ndarray] | None, optional
        Optional mean/std arrays. When omitted, raw log-mel features are kept.

    Returns
    -------
    dict[str, numpy.ndarray]
        Arrays ready to write to a split `.npz` file.
    """

    inputs: list[np.ndarray] = []
    labels: list[int] = []
    folds: list[int] = []
    clip_ids: list[str] = []
    crop_starts: list[int] = []
    variant_indices: list[int] = []
    pad_before: list[int] = []
    pad_after: list[int] = []

    for record in sorted(records, key=lambda item: item.clip_id):
        resampled = resample_audio(record.waveform, record.sample_rate)
        starts = (
            train_crop_starts(record.clip_id, resampled.shape[0], config)
            if split_name == "train"
            else [center_crop_start(resampled.shape[0])]
        )
        for variant_index, start in enumerate(starts):
            crop = crop_or_pad_waveform(resampled, start)
            features = feature_fn(crop.waveform, SAMPLE_RATE_HZ)
            if normalization is not None:
                features = normalize_features(features, normalization[0], normalization[1])
            inputs.append(features.astype(np.float32, copy=False))
            labels.append(int(record.class_id))
            folds.append(int(record.fold))
            clip_ids.append(str(record.clip_id))
            crop_starts.append(int(crop.crop_start_sample))
            variant_indices.append(int(variant_index if split_name == "train" else 0))
            pad_before.append(int(crop.pad_before_sample))
            pad_after.append(int(crop.pad_after_sample))

    input_array = (
        np.stack(inputs, axis=0).astype(np.float32, copy=False)
        if inputs
        else np.empty((0, EXPECTED_FRAMES, MEL_BINS), dtype=np.float32)
    )
    return {
        "inputs": input_array,
        "labels": np.asarray(labels, dtype=np.int64),
        "folds": np.asarray(folds, dtype=np.int64),
        "clip_ids": np.asarray(clip_ids, dtype=str),
        "crop_start_sample": np.asarray(crop_starts, dtype=np.int64),
        "crop_variant_index": np.asarray(variant_indices, dtype=np.int64),
        "pad_before_sample": np.asarray(pad_before, dtype=np.int64),
        "pad_after_sample": np.asarray(pad_after, dtype=np.int64),
    }


def expected_metadata(config: PrepConfig, mean: np.ndarray | None = None, std: np.ndarray | None = None) -> dict[str, Any]:
    """Build the metadata JSON payload for a generated cache.

    Parameters
    ----------
    config : PrepConfig
        Preparation configuration.
    mean : numpy.ndarray | None, optional
        Optional normalization mean.
    std : numpy.ndarray | None, optional
        Optional normalization standard deviation.

    Returns
    -------
    dict[str, object]
        Metadata payload.
    """

    zero_stats = [0.0 for _ in range(MEL_BINS)]
    return {
        "schema_version": 1,
        "dataset_name": DATASET_NAME,
        "component_name": COMPONENT_NAME,
        "cache_version": config.cache_version,
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
        "normalization_mean": (
            np.asarray(mean, dtype=np.float32).tolist() if mean is not None else zero_stats
        ),
        "normalization_std": (
            np.asarray(std, dtype=np.float32).tolist() if std is not None else zero_stats
        ),
        "normalization_epsilon": NORMALIZATION_EPSILON,
        "fold_split": {key: list(value) for key, value in FOLD_SPLIT.items()},
        "batch_period_ms": config.batch_period_ms,
        "train_crop_variants": config.train_crop_variants,
        "crop_seed": config.crop_seed,
        "calibration_examples": config.calibration_examples,
        "class_names": list(CLASS_NAMES),
        "label_encoding": LABEL_ENCODING,
    }


def write_cache(cache_dir: Path, metadata: dict[str, Any], payloads: dict[str, dict[str, np.ndarray]]) -> None:
    """Write metadata and split payloads to a versioned cache directory.

    Parameters
    ----------
    cache_dir : pathlib.Path
        Destination cache directory.
    metadata : dict[str, object]
        Cache metadata JSON payload.
    payloads : dict[str, dict[str, numpy.ndarray]]
        Mapping of split name to `.npz` array payload.
    """

    cache_dir.mkdir(parents=True, exist_ok=True)
    for split_name, payload in payloads.items():
        np.savez_compressed(cache_dir / f"{split_name}.npz", **payload)
    (cache_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def validate_cache(cache_dir: Path, config: PrepConfig) -> bool:
    """Validate an existing cache against the Phase 1 contract.

    Parameters
    ----------
    cache_dir : pathlib.Path
        Existing cache directory.
    config : PrepConfig
        Requested cache configuration.

    Returns
    -------
    bool
        True when the cache is valid.

    Raises
    ------
    ValueError
        If present cache files conflict with the requested configuration.
    """

    if not cache_dir.exists():
        return False
    metadata_path = cache_dir / "metadata.json"
    if not metadata_path.exists():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = expected_metadata(config)
    for field in REQUIRED_METADATA_FIELDS:
        if field not in metadata:
            raise ValueError(f"Cache metadata missing required field '{field}'.")
        if field in {"normalization_mean", "normalization_std"}:
            continue
        if metadata[field] != expected[field]:
            raise ValueError(f"Cache metadata field '{field}' does not match requested config.")
    for stat_field in ("normalization_mean", "normalization_std"):
        values = metadata.get(stat_field)
        if not isinstance(values, list) or len(values) != MEL_BINS:
            raise ValueError(f"Cache metadata field '{stat_field}' must be a list of {MEL_BINS} floats.")
        stat_values = np.asarray(values, dtype=np.float32)
        if stat_values.shape != (MEL_BINS,) or not np.all(np.isfinite(stat_values)):
            raise ValueError(f"Cache metadata field '{stat_field}' must contain finite float values.")

    for split_name in ("train", "val", "test", "calibration"):
        split_path = cache_dir / f"{split_name}.npz"
        if not split_path.exists():
            return False
        with np.load(split_path, allow_pickle=False) as split_data:
            for array_name in REQUIRED_CACHE_ARRAYS:
                if array_name not in split_data:
                    raise ValueError(f"{split_path} missing array '{array_name}'.")
            inputs = split_data["inputs"]
            if inputs.dtype != np.float32 or inputs.shape[1:] != (EXPECTED_FRAMES, MEL_BINS):
                raise ValueError(f"{split_path} has invalid inputs shape or dtype.")
            if not np.all(np.isfinite(inputs)):
                raise ValueError(f"{split_path} contains non-finite input features.")
            for array_name in REQUIRED_CACHE_ARRAYS[1:]:
                if split_data[array_name].shape[0] != inputs.shape[0]:
                    raise ValueError(f"{split_path} array '{array_name}' length does not match inputs.")
            int_arrays = (
                "labels",
                "folds",
                "crop_start_sample",
                "crop_variant_index",
                "pad_before_sample",
                "pad_after_sample",
            )
            for array_name in int_arrays:
                if split_data[array_name].dtype != np.int64:
                    raise ValueError(f"{split_path} array '{array_name}' must have dtype int64.")
            if split_data["clip_ids"].dtype.kind not in {"U", "S"}:
                raise ValueError(f"{split_path} array 'clip_ids' must be a string array.")
            labels = split_data["labels"]
            if np.any(labels < 0) or np.any(labels >= len(CLASS_NAMES)):
                raise ValueError(f"{split_path} contains labels outside the UrbanSound8K class range.")
            expected_folds = set(FOLD_SPLIT["train"] if split_name == "calibration" else FOLD_SPLIT[split_name])
            actual_folds = set(int(value) for value in split_data["folds"].tolist())
            if not actual_folds.issubset(expected_folds):
                raise ValueError(f"{split_path} contains folds outside the expected {split_name} split.")
            if split_name in {"val", "test", "calibration"} and np.any(split_data["crop_variant_index"] != 0):
                raise ValueError(f"{split_path} center-crop split must use crop_variant_index=0.")
            if split_name == "calibration":
                clip_ids = split_data["clip_ids"].tolist()
                if len(clip_ids) != len(set(clip_ids)):
                    raise ValueError(f"{split_path} contains duplicate calibration clip IDs.")
    return True


def build_cache_from_records(
    records: Sequence[ClipRecord],
    config: PrepConfig,
    *,
    force: bool = False,
    feature_fn: FeatureFn = compute_log_mel_features,
) -> Path:
    """Generate or reuse a deterministic UrbanSound8K feature cache.

    Parameters
    ----------
    records : Sequence[ClipRecord]
        UrbanSound8K clip records.
    config : PrepConfig
        Cache generation configuration.
    force : bool, optional
        Whether to overwrite conflicting existing caches.
    feature_fn : callable, optional
        Feature extractor used for production or tests.

    Returns
    -------
    pathlib.Path
        Generated or reused cache directory.
    """

    cache_dir = config.cache_dir
    if cache_dir.exists() and not force:
        if validate_cache(cache_dir, config):
            return cache_dir

    splits = split_records(records)
    # Generate train features before all other splits so normalization is based
    # on the expanded training crop rows and never deferred to the loader.
    train_payload_raw = build_split_payload(splits["train"], "train", config, feature_fn=feature_fn)
    mean, std = compute_normalization_stats(train_payload_raw["inputs"])
    normalization = (mean, std)
    payloads = {
        "train": {
            **train_payload_raw,
            "inputs": normalize_features(train_payload_raw["inputs"], mean, std),
        },
        "val": build_split_payload(splits["val"], "val", config, feature_fn=feature_fn, normalization=normalization),
        "test": build_split_payload(splits["test"], "test", config, feature_fn=feature_fn, normalization=normalization),
        "calibration": build_split_payload(
            select_calibration_records(splits["train"], config.calibration_examples),
            "calibration",
            config,
            feature_fn=feature_fn,
            normalization=normalization,
        ),
    }
    write_cache(cache_dir, expected_metadata(config, mean, std), payloads)
    return cache_dir


def require_soundata() -> Any:
    """Import soundata lazily so unit tests can mock production loading.

    Returns
    -------
    module
        Imported `soundata` module.
    """

    try:
        import soundata  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "soundata is required to prepare UrbanSound8K. "
            "Update the tinyodomex environment from environment.yml."
        ) from exc
    return soundata


def initialize_soundata_dataset(config: PrepConfig, *, download: bool, accept_license: bool) -> Any:
    """Initialize, optionally download, and validate the soundata dataset.

    Parameters
    ----------
    config : PrepConfig
        Preparation configuration.
    download : bool
        Whether to call `dataset.download()`.
    accept_license : bool
        Whether the caller accepted the UrbanSound8K license terms.

    Returns
    -------
    object
        Initialized soundata dataset object.
    """

    if download and not accept_license:
        raise ValueError("--accept-license is required when --download is used.")
    soundata = require_soundata()
    dataset = soundata.initialize("urbansound8k", data_home=str(config.raw_root), version="1.0")
    if download:
        # soundata 1.0.1 expects urllib.request to be loaded before it calls
        # urllib.request.urlretrieve inside its downloader.
        import urllib.request  # noqa: F401

        dataset.download()
    missing_files, invalid_files = dataset.validate(verbose=False)
    missing_count = count_validation_issues(missing_files)
    invalid_count = count_validation_issues(invalid_files)
    if missing_count or invalid_count:
        raise RuntimeError(
            "UrbanSound8K validation failed. "
            f"Missing files: {missing_count}; invalid files: {invalid_count}."
        )
    return dataset


def count_validation_issues(validation_payload: Any) -> int:
    """Count nested soundata validation issues.

    Parameters
    ----------
    validation_payload : Any
        Payload returned by ``soundata.Dataset.validate``. soundata 1.0.1
        returns dictionaries such as ``{"metadata": {}, "clips": {}}`` even
        when no files are missing or invalid.

    Returns
    -------
    int
        Number of nested validation issues.
    """

    if validation_payload is None:
        return 0
    if isinstance(validation_payload, dict):
        return sum(count_validation_issues(value) for value in validation_payload.values())
    if isinstance(validation_payload, (list, tuple, set)):
        return sum(count_validation_issues(value) if isinstance(value, dict) else 1 for value in validation_payload)
    return 1 if validation_payload else 0


def records_from_soundata(dataset: Any) -> list[ClipRecord]:
    """Load `ClipRecord` objects from a soundata UrbanSound8K dataset.

    Parameters
    ----------
    dataset : object
        soundata dataset object exposing `load_clips()`.

    Returns
    -------
    list[ClipRecord]
        Sorted clip records ready for cache generation.
    """

    records: list[ClipRecord] = []
    for clip_id, clip in sorted(dataset.load_clips().items()):
        audio_value = getattr(clip, "audio", None)
        if audio_value is None:
            audio_value = clip.load_audio()
        waveform, sample_rate = audio_value
        records.append(
            ClipRecord(
                clip_id=str(clip_id),
                fold=int(clip.fold),
                class_id=int(clip.class_id),
                class_label=str(clip.class_label),
                waveform=np.asarray(waveform),
                sample_rate=int(sample_rate),
            )
        )
    return records


def main() -> None:
    """Run the UrbanSound8K preparation workflow from the command line."""

    args = parse_args()
    config = config_from_args(args)
    dataset = initialize_soundata_dataset(
        config,
        download=bool(args.download),
        accept_license=bool(args.accept_license),
    )
    cache_dir = build_cache_from_records(
        records_from_soundata(dataset),
        config,
        force=bool(args.force),
    )
    print(f"UrbanSound8K cache prepared in {cache_dir}")


if __name__ == "__main__":
    main()
