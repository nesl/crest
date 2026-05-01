"""UrbanSound8K cached log-mel dataset adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..interfaces import DatasetABC
from ..pipeline_types import DataSplit, DatasetBundle
from .urbansound8k_common import (
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


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    """Return one field from a mapping- or attribute-like object.

    Parameters
    ----------
    config : Any
        Configuration object that may expose mapping keys or attributes.
    key : str
        Field name to resolve.
    default : Any, optional
        Value returned when the field is absent.

    Returns
    -------
    Any
        Resolved value or ``default``.
    """

    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)


def _require_positive_number(raw_value: Any, *, field_name: str) -> float:
    """Parse one positive finite numeric config value.

    Parameters
    ----------
    raw_value : Any
        Raw value to parse.
    field_name : str
        Human-readable field name for errors.

    Returns
    -------
    float
        Parsed positive finite value.

    Raises
    ------
    ValueError
        If the value is missing, boolean, nonnumeric, non-finite, or
        non-positive.
    """

    if raw_value in (None, "") or isinstance(raw_value, bool):
        raise ValueError(f"UrbanSound8KMelDataset requires positive numeric {field_name}.")
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"UrbanSound8KMelDataset requires numeric {field_name}.") from exc
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"UrbanSound8KMelDataset requires positive finite {field_name}.")
    return value


def _read_metadata(cache_dir: Path) -> dict[str, Any]:
    """Read cache metadata JSON.

    Parameters
    ----------
    cache_dir : pathlib.Path
        Exact cache-version directory.

    Returns
    -------
    dict[str, Any]
        Parsed metadata mapping.

    Raises
    ------
    FileNotFoundError
        If `metadata.json` is missing.
    ValueError
        If the file does not decode to a JSON object.
    """

    metadata_path = cache_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"UrbanSound8K cache is missing {metadata_path}.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("UrbanSound8K cache metadata must be a JSON object.")
    return metadata


def _expected_metadata_values() -> dict[str, Any]:
    """Return fixed metadata values owned by the cache contract.

    Returns
    -------
    dict[str, Any]
        Metadata fields that must exactly match the v1 cache contract.
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
        "normalization_epsilon": NORMALIZATION_EPSILON,
        "fold_split": {key: list(value) for key, value in FOLD_SPLIT.items()},
        "calibration_examples": CALIBRATION_EXAMPLES,
        "train_crop_variants": TRAIN_CROP_VARIANTS,
        "crop_seed": CROP_SEED,
        "class_names": list(CLASS_NAMES),
    }


def _validate_metadata(metadata: dict[str, Any], *, expected_batch_period_ms: float) -> None:
    """Validate cache metadata against the v1 contract.

    Parameters
    ----------
    metadata : dict[str, Any]
        Parsed cache metadata.
    expected_batch_period_ms : float
        Batch period requested by `dataset.params`.

    Returns
    -------
    None
        Raises on invalid metadata.
    """

    for field_name in REQUIRED_METADATA_FIELDS:
        if field_name not in metadata:
            raise ValueError(f"UrbanSound8K metadata missing required field '{field_name}'.")

    for field_name, expected_value in _expected_metadata_values().items():
        if metadata.get(field_name) != expected_value:
            raise ValueError(f"UrbanSound8K metadata field '{field_name}' does not match v1 contract.")

    if float(metadata["batch_period_ms"]) != float(expected_batch_period_ms):
        raise ValueError("UrbanSound8K metadata batch_period_ms does not match dataset.params.")

    for stat_field in ("normalization_mean", "normalization_std"):
        values = metadata.get(stat_field)
        if not isinstance(values, list) or len(values) != MEL_BINS:
            raise ValueError(f"UrbanSound8K metadata field '{stat_field}' must have {MEL_BINS} values.")
        if not np.all(np.isfinite(np.asarray(values, dtype=np.float32))):
            raise ValueError(f"UrbanSound8K metadata field '{stat_field}' must contain finite values.")


def _validate_split_arrays(
    split_name: str,
    arrays: dict[str, np.ndarray],
) -> None:
    """Validate one cached split payload.

    Parameters
    ----------
    split_name : str
        Split name being validated.
    arrays : dict[str, numpy.ndarray]
        Arrays loaded from one `.npz` file.

    Returns
    -------
    None
        Raises on invalid split contents.
    """

    missing = [array_name for array_name in REQUIRED_CACHE_ARRAYS if array_name not in arrays]
    if missing:
        raise ValueError(f"UrbanSound8K {split_name}.npz missing arrays: {', '.join(missing)}.")

    inputs = arrays["inputs"]
    if inputs.dtype != np.float32 or inputs.shape[1:] != (EXPECTED_FRAMES, MEL_BINS):
        raise ValueError(f"UrbanSound8K {split_name}.npz has invalid inputs dtype or shape.")
    if not np.all(np.isfinite(inputs)):
        raise ValueError(f"UrbanSound8K {split_name}.npz contains non-finite inputs.")

    row_count = inputs.shape[0]
    for array_name in REQUIRED_CACHE_ARRAYS[1:]:
        if arrays[array_name].shape[0] != row_count:
            raise ValueError(f"UrbanSound8K {split_name}.npz array '{array_name}' length mismatch.")

    for array_name in (
        "labels",
        "folds",
        "crop_start_sample",
        "crop_variant_index",
        "pad_before_sample",
        "pad_after_sample",
    ):
        if arrays[array_name].dtype != np.int64:
            raise ValueError(f"UrbanSound8K {split_name}.npz array '{array_name}' must be int64.")

    if arrays["clip_ids"].dtype.kind not in {"U", "S"}:
        raise ValueError(f"UrbanSound8K {split_name}.npz clip_ids must be strings.")

    labels = arrays["labels"]
    if np.any(labels < 0) or np.any(labels >= len(CLASS_NAMES)):
        raise ValueError(f"UrbanSound8K {split_name}.npz contains labels outside class range.")

    expected_folds = set(FOLD_SPLIT["train"] if split_name == "calibration" else FOLD_SPLIT[split_name])
    actual_folds = set(int(value) for value in arrays["folds"].tolist())
    if not actual_folds.issubset(expected_folds):
        raise ValueError(f"UrbanSound8K {split_name}.npz contains unexpected folds.")

    if split_name in {"val", "test", "calibration"} and np.any(arrays["crop_variant_index"] != 0):
        raise ValueError(f"UrbanSound8K {split_name}.npz must use center crop variant 0.")


def _load_split(cache_dir: Path, split_name: str) -> DataSplit:
    """Load and validate one cached split.

    Parameters
    ----------
    cache_dir : pathlib.Path
        Exact cache-version directory.
    split_name : str
        Split name to load.

    Returns
    -------
    DataSplit
        Generic split with inputs, integer labels, and split metadata arrays.
    """

    split_path = cache_dir / f"{split_name}.npz"
    if not split_path.is_file():
        raise FileNotFoundError(f"UrbanSound8K cache is missing {split_path}.")
    with np.load(split_path, allow_pickle=False) as loaded:
        arrays = {array_name: loaded[array_name] for array_name in loaded.files}

    _validate_split_arrays(split_name, arrays)
    return DataSplit(
        inputs=arrays["inputs"],
        targets=arrays["labels"],
        sample_weights=None,
        metadata={
            "folds": arrays["folds"],
            "clip_ids": arrays["clip_ids"],
            "crop_start_sample": arrays["crop_start_sample"],
            "crop_variant_index": arrays["crop_variant_index"],
            "pad_before_sample": arrays["pad_before_sample"],
            "pad_after_sample": arrays["pad_after_sample"],
        },
    )


class UrbanSound8KMelDataset(DatasetABC):
    """Dataset adapter for Phase 1 cached UrbanSound8K log-mel features."""

    @property
    def name(self) -> str:
        """Return the stable registry-facing dataset name.

        Returns
        -------
        str
            Stable dataset identifier.
        """

        return COMPONENT_NAME

    def validate_config(self, dataset_config: Any) -> None:
        """Validate pre-load UrbanSound8K dataset configuration.

        Parameters
        ----------
        dataset_config : Any
            `config.dataset.params` subtree.

        Returns
        -------
        None
            Raises if required pre-load fields are missing or invalid.
        """

        cache_dir = _cfg_get(dataset_config, "cache_dir", None)
        if cache_dir in (None, ""):
            raise ValueError("UrbanSound8KMelDataset requires dataset.params.cache_dir.")
        if _cfg_get(dataset_config, "cache_version", None) is not None:
            raise ValueError("UrbanSound8KMelDataset Phase 3 does not accept dataset.params.cache_version.")
        _require_positive_number(
            _cfg_get(dataset_config, "batch_period_ms", None),
            field_name="dataset.params.batch_period_ms",
        )

    def load(self, dataset_config: Any) -> DatasetBundle:
        """Load cached UrbanSound8K log-mel features into a dataset bundle.

        Parameters
        ----------
        dataset_config : Any
            `config.dataset.params` subtree.

        Returns
        -------
        DatasetBundle
            Dataset bundle containing train, validation, test, and calibration
            splits plus audio classification metadata.
        """

        self.validate_config(dataset_config)
        cache_dir = Path(str(_cfg_get(dataset_config, "cache_dir"))).expanduser().resolve()
        batch_period_ms = _require_positive_number(
            _cfg_get(dataset_config, "batch_period_ms"),
            field_name="dataset.params.batch_period_ms",
        )
        metadata = _read_metadata(cache_dir)
        _validate_metadata(metadata, expected_batch_period_ms=batch_period_ms)

        train = _load_split(cache_dir, "train")
        val = _load_split(cache_dir, "val")
        test = _load_split(cache_dir, "test")
        calibration = _load_split(cache_dir, "calibration")

        return DatasetBundle(
            train=train,
            val=val,
            test=test,
            calibration=calibration,
            input_shape=(EXPECTED_FRAMES, MEL_BINS),
            input_dtype=FEATURE_DTYPE,
            metadata={
                "dataset_name": DATASET_NAME,
                "component_name": COMPONENT_NAME,
                "cache_dir": str(cache_dir),
                "class_names": list(CLASS_NAMES),
                "num_classes": len(CLASS_NAMES),
                "fold_split": {key: list(value) for key, value in FOLD_SPLIT.items()},
                "batch_period_ms": batch_period_ms,
                "label_encoding": LABEL_ENCODING,
            },
        )

    def make_calibration_data(
        self,
        bundle: DatasetBundle,
        dataset_config: Any,
    ) -> DataSplit | None:
        """Return cache-owned calibration data without synthesis.

        Parameters
        ----------
        bundle : DatasetBundle
            Loaded dataset bundle.
        dataset_config : Any
            Dataset-local configuration subtree.

        Returns
        -------
        DataSplit | None
            Calibration split stored in the cache.
        """

        del dataset_config
        return bundle.calibration
