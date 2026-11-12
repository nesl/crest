# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Verified prepared FER+/RAF-DB mixed facial-expression dataset."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..interfaces import DatasetABC
from ..component_selection import cfg_get as _cfg_get
from ..pipeline_types import DataSplit, DatasetBundle

COMPONENT_NAME = "ferplus_rafdb_mixed"
DATASET_SCHEMA = "mixed_native_v1"
LABEL_SOURCE = "FER+_RAF-DB_mixed"
CLASS_NAMES = ("happy", "sad", "angry", "surprised", "neutral")
SOURCES = ("ferplus", "raf")
SUPPORTED_IMAGE_SIZES = (64, 96)
PREPROCESSING_POLICY = (
    "source crop; nearest floor-index resize; float32 pixel/127.5-1; external NHW"
)
FIXTURE_SCHEMA = "crest_fer_representative_fixture_v1"


def _config_keys(config: Any) -> set[str]:
    """Return explicit keys from a mapping-like config object."""
    if isinstance(config, Mapping):
        return set(config)
    items = getattr(config, "items", None)
    if callable(items):
        return {str(key) for key, _ in items()}
    return set(vars(config)) if hasattr(config, "__dict__") else set()


def _require_positive_integer(value: Any, *, field_name: str) -> int:
    """Validate a non-boolean positive integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return int(value)


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    """Return whether a value is a lowercase hexadecimal SHA-256 digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def preprocess_grayscale(images: np.ndarray, image_size: int) -> np.ndarray:
    """Resize grayscale images with the deployed nearest-floor rule.

    Parameters
    ----------
    images : numpy.ndarray
        Unsigned grayscale images with shape ``(N, H, W)``.
    image_size : int
        Destination height and width, either 64 or 96.

    Returns
    -------
    numpy.ndarray
        Float32 inputs with external shape ``(N, image_size, image_size)``.
    """
    if image_size not in SUPPORTED_IMAGE_SIZES:
        raise ValueError(f"image_size must be one of {SUPPORTED_IMAGE_SIZES}.")
    values = np.asarray(images)
    if values.ndim != 3 or values.dtype != np.uint8 or min(values.shape[1:]) < 1:
        raise ValueError("FER mixed images must be uint8 arrays with shape (N, H, W).")
    rows = np.arange(image_size) * values.shape[1] // image_size
    columns = np.arange(image_size) * values.shape[2] // image_size
    resized = values[:, rows, :][:, :, columns]
    return resized.astype(np.float32) / np.float32(127.5) - np.float32(1.0)


def equal_source_weights(sources: np.ndarray) -> np.ndarray:
    """Return weights giving FER+ and RAF-DB equal total mass."""
    normalized = np.asarray(sources)
    counts = Counter(str(value) for value in normalized.tolist())
    if set(counts) != set(SOURCES) or any(counts[source] <= 0 for source in SOURCES):
        raise ValueError("Both FER+ and RAF sources are required for equal source weighting.")
    return np.asarray(
        [len(normalized) / (len(SOURCES) * counts[str(source)]) for source in normalized],
        dtype=np.float32,
    )


def _read_manifest(data_dir: Path) -> tuple[dict[str, Any], str]:
    """Read and validate prepared-dataset manifest-level provenance."""
    path = data_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"FER mixed dataset is missing {path}.")
    manifest_sha256 = _sha256(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("FER mixed manifest must be a JSON object.")
    expected = {
        "classes": list(CLASS_NAMES),
        "label_source": LABEL_SOURCE,
        "schema": DATASET_SCHEMA,
        "fer_native_size": 48,
        "raf_native_size": 100,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"FER mixed manifest field '{key}' does not match the required schema.")
    if manifest.get("crop_mode") not in {"provided_aligned", "espressif_manifest"}:
        raise ValueError("FER mixed manifest has an unsupported crop_mode.")
    detector_hash = manifest.get("detector_boxes_sha256")
    if manifest["crop_mode"] == "provided_aligned" and detector_hash is not None:
        raise ValueError("Provided-aligned FER mixed data must not declare detector boxes.")
    if manifest["crop_mode"] == "espressif_manifest" and not _is_sha256(detector_hash):
        raise ValueError("Detector-cropped FER mixed data requires detector-box provenance.")
    if not isinstance(manifest.get("source_hashes"), dict) or not manifest["source_hashes"]:
        raise ValueError("FER mixed manifest requires source hash provenance.")
    for name, value in manifest["source_hashes"].items():
        if name.endswith("sha256") and not _is_sha256(value):
            raise ValueError(f"FER mixed manifest source hash '{name}' is invalid.")
    if not _is_sha256(manifest.get("records_sha256")):
        raise ValueError("FER mixed manifest records_sha256 is invalid.")
    records_path = data_dir / "records.jsonl"
    if not records_path.is_file() or _sha256(records_path) != manifest["records_sha256"]:
        raise ValueError("FER mixed records.jsonl hash does not match the manifest.")
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or set(splits) != {"train", "val", "test"}:
        raise ValueError("FER mixed manifest must define train, val, and test splits.")
    return manifest, manifest_sha256


def _validate_string_array(
    values: np.ndarray,
    *,
    field_name: str,
    count: int,
    unique: bool,
) -> np.ndarray:
    """Validate one one-dimensional unique Unicode metadata array."""
    array = np.asarray(values)
    if array.shape != (count,) or array.dtype.kind not in {"U", "S"}:
        raise ValueError(f"FER mixed {field_name} must be a one-dimensional string array.")
    normalized = array.astype(str)
    if any(not value for value in normalized.tolist()):
        raise ValueError(f"FER mixed {field_name} values must be nonempty.")
    if unique and len(set(normalized.tolist())) != count:
        raise ValueError(f"FER mixed {field_name} values must be unique.")
    return normalized


def _load_source(
    data_dir: Path,
    manifest: dict[str, Any],
    split_name: str,
    source: str,
    image_size: int,
) -> dict[str, np.ndarray]:
    """Load, validate, crop, and preprocess one source split."""
    split_spec = manifest["splits"][split_name]
    if not isinstance(split_spec, dict) or not isinstance(split_spec.get("sources"), dict):
        raise ValueError(f"FER mixed manifest split '{split_name}' is malformed.")
    if set(split_spec["sources"]) != set(SOURCES):
        raise ValueError(f"FER mixed manifest split '{split_name}' must contain exactly FER+ and RAF.")
    source_spec = split_spec["sources"].get(source)
    if not isinstance(source_spec, dict):
        raise ValueError(f"FER mixed manifest split '{split_name}' is missing source '{source}'.")
    expected_native = 48 if source == "ferplus" else 100
    if source_spec.get("native_size") != expected_native:
        raise ValueError(f"FER mixed {split_name}/{source} native size is invalid.")
    if not _is_sha256(source_spec.get("sha256")):
        raise ValueError(f"FER mixed {split_name}/{source} file hash is invalid.")
    filename = source_spec.get("file")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError(f"FER mixed {split_name}/{source} file name is invalid.")
    path = data_dir / filename
    if not path.is_file() or _sha256(path) != source_spec["sha256"]:
        raise ValueError(f"FER mixed {split_name}/{source} data hash does not match the manifest.")
    with np.load(path, allow_pickle=False) as loaded:
        required = {"images", "labels", "targets", "boxes", "ids", "canonical48_sha256"}
        if set(loaded.files) != required:
            raise ValueError(f"FER mixed {split_name}/{source} arrays do not match the schema.")
        arrays = {name: loaded[name].copy() for name in required}

    count = source_spec.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError(f"FER mixed {split_name}/{source} count is invalid.")
    images = arrays["images"]
    labels = arrays["labels"]
    targets = arrays["targets"]
    boxes = arrays["boxes"]
    if images.dtype != np.uint8 or images.shape != (count, expected_native, expected_native):
        raise ValueError(f"FER mixed {split_name}/{source} image shape or dtype is invalid.")
    if labels.dtype != np.int64 or labels.shape != (count,) or np.any((labels < 0) | (labels >= 5)):
        raise ValueError(f"FER mixed {split_name}/{source} labels are invalid.")
    if targets.shape != (count, 5) or targets.dtype != np.float32 or not np.isfinite(targets).all():
        raise ValueError(f"FER mixed {split_name}/{source} targets are invalid.")
    if np.any(targets < 0.0) or not np.allclose(targets.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError(f"FER mixed {split_name}/{source} targets must be probability vectors.")
    if not np.array_equal(np.argmax(targets, axis=1), labels):
        raise ValueError(f"FER mixed {split_name}/{source} target/label alignment is invalid.")
    if source == "ferplus" and np.any(np.max(targets, axis=1) <= 0.5):
        raise ValueError("FER mixed FER+ targets must preserve a strict selected-class majority.")
    if source == "raf" and not np.array_equal(targets, np.eye(5, dtype=np.float32)[labels]):
        raise ValueError("FER mixed RAF targets must be hard one-hot vectors.")
    if boxes.shape != (count, 3) or boxes.dtype.kind not in {"i", "u"}:
        raise ValueError(f"FER mixed {split_name}/{source} boxes are invalid.")
    ids = _validate_string_array(arrays["ids"], field_name="ids", count=count, unique=True)
    hashes = _validate_string_array(
        arrays["canonical48_sha256"],
        field_name="canonical48_sha256",
        count=count,
        unique=False,
    )
    expected_prefix = source + ":"
    if any(not value.startswith(expected_prefix) for value in ids.tolist()):
        raise ValueError(f"FER mixed {split_name}/{source} IDs have an invalid source prefix.")
    if any(not _is_sha256(value) for value in hashes.tolist()):
        raise ValueError(f"FER mixed {split_name}/{source} canonical hashes are invalid.")
    bx, by, side = (boxes[:, index].astype(np.int64) for index in range(3))
    if np.any(side < 1) or np.any(bx < 0) or np.any(by < 0):
        raise ValueError(f"FER mixed {split_name}/{source} crop boxes are invalid.")
    if np.any(bx + side > expected_native) or np.any(by + side > expected_native):
        raise ValueError(f"FER mixed {split_name}/{source} crop boxes exceed image bounds.")
    if source == "ferplus" and not np.array_equal(boxes, np.tile([0, 0, 48], (count, 1))):
        raise ValueError("FER mixed FER+ boxes must preserve the native 48-pixel image.")
    if (
        source == "raf"
        and manifest["crop_mode"] == "provided_aligned"
        and not np.array_equal(boxes, np.tile([0, 0, 100], (count, 1)))
    ):
        raise ValueError("Provided-aligned RAF boxes must preserve the full 100-pixel image.")

    if source == "ferplus":
        inputs = preprocess_grayscale(images, image_size)
    else:
        inputs = np.concatenate(
            [
                preprocess_grayscale(image[yy : yy + size, xx : xx + size][None, ...], image_size)
                for image, xx, yy, size in zip(images, bx, by, side)
            ],
            axis=0,
        )
    if source_spec.get("class_counts") != np.bincount(labels, minlength=5).tolist():
        raise ValueError(f"FER mixed {split_name}/{source} class counts do not match the manifest.")
    return {
        "inputs": inputs,
        "labels": labels,
        "training_targets": targets,
        "sources": np.repeat(np.asarray([source], dtype="<U7"), count),
        "ids": ids,
        "canonical48_sha256": hashes,
    }


def _load_split(
    data_dir: Path,
    manifest: dict[str, Any],
    split_name: str,
    image_size: int,
) -> DataSplit:
    """Load both sources in their fixed FER+-then-RAF order."""
    groups = [_load_source(data_dir, manifest, split_name, source, image_size) for source in SOURCES]
    combined = {
        key: np.concatenate([group[key] for group in groups], axis=0)
        for key in groups[0]
    }
    expected_count = manifest["splits"][split_name].get("count")
    if expected_count != len(combined["labels"]):
        raise ValueError(f"FER mixed split '{split_name}' count does not match the manifest.")
    if manifest["splits"][split_name].get("class_counts") != np.bincount(
        combined["labels"], minlength=5
    ).tolist():
        raise ValueError(f"FER mixed split '{split_name}' class counts do not match the manifest.")
    metadata = {
        "sources": combined["sources"],
        "ids": combined["ids"],
        "canonical48_sha256": combined["canonical48_sha256"],
    }
    if split_name in {"train", "val"}:
        metadata["training_targets"] = combined["training_targets"]
    return DataSplit(
        inputs=combined["inputs"],
        targets=combined["labels"],
        sample_weights=(equal_source_weights(combined["sources"]) if split_name == "train" else None),
        metadata=metadata,
    )


def _stable_group_indices(
    labels: np.ndarray,
    sources: np.ndarray,
    ids: np.ndarray,
    *,
    per_source_class: int,
    seed: int,
) -> np.ndarray:
    """Select resolution-independent rows by a stable hash of source IDs."""
    selected: list[int] = []
    for source in SOURCES:
        for label in range(len(CLASS_NAMES)):
            candidates = np.flatnonzero((sources == source) & (labels == label)).tolist()
            if not candidates:
                raise ValueError(f"FER mixed training data has no {source}/{CLASS_NAMES[label]} row.")
            candidates.sort(
                key=lambda index: hashlib.sha256(f"{seed}:{ids[index]}".encode()).hexdigest()
            )
            selected.extend(candidates[: min(per_source_class, len(candidates))])
    return np.asarray(selected, dtype=np.int64)


def _representative_indices(train: DataSplit, configured_ids: Sequence[str] | None) -> np.ndarray:
    """Select exactly one training example for every source/class pair."""
    labels = np.asarray(train.targets)
    sources = np.asarray(train.metadata["sources"])
    ids = np.asarray(train.metadata["ids"])
    if configured_ids is None:
        return _stable_group_indices(labels, sources, ids, per_source_class=1, seed=0)
    values = [str(value) for value in configured_ids]
    if len(values) != 10 or len(set(values)) != 10:
        raise ValueError("dataset.params.representative_ids must contain ten unique IDs.")
    positions = {value: index for index, value in enumerate(ids.tolist())}
    if any(value not in positions for value in values):
        raise ValueError("dataset.params.representative_ids contains an ID outside training data.")
    indices = np.asarray([positions[value] for value in values], dtype=np.int64)
    pairs = {(str(sources[index]), int(labels[index])) for index in indices}
    expected_pairs = {(source, label) for source in SOURCES for label in range(5)}
    if pairs != expected_pairs:
        raise ValueError("dataset.params.representative_ids must cover every source/class pair once.")
    return indices


def _validate_records(data_dir: Path, splits: Mapping[str, DataSplit]) -> None:
    """Cross-check the hashed record ledger against loaded split metadata."""
    expected: dict[str, tuple[str, int, str, str]] = {}
    for split_name, split in splits.items():
        for identifier, source, label, digest in zip(
            split.metadata["ids"],
            split.metadata["sources"],
            split.targets,
            split.metadata["canonical48_sha256"],
        ):
            expected[str(identifier)] = (str(source), int(label), str(digest), split_name)
    observed: dict[str, tuple[str, int, str, str]] = {}
    with (data_dir / "records.jsonl").open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
                identifier = str(record["id"])
                value = (
                    str(record["source"]),
                    int(record["label"]),
                    str(record["hash"]),
                    str(record["split"]),
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"FER mixed records.jsonl line {line_number} is malformed.") from exc
            if identifier in observed:
                raise ValueError("FER mixed records.jsonl contains duplicate IDs.")
            observed[identifier] = value
    if observed != expected:
        raise ValueError("FER mixed records.jsonl does not match prepared split metadata.")


def write_representative_fixture(
    bundle: DatasetBundle,
    npz_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Write the frozen backend-facing representative fixture contract.

    The NPZ keys are exactly ``inputs`` (float32 ``N,H,W``) and ``ids``
    (Unicode ``N``). Quantized tensors are deliberately not stored because the
    consumer must use each TFLite artifact's actual input scale and zero point.
    """
    indices = np.asarray(bundle.metadata.get("representative_fixture_indices"))
    if indices.ndim != 1 or len(indices) == 0 or indices.dtype.kind not in {"i", "u"}:
        raise ValueError("Dataset bundle is missing representative fixture indices.")
    inputs = np.asarray(bundle.train.inputs)[indices]
    ids = np.asarray(bundle.train.metadata["ids"])[indices].astype(str)
    if inputs.dtype != np.float32 or inputs.ndim != 3 or not np.isfinite(inputs).all():
        raise ValueError("Representative inputs must be finite float32 N,H,W tensors.")
    if np.any(inputs < -1.0) or np.any(inputs > 1.0):
        raise ValueError("Representative inputs must remain within [-1, 1].")
    if ids.shape != (len(indices),) or len(set(ids.tolist())) != len(indices):
        raise ValueError("Representative IDs must be unique training IDs.")
    npz_path = Path(npz_path)
    manifest_path = Path(manifest_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, inputs=inputs, ids=ids)
    payload = {
        "schema": FIXTURE_SCHEMA,
        "file": npz_path.name,
        "npz_sha256": _sha256(npz_path),
        "inputs_shape": list(inputs.shape),
        "inputs_dtype": str(inputs.dtype),
        "ids": ids.tolist(),
        "split": "train",
        "selection": "one stable example per source/class in ferplus-then-raf, class-order order",
        "dataset_identity": bundle.metadata["native_dataset_identity"],
        "preprocessing_identity": bundle.metadata["preprocessing_identity"],
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


class FERPlusRAFDBMixedDataset(DatasetABC):
    """Load the verified native-resolution mixed facial-expression cache."""

    @property
    def name(self) -> str:
        """Return the stable registry key."""
        return COMPONENT_NAME

    def validate_config(self, dataset_config: Any) -> None:
        """Validate dataset-local paths, resolution, and deterministic selection."""
        allowed = {
            "data_dir",
            "image_size",
            "crop_mode",
            "calibration_examples_per_source_class",
            "calibration_seed",
            "representative_ids",
        }
        unknown = sorted(_config_keys(dataset_config) - allowed)
        if unknown:
            raise ValueError(f"FER mixed dataset.params contains unsupported keys: {', '.join(unknown)}.")
        if _cfg_get(dataset_config, "data_dir", None) in (None, ""):
            raise ValueError("FER mixed dataset requires dataset.params.data_dir.")
        image_size = _cfg_get(dataset_config, "image_size", None)
        if isinstance(image_size, bool) or image_size not in SUPPORTED_IMAGE_SIZES:
            raise ValueError(f"dataset.params.image_size must be one of {SUPPORTED_IMAGE_SIZES}.")
        crop_mode = _cfg_get(dataset_config, "crop_mode", "provided_aligned")
        if crop_mode not in {"provided_aligned", "espressif_manifest"}:
            raise ValueError("dataset.params.crop_mode is unsupported.")
        _require_positive_integer(
            _cfg_get(dataset_config, "calibration_examples_per_source_class", 50),
            field_name="dataset.params.calibration_examples_per_source_class",
        )
        seed = _cfg_get(dataset_config, "calibration_seed", 42)
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("dataset.params.calibration_seed must be a nonnegative integer.")
        representative_ids = _cfg_get(dataset_config, "representative_ids", None)
        if representative_ids is not None and (
            isinstance(representative_ids, (str, bytes))
            or not isinstance(representative_ids, Sequence)
        ):
            raise ValueError("dataset.params.representative_ids must be a sequence of IDs.")

    def load(self, dataset_config: Any) -> DatasetBundle:
        """Load train/validation/test data and deterministic training calibration."""
        self.validate_config(dataset_config)
        data_dir = Path(str(_cfg_get(dataset_config, "data_dir"))).expanduser().resolve()
        image_size = int(_cfg_get(dataset_config, "image_size"))
        manifest, manifest_sha256 = _read_manifest(data_dir)
        requested_crop = str(_cfg_get(dataset_config, "crop_mode", "provided_aligned"))
        if requested_crop != manifest["crop_mode"]:
            raise ValueError("FER mixed config crop_mode does not match the prepared dataset.")
        train = _load_split(data_dir, manifest, "train", image_size)
        val = _load_split(data_dir, manifest, "val", image_size)
        test = _load_split(data_dir, manifest, "test", image_size)
        _validate_records(data_dir, {"train": train, "val": val, "test": test})
        all_ids = np.concatenate(
            [train.metadata["ids"], val.metadata["ids"], test.metadata["ids"]]
        ).tolist()
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("FER mixed IDs must be unique across train, validation, and test.")

        calibration_count = _require_positive_integer(
            _cfg_get(dataset_config, "calibration_examples_per_source_class", 50),
            field_name="dataset.params.calibration_examples_per_source_class",
        )
        calibration_seed = int(_cfg_get(dataset_config, "calibration_seed", 42))
        calibration_indices = _stable_group_indices(
            np.asarray(train.targets),
            np.asarray(train.metadata["sources"]),
            np.asarray(train.metadata["ids"]),
            per_source_class=calibration_count,
            seed=calibration_seed,
        )
        calibration = DataSplit(
            inputs=np.asarray(train.inputs)[calibration_indices],
            targets=np.asarray(train.targets)[calibration_indices],
            metadata={
                key: np.asarray(value)[calibration_indices]
                for key, value in train.metadata.items()
                if key != "training_targets"
            },
        )
        representative_indices = _representative_indices(
            train, _cfg_get(dataset_config, "representative_ids", None)
        )
        native_identity = {
            "manifest_sha256": manifest_sha256,
            "records_sha256": manifest["records_sha256"],
            "schema": DATASET_SCHEMA,
            "label_source": LABEL_SOURCE,
            "crop_mode": manifest["crop_mode"],
        }
        preprocessing_payload = {
            "native_dataset_identity": native_identity,
            "image_size": image_size,
            "policy": PREPROCESSING_POLICY,
            "dtype": "float32",
            "external_shape": [image_size, image_size],
        }
        preprocessing_identity = hashlib.sha256(
            json.dumps(preprocessing_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return DatasetBundle(
            train=train,
            val=val,
            test=test,
            calibration=calibration,
            input_shape=(image_size, image_size),
            input_dtype="float32",
            metadata={
                "dataset_name": COMPONENT_NAME,
                "data_dir": str(data_dir),
                "class_names": list(CLASS_NAMES),
                "num_classes": len(CLASS_NAMES),
                "label_encoding": "class_index",
                "label_source": LABEL_SOURCE,
                "crop_mode": manifest["crop_mode"],
                "image_size": image_size,
                "preprocessing_policy": PREPROCESSING_POLICY,
                "native_dataset_identity": native_identity,
                "preprocessing_identity": preprocessing_identity,
                "calibration_ids": calibration.metadata["ids"].tolist(),
                "representative_fixture_indices": representative_indices.tolist(),
                "representative_fixture_ids": train.metadata["ids"][representative_indices].tolist(),
            },
        )

    def make_calibration_data(
        self,
        bundle: DatasetBundle,
        dataset_config: Any,
    ) -> DataSplit | None:
        """Return the deterministic training-only representative split."""
        del dataset_config
        return bundle.calibration
