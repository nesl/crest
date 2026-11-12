# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Focused tests for the mixed FER dataset and probability task."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
import yaml
from addict import Dict

from crest.builtin_components import ensure_fer_components_registered
from crest.component_selection import resolve_component_selection
from crest.datasets.ferplus_rafdb_mixed import (
    CLASS_NAMES,
    FERPlusRAFDBMixedDataset,
    equal_source_weights,
    preprocess_grayscale,
    write_representative_fixture,
)
from crest.model import load_config
from crest.pipeline_types import DataSplit
from crest.registry import dataset_registry, model_family_registry, task_registry
from crest.runtime_bootstrap import bootstrap_pipeline
from crest.tasks.facial_expression_classification import FacialExpressionClassificationTask


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_payload(source: str, split: str) -> dict[str, np.ndarray]:
    native = 48 if source == "ferplus" else 100
    labels = np.arange(5, dtype=np.int64)
    images = np.stack(
        [np.full((native, native), 20 + 30 * label + (1 if source == "raf" else 0), np.uint8) for label in labels]
    )
    if source == "raf":
        images[0] = (np.arange(native * native).reshape(native, native) % 256).astype(np.uint8)
    targets = np.eye(5, dtype=np.float32)[labels]
    if source == "ferplus":
        targets = targets * np.float32(0.75) + np.roll(targets, 1, axis=1) * np.float32(0.25)
    boxes = np.tile([0, 0, native], (5, 1)).astype(np.int64)
    ids = np.asarray([f"{source}:{split}_{label}" for label in labels])
    hashes = np.asarray([hashlib.sha256(images[label].tobytes()).hexdigest() for label in labels])
    return {
        "images": images,
        "labels": labels,
        "targets": targets,
        "boxes": boxes,
        "ids": ids,
        "canonical48_sha256": hashes,
    }


def _write_dataset(path: Path) -> None:
    path.mkdir()
    records = []
    manifest = {
        "classes": list(CLASS_NAMES),
        "label_source": "FER+_RAF-DB_mixed",
        "schema": "mixed_native_v1",
        "crop_mode": "provided_aligned",
        "detector_boxes_sha256": None,
        "fer_native_size": 48,
        "raf_native_size": 100,
        "source_hashes": {"fer_manifest_sha256": "0" * 64},
        "splits": {},
    }
    for split in ("train", "val", "test"):
        split_sources = {}
        for source in ("ferplus", "raf"):
            payload = _source_payload(source, split)
            filename = f"{split}_{source}.npz"
            np.savez_compressed(path / filename, **payload)
            split_sources[source] = {
                "file": filename,
                "count": 5,
                "sha256": _sha256(path / filename),
                "native_size": 48 if source == "ferplus" else 100,
                "class_counts": [1, 1, 1, 1, 1],
            }
            records.extend(
                json.dumps(
                    {
                        "id": identifier,
                        "source": source,
                        "label": int(label),
                        "hash": digest,
                        "split": split,
                    }
                )
                + "\n"
                for identifier, label, digest in zip(
                    payload["ids"].tolist(),
                    payload["labels"].tolist(),
                    payload["canonical48_sha256"].tolist(),
                )
            )
        manifest["splits"][split] = {
            "count": 10,
            "sources": split_sources,
            "class_counts": [2, 2, 2, 2, 2],
        }
    (path / "records.jsonl").write_text("".join(records), encoding="utf-8")
    manifest["records_sha256"] = _sha256(path / "records.jsonl")
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _config(path: Path, image_size: int = 64) -> Dict:
    return Dict(
        data_dir=str(path),
        image_size=image_size,
        crop_mode="provided_aligned",
        calibration_examples_per_source_class=1,
        calibration_seed=42,
    )


def test_builtin_registration_and_fixed_resolution_fragments() -> None:
    ensure_fer_components_registered()
    assert dataset_registry.get("ferplus_rafdb_mixed") is FERPlusRAFDBMixedDataset
    assert task_registry.get("facial_expression_classification") is FacialExpressionClassificationTask
    assert model_family_registry.get("fer_mini_xception").__name__ == "FERMiniXceptionFamily"
    config_dir = Path(__file__).resolve().parents[1] / "src" / "config"
    for size in (64, 96):
        raw = yaml.safe_load((config_dir / f"fer_components_{size}.yaml").read_text())
        selection = resolve_component_selection(Dict(raw))
        assert selection["dataset_config"].image_size == size
        assert selection["task_config"].runtime_training_batch_size == 64
        assert "runtime_training_batch_size" not in raw["task"]["params"]


@pytest.mark.parametrize("explicit_fixed_split", [False, True])
def test_loaded_composed_hci_config_bootstraps_fixed_split_probability_pipeline(
    tmp_path: Path,
    explicit_fixed_split: bool,
) -> None:
    data_dir = tmp_path / "mixed"
    _write_dataset(data_dir)
    config_dir = Path(__file__).resolve().parents[1] / "src" / "config"
    raw = yaml.safe_load((config_dir / "fer_components_96.yaml").read_text())
    raw["dataset"]["params"].update(
        data_dir=str(data_dir),
        calibration_examples_per_source_class=1,
    )
    if explicit_fixed_split:
        raw["task"]["params"]["evaluation"] = {"protocol": "fixed_split"}
    raw["device"] = {
        "name": "FREENOVE_ESP32_S3_WROOM",
        "hil": True,
        "compile_when_hil_disabled": "auto",
        "runtime_mode": "back_to_back",
        "measured_inference_runs": 10,
    }
    raw["training"].update(
        nas_trials=1,
        train=True,
        energy_aware=False,
        quantization={"mode": "int8_ptq", "search": False, "choices": ["int8_ptq"]},
    )
    raw["nas"] = {
        "hil_first": False,
        "score": {
            "type": "scoring-function",
            "params": {
                "terms": [{"type": "weighted", "metric": "macro_f1", "weight": 1.0}]
            },
        },
        "prune": {"rules": []},
        "feasibility": {"train_if_infeasible": False, "rules": []},
    }
    raw["outputs"] = {
        "models_dir": str(tmp_path / "models"),
        "candidate_dir": str(tmp_path / "candidates"),
        "artifact_stem": "CREST_FER96_Test",
    }
    config_path = tmp_path / "composed_fer96.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    loaded = load_config(config_path)
    assert loaded.task.params.evaluation.protocol == "fixed_split"
    assert loaded.task.params.evaluation.fold_rotation.test_folds == list(range(1, 11))
    ensure_fer_components_registered()
    pipeline = bootstrap_pipeline(loaded)
    task_config = pipeline.selection["task_config"]
    assert task_config.runtime_training_batch_size == 64
    assert task_config.evaluation.protocol == "fixed_split"
    assert pipeline.target_spec.output_names == ["probabilities"]
    assert pipeline.target_spec.metadata["from_logits"] is False
    assert pipeline.model_build_context.input_shape == (96, 96)

    plan = pipeline.task.build_fit_plan(
        pipeline.bundle,
        task_config,
        pipeline.target_spec,
        mode="search",
        combine_train_val=False,
    )
    batches = list(plan.fit_kwargs["x"].as_numpy_iterator())
    assert len(batches) == 1
    assert batches[0][0].shape == (10, 96, 96)
    assert batches[0][1].shape == (10, 5)
    assert plan.monitor_metric == "val_loss"


def test_facial_expression_task_explicitly_rejects_fold_rotation(tmp_path: Path) -> None:
    task = FacialExpressionClassificationTask(checkpoint_path=tmp_path / "best.keras")
    with pytest.raises(ValueError, match="only supports fixed_split"):
        task.validate_config(Dict(evaluation=Dict(protocol="fold_rotation")))


def test_loader_preserves_source_order_soft_targets_weights_and_resolution_identity(tmp_path: Path) -> None:
    data_dir = tmp_path / "mixed"
    _write_dataset(data_dir)
    dataset = FERPlusRAFDBMixedDataset()
    bundle64 = dataset.load(_config(data_dir, 64))
    bundle96 = dataset.load(_config(data_dir, 96))

    assert bundle64.input_shape == (64, 64)
    assert bundle64.train.inputs.shape == (10, 64, 64)
    assert bundle64.train.inputs.dtype == np.float32
    assert bundle64.train.metadata["sources"].tolist() == ["ferplus"] * 5 + ["raf"] * 5
    assert bundle64.train.metadata["training_targets"].shape == (10, 5)
    assert np.array_equal(
        bundle64.train.metadata["training_targets"][0],
        np.asarray([0.75, 0.25, 0.0, 0.0, 0.0], np.float32),
    )
    assert np.allclose(bundle64.train.sample_weights[:5].sum(), 5.0)
    assert np.allclose(bundle64.train.sample_weights[5:].sum(), 5.0)
    assert bundle64.metadata["native_dataset_identity"] == bundle96.metadata["native_dataset_identity"]
    assert bundle64.metadata["preprocessing_identity"] != bundle96.metadata["preprocessing_identity"]
    assert bundle64.metadata["representative_fixture_ids"] == bundle96.metadata["representative_fixture_ids"]
    assert bundle64.calibration.metadata["ids"].tolist() == bundle96.calibration.metadata["ids"].tolist()
    native_raf = _source_payload("raf", "train")["images"][0]
    rows = np.arange(64) * 100 // 64
    columns = np.arange(64) * 100 // 64
    expected_raf = native_raf[rows][:, columns].astype(np.float32) / np.float32(127.5) - 1.0
    assert np.array_equal(bundle64.train.inputs[5], expected_raf)


def test_preprocess_nearest_floor_rule_and_unequal_source_weights() -> None:
    image = (np.arange(15).reshape(3, 5) * 17).astype(np.uint8)
    actual = preprocess_grayscale(image[None, ...], 64)[0]
    rows = np.arange(64) * 3 // 64
    columns = np.arange(64) * 5 // 64
    expected = image[rows][:, columns].astype(np.float32) / np.float32(127.5) - 1.0
    assert np.array_equal(actual, expected)
    sources = np.asarray(["ferplus", "ferplus", "ferplus", "raf"])
    weights = equal_source_weights(sources)
    assert np.allclose(weights[:3], 2.0 / 3.0)
    assert np.isclose(weights[:3].sum(), 2.0)
    assert np.isclose(weights[3], 2.0)


def test_loader_rejects_schema_drift(tmp_path: Path) -> None:
    data_dir = tmp_path / "mixed"
    _write_dataset(data_dir)
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["classes"] = list(reversed(CLASS_NAMES))
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="classes"):
        FERPlusRAFDBMixedDataset().load(_config(data_dir))


def test_loader_rejects_source_file_hash_drift(tmp_path: Path) -> None:
    data_dir = tmp_path / "mixed"
    _write_dataset(data_dir)
    with (data_dir / "train_raf.npz").open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="data hash"):
        FERPlusRAFDBMixedDataset().load(_config(data_dir))


def test_loader_rejects_nonfull_provided_aligned_raf_crop(tmp_path: Path) -> None:
    data_dir = tmp_path / "mixed"
    _write_dataset(data_dir)
    path = data_dir / "train_raf.npz"
    with np.load(path, allow_pickle=False) as loaded:
        payload = {name: loaded[name].copy() for name in loaded.files}
    payload["boxes"][0] = [0, 0, 99]
    np.savez_compressed(path, **payload)
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["splits"]["train"]["sources"]["raf"]["sha256"] = _sha256(path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="full 100-pixel"):
        FERPlusRAFDBMixedDataset().load(_config(data_dir))


def test_representative_fixture_contract_is_exact_and_self_consistent(tmp_path: Path) -> None:
    data_dir = tmp_path / "mixed"
    _write_dataset(data_dir)
    bundle = FERPlusRAFDBMixedDataset().load(_config(data_dir))
    npz_path = tmp_path / "representative_64.npz"
    manifest_path = tmp_path / "representative_64.json"
    manifest = write_representative_fixture(bundle, npz_path, manifest_path)
    with np.load(npz_path, allow_pickle=False) as loaded:
        assert set(loaded.files) == {"inputs", "ids"}
        assert loaded["inputs"].shape == (10, 64, 64)
        assert loaded["inputs"].dtype == np.float32
        assert loaded["ids"].tolist() == manifest["ids"]
    assert manifest["npz_sha256"] == _sha256(npz_path)
    assert manifest["inputs_shape"] == [10, 64, 64]
    assert manifest["inputs_dtype"] == "float32"


def test_task_prebatched_augmentation_restores_external_shape_and_keeps_tail(tmp_path: Path) -> None:
    data_dir = tmp_path / "mixed"
    _write_dataset(data_dir)
    bundle = FERPlusRAFDBMixedDataset().load(_config(data_dir))
    task = FacialExpressionClassificationTask(checkpoint_path=tmp_path / "best.keras")
    task_config = Dict(
        runtime_training_batch_size=6,
        augmentation_seed=2026,
        learning_rate=0.001,
        lr_patience=2,
        report_validation_macro_f1=False,
    )
    target_spec = task.build_target_spec(bundle, task_config)
    plan = task.build_fit_plan(
        bundle, task_config, target_spec, mode="search", combine_train_val=False
    )
    batches = list(plan.fit_kwargs["x"].as_numpy_iterator())
    assert [batch[0].shape for batch in batches] == [(6, 64, 64), (4, 64, 64)]
    assert all(batch[1].shape[-1] == 5 for batch in batches)
    assert all(np.isfinite(batch[0]).all() for batch in batches)
    assert plan.monitor_metric == "val_loss"
    checkpoint = next(
        callback for callback in plan.callbacks if isinstance(callback, tf.keras.callbacks.ModelCheckpoint)
    )
    assert checkpoint.monitor == "val_loss"


def test_task_compile_recipe_and_keras_loss_includes_regularization(tmp_path: Path) -> None:
    inputs = tf.keras.layers.Input((2,), name="features")
    probabilities = tf.keras.layers.Dense(
        5,
        activation="softmax",
        kernel_initializer=tf.keras.initializers.Constant(0.2),
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        name="probabilities",
    )(inputs)
    model = tf.keras.Model(inputs, probabilities)
    task = FacialExpressionClassificationTask(checkpoint_path=tmp_path / "best.keras")
    task.compile_model(model, Dict(learning_rate=0.001), None)

    assert isinstance(model.optimizer, tf.keras.optimizers.Adam)
    assert np.isclose(float(model.optimizer.learning_rate.numpy()), 0.001)
    assert isinstance(model.loss, tf.keras.losses.CategoricalCrossentropy)
    assert model.loss.from_logits is False

    features = np.asarray([[1.0, 2.0], [-1.0, 0.5]], np.float32)
    targets = np.eye(5, dtype=np.float32)[[0, 3]]
    predictions = model(features, training=False)
    prediction_cce = float(
        tf.reduce_mean(tf.keras.losses.categorical_crossentropy(targets, predictions)).numpy()
    )
    regularization = float(tf.add_n(model.losses).numpy())
    evaluated = model.evaluate(features, targets, verbose=0, return_dict=True)
    assert regularization > 0.0
    assert np.isclose(evaluated["loss"], prediction_cce + regularization, atol=1e-6)


def test_effective_batch64_and_partial_tail_flow_through_runner_fit_kwargs(tmp_path: Path) -> None:
    from nas_model_client import _runner_fit_kwargs

    data_dir = tmp_path / "mixed"
    _write_dataset(data_dir)
    bundle = FERPlusRAFDBMixedDataset().load(_config(data_dir))
    repeats = 7
    bundle.train.inputs = np.tile(bundle.train.inputs, (repeats, 1, 1))
    bundle.train.targets = np.tile(bundle.train.targets, repeats)
    bundle.train.sample_weights = np.tile(bundle.train.sample_weights, repeats)
    for key in ("sources", "ids", "canonical48_sha256", "training_targets"):
        bundle.train.metadata[key] = np.tile(
            bundle.train.metadata[key], (repeats, 1) if key == "training_targets" else repeats
        )
    task = FacialExpressionClassificationTask(checkpoint_path=tmp_path / "best.keras")
    config = Dict(runtime_training_batch_size=64, report_validation_macro_f1=False)
    spec = task.build_target_spec(bundle, config)
    plan = task.build_fit_plan(bundle, config, spec, mode="search", combine_train_val=False)
    batches = list(plan.fit_kwargs["x"].as_numpy_iterator())
    assert [batch[0].shape[0] for batch in batches] == [64, 6]
    fit_kwargs = _runner_fit_kwargs(plan, epochs=1, batch_size=64)
    assert "batch_size" not in fit_kwargs
    assert fit_kwargs["x"] is plan.fit_kwargs["x"]


def test_task_final_combines_soft_train_hard_val_and_reweights_sources(tmp_path: Path) -> None:
    data_dir = tmp_path / "mixed"
    _write_dataset(data_dir)
    bundle = FERPlusRAFDBMixedDataset().load(_config(data_dir))
    task = FacialExpressionClassificationTask(checkpoint_path=tmp_path / "best.keras")
    config = Dict(runtime_training_batch_size=64, report_validation_macro_f1=False)
    spec = task.build_target_spec(bundle, config)
    plan = task.build_fit_plan(bundle, config, spec, mode="final", combine_train_val=True)
    batch = next(iter(plan.fit_kwargs["x"].as_numpy_iterator()))
    assert batch[0].shape == (20, 64, 64)
    assert np.allclose(batch[2][:10].sum() + batch[2][10:].sum(), 20.0)
    assert plan.monitor_metric == "loss"
    assert "validation_data" not in plan.fit_kwargs


def test_task_accepts_dequantized_softmax_rounding_and_source_subsets(tmp_path: Path) -> None:
    task = FacialExpressionClassificationTask(checkpoint_path=tmp_path / "best.keras")
    inputs = np.zeros((2, 64, 64), np.float32)
    split = DataSplit(
        inputs=inputs,
        targets=np.asarray([0, 1], np.int64),
        metadata={"sources": np.asarray(["ferplus", "ferplus"])},
    )
    predictions = np.asarray([[255 / 256, 0, 0, 0, 0], [0, 255 / 256, 0, 0, 0]], np.float32)
    result = task.evaluate_predictions(predictions, split, Dict(), None)
    assert result.metrics["accuracy"] == 1.0
    assert result.artifacts["sources"]["raf"] is None
    assert result.artifacts["equal_source_mean_macro_f1"] is None
    assert np.asarray(result.predictions).shape == (2, 5)
    assert "excludes model regularization" in result.artifacts["loss_semantics"]


@pytest.fixture(autouse=True)
def clear_tensorflow_session():
    yield
    tf.keras.backend.clear_session()
