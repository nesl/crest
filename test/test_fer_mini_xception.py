# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Focused decoder, reference-parity, reload, and INT8 checks for FER models."""

from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path

import numpy as np
import optuna
import pytest
import tensorflow as tf
from addict import Dict

from crest.hardware import convert_to_tflite_model, predict_tflite_model
from crest.model_families.fer_mini_xception import (
    DEFAULT_FER_MINI_XCEPTION_SEED,
    FER_MINI_XCEPTION_SEARCH_CHOICES,
    FERMiniXceptionFamily,
    decode_attention_indices,
    decode_block_strides,
    decode_block_widths,
    decoded_architecture_identity,
)
from crest.optimizers.search_space import SearchParam
from crest.pipeline_types import ModelBuildContext, TargetSpec

HCI_MODEL_SOURCE = Path("/home/joseph/Projects/HCI-final-project/training/hci_fer/model.py")


def _context(image_size: int = 96) -> ModelBuildContext:
    return ModelBuildContext(
        input_shape=(image_size, image_size),
        input_dtype="float32",
        target_spec=TargetSpec(
            task_type="classification",
            output_names=["probabilities"],
            output_shapes=[(5,)],
            metadata={"num_classes": 5, "from_logits": False},
        ),
    )


def _config(**params) -> Dict:
    return Dict(
        family="fer_mini_xception",
        params=Dict(
            export_variant="untrained",
            stem_convolutions=2,
            separable_convolutions_per_block=2,
            se_reduction_ratio=8,
            l2=0.001,
            **params,
        ),
        search=Dict(),
    )


def test_all_3456_raw_combinations_decode_valid_structures_and_canonical_identities() -> None:
    names = tuple(FER_MINI_XCEPTION_SEARCH_CHOICES)
    identities: dict[tuple, dict] = {}
    count = 0
    for choices in itertools.product(*(FER_MINI_XCEPTION_SEARCH_CHOICES[name] for name in names)):
        raw = dict(zip(names, choices))
        widths = decode_block_widths(raw)
        strides = decode_block_strides(raw)
        attention = decode_attention_indices(raw)
        identity = decoded_architecture_identity(raw)
        assert len(widths) == raw["num_blocks"]
        assert len(strides) == raw["num_blocks"]
        assert all(1 <= width <= raw["channel_cap"] for width in widths)
        assert all(stride in {1, 2} for stride in strides)
        assert all(0 <= index < raw["num_blocks"] for index in attention)
        signature = (
            raw["stem_channels"],
            tuple(widths),
            tuple(strides),
            raw["kernel_size"],
            tuple(attention),
        )
        if signature in identities:
            assert identity == identities[signature]
        else:
            identities[signature] = identity
        count += 1
    assert count == 3456
    assert len(identities) < count


def test_family_descriptor_fixed_trial_sampling_persisted_decode_and_search_override() -> None:
    family = FERMiniXceptionFamily()
    context = _context()
    config = _config()
    declarations = family.trial_search_space(context, config)
    assert all(isinstance(declaration, SearchParam) for declaration in declarations)
    assert [declaration.name for declaration in declarations] == list(
        FER_MINI_XCEPTION_SEARCH_CHOICES
    )
    assert all(declaration.kind == "categorical" for declaration in declarations)
    assert {
        declaration.name: declaration.choices for declaration in declarations
    } == FER_MINI_XCEPTION_SEARCH_CHOICES

    sampled = family.sample_hparams(
        optuna.trial.FixedTrial(DEFAULT_FER_MINI_XCEPTION_SEED), context, config
    )
    assert sampled == DEFAULT_FER_MINI_XCEPTION_SEED
    persisted = {**sampled, "channel_growth": 2}
    decoded = family.decode_trial_hparams(persisted, context, config)
    assert decoded == DEFAULT_FER_MINI_XCEPTION_SEED
    assert type(decoded["channel_growth"]) is float

    overridden = _config()
    overridden.search = Dict(kernel_size=[5])
    override_declarations = {
        declaration.name: declaration
        for declaration in family.trial_search_space(context, overridden)
    }
    assert override_declarations["kernel_size"].choices == (5,)
    assert override_declarations["num_blocks"].choices == FER_MINI_XCEPTION_SEARCH_CHOICES[
        "num_blocks"
    ]
    override_seed = {**DEFAULT_FER_MINI_XCEPTION_SEED, "kernel_size": 5}
    override_sampled = family.sample_hparams(
        optuna.trial.FixedTrial(override_seed), context, overridden
    )
    assert family.decode_trial_hparams(override_sampled, context, overridden) == override_seed


def test_reference_architecture_matches_selected_hci_shape_and_parameter_count() -> None:
    family = FERMiniXceptionFamily()
    model = family.build_model(DEFAULT_FER_MINI_XCEPTION_SEED, _context(), _config())
    assert model.input_shape == (None, 96, 96)
    assert model.output_shape == (None, 5)
    assert model.output_names == ["probabilities"]
    assert model.count_params() == 61453
    assert decode_block_widths(DEFAULT_FER_MINI_XCEPTION_SEED) == (16, 32, 64, 128)
    assert decode_attention_indices(DEFAULT_FER_MINI_XCEPTION_SEED) == (2, 3)
    assert isinstance(model.get_layer("face_gray_channel"), tf.keras.layers.Reshape)
    assert isinstance(model.layers[-1], tf.keras.layers.Softmax)


@pytest.mark.parametrize(
    "overrides,image_size",
    [
        ({"num_blocks": 3, "downsampling": "preserve_first", "attention_placement": "none"}, 64),
        ({"kernel_size": 5, "attention_placement": "last"}, 64),
        ({"stem_channels": 16, "first_block_channels": 32, "channel_growth": 1.5, "channel_cap": 64}, 96),
    ],
)
def test_representative_boundary_models_build_and_emit_probabilities(overrides: dict, image_size: int) -> None:
    raw = {**DEFAULT_FER_MINI_XCEPTION_SEED, **overrides}
    model = FERMiniXceptionFamily().build_model(raw, _context(image_size), _config())
    values = np.asarray(model(np.zeros((2, image_size, image_size), np.float32), training=False))
    assert values.shape == (2, 5)
    assert np.allclose(values.sum(axis=1), 1.0)


def test_reference_model_save_reload_preserves_outputs(tmp_path: Path) -> None:
    family = FERMiniXceptionFamily()
    model = family.build_model(DEFAULT_FER_MINI_XCEPTION_SEED, _context(), _config())
    inputs = np.random.default_rng(2026).uniform(-1, 1, (2, 96, 96)).astype(np.float32)
    expected = np.asarray(model(inputs, training=False))
    path = tmp_path / "reference.keras"
    model.save(path)
    loaded = family.load_model(path, _context(), _config())
    assert np.allclose(np.asarray(loaded(inputs, training=False)), expected, atol=1e-7)


@pytest.mark.skipif(not HCI_MODEL_SOURCE.is_file(), reason="audited HCI source is not available")
def test_reference_float_outputs_match_audited_hci_source_after_weight_transfer() -> None:
    spec = importlib.util.spec_from_file_location("audited_hci_fer_model", HCI_MODEL_SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    reference = module.build_model(image_size=96, l2=0.001, architecture="se")
    candidate = FERMiniXceptionFamily().build_model(
        DEFAULT_FER_MINI_XCEPTION_SEED, _context(), _config()
    )
    for layer in reference.layers:
        if layer.weights:
            candidate.get_layer(layer.name).set_weights(layer.get_weights())
    inputs = np.random.default_rng(2026).uniform(-1, 1, (3, 96, 96)).astype(np.float32)
    expected = np.asarray(reference(inputs[..., None], training=False))
    actual = np.asarray(candidate(inputs, training=False))
    assert reference.count_params() == candidate.count_params() == 61453
    assert np.array_equal(actual, expected)


def test_existing_crest_int8_export_has_supported_ops_and_probability_parity(tmp_path: Path) -> None:
    family = FERMiniXceptionFamily()
    raw = {
        **DEFAULT_FER_MINI_XCEPTION_SEED,
        "num_blocks": 3,
        "first_block_channels": 8,
        "channel_growth": 1.0,
        "channel_cap": 64,
        "attention_placement": "none",
    }
    model = family.build_model(raw, _context(64), _config())
    class_maps = model.get_layer("class_maps")
    kernel, bias = class_maps.get_weights()
    kernel.fill(0.0)
    bias[:] = np.asarray([1.0, 0.0, -1.0, 2.0, -2.0], dtype=np.float32)
    class_maps.set_weights([kernel, bias])
    calibration = np.random.default_rng(42).uniform(-1, 1, (10, 64, 64)).astype(np.float32)
    output_path = tmp_path / "model_int8.tflite"
    convert_to_tflite_model(model, calibration, "int8_ptq", output_path)
    interpreter = tf.lite.Interpreter(model_path=str(output_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    assert input_detail["dtype"] == np.int8
    assert output_detail["dtype"] == np.int8
    operators = {
        entry["op_name"]
        for entry in interpreter._get_ops_details()
        if entry["op_name"] != "DELEGATE"
    }
    assert operators <= {
        "ADD",
        "AVERAGE_POOL_2D",
        "CONV_2D",
        "DEPTHWISE_CONV_2D",
        "MAX_POOL_2D",
        "PACK",
        "RESHAPE",
        "SHAPE",
        "SOFTMAX",
        "STRIDED_SLICE",
    }
    assert {"RESHAPE", "SHAPE", "PACK", "STRIDED_SLICE"} <= operators
    inputs = calibration[:3]
    float_predictions = np.asarray(model(inputs, training=False))
    int8_predictions = np.asarray(predict_tflite_model(output_path, inputs))
    assert np.array_equal(float_predictions.argmax(axis=1), int8_predictions.argmax(axis=1))
    assert float(np.max(np.abs(float_predictions - int8_predictions))) < 0.05


@pytest.fixture(autouse=True)
def clear_tensorflow_session():
    yield
    tf.keras.backend.clear_session()
