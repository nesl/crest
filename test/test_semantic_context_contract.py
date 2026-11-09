# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Canonical context hashes preserve persisted prompt identity across refactors."""
import hashlib
import json

import pytest

from crest.optimizers.context import build_semantic_context, disabled_semantic_context
from crest.optimizers.search_space import SearchParam, SearchSpaceDescriptor


def _context(*, seed=None, enabled=True):
    training = {"train": True, "quantization": {"mode": "float"}}
    if seed is not None:
        training.update(random_seed=seed, deterministic_ops=True, nas_epochs=5)
    return build_semantic_context(
        enabled=enabled,
        dataset_name="urbansound8k_mel",
        dataset_config={"sample_rate_hz": 16000, "mel_bins": 40},
        dataset_bundle={"input_shape": (98, 40, 1), "metadata": {"sample_rate_hz": 22050}},
        model_build_context={"input_dtype": "float32", "dataset_metadata": {"sample_rate_hz": 32000}},
        task_name="classification",
        target_spec={"task_type": "classification", "output_names": ["classes"], "output_shapes": [(10,)]},
        metric_contract={"primary_metric": "accuracy", "available_metrics": ["accuracy"]},
        score_config={"type": "scoring-function", "params": {"terms": [{"metric": "accuracy", "weight": 1}]}},
        feasibility_config={},
        model_family_name="audio_dscnn",
        descriptor=SearchSpaceDescriptor((
            SearchParam("width", "int", low=2, high=8, description="channel count", units="channels"),
            SearchParam("quantization_mode", "categorical", choices=("float", "int8_ptq")),
        )),
        device_config={"name": "NICLA_SENSE_ME", "hil": True, "compile_when_hil_disabled": "true"},
        training_config=training,
        collect_compile_metrics=True,
    )


@pytest.mark.parametrize("seed,expected_hash", [(None, "4f40473cc353d42d0db9b88f5b403779e25fb8abb81d112b858632ded2ec93f7"), (42, "fd3c02deb3ba5e81c8343537a2d88fa9d7ce27f6ed4fde0b6e88a7ca4a82792f")])
def test_semantic_payload_retains_original_canonical_hash(seed, expected_hash):
    """Digests were captured from context.py before the domain-module extraction."""
    context = _context(seed=seed)
    assert context["hash"] == expected_hash
    encoded = json.dumps(context["payload"], sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert context["hash"] == hashlib.sha256(encoded.encode()).hexdigest()
    assert context["payload"]["dataset_context"]["sampling_rate_hz"] == 32000


def test_disabled_context_is_identical_for_builder_and_shortcut():
    assert _context(enabled=False) == disabled_semantic_context()
