# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Synthetic replay matching without private study artifacts."""

import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).parents[1] / "analysis_scripts/cs3_audit/replay.py"
SPEC = importlib.util.spec_from_file_location("cs3_replay", PATH)
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


def inputs():
    """Return a selected source and a replay with a different CSV row index."""
    source = dict(
        board="board-a",
        method="search-x",
        trial=12,
        study_name="experiment",
        f1=0.8,
        energy_mj=10,
        raw_params_json=json.dumps({"width": 4}),
        hyperparameters_json=json.dumps({"width": 4}),
    )
    measured = {
        "source__study_name": "experiment",
        "source_row_index": "11",
        "source__metric__macro_f1": ".8",
        "family_hparams_json": '{"width":4}',
        "target__error_code": "1",
        "target__energy_mj_per_inference": "7",
    }
    return source, measured


def test_trial_identity_does_not_use_csv_position():
    source, measured = inputs()
    rows = replay.audit([source], [("board-b", [measured])])
    assert [r["trial"] for r in rows] == [12, 12]
    assert rows[1]["score"] == pytest.approx(0.8 - 0.1 * 7 / 400)
    assert rows[1]["model_variant"] == "not recorded"


@pytest.mark.parametrize(
    "field,value",
    [
        ("family_hparams_json", '{"width":8}'),
        ("source__metric__macro_f1", ".7"),
        ("target__error_code", "2"),
        ("target__energy_mj_per_inference", "nan"),
    ],
)
def test_rejects_mismatched_or_failed_replays(field, value):
    source, measured = inputs()
    measured[field] = value
    with pytest.raises(ValueError):
        replay.audit([source], [("board-b", [measured])])
