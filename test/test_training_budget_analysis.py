# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Offline training-budget regression checks with synthetic frozen trajectories."""

import importlib.util
import json
from pathlib import Path

import pytest

PATH = (
    Path(__file__).resolve().parents[1] / "analysis_scripts/appendix_training_budget/analyze.py"
)
spec = importlib.util.spec_from_file_location("training_budget_analysis", PATH)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def dump(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def panel(tmp_path):
    folder = tmp_path / "panel"
    folder.mkdir()
    (folder / "source_config.yaml").write_text("training: {}\n")
    (folder / "effective_config.yaml").write_text("training: {}\noutputs: {}\n")
    m = dict(
        budgets=[2, 4],
        seeds=[7, 19],
        source_csv_sha256="source",
        snapshot_sha256=analysis.sha256(folder / "source_config.yaml"),
        candidates=[],
    )
    for i in range(3):
        cid = f"c{i}"
        m["candidates"].append(
            dict(
                candidate_id=cid,
                replay_payload_key=cid,
                historical_energy_mj=i + 1,
                quantization_mode="float",
            )
        )
        for seed in m["seeds"]:
            w = folder / cid / f"seed_{seed}"
            w.mkdir(parents=True)
            h = [
                dict(epoch=e, val_loss=5 - e, elapsed_s=e * 10, virtual_stop_epoch=None)
                for e in range(1, 5)
            ]
            rows, cp = [], []
            for b in m["budgets"]:
                for p in analysis.POLICIES:
                    s = dict(
                        budget=b,
                        policy=p,
                        selected_epoch=b,
                        checkpoint=f"{b}_{p}.keras",
                        sha256="a" * 64,
                        virtual_stop_epoch=None,
                    )
                    cp.append(s)
                    rows.append(
                        dict(
                            s,
                            candidate_id=cid,
                            seed=seed,
                            status="ok",
                            energy_policy="fixed_historical_measurement",
                            energy_mj_per_inference=i + 1,
                            quantization_mode="float",
                            metrics=dict(rmse_total=3 - i + (4 - b) / 10),
                        )
                    )
            for name, obj in [
                ("history.json", h),
                ("results.json", rows),
                ("checkpoints.json", cp),
                ("complete.json", dict(elapsed_s=9999)),
            ]:
                dump(w / name, obj)
    dump(folder / "manifest.json", m)
    dump(
        folder / "execution.json",
        dict(
            identity=dict(
                manifest_sha256=analysis.sha256(folder / "manifest.json"),
                dataset_dir="external",
                split_hashes={},
            ),
            effective_config_sha256=analysis.sha256(folder / "effective_config.yaml"),
            test_loaded=False,
            tensorflow_version="unused",
            python="synthetic",
            train_shape=[1],
            validation_shape=[1],
        ),
    )
    return folder


def test_arbitrary_panel_seeds_reference_and_prefix_timing(panel, tmp_path):
    s = analysis.analyze([panel], tmp_path / "out", reference_budget=4)
    assert s["candidate_count"] == 3
    assert set(s["seeds"]) == {"7", "19"}
    assert s["seeds"]["7"]["cumulative_training_hours"]["2"] == 60 / 3600
    assert (tmp_path / "out/figure10_training_budget_seed_19.csv").is_file()
    analysis.analyze(
        [panel],
        tmp_path / "one",
        reference_budget=2,
        reference_policy="original_policy",
        seeds=[19],
    )
    assert (tmp_path / "one/figure10_training_budget.csv").read_text().splitlines()[
        0
    ] == "candidate_id,budget,rmse_total,energy_mj,source"


@pytest.mark.parametrize(
    "kind",
    [
        "duplicate",
        "incomplete",
        "history",
        "selection",
        "energy",
        "checkpoint",
        "metric",
        "hash",
        "test_loaded",
    ],
)
def test_reject_corrupt_panel(panel, kind):
    w = panel / "c0/seed_7"
    if kind == "hash":
        (panel / "source_config.yaml").write_text("changed")
    elif kind == "test_loaded":
        e = analysis.read(panel / "execution.json")
        e["test_loaded"] = True
        dump(panel / "execution.json", e)
    elif kind == "incomplete":
        (w / "complete.json").unlink()
    elif kind == "history":
        h = analysis.read(w / "history.json")
        h[1]["elapsed_s"] = -1
        dump(w / "history.json", h)
    else:
        rows = analysis.read(w / "results.json")
        if kind == "duplicate":
            rows.append(rows[0])
        if kind == "selection":
            rows[0]["selected_epoch"] = 1
        if kind == "energy":
            rows[0]["energy_mj_per_inference"] = 99
        if kind == "checkpoint":
            (w / rows[0]["checkpoint"]).write_bytes(b"bad checkpoint")
        if kind == "metric":
            rows[0]["metrics"]["rmse_total"] = float("nan")
        dump(w / "results.json", rows)
    with pytest.raises(ValueError):
        analysis.load_panels([panel])


def test_ties_fronts_regret_and_coverage():
    early = {"a": (1, 1), "b": (1, 2), "c": (3, 3)}
    final = {"a": (3, 1), "b": (2, 2), "c": (1, 3)}
    result = analysis.reference_comparison(early, final)
    assert result["tied"] == 1 and result["discordant"] == 2
    assert result["early_front"] == ["a"]
    assert result["regret"][-1]["absolute_regret"] == 2
    assert (
        analysis.reference_comparison({"a": (0, 1)}, {"a": (0, 1)})["spearman_rho"]
        is None
    )


def test_duplicate_panels_rejected(panel):
    with pytest.raises(ValueError, match="Duplicate candidate"):
        analysis.load_panels([panel, panel])
