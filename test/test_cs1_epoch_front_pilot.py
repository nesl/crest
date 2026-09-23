# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Local synthetic tests: no real dataset, training job, network or HIL."""
import argparse
import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "analysis_scripts/cs1_epoch_sensitivity/epoch_front_pilot.py"
spec = importlib.util.spec_from_file_location("epoch_front_pilot", SCRIPT)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


@pytest.fixture
def source(tmp_path):
    folder = tmp_path / "source"
    folder.mkdir()
    config = {
        "device": {"name": "PORTENTA_H7", "runtime_mode": "back_to_back"},
        "dataset": {"name": "oxiod"}, "model": {"family": "odom_tcn"},
        "task": {"name": "odometry_regression"},
        "nas": {"score": {"type": "multi-objective", "params": {"objectives": [
            {"metric": "rmse_total", "direction": "minimize"},
            {"metric": "energy_mj_per_inference", "direction": "minimize"},
        ]}}},
    }
    (folder / "nas_config.yaml").write_text(yaml.safe_dump(config))
    rows = []
    for i in range(7):
        rows.append({
            "metric__rmse_total": 7 - i, "energy_mj_per_inference": i + 1,
            "latency_ms": 10, "error_code": 1, "pruned": False,
            "quantization_mode": "float", "flops": 100,
            "hparam__batch_size": 256, "hparam__timesteps": 200,
            "hparam__input_dim": 10, "hparam__nb_filters": i + 2,
            "hparam__kernel_size": 2, "hparam__dilations": "[1, 2]",
            "hparam__dropout_rate": 0, "hparam__norm_flag": False,
            "hparam__use_skip_connections": False,
        })
    rows.append({**rows[0], "metric__rmse_total": 0, "energy_mj_per_inference": -1})
    rows.append({**rows[0], "metric__rmse_total": 8, "energy_mj_per_inference": 8})
    with (folder / "log_NAS_test.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    return folder


def prepared(source, tmp_path):
    out = tmp_path / "experiment"
    args = SimpleNamespace(source_run_dir=source, source_csv=None, source_config=None,
                           output_dir=out, count=4, budgets=[15, 55, 100], seeds=[17])
    pilot.prepare(args)
    return out, args


def test_selection_spans_front_and_excludes_sentinels(source):
    selected, _, _, _, front_count, valid_count = pilot.select_front(source)
    assert front_count == 7
    assert valid_count == 8
    assert [c["source_row_index"] for c in selected] == [0, 2, 4, 6]
    assert all(c["source_trial_number"] is None for c in selected)


def test_too_small_front_fails_without_silent_panel_change(source):
    with pytest.raises(ValueError, match="found 7"):
        pilot.select_front(source, count=8)


def test_reject_proxy_source(source):
    path = source / "nas_config.yaml"
    cfg = yaml.safe_load(path.read_text())
    cfg["nas"]["score"]["params"]["objectives"][1]["metric"] = "flops"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="proxy"):
        pilot.select_front(source)


def test_prepare_hashes_and_never_overwrites(source, tmp_path):
    out, args = prepared(source, tmp_path)
    manifest = pilot.load_manifest(out)
    assert len(manifest["candidates"]) == 4
    assert manifest["energy_policy"] == "fixed_historical_measurement"
    with pytest.raises(FileExistsError):
        pilot.prepare(args)
    (out / "source_config.yaml").write_text("modified")
    with pytest.raises(ValueError, match="modified"):
        pilot.load_manifest(out)


@pytest.mark.parametrize("text", ["55", "15,100", "0,55,100", "55,15,100", "15,55,55,100"])
def test_invalid_budgets(text):
    with pytest.raises(argparse.ArgumentTypeError):
        pilot.budgets_arg(text)


def test_run_requires_explicit_execute_before_any_access():
    with pytest.raises(ValueError, match="--execute"):
        pilot.run(SimpleNamespace(execute=False))


def test_prepare_import_does_not_load_training_runtime():
    code = (
        "import runpy, sys; runpy.run_path(sys.argv[1], run_name='safe_import'); "
        "assert 'tensorflow' not in sys.modules; assert 'nas_model_client' not in sys.modules; "
        "assert 'hil_server' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code, str(SCRIPT)], check=True)


def test_reference_comparison_reports_reversal():
    early = {"a": (1.0, 1.0), "b": (2.0, 2.0)}
    final = {"a": (0.8, 1.0), "b": (0.4, 2.0)}
    result = pilot.reference_comparison(early, final)
    assert result["early_front"] == ["a"]
    assert result["reference_front"] == ["a", "b"]
    assert result["regret"][1]["absolute_regret"] == pytest.approx(0.4)


def test_reference_comparison_reports_coverage_failure():
    result = pilot.reference_comparison({"a": (1, 1), "b": (2, 2)},
                                        {"a": (1, 3), "b": (0.5, 1)})
    assert result["regret"][0]["coverage_failure"]
    assert result["regret"][0]["absolute_regret"] is None


def test_same_front_improves_without_regret():
    result = pilot.reference_comparison({"a": (2, 1), "b": (1, 2)},
                                        {"a": (1, 1), "b": (0.5, 2)})
    assert all(r["absolute_regret"] == 0 for r in result["regret"])


def test_analyze_refuses_incomplete_panel(source, tmp_path):
    out, _ = prepared(source, tmp_path)
    with pytest.raises(ValueError, match="Incomplete panel"):
        pilot.analyze(SimpleNamespace(experiment_dir=out))


def test_complete_analysis_retains_all_candidates_and_policies(source, tmp_path, monkeypatch):
    out, _ = prepared(source, tmp_path)
    manifest = pilot.load_manifest(out)
    for c in manifest["candidates"]:
        work = out / c["candidate_id"] / "seed_17"
        work.mkdir(parents=True)
        pilot.write_json(work / "complete.json", {})
        rows = [dict(candidate_id=c["candidate_id"], seed=17, budget=b, policy=p,
                     selected_epoch=b, status="ok", quantization_mode="float",
                     energy_mj_per_inference=c["historical_energy_mj"],
                     energy_policy="fixed_historical_measurement",
                     metrics={"rmse_total": c["historical_rmse_total"] / b})
                for b in manifest["budgets"] for p in ("budget_only", "original_policy")]
        pilot.write_json(work / "results.json", rows)
    pilot.analyze(SimpleNamespace(experiment_dir=out))
    summary = json.loads((out / "summary.json").read_text())
    assert summary["seeds"]["17"]["panel_size"] == 4
    assert len(list(csv.DictReader((out / "checkpoint_metrics.csv").open()))) == 24
    assert (out / "conditional_fronts_seed_17.png").is_file()


def test_epoch_recorder_uses_real_keras_stopping_without_training(tmp_path):
    """Drive real callbacks manually with synthetic losses; never call fit."""
    tf = pytest.importorskip("tensorflow")
    model = tf.keras.Sequential([tf.keras.layers.Input((1,)), tf.keras.layers.Dense(1)])
    best = tmp_path / "best.keras"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(best, monitor="val_loss", mode="min", save_best_only=True)
    early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=2,
                                            restore_best_weights=True)
    record = pilot.make_epoch_recorder(tf, early, best, tmp_path, [2, 4, 5])
    for callback in (checkpoint, record):
        callback.set_model(model)
        callback.set_params({"epochs": 5})
        callback.on_train_begin()
    model.stop_training = False
    for epoch, loss in enumerate([5.0, 4.0, 4.5, 4.6, 1.0]):
        logs = {"loss": loss, "val_loss": loss}
        checkpoint.on_epoch_end(epoch, logs)
        record.on_epoch_end(epoch, logs)
        assert not model.stop_training
    assert record.stop_epoch == 4
    by_key = {(x["budget"], x["policy"]): x for x in record.snapshots}
    assert by_key[5, "original_policy"]["selected_epoch"] == 2
    assert by_key[5, "budget_only"]["selected_epoch"] == 5
    assert by_key[2, "budget_only"]["sha256"] == by_key[5, "original_policy"]["sha256"]


def test_split_hashes_ignore_test_data(tmp_path):
    (tmp_path / "Train.txt").write_text("train_sequence")
    (tmp_path / "Valid.txt").write_text("validation_sequence")
    (tmp_path / "Test.txt").write_text("held_out_sequence")
    assert set(pilot.split_hashes(tmp_path)) == {"Train.txt", "Valid.txt"}


def test_runtime_bootstrap_reuses_real_crest_without_fit_or_hil(source, tmp_path, monkeypatch):
    """Real config/model/task wiring with synthetic data, never invoke model.fit."""
    pytest.importorskip("tensorflow")
    pytest.importorskip("optuna")
    import numpy as np
    from crest.pipeline_types import DataSplit, DatasetBundle
    config = ROOT / "src/config/case_study_configs/nas_config_case1_3_portenta_m7_b2b_oxiod.yaml"
    (source / "nas_config.yaml").write_text(config.read_text())
    out, _ = prepared(source, tmp_path)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "Train.txt").write_text("train")
    (dataset / "Valid.txt").write_text("valid")
    split = DataSplit(inputs=np.zeros((4, 200, 10), dtype="float32"),
                      targets={"velx": np.zeros((4, 1)), "vely": np.zeros((4, 1))})
    bundle = DatasetBundle(train=split, val=split, calibration=split, test=None,
                           input_shape=(200, 10), input_dtype="float32")
    monkeypatch.setattr(pilot, "load_training_bundle", lambda params: bundle)
    calls = []

    def inspect_candidate(client, candidate, seed, budgets, work, tf):
        assert client.socket is None and client.context is None
        assert client.config.device.hil is False
        assert client.dataset_bundle.test is None
        assert client.config.training.train is True
        assert str(client.config.outputs.models_dir).startswith(str(out))
        model = client.model_family.build_model(candidate["family_hparams"],
                                                client.model_build_context, client.model_config)
        task = client._instantiate_task(client.task_name, client.config, client.task_config,
                                        checkpoint_path=work / "best.keras")
        task.compile_model(model, client.task_config, client.target_spec)
        fit = task.build_fit_plan(bundle, client.task_config, client.target_spec,
                                  mode="search", combine_train_val=False)
        assert fit.monitor_metric == "val_loss"
        assert len(fit.callbacks) == 2
        assert fit.fit_kwargs["x"] is split.inputs
        calls.append(candidate["candidate_id"])

    monkeypatch.setattr(pilot, "train_candidate", inspect_candidate)
    monkeypatch.setattr(pilot, "analyze", lambda args: None)
    args = SimpleNamespace(execute=True, experiment_dir=out, dataset_dir=dataset, skip_completed=False)
    pilot.run(args)
    assert len(calls) == 4
    assert json.loads((out / "execution.json").read_text())["test_loaded"] is False
    args.skip_completed = True
    pilot.run(args)
    assert len(calls) == 4


def test_train_candidate_wiring_uses_real_task_and_evaluation_seam(tmp_path, monkeypatch):
    """Use fake fit events with real models/callbacks; no gradient updates."""
    tf = pytest.importorskip("tensorflow")
    pytest.importorskip("optuna")
    import numpy as np
    from addict import Dict
    from crest.model_families.odom_tcn import OdomTCNFamily
    from crest.tasks.odometry_regression import OdometryRegressionTask
    from crest.pipeline_types import DataSplit, DatasetBundle, ModelBuildContext, TargetSpec
    split = DataSplit(inputs=np.zeros((4, 16, 10), dtype="float32"),
                      targets={"velx": np.zeros((4, 1)), "vely": np.zeros((4, 1))})
    bundle = DatasetBundle(train=split, val=split, input_shape=(16, 10), input_dtype="float32")
    target = TargetSpec(task_type="regression", output_names=["velx", "vely"], output_shapes=[(1,), (1,)])
    context = ModelBuildContext(input_shape=(16, 10), input_dtype="float32", target_spec=target)
    task = OdometryRegressionTask(checkpoint_path=tmp_path / "best.keras", early_stopping_patience=40)
    calls = []

    def fake_fit(model, **kwargs):
        assert kwargs["batch_size"] == 256
        assert kwargs["x"] is split.inputs
        model.optimizer.build(model.trainable_variables)
        callbacks = kwargs["callbacks"]
        for cb in callbacks:
            cb.set_model(model)
            cb.set_params({"epochs": kwargs["epochs"]})
            cb.on_train_begin()
        model.stop_training = False
        for epoch in range(kwargs["epochs"]):
            # Simulate distinct checkpoint contents without taking gradient steps.
            model.weights[0].assign_add(tf.ones_like(model.weights[0]) * 0.001)
            for cb in callbacks:
                cb.on_epoch_end(epoch, {"val_loss": 1 / (epoch + 1), "loss": 1 / (epoch + 1)})
        for cb in callbacks:
            cb.on_train_end()

    def fake_evaluate(**kwargs):
        assert kwargs["split"] is split and kwargs["evaluation_backend"] == "tflite"
        assert kwargs["quantization_mode"] == "float"
        (tmp_path / f"pilot_{kwargs['split_name']}_eval.tflite").write_bytes(b"test export placeholder")
        calls.append(kwargs["split_name"])
        return SimpleNamespace(metrics={"rmse_total": 0.2, "rmse_vel_x": 0.1, "rmse_vel_y": 0.1})

    monkeypatch.setattr(tf.keras.Model, "fit", fake_fit)
    client = SimpleNamespace(
        dataset_bundle=bundle, model_family=OdomTCNFamily(), model_build_context=context,
        model_config=Dict(), task_config=Dict(), task_name="odometry_regression", config=Dict(),
        target_spec=target, study_name="pilot", _artifacts_dir=lambda: tmp_path,
        _instantiate_task=lambda *a, **k: task, _evaluate_model_with_backend=fake_evaluate,
    )
    candidate = {
        "candidate_id": "test", "historical_energy_mj": 10, "quantization_mode": "float",
        "runtime_metadata": {"batch_size": 256, "timesteps": 16, "input_dim": 10},
        "family_hparams": {"nb_filters": 2, "kernel_size": 2, "dilations": [1, 2],
                           "dropout_rate": 0.0, "use_skip_connections": False, "norm_flag": False},
    }
    pilot.train_candidate(client, candidate, 17, [1, 2], tmp_path, tf)
    rows = json.loads((tmp_path / "results.json").read_text())
    assert len(rows) == 4 and all(r["status"] == "ok" for r in rows)
    assert len(calls) == 2  # Shared budget/original-policy hashes evaluated once.
    assert (tmp_path / "last.keras").is_file()
