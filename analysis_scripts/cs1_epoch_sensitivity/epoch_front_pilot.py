#!/usr/bin/env python3
# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Front-only CS1 epoch pilot; prepare/analyze never train or access hardware.

Reuses CREST's Pareto reconstruction, OxIOD loader, task/model contracts, fit
callbacks, TFLite conversion and evaluation. Historical energy is held fixed:
these are conditional fronts, NOT newly measured deployment fronts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import yaml
from crest import pareto_replay as replay

LIMITATION = (
    "Four-point original-front-only pilot; cannot detect slow starters outside "
    "the panel. Energy is historical and fixed, not remeasured at checkpoints. "
    "Report reversals and negative results as well as supporting evidence."
)
OBJECTIVES = (
    replay.ObjectiveSpec("rmse_total", "minimize", "rmse_total"),
    replay.ObjectiveSpec("energy", "minimize", "energy"),
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    """Atomically replace only an experiment-owned JSON file."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def git_revision():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def budgets_arg(value):
    try:
        budgets = [int(x) for x in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Budgets must be comma-separated integers") from exc
    if budgets != sorted(set(budgets)) or not budgets or budgets[0] < 1:
        raise argparse.ArgumentTypeError("Budgets must be positive, unique and increasing")
    if 55 not in budgets or budgets[-1] <= 55:
        raise argparse.ArgumentTypeError("Include 55 and a reference budget greater than 55")
    return budgets


def select_front(source_dir, source_csv=None, source_config=None, count=4):
    """Reconstruct valid original front, then select energy-spaced positions."""
    config_path = source_config or replay.find_config_path(source_dir)
    if config_path is None:
        raise ValueError("A saved source config is required")
    config = replay.load_yaml_config(config_path)
    if (config.get("dataset", {}).get("name") != "oxiod"
            or config.get("model", {}).get("family") != "odom_tcn"
            or config.get("task", {}).get("name") != "odometry_regression"
            or config.get("device", {}).get("runtime_mode") != "back_to_back"):
        raise ValueError("Expected an OxIOD/odom_tcn back-to-back CS1 config")
    configured = config.get("nas", {}).get("score", {}).get("params", {}).get("objectives", [])
    if {(x.get("metric"), x.get("direction")) for x in configured} != {
        ("rmse_total", "minimize"), ("energy_mj_per_inference", "minimize")
    }:
        raise ValueError("Use a measured-energy CS1 source, not a proxy run")
    csv_path = source_csv or replay.find_csv_path(source_dir, config)
    rows, columns = replay.read_csv_rows(csv_path)
    specs = replay.resolve_objective_specs(config=config, rows=rows, columns=columns, override=None)
    rmse_col = next(x.column for x in specs if x.metric == "rmse_total")
    energy_col = next(x.column for x in specs if x.metric == "energy_mj_per_inference")
    eligible = []
    original_indices = []
    for index, row in enumerate(rows):
        rmse = replay.parse_finite_float(row.get(rmse_col))
        energy = replay.parse_finite_float(row.get(energy_col))
        latency = replay.parse_finite_float(row.get("latency_ms"))
        # Additional guards: replay's generic validity check allows some sentinels.
        if (replay.is_source_row_valid(row, specs)
                and rmse is not None and rmse >= 0
                and energy is not None and energy > 0
                and latency is not None and latency > 0):
            eligible.append(row)
            original_indices.append(index)
    candidates = replay.dedupe_candidates([
        replay.build_replay_candidate(
            source_row_index=original_indices[i], row=eligible[i], objectives=specs
        ) for i in replay.pareto_indices(eligible, specs)
    ])
    candidates.sort(key=lambda c: (c.objective_values[energy_col],
                                   c.objective_values[rmse_col], c.source_row_index))
    if count < 2 or len(candidates) < count:
        raise ValueError(f"Need at least {count} unique valid front points; found {len(candidates)}")
    positions = [round(i * (len(candidates) - 1) / (count - 1)) for i in range(count)]
    selected = []
    for pos in positions:
        candidate = candidates[pos]
        record = asdict(candidate)
        record.update({
            "candidate_id": f"row_{candidate.source_row_index:04d}",
            "historical_rmse_total": candidate.objective_values[rmse_col],
            "historical_energy_mj": candidate.objective_values[energy_col],
            "historical_latency_ms": float(candidate.source_row["latency_ms"]),
            "source_trial_number": candidate.source_row.get("trial_number") or None,
        })
        # Row index is deliberately not represented as an Optuna trial number.
        selected.append(record)
    return selected, config, Path(config_path), Path(csv_path), len(candidates), len(eligible)


def prepare(args):
    selected, config, config_path, csv_path, front_count, valid_count = select_front(
        args.source_run_dir.resolve(), args.source_csv, args.source_config, args.count
    )
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite experiment directory: {out}")
    out.mkdir(parents=True)
    snapshot = out / "source_config.yaml"
    snapshot.write_text(yaml.safe_dump(config, sort_keys=False))
    write_json(out / "manifest.json", {
        "schema_version": 1, "limitation": LIMITATION,
        "selection": "energy-ordered evenly spaced original-front positions, including extremes",
        "valid_source_rows": valid_count, "original_front_count": front_count,
        "source_csv": str(csv_path.resolve()), "source_csv_sha256": sha256(csv_path),
        "source_config": str(config_path.resolve()), "source_config_sha256": sha256(config_path),
        "snapshot_sha256": sha256(snapshot), "code_revision": git_revision(),
        "budgets": args.budgets, "seeds": args.seeds,
        "energy_policy": "fixed_historical_measurement", "candidates": selected,
    })
    print(f"Prepared {len(selected)} of {front_count} original-front points in {out}")
    print(LIMITATION)


def load_manifest(out):
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest["schema_version"] != 1 or sha256(out / "source_config.yaml") != manifest["snapshot_sha256"]:
        raise ValueError("Unsupported or modified experiment snapshot")
    return manifest


def load_training_bundle(params):
    """Reuse the legacy split loader/adapter, intentionally never load Test.txt."""
    from crest.data import import_oxiod_dataset
    from crest.datasets.oxiod import OxIODDataset, _OXIOD_SUBFOLDERS
    from crest.pipeline_types import DatasetBundle
    adapter = OxIODDataset()
    adapter.validate_config(params)
    common = dict(
        dataset_folder=str(params.directory), sub_folders=list(_OXIOD_SUBFOLDERS),
        sampling_rate=int(params.sampling_rate_hz), window_size=int(params.window_size),
        stride=int(params.stride), verbose=False, useMagnetometer=True,
        useStepCounter=True, AugmentationCopies=0,
    )
    train = adapter._to_split(import_oxiod_dataset(type_flag=2, **common))
    val = adapter._to_split(import_oxiod_dataset(type_flag=3, **common))
    calibration = None
    if params.get("calibration_windows") is not None:
        calibration = adapter._to_split(import_oxiod_dataset(
            type_flag=2, max_windows=int(params.calibration_windows), **common
        ))
    return DatasetBundle(
        train=train, val=val, calibration=calibration, test=None,
        input_shape=tuple(train.inputs.shape[1:]), input_dtype=str(train.inputs.dtype),
        metadata={"sampling_rate_hz": int(params.sampling_rate_hz),
                  "window_size": int(params.window_size), "stride": int(params.stride),
                  "input_dim": int(train.inputs.shape[2])},
    )


def split_hashes(dataset_dir):
    """Freeze train/validation sequence lists without opening held-out test lists."""
    files = sorted(p for p in dataset_dir.rglob("*.txt") if p.name in {"Train.txt", "Valid.txt"})
    if not files:
        raise ValueError("No OxIOD Train.txt/Valid.txt lists found")
    return {str(p.relative_to(dataset_dir)): sha256(p) for p in files}


def make_epoch_recorder(tf, early_stop, best_path, work, budgets):
    """Observe the real task EarlyStopping callback without terminating training.

    ModelCheckpoint runs first. The original callback decides virtual stopping;
    only stop_training/restore-best are disabled for the continuing trajectory.
    Evaluation happens AFTER fit, never by replacing the live training model.
    """
    class Recorder(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            early_stop.restore_best_weights = False
            early_stop.set_model(self.model)
            early_stop.set_params(self.params)
            early_stop.on_train_begin(logs)
            self.rows, self.snapshots = [], []
            self.best_loss, self.best_epoch, self.stop_epoch = math.inf, None, None
            self.frozen_best_epoch = None
            self.started = time.monotonic()

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            value = float(logs["val_loss"])
            if not math.isfinite(value):
                raise ValueError(f"Nonfinite validation loss at epoch {epoch + 1}")
            if value < self.best_loss:
                self.best_loss, self.best_epoch = value, epoch + 1
            if self.stop_epoch is None:
                early_stop.on_epoch_end(epoch, logs)
                if self.model.stop_training:
                    self.stop_epoch = epoch + 1
                    self.frozen_best_epoch = self.best_epoch
                    shutil.copy2(best_path, work / "original_policy_best.keras")
                    self.model.stop_training = False
            row = {"epoch": epoch + 1, "elapsed_s": time.monotonic() - self.started,
                   "best_epoch": self.best_epoch, "virtual_stop_epoch": self.stop_epoch,
                   **{k: float(v) for k, v in logs.items()}}
            self.rows.append(row)
            write_json(work / "history.json", self.rows)
            if epoch + 1 in budgets:
                for policy in ("budget_only", "original_policy"):
                    frozen = policy == "original_policy" and self.stop_epoch is not None
                    src = work / "original_policy_best.keras" if frozen else best_path
                    path = work / f"budget_{epoch + 1:04d}_{policy}.keras"
                    shutil.copy2(src, path)
                    self.snapshots.append({
                        "budget": epoch + 1, "policy": policy, "checkpoint": path.name,
                        "selected_epoch": self.frozen_best_epoch if frozen else self.best_epoch,
                        "virtual_stop_epoch": self.stop_epoch, "sha256": sha256(path),
                    })
                write_json(work / "checkpoints.json", self.snapshots)
    return Recorder()


def train_candidate(client, candidate, seed, budgets, work, tf):
    """Train using the existing family and task fit plan, then evaluate snapshots."""
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    runtime = candidate["runtime_metadata"]
    if (int(runtime["timesteps"]), int(runtime["input_dim"])) != tuple(client.dataset_bundle.input_shape):
        raise ValueError("Reconstructed input shape differs from source candidate")
    batch_size = int(runtime["batch_size"])
    if batch_size <= 0:
        raise ValueError("Invalid source batch size")
    family = client.model_family
    hparams = candidate["family_hparams"]
    family.validate_hparams(hparams, client.model_build_context, client.model_config)
    model = family.build_model(hparams, client.model_build_context, client.model_config)
    best_path = work / "best.keras"
    task = client._instantiate_task(
        client.task_name, client.config, client.task_config, checkpoint_path=best_path
    )
    task.validate_model_outputs(model, client.target_spec)
    task.compile_model(model, client.task_config, client.target_spec)
    fit = task.build_fit_plan(client.dataset_bundle, client.task_config,
                              client.target_spec, mode="search", combine_train_val=False)
    checkpoints = [x for x in fit.callbacks if isinstance(x, tf.keras.callbacks.ModelCheckpoint)]
    stops = [x for x in fit.callbacks if isinstance(x, tf.keras.callbacks.EarlyStopping)]
    if len(checkpoints) != 1 or len(stops) != 1 or len(fit.callbacks) != 2:
        raise ValueError("Unexpected task callbacks; review epoch policy before running")
    if checkpoints[0].monitor != "val_loss" or stops[0].monitor != "val_loss":
        raise ValueError("Pilot expects the source validation-loss checkpoint policy")
    recorder = make_epoch_recorder(tf, stops[0], best_path, work, budgets)
    model.fit(**fit.fit_kwargs, callbacks=[checkpoints[0], recorder],
              epochs=max(budgets), batch_size=batch_size)
    model.save(work / "last.keras")
    del model
    results, cache = [], {}
    for snapshot in recorder.snapshots:
        row = {**snapshot, "candidate_id": candidate["candidate_id"], "seed": seed,
               "energy_mj_per_inference": candidate["historical_energy_mj"],
               "energy_policy": "fixed_historical_measurement",
               "quantization_mode": candidate["quantization_mode"]}
        try:
            key = snapshot["sha256"]
            if key not in cache:
                loaded = family.load_model(work / snapshot["checkpoint"],
                                           client.model_build_context, client.model_config)
                label = f"b{snapshot['budget']}_{snapshot['policy']}"
                evaluated = client._evaluate_model_with_backend(
                    model=loaded, split=client.dataset_bundle.val, split_name=label,
                    quantization_mode=candidate["quantization_mode"], evaluation_backend="tflite",
                )
                metrics = {k: float(v) for k, v in evaluated.metrics.items()}
                if not all(math.isfinite(v) and v >= 0 for v in metrics.values()):
                    raise ValueError("Invalid validation metric")
                export = client._artifacts_dir() / f"{client.study_name}_{label}_eval.tflite"
                cache[key] = {"metrics": metrics, "tflite_path": str(export),
                              "tflite_sha256": sha256(export)}
                del loaded
            row.update(cache[key], status="ok")
        except Exception as exc:
            row.update(status="evaluation_failed", error=f"{type(exc).__name__}: {exc}")
        results.append(row)
        write_json(work / "results.json", results)
    if any(row["status"] != "ok" for row in results):
        raise RuntimeError("Some checkpoint evaluations failed; see results.json")


def run(args):
    if not args.execute:
        raise ValueError("Training requires the explicit --execute flag; nothing was started")
    out = args.experiment_dir.resolve()
    manifest = load_manifest(out)
    if git_revision() != manifest["code_revision"]:
        raise ValueError("Code revision differs from preparation; prepare a new experiment")
    dataset_dir = args.dataset_dir.resolve()
    hashes = split_hashes(dataset_dir)
    execution_path = out / "execution.json"
    identity = {"manifest_sha256": sha256(out / "manifest.json"),
                "dataset_dir": str(dataset_dir), "split_hashes": hashes}
    if execution_path.exists():
        previous = json.loads(execution_path.read_text())
        if not args.skip_completed or previous["identity"] != identity:
            raise ValueError("Existing execution: use matching --skip-completed or a new experiment")
        for candidate in manifest["candidates"]:
            for seed in manifest["seeds"]:
                work = out / candidate["candidate_id"] / f"seed_{seed}"
                if work.exists() and not (work / "complete.json").exists():
                    raise ValueError(f"Partial run must not be overwritten or spliced: {work}")
    # Heavy runtime imports are intentionally unreachable from prepare/analyze/help.
    import tensorflow as tf
    from nas_model_client import NASModelClient
    from crest.builtin_components import ensure_builtin_components_registered
    from crest.model import load_config
    from crest.runtime_bootstrap import bootstrap_pipeline
    ensure_builtin_components_registered()
    raw = yaml.safe_load((out / "source_config.yaml").read_text())
    raw["dataset"]["params"]["directory"] = str(dataset_dir) + "/"
    raw["device"].update(hil=False, compile_when_hil_disabled="false")
    raw["training"].update(energy_aware=False, train=True)
    # No hardware objective is evaluated in this standalone training experiment.
    raw["nas"] = {"score": {"type": "multi-objective", "params": {"objectives": [
        {"metric": "rmse_total", "direction": "minimize"}]}}, "prune": {"rules": []}}
    raw["outputs"] = {"models_dir": str(out / "exports"),
                      "candidate_dir": str(out / "unused_candidate_dir"),
                      "artifact_stem": "epoch_pilot", "log_file_name": "unused.csv"}
    effective = out / "effective_config.yaml"
    effective.write_text(yaml.safe_dump(raw, sort_keys=False))
    client = NASModelClient.__new__(NASModelClient)
    client.config_path = effective
    client.config = load_config(effective)
    tf.keras.utils.set_random_seed(manifest["seeds"][0])
    bundle = load_training_bundle(client.config.dataset.params)
    client._attach_bootstrapped_pipeline(bootstrap_pipeline(client.config, bundle=bundle))
    client.context = client.socket = None  # No HIL/network initialization.
    write_json(execution_path, {
        "identity": identity, "tensorflow_version": tf.__version__,
        "python": sys.version, "code_revision": git_revision(),
        "effective_config_sha256": sha256(effective),
        "train_shape": list(bundle.train.inputs.shape),
        "validation_shape": list(bundle.val.inputs.shape), "test_loaded": False,
    })
    failures = []
    for candidate in manifest["candidates"]:
        for seed in manifest["seeds"]:
            work = out / candidate["candidate_id"] / f"seed_{seed}"
            if (work / "complete.json").exists():
                continue
            work.mkdir(parents=True, exist_ok=False)
            client.study_name = f"{candidate['candidate_id']}_seed_{seed}"
            started = time.monotonic()
            try:
                train_candidate(client, candidate, seed, manifest["budgets"], work, tf)
                write_json(work / "complete.json", {"elapsed_s": time.monotonic() - started})
            except Exception as exc:
                failure = {"candidate_id": candidate["candidate_id"], "seed": seed,
                           "error": f"{type(exc).__name__}: {exc}"}
                write_json(work / "failure.json", failure)
                failures.append(failure)
    if failures:
        raise RuntimeError(f"{len(failures)} trajectories failed; all failures preserved")
    analyze(argparse.Namespace(experiment_dir=out))


def front_ids(points):
    return [key for key, values in points.items() if not any(
        other != key and replay.dominates(other_values, values, OBJECTIVES)
        for other, other_values in points.items()
    )]


def reference_comparison(early, final):
    """Same final performance on both sides; no reward for simple loss reduction."""
    selected = front_ids(early)
    reference = front_ids(final)
    caps = sorted({values[1] for values in final.values()})
    regret = []
    for cap in caps:
        ref = min(v[0] for v in final.values() if v[1] <= cap)
        kept = [final[k][0] for k in selected if final[k][1] <= cap]
        regret.append({"energy_cap_mj": cap, "reference_rmse": ref,
                       "selected_rmse": min(kept) if kept else None,
                       "absolute_regret": min(kept) - ref if kept else None,
                       "coverage_failure": not bool(kept)})
    return {"early_front": selected, "reference_front": reference,
            "reference_evaluated_early_front": front_ids({k: final[k] for k in selected}),
            "regret": regret, "panel_size": len(final)}


def analyze(args):
    out = args.experiment_dir.resolve()
    manifest = load_manifest(out)
    rows = []
    for candidate in manifest["candidates"]:
        for seed in manifest["seeds"]:
            work = out / candidate["candidate_id"] / f"seed_{seed}"
            if not (work / "complete.json").exists():
                raise ValueError(f"Incomplete panel: {work}; refusing success-only analysis")
            rows.extend(json.loads((work / "results.json").read_text()))
    indexed = {(r["candidate_id"], r["seed"], r["budget"], r["policy"]): r for r in rows}
    expected = len(manifest["candidates"]) * len(manifest["seeds"]) * len(manifest["budgets"]) * 2
    if len(indexed) != expected or len(rows) != expected or any(r["status"] != "ok" for r in rows):
        raise ValueError("Missing, duplicated or failed checkpoint records")
    summary = {"limitation": LIMITATION, "reference_budget": max(manifest["budgets"]), "seeds": {}}
    for seed in manifest["seeds"]:
        def points(budget, policy):
            result = {}
            for c in manifest["candidates"]:
                row = indexed[c["candidate_id"], seed, budget, policy]
                result[c["candidate_id"]] = (row["metrics"]["rmse_total"], row["energy_mj_per_inference"])
            return result
        final = points(max(manifest["budgets"]), "budget_only")
        entry = reference_comparison(points(55, "original_policy"), final)
        entry["budget_only_fronts"] = {str(b): front_ids(points(b, "budget_only")) for b in manifest["budgets"]}
        summary["seeds"][str(seed)] = entry
    write_json(out / "summary.json", summary)
    with (out / "checkpoint_metrics.csv").open("w", newline="") as stream:
        fields = ["candidate_id", "seed", "budget", "policy", "selected_epoch", "rmse_total",
                  "energy_mj_per_inference", "energy_policy", "quantization_mode"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row["metrics"][k] if k == "rmse_total" else row[k] for k in fields})
    plot_results(out, manifest, indexed)
    print(f"Wrote conditional fronts and reference comparisons to {out}")


def plot_results(out, manifest, indexed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for seed in manifest["seeds"]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        def draw(ax, budget, policy, ids, label, **kwargs):
            points = {key: (indexed[key, seed, budget, policy]["metrics"]["rmse_total"],
                            indexed[key, seed, budget, policy]["energy_mj_per_inference"])
                      for key in ids}
            ax.scatter([x[1] for x in points.values()], [x[0] for x in points.values()], alpha=0.35)
            front = sorted((points[key] for key in front_ids(points)), key=lambda x: x[1])
            style = {"linestyle": "-", **kwargs}
            ax.plot([x[1] for x in front], [x[0] for x in front], marker="o", label=label, **style)
            return points
        ids = [c["candidate_id"] for c in manifest["candidates"]]
        for b in manifest["budgets"]:
            draw(axes[0], b, "budget_only", ids, f"Budget {b}")
        early = draw(axes[0], 55, "original_policy", ids, "55 original policy", linestyle="--")
        ref = max(manifest["budgets"])
        draw(axes[1], ref, "budget_only", ids, f"All panel candidates at {ref}")
        draw(axes[1], ref, "budget_only", front_ids(early), f"55-selected candidates at {ref}", linestyle="--")
        for ax in axes:
            ax.set_xlabel("Historical energy / inference (mJ; HELD FIXED)")
            ax.set_ylabel("Validation TFLite RMSE total (lower is better)")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
        axes[0].set_title("Training-budget sensitivity within original-front panel")
        axes[1].set_title("Reference performance of early-selected subset")
        fig.suptitle(f"Front-only conditional pilot — seed {seed}; no new HIL energy")
        fig.tight_layout()
        fig.savefig(out / f"conditional_fronts_seed_{seed}.png", dpi=180)
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Freeze original-front selection; no training/data loading/HIL")
    p.add_argument("--source-run-dir", type=Path, required=True)
    p.add_argument("--source-csv", type=Path)
    p.add_argument("--source-config", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--count", type=int, default=4)
    p.add_argument("--budgets", type=budgets_arg, default=budgets_arg("15,30,55,100,200,300"))
    p.add_argument("--seeds", type=int, nargs="+", default=[17])
    p.set_defaults(func=prepare)
    p = sub.add_parser("run", help="Train/evaluate only with --execute; never access HIL")
    p.add_argument("--experiment-dir", type=Path, required=True)
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--skip-completed", action="store_true", help="Skip complete trajectories; refuse partial ones")
    p.set_defaults(func=run)
    p = sub.add_parser("analyze", help="Analyze a complete panel without training")
    p.add_argument("--experiment-dir", type=Path, required=True)
    p.set_defaults(func=analyze)
    args = parser.parse_args(argv)
    if args.command == "prepare" and (len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds)):
        parser.error("Seeds must be unique nonnegative integers")
    args.func(args)


if __name__ == "__main__":
    main()
