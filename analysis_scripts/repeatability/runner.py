#!/usr/bin/env python3
# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Frozen, bounded, restart-safe paired repeatability schedule. No hardware on import."""
import argparse
import datetime as dt
import hashlib
import json
import random
import time
import uuid

from . import experiment
from pathlib import Path


def stamp():
    """Return the current UTC timestamp.

    Returns
    -------
    str
        ISO 8601 timestamp.
    """
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path):
    """Hash the complete file contents.

    Parameters
    ----------
    path : str or pathlib.Path
        File or directory location.

    Returns
    -------
    str
        SHA-256 hexadecimal digest.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    """Write JSON evidence to the requested path.

    Parameters
    ----------
    path : str or pathlib.Path
        File or directory location.
    data : object
        JSON-serializable evidence.
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def prepare(manifest_path, output, rounds, seed, max_attempts=2):
    """Freeze a randomized paired schedule without importing hardware code.

    Parameters
    ----------
    manifest_path : str or pathlib.Path
        Candidate manifest with pairs, dimensions, cadence, runs, and model seed.
    output : str or pathlib.Path
        New study directory; existing directories are never reused.
    rounds : int
        Positive repetitions per pair, with balanced AB/BA ordering.
    seed : int
        Schedule randomization seed, independent of model initialization.
    max_attempts : int, optional
        Maximum attempts per evaluation across restarts.

    Returns
    -------
    pathlib.Path
        Created sealed study directory.
    """
    manifest = json.loads(Path(manifest_path).read_text())
    from .experiment import normalized_config

    for candidate in manifest["candidates"]:
        normalized_config(runtime_config(candidate, manifest))
    candidates = {c["candidate_id"]: c for c in manifest["candidates"]}
    pairs = manifest["pairs"]
    if len(candidates) != len(manifest["candidates"]) or not candidates or not pairs:
        raise ValueError("unique candidates and nonempty pairs required")
    if len({p["pair_id"] for p in pairs}) != len(pairs):
        raise ValueError("pair IDs must be unique")
    if rounds < 1 or max_attempts < 1:
        raise ValueError("rounds and max_attempts must be positive")
    for value in [*candidates, *(p["pair_id"] for p in pairs)]:
        if (
            not isinstance(value, str)
            or not value
            or value in (".", "..")
            or "/" in value
            or "\\" in value
        ):
            raise ValueError("candidate/pair IDs must be plain path components")
    if sorted(c for p in pairs for c in [p["candidate_a"], p["candidate_b"]]) != sorted(
        candidates
    ):
        raise ValueError("pairs must partition candidates")
    rng = random.Random(seed)
    orders = {}
    for p in pairs:
        seq = (
            ["AB"] * (rounds // 2)
            + ["BA"] * (rounds // 2)
            + ([rng.choice(["AB", "BA"])] if rounds % 2 else [])
        )
        rng.shuffle(seq)
        orders[p["pair_id"]] = seq
    study_id = (
        "repeatability-"
        + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    schedule = []
    for r in range(1, rounds + 1):
        block = list(pairs)
        rng.shuffle(block)
        for p in block:
            order = orders[p["pair_id"]][r - 1]
            for side in order:
                cid = p["candidate_a" if side == "A" else "candidate_b"]
                schedule.append(
                    dict(
                        study_id=study_id,
                        round_id=f"r{r:02d}",
                        pair_id=p["pair_id"],
                        config_id=cid,
                        pair_order=order,
                        side=side,
                        evaluation_id=f"{study_id}/r{r:02d}/{cid}",
                    )
                )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save(output / "manifest.json", manifest)
    save(
        output / "schedule.json",
        dict(
            schema_version=1,
            study_id=study_id,
            rounds=rounds,
            seed=seed,
            created_at=stamp(),
            max_attempts=max_attempts,
            repeat_unit="full continuous + cadenced paired evaluation",
            evaluations=schedule,
        ),
    )
    save(
        output / "seal.json",
        {name: digest(output / name) for name in ["manifest.json", "schedule.json"]},
    )
    return output


def verify(root):
    """Verify immutable plan hashes and reject superseded studies.

    Parameters
    ----------
    root : pathlib.Path
        Study directory containing manifest, schedule, and seal.
    """
    if (root / "SUPERSEDED.md").exists():
        raise ValueError("study plan is superseded; read SUPERSEDED.md")
    for name, expected in json.loads((root / "seal.json").read_text()).items():
        if digest(root / name) != expected:
            raise ValueError("sealed file changed: " + name)


def runtime_config(candidate, manifest):
    """Combine candidate identity with explicit study measurement settings.

    Parameters
    ----------
    candidate : dict
        Candidate architecture and clock specification.
    manifest : dict
        Study dimensions, cadence, run count, and initialization seed.

    Returns
    -------
    dict
        Assembled runtime configuration; validation is performed separately.
    """
    return dict(
        candidate,
        cpu_clock_mhz=int(candidate["clock_mhz"]),
        window_size=int(manifest["timesteps"]),
        num_channels=int(manifest["input_dim"]),
        period_ms=manifest["period_ms"],
        measured_runs=manifest["measured_runs"],
        initialization_seed=manifest["initialization_seed"],
    )


def execute(
    root,
    dut_port,
    harness_port,
    board_ready=False,
    max_evaluations=None,
):
    """Measure unfinished paired evaluations, one experiment at a time.

    Parameters
    ----------
    root : str or pathlib.Path
        Sealed study directory with prepared or bound artifacts.
    dut_port, harness_port : str
        Explicit serial paths for the current deployment.
    board_ready : bool, optional
        Explicit acknowledgement permitting hardware programming and measurement.
    max_evaluations : int, optional
        Bound on new evaluation units in this invocation.
    """
    if max_evaluations is not None and max_evaluations <= 0:
        raise ValueError("max_evaluations must be positive")
    if not dut_port or not harness_port:
        raise ValueError("explicit DUT and harness ports required")
    if not board_ready:
        raise ValueError(
            "measurement requires explicit --board-ready after board review"
        )
    root = Path(root).resolve()
    import fcntl

    with (root / "run.lock").open("a") as study_lock:
        fcntl.flock(study_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify(root)
        manifest = json.loads((root / "manifest.json").read_text())
        plan = json.loads((root / "schedule.json").read_text())
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
        module = experiment
        measurement = module.Experiment(manifest, root)
        configs = {c["candidate_id"]: dict(c) for c in manifest["candidates"]}
        binding = root / "artifact_binding.json"
        if binding.exists():
            bound = json.loads(binding.read_text())
            for cid, entry in bound["candidates"].items():
                if digest(entry["path"]) != entry["sha256"]:
                    raise ValueError("artifact binding changed: " + cid)
                configs[cid]["prepared_manifest"] = entry["path"]
        for candidate in configs.values():
            candidate.update(dut_port=dut_port, harness_port=harness_port)
            checked = measurement.preflight(runtime_config(candidate, manifest))
            if checked.get("ok") is not True:
                raise RuntimeError(
                    "study preflight rejected: " + candidate["candidate_id"]
                )
        processed = 0
        for item in plan["evaluations"]:
            directory = root / "evaluations" / item["round_id"] / item["config_id"]
            directory.mkdir(parents=True, exist_ok=True)
            if (directory / "complete.json").exists():
                continue
            for attempt in range(1, plan["max_attempts"] + 1):
                dest = directory / f"attempt-{attempt:02d}"
                if dest.exists():
                    if (dest / "result.json").exists():
                        prior = json.loads((dest / "result.json").read_text())
                        if prior.get("accepted") is True:
                            save(
                                directory / "complete.json",
                                dict(
                                    attempt_id=prior["attempt_id"],
                                    result=str(
                                        (dest / "result.json").relative_to(root)
                                    ),
                                ),
                            )
                            break
                    if (dest / "failure.json").exists() and json.loads(
                        (dest / "failure.json").read_text()
                    ).get("requires_inspection"):
                        raise RuntimeError(
                            "failure requires manual inspection: " + str(dest)
                        )
                    # Never resume after uncertain hardware interruption without inspection.
                    if (
                        not (dest / "result.json").exists()
                        and not (dest / "failure.json").exists()
                    ):
                        raise RuntimeError(
                            "interrupted attempt requires manual inspection: "
                            + str(dest)
                        )
                    continue
                dest.mkdir()
                context = dict(
                    item,
                    attempt_id=f"a{attempt:02d}",
                    attempt_dir=str(dest.resolve()),
                    phase_ids={
                        p: f"{item['evaluation_id']}/{p}/a{attempt:02d}"
                        for p in ["continuous", "cadenced"]
                    },
                )
                started_at = stamp()
                started_clock = time.monotonic()
                save(dest / "started.json", dict(context, started_at=started_at))
                config = runtime_config(configs[item["config_id"]], manifest)
                config.setdefault(
                    "prepared_manifest",
                    str(
                        Path("artifacts") / item["config_id"] / "prepared_manifest.json"
                    ),
                )
                try:
                    preflight = measurement.preflight(config)
                    save(dest / "preflight.json", preflight)
                    if preflight.get("ok") is not True:
                        raise RuntimeError("preflight rejected")
                    raw = measurement.evaluate(config, context)
                    save(dest / "raw.json", raw)
                    assessment = module.validate_evaluation(raw, config)
                    save(
                        dest / "result.json",
                        dict(
                            context,
                            started_at=started_at,
                            finished_at=stamp(),
                            wall_time_s=time.monotonic() - started_clock,
                            **assessment,
                        ),
                    )
                    if assessment["accepted"]:
                        save(
                            directory / "complete.json",
                            dict(
                                attempt_id=context["attempt_id"],
                                result=str((dest / "result.json").relative_to(root)),
                            ),
                        )
                        break
                except Exception as exc:
                    save(
                        dest / "failure.json",
                        dict(
                            context,
                            started_at=started_at,
                            finished_at=stamp(),
                            wall_time_s=time.monotonic() - started_clock,
                            error_type=type(exc).__name__,
                            error=str(exc),
                            requires_inspection=True,
                        ),
                    )
                    raise
            if not (directory / "complete.json").exists():
                raise RuntimeError(
                    "attempt budget exhausted for incomplete evaluation: "
                    + item["evaluation_id"]
                )
            processed += 1
            if max_evaluations is not None and processed >= max_evaluations:
                break


def build_artifacts(root, config_path):
    """Export initialized models and compile immutable paired artifacts.

    Parameters
    ----------
    root : str or pathlib.Path
        Sealed study receiving new artifacts.
    config_path : str or pathlib.Path
        Explicit CREST configuration with calibration data and toolchain paths.
    """
    root = Path(root).resolve()
    verify(root)
    repo = Path(__file__).resolve().parents[2]
    import sys

    sys.path.insert(0, str(repo / "src"))
    from hil_server import HILServer

    module = experiment
    manifest = json.loads((root / "manifest.json").read_text())
    server = HILServer(config_path=Path(config_path).resolve())
    try:
        # Redirect all backend staging paths before lazy model/data preparation.
        server.config.outputs.models_dir = str(root / "build_records")
        server.config.outputs.candidate_dir = str(root / "build_staging")
        server.config.outputs.tflite_model_path = str(
            root / "build_staging" / "model.tflite"
        )
        for candidate in manifest["candidates"]:
            destination = root / "artifacts" / candidate["candidate_id"]
            prepared = destination / "prepared_manifest.json"
            if prepared.exists():
                module.verify_bundle(prepared)
                continue
            if destination.exists():
                raise RuntimeError(
                    "incomplete artifact preparation retained; inspect and use a new study: "
                    + str(destination)
                )
            module.prepare_with_server(
                server, runtime_config(candidate, manifest), destination
            )
        save(
            root / "artifacts_ready.json",
            {
                "prepared_at": stamp(),
                "config_path": str(Path(config_path).resolve()),
                "config_sha256": digest(config_path),
                "candidates": {
                    c["candidate_id"]: digest(
                        root
                        / "artifacts"
                        / c["candidate_id"]
                        / "prepared_manifest.json"
                    )
                    for c in manifest["candidates"]
                },
            },
        )
    finally:
        server.socket.close(linger=0)


def bind_artifacts(root, artifact_root):
    """Bind verified existing bundles before the first measurement attempt.

    Parameters
    ----------
    root : str or pathlib.Path
        New sealed study without existing attempts or artifact binding.
    artifact_root : str or pathlib.Path
        Directory containing one prepared bundle per candidate ID.
    """
    root = Path(root).resolve()
    verify(root)
    artifact_root = Path(artifact_root).resolve()
    if (root / "artifact_binding.json").exists():
        raise FileExistsError("study already bound")
    if list(root.glob("evaluations/*/*/attempt-*")):
        raise ValueError("cannot change artifacts after an attempt")
    module = experiment
    manifest = json.loads((root / "manifest.json").read_text())
    candidates = manifest["candidates"]
    entries = {}
    measurement = module.Experiment(manifest, root)
    for candidate in candidates:
        path = artifact_root / candidate["candidate_id"] / "prepared_manifest.json"
        _, bundle = module.verify_bundle(path)
        if bundle["config"]["candidate_id"] != candidate["candidate_id"]:
            raise ValueError("artifact candidate mismatch")
        checked = measurement.preflight(
            dict(runtime_config(candidate, manifest), prepared_manifest=str(path))
        )
        if checked.get("ok") is not True:
            raise ValueError("artifact preflight rejected")
        entries[candidate["candidate_id"]] = {"path": str(path), "sha256": digest(path)}
    save(root / "artifact_binding.json", dict(bound_at=stamp(), candidates=entries))


def main():
    """Parse command-line arguments and dispatch the requested operation."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--rounds", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--max-attempts", type=int, default=2)
    p = sub.add_parser("bind-artifacts")
    p.add_argument("--study", required=True)
    p.add_argument("--artifact-root", required=True)
    p = sub.add_parser("build-artifacts")
    p.add_argument("--study", required=True)
    p.add_argument("--config", required=True)
    p = sub.add_parser("run")
    p.add_argument("--study", required=True)
    p.add_argument("--board-ready", action="store_true")
    p.add_argument("--max-evaluations", type=int)
    p.add_argument("--dut-port", required=True)
    p.add_argument("--harness-port", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(
            prepare(
                args.manifest, args.output, args.rounds, args.seed, args.max_attempts
            )
        )
    elif args.command == "bind-artifacts":
        bind_artifacts(args.study, args.artifact_root)
    elif args.command == "build-artifacts":
        build_artifacts(args.study, args.config)
    else:
        execute(
            args.study,
            args.dut_port,
            args.harness_port,
            args.board_ready,
            args.max_evaluations,
        )


if __name__ == "__main__":
    main()
