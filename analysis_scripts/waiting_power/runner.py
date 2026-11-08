# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Build, measure, and analyze waiting power with compiled STM32 firmware variants.

Hardware execution is explicit; imports and dry runs never open devices.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import hashlib
import json
import math
import random
import shutil
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIRMWARE = REPO / "sketches/stm32/crest_stm32_lrun/Appli/Src/crest_dut_runner.c"
EXPECTED_PROTOCOL = "rtc_wall_v4_dwt_active_stop_evidence_awake_tail"  # codex/llm-generator-mvp @ 0445a49
BUSY_SCRIPTS = (
    "runner.py",
    "backend_adapter.py",
    "hil_server.py",
    "nas_model_client.py",
    "pareto_hil_replay.py",
    "waiting_power_probe.py",
    "run_micro_workload_energy_probe.py",
)
BUSY_TOOLS = ("STM32_Programmer_CLI", "arduino-cli", "ST-LINK_gdbserver")
PROVENANCE_FILES = [
    FIRMWARE,
    REPO / "sketches/harness/harness.ino",
    REPO / "src/crest/microcontrollers/stm32_nucleo_n657x0.py",
    REPO / "src/crest/microcontrollers/stm32_measurement.py",
    REPO / "src/crest/microcontrollers/stm32_runtime.py",
]
WAKE_MARGIN_US = 5000
MIN_SLEEP_US = 5000
# Rough planning numbers only (not measured on this host).
EST_BUILD_S, EST_EXPORT_S, EST_MEASURE_S, EST_HARNESS_FLASH_S = 45.0, 180.0, 45.0, 60.0


def now():
    """Return the current UTC timestamp.

    Returns
    -------
    str
        ISO 8601 timestamp.
    """
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path):
    """Hash file contents with bounded read buffers.

    Parameters
    ----------
    path : str or pathlib.Path
        File or directory location.

    Returns
    -------
    str
        SHA-256 hexadecimal digest.
    """
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def jdump(value):
    """Serialize sorted, indented JSON with string fallback conversion.

    Parameters
    ----------
    value : object
        Value to convert or inspect.

    Returns
    -------
    str
        JSON text with a trailing newline.
    """
    return json.dumps(value, indent=2, sort_keys=True, default=str) + "\n"


def save(path, value):
    """Write JSON evidence to the requested path.

    Parameters
    ----------
    path : str or pathlib.Path
        File or directory location.
    value : object
        Value to convert or inspect.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(jdump(value))


def shapes(low_runs):
    """Define the continuous, long cadenced, and short cadenced windows.

    Parameters
    ----------
    low_runs : int
        Short window run count in the range 1 through 9.

    Returns
    -------
    dict
        Shape names mapped to native phase and run count.
    """
    if not 1 <= low_runs < 10:
        raise SystemExit("--low-runs must be in [1, 9]")
    return {
        "continuous": ("back_to_back", 10),
        "w10": ("cadenced", 10),
        f"w{low_runs}": ("cadenced", low_runs),
    }


def firmware_protocol(path=None):
    """Read the firmware measurement protocol macro.

    Parameters
    ----------
    path : str or pathlib.Path
        File or directory location.

    Returns
    -------
    str or None
        Protocol marker, or None when the macro is absent.
    """
    path = FIRMWARE if path is None else path
    import re

    m = re.search(
        r'#define\s+CREST_DUT_MEASUREMENT_PROTOCOL\s+"([^"]+)"', Path(path).read_text()
    )
    return m.group(1) if m else None


def git(*args):
    """Run a read-only Git command without optional index locks.

    Parameters
    ----------
    args : str
        Git command arguments.

    Returns
    -------
    str
        Command stdout, or an unavailable message on an OS error.
    """
    try:
        # --no-optional-locks: read-only, never refreshes the index of a shared checkout.
        return subprocess.run(
            ["git", "--no-optional-locks", "-C", str(REPO), *args],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
    except OSError as exc:
        return f"<git unavailable: {exc}>"


def provenance():
    """Capture source hashes and read-only Git evidence.

    Returns
    -------
    tuple[dict, str]
        Provenance metadata and firmware diff text.
    """
    rel = str(FIRMWARE)
    diff = git("diff", "HEAD", "--", rel)
    return {
        "captured_utc": now(),
        "firmware_path": rel,
        "firmware_protocol": firmware_protocol(),
        "git_head": git("rev-parse", "HEAD").strip(),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD").strip(),
        "firmware_last_commit": git(
            "log", "-1", "--format=%H %cI %s", "--", rel
        ).strip(),
        "firmware_status_porcelain": git("status", "--porcelain", "--", rel).strip(),
        "firmware_dirty_vs_head": bool(diff),
        "firmware_diff_vs_head_sha256": (
            hashlib.sha256(diff.encode()).hexdigest() if diff else None
        ),
        "firmware_head_blob": git("rev-parse", f"HEAD:{rel}").strip(),
        "firmware_worktree_blob": git("hash-object", rel).strip(),
        "stm32_project_status_porcelain": git(
            "status", "--porcelain", "--", "sketches/stm32/crest_stm32_lrun"
        ).splitlines(),
        "file_sha256": {str(p): sha256(p) for p in PROVENANCE_FILES if p.is_file()},
    }, diff


def snapshot_provenance(dest):
    """Save provenance metadata, firmware source, and diff evidence.

    Parameters
    ----------
    dest : pathlib.Path
        Destination directory.

    Returns
    -------
    dict
        Captured provenance metadata.
    """
    prov, diff = provenance()
    dest.mkdir(parents=True, exist_ok=True)
    save(dest / "firmware_git_state.json", prov)
    shutil.copy2(FIRMWARE, dest / "crest_dut_runner.c")
    (dest / "firmware_diff_vs_HEAD.patch").write_text(diff)
    return prov


def load_yaml(path):
    """Read a YAML configuration using the safe loader.

    Parameters
    ----------
    path : str or pathlib.Path
        File or directory location.

    Returns
    -------
    object
        Decoded YAML value.
    """
    import yaml

    return yaml.safe_load(Path(path).read_text())


def load_candidate(manifest_path, candidate_id):
    """Find a candidate and assemble integer input dimensions.

    Parameters
    ----------
    manifest_path : str or pathlib.Path
        Candidate manifest file.
    candidate_id : str
        Candidate identity to select.

    Returns
    -------
    dict
        Candidate fields with window_size and num_channels; missing IDs exit.
    """
    m = json.loads(Path(manifest_path).read_text())
    for c in m["candidates"]:
        if c["candidate_id"] == candidate_id:
            return dict(
                c, window_size=int(m["timesteps"]), num_channels=int(m["input_dim"])
            )
    raise SystemExit(f"candidate {candidate_id} not in {manifest_path}")


def tools_on_path():
    """Locate build tools using PATH and the repository CLI fallback.

    Returns
    -------
    dict[str, str or None]
        Executable locations or None for missing tools.
    """
    found = {
        n: shutil.which(n)
        for n in (
            "stedgeai",
            "arm-none-eabi-gcc",
            "make",
            "STM32_Programmer_CLI",
            "arduino-cli",
        )
    }
    repo_cli = (
        REPO / "tools/bin/arduino-cli"
    )  # arduino_base uses this when ARDUINO_CLI_BIN is unset
    if found["arduino-cli"] is None and repo_cli.is_file():
        found["arduino-cli"] = f"{repo_cli} (repo tools/bin, used by arduino_base)"
    return found


def busy_processes():
    """List other hardware runner and programmer processes without signaling them.

    Returns
    -------
    list[str]
        Matching process lines, or an unavailable marker when ps cannot run.
    """
    import os

    try:
        lines = subprocess.run(
            ["ps", "-eo", "pid,args"], capture_output=True, text=True, check=False
        ).stdout.splitlines()
    except OSError:
        return ["<ps unavailable>"]
    busy = []
    for line in lines[1:]:
        pid, _, cmd = line.strip().partition(" ")
        argv = cmd.split()
        if (
            not argv
            or pid == str(os.getpid())
            or "--dry-run" in argv
            or "--help" in argv
        ):
            continue
        exe = Path(argv[0]).name
        if exe in BUSY_TOOLS or (
            exe.startswith("python")
            and any(Path(a).name in BUSY_SCRIPTS for a in argv[1:3])
        ):
            busy.append(line.strip())
    return busy


def variant_name(clock, shape):
    """Construct the stable clock and shape variant identifier.

    Parameters
    ----------
    clock : int
        Clock frequency in MHz.
    shape : str
        Measurement shape name.

    Returns
    -------
    str
        Variant identifier used in study metadata.
    """
    return f"{clock}MHz_{shape}"


def study_period_ms(meta):
    """Read a positive, internally consistent frozen release period.

    Parameters
    ----------
    meta : dict
        Built study metadata containing explicit period_ms.

    Returns
    -------
    int
        Release period in milliseconds.
    """
    period = meta.get("period_ms")
    if isinstance(period, bool) or not isinstance(period, int) or period <= 0:
        raise SystemExit(f"study.json period_ms={period!r} must be a positive integer")
    if "period_s" in meta and not math.isclose(meta["period_s"], period / 1000.0):
        raise SystemExit(
            f'study.json period_s={meta["period_s"]} disagrees with period_ms={period}'
        )
    return int(period)


def device_options(base_device_cfg, clock, period_ms):
    """Resolve native device options for a single clock and release period.

    Parameters
    ----------
    base_device_cfg : object
        Configured device settings.
    clock : int
        CPU clock in MHz.
    period_ms : int
        Cadence period in milliseconds.

    Returns
    -------
    dict
        Resolved options including STOP timing margins.
    """
    from crest.microcontrollers import resolve_device_options

    opts = resolve_device_options("STM32_NUCLEO_N657X0_Q", base_device_cfg) or {}
    opts = dict(opts)
    opts.update(
        cpu_clock_mhz=int(clock),
        latency_budget_ms=int(period_ms),
        runtime_mode="cadenced",
        wake_margin_us=WAKE_MARGIN_US,
        min_sleep_us=MIN_SLEEP_US,
    )
    return opts


# --------------------------------------------------------------------------- build


def cmd_build(args):
    """Export initialized weights and compile all requested experiment variants.

    Parameters
    ----------
    args : argparse.Namespace
        Build arguments; dry_run restricts execution to read-only planning.
    """
    clocks = sorted(set(args.clocks))
    shp = shapes(args.low_runs)
    period = args.period_ms
    out = Path(args.output).resolve()
    candidate = load_candidate(args.manifest, args.candidate)
    proto = firmware_protocol()
    plan = {
        "output": str(out),
        "candidate": candidate["candidate_id"],
        "architecture": candidate["architecture"],
        "source_clock_mhz": candidate["clock_mhz"],
        "clocks_mhz": clocks,
        "period_ms": period,
        "period_s": period / 1000.0,
        "wake_margin_us": WAKE_MARGIN_US,
        "min_sleep_us": MIN_SLEEP_US,
        "w10_window_s": 10 * period / 1000.0,
        "shapes": {k: {"phase": v[0], "measured_runs": v[1]} for k, v in shp.items()},
        "builds": len(clocks) * len(shp),
        "config": str(Path(args.config).resolve()),
        "firmware_protocol": proto,
        "expected_protocol": args.expect_protocol,
        "protocol_ok": proto == args.expect_protocol,
        "tools_on_path": tools_on_path(),
        "estimated_build_wall_s": EST_EXPORT_S + EST_BUILD_S * len(clocks) * len(shp),
    }
    if args.dry_run:
        prov, _ = provenance()
        print(jdump({"dry_run": True, "plan": plan, "firmware_git_state": prov}))
        missing = [k for k, v in plan["tools_on_path"].items() if v is None]
        if missing:
            hint = ""
            print(f"WARNING: not on PATH: {missing}{hint}")
        if not plan["protocol_ok"]:
            print(
                f"WARNING: firmware protocol {proto!r} != expected {args.expect_protocol!r}; build would refuse"
            )
        return
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(REPO / "src"))
    prov = snapshot_provenance(out / "provenance" / "build")
    if proto != args.expect_protocol:
        raise SystemExit(
            f"firmware protocol {proto!r} != expected {args.expect_protocol!r}"
        )
    from crest.microcontrollers import stm32_measurement

    if stm32_measurement.PROTOCOL != proto:
        raise SystemExit(
            f"host validator expects {stm32_measurement.PROTOCOL}, firmware emits {proto}"
        )
    save(
        out / "study.json",
        dict(
            plan,
            created_utc=now(),
            status="building",
            firmware_git_state=prov,
            manifest=str(Path(args.manifest).resolve()),
            manifest_sha256=sha256(args.manifest),
            config_sha256=sha256(args.config),
            low_runs=args.low_runs,
            timesteps=candidate["window_size"],
            input_dim=candidate["num_channels"],
            initialization_seed=args.initialization_seed,
        ),
    )
    import numpy as np
    import tensorflow as tf
    from dataclasses import replace
    from hil_server import HILServer
    from crest.microcontrollers import get_device, stm32_nucleo_n657x0 as stm

    server = HILServer(config_path=Path(args.config).resolve())
    try:
        server.config.outputs.models_dir = str(out / "build_records")
        server.config.outputs.candidate_dir = str(out / "build_staging")
        server.config.outputs.tflite_model_path = str(
            out / "build_staging" / "model.tflite"
        )
        seed = args.initialization_seed
        random.seed(seed)
        np.random.seed(seed)
        tf.keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()
        export_backend = get_device(
            "STM32_NUCLEO_N657X0_Q",
            device_options=device_options(server.config.device, clocks[-1], period),
        )
        model = server._materialize_candidate_model(
            candidate["architecture"],
            model_variant="approx_trained",
            checkpoint_path=None,
        )
        calib = server._ensure_calibration_split()
        request = server._build_candidate_prepare_request(
            model=model,
            model_variant="approx_trained",
            checkpoint_path=None,
            calibration_split=calib,
            quantization_mode="int8_ptq",
        )
        art = out / "artifacts"
        art.mkdir()
        np.save(
            art / "calibration_inputs.npy", np.asarray(calib.inputs), allow_pickle=False
        )
        request = replace(request, artifact_root=art / "export_root")
        export = Path(export_backend.prepare_candidate(request=request)).resolve()
        export_sidecar = json.loads(
            stm._manifest_path(export_backend._resolve_paths(export)).read_text()
        )
        weight_fields = {
            k: export_sidecar.get(k)
            for k in (
                "weights_blob_path",
                "weights_blob_size",
                "weights_blob_sha256",
                "weights_flash_address",
                "weights_external_loader",
                "weight_storage_mode",
            )
        }
        variants = {}
        for clock in clocks:
            for shape, (phase, runs) in shp.items():
                name = variant_name(clock, shape)
                root = art / "variants" / name
                shutil.copytree(export, root)
                opts = device_options(server.config.device, clock, period)
                backend = get_device("STM32_NUCLEO_N657X0_Q", device_options=opts)
                paths = backend._resolve_paths(root)
                sidecar = stm._manifest_path(paths)
                text = sidecar.read_text().replace(str(export), str(root))
                storage = json.loads(text)
                storage["staged_workspace_root"] = str(root)
                sidecar.write_text(jdump(storage))
                backend._write_runtime_phase_config(
                    paths=paths, selected_phase=phase, measured_runs=runs
                )
                t0 = time.monotonic()
                result = backend.compile(
                    sketch_path=root,
                    arena_kb=-1,
                    window_size=candidate["window_size"],
                    num_channels=candidate["num_channels"],
                )
                if not result.success or result.overflow_kind:
                    raise RuntimeError(f"{name} compile failed: {result.log[-2000:]}")
                repairs = ensure_weight_fields(sidecar, weight_fields, root, export)
                variants[name] = {
                    "clock_mhz": clock,
                    "shape": shape,
                    "phase": phase,
                    "measured_runs": runs,
                    "project": str(root),
                    "compile_s": time.monotonic() - t0,
                    "phase_header_sha256": sha256(
                        paths.inc_dir / "crest_dut_phase_config.h"
                    ),
                    "sidecar_repairs": repairs,
                    "device_options": json.loads(jdump(opts)),
                    "compile_result": json.loads(jdump(dataclasses.asdict(result))),
                }
                save(art / "variants.json", variants)
                print(f'built {name} in {variants[name]["compile_s"]:.1f}s', flush=True)
        save(
            out / "study.json",
            dict(
                json.loads((out / "study.json").read_text()),
                status="built",
                built_utc=now(),
                export_root=str(export),
                weight_fields=weight_fields,
            ),
        )
    finally:
        try:
            server.socket.close(linger=0)
        except Exception:
            pass
    print(f"built study at {out}")


def ensure_weight_fields(sidecar, weight_fields, root, export):
    """Restore missing exported weight fields and require an existing blob.

    Parameters
    ----------
    sidecar : pathlib.Path
        Variant storage manifest to repair in place.
    weight_fields : dict
        Original exported weight metadata.
    root : pathlib.Path
        Variant workspace replacing the export prefix in string fields.
    export : pathlib.Path
        Original export workspace.

    Returns
    -------
    dict[str, str]
        Repair explanations by field; an invalid blob raises RuntimeError.
    """
    storage = json.loads(sidecar.read_text())
    repairs = {}
    for k, v in weight_fields.items():
        v = v.replace(str(export), str(root)) if isinstance(v, str) else v
        if storage.get(k) in (None, "") and v is not None:
            storage[k] = v
            repairs[k] = "restored from export sidecar"
    if repairs:
        sidecar.write_text(jdump(storage))
    blob = storage.get("weights_blob_path")
    if not blob or not Path(blob).is_file():
        raise RuntimeError(f"variant sidecar lacks a valid weights blob: {sidecar}")
    return repairs


# --------------------------------------------------------------------------- schedule


def schedule(rounds, clocks, low_runs, seed, order):
    """Create deterministic randomized blocks for all clocks and shapes.

    Parameters
    ----------
    rounds : int
        Number of complete measurement rounds.
    clocks : list[int]
        CPU clocks in MHz.
    low_runs : int
        Short cadenced window run count.
    seed : int
        Schedule randomization seed.
    order : str
        Grouped or fully shuffled measurement order.

    Returns
    -------
    list[dict]
        Ordered measurements with stable round and variant identifiers.
    """
    rng = random.Random(seed)
    low = f"w{low_runs}"
    block10 = [(c, s) for c in clocks for s in ("continuous", "w10")]
    block_low = [(c, low) for c in clocks]
    items = []
    for r in range(1, rounds + 1):
        if order == "grouped":
            a, b = block10[:], block_low[:]
            rng.shuffle(a)
            rng.shuffle(b)
            # Alternate block order (one harness run-count change per round boundary) and end the
            # study on the runs=10 block so the harness is left flashed with CREST_INFERENCE_RUNS=10.
            seq = b + a if (rounds - r) % 2 == 0 else a + b
        else:
            seq = block10 + block_low
            rng.shuffle(seq)
        items += [
            {
                "id": f"r{r:02d}_{variant_name(c, s)}",
                "round": r,
                "clock_mhz": c,
                "shape": s,
            }
            for c, s in seq
        ]
    return items


def harness_reflashes(items, shp):
    """Count run-count transitions required by a measurement schedule.

    Parameters
    ----------
    items : list[dict]
        Scheduled measurement items.
    shp : dict
        Shape-to-phase/run-count mapping.

    Returns
    -------
    int
        Initial flash plus run-count changes.
    """
    runs = [shp[i["shape"]][1] for i in items]
    return (1 if runs else 0) + sum(1 for a, b in zip(runs, runs[1:]) if a != b)


# --------------------------------------------------------------------------- run


def accept(metrics, error_code, phase, runs, clock, proto, dut_log):
    """Check native instrumentation and waiting-power protocol metrics.

    Parameters
    ----------
    metrics : dict
        Native power metrics.
    error_code : int
        Native backend result code.
    phase : str
        Expected native phase.
    runs : int
        Expected measured run count.
    clock : int
        Expected clock in MHz.
    proto : str
        Required protocol marker.
    dut_log : list[str]
        Raw device output containing the protocol evidence.

    Returns
    -------
    list[str]
        Rejection reasons; empty only when all checks pass.
    """
    reasons = []
    if error_code != 0:
        reasons.append(f"error_code={error_code}")
    for flag in (
        "measurement_instrumentation_valid",
        "measurement_deployment_eligible",
        "appli_programmed",
        "weights_programmed",
    ):
        if metrics.get(flag) is not True:
            reasons.append(f"{flag} not true")
    if metrics.get("runs") != runs:
        reasons.append("runs mismatch")
    if metrics.get("clock_hz") != clock * 1_000_000:
        reasons.append("clock mismatch")
    if metrics.get("phase") != phase:
        reasons.append("phase mismatch")
    e = metrics.get("energy_mj_per_inference")
    if not (
        isinstance(e, (int, float))
        and not isinstance(e, bool)
        and math.isfinite(e)
        and e > 0
    ):
        reasons.append("invalid energy")
    if f"measurement protocol output: {proto}" not in "\n".join(map(str, dut_log)):
        reasons.append("protocol line missing")
    if phase == "cadenced":
        if metrics.get("deadline_miss_count") != 0:
            reasons.append("deadline misses")
        if (
            metrics.get("stop_entries") != runs - 1
            or metrics.get("slept_slots") != runs - 1
        ):
            reasons.append("stop entries != runs-1")
    return reasons


def _prepare_runtime_backend(v, adir, stm, CompileResult, phase, runs, item, period):
    """Configure a variant backend without rebuilding or changing phase headers.

    Parameters
    ----------
    v : dict
        Frozen variant options, project path, header digest, and compile result.
    adir : pathlib.Path
        Current attempt directory receiving native measurement records.
    stm : module
        STM32 backend module imported by the runner.
    CompileResult : type
        Compile result class imported by the runner.
    phase : str
        Native phase selected for the current variant.
    runs : int
        Compiled inference count for the current variant.
    item : dict
        Scheduled item containing the CPU clock.
    period : int
        Frozen study release period in milliseconds.

    Returns
    -------
    tuple[object, pathlib.Path, int]
        Configured backend, project root, and clock in MHz. Setup failures
        propagate before the runner's measurement exception handler.
    """
    opts = dict(v["device_options"])  # resolved when the variant was compiled
    opts["measurement_records_dir"] = str(adir / "measurement_records")
    backend = stm.STM32NucleoN657X0QDevice(device_options=opts)
    root = Path(v["project"])
    paths = backend._resolve_paths(root)
    sidecar = stm._manifest_path(paths)
    storage = json.loads(sidecar.read_text())
    for key in list(
        storage
    ):  # force app+weights programming regardless of workspace history
        if key.startswith("last_programmed_"):
            storage[key] = None
    sidecar.write_text(jdump(storage))
    if sha256(paths.inc_dir / "crest_dut_phase_config.h") != v["phase_header_sha256"]:
        raise SystemExit(f"phase header changed after build: {root}")
    fields = dict(v["compile_result"])
    for key in (
        "build_dir",
        "boot_build_dir",
        "boot_elf_path",
        "app_build_dir",
        "app_elf_path",
        "signed_app_bin_path",
    ):
        if fields.get(key) is not None:
            fields[key] = Path(fields[key])
    cached = CompileResult(**fields)
    backend.compile = types.MethodType(lambda self, **kw: cached, backend)
    clock = item["clock_mhz"]

    def validate_only(
        self, *, paths, selected_phase, measured_runs, _p=phase, _r=runs, _c=clock
    ):
        """Validate the requested phase and existing compiled header.

        Parameters
        ----------
        paths : object
            Resolved backend project paths.
        selected_phase : str
            Requested native phase.
        measured_runs : int
            Requested inference count.
        _p : str, optional
            Compiled phase captured when the validator is created.
        _r : int, optional
            Compiled run count captured when the validator is created.
        _c : int, optional
            Compiled clock captured when the validator is created.
        """
        if selected_phase != _p or measured_runs != _r:
            raise ValueError(
                "runtime requested a different phase/run count than was compiled"
            )
        stm._validate_phase_config_header(
            paths.inc_dir / "crest_dut_phase_config.h",
            cpu_clock_mhz=_c,
            selected_phase=_p,
            measured_runs=_r,
            latency_budget_ms=period,
            wake_margin_us=WAKE_MARGIN_US,
            min_sleep_us=MIN_SLEEP_US,
        )

    backend._write_runtime_phase_config = types.MethodType(validate_only, backend)
    return backend, root, clock


def cmd_run(args):
    """Measure unfinished scheduled variants using compiled firmware variants.

    Parameters
    ----------
    args : argparse.Namespace
        Explicit study, serial identities, and schedule arguments.
    """
    study = Path(args.study).resolve() if args.study else None
    clocks = sorted(set(args.clocks))
    low_runs = args.low_runs
    period = None  # only ever taken from study.json
    if study and (study / "study.json").is_file():
        meta = json.loads((study / "study.json").read_text())
        low_runs = meta.get("low_runs", low_runs)
        period = study_period_ms(meta)
        if not set(clocks) <= set(meta["clocks_mhz"]):
            raise SystemExit(f'clocks {clocks} not all built ({meta["clocks_mhz"]})')
    shp = shapes(low_runs)
    items = schedule(args.rounds, clocks, low_runs, args.seed, args.order)
    flashes = harness_reflashes(items, shp)
    est = len(items) * EST_MEASURE_S + flashes * EST_HARNESS_FLASH_S
    if args.dry_run:
        print(
            jdump(
                {
                    "dry_run": True,
                    "study": str(study) if study else None,
                    "rounds": args.rounds,
                    "period_ms": period,
                    "clocks_mhz": clocks,
                    "shapes": {k: list(v) for k, v in shp.items()},
                    "order": args.order,
                    "seed": args.seed,
                    "measurements": len(items),
                    "dut_programs_app_and_weights": len(items),
                    "harness_reflashes_expected": flashes,
                    "estimated_measure_wall_s": est,
                    "dut_port": args.dut_port,
                    "dut_port_present": Path(args.dut_port).exists(),
                    "harness_port": args.harness_port,
                    "harness_port_present": Path(args.harness_port).exists(),
                    "harness_auto_flash": args.harness_auto_flash,
                    "other_hil_processes": busy_processes(),
                    "note": "port presence is a filesystem check only; no port was opened",
                }
            )
        )
        for i in items:
            ph, rn = shp[i["shape"]]
            print(f"{i['id']:<28} phase={ph:<12} runs={rn}")
        return
    if not args.board_ready:
        raise SystemExit("refusing to touch hardware without --board-ready")
    if not study or not (study / "artifacts" / "variants.json").is_file():
        raise SystemExit("--study must point at a completed build")
    if not args.dut_port or not args.harness_port:
        raise SystemExit("--dut-port and --harness-port are required")
    for p in (args.dut_port, args.harness_port):
        if not Path(p).exists():
            raise SystemExit(f"serial port missing: {p}")
    meta = json.loads((study / "study.json").read_text())
    variants = json.loads((study / "artifacts" / "variants.json").read_text())
    period = study_period_ms(meta)
    bad = {
        n: v["device_options"].get("latency_budget_ms")
        for n, v in variants.items()
        if v["device_options"].get("latency_budget_ms") != period
    }
    if bad:
        raise SystemExit(
            f"variant latency_budget_ms differs from study period_ms={period}: {bad}"
        )
    print(f"study period T = {period} ms (from study.json)", flush=True)
    sched_path = study / "schedule.json"
    if sched_path.exists():
        saved = json.loads(sched_path.read_text())
        if saved["items"] != items:
            raise SystemExit(
                "schedule differs from the saved one; rerun with the original --rounds/--clocks/--seed/--order"
            )
    sys.path.insert(0, str(REPO / "src"))
    from crest.devices import CompileResult
    from crest.microcontrollers import stm32_nucleo_n657x0 as stm
    from crest.microcontrollers import stm32_measurement

    proto = meta["firmware_protocol"]
    if stm32_measurement.PROTOCOL != proto:
        raise SystemExit("host validator protocol differs from built firmware protocol")
    if sha256(meta["config"]) != meta["config_sha256"]:
        raise ValueError("build configuration changed")
    dev = load_yaml(meta["config"])["device"]
    timesteps = int(meta["timesteps"])
    input_dim = int(meta["input_dim"])
    runtime = {
        "serial_port": args.dut_port,
        "harness_serial_port": args.harness_port,
        "baud_rate": 115200,
        "harness_fqbn": dev.get("harness_fqbn", "arduino:mbed_nano:nano33ble"),
        "harness_auto_flash": args.harness_auto_flash,
        "dut_ready_timeout_s": float(dev.get("dut_ready_timeout_s", 12.0)),
        "serial_timeout_s": max(30.0, float(dev.get("serial_timeout_s", 12.0))),
        # Harness defines must match the compiled variants (backend defaults).
        "harness_arm_pin": int(dev.get("harness_arm_pin", 3)),
        "harness_trigger_pin": int(dev.get("harness_trigger_pin", 2)),
        "dut_arm_hold_ms": int(dev.get("dut_arm_hold_ms", 600)),
        "harness_stable_low_ms": int(dev.get("harness_stable_low_ms", 500)),
        **{
            k: float(dev[k])
            for k in (
                "harness_ready_timeout_s",
                "harness_arm_timeout_s",
                "harness_active_timeout_s",
                "harness_done_timeout_s",
            )
            if k in dev
        },
    }
    if not sched_path.exists():
        save(
            sched_path,
            {
                "created_utc": now(),
                "rounds": args.rounds,
                "clocks_mhz": clocks,
                "seed": args.seed,
                "order": args.order,
                "items": items,
            },
        )
    prov = snapshot_provenance(
        study / "provenance" / f'run_{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    if prov["file_sha256"].get(str(FIRMWARE)) != meta["firmware_git_state"][
        "file_sha256"
    ].get(str(FIRMWARE)):
        print(
            "WARNING: firmware source changed since build; flashed binaries are the BUILD snapshot (see provenance/build)."
        )
    save(study / "runtime_kwargs.json", runtime)
    for n, item in enumerate(items, 1):
        mdir = study / "measurements" / item["id"]
        res_path = mdir / "result.json"
        prior = json.loads(res_path.read_text()) if res_path.exists() else None
        if prior and prior["status"] == "accepted":
            continue
        attempts = sorted(mdir.glob("attempt-*")) if mdir.exists() else []
        if attempts:
            last = attempts[-1] / "result.json"
            if (
                not last.exists()
                or json.loads(last.read_text()).get("status") == "exception"
            ):
                raise RuntimeError(
                    f"interrupted or failed attempt requires inspection: {attempts[-1]}"
                )
        if len(attempts) >= args.max_attempts:
            raise RuntimeError(
                "attempt budget exhausted for incomplete measurement: " + item["id"]
            )
        v = variants[variant_name(item["clock_mhz"], item["shape"])]
        phase, runs = v["phase"], v["measured_runs"]
        adir = mdir / f"attempt-{len(attempts) + 1:02d}"
        adir.mkdir(parents=True)
        save(adir / "started.json", dict(item, started_utc=now(), variant=v["project"]))
        print(f'[{n}/{len(items)}] {item["id"]} phase={phase} runs={runs}', flush=True)
        t0 = time.monotonic()
        backend, root, clock = _prepare_runtime_backend(
            v, adir, stm, CompileResult, phase, runs, item, period
        )
        try:
            r = backend._evaluate_single_phase(
                dirpath=root,
                phase=phase,
                arena_kb=-1,
                window_size=timesteps,
                num_channels=input_dim,
                measured_inference_runs=runs,
                **runtime,
            )
            rj = json.loads(jdump(dataclasses.asdict(r)))
            pm = rj.get("power_metrics") or {}
            rec_paths = pm.get("measurement_record_paths") or []
            dut_log = (
                json.loads(Path(rec_paths[-1]).read_text()).get("dut_log", [])
                if rec_paths
                else []
            )
            reasons = accept(
                pm, rj.get("error_code"), phase, runs, clock, proto, dut_log
            )
            out = dict(
                item,
                phase=phase,
                runs=runs,
                status="accepted" if not reasons else "rejected",
                reasons=reasons,
                wall_s=time.monotonic() - t0,
                finished_utc=now(),
                attempt=adir.name,
                latency_s=rj.get("latency_s"),
                error_code=rj.get("error_code"),
                power_metrics=pm,
                measurement_record_paths=rec_paths,
                protocol_lines=[
                    l for l in map(str, dut_log) if "measurement protocol output" in l
                ],
            )
        except Exception as exc:  # preserve evidence; stop for inspection
            out = dict(
                item,
                phase=phase,
                runs=runs,
                status="exception",
                reasons=[f"{type(exc).__name__}: {exc}"],
                wall_s=time.monotonic() - t0,
                finished_utc=now(),
                attempt=adir.name,
            )
            save(adir / "result.json", out)
            save(res_path, out)
            raise
        save(adir / "result.json", out)
        save(res_path, out)
        print(
            f'    {out["status"]} E={pm.get("energy_mj_per_inference")} mJ wall={out["wall_s"]:.0f}s {reasons or ""}',
            flush=True,
        )
    incomplete = [
        item["id"]
        for item in items
        if not (study / "measurements" / item["id"] / "result.json").exists()
        or json.loads(
            (study / "measurements" / item["id"] / "result.json").read_text()
        ).get("status")
        != "accepted"
    ]
    if incomplete:
        raise RuntimeError(
            "measurements remain incomplete or rejected: " + ", ".join(incomplete)
        )
    print("run finished; analyze with: analyze --study", study)


# --------------------------------------------------------------------------- analyze


def mean_sd(xs):
    """Summarize finite recorded values without inventing missing samples.

    Parameters
    ----------
    xs : iterable[float or None]
        Values to summarize, optionally containing None or nonfinite values.

    Returns
    -------
    tuple
        Mean (None when empty), sample standard deviation (None with fewer
        than two values), and number of finite values.
    """
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    if not xs:
        return (None, None, 0)
    return (
        statistics.fmean(xs),
        statistics.stdev(xs) if len(xs) > 1 else None,
        len(xs),
    )


def solve(E, C, W_lo, lo, W_hi, hi, T):
    """Solve busy and effective sleep power from two cadenced window energies.

    Parameters
    ----------
    E : float
        Continuous energy per inference, in mJ.
    C : float
        Active time per inference, in seconds.
    W_lo, W_hi : float
        Total measured window energy in mJ at each run count.
    lo, hi : int or float
        Finite numeric run counts satisfying ``hi > lo >= 1``. Integer
        counts are conventional but are not required by this solver.
    T : float
        Release period in seconds, strictly greater than ``C``.

    Returns
    -------
    tuple[float, float]
        Busy-wait power and effective sleep power in mW.
    """
    if (
        not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            for v in (E, C, W_lo, lo, W_hi, hi, T)
        )
        or T <= C
        or hi <= lo
        or lo < 1
    ):
        raise ValueError("finite measurements, T > C, and hi > lo >= 1 required")
    gap = T - C
    p_stop = ((W_hi - hi * E) - (W_lo - lo * E)) / ((hi - lo) * gap)
    p_busy = (W_lo - lo * E) / gap - (lo - 1) * p_stop
    return p_busy, p_stop


def cmd_analyze(args):
    """Analyze accepted recorded waiting-power triples without hardware access.

    Parameters
    ----------
    args : argparse.Namespace
        Study path, analysis output path, and active-time source.
    """
    study = Path(args.study).resolve()
    out = Path(args.output).resolve() if args.output else study / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((study / "study.json").read_text())
    lo = int(meta.get("low_runs", 1))
    low = f"w{lo}"
    T = study_period_ms(meta) / 1000.0
    results = [
        json.loads(p.read_text())
        for p in sorted((study / "measurements").glob("*/result.json"))
    ]
    raw_rows, by = [], {}
    for r in results:
        pm = r.get("power_metrics") or {}
        row = {
            k: r.get(k)
            for k in (
                "id",
                "round",
                "clock_mhz",
                "shape",
                "phase",
                "runs",
                "status",
                "attempt",
                "wall_s",
            )
        }
        row.update(
            {
                k: pm.get(k)
                for k in (
                    "energy_mj_per_inference",
                    "avg_power_mw",
                    "timer_per_inference_s",
                    "dwt_cycles_per_inference",
                    "timer_per_window_s",
                    "harness_latency_s",
                    "clock_hz",
                    "slept_slots",
                    "stop_entries",
                    "rtc_sleep_total_ms",
                    "wake_recovery_us",
                    "deadline_miss_count",
                )
            }
        )
        cyc, hz = pm.get("dwt_cycles_per_inference"), pm.get("clock_hz")
        row["C_dwt_s"] = (
            cyc / hz
            if isinstance(cyc, (int, float))
            and isinstance(hz, (int, float))
            and cyc > 0
            and hz > 0
            else None
        )
        row["reasons"] = ";".join(r.get("reasons") or [])
        row["protocol"] = ";".join(
            l.split(":", 1)[-1].strip() for l in r.get("protocol_lines") or []
        )
        row["measurement_record_paths"] = ";".join(
            r.get("measurement_record_paths") or []
        )
        raw_rows.append(row)
        if r["status"] == "accepted":
            by[(r["round"], r["clock_mhz"], r["shape"])] = row
    per_round = []
    for rnd, clock in sorted({(k[0], k[1]) for k in by}):
        c, w10, wl = (
            by.get((rnd, clock, "continuous")),
            by.get((rnd, clock, "w10")),
            by.get((rnd, clock, low)),
        )
        if not (c and w10 and wl):
            continue
        E = c["energy_mj_per_inference"]
        # v4: continuous timer_per_inference_s = RTC window / runs; DWT = active invoke cycles / clock.
        C_rtc, C_dwt = c["timer_per_inference_s"], c["C_dwt_s"]
        c_source = "dwt" if (args.latency_source == "dwt" and C_dwt) else "rtc"
        C = C_dwt if c_source == "dwt" else C_rtc
        if not (C and 0 < C < T):
            print(f"skip round {rnd} {clock} MHz: C={C} not in (0, T={T})")
            continue
        W10, WL = (
            10 * w10["energy_mj_per_inference"],
            lo * wl["energy_mj_per_inference"],
        )
        p_busy, p_stop = solve(E, C, WL, lo, W10, 10, T)
        # Split folded overhead: awake (margin + recovery + poll) at P_busy, requested sleep S in STOP.
        S = (w10["rtc_sleep_total_ms"] or 0) / 1000.0
        awake_wait = 9 * (T - C) - S
        p_stop_sleep = (
            ((W10 - 10 * E - (T - C) * p_busy - awake_wait * p_busy) / S)
            if S > 0
            else None
        )
        per_round.append(
            {
                "round": rnd,
                "clock_mhz": clock,
                "E_inf_mj": E,
                "C_s": C,
                "C_source": c_source,
                "C_rtc_continuous_s": C_rtc,
                "C_dwt_continuous_s": C_dwt,
                "C_cadenced_w10_s": w10["timer_per_inference_s"],
                f"C_cadenced_{low}_s": wl["timer_per_inference_s"],
                "W10_mj": W10,
                f"W{lo}_mj": WL,
                "P_busy_mw": p_busy,
                "P_stop_eff_mw": p_stop,
                "P_stop_sleep_only_mw": p_stop_sleep,
                "rtc_sleep_total_s_w10": S,
                "release_mj_measured": W10 / 10,
                "release_mj_stop_tail": (W10 - (T - C) * (p_busy - p_stop)) / 10,
                "protocol": ";".join(
                    sorted({c["protocol"], w10["protocol"], wl["protocol"]})
                ),
                "records": ";".join(
                    x["measurement_record_paths"] for x in (c, w10, wl)
                ),
            }
        )
    summary = []
    for clock in sorted({r["clock_mhz"] for r in per_round}):
        rows = [r for r in per_round if r["clock_mhz"] == clock]
        s = {"clock_mhz": clock}
        for key, name in (
            ("P_busy_mw", "P_busy_mw"),
            ("P_stop_eff_mw", "P_stop_eff_mw"),
            ("P_stop_sleep_only_mw", "P_stop_sleep_only_mw"),
            ("E_inf_mj", "E_inf_mj"),
            ("C_s", "C_s"),
            ("C_rtc_continuous_s", "C_rtc_continuous_s"),
            ("C_dwt_continuous_s", "C_dwt_continuous_s"),
            ("release_mj_measured", "release_mj_measured"),
            ("release_mj_stop_tail", "release_mj_stop_tail"),
        ):
            m, sd, n = mean_sd([r[key] for r in rows])
            s[f"{name}_mean"], s[f"{name}_sd"] = m, sd
        s["n"] = len(rows)
        summary.append(s)

    def write(name, rows):
        """Write rows to a CSV file using first-seen column order.

        Parameters
        ----------
        name : str
            Output filename.
        rows : list[dict]
            Rows to write; an empty list produces an empty file.
        """
        if not rows:
            (out / name).write_text("")
            return
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with (out / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    write("waiting_power_by_clock.csv", summary)
    write("waiting_power_per_round.csv", per_round)
    write("raw_measurements.csv", raw_rows)
    protocols = sorted({r["protocol"] for r in raw_rows if r["protocol"]})
    save(
        out / "summary.json",
        {
            "study": str(study),
            "analyzed_utc": now(),
            "period_s": T,
            "period_ms": round(T * 1000),
            "low_runs": lo,
            "latency_source": args.latency_source,
            "equations": {
                "P_busy": f"(W{lo} - {lo} E_inf)/(T-C) - ({lo}-1) P_stop_eff",
                "P_stop_eff": f"((W10 - 10 E_inf) - (W{lo} - {lo} E_inf)) / ((10-{lo}) (T-C))",
            },
            "measurements": len(results),
            "accepted": sum(r["status"] == "accepted" for r in results),
            "complete_round_clock_triples": len(per_round),
            "dut_protocols_observed": protocols,
            "built_firmware_protocol": meta.get("firmware_protocol"),
            "build_firmware_git_state": meta.get("firmware_git_state"),
            "by_clock": summary,
        },
    )
    for s in summary:
        f = lambda v: "nan" if v is None else f"{v:.1f}"
        print(
            f"{s['clock_mhz']:>4} MHz n={s['n']}  P_busy={f(s['P_busy_mw_mean'])}±{f(s['P_busy_mw_sd'])} mW "
            f"P_stop_eff={f(s['P_stop_eff_mw_mean'])}±{f(s['P_stop_eff_mw_sd'])} mW"
        )
    print(f"wrote {out}")


def main(argv=None):
    """Parse command-line arguments and dispatch the requested operation.

    Parameters
    ----------
    argv : list[str] or None, optional
        Arguments to parse; None uses the process command line.
    """
    global FIRMWARE
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser(
        "build",
        help="export the candidate once and compile one image per (clock, shape); no hardware",
    )
    b.add_argument(
        "--output", required=True, help="new study directory; must not exist"
    )
    b.add_argument(
        "--firmware",
        type=Path,
        default=FIRMWARE,
        help="firmware source used for protocol/provenance checks",
    )
    b.add_argument("--initialization-seed", type=int, required=True)
    b.add_argument(
        "--period-ms",
        type=int,
        required=True,
        help="positive cadenced release period in ms",
    )
    b.add_argument("--config", required=True)
    b.add_argument(
        "--manifest",
        required=True,
        help="candidate manifest with timesteps and input_dim",
    )
    b.add_argument("--candidate", required=True)
    b.add_argument("--clocks", type=int, nargs="+", required=True)
    b.add_argument(
        "--low-runs",
        type=int,
        default=1,
        help="run count of the short cadenced window (default 1)",
    )
    b.add_argument(
        "--expect-protocol",
        default=EXPECTED_PROTOCOL,
        help="required CREST_DUT_MEASUREMENT_PROTOCOL",
    )
    b.add_argument("--dry-run", action="store_true")
    r = sub.add_parser(
        "run", help="program and measure the interleaved schedule (hardware)"
    )
    r.add_argument("--study", required=True, help="study directory produced by build")

    r.add_argument("--firmware", type=Path, default=FIRMWARE)
    r.add_argument("--rounds", type=int, required=True)
    r.add_argument("--clocks", type=int, nargs="+", required=True)
    r.add_argument(
        "--low-runs", type=int, default=1, help="only used by --dry-run without a study"
    )
    r.add_argument("--seed", type=int, required=True)
    r.add_argument(
        "--order",
        choices=("grouped", "shuffled"),
        default="grouped",
        help="grouped: per round, shuffled runs=10 block and shuffled low-runs block, alternating block order "
        "(limits harness reflashes); shuffled: full per-round shuffle",
    )
    r.add_argument("--dut-port", required=True)
    r.add_argument("--harness-port", required=True)
    r.add_argument(
        "--harness-auto-flash",
        choices=("once", "always"),
        default="once",
        help="'once' reflashes the harness whenever CREST_INFERENCE_RUNS changes (required for low-runs)",
    )
    r.add_argument(
        "--max-attempts",
        type=int,
        default=2,
        help="attempts per scheduled measurement across invocations",
    )
    r.add_argument(
        "--board-ready", action="store_true", help="required to touch hardware"
    )
    r.add_argument("--dry-run", action="store_true")
    a = sub.add_parser("analyze", help="offline per-clock P_busy / P_stop_eff")
    a.add_argument("--study", required=True)
    a.add_argument("--output", help="default <study>/analysis")
    a.add_argument(
        "--latency-source",
        choices=("dwt", "rtc"),
        default="dwt",
        help="C for T-C: continuous DWT active time (default; falls back to RTC) or RTC window/runs",
    )
    args = p.parse_args(argv)
    if hasattr(args, "initialization_seed") and args.initialization_seed < 0:
        p.error("--initialization-seed must be nonnegative")
    if hasattr(args, "firmware"):
        FIRMWARE = args.firmware.resolve()
    for field in ("period_ms", "rounds", "max_attempts"):
        if hasattr(args, field) and getattr(args, field) <= 0:
            p.error(f"--{field.replace(chr(95), chr(45))} must be positive")
    if hasattr(args, "clocks") and any(c <= 0 for c in args.clocks):
        p.error("--clocks must contain positive integers")
    {"build": cmd_build, "run": cmd_run, "analyze": cmd_analyze}[args.command](args)


if __name__ == "__main__":
    main()
