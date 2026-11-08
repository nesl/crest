# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Synthetic tests: no backend imports, compilation, ports or hardware."""
import copy
import json
from pathlib import Path

import pytest

from analysis_scripts.policy_replay import replay


def manifest(tmp_path):
    header = tmp_path / "phase.h"
    header.write_text(
        "#define CREST_DUT_SELECTED_PHASE CREST_DUT_PHASE_CADENCED\n"
        "#define CREST_DUT_MEASURED_RUNS 10\n#define CREST_DUT_CPU_CLOCK_MHZ 400\n"
        "#define CREST_DUT_LATENCY_BUDGET_MS 500\n"
    )
    elf = tmp_path / "app.elf"
    elf.write_bytes(replay.PROTOCOLS["awake"].encode())
    variant = dict(
        name="model_awake",
        candidate="model",
        phase="cadenced",
        policy="awake",
        runs=10,
        clock_mhz=400,
        period_ms=500,
        device_options={"latency_budget_ms": 500},
        project=".",
        phase_header="phase.h",
        phase_header_sha256=replay.sha256(header),
        app_elf_sha256=replay.sha256(elf),
        compile_result={"success": True, "app_elf_path": "app.elf"},
    )
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(dict(schema_version=1, variants=[variant], blocks=[["model_awake"]]))
    )
    return path


def evidence():
    return dict(
        error_code=0,
        latency_s=0.01,
        dut_log=[
            f'measurement protocol output: {replay.PROTOCOLS["awake"]}',
            "cadence budget (ms): 500",
        ],
        harness_log=["energy evidence"],
        power_metrics=dict(
            measurement_instrumentation_valid=True,
            measurement_deployment_eligible=True,
            appli_programmed=True,
            weights_programmed=True,
            phase="cadenced",
            clock_hz=400000000,
            runs=10,
            timer_per_window_s=5.0,
            harness_latency_s=0.5,
            rtc_clock_source="LSE",
            rtc_clock_hz_nominal=32768,
            energy_mj_per_inference=1.0,
            slept_slots=9,
            stop_entries=9,
            deadline_miss_count=0,
        ),
    )


def test_freeze_identity_and_tamper(tmp_path):
    source = manifest(tmp_path)
    study = replay.prepare(source, tmp_path / "out")
    data = json.loads((study / "study.json").read_text())
    variant = data["variants"][0]
    replay.validate_variant(variant, study)
    (tmp_path / "app.elf").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="digest mismatch"):
        replay.validate_variant(variant, study)


def test_schedule_explicit_partition():
    names = ["x", "y", "z"]
    first = replay.schedule(3, names, [["x", "y"], ["z"]], 42)
    assert first == replay.schedule(3, names, [["x", "y"], ["z"]], 42)
    assert len({item["id"] for item in first}) == 9
    with pytest.raises(ValueError):
        replay.schedule(1, names, [["x", "x"], ["z"]], 42)


def test_real_acceptance_fails_closed(tmp_path):
    path = manifest(tmp_path)
    variant = json.loads(path.read_text())["variants"][0]
    result = evidence()
    assert replay.accept(result, variant) == []
    bad = copy.deepcopy(result)
    bad["power_metrics"]["stop_entries"] = 10
    assert replay.accept(bad, variant)
    bad = copy.deepcopy(result)
    bad["power_metrics"]["rtc_clock_source"] = "LSI"
    assert replay.accept(bad, variant)
    bad = copy.deepcopy(result)
    bad["dut_log"] = []
    assert replay.accept(bad, variant)


def test_run_resume_and_stop_after_failure(tmp_path, monkeypatch):
    from analysis_scripts.policy_replay import native

    study = replay.prepare(manifest(tmp_path), tmp_path / "out")
    calls = []

    def evaluate(*args):
        calls.append(args)
        return evidence()

    monkeypatch.setattr(native, "evaluate", evaluate)
    kwargs = dict(
        rounds=1,
        seed=42,
        device={"device_id": "synthetic-policy-test"},
        board_ready=True,
        lock_dir=tmp_path / "locks",
    )
    replay.run(study, **kwargs)
    replay.run(study, **kwargs)
    assert len(calls) == 1
    with pytest.raises(ValueError, match="saved schedule"):
        replay.run(study, **dict(kwargs, rounds=2))
    other = replay.prepare(manifest(tmp_path), tmp_path / "other")
    monkeypatch.setattr(native, "evaluate", lambda *args: dict(error_code=1))
    with pytest.raises(RuntimeError, match="rejected"):
        replay.run(other, **kwargs)
    with pytest.raises(RuntimeError, match="requires inspection"):
        replay.run(other, **kwargs)


def test_stop_policy_and_hardware_authorization(tmp_path):
    variant = json.loads(manifest(tmp_path).read_text())["variants"][0]
    variant["policy"] = "stop"
    result = evidence()
    result["dut_log"][0] = f'measurement protocol output: {replay.PROTOCOLS["stop"]}'
    result["power_metrics"].update(slept_slots=10, stop_entries=10)
    assert replay.accept(result, variant) == []
    with pytest.raises(ValueError, match="board_ready"):
        replay.run(tmp_path, rounds=1, seed=0, device={})


@pytest.fixture
def native_case(tmp_path, monkeypatch):
    """Install a synthetic native backend and prepared evidence without hardware."""
    import dataclasses
    import sys
    import types
    from analysis_scripts.policy_replay import native

    study = replay.prepare(manifest(tmp_path), tmp_path / "study")
    variant = json.loads((study / "study.json").read_text())["variants"][0]
    variant.update(window_size=1, num_channels=1)
    attempt = study / "attempt"
    attempt.mkdir()
    inc = tmp_path / "inc"
    inc.mkdir()
    header = inc / "crest_dut_phase_config.h"
    header.write_bytes((tmp_path / "phase.h").read_bytes())
    variant["phase_header"] = str(header)
    weights = tmp_path / "weights.bin"
    loader = tmp_path / "loader"
    weights.write_bytes(b"weights")
    loader.touch()
    sidecar = tmp_path / "storage.json"
    replay.save(
        sidecar,
        dict(
            weights_blob_path=str(weights),
            weights_external_loader=str(loader),
            weights_blob_size=weights.stat().st_size,
            weights_blob_sha256=replay.sha256(weights),
            last_programmed_app="old",
        ),
    )
    record = attempt / "raw.json"
    metrics = {"energy_mj_per_inference": 1.0}
    raw = dict(
        finished_utc="finished",
        request={"phase": "cadenced"},
        acceptance={"instrumentation_valid": True, "deployment_eligible": True},
        result={"power_metrics": dict(metrics)},
        dut_log=["dut"],
        harness_log=["harness"],
    )
    replay.save(record, raw)
    original = object()
    stm = types.ModuleType("crest.microcontrollers.stm32_nucleo_n657x0")
    stm.validate_measurement = original
    stm._manifest_path = lambda paths: sidecar
    checks = []
    stm._validate_phase_config_header = lambda *args, **kwargs: checks.append(kwargs)

    @dataclasses.dataclass
    class Result:
        power_metrics: dict

    class Backend:
        def __init__(self, **kwargs):
            self.options = kwargs

        def _resolve_paths(self, work):
            return types.SimpleNamespace(inc_dir=inc)

        def _evaluate_single_phase(self, **kwargs):
            assert self.compile() is cached[-1]
            assert json.loads(sidecar.read_text())["last_programmed_app"] is None
            assert (
                stm.validate_measurement is native.validator_awake.validate_measurement
            )
            self._write_runtime_phase_config(
                paths=self._resolve_paths(None),
                selected_phase=state.get("phase", "cadenced"),
                measured_runs=state.get("runs", 10),
            )
            if state.get("raise"):
                raise RuntimeError("synthetic backend failure")
            return Result(
                dict(
                    metrics, measurement_record_paths=[str(state.get("record", record))]
                )
            )

    cached = []

    def compile_result(**kwargs):
        value = types.SimpleNamespace(**kwargs)
        cached.append(value)
        return value

    devices = types.ModuleType("crest.devices")
    devices.CompileResult = compile_result
    microcontrollers = types.ModuleType("crest.microcontrollers")
    microcontrollers.stm32_nucleo_n657x0 = stm
    monkeypatch.setitem(sys.modules, "crest.devices", devices)
    monkeypatch.setitem(sys.modules, "crest.microcontrollers", microcontrollers)
    monkeypatch.setitem(sys.modules, "crest.microcontrollers.stm32_nucleo_n657x0", stm)
    stm.STM32NucleoN657X0QDevice = Backend
    state = {}
    device = {
        "runtime_kwargs": {
            "serial_port": str(loader),
            "harness_serial_port": str(loader),
        }
    }
    return types.SimpleNamespace(
        evaluate=lambda: native.evaluate(variant, study, attempt, device),
        state=state,
        raw=raw,
        record=record,
        stm=stm,
        original=original,
        checks=checks,
        tmp=tmp_path,
    )


def test_native_cached_setup_and_validator_restoration(native_case):
    case = native_case
    assert case.evaluate()["dut_log"] == ["dut"]
    assert len(case.checks) == 1
    assert case.stm.validate_measurement is case.original
    case.state["raise"] = True
    with pytest.raises(RuntimeError, match="synthetic backend failure"):
        case.evaluate()
    assert case.stm.validate_measurement is case.original


@pytest.mark.parametrize(
    "fault", ["outside", "unfinished", "phase", "metrics", "acceptance"]
)
def test_native_rejects_raw_evidence(native_case, fault):
    case = native_case
    if fault == "outside":
        case.state["record"] = case.tmp / "outside.json"
    elif fault == "unfinished":
        case.raw["finished_utc"] = None
    elif fault == "phase":
        case.raw["request"]["phase"] = "back_to_back"
    elif fault == "metrics":
        case.raw["result"]["power_metrics"]["energy_mj_per_inference"] = 2.0
    else:
        case.raw["acceptance"]["instrumentation_valid"] = False
    replay.save(case.record, case.raw)
    with pytest.raises(ValueError):
        case.evaluate()
    assert case.stm.validate_measurement is case.original


@pytest.mark.parametrize("changed_request", [{"phase": "back_to_back"}, {"runs": 11}])
def test_native_header_hook_rejects_changed_request(native_case, changed_request):
    case = native_case
    case.state.update(changed_request)
    with pytest.raises(ValueError, match="requested phase/run count"):
        case.evaluate()
    assert case.stm.validate_measurement is case.original
