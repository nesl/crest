# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Hardware-free scheduling, acceptance, and waiting-power algebra tests."""
import pytest

from .runner import accept, schedule, shapes, solve, study_period_ms


def test_schedules_are_deterministic_and_end_in_ten_run_block():
    items = schedule(3, [80, 160], 2, 9, "grouped")
    assert items == schedule(3, [80, 160], 2, 9, "grouped")
    assert len(items) == 18
    assert shapes(2)[items[-1]["shape"]][1] == 10


def test_waiting_power_solution_and_invalid_gap():
    # E=2mJ, C=.1s, T=.5s; busy=50mW, stop=5mW.
    assert solve(2, 0.1, 22, 1, 58, 10, 0.5) == pytest.approx((50, 5))
    with pytest.raises(ValueError):
        solve(2, 0.5, 22, 1, 58, 10, 0.5)
    with pytest.raises(SystemExit):
        study_period_ms({"period_ms": 0})
    assert study_period_ms({"period_ms": 250, "period_s": 0.25}) == 250


def test_acceptance_fails_closed():
    assert accept({}, 0, "cadenced", 1, 80, "protocol", [])


@pytest.fixture
def fake_study(tmp_path, monkeypatch):
    import dataclasses
    import json
    import sys
    import types
    from . import runner

    study = tmp_path / "study"
    (study / "artifacts").mkdir(parents=True)
    config = study / "config.yaml"
    config.write_text("device: {}")
    events = []
    proto = "fake-protocol"
    meta = dict(
        period_ms=250,
        low_runs=1,
        clocks_mhz=[80],
        firmware_protocol=proto,
        config=str(config),
        config_sha256=runner.sha256(config),
        timesteps=16,
        input_dim=3,
        firmware_git_state={"file_sha256": {}},
    )
    runner.save(study / "study.json", meta)
    variants = {}
    for shape, (phase, runs) in runner.shapes(1).items():
        root = study / shape
        root.mkdir()
        (root / "crest_dut_phase_config.h").write_text(shape)
        runner.save(root / "sidecar.json", {"last_programmed_app": "old", "keep": 1})
        variants[runner.variant_name(80, shape)] = dict(
            project=str(root),
            phase=phase,
            measured_runs=runs,
            device_options={"latency_budget_ms": 250},
            phase_header_sha256=runner.sha256(root / "crest_dut_phase_config.h"),
            compile_result={"success": True},
        )
    runner.save(study / "artifacts" / "variants.json", variants)

    @dataclasses.dataclass
    class Result:
        error_code: int = 0
        power_metrics: dict = dataclasses.field(default_factory=dict)

    class Backend:
        def __init__(self, device_options):
            events.append(("backend", device_options))

        def _resolve_paths(self, root):
            return types.SimpleNamespace(inc_dir=root)

        def _evaluate_single_phase(self, **kwargs):
            root = kwargs["dirpath"]
            assert self.compile().success is True
            self._write_runtime_phase_config(
                paths=self._resolve_paths(root),
                selected_phase=kwargs["phase"],
                measured_runs=kwargs["measured_inference_runs"],
            )
            events.append(("evaluate", root.name))
            return Result()

    def validate(path, **kwargs):
        events.append(("validate", path.name, kwargs))

    stm = types.SimpleNamespace(
        STM32NucleoN657X0QDevice=Backend,
        _manifest_path=lambda paths: paths.inc_dir / "sidecar.json",
        _validate_phase_config_header=validate,
    )
    monkeypatch.setitem(
        sys.modules,
        "crest.devices",
        types.SimpleNamespace(CompileResult=lambda **kw: types.SimpleNamespace(**kw)),
    )
    monkeypatch.setitem(
        sys.modules,
        "crest.microcontrollers",
        types.SimpleNamespace(
            stm32_nucleo_n657x0=stm,
            stm32_measurement=types.SimpleNamespace(PROTOCOL=proto),
        ),
    )
    monkeypatch.setattr(runner, "snapshot_provenance", lambda dest: {"file_sha256": {}})
    monkeypatch.setattr(runner, "accept", lambda *args: [])
    args = types.SimpleNamespace(
        study=str(study),
        clocks=[80],
        low_runs=1,
        rounds=1,
        seed=2,
        order="grouped",
        dry_run=False,
        board_ready=True,
        dut_port=str(config),
        harness_port=str(config),
        harness_auto_flash="once",
        max_attempts=2,
    )
    return runner, args, study, variants, events


def test_cached_compile_header_validation_and_restart_skip(fake_study):
    import json

    runner, args, study, variants, events = fake_study
    runner.cmd_run(args)
    assert len([e for e in events if e[0] == "evaluate"]) == 3
    checks = [e[2] for e in events if e[0] == "validate"]
    assert len(checks) == 3
    assert all(
        c["cpu_clock_mhz"] == 80 and c["latency_budget_ms"] == 250 for c in checks
    )
    for v in variants.values():
        storage = json.loads((study / v["project"] / "sidecar.json").read_text())
        assert storage == {"last_programmed_app": None, "keep": 1}
    runner.cmd_run(args)
    assert len([e for e in events if e[0] == "evaluate"]) == 3


def test_header_change_stops_before_evaluation_and_restart_requires_inspection(
    fake_study,
):
    from pathlib import Path

    runner, args, study, variants, events = fake_study
    first = runner.schedule(1, [80], 1, 2, "grouped")[0]
    v = variants[runner.variant_name(80, first["shape"])]
    (Path(v["project"]) / "crest_dut_phase_config.h").write_text("changed")
    with pytest.raises(SystemExit, match="phase header changed"):
        runner.cmd_run(args)
    assert not any(e[0] == "evaluate" for e in events)
    with pytest.raises(RuntimeError, match="requires inspection"):
        runner.cmd_run(args)


def test_backend_exception_is_recorded_and_stops_restart(fake_study, monkeypatch):
    import json
    import sys

    runner, args, study, variants, events = fake_study
    backend = sys.modules[
        "crest.microcontrollers"
    ].stm32_nucleo_n657x0.STM32NucleoN657X0QDevice

    def interrupted(self, **kwargs):
        events.append(("interrupted", kwargs["phase"]))
        raise RuntimeError("fake interrupted measurement")

    monkeypatch.setattr(backend, "_evaluate_single_phase", interrupted)
    with pytest.raises(RuntimeError, match="fake interrupted"):
        runner.cmd_run(args)
    result = next(study.glob("measurements/*/result.json"))
    assert json.loads(result.read_text())["status"] == "exception"
    with pytest.raises(RuntimeError, match="requires inspection"):
        runner.cmd_run(args)
    assert len([e for e in events if e[0] == "interrupted"]) == 1


def test_relocated_provenance_sources_exist():
    from .runner import PROVENANCE_FILES

    assert PROVENANCE_FILES
    assert all(path.is_file() for path in PROVENANCE_FILES)
