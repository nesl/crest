"""Synthetic tests: no backend imports, compilation, ports or hardware."""
import copy
import json
from pathlib import Path

import pytest

from analysis_scripts.policy_replay import replay


def manifest(tmp_path):
    header = tmp_path / 'phase.h'
    header.write_text('#define CREST_DUT_SELECTED_PHASE CREST_DUT_PHASE_CADENCED\n'
        '#define CREST_DUT_MEASURED_RUNS 10\n#define CREST_DUT_CPU_CLOCK_MHZ 400\n'
        '#define CREST_DUT_LATENCY_BUDGET_MS 500\n')
    elf = tmp_path / 'app.elf'
    elf.write_bytes(replay.PROTOCOLS['awake'].encode())
    variant = dict(name='model_awake', candidate='model', phase='cadenced', policy='awake',
        runs=10, clock_mhz=400, period_ms=500, device_options={'latency_budget_ms': 500},
        project='.', phase_header='phase.h', phase_header_sha256=replay.sha256(header),
        app_elf_sha256=replay.sha256(elf), compile_result={'success': True, 'app_elf_path': 'app.elf'})
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(dict(schema_version=1, variants=[variant], blocks=[['model_awake']])))
    return path


def evidence():
    return dict(error_code=0, latency_s=.01, dut_log=[
        f'measurement protocol output: {replay.PROTOCOLS["awake"]}', 'cadence budget (ms): 500'],
        harness_log=['energy evidence'], power_metrics=dict(measurement_instrumentation_valid=True,
        measurement_deployment_eligible=True, appli_programmed=True, weights_programmed=True,
        phase='cadenced', clock_hz=400000000, runs=10, timer_per_window_s=5., harness_latency_s=.5,
        rtc_clock_source='LSE', rtc_clock_hz_nominal=32768, energy_mj_per_inference=1.,
        slept_slots=9, stop_entries=9, deadline_miss_count=0))


def test_freeze_identity_and_tamper(tmp_path):
    source = manifest(tmp_path)
    study = replay.prepare(source, tmp_path / 'out')
    data = json.loads((study / 'study.json').read_text())
    variant = data['variants'][0]
    replay.validate_variant(variant, study)
    (tmp_path / 'app.elf').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='digest mismatch'):
        replay.validate_variant(variant, study)


def test_schedule_explicit_partition():
    names = ['x', 'y', 'z']
    first = replay.schedule(3, names, [['x', 'y'], ['z']], 42)
    assert first == replay.schedule(3, names, [['x', 'y'], ['z']], 42)
    assert len({item['id'] for item in first}) == 9
    with pytest.raises(ValueError):
        replay.schedule(1, names, [['x', 'x'], ['z']], 42)


def test_real_acceptance_fails_closed(tmp_path):
    path = manifest(tmp_path)
    variant = json.loads(path.read_text())['variants'][0]
    result = evidence()
    assert replay.accept(result, variant) == []
    bad = copy.deepcopy(result)
    bad['power_metrics']['stop_entries'] = 10
    assert replay.accept(bad, variant)
    bad = copy.deepcopy(result)
    bad['power_metrics']['rtc_clock_source'] = 'LSI'
    assert replay.accept(bad, variant)
    bad = copy.deepcopy(result)
    bad['dut_log'] = []
    assert replay.accept(bad, variant)


def test_run_resume_and_stop_after_failure(tmp_path, monkeypatch):
    from analysis_scripts.policy_replay import native
    study = replay.prepare(manifest(tmp_path), tmp_path / 'out')
    calls = []
    def evaluate(*args):
        calls.append(args)
        return evidence()
    monkeypatch.setattr(native, 'evaluate', evaluate)
    kwargs = dict(rounds=1, seed=42, device={'device_id': 'synthetic-policy-test'}, board_ready=True,
                  lock_dir=tmp_path / 'locks')
    replay.run(study, **kwargs)
    replay.run(study, **kwargs)
    assert len(calls) == 1
    with pytest.raises(ValueError, match='saved schedule'):
        replay.run(study, **dict(kwargs, rounds=2))
    other = replay.prepare(manifest(tmp_path), tmp_path / 'other')
    monkeypatch.setattr(native, 'evaluate', lambda *args: dict(error_code=1))
    with pytest.raises(RuntimeError, match='rejected'):
        replay.run(other, **kwargs)
    with pytest.raises(RuntimeError, match='requires inspection'):
        replay.run(other, **kwargs)


def test_stop_policy_and_hardware_authorization(tmp_path):
    variant = json.loads(manifest(tmp_path).read_text())['variants'][0]
    variant['policy'] = 'stop'
    result = evidence()
    result['dut_log'][0] = f'measurement protocol output: {replay.PROTOCOLS["stop"]}'
    result['power_metrics'].update(slept_slots=10, stop_entries=10)
    assert replay.accept(result, variant) == []
    with pytest.raises(ValueError, match='board_ready'):
        replay.run(tmp_path, rounds=1, seed=0, device={})
