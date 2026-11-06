"""Hardware-free repeatability planning and evidence tests."""
import json
from pathlib import Path

import pytest

from . import experiment as backend
from .runner import prepare, verify


def manifest():
    return dict(candidates=[dict(candidate_id=k, clock_mhz=80, architecture={}) for k in ('a', 'b')],
                pairs=[dict(pair_id='pair', candidate_a='a', candidate_b='b')],
                timesteps=16, input_dim=3, period_ms=250, measured_runs=4,
                initialization_seed=7, prospective_protocol=backend.PROTOCOL)


def test_arbitrary_pair_count_odd_rounds_and_seal(tmp_path):
    source = tmp_path / 'source.json'
    source.write_text(json.dumps(manifest()))
    study = prepare(source, tmp_path / 'study', rounds=3, seed=17)
    plan = json.loads((study / 'schedule.json').read_text())
    assert len(plan['evaluations']) == 6
    assert {x['config_id'] for x in plan['evaluations']} == {'a', 'b'}
    verify(study)
    (study / 'manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='sealed file changed'):
        verify(study)


def test_refuses_unsafe_candidate_paths_before_output(tmp_path):
    data = manifest()
    data['candidates'][0]['candidate_id'] = '../escape'
    source = tmp_path / 'source.json'
    source.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='path components'):
        prepare(source, tmp_path / 'study', 1, 0)
    assert not (tmp_path / 'study').exists()


def test_configurable_protocol_validates_raw_records(tmp_path):
    config = dict(cpu_clock_mhz=80, window_size=16, num_channels=3, period_ms=250,
                  measured_runs=4, initialization_seed=7, dut_port='/dut', harness_port='/harness')
    phases = {}
    for label, phase in backend.PHASES.items():
        metrics = dict(measurement_instrumentation_valid=True, measurement_deployment_eligible=True,
                       appli_programmed=True, weights_programmed=True, energy_mj_per_inference=2.,
                       clock_hz=80_000_000, runs=4, phase=phase, timer_per_inference_s=.1,
                       timer_per_window_s=1. if label == 'cadenced' else .4,
                       deadline_miss_count=0, stop_entries=3)
        path = tmp_path / (label + '.json')
        metrics['measurement_record_paths'] = [str(path)]
        path.write_text(json.dumps(dict(finished_utc='now', request=dict(phase=phase,
            measured_inference_runs=4, window_size=16, num_channels=3,
            serial_port='/dut', harness_serial_port='/harness'),
            acceptance=dict(instrumentation_valid=True, deployment_eligible=True,
                            checks={'complete': True}, energy_mj_per_release=2.),
            result={'power_metrics': metrics}, dut_log=[backend.PROTOCOL])))
        phases[label] = dict(error_code=0, power_metrics=metrics)
    assert backend.validate_evaluation({'phases': phases}, config)['accepted']
    raw = json.loads((tmp_path / 'cadenced.json').read_text())
    raw['acceptance']['checks']['complete'] = False
    (tmp_path / 'cadenced.json').write_text(json.dumps(raw))
    assert not backend.validate_evaluation({'phases': phases}, config)['accepted']


def test_restart_after_backend_exception_requires_inspection(tmp_path, monkeypatch):
    import sys
    import types
    from .runner import execute

    source = tmp_path / 'source.json'
    source.write_text(json.dumps(manifest()))
    study = prepare(source, tmp_path / 'study', 1, 0)
    module = types.ModuleType('test_repeatability_adapter')
    calls = []

    class Adapter:
        def __init__(self, *args):
            pass

        def preflight(self, config):
            return {'ok': True}

        def evaluate(self, config, context):
            calls.append(context)
            raise RuntimeError('uncertain measurement interruption')

    module.Experiment = Adapter
    from . import runner
    monkeypatch.setattr(runner, 'experiment', module)
    kwargs = dict(root=study, device_id='test-device', dut_port='/dut', harness_port='/harness',
                  lock_dir=tmp_path / 'locks', board_ready=True)
    with pytest.raises(RuntimeError, match='uncertain measurement'):
        execute(**kwargs)
    with pytest.raises(RuntimeError, match='manual inspection'):
        execute(**kwargs)
    assert len(calls) == 1


def test_rejected_evaluation_exhaustion_is_failure(tmp_path, monkeypatch):
    import sys
    import types
    from .runner import execute

    source = tmp_path / 'source.json'
    source.write_text(json.dumps(manifest()))
    study = prepare(source, tmp_path / 'study', 1, 0, max_attempts=1)
    module = types.ModuleType('test_rejected_adapter')
    calls = []

    class Adapter:
        def __init__(self, *args):
            pass

        def preflight(self, config):
            return {'ok': True}

        def evaluate(self, config, context):
            calls.append(context)
            return {'accepted': False}

    module.Experiment = Adapter
    module.validate_evaluation = lambda raw, config: raw
    from . import runner
    monkeypatch.setattr(runner, 'experiment', module)
    kwargs = dict(root=study, device_id='device', dut_port='/dut', harness_port='/harness',
                  lock_dir=tmp_path / 'locks', board_ready=True)
    for _ in range(2):
        with pytest.raises(RuntimeError, match='attempt budget exhausted.*repeatability-'):
            execute(**kwargs)
    assert len(calls) == 1
