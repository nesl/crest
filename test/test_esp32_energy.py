# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Strict energy evidence and shared protocol reuse tests."""
from unittest.mock import Mock

import pytest

from crest.hil.protocol import HandshakeResult
from crest.microcontrollers import arduino_base
from crest.microcontrollers.esp32_idf import ESP32IDFDevice
from crest.microcontrollers.esp32_measurement import validate_energy_log


def energy_log(count=20):
    """Return representative shared-logger telemetry.

    Parameters
    ----------
    count : int
        Completed inference windows.

    Returns
    -------
    list[str]
        Harness serial evidence in shared merged-log format.
    """
    return ['HARNESS: ' + line for line in ['WINDOWS READY 1',
        f'energy windows: {count} elapsed_us: 4000000', f'runs: {count}',
        'energy output (mJ): 10.0', 'avg power output (mW): 50.0',
        'bus voltage output (V): 3.3', 'harness timer output: 0.2', 'DONE']]


def test_energy_normalization_counts_all_fixtures():
    """Energy normalization counts all fixtures.
    """
    lines = energy_log()
    metrics = arduino_base._parse_power_metrics([s.removeprefix('HARNESS: ') for s in lines])
    evidence, reasons = validate_energy_log(lines=lines, metrics=metrics, expected_runs=20, invocation_us=4000000)
    assert reasons == []
    assert evidence['measurement_energy_total_mj'] == 200
    assert evidence['measurement_energy_idle_subtracted'] is False
    assert evidence['measurement_energy_raw_trace_available'] is False


@pytest.mark.parametrize('bad', [None, -1, 0, float('nan'), float('inf'), True])
def test_invalid_energy_fails(bad):
    """Invalid energy fails.

    Parameters
    ----------
    bad : object
        Test fixture or parameter for this regression.
    """
    evidence, reasons = validate_energy_log(lines=energy_log(), metrics={
        'energy_mj_per_inference': bad, 'avg_power_mw': 50, 'bus_voltage_v': 3.3,
        'harness_latency_s': 0.2}, expected_runs=20, invocation_us=4000000)
    assert not evidence['measurement_energy_valid']
    assert 'invalid_energy_mj_per_inference' in reasons


@pytest.mark.parametrize('lines', [[], energy_log(2), energy_log()[:-1],
    energy_log() + ['HARNESS: harness error: active_timeout']])
def test_missing_protocol_count_or_done_fails(lines):
    """Missing protocol count or done fails.

    Parameters
    ----------
    lines : object
        Test fixture or parameter for this regression.
    """
    evidence, reasons = validate_energy_log(lines=lines, metrics={}, expected_runs=20, invocation_us=4000000)
    assert reasons and not evidence['measurement_energy_valid']


def test_windowed_handshake_failure_never_falls_back_or_flashes(monkeypatch):
    """Windowed handshake failure never falls back or flashes.

    Parameters
    ----------
    monkeypatch : object
        Test fixture or parameter for this regression.
    """
    monkeypatch.setattr(arduino_base, 'ensure_harness_firmware', Mock(return_value=False))
    handshake = Mock(return_value=HandshakeResult([], [], False, False, None, None, 'unsupported'))
    monkeypatch.setattr(arduino_base.hil_protocol, 'run_handshake', handshake)
    fallback = Mock(side_effect=AssertionError('DUT fallback forbidden'))
    monkeypatch.setattr(arduino_base.hil_protocol, 'run_dut_only', fallback)
    result = arduino_base.measure_serial(serial_port='dut', baud_rate=115200,
        serial_timeout_s=1, harness_serial_port='harness', harness_auto_flash='never', windowed_energy=True)
    assert result.latency_s is None
    assert handshake.call_args.kwargs['windowed_energy'] is True
    fallback.assert_not_called()


@pytest.mark.parametrize("flash_policy", ["never", "once", "always"])
def test_esp32_requires_energy_after_shared_helper(monkeypatch, flash_policy):
    """Esp32 requires energy after shared helper.

    Parameters
    ----------
    monkeypatch : object
        Test fixture or parameter for this regression.
    flash_policy : str
        Configured harness installation policy forwarded to the shared helper.
    """
    from crest.devices import MeasureResult
    helper = Mock(return_value=MeasureResult(0.2, None, [], None))
    monkeypatch.setattr(arduino_base, 'measure_serial', helper)
    device = ESP32IDFDevice()
    device._active_manifest = {'fixture_count': 10, 'measured_runs_per_fixture': 2}
    result = device.measure(serial_port='dut', baud_rate=115200, serial_timeout_s=1,
                            harness_serial_port='harness', harness_auto_flash=flash_policy)
    assert helper.call_args.kwargs['measured_inference_runs'] == 20
    assert helper.call_args.kwargs['harness_auto_flash'] == flash_policy
    assert not result.power_metrics['measurement_energy_valid']
    assert not result.power_metrics['measurement_instrumentation_valid']


def test_unsupported_window_protocol_does_not_start_dut(monkeypatch):
    """Unsupported window protocol does not start dut.

    Parameters
    ----------
    monkeypatch : object
        Test fixture or parameter for this regression.
    """
    from crest.hil import protocol
    harness = Mock()
    harness.__enter__ = Mock(return_value=harness)
    harness.__exit__ = Mock(return_value=False)
    serial = Mock(return_value=harness)
    monkeypatch.setattr(protocol.serial, 'Serial', serial)
    monkeypatch.setattr(protocol, '_wait_for_token', Mock(side_effect=[True, False]))
    result = protocol.run_handshake(dut_port='dut', harness_port='harness', baud_rate=115200,
        dut_ready_timeout_s=1, dut_timer_timeout_s=1, harness_ready_timeout_s=1,
        harness_active_timeout_s=1, harness_done_timeout_s=1, windowed_energy=True)
    assert result.error == 'harness_window_protocol_unsupported'
    assert serial.call_count == 1
    assert serial.call_args.kwargs['exclusive'] is True
    assert [call.args[0] for call in harness.write.call_args_list] == [b'PING\n', b'WINDOWS 1\n']


def test_prefixed_identity_evidence_remains_valid():
    """Prefixed identity evidence remains valid.
    """
    from test.test_esp32_backend import ESP32MeasurementEvidenceTests
    from crest.microcontrollers.esp32_measurement import validate_measurement_log
    case = ESP32MeasurementEvidenceTests()
    log = ['DUT: ' + line for line in case._valid_log()]
    evidence, _ = validate_measurement_log(lines=log, manifest=case._manifest(),
                                          latency_s=3.55, psram_bytes=8*1024*1024)
    assert evidence['measurement_instrumentation_valid']
    log = [line.replace('model_sha256=' + 'a'*64, 'model_sha256=' + 'c'*64) for line in log]
    evidence, _ = validate_measurement_log(lines=log, manifest=case._manifest(),
                                          latency_s=3.55, psram_bytes=8*1024*1024)
    assert not evidence['measurement_instrumentation_valid']


@pytest.mark.parametrize('resolution,trial', [(64, 114), (96, 71)])
def test_staged_saved_artifact_fixtures_without_conversion(tmp_path, monkeypatch, resolution, trial):
    """Staged saved artifact fixtures without conversion.

    Parameters
    ----------
    tmp_path : object
        Test fixture or parameter for this regression.
    monkeypatch : object
        Test fixture or parameter for this regression.
    resolution : object
        Test fixture or parameter for this regression.
    trial : object
        Test fixture or parameter for this regression.
    """
    import os
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from crest.devices import CandidatePrepareRequest
    configured = os.environ.get('CREST_SAVED_REPLAY_INPUTS')
    if not configured:
        pytest.skip('Set CREST_SAVED_REPLAY_INPUTS to the saved campaign handoff directory')
    root = Path(configured)
    project = tmp_path/'project'
    project.mkdir()
    (project/'CMakeLists.txt').write_text('project(test)\n')
    device = ESP32IDFDevice(device_options={'project_root': project,
        'fixture_npz_path': root/f'fixtures/representative_{resolution}.npz',
        'fixture_manifest_path': root/f'fixtures/representative_{resolution}.json'})
    monkeypatch.setattr('crest.hardware.convert_to_tflite_model', Mock(side_effect=AssertionError('reconversion forbidden')))
    path = root/f'artifacts/trial_{trial}.tflite'
    request = CandidatePrepareRequest(config=SimpleNamespace(device=SimpleNamespace(measured_inference_runs=2)),
        model=None, model_variant='supplied_tflite', artifact_root=tmp_path/'candidates',
        tflite_model_path=tmp_path/'unused.tflite', calibration_split=None, quantization_mode='int8_ptq',
        input_shape=(resolution,resolution), checkpoint_path=None, source_tflite_path=path)
    staged = device.prepare_candidate(request=request)
    assert (staged/'main/crest_candidate.tflite').read_bytes() == path.read_bytes()
    manifest = json.loads((staged/'main/crest_model_manifest.json').read_text())
    assert manifest['fixture_count'] == 10
    assert manifest['measured_runs_per_fixture'] == 2


def test_windowed_session_refreshes_dut_readiness(monkeypatch):
    """Recover a readiness banner lost while opening the DUT after flash.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Stub serial ports and waits without hardware access.
    """
    from crest.hil import protocol
    ports = [Mock(), Mock()]
    for port in ports:
        port.__enter__ = Mock(return_value=port)
        port.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(protocol.serial, 'Serial', Mock(side_effect=ports))
    monkeypatch.setattr(protocol, '_wait_for_token', Mock(return_value=True))
    monkeypatch.setattr(protocol, '_read_until_prefix', Mock(return_value=True))
    monkeypatch.setattr(protocol, '_read_until_done', Mock(return_value=True))
    result = protocol.run_handshake(dut_port='dut', harness_port='harness', baud_rate=115200,
        dut_ready_timeout_s=1, dut_timer_timeout_s=1, harness_ready_timeout_s=1,
        harness_active_timeout_s=1, harness_done_timeout_s=1, windowed_energy=True)
    assert result.error is None
    assert [call.args[0] for call in ports[1].write.call_args_list] == [b'PING\n', b'START\n']
