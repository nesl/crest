# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Exact replay provenance, preflight and resume regressions."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from crest import pareto_replay as replay
from test.test_pareto_replay import base_row, source_config, write_source_run, write_target_config, make_args


@pytest.fixture
def saved_run(tmp_path, monkeypatch):
    """Prepare source evidence with a stubbed executable-flatbuffer reader.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test workspace.
    monkeypatch : pytest.MonkeyPatch
        Dependency patching fixture.

    Returns
    -------
    tuple
        Replay config, artifact path and manifest path.
    """
    from crest.hil import artifact_transport
    monkeypatch.setattr(artifact_transport, 'read_tflite_artifact',
                        lambda path, **kwargs: (Path(path).read_bytes(), {}))
    artifact = tmp_path / 'model.tflite'
    artifact.write_bytes(b'exact-trained-artifact')
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    config = source_config()
    config['nas']['hil_first'] = False
    config['training'] = {'train': True}
    row = base_row(study_name='study', trial_number=114, hparam__resolution=64,
                   quantization_mode='int8_ptq', hil_first=False,
                   artifact_summary_json=json.dumps({'hil_first': False, 'tflite_sha256': digest}))
    source, _, _ = write_source_run(tmp_path, [row], config)
    manifest = tmp_path / 'artifacts.json'
    manifest.write_text(json.dumps({'candidates': [dict(source_study='study', source_trial=114,
        resolution=64, staged_tflite_path=str(artifact), tflite_sha256=digest,
        byte_length=artifact.stat().st_size)]}))
    target = write_target_config(tmp_path)
    args = replace(make_args(source_run_dir=source, target_config=target,
                             output_dir=tmp_path/'replay', dry_run=True), artifact_manifest=manifest)
    return args, artifact, manifest


def test_exact_replay_dry_run_never_completes_measurement(saved_run):
    """Exact replay dry run never completes measurement.

    Parameters
    ----------
    saved_run : object
        Test fixture or parameter for this regression.
    """
    args, _, _ = saved_run
    callback = Mock(return_value={'error_code': 1})
    assert replay.run_replay(args, metrics_callback=callback) == 0
    callback.assert_not_called()
    live = replace(args, dry_run=False, resume=True)
    assert replay.run_replay(live, metrics_callback=callback) == 0
    assert callback.call_count == 1
    assert callback.call_args.kwargs['source_tflite_path'].is_file()
    assert callback.call_args.kwargs['checkpoint_path'] is None
    assert replay.run_replay(live, metrics_callback=callback) == 0
    assert callback.call_count == 1
    args.target_config.write_text('device:\n  measured_inference_runs: 3\n')
    assert replay.run_replay(live, metrics_callback=callback) == 0
    assert callback.call_count == 2


@pytest.mark.parametrize('mutation', ['hash', 'missing', 'length', 'resolution', 'study', 'duplicate'])
def test_exact_replay_rejects_invalid_artifact_before_server(saved_run, mutation):
    """Exact replay rejects invalid artifact before server.

    Parameters
    ----------
    saved_run : object
        Test fixture or parameter for this regression.
    mutation : object
        Test fixture or parameter for this regression.
    """
    args, artifact, manifest = saved_run
    data = json.loads(manifest.read_text())
    entry = data['candidates'][0]
    if mutation == 'hash': artifact.write_bytes(b'wrong-trained-artifact')
    elif mutation == 'missing': artifact.unlink()
    elif mutation == 'length': entry['byte_length'] += 1
    elif mutation == 'resolution': entry['resolution'] = 96
    elif mutation == 'study': entry['source_study'] = 'different'
    elif mutation == 'duplicate': data['candidates'].append(entry)
    manifest.write_text(json.dumps(data))
    with pytest.raises((ValueError, FileNotFoundError)):
        replay.run_replay(args, server_factory=Mock(side_effect=AssertionError('hardware touched')))
    assert not args.output_dir.exists()


def test_exact_failure_is_not_resumed(saved_run):
    """Exact failure is not resumed.

    Parameters
    ----------
    saved_run : object
        Test fixture or parameter for this regression.
    """
    args, _, _ = saved_run
    args = replace(args, dry_run=False, resume=True)
    callback = Mock(return_value={'error_code': -1})
    assert replay.run_replay(args, metrics_callback=callback) == 1
    callback.return_value = {'error_code': 1}
    assert replay.run_replay(args, metrics_callback=callback) == 0
    assert callback.call_count == 2


def test_energy_required_missing_energy_fails(saved_run):
    """Energy required missing energy fails.

    Parameters
    ----------
    saved_run : object
        Test fixture or parameter for this regression.
    """
    args, _, _ = saved_run
    args.target_config.write_text('training:\n  energy_aware: true\n')
    assert replay.run_replay(replace(args, dry_run=False),
                             metrics_callback=Mock(return_value={'error_code': 1})) == 1


def test_infeasible_row_cannot_enter_front():
    """Infeasible row cannot enter front.
    """
    row = base_row(feasible=False)
    assert replay.pareto_indices([row], [replay.ObjectiveSpec('flops', 'minimize', 'flops')]) == []


def test_fixture_contents_bind_resume(tmp_path):
    """Fixture contents bind resume.

    Parameters
    ----------
    tmp_path : object
        Test fixture or parameter for this regression.
    """
    fixture = tmp_path/'fixture.npz'
    fixture.write_bytes(b'first')
    candidate = replay.build_replay_candidate(source_row_index=0, row=base_row(hparam__resolution=64),
        objectives=[replay.ObjectiveSpec('flops', 'minimize', 'flops')])
    target = {'device': {'esp32': {'fixture_npz_path': str(fixture)}}}
    before = replay.measurement_identity(target, candidate)
    fixture.write_bytes(b'changed')
    assert before != replay.measurement_identity(target, candidate)
