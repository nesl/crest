# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Prepare and orchestrate policy replay using explicit prepared firmware artifacts.

Prepared projects must come from a controlled policy-specific native build.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import random
import re

PROTOCOLS = {p: f'rtc_wall_v4_dwt_active_stop_evidence_{p}_tail' for p in ('awake', 'stop')}


def sha256(path):
    """Return a file digest.

    Parameters
    ----------
    path : path-like
        Existing artifact.

    Returns
    -------
    str
        Hexadecimal SHA-256 digest.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    """Atomically replace a JSON record.

    Parameters
    ----------
    path : path-like
        Destination record.
    value : object
        JSON-serializable content.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def validate_variant(v, base):
    """Verify prepared image identity and supported protocol, without hardware.

    Parameters
    ----------
    v : dict
        Prepared project metadata, phase, policy and device options.
    base : pathlib.Path
        Root for relative prepared input paths.

    Raises
    ------
    ValueError
        Metadata or an artifact does not match its declared identity.
    """
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', v['name']):
        raise ValueError('variant name must be a safe path component')
    if v['policy'] not in PROTOCOLS or v['phase'] not in ('cadenced', 'back_to_back'):
        raise ValueError('unsupported policy or phase')
    if v['phase'] == 'back_to_back' and v['policy'] != 'awake':
        raise ValueError('continuous baseline must use awake protocol')
    for key in ('runs', 'clock_mhz', 'period_ms'):
        if type(v[key]) is not int or v[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if v['device_options'].get('latency_budget_ms') != v['period_ms']:
        raise ValueError('device latency budget differs from prepared cadence')
    project = base / v['project']
    header_path = project / v['phase_header']
    elf = base / v['compile_result']['app_elf_path']
    if v['compile_result'].get('success') is not True or v['compile_result'].get('overflow_kind'):
        raise ValueError('prepared compile must succeed without overflow')
    if sha256(header_path) != v['phase_header_sha256'] or sha256(elf) != v['app_elf_sha256']:
        raise ValueError('prepared header or ELF digest mismatch')
    header = header_path.read_text()
    expected = {'CREST_DUT_MEASURED_RUNS': str(v['runs']), 'CREST_DUT_CPU_CLOCK_MHZ': str(v['clock_mhz']),
                'CREST_DUT_LATENCY_BUDGET_MS': str(v['period_ms']),
                'CREST_DUT_SELECTED_PHASE': 'CREST_DUT_PHASE_' + v['phase'].upper()}
    for key, value in expected.items():
        if re.findall(r'^#define\s+' + key + r'\s+(\S+)\s*$', header, re.M) != [value]:
            raise ValueError(f'phase header mismatch: {key}')
    blob = elf.read_bytes()
    if PROTOCOLS[v['policy']].encode() not in blob or any(
            value.encode() in blob for policy, value in PROTOCOLS.items() if policy != v['policy']):
        raise ValueError('ELF must contain exactly the selected supported protocol')


def prepare(manifest, output):
    """Record explicitly supplied prepared project metadata in a new study.

    Parameters
    ----------
    manifest : path-like
        JSON file containing schema_version=1, variants and blocks.
    output : path-like
        New study directory; existing destinations are refused.

    Returns
    -------
    pathlib.Path
        Study root referencing the prepared projects.
    """
    manifest = Path(manifest).resolve()
    data = json.loads(manifest.read_text())
    if data.get('schema_version') != 1 or not data.get('variants'):
        raise ValueError('expected schema_version=1 and nonempty variants')
    names = [v['name'] for v in data['variants']]
    if len(set(names)) != len(names):
        raise ValueError('duplicate variant names')
    schedule(1, names, data['blocks'], 0)
    for variant in data['variants']:
        validate_variant(variant, manifest.parent)
    for variant in data['variants']:
        variant['project'] = str((manifest.parent / variant['project']).resolve())
        for key in ('build_dir', 'boot_build_dir', 'boot_elf_path', 'app_build_dir', 'app_elf_path', 'signed_app_bin_path'):
            if variant['compile_result'].get(key) is not None:
                variant['compile_result'][key] = str((manifest.parent / variant['compile_result'][key]).resolve())
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    data['source_manifest_sha256'] = sha256(manifest)
    save(output / 'study.json', data)
    return output


def schedule(rounds, names, blocks, seed):
    """Shuffle explicit pair blocks and their members reproducibly each round.

    Parameters
    ----------
    rounds : int
        Positive number of repeats.
    names : sequence of str
        All variant names.
    blocks : sequence of sequences of str
        Nonempty blocks covering each variant exactly once.
    seed : int
        Local random generator seed.

    Returns
    -------
    list of dict
        Stable item identities, round numbers and variant names.
    """
    flat = [name for block in blocks for name in block]
    if type(rounds) is not int or rounds < 1 or not blocks or any(not b for b in blocks) or sorted(flat) != sorted(names):
        raise ValueError('positive rounds and exact nonempty block partition required')
    rng, result = random.Random(seed), []
    for rnd in range(1, rounds + 1):
        order = [list(block) for block in blocks]
        rng.shuffle(order)
        for block in order:
            rng.shuffle(block)
            result.extend({'id': f'r{rnd:02d}_{name}', 'round': rnd, 'variant': name} for name in block)
    return result


def accept(result, variant):
    """Apply archived energy replay acceptance and policy-specific validation.

    Parameters
    ----------
    result : dict
        Native result with error_code, latency_s, power_metrics, dut_log and harness_log.
    variant : dict
        Prepared condition metadata.

    Returns
    -------
    list of str
        Rejection reasons; empty means accepted.
    """
    from . import validator_awake, validator_stop
    pm = result.get('power_metrics') or {}
    validator = validator_awake if variant['policy'] == 'awake' else validator_stop
    checked = validator.validate_measurement(pm, latency_s=result.get('latency_s'), phase=variant['phase'],
        clock_mhz=variant['clock_mhz'], runs=variant['runs'], period_ms=variant['period_ms'],
        dut_log=result.get('dut_log', []), harness_log=result.get('harness_log', []), require_harness=True)
    reasons = []
    if result.get('error_code') != 0:
        reasons.append('backend error')
    for flag in ('measurement_instrumentation_valid', 'measurement_deployment_eligible', 'appli_programmed', 'weights_programmed'):
        if pm.get(flag) is not True:
            reasons.append(f'{flag} not true')
    if not checked.get('instrumentation_valid') or not checked.get('deployment_eligible'):
        reasons.append('policy validator rejected telemetry')
    for key, expected in [('runs', variant['runs']), ('clock_hz', variant['clock_mhz'] * 1_000_000), ('phase', variant['phase'])]:
        if pm.get(key) != expected:
            reasons.append(f'{key} mismatch')
    energy = pm.get('energy_mj_per_inference')
    if isinstance(energy, bool) or not isinstance(energy, (float, int)) or not math.isfinite(energy) or energy <= 0:
        reasons.append('invalid energy')
    if variant['phase'] == 'cadenced':
        expected = variant['runs'] - (variant['policy'] == 'awake')
        if pm.get('deadline_miss_count') != 0 or pm.get('stop_entries') != expected or pm.get('slept_slots') != expected:
            reasons.append('cadenced sleep/deadline evidence mismatch')
    return reasons


@contextmanager
def study_lock(study):
    """Hold a persistent nonblocking POSIX flock for one output study.

    Parameters
    ----------
    study : pathlib.Path
        Existing study directory.

    Yields
    ------
    None
        Control while the study is exclusively owned.
    """
    import fcntl
    with (study / '.replay.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('policy study is already in use') from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run(study, *, rounds, seed, device, board_ready=False, lock_dir=None):
    """Run or resume a schedule using the native STM32 backend.

    Parameters
    ----------
    study : path-like
        Prepared study root.
    rounds, seed : int
        Immutable schedule parameters.
    device : dict
        Explicit device_id and native runtime settings (ports, etc.).
    board_ready : bool
        Explicit authorization to program and measure the device.
    lock_dir : path-like, optional
        Shared hardware lock directory; defaults to the common environment/default.
    """
    if not board_ready:
        raise ValueError('hardware requires board_ready=True')
    from analysis_scripts.experiment_support.device_lock import device_lock
    study = Path(study).resolve()
    with study_lock(study), device_lock(device['device_id'], lock_dir=lock_dir):
        data = json.loads((study / 'study.json').read_text())
        variants = {v['name']: v for v in data['variants']}
        items = schedule(rounds, list(variants), data['blocks'], seed)
        identity = {'rounds': rounds, 'seed': seed, 'items': items, 'device': device,
                    'study_sha256': sha256(study / 'study.json')}
        path = study / 'schedule.json'
        if path.exists() and json.loads(path.read_text()) != identity:
            raise ValueError('saved schedule/device/study differs; resume original configuration')
        save(path, identity)
        for item in items:
            variant = variants[item['variant']]
            validate_variant(variant, study)
            root = study / 'measurements' / item['id']
            result_path = root / 'result.json'
            if result_path.exists() and json.loads(result_path.read_text())['status'] == 'accepted':
                continue
            if root.exists():
                raise RuntimeError(f'failed or interrupted measurement requires inspection; use a new output: {root}')
            attempt = root / 'attempt-01'
            attempt.mkdir(parents=True, exist_ok=False)
            save(attempt / 'started.json', item)
            try:
                from .native import evaluate
                evidence = evaluate(variant, study, attempt, device)
                reasons = accept(evidence, variant)
                result = dict(item, **{key: variant[key] for key in ('candidate', 'clock_mhz', 'phase', 'period_ms', 'policy', 'runs')},
                              shape='continuous' if variant['phase'] == 'back_to_back' else 'w10',
                              status='rejected' if reasons else 'accepted', reasons=reasons,
                              evidence=evidence, attempt=attempt.name)
            except Exception as exc:
                result = dict(item, status='exception', reasons=[f'{type(exc).__name__}: {exc}'], attempt=attempt.name)
                save(attempt / 'result.json', result)
                save(result_path, result)
                raise
            save(attempt / 'result.json', result)
            save(result_path, result)
            if reasons:
                raise RuntimeError(f'rejected measurement: {reasons}')


def main():
    """Parse explicit preparation, preview and native-run commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--manifest', required=True)
    prep.add_argument('--output', required=True)
    for name in ('plan', 'run'):
        cmd = sub.add_parser(name)
        cmd.add_argument('--study', required=True)
        cmd.add_argument('--rounds', type=int, required=True)
        cmd.add_argument('--seed', type=int, required=True)
        if name == 'run':
            cmd.add_argument('--device-config', required=True)
            cmd.add_argument('--board-ready', action='store_true')
            cmd.add_argument('--lock-dir', help='shared device lock directory')
    args = parser.parse_args()
    if args.command == 'prepare':
        print(prepare(args.manifest, args.output))
    elif args.command == 'plan':
        data = json.loads((Path(args.study) / 'study.json').read_text())
        print(json.dumps(schedule(args.rounds, [v['name'] for v in data['variants']], data['blocks'], args.seed), indent=2))
    else:
        if not args.board_ready:
            parser.error('run requires --board-ready')
        run(args.study, rounds=args.rounds, seed=args.seed,
            device=json.loads(Path(args.device_config).read_text()),
            board_ready=True, lock_dir=args.lock_dir)


if __name__ == '__main__':
    main()
