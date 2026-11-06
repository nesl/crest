"""Native execution of the prepared policy replay experiment."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import types
import sys

from .replay import save, sha256, validate_variant
from . import validator_awake, validator_stop


def evaluate(variant, study_root, attempt_root, device):
    """Replay one prepared phase and preserve raw evidence.

    Parameters
    ----------
    variant : dict
        Prepared project metadata and native compile_result.
    study_root, attempt_root : pathlib.Path
        Frozen study and new attempt directory respectively.
    device : dict
        Explicit runtime_kwargs and device_id.
        Harness firmware must already be installed; auto-flashing is disabled.

    Returns
    -------
    dict
        Native metrics and raw DUT/harness logs for policy acceptance.

    Notes
    -----
    The caller holds the shared per-device lock. Execution is synchronous;
    temporary module validator replacement is restored even on exceptions.
    """
    validate_variant(variant, study_root)
    runtime = dict(device['runtime_kwargs'])
    runtime['harness_auto_flash'] = 'never'
    for port in ('serial_port', 'harness_serial_port'):
        if not Path(runtime[port]).exists():
            raise ValueError(f'missing runtime port: {port}')
    work = Path(variant['project'])
    fields = dict(variant['compile_result'])
    for field in ('build_dir', 'boot_build_dir', 'boot_elf_path', 'app_build_dir', 'app_elf_path', 'signed_app_bin_path'):
        if fields.get(field) is not None:
            fields[field] = Path(fields[field])
    source_root = Path(__file__).resolve().parents[2] / 'src'
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from crest.devices import CompileResult
    from crest.microcontrollers import stm32_nucleo_n657x0 as stm
    cached = CompileResult(**fields)
    backend = stm.STM32NucleoN657X0QDevice(device_options=dict(variant['device_options'],
        measurement_records_dir=str(attempt_root / 'measurement_records')))
    paths = backend._resolve_paths(work)
    storage_path = stm._manifest_path(paths)
    storage = json.loads(storage_path.read_text())
    weights = Path(storage['weights_blob_path'])
    loader = Path(storage['weights_external_loader'])
    if not weights.is_file() or not loader.is_file():
        raise ValueError('prepared weights and external loader must exist')
    if weights.stat().st_size != storage['weights_blob_size'] or sha256(weights) != storage['weights_blob_sha256']:
        raise ValueError('prepared weights disagree with native sidecar')
    if (paths.inc_dir / 'crest_dut_phase_config.h').resolve() != (work / variant['phase_header']).resolve():
        raise ValueError('phase header must be the native runtime header')
    for key in storage:
        if key.startswith('last_programmed_'):
            storage[key] = None
    save(storage_path, storage)
    backend.compile = types.MethodType(lambda self, **kwargs: cached, backend)

    def validate_header(self, *, paths, selected_phase, measured_runs):
        if selected_phase != variant['phase'] or measured_runs != variant['runs']:
            raise ValueError('requested phase/run count differs from prepared image')
        stm._validate_phase_config_header(paths.inc_dir / 'crest_dut_phase_config.h',
            cpu_clock_mhz=variant['clock_mhz'], selected_phase=selected_phase, measured_runs=measured_runs,
            latency_budget_ms=variant['period_ms'], wake_margin_us=5000, min_sleep_us=5000)
        if sha256(paths.inc_dir / 'crest_dut_phase_config.h') != variant['phase_header_sha256']:
            raise ValueError('runtime phase header changed')
    backend._write_runtime_phase_config = types.MethodType(validate_header, backend)
    original_validator = stm.validate_measurement
    validator = validator_awake if variant['policy'] == 'awake' else validator_stop
    try:
        stm.validate_measurement = validator.validate_measurement
        native = backend._evaluate_single_phase(dirpath=work, phase=variant['phase'], arena_kb=-1,
            window_size=variant['window_size'], num_channels=variant['num_channels'],
            measured_inference_runs=variant['runs'], **runtime)
    finally:
        stm.validate_measurement = original_validator
    result = json.loads(json.dumps(dataclasses.asdict(native), default=str))
    records = (result.get('power_metrics') or {}).get('measurement_record_paths') or []
    if len(records) != 1:
        raise ValueError('expected exactly one raw phase record')
    record_path = Path(records[0]).resolve()
    if not record_path.is_relative_to(attempt_root.resolve()):
        raise ValueError('raw evidence must be within the current attempt')
    raw = json.loads(record_path.read_text())
    if not raw.get('finished_utc') or raw['request']['phase'] != variant['phase']:
        raise ValueError('unfinished or wrong raw measurement phase')
    for flag in ('instrumentation_valid', 'deployment_eligible'):
        if raw['acceptance'].get(flag) is not True:
            raise ValueError('raw policy acceptance rejected measurement')
    if raw['result']['power_metrics'] != result['power_metrics']:
        # Record paths are appended after the raw record is written.
        raw_metrics = dict(raw['result']['power_metrics'])
        actual_metrics = dict(result['power_metrics'])
        raw_metrics.pop('measurement_record_paths', None)
        actual_metrics.pop('measurement_record_paths', None)
        if raw_metrics != actual_metrics:
            raise ValueError('raw metrics disagree with backend result')
    result.update(dut_log=raw.get('dut_log', []), harness_log=raw.get('harness_log', []))
    validate_variant(variant, study_root)
    return result
