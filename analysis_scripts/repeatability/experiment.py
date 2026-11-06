# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Original paired experiment compile-once STM32 recipe. Importing this module never imports CREST or opens hardware."""
from __future__ import annotations
import dataclasses
import hashlib
import inspect
import json
import math
import random
import shutil
import types
from pathlib import Path

PHASES = {'continuous': 'back_to_back', 'cadenced': 'cadenced'}
PROTOCOL = 'rtc_wall_v4_dwt_active_stop_evidence_awake_tail'

class InspectionRequired(RuntimeError):
    requires_inspection = True

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def _json(value):
    if dataclasses.is_dataclass(value):
        return _json(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'item'):
        return value.item()
    return value

def _write(path, value):
    Path(path).write_text(json.dumps(_json(value), indent=2, sort_keys=True) + '\n')

def _inside(path, root):
    path = Path(path).resolve()
    path.relative_to(Path(root).resolve())
    return path

def _analysis_destination(path):
    path = Path(path).resolve()
    return path

def _inventory(root):
    return {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob('*')) if p.is_file()}

def normalized_config(config):
    """Validate explicit candidate measurement and model settings.

    Parameters
    ----------
    config : dict
        Candidate and measurement configuration.

    Returns
    -------
    dict
        Validated configuration with normalized clock and input dimensions.
    """
    c = dict(config)
    c.setdefault('cpu_clock_mhz', c.get('clock_mhz'))
    c.setdefault('window_size', c.get('timesteps', c.get('architecture', {}).get('timesteps')))
    c.setdefault('num_channels', c.get('input_dim', c.get('architecture', {}).get('input_dim')))
    for name in ('cpu_clock_mhz', 'window_size', 'num_channels', 'period_ms', 'measured_runs'):
        if name not in c or isinstance(c[name], bool) or not isinstance(c[name], int) or c[name] <= 0:
            raise ValueError(f'{name} must be an explicit positive integer')
    if isinstance(c.get('initialization_seed'), bool) or not isinstance(c.get('initialization_seed'), int) or c['initialization_seed'] < 0:
        raise ValueError('initialization_seed must be an explicit nonnegative integer')
    return c

def sanitize_runtime_kwargs(values):
    """Validate forwarded native runtime options before deployment.

    Parameters
    ----------
    values : dict
        Serial, harness, timeout, and handshake options.

    Returns
    -------
    dict
        Sanitized kwargs and recorded normalization repairs.
    """
    string_fields = {'serial_port', 'harness_serial_port', 'harness_fqbn', 'harness_auto_flash'}
    integer_fields = {'baud_rate', 'harness_arm_pin', 'harness_trigger_pin', 'dut_arm_hold_ms', 'harness_stable_low_ms'}
    timeout_fields = {'serial_timeout_s', 'dut_ready_timeout_s', 'harness_ready_timeout_s',
                      'harness_arm_timeout_s', 'harness_active_timeout_s', 'harness_done_timeout_s'}
    if not isinstance(values, dict):
        raise ValueError('Runtime options must be a mapping')
    allowed = string_fields | integer_fields | timeout_fields
    kwargs, repairs = {}, {}
    for key, value in values.items():
        if key not in allowed:
            raise ValueError(f'Unknown runtime option: {key}')
        if value is None or value == {}:
            repairs[key] = 'missing optional value omitted'
            continue
        if key in string_fields:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'{key} must be a nonempty string')
        elif key in integer_fields:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0 or (key == 'baud_rate' and value == 0):
                raise ValueError(f'{key} must be an integer in its valid range')
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be a positive finite timeout')
        kwargs[key] = value
    if 'baud_rate' not in kwargs:
        kwargs['baud_rate'] = 115200
        repairs['baud_rate'] = 'explicit native default 115200'
    return {'kwargs': kwargs, 'repairs': repairs}

def prepare_with_server(server, config, destination):
    """Export and compile through the configured HIL model/calibration plumbing.

    Parameters
    ----------
    server : HILServer
        Configured server used only for candidate preparation.
    config : dict
        Explicit candidate architecture and measurement settings.
    destination : str or pathlib.Path
        New bundle directory.

    Returns
    -------
    pathlib.Path
        Sealed prepared manifest.
    """
    from crest.microcontrollers import get_device, resolve_device_options
    config = normalized_config(config)
    options = resolve_device_options(server._normalized_device_name(), server.config.device)
    options.update(cpu_clock_mhz=int(config['cpu_clock_mhz']), latency_budget_ms=config['period_ms'],
                   runtime_mode='cadenced', wake_margin_us=5000, min_sleep_us=5000)
    backend = get_device(server._normalized_device_name(), device_options=options)
    def factory(seed):
        model = server._materialize_candidate_model(config['architecture'],
                       model_variant='approx_trained', checkpoint_path=None)
        calibration = server._ensure_calibration_split()
        return server._build_candidate_prepare_request(model=model, model_variant='approx_trained',
                   checkpoint_path=None, calibration_split=calibration, quantization_mode='int8_ptq')
    runtime_names = ('serial_port', 'baud_rate', 'serial_timeout_s', 'dut_ready_timeout_s',
                     'harness_serial_port', 'harness_fqbn', 'harness_ready_timeout_s',
                     'harness_arm_timeout_s', 'harness_active_timeout_s', 'harness_done_timeout_s',
                     'harness_arm_pin', 'harness_trigger_pin', 'dut_arm_hold_ms', 'harness_stable_low_ms')
    config['runtime_kwargs'] = {k: getattr(server.config.device, k) for k in runtime_names
                                if getattr(server.config.device, k, None) is not None}
    config['runtime_kwargs'] = sanitize_runtime_kwargs(config['runtime_kwargs'])['kwargs']
    return prepare_bundle(config=config, destination=destination, backend=backend,
                          request_factory=factory, provenance={'config_repr': str(server.config)})

def prepare_bundle(*, config, destination, backend, request_factory, provenance):
    """Seed model construction, export once, and compile each native phase.

    Parameters
    ----------
    config : dict
        Explicit candidate and measurement settings.
    destination : str or pathlib.Path
        New directory containing all frozen inputs and outputs.
    backend : Device
        Native export/compiler backend; no evaluation is performed.
    request_factory : callable
        Factory accepting a seed and returning a fresh initialized candidate request.
    provenance : dict
        Caller-provided build context recorded with the bundle.

    Returns
    -------
    pathlib.Path
        Prepared manifest with a complete hash inventory.
    """
    import numpy as np
    import tensorflow as tf
    from dataclasses import replace
    destination = _analysis_destination(destination)
    destination.mkdir(parents=True, exist_ok=False)
    config = normalized_config(config)
    seed = int(config['initialization_seed'])
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    request = request_factory(seed)
    if request.quantization_mode != 'int8_ptq' or request.calibration_split is None:
        raise ValueError('int8_ptq and representative calibration inputs required')
    calibration = np.asarray(request.calibration_split.inputs)
    if not calibration.size or not np.isfinite(calibration).all():
        raise ValueError('Calibration inputs must be nonempty and finite')
    np.save(destination / 'calibration_inputs.npy', calibration, allow_pickle=False)
    request.model.save(destination / 'initialized_model.keras')
    np.savez(destination / 'initialized_weights.npz', *request.model.get_weights())
    request = replace(request, artifact_root=destination / 'export')
    template = Path(backend.resolved_options.project_root)
    protocol_sources = list(template.rglob('crest_dut_runner.c'))
    if not protocol_sources or not any(f'\"{PROTOCOL}\"' in p.read_text() for p in protocol_sources):
        raise ValueError(f'Template protocol does not match required {PROTOCOL}; no relabeling allowed')
    staged = Path(backend.prepare_candidate(request=request)).resolve()
    paths = backend._resolve_paths(staged)
    # Snapshot model + generated weights once; every phase starts from identical export.
    export_root = staged
    phases = {}
    for label, phase in PHASES.items():
        phase_root = destination / 'phases' / label
        shutil.copytree(export_root, phase_root)
        # Sidecars contain absolute generated-output/weight paths; they remain inside this bundle.
        phase_paths = backend._resolve_paths(phase_root)
        from crest.microcontrollers import stm32_nucleo_n657x0 as stm
        sidecar = stm._manifest_path(phase_paths)
        initial_storage = json.loads(sidecar.read_text())
        initial_storage['staged_workspace_root'] = str(phase_root.resolve())
        _write(sidecar, initial_storage)
        backend._write_runtime_phase_config(paths=phase_paths, selected_phase=phase, measured_runs=config['measured_runs'])
        result = backend.compile(sketch_path=phase_root, arena_kb=-1,
                                 window_size=int(config['window_size']), num_channels=int(config['num_channels']))
        if not result.success or result.overflow_kind:
            raise RuntimeError(f'{label} compilation failed: {result.log}')
        phases[label] = {'project': str(phase_root.relative_to(destination)), 'compile_result': _json(result)}
    source = Path(inspect.getfile(type(backend))).resolve()
    source_root = source.parents[2]
    code_hashes = {str(p): sha256(p) for p in source_root.rglob('*.py')}
    tools = {}
    for name in ('stedgeai', 'arm-none-eabi-gcc', 'make', 'STM32_Programmer_CLI'):
        executable = shutil.which(name)
        if executable:
            tools[str(Path(executable).resolve())] = sha256(executable)
    for key, value in dataclasses.asdict(backend.resolved_options).items():
        if isinstance(value, Path) and value.is_file():
            tools[str(value)] = sha256(value)
    descriptor = {'schema_version': 1, 'config': config, 'seed': seed,
                  'training_performed': False, 'quantization_mode': 'int8_ptq',
                  'protocol': PROTOCOL, 'measured_runs': config['measured_runs'], 'period_ms': config['period_ms'],
                  'calibration_shape': list(calibration.shape), 'calibration_dtype': str(calibration.dtype),
                  'provenance': provenance, 'backend_source': str(source),
                  'backend_source_sha256': sha256(source), 'code_hashes': code_hashes, 'tool_hashes': tools, 'phases': phases,
                  'device_options': _json({k: v for k, v in dataclasses.asdict(backend.resolved_options).items() if k not in ('runtime_project_name',)}),
                  'runtime_kwargs': config.get('runtime_kwargs', {})}
    _write(destination / 'bundle.json', descriptor)
    # Absolute backend paths make this bundle intentionally host/path bound.
    manifest = {'schema_version': 1, 'root': str(destination), 'files': _inventory(destination)}
    _write(destination / 'prepared_manifest.json', manifest)
    verify_bundle(destination / 'prepared_manifest.json')
    return destination / 'prepared_manifest.json'

def verify_bundle(manifest_path):
    """Validate frozen files, code, tool hashes, and referenced artifacts.

    Parameters
    ----------
    manifest_path : str or pathlib.Path
        Prepared manifest at its original host path.

    Returns
    -------
    tuple[pathlib.Path, dict]
        Verified bundle root and descriptor.
    """
    manifest_path = Path(manifest_path).resolve()
    m = json.loads(manifest_path.read_text())
    root = Path(m['root']).resolve()
    if manifest_path.parent != root:
        raise ValueError('Bundle was moved; absolute backend paths require explicit re-preparation')
    for rel, digest in m['files'].items():
        p = _inside(root / rel, root)
        if not p.is_file() or sha256(p) != digest:
            raise ValueError(f'Frozen artifact mismatch: {rel}')
    bundle = json.loads((root / 'bundle.json').read_text())
    for filename, digest in {**bundle.get('code_hashes', {}), **bundle.get('tool_hashes', {})}.items():
        if sha256(filename) != digest:
            raise ValueError(f'Preparation code/tool changed: {filename}')
    for rel in m['files']:
        candidate = root / rel
        if candidate.suffix == '.json' and 'manifest' in candidate.name:
            try:
                sidecar = json.loads(candidate.read_text())
            except (ValueError, UnicodeDecodeError):
                continue
            for key in ('weights_blob_path', 'appli_signed_image_path', 'boot_elf_path'):
                value = sidecar.get(key) if isinstance(sidecar, dict) else None
                if value:
                    target = _inside(value, root)
                    if str(target.relative_to(root)) not in m['files']:
                        raise ValueError(f'Unsealed sidecar artifact: {value}')
    for entry in bundle['phases'].values():
        for key in ('boot_elf_path', 'app_elf_path', 'signed_app_bin_path'):
            value = entry['compile_result'].get(key)
            if value:
                artifact = _inside(value, root)
                if str(artifact.relative_to(root)) not in m['files']:
                    raise ValueError('Unsealed compiled artifact')
    if sha256(bundle['backend_source']) != bundle['backend_source_sha256']:
        raise ValueError('Backend source changed since preparation')
    return root, bundle

def effective_weights_mapping(manifest_path):
    """Resolve the unique sealed external-weight export sidecar.

    Parameters
    ----------
    manifest_path : str or pathlib.Path
        Prepared bundle manifest.

    Returns
    -------
    dict
        Verified runtime weight metadata and supporting hashes.
    """
    root, bundle = verify_bundle(manifest_path)
    inventory = json.loads(Path(manifest_path).read_text())['files']
    candidates = []
    for rel in inventory:
        if rel.startswith('export/') and Path(rel).name == 'crest_stm32_manifest.json':
            data = json.loads((root / rel).read_text())
            if data.get('weights_blob_path'):
                candidates.append((rel, data))
    if len(candidates) != 1:
        raise ValueError('Exactly one sealed export weight sidecar required')
    rel, storage = candidates[0]
    if storage.get('weight_storage_mode') != 'external_flash':
        raise ValueError('External flash weight storage required')
    blob = _inside(storage['weights_blob_path'], root)
    blob_rel = str(blob.relative_to(root))
    digest = inventory.get(blob_rel)
    if digest is None or sha256(blob) != digest or storage.get('weights_blob_size') != blob.stat().st_size:
        raise ValueError('Exported weight size/hash differs from sealed inventory')
    if storage.get('weights_blob_sha256') != digest:
        raise ValueError('Export sidecar weight hash differs from sealed inventory')
    if str(storage.get('weights_flash_address')).lower() != '0x71000000':
        raise ValueError('Unexpected external weights flash address')
    fields = {k: storage.get(k) for k in ('weights_blob_path', 'weights_blob_size', 'weights_blob_sha256',
              'weights_flash_address', 'weights_external_loader', 'weight_storage_mode')}
    return {'export_sidecar': rel, 'export_sidecar_sha256': inventory[rel],
            'weights_sha256': digest, 'fields': fields}

def runtime_storage_manifest(storage, mapping, work):
    """Create disposable storage metadata without changing frozen inputs.

    Parameters
    ----------
    storage : dict
        Phase storage sidecar metadata.
    mapping : dict
        Verified export weight mapping.
    work : str or pathlib.Path
        Disposable runtime workspace.

    Returns
    -------
    dict
        Runtime sidecar with verified weights and cleared programming cache.
    """
    storage = dict(storage)
    storage.update(mapping['fields'])
    storage['staged_workspace_root'] = str(Path(work).resolve())
    for key in list(storage):
        if key.startswith('last_programmed_'):
            storage[key] = None
    return storage

class Experiment:
    def __init__(self, manifest, study_dir):
        self.manifest, self.study_dir = manifest, Path(study_dir).resolve()

    def _path(self, config):
        return self.study_dir / config.get('prepared_manifest', f"artifacts/{config['candidate_id']}/prepared_manifest.json")

    def preflight(self, config):
        """Verify a candidate bundle and runtime options without opening hardware.

        Parameters
        ----------
        config : dict
            Explicit candidate configuration matching the frozen bundle.

        Returns
        -------
        dict
            Verified bundle identity and runtime normalization evidence.
        """
        config = normalized_config(config)
        path = self._path(config)
        root, b = verify_bundle(path)
        if b['period_ms'] != config['period_ms'] or b['measured_runs'] != config['measured_runs'] or b['quantization_mode'] != 'int8_ptq':
            raise ValueError('Frozen protocol differs from planned repeatability protocol')
        for key in ('candidate_id', 'architecture', 'window_size', 'num_channels', 'initialization_seed'):
            if b['config'].get(key) != config.get(key):
                raise ValueError(f'Frozen candidate identity mismatch: {key}')
        if self.manifest.get('prospective_protocol') != PROTOCOL:
            raise ValueError('Study manifest lacks explicit prospective protocol')
        if b['protocol'] != PROTOCOL:
            raise ValueError('Frozen firmware protocol mismatch')
        if int(b['config']['cpu_clock_mhz']) != int(config['cpu_clock_mhz']):
            raise ValueError('Candidate source clock changed')
        runtime = sanitize_runtime_kwargs(b['runtime_kwargs'])
        mapping = effective_weights_mapping(path)
        return {'runtime_normalization': runtime, 'ok': True, 'prepared_manifest': str(path), 'sha256': sha256(path), 'effective_weights_mapping': mapping}

    def evaluate(self, config, context):
        """Measure both phases using disposable workspaces and frozen artifacts.

        Parameters
        ----------
        config : dict
            Validated candidate configuration including explicit serial paths.
        context : dict
            Study identity and unique attempt directory.

        Returns
        -------
        dict
            Native phase results and protocol acceptance assessment.
        """
        config = normalized_config(config)
        from crest.devices import CompileResult
        from crest.microcontrollers.stm32_nucleo_n657x0 import STM32NucleoN657X0QDevice
        self.preflight(config)
        path = self._path(config)
        root, bundle = verify_bundle(path)
        attempt = _analysis_destination(context['attempt_dir'])
        attempt.mkdir(parents=True, exist_ok=True)
        runtime = sanitize_runtime_kwargs(bundle['runtime_kwargs'])
        deployment = dict(runtime['kwargs'], serial_port=config['dut_port'], harness_serial_port=config['harness_port'],
                          harness_auto_flash='never')
        _write(attempt / 'deployment_override.json', {'kwargs': deployment, 'repairs': runtime['repairs']})
        if not Path(config['dut_port']).exists() or not Path(config['harness_port']).exists():
            raise InspectionRequired('Expected stable STM32 DUT/harness IDs absent; reconnect documented boards')
        outputs = {}
        for label, phase in PHASES.items():
            entry = bundle['phases'][label]
            # Backend modifies programming-cache sidecars. Only disposable copy is writable.
            work = attempt / label
            shutil.copytree(root / entry['project'], work)
            options = dict(bundle['device_options'])
            options['measurement_records_dir'] = str(work / 'measurement_records')
            backend = STM32NucleoN657X0QDevice(device_options=options)
            original = root / entry['project']
            def relocate(value):
                if isinstance(value, str):
                    return value.replace(str(original), str(work))
                if isinstance(value, dict):
                    return {k: relocate(v) for k, v in value.items()}
                return value
            # Update path references and reset cache so BOTH app and weights program each phase.
            paths = backend._resolve_paths(work)
            from crest.microcontrollers import stm32_nucleo_n657x0 as stm
            sidecar = stm._manifest_path(paths)
            storage = relocate(json.loads(sidecar.read_text()))
            mapping = effective_weights_mapping(path)
            storage = runtime_storage_manifest(storage, mapping, work)
            _write(attempt / f'{label}_weights_mapping.json', mapping)
            _write(sidecar, storage)
            fields = relocate(entry['compile_result'])
            for key in ('build_dir', 'boot_build_dir', 'boot_elf_path', 'app_build_dir', 'app_elf_path', 'signed_app_bin_path'):
                if fields.get(key) is not None:
                    fields[key] = Path(fields[key])
            cached = CompileResult(**fields)
            backend.compile = types.MethodType(lambda self, **kw: cached, backend)
            def validate_header(self, *, paths, selected_phase, measured_runs):
                if selected_phase != phase or measured_runs != config['measured_runs']:
                    raise ValueError('Runtime requested different phase or run count')
                # Preflight hashes already validated the compiled header. No rewrite allowed.
            backend._write_runtime_phase_config = types.MethodType(validate_header, backend)
            kwargs = dict(deployment)
            result = backend._evaluate_single_phase(dirpath=work, phase=phase, arena_kb=-1,
                       window_size=int(config['window_size']), num_channels=int(config['num_channels']),
                       measured_inference_runs=config['measured_runs'], **kwargs)
            outputs[label] = _json(result)
            _write(attempt / f'{label}_metrics.json', outputs[label])
            verify_bundle(path)
            if outputs[label].get('error_code') != 0:
                raise InspectionRequired(f'{label} native backend failed; raw evidence preserved, inspect before retry')
        result = {'phases': outputs, 'artifact_manifest_sha256': sha256(path), 'deployment_override': deployment}
        result.update(validate_evaluation(result, config))
        return result

    def validate_result(self, result, config):
        """Validate native paired results against the configured protocol.

        Parameters
        ----------
        result : dict
            Native paired backend output.
        config : dict
            Explicit measurement settings and serial identities.

        Returns
        -------
        dict
            Acceptance assessment and rejection reasons.
        """
        return validate_evaluation(result, config)

def validate_evaluation(result, config, *, expected_protocol=PROTOCOL):
    """Validate paired metrics against each completed native acceptance record.

    Parameters
    ----------
    result : dict
        Backend result containing both continuous and cadenced phases.
    config : dict
        Explicit candidate, dimensions, clock, cadence, run count, and ports.
    expected_protocol : str, optional
        Native firmware protocol marker required in both raw records.

    Returns
    -------
    dict
        Acceptance flag, rejection reasons, phase energies, and timing evidence.
    """
    reasons, energies, timing, record_paths = [], {}, {}, []
    clock = float(config.get('clock_mhz', config.get('cpu_clock_mhz', -1))) * 1000000
    def positive(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0
    for label, expected_phase in PHASES.items():
        phase = result.get('phases', {}).get(label, {})
        pm = phase.get('power_metrics') or {}
        if phase.get('error_code') != 0:
            reasons.append(f'{label}: backend error')
        for flag in ('measurement_instrumentation_valid', 'measurement_deployment_eligible', 'appli_programmed', 'weights_programmed'):
            if pm.get(flag) is not True:
                reasons.append(f'{label}: {flag} missing/false')
        energy = pm.get('energy_mj_per_inference')
        if not positive(energy):
            reasons.append(f'{label}: invalid energy'); energy = None
        energies[label] = energy
        timing[label] = {'latency_s': phase.get('latency_s'), **{k: v for k, v in pm.items()
                         if any(s in k for s in ('window', 'clock', 'deadline', 'run', 'protocol', 'rtc', 'sleep'))}}
        if pm.get('clock_hz') != clock:
            reasons.append(f'{label}: clock mismatch')
        if pm.get('runs') != config['measured_runs'] or pm.get('phase') != expected_phase:
            reasons.append(f'{label}: phase/run count mismatch')
        if not positive(pm.get('timer_per_inference_s')) or not positive(pm.get('timer_per_window_s')):
            reasons.append(f'{label}: invalid timing')
        if label == 'cadenced':
            if pm.get('deadline_miss_count') != 0 or pm.get('stop_entries') != config['measured_runs'] - 1:
                reasons.append('cadenced: deadline/Stop count mismatch')
            if not math.isclose(pm.get('timer_per_window_s', -1), config['measured_runs'] * config['period_ms'] / 1000) or not positive(pm.get('timer_per_inference_s')) or pm['timer_per_inference_s'] > config['period_ms'] / 1000:
                reasons.append('cadenced: cadence window mismatch')
        paths = pm.get('measurement_record_paths')
        if not isinstance(paths, list) or len(paths) != 1:
            reasons.append(f'{label}: expected one raw phase record'); continue
        try:
            path = Path(paths[0]).resolve(); raw = json.loads(path.read_text()); record_paths.append(str(path))
            acceptance = raw['acceptance']; raw_pm = raw['result']['power_metrics']
            if not raw.get('finished_utc') or raw['request']['phase'] != expected_phase:
                reasons.append(f'{label}: unfinished/wrong raw phase')
            if raw['request'].get('measured_inference_runs') != config['measured_runs'] or raw['request'].get('window_size') != config['window_size'] or raw['request'].get('num_channels') != config['num_channels']:
                reasons.append(f'{label}: raw request mismatch')
            if acceptance.get('instrumentation_valid') is not True or acceptance.get('deployment_eligible') is not True or not acceptance.get('checks') or not all(v is True for v in acceptance['checks'].values()):
                reasons.append(f'{label}: raw acceptance rejected')
            if expected_protocol == PROTOCOL and (raw['request'].get('serial_port') != config['dut_port'] or raw['request'].get('harness_serial_port') != config['harness_port']):
                reasons.append(f'{label}: stable DUT/harness identity mismatch')
            if expected_protocol not in str(raw.get('dut_log', '')):
                reasons.append(f'{label}: protocol missing/mismatch')
            if acceptance.get('energy_mj_per_release') != energy:
                reasons.append(f'{label}: raw accepted energy mismatch')
            for key in ('energy_mj_per_inference', 'clock_hz', 'runs', 'phase', 'timer_per_window_s', 'timer_per_inference_s', 'appli_programmed', 'weights_programmed'):
                if raw_pm.get(key) != pm.get(key): reasons.append(f'{label}: raw metric mismatch: {key}')
            if label == 'cadenced':
                for key in ('deadline_miss_count', 'stop_entries'):
                    if raw_pm.get(key) != pm.get(key): reasons.append(f'{label}: raw metric mismatch: {key}')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            reasons.append(f'{label}: unreadable/incomplete raw record: {exc}')
    if len(record_paths) == 2 and len(set(record_paths)) != 2:
        reasons.append('continuous/cadenced raw records must be distinct')
    return {'accepted': not reasons, 'reasons': reasons,
            'continuous_energy_mj_per_inference': energies['continuous'],
            'cadenced_energy_mj_per_release': energies['cadenced'], 'timing': timing}
