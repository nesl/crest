#!/usr/bin/env python3
"""Extract CS3 warm-up from read-only historical SQLite; cross-check CSV.

Run from the repository with its Python environment (no training, network, or HIL):
  PYTHONPATH=src python experiments/cs3/extract_baseline.py \
    --source-dir /path/to/downloaded/baseline --output-dir /tmp/cs3-reextracted
The source directory must contain the original DB, trials.csv, and archived YAML.
"""
from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sqlite3

import yaml
from optuna.distributions import json_to_distribution

from crest.model_families.audio_dscnn import AudioDSCNNFamily
from crest.pipeline_types import ModelBuildContext, TargetSpec
from crest.search_space import build_search_space_descriptor, validate_candidate

STUDY = 'UrbanSound8K_DSCNN_STM32_AUDIO_case3_2_t1'
YAML_NAME = 'nas_config_case3_2_t1_stm32_audio.yaml'


def digest(value):
    """Match the warm-up identity's canonical ordered-content digest.

    Parameters
    ----------
    value : JSON-serializable object
        Ordered candidate collection to serialize canonically.

    Returns
    -------
    str
        Hexadecimal SHA-256 digest of the canonical JSON representation.
    """
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def extract(source: Path, output: Path):
    """Validate and write the historical CS3 warm-up and its provenance.

    Parameters
    ----------
    source : pathlib.Path
        Directory containing the original SQLite database, trial CSV export,
        and archived CS3 YAML configuration.
    output : pathlib.Path
        Destination for ``warmup_params.json`` and ``baseline_provenance.json``.
        The directory is created if necessary.

    Returns
    -------
    None
        Writes the validated artifacts and prints a JSON extraction summary.

    Raises
    ------
    ValueError
        If study identity, trial ordering, recorded parameters, CSV evidence,
        input shapes, or the active search-space descriptor fail validation.
    OSError
        If source files cannot be read or output artifacts cannot be written.
    sqlite3.Error
        If the historical database cannot be queried.
    """
    cfg = yaml.safe_load((source / YAML_NAME).read_text())
    db = sqlite3.connect((source / 'optuna.db').resolve().as_uri() + '?mode=ro', uri=True)
    names = db.execute('SELECT study_id, study_name FROM studies').fetchall()
    if len(names) != 1 or names[0][1] != STUDY:
        raise ValueError(f'Unexpected study identities: {names}')
    trials = db.execute('SELECT trial_id, number, state, datetime_start, datetime_complete FROM trials WHERE study_id=? ORDER BY number', (names[0][0],)).fetchall()
    if [t[1] for t in trials[:15]] != list(range(15)):
        raise ValueError('Missing historical trial numbers 0-14')
    with (source / 'trials.csv').open(newline='') as handle:
        csv_rows = list(csv.DictReader(handle))
    by_number = {int(row['number']): row for row in csv_rows}
    if len(by_number) != len(csv_rows):
        raise ValueError('Duplicate CSV trial numbers')
    params, records = [], []
    for tid, number, state, start, complete in trials[:15]:
        candidate = {}
        for name, value, distribution in db.execute('SELECT param_name,param_value,distribution_json FROM trial_params WHERE trial_id=? ORDER BY param_name', (tid,)):
            candidate[name] = json_to_distribution(distribution).to_external_repr(value)
        attrs = {k: json.loads(v) for k, v in db.execute('SELECT key,value_json FROM trial_user_attributes WHERE trial_id=?', (tid,))}
        row = by_number[number]
        csv_keys = {k.removeprefix('params_') for k,v in row.items() if k.startswith('params_') and v != ''}
        if set(candidate) != csv_keys or row['state'] != state:
            raise ValueError(f'CSV key/state mismatch at trial {number}')
        for key,value in candidate.items():
            raw = row['params_' + key]
            try:
                parsed = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                parsed = raw
            if parsed != value or (isinstance(value, bool) and type(parsed) is not bool):
                raise ValueError(f'CSV mismatch at trial {number}: {key}')
            if attrs['hyperparameters'][key] != value:
                raise ValueError(f'Hyperparameter attribute mismatch at trial {number}: {key}')
        params.append(candidate)
        records.append(dict(number=number, state=state, datetime_start=start, datetime_complete=complete,
                            raw_parameter_count=len(candidate), error_code=attrs.get('error_code'),
                            error_label=attrs.get('error_code_label'), prune_reason=attrs.get('prune_reason'),
                            input_shape=[attrs['hyperparameters']['timesteps'], attrs['hyperparameters']['input_dim']]))
    if any(r['state'] not in ('COMPLETE', 'PRUNED') for r in records):
        raise ValueError('Startup segment includes a state not counted by TPE')
    if any(a['datetime_complete'] > b['datetime_start'] for a,b in zip(records, records[1:])):
        raise ValueError('Startup trials overlap')
    shapes = {tuple(r['input_shape']) for r in records}
    if len(shapes) != 1:
        raise ValueError('Warm-up input shapes differ')
    ctx = ModelBuildContext(input_shape=tuple(records[0]['input_shape']), input_dtype='float32',
                           target_spec=TargetSpec(task_type='classification', output_names=['class_logits'],
                                                  output_shapes=[(10,)], metadata={'num_classes': 10, 'label_encoding': 'class_index', 'from_logits': True}))
    descriptor = build_search_space_descriptor(AudioDSCNNFamily(), ctx, cfg['model'], cfg, collect_compile_metrics=True)
    for candidate in params:
        validate_candidate(candidate, descriptor)
    output.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(params, indent=2, allow_nan=False) + '\n'
    (output / 'warmup_params.json').write_text(payload)
    manifest = dict(study_name=STUDY, source_inspection_date='2026-10-13', authoritative_source='Optuna SQLite trial_params decoded using recorded distribution_json',
                    remote_host='m202-GPU', remote_directory='/home/joseph/Projects/tinyodom-ex/models/' + STUDY,
                    extracted_trial_numbers=list(range(15)), warmup_params_hash=digest(params),
                    warmup_file_sha256=hashlib.sha256(payload.encode()).hexdigest(),
                    inputs={name: dict(bytes=(source/name).stat().st_size, sha256=hashlib.sha256((source/name).read_bytes()).hexdigest()) for name in ['optuna.db', 'trials.csv', YAML_NAME]},
                    csv_crosscheck='All 15 trial numbers, states, and every nonempty params_* field match decoded DB parameters; hyperparameters attributes also match.',
                    descriptor_validation=dict(all_candidates_valid=True, library_version_for_decoding_and_startup_check=__import__('optuna').__version__, collect_compile_metrics=True, input_shape=list(ctx.input_shape), parameters=[asdict(p) for p in descriptor.params]),
                    historical_states=records, state_counts={s:sum(t[2]==s for t in trials) for s in sorted({t[2] for t in trials})},
                    startup_boundary=dict(configured_n_startup_trials=15, first_post_startup_trial_number=15,
                        preceding_complete_or_pruned_count=sum(t[2] in ('COMPLETE','PRUNED') for t in trials[:15]),
                        evidence='The pre-run committed sampler source and current remote source both set n_startup_trials=15; trials 0-14 are 14 COMPLETE plus one PRUNED, sequential without overlap. Optuna TPE counts COMPLETE and PRUNED trials toward startup.',
                        certainty='Source-supported boundary inference. No deployed run commit, sampler state, Optuna version, or per-trial sampling-method trace was retained; exact historical execution cannot be independently certified.',
                        historical_source=dict(commit='31d27ba7235846a23313f9765fbf841a2e918efd', path='src/nas_model_client.py', sha256='62854b213fcc8b2d8d504c7df69d9226b9c65bb7d35343f8534a618751ad7d7a', lines=[1711,1712], selection='Last commit changing this path before 2026-05-17 18:50:54; not a proven run revision.'),
                        observed_remote_source=dict(head='20d578d2f86e1821c4216a18b620cf7ff49d73d5', path='src/nas_model_client.py', sha256='ea4fce91dd3e9d7e3788132ee680382a18ea71b66bfd87fc0627813987fd2914', n_startup_trials_line=796)),
                    limitations=['No missing parameters were filled or inferred.', 'Historical scores are not injected into the new campaign.', 'Trial 6 retains its recorded PRUNED HIL_MASTER_FATAL attempt and its complete raw parameters.'])
    (output/'baseline_provenance.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(trials=len(params), raw_parameters_per_trial=len(params[0]), warmup_params_hash=digest(params), output=str(output))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    extract(args.source_dir, args.output_dir)
