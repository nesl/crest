"""Synthetic native extension on temporary backups of historical campaign evidence.

Original databases are opened read-only solely for SQLite backup. No original
campaign is extended and no hardware, training or provider implementation runs.
"""
from pathlib import Path
import hashlib
import json
import sqlite3
import tempfile
from unittest.mock import patch
import sys
import yaml
import optuna
from addict import Dict

ROOT = Path('/Users/jzales/.codex/worktrees/crest-proposal-contract/fall-2026-nesl-crest')
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT))
from test.test_nas_model_client import _build_test_client
from nas_model_client import OPTIMIZER_SIGNATURE_ATTR, LEGACY_OPTIMIZER_ADOPTION_ATTR

SOURCE = Path('/Users/jzales/Documents/Projects/fall-2026-nesl-crest/outputs/search_budget_2026-09-24')
CAMPAIGNS = [
    ('stm32_cs1', 'OxIOD_STM32_B2B_case1_5_t1', 'nas_config_case1_5_stm32_b2b_oxiod.yaml'),
    ('portenta_m7', 'OxIOD_PORTENTA_M7_B2B_case1_3_t1', 'nas_config_case1_3_portenta_m7_b2b_oxiod.yaml'),
]

def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def evidence_hash(trials):
    return hashlib.sha256(json.dumps([
        {'number': t.number, 'state': t.state.name, 'params': t.params,
         'values': t.values, 'user_attrs': t.user_attrs, 'system_attrs': t.system_attrs,
         'intermediate_values': t.intermediate_values,
         'distributions': {k: optuna.distributions.distribution_to_json(v) for k,v in t.distributions.items()},
         'datetime_start': t.datetime_start, 'datetime_complete': t.datetime_complete}
        for t in trials], sort_keys=True, default=str, allow_nan=True).encode()).hexdigest()

optuna.logging.set_verbosity(optuna.logging.ERROR)
for folder, name, yaml_name in CAMPAIGNS:
    source = SOURCE / folder / 'source' / 'optuna.db'
    config_path = source.parent / yaml_name
    original_hash = file_hash(source)
    with tempfile.TemporaryDirectory(prefix=f'crest-legacy-{folder}-') as temporary:
        temporary = Path(temporary)
        copy = temporary / 'copy.db'
        # URI mode=ro ensures the source cannot be changed by this connection.
        with sqlite3.connect(source.as_uri()+'?mode=ro', uri=True) as original:
            with sqlite3.connect(copy) as destination:
                original.backup(destination)
        storage = f'sqlite:///{copy}'
        study = optuna.load_study(study_name=name, storage=storage)
        old_count = len(study.trials)
        complete = sum(t.state.name == 'COMPLETE' for t in study.trials)
        assert OPTIMIZER_SIGNATURE_ATTR not in study.user_attrs
        old_evidence = evidence_hash(study.trials)
        saved = Dict(yaml.safe_load(config_path.read_text()))
        client = _build_test_client(temporary)
        # Restore original objective/directions and sampler settings only. Fake
        # objective below supplies simulated values, not campaign measurements.
        client.config.nas = saved.nas
        client.config.training = saved.training
        client.config.optimizer = Dict(type='optuna')
        client.config.training.nas_trials = complete + 1
        client.config.training.max_total_trials = old_count + 1
        calls = []
        latest = next(t for t in reversed(study.trials) if t.state.name == 'COMPLETE')
        distributions = latest.distributions
        def fake_objective(trial):
            calls.append(trial.number)
            for key, dist in distributions.items():
                if isinstance(dist, optuna.distributions.IntDistribution):
                    trial.suggest_int(key, dist.low, dist.high, step=dist.step, log=dist.log)
                elif isinstance(dist, optuna.distributions.FloatDistribution):
                    trial.suggest_float(key, dist.low, dist.high, step=dist.step, log=dist.log)
                else:
                    trial.suggest_categorical(key, dist.choices)
            trial.set_user_attr('synthetic_legacy_copy_probe', True)
            return (1.0, 1.0)
        client.objective = fake_objective
        try:
            client.run_nas(name, storage=storage)
        except RuntimeError as error:
            assert 'no CREST optimizer signature' in str(error), str(error)
        else:
            raise AssertionError('Unsigned historical study resumed without explicit opt-in')
        assert not calls
        assert evidence_hash(optuna.load_study(study_name=name, storage=storage).trials) == old_evidence
        client.config.optimizer.adopt_legacy_study = True
        with patch('nas_model_client.build_search_space_descriptor', side_effect=AssertionError('native descriptor forbidden')), \
             patch('nas_model_client.build_semantic_context', side_effect=AssertionError('native semantics forbidden')):
            result = client.run_nas(name, storage=storage)
        assert calls == [old_count], calls
        assert len(result.trials) == old_count + 1
        assert evidence_hash(result.trials[:old_count]) == old_evidence
        assert result.user_attrs[LEGACY_OPTIMIZER_ADOPTION_ATTR]['trial_count'] == old_count
        assert result.user_attrs[OPTIMIZER_SIGNATURE_ATTR]['sampler']['class'] == 'optuna.samplers.NSGAIISampler'
        signature = result.user_attrs[OPTIMIZER_SIGNATURE_ATTR]
        first_extension_evidence = evidence_hash(result.trials)
        # Signed campaigns then extend with opt-in disabled and a budget-only change.
        client.config.optimizer.adopt_legacy_study = False
        client.config.training.nas_trials += 1
        client.config.training.max_total_trials += 1
        result = client.run_nas(name, storage=storage)
        assert calls == [old_count, old_count+1]
        assert result.user_attrs[OPTIMIZER_SIGNATURE_ATTR] == signature
        assert evidence_hash(result.trials[:old_count+1]) == first_extension_evidence
        assert evidence_hash(result.trials[:old_count]) == old_evidence
        assert file_hash(source) == original_hash
        print(json.dumps({'campaign': name, 'source_file_sha256_before_after': original_hash,
            'original_trial_count': old_count, 'original_complete_count': complete,
            'original_trial_evidence_sha256_before_after': old_evidence,
            'copied_new_synthetic_trials': 2, 'native_sampler_signature': signature['sampler'],
            'unsigned_rejected_by_default': True, 'signed_budget_extension': True,
            'old_trials_unchanged': True, 'original_db_unchanged': True}, sort_keys=True))
