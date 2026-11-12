"""Joint FER resolution and decoded-duplicate regression checks."""
from types import SimpleNamespace
from unittest.mock import patch
import optuna
import pytest
from addict import Dict
from crest.model_families.fer_mini_xception.decode import DEFAULT_FER_MINI_XCEPTION_SEED, decoded_architecture_identity
from crest.model_families.fer_mini_xception.sampler import UniqueFERNSGAIISampler
from crest.model_families.fer_mini_xception.runtime import resolution_pipeline
from crest.interfaces import OptimizerBudgetExhausted
from test.test_fer_mini_xception import _config, _context
from crest.model_families.fer_mini_xception.family import FERMiniXceptionFamily


def space():
    values = {k: [v] for k, v in DEFAULT_FER_MINI_XCEPTION_SEED.items()}
    values.update(channel_growth=[1.0], channel_cap=[64, 96, 128], resolution=[64, 96])
    return values


def test_duplicate_caps_do_not_consume_evaluations_and_resolution_is_distinct():
    sampler = UniqueFERNSGAIISampler(choices=space(), fixed_resolution=96, population_size=2, seed=42)
    study = optuna.create_study(directions=['maximize', 'minimize'], sampler=sampler)
    params = []
    def objective(trial):
        candidate = {k: trial.suggest_categorical(k, v) for k, v in space().items()}
        params.append(candidate)
        return 0.8, 1000.
    study.optimize(objective, n_trials=2)
    assert len(params) == 2
    assert {p['resolution'] for p in params} == {64, 96}
    assert decoded_architecture_identity(params[0]) != decoded_architecture_identity(params[1])
    with pytest.raises(OptimizerBudgetExhausted):
        study.optimize(objective, n_trials=1)
    assert len(params) == 2


def test_resume_uses_persisted_params_for_dedup(tmp_path):
    storage = 'sqlite:///' + str(tmp_path / 'study.db')
    sampler = UniqueFERNSGAIISampler(choices=space(), fixed_resolution=96, population_size=2, seed=42)
    study = optuna.create_study(study_name='joint', directions=['maximize','minimize'], storage=storage, sampler=sampler)
    t = study.ask()
    first = {k: t.suggest_categorical(k,v) for k,v in space().items()}
    study.tell(t, [0.8,1000])
    study = optuna.load_study(study_name='joint', storage=storage, sampler=UniqueFERNSGAIISampler(choices=space(), fixed_resolution=96, population_size=2, seed=42))
    t = study.ask()
    second = {k: t.suggest_categorical(k,v) for k,v in space().items()}
    assert second['resolution'] != first['resolution']


def test_family_builds_both_resolutions_and_rejects_context_mismatch():
    family = FERMiniXceptionFamily()
    config = _config()
    config.search.resolution = [64,96]
    assert 'resolution' in {p.name for p in family.trial_search_space(_context(), config)}
    for size in (64,96):
        params = dict(DEFAULT_FER_MINI_XCEPTION_SEED, resolution=size)
        model = family.build_model(params, _context(size), config)
        assert model.input_shape == (None,size,size)
    with pytest.raises(ValueError, match='active dataset'):
        family.build_model(params, _context(64), config)
    assert family.default_seed_trial(_context(),config)['resolution'] == 96


def test_pipeline_cache_and_server_selection_do_not_change_base_config():
    from hil_server import HILServer
    config = Dict(model=Dict(family='fer_mini_xception',search=Dict(resolution=[64,96])),dataset=Dict(name='ferplus_rafdb_mixed',params=Dict(image_size=96)))
    owner = SimpleNamespace(config=config)
    seen=[]
    def bootstrap(cfg):
        seen.append(cfg.dataset.params.image_size)
        return object()
    with patch('crest.model_families.fer_mini_xception.runtime.bootstrap_pipeline', side_effect=bootstrap):
        a=resolution_pipeline(owner, {'resolution':64})
        resolution_pipeline(owner, {'resolution':96})
        assert resolution_pipeline(owner, {'resolution':64}) is a
    assert seen == [64,96]
    assert config.dataset.params.image_size == 96
    with pytest.raises(ValueError):
        resolution_pipeline(owner, {'resolution':128})


def test_125_proposals_are_unique_across_nsga_generations():
    from crest.model_families.fer_mini_xception.decode import FER_MINI_XCEPTION_SEARCH_CHOICES
    choices = dict(FER_MINI_XCEPTION_SEARCH_CHOICES, resolution=(64,96))
    sampler = UniqueFERNSGAIISampler(choices=choices, fixed_resolution=96, population_size=25, seed=42)
    study = optuna.create_study(directions=['maximize','minimize'], sampler=sampler)
    def objective(trial):
        p = {k:trial.suggest_categorical(k,v) for k,v in choices.items()}
        return p['stem_channels'] / 16, p['num_blocks'] * p['resolution']
    study.optimize(objective, n_trials=125)
    keys = [sampler._identity(t.params) for t in study.trials]
    assert len(set(keys)) == 125
    assert max(sampler.get_trial_generation(study,t) for t in study.trials) == 4


def test_joint_client_server_real_bootstrap_and_fixture_identity(tmp_path):
    import json
    from pathlib import Path
    import yaml
    from unittest.mock import MagicMock
    from test.test_fer_model_data import _write_dataset
    from nas_model_client import NASModelClient
    from hil_server import HILServer
    from crest.model_families.fer_mini_xception.runtime import attach_pipeline
    from crest.datasets.ferplus_rafdb_mixed import write_representative_fixture
    data = tmp_path / 'data'
    _write_dataset(data)
    raw = yaml.safe_load((Path(__file__).parents[1]/'src/config/fer_components_96.yaml').read_text())
    raw['dataset']['params'].update(data_dir=str(data),calibration_examples_per_source_class=1)
    raw['model']['search']['resolution']=[64,96]
    raw['model']['params']['avoid_duplicate_architectures']=True
    raw['device']={'name':'FREENOVE_ESP32_S3_WROOM','hil':True,'runtime_mode':'back_to_back'}
    raw['training'].update(nas_trials=2,max_total_trials=2,nas_multiobjective_population_size=2,nas_epochs=75,train=True,energy_aware=False,quantization={'mode':'int8_ptq','search':False,'choices':['int8_ptq']})
    raw['nas']={'hil_first':False,'score':{'type':'multi-objective','params':{'objectives':[{'metric':'macro_f1','direction':'maximize'},{'metric':'latency_ms','direction':'minimize'}]}},'prune':{'rules':[]},'feasibility':{'rules':[]}}
    raw['outputs']={'models_dir':str(tmp_path/'models'),'candidate_dir':str(tmp_path/'candidates'),'artifact_stem':'JOINT_TEST'}
    path=tmp_path/'config.yaml'; path.write_text(yaml.safe_dump(raw))
    client=NASModelClient(path)
    server=HILServer(path)
    prior_ids=None
    try:
        for size in (64,96,64):
            hparams=dict(DEFAULT_FER_MINI_XCEPTION_SEED,resolution=size)
            pipeline=resolution_pipeline(client,hparams)
            attach_pipeline(client,pipeline)
            server._select_candidate_resolution(hparams)
            server._validate_request_dimensions(requested_timesteps=size,requested_input_dim=size)
            assert client.dataset_bundle.val.inputs.shape[1:]==(size,size)
            assert server.dataset_bundle.calibration.inputs.shape[1:]==(size,size)
            manifest=write_representative_fixture(client.dataset_bundle,tmp_path/f'{size}.npz',tmp_path/f'{size}.json')
            if prior_ids is not None:
                assert manifest['ids']==prior_ids
            prior_ids=manifest['ids']
            metadata={'sha256':'a'*64,'byte_length':12,'inputs':[{'shape':[1,size,size],'dtype':'int8'}],'outputs':[{'shape':[1,5],'dtype':'int8'}]}
            device=MagicMock(); device.supports_tflite_artifact.return_value=True
            payload={'request_type':'candidate_artifact','protocol_version':1,'request_id':f'trial-{size}','artifact':{},'family_hparams':hparams,'runtime_metadata':{'timesteps':size,'input_dim':size,'flops':1},'quantization_mode':'int8_ptq'}
            with patch.object(server,'_artifact_device',return_value=device), patch('hil_server.validate_artifact_metadata',return_value=metadata), patch.object(server,'determine_metrics',return_value={'error_code':1}):
                assert server._handle_artifact_request(payload,b'exact-model!')['artifact_sha256']=='a'*64
                metadata['inputs'][0]['shape']=[1,48,48]
                with pytest.raises(ValueError,match='input shape'):
                    server._handle_artifact_request(payload,b'exact-model!')
            server.calibration_split=object(); server._calibration_split_resolved=True
        assert client.config.dataset.params.image_size==96
        assert len(client._fer_resolution_pipelines)==2
        assert isinstance(client._build_sampler(),UniqueFERNSGAIISampler)
    finally:
        client.close()
        server.socket.close(linger=0)
