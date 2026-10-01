"""Real-Optuna acceptance tests of CREST's single proposal execution path."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import optuna
import pytest
import yaml
from addict import Dict
from optuna.trial import TrialState
from crest.model import load_config
from crest.pipeline_types import BudgetSnapshot, CandidateProposal, ExplicitRound, NativeRound
from crest.search_space import SearchParam, SearchSpaceDescriptor
from crest.registry import optimizer_registry
from nas_model_client import NASModelClient, OPTIMIZER_SIGNATURE_ATTR, PROPOSAL_ATTR, _family_trial_params
from test.proposal_plugin import ThirdProposer
from test.test_nas_model_client import _build_test_client
from crest.hardware import HIL_MASTER_FATAL, HIL_MASTER_SUCCESS, HIL_MASTER_DEVICE_NOT_FOUND

NAME = "test_third_proposer"
if NAME not in optimizer_registry:
    optimizer_registry.register(NAME, ThirdProposer)

@pytest.fixture
def client(tmp_path):
    ThirdProposer.rounds = []
    ThirdProposer.calls = []
    ThirdProposer.contexts = []
    c = _build_test_client(tmp_path)
    c.config.optimizer = Dict(type=NAME, setting=1)
    c.config.device.hil = False
    c.config.device.compile_when_hil_disabled = "false"
    c.config.training.train = False
    c.config.training.nas_trials = 2
    c.config.training.max_total_trials = 2
    c.config.nas.score = Dict(type="scoring-function", metrics={}, params={"terms": [{"type":"weighted", "metric":"flops", "weight":-1.0}]})
    c.model_family.trial_search_space = MagicMock(return_value=[SearchParam("width", "int", low=2, high=8)])
    c.objective = lambda trial: float(trial.suggest_int("width", 2, 8))
    return c

def execute(c, tmp_path, name="acceptance"):
    return c.run_nas(name, storage=f"sqlite:///{tmp_path / 'study.db'}")

@pytest.mark.parametrize("round", [NativeRound(0), NativeRound(True), NativeRound(3), ExplicitRound(()), ExplicitRound((CandidateProposal({"width": 3}), CandidateProposal({}))), ExplicitRound((CandidateProposal({"width": 3}), CandidateProposal({"width": 9}))), ExplicitRound((CandidateProposal({"width": 3}), CandidateProposal({"width": 4}, {"bad": object()}))), object()])
def test_invalid_rounds_do_not_enqueue_or_evaluate(client, tmp_path, round):
    ThirdProposer.rounds = [round]
    client.objective = MagicMock()
    with pytest.raises((ValueError, TypeError)):
        execute(client, tmp_path)
    study = optuna.load_study(study_name="acceptance", storage=f"sqlite:///{tmp_path / 'study.db'}")
    assert study.trials == []
    client.objective.assert_not_called()


def test_third_proposer_values_provenance_context_and_partial_round(client, tmp_path):
    ThirdProposer.rounds = [[{"width":3}], [{"width":7}]]
    with patch("nas_model_client.build_semantic_context", wraps=__import__("crest.semantic_context", fromlist=["build_semantic_context"]).build_semantic_context) as builder:
        study = execute(client, tmp_path)
    assert builder.call_count == 1
    assert [t.params for t in study.trials] == [{"width":3}, {"width":7}]
    assert [t.value for t in study.trials] == [3, 7]
    assert [call[1].permitted_round_size for call in ThirdProposer.calls] == [2, 1]
    assert ThirdProposer.calls[1][0][0].number == 0
    context = ThirdProposer.contexts[0]
    assert context.objective_summary == json.dumps(client.config.nas.score, sort_keys=True, default=str)
    with pytest.raises(TypeError):
        context.semantic_context["new"] = 1
    for t in study.trials:
        assert t.user_attrs[PROPOSAL_ATTR]["proposal_params"] == t.params
        assert t.user_attrs[PROPOSAL_ATTR]["optimizer"] == NAME
        assert _family_trial_params(t.params) == t.params


@pytest.mark.parametrize("name", ["MixedCasePlugin", "OPTUNA", "LLM_GENERATOR"])
def test_exact_plugin_registration_before_load_config_runs_with_registered_identity(client, tmp_path, name):
    optimizer_registry.register(name, ThirdProposer)
    try:
        config_path = Path(__file__).resolve().parents[1] / "src/config/nas_config_flops_rmse.yaml"
        config = yaml.safe_load(config_path.read_text())
        config["optimizer"] = {"type": " " + name + " ", "setting": 7, "custom_payload": {"seed": 3}}
        config["outputs"]["models_dir"] = str(tmp_path / "models")
        config["outputs"]["candidate_dir"] = str(tmp_path / "candidate")
        path = tmp_path / "plugin_config.yaml"
        path.write_text(yaml.safe_dump(config))
        loaded = load_config(path)
        assert loaded.optimizer.type == name
        assert loaded.optimizer.custom_payload.seed == 3
        assert "llm" not in loaded.optimizer
        client.config.optimizer = loaded.optimizer
        ThirdProposer.rounds = [[{"width": 3}, {"width": 7}]]
        study = execute(client, tmp_path)
        assert [trial.params for trial in study.trials] == [{"width": 3}, {"width": 7}]
        assert [trial.value for trial in study.trials] == [3, 7]
        assert study.user_attrs[OPTIMIZER_SIGNATURE_ATTR]["optimizer"] == name
        assert study.user_attrs[OPTIMIZER_SIGNATURE_ATTR]["config"] == {"setting": 7}
        assert all(trial.user_attrs[PROPOSAL_ATTR]["optimizer"] == name for trial in study.trials)
    finally:
        optimizer_registry._items.pop(name)


def sign(c, study):
    c._select_optimizer(study, c._build_sampler())

@pytest.mark.parametrize("tagged", [False, True])
def test_waiting_at_cap_consumes_exact_oldest_prefix_before_proposals(client, tmp_path, tagged):
    storage=f"sqlite:///{tmp_path / 'study.db'}"
    study=optuna.create_study(study_name="acceptance", storage=storage, direction="maximize")
    sign(client, study)
    for i, width in enumerate([3, 4, 7]):
        attrs={PROPOSAL_ATTR:{"round_id":"first" if i<2 else "next", "proposal_params":{"width":width}}} if tagged else {}
        study.enqueue_trial({"width":width}, user_attrs=attrs)
    assert all(t.params == {} for t in study.trials)
    client.config.training.max_total_trials=3
    client.config.training.nas_trials=2
    sizes=[]
    original=optuna.Study.optimize
    def optimize(study, objective, n_trials):
        sizes.append(n_trials)
        return original(study, objective, n_trials=n_trials)
    with patch.object(optuna.Study, "optimize", optimize):
        result=execute(client,tmp_path)
    assert sizes == ([2] if tagged else [3])
    assert ThirdProposer.calls == []
    assert len(result.trials)==3
    assert [t.state for t in result.trials] == ([TrialState.COMPLETE]*2+[TrialState.WAITING] if tagged else [TrialState.COMPLETE]*3)


def test_partial_enqueue_restart_consumes_reservation_before_new_request(client,tmp_path):
    ThirdProposer.rounds=[[{"width":3},{"width":4}]]
    original=optuna.Study.enqueue_trial
    count=0
    def enqueue(study,params,**kwargs):
        nonlocal count
        count+=1
        if count==2:
            raise RuntimeError("synthetic interruption")
        return original(study,params,**kwargs)
    with patch.object(optuna.Study,"enqueue_trial",enqueue),pytest.raises(RuntimeError, match="interruption"):
        execute(client,tmp_path)
    ThirdProposer.calls=[]
    ThirdProposer.rounds=[[{"width":7}]]
    observed=[]
    def objective(trial):
        observed.append((trial.number,len(ThirdProposer.calls)))
        return float(trial.suggest_int("width",2,8))
    client.objective=objective
    study=execute(client,tmp_path)
    assert observed==[(0,0),(1,1)]
    assert len(study.trials)==2

@pytest.mark.parametrize("change",["setting","native"])
def test_identity_mismatch_rejects_waiting_before_initialization(client,tmp_path,change):
    storage=f"sqlite:///{tmp_path / 'study.db'}"
    study=optuna.create_study(study_name="acceptance", storage=storage,direction="maximize")
    sign(client,study)
    study.enqueue_trial({"width":3})
    if change=="setting": client.config.optimizer.setting=2
    else: client.config.optimizer=Dict(type="optuna")
    client.objective=MagicMock()
    with pytest.raises(RuntimeError,match="optimizer signature"):
        execute(client,tmp_path)
    client.objective.assert_not_called()
    assert ThirdProposer.contexts==[]
    assert optuna.load_study(study_name="acceptance",storage=storage).trials[0].state==TrialState.WAITING


def test_unsigned_legacy_fails_and_budget_extension_resumes(client,tmp_path):
    storage=f"sqlite:///{tmp_path / 'study.db'}"
    study=optuna.create_study(study_name="legacy", storage=storage,direction="maximize")
    study.enqueue_trial({"width":3})
    with pytest.raises(RuntimeError,match="no CREST optimizer signature"):
        client.run_nas("legacy",storage)
    ThirdProposer.rounds=[[{"width":3},{"width":4}]]
    result=execute(client,tmp_path)
    signature=result.user_attrs[OPTIMIZER_SIGNATURE_ATTR]
    client.config.training.nas_trials=3
    client.config.training.max_total_trials=3
    ThirdProposer.rounds=[[{"width":7}]]
    result=execute(client,tmp_path)
    assert result.user_attrs[OPTIMIZER_SIGNATURE_ATTR]==signature
    assert len(result.trials)==3


def test_target_already_satisfied_leaves_waiting_untouched(client,tmp_path):
    ThirdProposer.rounds=[[{"width":3},{"width":4}]]
    study=execute(client,tmp_path)
    study.enqueue_trial({"width":7})
    result=execute(client,tmp_path)
    assert result.trials[-1].state==TrialState.WAITING
    assert len(ThirdProposer.calls)==1


def test_native_descriptor_optional_and_running_still_consumes_cap(client,tmp_path):
    client.config.optimizer=Dict(type="optuna")
    client.model_family.trial_search_space=MagicMock(side_effect=NotImplementedError)
    with patch("nas_model_client.build_search_space_descriptor", side_effect=ValueError("irrelevant descriptor")) as descriptor, \
         patch("nas_model_client.build_semantic_context", side_effect=ValueError("irrelevant semantics")) as semantics:
        study=execute(client,tmp_path)
    descriptor.assert_not_called()
    semantics.assert_not_called()
    client.model_family.trial_search_space.assert_not_called()
    assert len(study.trials)==2
    assert all(PROPOSAL_ATTR not in t.user_attrs for t in study.trials)
    storage=f"sqlite:///{tmp_path / 'orphan.db'}"
    study=optuna.create_study(study_name="orphan",storage=storage,direction="maximize")
    sign(client,study)
    study.ask()
    client.config.training.max_total_trials=1
    result=client.run_nas("orphan",storage)
    assert len(result.trials)==1 and result.trials[0].state==TrialState.RUNNING


def hil_metrics(error=HIL_MASTER_SUCCESS):
    return {"error_code":error,"ram_bytes":512,"flash_bytes":512,"arena_bytes":1024,"latency_ms":10.0,
            "energy_mj_per_inference":1.0,"avg_power_mw":1.0,"avg_current_ma":1.0,"bus_voltage_v":3.3}


def real_objective_client(tmp_path,multi=False):
    c=_build_test_client(tmp_path)
    c.config.optimizer=Dict(type=NAME,setting=1)
    c.config.training.nas_trials=2
    c.config.training.max_total_trials=2
    c.model_family.trial_search_space=MagicMock(return_value=[SearchParam("width","int",low=2,high=8)])
    c.model_family.sample_hparams=lambda trial,*args:{"width":trial.suggest_int("width",2,8)}
    if multi:
        c.config.nas.score=Dict(type="multi-objective",metrics={},params={"objectives":[{"metric":"rmse_total","direction":"minimize"},{"metric":"latency_ms","direction":"minimize"}]})
    return c

@pytest.mark.parametrize("multi",[False,True])
def test_prefit_failure_then_viable_candidate_trains_once_and_logs_real_csv(tmp_path,multi):
    ThirdProposer.rounds=[[{"width":3},{"width":4}]]
    ThirdProposer.calls=[]
    c=real_objective_client(tmp_path,multi)
    requests=[]
    def hil(payload):
        requests.append(payload)
        if len(requests)==1:
            assert c.task.build_fit_plan.call_count==0
            return hil_metrics(HIL_MASTER_FATAL)
        assert c.task.build_fit_plan.call_count==0
        return hil_metrics()
    c._hil_request=hil
    with patch("nas_model_client.return_hardware_specs",return_value=(2048,4096)):
        study=execute(c,tmp_path)
    assert c.task.build_fit_plan.call_count==1
    assert c.model_family.build_model.return_value.fit.call_count==1
    assert len(study.trials)==2
    assert study.trials[0].state==(TrialState.COMPLETE if multi else TrialState.PRUNED)
    assert study.trials[0].values==([1e12,1e12] if multi else [-float("inf")])
    if not multi: assert study.trials[0].intermediate_values=={0:-float("inf")}
    assert study.trials[1].state==TrialState.COMPLETE
    assert study.trials[1].params=={"width":4}
    assert study.trials[0].user_attrs["pruned"] is True
    rows=list(csv.DictReader((c._artifacts_dir()/c.config.outputs.log_file_name).open()))
    assert len(rows)==2 and rows[0]["pruned"]=="True"
    assert all("crest_proposal" not in key for key in rows[0])


def test_fixed_candidate_shared_objective_matches_prior_enqueued_execution(tmp_path):
    candidates=[{"width":3},{"width":4}]
    outcomes=[]
    for shared in [False,True]:
        c=real_objective_client(tmp_path/str(shared))
        c.study_name="parity"
        c._hil_request=MagicMock(side_effect=[hil_metrics() for _ in candidates])
        study=optuna.create_study(direction="maximize")
        with patch("nas_model_client.return_hardware_specs",return_value=(2048,4096)),patch("crest.model.time.time",return_value=12345),patch("crest.model.time.strftime",return_value="fixed"):
            if shared:
                ThirdProposer.rounds=[candidates]
                c._execute_search_round(study,ThirdProposer(),NAME,SearchSpaceDescriptor((SearchParam("width","int",low=2,high=8),)),BudgetSnapshot(0,0,2,2,2))
            else:
                for params in candidates: study.enqueue_trial(params)
                study.optimize(c.objective,n_trials=2)
        trials=[(t.params,t.values,{k:v for k,v in t.user_attrs.items() if k!=PROPOSAL_ATTR},t.intermediate_values,t.system_attrs) for t in study.trials]
        csv_text=(c._artifacts_dir()/c.config.outputs.log_file_name).read_text()
        inputs=[call.args[0] for call in c._hil_request.call_args_list]
        outcomes.append((trials,csv_text,inputs))
    assert outcomes[0]==outcomes[1]


def test_invalid_declared_descriptor_propagates_before_objective(client,tmp_path):
    client.model_family.trial_search_space=MagicMock(side_effect=ValueError("invalid descriptor"))
    client.objective=MagicMock()
    with pytest.raises(ValueError,match="invalid descriptor"):
        execute(client,tmp_path)
    client.objective.assert_not_called()


@pytest.mark.parametrize("multi,cap,expected",[(False,10,1),(True,10,8),(True,3,3)])
def test_native_round_boundaries_preserve_population_and_target_rules(client,tmp_path,multi,cap,expected):
    client.config.optimizer=Dict(type="optuna")
    client.config.training.nas_trials=1
    client.config.training.max_total_trials=cap
    if multi:
        client.config.nas.score=Dict(type="multi-objective",metrics={},params={"objectives":[{"metric":"flops","direction":"minimize"},{"metric":"weight_bytes","direction":"minimize"}]})
        client.objective=lambda trial:(float(trial.suggest_int("width",2,8)),1.0)
    sizes=[]
    original=optuna.Study.optimize
    def optimize(study,objective,n_trials):
        sizes.append(n_trials)
        return original(study,objective,n_trials=n_trials)
    with patch.object(optuna.Study,"optimize",optimize),patch.object(optuna.Study,"enqueue_trial",side_effect=AssertionError("native must never enqueue")):
        result=execute(client,tmp_path)
    assert sizes==[expected]
    assert len(result.trials)==expected


def test_smoke_third_proposer_runs_exact_attempts_and_restores_budget(client,tmp_path):
    ThirdProposer.rounds=[[{"width":3}],[{"width":4}]]
    count=0
    def objective(trial):
        nonlocal count
        count+=1
        trial.suggest_int("width",2,8)
        if count==1: raise optuna.TrialPruned()
        return 1.0
    client.objective=objective
    original=(client.config.training.nas_trials,client.config.training.max_total_trials)
    client.smoke_test(train=False,hil=False,trials=2,epochs=1,study_name="smoke")
    assert count==2
    assert [b.target for _,b in ThirdProposer.calls]==[2,2]
    assert (client.config.training.nas_trials,client.config.training.max_total_trials)==original


def test_explicit_active_runtime_fields_all_consumed_by_real_objective(tmp_path):
    c=real_objective_client(tmp_path)
    c.config.training.quantization=Dict(mode="float",search=True,choices=["float","int8_ptq"])
    c.config.device.cpu_clock_mhz_options=[80,160]
    c.config.training.nas_trials=1
    c.config.training.max_total_trials=1
    params={"width":3,"quantization_mode":"float","cpu_clock_mhz_index":1}
    ThirdProposer.rounds=[[params]]
    c._hil_request=MagicMock(return_value=hil_metrics())
    with patch("nas_model_client.return_hardware_specs",return_value=(2048,4096)):
        study=execute(c,tmp_path)
    assert study.trials[0].params==params
    assert _family_trial_params(study.best_trial.params)=={"width":3}
    assert c._hil_request.call_args.args[0]["device_options_overrides"]=={"cpu_clock_mhz":160}


def test_build_failure_keeps_intended_vs_sampled_fields_and_aborts(tmp_path):
    c=real_objective_client(tmp_path)
    c.config.training.quantization=Dict(mode="float",search=True,choices=["float","int8_ptq"])
    ThirdProposer.rounds=[[{"width":3,"quantization_mode":"float"},{"width":4,"quantization_mode":"float"}]]
    c.model_family.build_model.side_effect=RuntimeError("build failed")
    with pytest.raises(RuntimeError,match="build failed"):
        execute(c,tmp_path)
    study=optuna.load_study(study_name="acceptance",storage=f"sqlite:///{tmp_path / 'study.db'}")
    assert [t.state for t in study.trials]==[TrialState.FAIL,TrialState.WAITING]
    assert study.trials[0].params=={"width":3}
    assert study.trials[0].user_attrs[PROPOSAL_ATTR]["proposal_params"]=={"width":3,"quantization_mode":"float"}
    c.task.build_fit_plan.assert_not_called()


@pytest.mark.parametrize("field,value",[("model","changed-model"),("prompt_version","v2"),("type","optuna")])
def test_llm_signature_mismatch_with_waiting_never_initializes_provider(client,tmp_path,field,value):
    client.config.optimizer=Dict(type="llm_generator",llm={"provider":"fake","responses":[{"candidates":[{"width":3}]}]})
    storage=f"sqlite:///{tmp_path / 'study.db'}"
    study=optuna.create_study(study_name="acceptance",storage=storage,direction="maximize")
    sign(client,study)
    study.enqueue_trial({"width":3})
    if field=="type": client.config.optimizer.type=value
    else: client.config.optimizer.llm[field]=value
    client.objective=MagicMock()
    with patch("crest.optimizers.llm.component.build_provider",side_effect=AssertionError("provider initialization forbidden")),pytest.raises(RuntimeError,match="optimizer signature"):
        execute(client,tmp_path)
    client.objective.assert_not_called()
    assert optuna.load_study(study_name="acceptance",storage=storage).trials[0].state==TrialState.WAITING


def test_audio_family_smoke_uses_real_shared_objective_and_all_fields(client,tmp_path):
    from crest.model_families.audio_dscnn import AudioDSCNNFamily
    from test.test_audio_dscnn import make_context
    client.model_family=AudioDSCNNFamily()
    client.model_family_name="audio_dscnn"
    client.model_config=Dict(family="audio_dscnn",params={},search={})
    client.config.model=client.model_config
    client.model_build_context=make_context(input_shape=(32,16))
    client.target_spec=client.model_build_context.target_spec
    client.dataset_name="urbansound8k_mel"
    client.task_name="sound_classification"
    params=client.model_family.default_seed_trial(client.model_build_context,client.model_config)
    ThirdProposer.rounds=[[params]]
    client.objective=NASModelClient.objective.__get__(client)
    with patch.object(client,"_hil_request",side_effect=AssertionError("HIL must not run")):
        client.smoke_test(train=False,hil=False,trials=1,epochs=1,study_name="audio-contract-smoke")
    study=optuna.load_study(study_name="audio-contract-smoke",storage=f"sqlite:///{client._artifacts_dir() / 'optuna_smoke_test.db'}")
    assert study.trials[0].state==TrialState.COMPLETE
    assert study.trials[0].params==params
    assert study.trials[0].user_attrs[PROPOSAL_ATTR]["proposal_params"]==params
    assert client.model_family.decode_trial_hparams(_family_trial_params(study.best_trial.params),client.model_build_context,client.model_config)==params
