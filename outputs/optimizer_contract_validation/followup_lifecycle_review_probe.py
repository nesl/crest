import copy
from unittest.mock import MagicMock, patch
from addict import Dict
import optuna
from optuna.trial import TrialState, create_trial
import pytest
from nas_model_client import OPTIMIZER_SIGNATURE_ATTR, LEGACY_OPTIMIZER_ADOPTION_ATTR, FEASIBILITY_POLICY_SIGNATURE_ATTR, PROPOSAL_ATTR
from test.test_optimizer_resume import campaign

@pytest.mark.parametrize('adopt', [False, True])
def test_old_signed_tpe_alias_drains_queue_at_cap_without_restamp(tmp_path, adopt):
    c, storage = campaign.__wrapped__(tmp_path)
    s = optuna.create_study(study_name='campaign',storage=storage,direction='maximize')
    c._select_optimizer(s,c._build_sampler())
    signature=s.user_attrs[OPTIMIZER_SIGNATURE_ATTR]
    signature['sampler']['class']='optuna.samplers._tpe.sampler.TPESampler'
    s.set_user_attr(OPTIMIZER_SIGNATURE_ATTR,signature)
    s.enqueue_trial({'width':3},user_attrs={PROPOSAL_ATTR:{'round_id':'older'}})
    s.enqueue_trial({'width':7},user_attrs={PROPOSAL_ATTR:{'round_id':'next'}})
    c.config.optimizer.adopt_legacy_study=adopt
    c.config.training.nas_trials=1
    c.config.training.max_total_trials=1
    with patch('nas_model_client.build_search_space_descriptor',side_effect=AssertionError('native setup must skip descriptor')):
        result=c.run_nas('campaign',storage)
    assert len(result.trials)==2
    assert result.trials[0].params=={'width':3}
    assert [t.state for t in result.trials]==[TrialState.COMPLETE,TrialState.WAITING]
    assert result.user_attrs[OPTIMIZER_SIGNATURE_ATTR]==signature
    assert LEGACY_OPTIMIZER_ADOPTION_ATTR not in result.user_attrs

def test_nonempty_signed_nsga_private_alias_extends_exact_budget_preserves_evidence(tmp_path):
    c,storage=campaign.__wrapped__(tmp_path)
    c.config.nas.score=Dict(type='multi-objective',metrics={},params={'objectives':[
        {'metric':'flops','direction':'minimize'},{'metric':'ram_bytes','direction':'minimize'}]})
    c.objective=lambda t: (float(t.suggest_int('width',2,8)),1.0)
    s=optuna.create_study(study_name='campaign',storage=storage,directions=['minimize','minimize'])
    c._select_optimizer(s,c._build_sampler())
    signature=s.user_attrs[OPTIMIZER_SIGNATURE_ATTR]
    signature['sampler']['class']='optuna.samplers.nsgaii._sampler.NSGAIISampler'
    s.set_user_attr(OPTIMIZER_SIGNATURE_ATTR,signature)
    for width in [3,4]:
        t=s.ask();t.suggest_int('width',2,8);t.set_user_attr('history','preserve');s.tell(t,(float(width),1.0))
    before=copy.deepcopy(s.trials)
    c.config.training.nas_trials=3;c.config.training.max_total_trials=3
    c.config.optimizer.adopt_legacy_study=True
    result=c.run_nas('campaign',storage)
    assert result.trials[:2]==before and len(result.trials)==3
    assert result.user_attrs[OPTIMIZER_SIGNATURE_ATTR]==signature
    assert LEGACY_OPTIMIZER_ADOPTION_ATTR not in result.user_attrs

def test_direction_guard_runs_before_missing_feasibility_or_adoption_stamp(tmp_path):
    c,storage=campaign.__wrapped__(tmp_path)
    c.config.optimizer.adopt_legacy_study=True
    c.config.nas.feasibility=Dict(train_if_infeasible=False,rules=[{
      'rule':'latency_budget','metric':'latency_ms','condition':'<=','reference':{'type':'literal','value':10}}])
    s=optuna.create_study(study_name='campaign',storage=storage,direction='minimize')
    s.add_trial(create_trial(value=3.0,user_attrs={'legacy':'retain'},system_attrs={'original':'evidence'}))
    before=copy.deepcopy(s.trials);attrs=copy.deepcopy(s.user_attrs)
    c.objective=MagicMock()
    with pytest.raises(RuntimeError,match='study directions'):
        c.run_nas('campaign',storage)
    result=optuna.load_study(study_name='campaign',storage=storage)
    assert result.trials==before and result.user_attrs==attrs
    c.objective.assert_not_called()
    for key in [OPTIMIZER_SIGNATURE_ATTR,LEGACY_OPTIMIZER_ADOPTION_ATTR,FEASIBILITY_POLICY_SIGNATURE_ATTR]:
        assert key not in result.user_attrs
