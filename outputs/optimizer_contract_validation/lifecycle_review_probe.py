from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import optuna
from optuna.trial import TrialState
from test.test_optimizer_runner import client, sign, ThirdProposer, PROPOSAL_ATTR, NAME

optuna.logging.set_verbosity(optuna.logging.ERROR)

with TemporaryDirectory(prefix='crest-review-mixed-') as root:
    path = Path(root)
    c = client.__wrapped__(path)
    storage = f'sqlite:///{path / "study.db"}'
    study = optuna.create_study(study_name='mixed', storage=storage, direction='maximize')
    sign(c, study)
    t = study.ask(); t.suggest_int('width', 2, 8); study.tell(t, 3.0)
    orphan = study.ask()
    for width, group in [(4, 'old'), (5, 'old'), (6, 'new')]:
        study.enqueue_trial({'width': width}, user_attrs={PROPOSAL_ATTR: {'round_id': group}})
    c.config.training.nas_trials = 3
    c.config.training.max_total_trials = 1
    sizes = []
    original = optuna.Study.optimize
    def record(study, objective, n_trials):
        sizes.append(n_trials)
        return original(study, objective, n_trials=n_trials)
    with patch.object(optuna.Study, 'optimize', record):
        result = c.run_nas('mixed', storage)
    assert sizes == [2], sizes
    assert [t.state for t in result.trials] == [TrialState.COMPLETE, TrialState.RUNNING, TrialState.COMPLETE, TrialState.COMPLETE, TrialState.WAITING]
    assert [t.params['width'] for t in result.trials[2:4]] == [4,5]
    assert len(result.trials) == 5 and not ThirdProposer.calls
    print('PASS mixed COMPLETE/RUNNING + two WAITING groups, cap below reservations, exact [2] consumption')

with TemporaryDirectory(prefix='crest-review-smoke-') as root:
    path = Path(root)
    c = client.__wrapped__(path)
    c.study_name = 'resume'
    storage = f'sqlite:///{c._artifacts_dir() / "optuna_smoke_test.db"}'
    study = optuna.create_study(study_name='resume', storage=storage, direction='maximize')
    sign(c, study)
    for width in [3,4,5]:
        study.enqueue_trial({'width':width}, user_attrs={PROPOSAL_ATTR:{'round_id':'old'}})
    c.smoke_test(train=False, hil=False, trials=2, epochs=1, study_name='resume')
    result = optuna.load_study(study_name='resume',storage=storage)
    assert [t.state for t in result.trials] == [TrialState.COMPLETE,TrialState.COMPLETE,TrialState.WAITING]
    assert not ThirdProposer.calls
    ThirdProposer.rounds = [[{'width':6}]]
    c.smoke_test(train=False, hil=False, trials=2, epochs=1, study_name='resume')
    result = optuna.load_study(study_name='resume',storage=storage)
    assert [t.params['width'] for t in result.trials] == [3,4,5,6]
    assert len(ThirdProposer.calls) == 1
    history,budget = ThirdProposer.calls[0]
    assert len(history) == 3 and all(t.state == 'COMPLETE' for t in history)
    assert budget.permitted_round_size == 1 and budget.target == 4
    print('PASS smoke splits reserved group, next run drains remainder before exactly one new proposal')

with TemporaryDirectory(prefix='crest-review-interrupt-') as root:
    path = Path(root)
    c = client.__wrapped__(path)
    ThirdProposer.rounds = [[{'width':3},{'width':4}]]
    original = optuna.Study.enqueue_trial
    calls = 0
    def interrupted(study,params,**kwargs):
        global calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt('probe')
        return original(study,params,**kwargs)
    storage = f'sqlite:///{path / "study.db"}'
    try:
        with patch.object(optuna.Study,'enqueue_trial',interrupted):
            c.run_nas('interrupted',storage)
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError('interruption did not propagate')
    ThirdProposer.calls=[]
    c.config.training.max_total_trials=1
    result=c.run_nas('interrupted',storage)
    assert len(result.trials)==1 and result.trials[0].state==TrialState.COMPLETE
    assert result.trials[0].params=={'width':3} and not ThirdProposer.calls
    print('PASS KeyboardInterrupt during partial enqueue persists first reservation and executes it at cap')
