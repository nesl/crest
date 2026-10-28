"""Use production objective evidence to distinguish failed designs from outages."""
from unittest.mock import MagicMock

import optuna
import pytest

from crest.optimizers.history import build_trial_snapshot
from crest.optimizers.llm.enqueue import _history_candidates
from crest.optimizers.search_space import SearchParam, SearchSpaceDescriptor
from test.test_nas_model_client import _build_test_client


@pytest.mark.parametrize('failure', [ValueError('invalid candidate shape'), NotImplementedError('unsupported op')])
def test_objective_failure_evidence_blocks_failed_candidate(tmp_path, failure):
    """Real objective failures supply dedup evidence without a test-only class label."""
    client = _build_test_client(tmp_path)
    client.model_family.build_model.side_effect = failure
    client._hil_request = MagicMock()
    study = optuna.create_study()
    study.enqueue_trial({'width': 3}, user_attrs={'crest_proposal': {'proposal_params': {'width': 3}}})
    with pytest.raises(type(failure)):
        study.optimize(client.objective, n_trials=1)
    record = study.trials[0]
    assert record.state.name == 'FAIL'
    assert record.user_attrs['candidate_failure_stage'] == 'model_construction'
    descriptor = SearchSpaceDescriptor((SearchParam('width', 'int', low=2, high=8),))
    assert _history_candidates(build_trial_snapshot(study), descriptor, {'FAIL'}) == [{'width': 3}]
    client._hil_request.assert_not_called()


@pytest.mark.parametrize('failure', [OSError('disk unavailable'), RuntimeError('unknown backend failure'), MemoryError('out of memory')])
def test_unknown_construction_failure_stays_retryable(tmp_path, failure):
    """Do not turn unknown infrastructure/resource errors into design rejections."""
    client = _build_test_client(tmp_path)
    client.model_family.build_model.side_effect = failure
    study = optuna.create_study()
    study.enqueue_trial({'width': 3}, user_attrs={'crest_proposal': {'proposal_params': {'width': 3}}})
    with pytest.raises(type(failure)):
        study.optimize(client.objective, n_trials=1)
    assert 'candidate_failure_class' not in study.trials[0].user_attrs
    descriptor = SearchSpaceDescriptor((SearchParam('width', 'int', low=2, high=8),))
    assert _history_candidates(build_trial_snapshot(study), descriptor, {'FAIL'}) == []
