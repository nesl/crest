# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Exercise campaign cadence and restart persistence without external services."""

from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from addict import Dict
import optuna
from optuna.trial import TrialState
import pytest
import yaml

from crest.optimizers.llm.history import build_best_trial_anchors
from crest.optimizers.llm.memory import ExperimentalMemory, MemoryConfig
from crest.optimizers.llm.provider import FakeProvider
from crest.search_space import SearchParam
from nas_model_client import PROPOSAL_ATTR
from test.test_optimizer_resume import campaign  # noqa: F401
from test.test_llm_memory import add, context, finding


def _batch(first):
    """Build five distinct synthetic proposals.

    Parameters
    ----------
    first : int
        First width in the response.

    Returns
    -------
    dict
        FakeProvider candidate response.
    """
    return {"candidates": [{"width": width} for width in range(first, first + 5)]}


def _jsonl(path):
    """Read a persistent ledger stream.

    Parameters
    ----------
    path : pathlib.Path
        JSONL ledger file.

    Returns
    -------
    list of dict
        Decoded ledger records.
    """
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_cs3_offline_campaign_memory_and_partial_batch_restart(campaign, tmp_path):
    """Preserve warmup, queued work, memory cadence, and provenance on restart.

    Parameters
    ----------
    campaign : tuple
        Existing lightweight client and real SQLite storage fixture.
    tmp_path : pathlib.Path
        Isolated campaign artifacts.

    Notes
    -----
    The synthetic objective replaces model construction, training, and hardware.
    Every provider response is deterministic; HTTP and socket connections fail
    immediately if any production path accidentally attempts external access.
    """
    client, storage = campaign
    warmup = [{"width": width} for width in range(2, 17)]
    warmup_path = tmp_path / "warmup.json"
    warmup_path.write_text(json.dumps(warmup))
    client.model_family.trial_search_space = MagicMock(
        return_value=[SearchParam("width", "int", low=2, high=100)])
    client.objective = lambda trial: float(trial.suggest_int("width", 2, 100))
    client.config.optimizer = Dict(type="llm_generator", llm={
        "provider": "fake", "responses": [_batch(17)], "batch_size": 5,
        "warmup_params_file": str(warmup_path), "semantic_context": False,
        "recent_trial_window": 15, "anchor_count": 5,
        "memory": {"enabled": True, "start_after_trials": 15, "interval_trials": 15, "lag_trials": 15},
    })
    client.config.training.nas_trials = 15
    client.config.training.max_total_trials = 40
    root = Path(client.config.outputs.models_dir) / "campaign" / "llm_optimizer"
    finding = {
        "id": "synthetic-width", "observation": "Synthetic score increases with width.",
        "conditions": "Deterministic width-only objective; widths 2 through 16.",
        "evidence": "Fifteen completed warmup trials returned their width as score.",
        "exceptions": "No exceptions in these fifteen synthetic trials.",
        "uncertainty": "This validates orchestration only; no model quality was measured.",
        "trial_numbers": list(range(15)),
    }
    summary = {"updates": [finding], "retire": []}
    warmup_provider = FakeProvider([])
    first_provider = FakeProvider([_batch(17)])
    middle_provider = FakeProvider([_batch(22), _batch(27)])
    final_provider = FakeProvider([summary, _batch(32)])
    original_optimize = optuna.Study.optimize

    def interrupt_partial_batch(study, objective, **kwargs):
        """Stop after two successful evaluations with three reservations intact.

        Parameters
        ----------
        study : optuna.Study
            Active persistent study.
        objective : callable
            Synthetic evaluation function.
        **kwargs : dict
            Runner optimization controls.

        Raises
        ------
        RuntimeError
            After the first generated batch has partially completed.
        """
        assert kwargs["n_trials"] == 5
        original_optimize(study, objective, **{**kwargs, "n_trials": 2})
        raise RuntimeError("offline partial batch interruption")

    with patch("urllib.request.urlopen", side_effect=AssertionError("HTTP forbidden")), \
            patch("socket.socket.connect", side_effect=AssertionError("network forbidden")):
        with patch("crest.optimizers.llm.component.build_provider", return_value=warmup_provider):
            first = client.run_nas("campaign", storage)
        warmup_trials = first.trials
        assert len(warmup_trials) == 15
        assert warmup_provider.requests == []
        assert not (root / "memory" / "state.json").exists()

        client.config.training.nas_trials = 30
        with patch("crest.optimizers.llm.component.build_provider", return_value=first_provider), \
                patch.object(optuna.Study, "optimize", interrupt_partial_batch), \
                pytest.raises(RuntimeError, match="partial batch interruption"):
            client.run_nas("campaign", storage)
        interrupted = optuna.load_study(study_name="campaign", storage=storage)
        assert [t.state for t in interrupted.trials] == [TrialState.COMPLETE] * 17 + [TrialState.WAITING] * 3
        assert interrupted.trials[:15] == warmup_trials
        assert not (root / "memory" / "state.json").exists()

        with patch("crest.optimizers.llm.component.build_provider", return_value=middle_provider):
            middle = client.run_nas("campaign", storage)
        middle_trials = middle.trials
        assert len(middle_trials) == 30
        assert all(t.state == TrialState.COMPLETE for t in middle_trials)
        assert middle_trials[:17] == interrupted.trials[:17]
        # The preserved reservations consume widths 19..21 before fresh work.
        assert [t.params["width"] for t in middle_trials] == list(range(2, 32))
        # Summary starts on the next proposal at thirty completed trials.
        assert not (root / "memory" / "state.json").exists()

        client.config.training.nas_trials = 35
        with patch("crest.optimizers.llm.component.build_provider", return_value=final_provider):
            final = client.run_nas("campaign", storage)

    assert final.trials[:30] == middle_trials
    assert len(final.trials) == 35
    assert all(t.state == TrialState.COMPLETE for t in final.trials)
    assert [t.params["width"] for t in final.trials] == list(range(2, 37))
    snapshots = sorted((root / "memory" / "snapshots").glob("*.json"))
    assert [len(json.loads(p.read_text())["covered"]) for p in snapshots] == [15]
    state = json.loads((root / "memory" / "state.json").read_text())
    assert state == json.loads(snapshots[-1].read_text())
    assert state["version"] == 1
    assert state["findings"] == [finding]
    assert set(state["covered"]) == {str(n) for n in range(15)}

    requests = first_provider.requests + middle_provider.requests + final_provider.requests
    summaries = [r for r in requests if r.metadata.get("purpose") == "summarization"]
    assert [r.metadata["new_trial_numbers"] for r in summaries] == [list(range(15))]
    assert len(requests) == 5  # One summary and four five-candidate generations.
    for index, trial in enumerate(final.trials[:15]):
        provenance = trial.user_attrs[PROPOSAL_ATTR]
        assert provenance["proposal_source"] == "llm_warmup"
        assert provenance["warmup_index"] == index
        assert provenance["proposal_params"] == warmup[index]
    returned = _jsonl(root / "returned_candidates.jsonl")
    assert len(returned) == 20
    for trial, record in zip(final.trials[15:], returned):
        provenance = trial.user_attrs[PROPOSAL_ATTR]
        assert provenance["proposal_source"] == "llm"
        assert provenance["proposal_params"] == record["candidate"] == trial.params
        assert provenance["request_id"] == record["request_id"]
        assert provenance["batch_id"] == record["batch_id"]
        assert provenance["batch_index"] == record["batch_index"]
    assert {r["request_id"] for r in returned} == {1, 2, 3, 4}
    for ledger_root, count in ((root, 4), (root / "memory", 1)):
        assert len(list((ledger_root / "requests").glob("*.request.json"))) == count
        assert len(list((ledger_root / "requests").glob("*.response.json"))) == count
    assert [r["event"] for r in _jsonl(root / "memory" / "optimizer_events.jsonl")] == [
        "summary_committed"]


def test_cs3_suspended_backlog_uses_configured_fifteen_record_bound(tmp_path):
    """Retain all evidence while bounding resumed CS3 pending prompt records.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated persistent memory ledger and snapshots.

    Notes
    -----
    Real memory transitions and restarts exercise the campaign YAML settings.
    Deterministic fake responses establish accepted memory and suspension; the
    resumed suspended owner must make no provider calls as the backlog grows.
    """
    config_path = Path(__file__).resolve().parents[1] / "experiments/cs3/nas_config_cs3_luna.yaml"
    settings = yaml.safe_load(config_path.read_text())["optimizer"]["llm"]
    assert settings["recent_trial_window"] == 15
    assert settings["anchor_count"] == 5
    memory_config = MemoryConfig.from_config(settings["memory"])
    assert memory_config.start_after_trials == memory_config.interval_trials == 15
    assert memory_config.lag_trials == 15
    assert memory_config.max_pending_trials == 15
    study = optuna.create_study(study_name="cs3-suspended-backlog")
    for _ in range(30):
        add(study)
    root = tmp_path / "memory"
    accepted_provider = FakeProvider([{"updates": [finding()], "retire": []}])
    with patch("urllib.request.urlopen", side_effect=AssertionError("HTTP forbidden")), \
            patch("socket.socket.connect", side_effect=AssertionError("network forbidden")):
        memory = ExperimentalMemory(root, memory_config)
        memory.enrich(study, context(study, 15), accepted_provider)
        accepted = json.loads(memory.state_path.read_text())
        snapshot_path = root / "snapshots/000001.json"
        snapshot = snapshot_path.read_bytes()
        for _ in range(15):
            add(study)
        invalid = {"updates": [{"id": "incomplete"}], "retire": []}
        for failure_count in range(1, 4):
            memory = ExperimentalMemory(root, memory_config)
            rejected_provider = FakeProvider([invalid] * 2)
            memory.enrich(study, context(study, 15), rejected_provider)
            assert len(rejected_provider.requests) == 2
            assert memory.state["summary_failure_count"] == failure_count
        for _ in range(120):
            add(study)
        suspended_state = memory.state_path.read_bytes()
        provider = FakeProvider([])
        for _ in range(2):
            restored = ExperimentalMemory(root, memory_config)
            prompt_context = replace(
                context(study, settings["recent_trial_window"]),
                anchors=build_best_trial_anchors(study, anchor_count=settings["anchor_count"]),
            )
            result = restored.enrich(study, prompt_context, provider)
            assert [r["number"] for r in result.pending_trials] == list(range(135, 150))
            assert [r["number"] for r in result.recent_trials] == list(range(150, 165))
            assert len(result.anchors) == 5
            assert result.knowledge_base["summary_status"] == "suspended"
            assert result.knowledge_base["pending_trial_count"] == 150
            assert result.knowledge_base["omitted_pending_trial_count"] == 120
            assert result.knowledge_base["covered_trial_count"] == 15
            assert "No int8 model has been tested." in result.knowledge_base["summary"]
            for field in ("covered", "findings", "version"):
                assert restored.state[field] == accepted[field]
            assert restored.state_path.read_bytes() == suspended_state
            assert snapshot_path.read_bytes() == snapshot
        assert provider.requests == []
    assert len(study.trials) == 165
    assert len(list((root / "snapshots").glob("*.json"))) == 1
    backlog_events = [r for r in _jsonl(root / "optimizer_events.jsonl")
                      if r["event"] == "summary_backlog"]
    assert [r["omitted_pending_trial_count"] for r in backlog_events] == [120, 120]
