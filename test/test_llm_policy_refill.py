"""Offline checks for explicit allocations and retained refill budgets."""
import json
import random
import tempfile
from dataclasses import replace
from pathlib import Path
from test.test_llm_enqueue import context, DESCRIPTOR
from crest.optimizers.llm.phase_policy import resolve_allocation
from crest.optimizers.llm.schemas import CandidateBatch, validate_policy_batch
from crest.optimizers.llm.enqueue import generate_llm_batch
from crest.optimizers.llm.provider import FakeProvider
from crest.optimizers.llm.ledger import LLMLedger


def entry(width, intent="exploration", **extra):
    """Build one parseable policy entry with independent intent metadata."""
    return {"params": {"width": width, "mode": "small"},
            "rationale": "Test a new width", "intent": intent, **extra}


def test_explicit_small_batch_allocations():
    # Freeze all fifteen policy cells and the no-parent conversion.
    expected = {"exploration": [(0,1),(1,1),(1,2),(2,2),(2,3)],
                "specification": [(1,0),(1,1),(2,1),(2,2),(3,2)],
                "finalization": [(1,0),(1,1),(2,1),(3,1),(4,1)]}
    for phase, rows in expected.items():
        for size, (local, exploration) in enumerate(rows, 1):
            assert resolve_allocation(phase, size) == {"local":local,"exploration":exploration}
            assert resolve_allocation(phase,size,has_feasible_parent=False) == {"local":0,"exploration":size}


def test_quota_overflow_preserves_valid_subset():
    accepted, rejected = validate_policy_batch(CandidateBatch(candidates=[entry(2),entry(3),entry(4,"bad")]),
                                               DESCRIPTOR, quotas={"local":1,"exploration":1})
    assert [x["params"]["width"] for x in accepted] == [2]
    assert [x.code for x in rejected] == ["quota_overflow","invalid_policy_envelope"]


def test_retains_candidates_and_shared_budgets():
    # A malformed refill consumes the single shared repair, not a fresh budget.
    provider = FakeProvider([{"candidates":[entry(2)]}, "bad-json",
                             {"candidates":[entry(3)]}, {"candidates":[entry(3)]}])
    ctx = replace(context(3), full_history_mode=True, allocation={"local":0,"exploration":3})
    with tempfile.TemporaryDirectory() as directory:
        ledger = LLMLedger(Path(directory))
        proposals = generate_llm_batch((), provider, ctx, ledger, prompt_version="test",
                                      max_repair_attempts=1,max_refill_attempts=2,rng=random.Random(0))
        assert [p.params["width"] for p in proposals] == [2,3]
        assert proposals[0].provenance["request_id"] != proposals[1].provenance["request_id"]
        events = [json.loads(line) for line in (Path(directory)/"optimizer_events.jsonl").read_text().splitlines()]
        final = next(x for x in events if x["event"] == "batch_policy_exhausted")
        assert final["repair_calls"] == 1 and final["refill_calls"] == 2


def test_target_membership_and_parent_diversity():
    from unittest.mock import patch
    # Distinct decoded parents remain mandatory across refills until exhausted.
    parents = [{"trial_number": 1, "params": {"width": 2, "mode": "small"}},
               {"trial_number": 2, "params": {"width": 3, "mode": "small"}}]
    with patch("crest.optimizers.llm.full_history.decoded_fingerprint", side_effect=lambda p: str(p["width"])), \
         patch("crest.optimizers.llm.full_history.candidate_cell", side_effect=lambda p: str(p["width"])):
        accepted, rejected = validate_policy_batch(
            CandidateBatch(candidates=[entry(4, "local", parent_trial=1),
                                       entry(5, "local", parent_trial=2), entry(6)]),
            DESCRIPTOR, quotas={"local": 1, "exploration": 1}, parents=parents,
            used_parents={"2"}, targets=[{"cell_id": "7"}])
        assert [x["parent_trial"] for x in accepted] == [2]
        assert [x.code for x in rejected] == ["parent_reused", "wrong_exploration_target"]



def test_target_multiplicity_and_post_goal_exploration():
    from unittest.mock import patch
    # Repeated targets are separate slots; consuming them never opens a free slot.
    with patch("crest.optimizers.llm.full_history.candidate_cell", return_value="cell"):
        accepted, rejected = validate_policy_batch(
            CandidateBatch(candidates=[entry(2), entry(3), entry(4)]), DESCRIPTOR,
            quotas={"exploration": 3}, targets=[{"cell_id": "cell"}, {"cell_id": "cell"}])
    assert len(accepted) == 2
    assert rejected[0].code == "wrong_exploration_target"
    accepted, rejected = validate_policy_batch(CandidateBatch(candidates=[entry(2)]),
                                               DESCRIPTOR, quotas={"exploration": 1})
    assert len(accepted) == 1 and not rejected


def test_retained_decoded_alias_rejected():
    from unittest.mock import patch
    # Raw differences do not bypass a retained training-network identity.
    with patch("crest.optimizers.llm.full_history.decoded_fingerprint", return_value="same-network"):
        accepted, rejected = validate_policy_batch(CandidateBatch(candidates=[entry(3)]),
            DESCRIPTOR, quotas={"exploration": 1}, retained=[entry(2)])
    assert not accepted and rejected[0].code == "duplicate_decoded"


def test_random_fallback_rejects_decoded_alias():
    from unittest.mock import patch
    from crest.pipeline_types import TrialRecord
    prior = TrialRecord(number=0, state="COMPLETE", params={"width": 2, "mode": "small"},
                        values=(0.1,), user_attrs={}, system_attrs={})
    provider = FakeProvider(["bad-json"])
    ctx = replace(context(1), full_history_mode=True, allocation={"local":0,"exploration":1})
    def identity(params):
        return "old-network" if params["width"] in (2,3) else "new-network"
    with tempfile.TemporaryDirectory() as directory, \
         patch("crest.optimizers.llm.full_history.decoded_fingerprint", side_effect=identity), \
         patch("crest.optimizers.llm.enqueue.sample_random_candidate", side_effect=[entry(3)["params"], entry(4)["params"]]):
        proposals = generate_llm_batch((prior,), provider, ctx, LLMLedger(Path(directory)),
            prompt_version="test", max_repair_attempts=0, max_refill_attempts=0, rng=random.Random(0))
    assert proposals[0].params["width"] == 4
    assert proposals[0].provenance["proposal_source"] == "random_fallback"



def test_boolean_only_failure_is_retryable():
    from crest.pipeline_types import TrialRecord
    # A negative boolean without explicit status is not confirmed infeasibility.
    prior = TrialRecord(number=0, state="FAIL", params={"width": 2, "mode": "small"},
                        values=None, user_attrs={"feasible": False}, system_attrs={})
    provider = FakeProvider([{"candidates": [entry(2), entry(3)]}])
    ctx = replace(context(1), full_history_mode=True, allocation={"local": 0, "exploration": 1})
    with tempfile.TemporaryDirectory() as directory:
        proposals = generate_llm_batch((prior,), provider, ctx, LLMLedger(Path(directory)),
            prompt_version="test", max_repair_attempts=0, max_refill_attempts=0, rng=random.Random(0))
    assert proposals[0].params["width"] == 2



def test_running_work_blocks_reproposal_despite_old_failure_metadata():
    from crest.pipeline_types import TrialRecord
    # Retry eligibility applies to finished infrastructure failures, not running work.
    prior = TrialRecord(number=0, state="RUNNING", params={"width": 2, "mode": "small"},
                        values=None, user_attrs={"measurement_failure_class": "infrastructure"}, system_attrs={})
    provider = FakeProvider([{"candidates": [entry(2), entry(3)]}])
    ctx = replace(context(1), full_history_mode=True, allocation={"local": 0, "exploration": 1})
    with tempfile.TemporaryDirectory() as directory:
        proposals = generate_llm_batch((prior,), provider, ctx, LLMLedger(Path(directory)),
            prompt_version="test", max_repair_attempts=0, max_refill_attempts=0, rng=random.Random(0))
    assert proposals[0].params["width"] == 3



def test_duplicate_feedback_excludes_unrelated_artifact_logs():
    from unittest.mock import patch
    from crest.pipeline_types import TrialRecord
    marker = "UNRELATED_BUILD_ARTIFACT_" * 10000
    score = 0.123456789012345
    prior = TrialRecord(number=7, state="COMPLETE", params=entry(2)["params"], values=(score,),
        user_attrs={"build_log": marker, "task_metrics": {"macro_f1": score},
                    "latency_ms": 1.23456789012345, "energy_mj_per_inference": 0.000123456789,
                    "failure_reason": "x" * 20000}, system_attrs={})
    provider = FakeProvider([{"candidates": [entry(2)]}, {"candidates": [entry(3)]}])
    ctx = replace(context(1), full_history_mode=True, allocation={"local":0,"exploration":1})
    with tempfile.TemporaryDirectory() as directory:
        generate_llm_batch((prior,), provider, ctx, LLMLedger(Path(directory)),
            prompt_version="test", max_repair_attempts=0, max_refill_attempts=1,rng=random.Random(0))
        feedback = (Path(directory)/"rejected_candidates.jsonl").read_text()
        assert "UNRELATED_BUILD_ARTIFACT" not in feedback and len(feedback) < 3000
        assert str(score) in feedback and "1.23456789012345" in feedback
        assert "truncated failure detail" not in feedback  # Successful work has no failure label.


def test_many_duplicate_matches_keep_ids_without_repeating_details():
    matches = [{"trial": n, "params": entry(2)["params"], "outcome": {"score": n / 1000}}
               for n in range(300)]
    accepted, rejected = validate_policy_batch(CandidateBatch(candidates=[entry(2)]),
        DESCRIPTOR, quotas={"exploration":1}, completed_candidates=[entry(2)["params"]],
        history_matches=matches)
    message = rejected[0].message.split(": ", 1)[1]
    feedback = json.loads(message)
    assert feedback["matching_trials"] == list(range(300))
    assert len(feedback["match_details"]) == 3
    assert feedback["additional_details_in_full_history"] == 297
    assert len(message) < 3000


def test_direct_full_mode_budget_limits(tmp_path):
    import pytest
    # The public generation boundary rejects excessive budgets before any call.
    ctx = replace(context(1), full_history_mode=True, allocation={"exploration": 1})
    for repairs, refills in [(2, 0), (0, 3), (-1, 0), (True, 0)]:
        provider = FakeProvider([])
        with pytest.raises(ValueError, match="repair attempts"):
            generate_llm_batch((), provider, ctx, LLMLedger(tmp_path), prompt_version="test",
                max_repair_attempts=repairs, max_refill_attempts=refills, rng=random.Random(0))
        assert not provider.requests


def test_partial_input_guard_preserves_accepted_and_actual_reason(tmp_path):
    from crest.optimizers.llm.spending import SpendTracker
    from crest.optimizers.llm.prompt_builder import build_candidate_request
    from test.test_llm_spending import configuration
    # Exercise the real pre-send size guard: initial request fits exactly, while
    # the retained-candidate refill exceeds it without reaching another fake call.
    ctx = replace(context(2), full_history_mode=True, allocation={"exploration": 2})
    initial = build_candidate_request(ctx, prompt_version="test")
    bound = len(json.dumps(initial.messages(), separators=(",", ":")).encode()) + 1024
    tracker = SpendTracker(tmp_path / "spend", configuration(max_usd="1", max_input_tokens=bound))
    class GuardedProvider(FakeProvider):
        def complete_json(self, request):
            """Apply the production size guard before any simulated content call."""
            tracker.reserve({"messages": request.messages()})
            return super().complete_json(request)
    provider = GuardedProvider([{"candidates": [entry(2)]}])
    result = generate_llm_batch((), provider, ctx, LLMLedger(tmp_path), prompt_version="test",
        max_repair_attempts=1, max_refill_attempts=2, rng=random.Random(0))
    assert [p.params["width"] for p in result] == [2]
    assert len(provider.requests) == 1
    events = [json.loads(x) for x in (tmp_path / "optimizer_events.jsonl").read_text().splitlines()]
    assert events[-1]["unfilled_reason"] == "input_guard"


def test_penalty_pruned_retry_eligibility_matches_rows_and_coverage(tmp_path):
    from test.test_llm_full_history import params, trial
    from test.test_llm_full_context import SMOKE, ROOT
    from crest.optimizers.llm.full_history import build_full_history, build_coverage, candidate_cell
    from crest.optimizers.llm.prompt_builder import enrich_full_history
    # Use actual penalty-shaped attributes and real decoder identity throughout.
    saved = json.loads((ROOT / "test/fixtures/cs3_legacy_request_000088.json").read_text())
    path = tmp_path / "context.jsonl"
    path.write_text(saved["messages"][1]["content"] + "\n")
    base = SMOKE.load_context(path)
    raw = {name: param.choices[0] for name, param in base.descriptor.items()}
    penalty = dict(feasible=False, feasibility_status="not_evaluated", pruned=True,
        prune_reason="HIL fatal error", latency_ms=10000.0, energy_mj_per_inference=10000.0)
    history = [trial(0, "PRUNED", raw, **penalty)]
    coverage = build_coverage(history, base.descriptor)
    cell = next(c for c in coverage["cells"] if c["cell_id"] == candidate_cell(raw))
    assert cell["informative"] == 0 and cell["unclassified"] == 1
    row = dict(zip(build_full_history(history)["columns"], build_full_history(history)["rows"][0]))
    assert row["failure_class"] == "unclassified" and row["latency_ms"] is None
    ctx = replace(enrich_full_history(base, history), allocation={"exploration": 1}, batch_size=1,
                  exploration_targets=({"cell_id": candidate_cell(raw)},))
    for state, classification, accepted in [("PRUNED", None, True), ("FAIL", None, True),
            ("PRUNED", "infrastructure", True), ("FAIL", "deterministic", False)]:
        attrs = dict(penalty)
        if classification:
            attrs["measurement_failure_class"] = classification
        prior = trial(0, state, raw, **attrs)
        provider = FakeProvider([{"candidates": [{"params": raw, "intent": "exploration", "rationale": "Retry unknown evidence"}]}])
        result = generate_llm_batch((prior,), provider, ctx, LLMLedger(tmp_path / f"{state}-{classification}"),
            prompt_version="test", max_repair_attempts=0, max_refill_attempts=0, rng=random.Random(0))
        assert (result[0].provenance["proposal_source"] == "llm") == accepted
