"""Timeout attribution tests with synthetic, non-paper campaign sizes."""

import importlib.util
from pathlib import Path

import pytest

MODULE = (
    Path(__file__).parents[1] / "analysis_scripts/timing_attribution/prepare_inputs.py"
)
SPEC = importlib.util.spec_from_file_location("timing_inputs", MODULE)
inputs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inputs)


def campaign(tmp_path):
    """Construct two candidates per condition on an unfamiliar server."""
    spans, reports = [], {}
    for condition in ("continuous", "cadenced"):
        run = "new-" + condition
        reports[condition] = {"run": {"run_id": run}, "counts": {"trained": 2}}
        base = dict(run_id=run, condition=condition, host="custom-server", pid=42)
        root = dict(
            base,
            span_id=run,
            parent_id=None,
            stage="hil.service",
            start_ns=0,
            end_ns=100,
            inclusive_s=100e-9,
        )
        spans.append(root)
        for number, (lo, hi) in enumerate([(10, 20), (30, 50), (60, 70)]):
            spans.append(
                dict(
                    base,
                    span_id=f"{run}-{number}",
                    parent_id=run,
                    stage="backend.measure",
                    start_ns=lo,
                    end_ns=hi,
                    inclusive_s=(hi - lo) / 1e9,
                    candidate_id=f"c{number%2}",
                )
            )
        directory = tmp_path / f"server/runs/custom-{condition}"
        directory.mkdir(parents=True)
        (directory / "server.console.log").write_text(
            "\n".join(
                f"HIL_controller attempt idx={i} arena=8 KiB err_flag={flag}"
                for i, flag in enumerate([0, 2, 0])
            )
        )
    return spans, reports


def test_timeout_excludes_successful_measurements(tmp_path):
    spans, reports = campaign(tmp_path)
    result = inputs.arena_search_split(
        tmp_path, spans, reports, "custom-", "arena-flags"
    )["conditions"]
    for condition in result.values():
        assert condition["candidates"] == 2
        assert condition["base_attempts"] == 3
        assert condition["failed_attempts"] == 1
        assert condition["failed_probe_measurement_s"] == pytest.approx(20e-9)


def test_timeout_rejects_build_inside_failed_session(tmp_path):
    spans, reports = campaign(tmp_path)
    bad = dict(
        spans[2],
        span_id="nested-build",
        parent_id=spans[2]["span_id"],
        stage="backend.compile",
        start_ns=31,
        end_ns=35,
        inclusive_s=4e-9,
    )
    with pytest.raises(ValueError, match="build/programming"):
        inputs.arena_search_split(
            tmp_path, spans + [bad], reports, "custom-", "arena-flags"
        )


def test_timeout_rejects_ambiguous_flags(tmp_path):
    spans, reports = campaign(tmp_path)
    (tmp_path / "server/runs/custom-continuous/server.console.log").write_text("")
    with pytest.raises(ValueError, match="match arena attempts"):
        inputs.arena_search_split(tmp_path, spans, reports, "custom-", "arena-flags")


def test_pipeline_cannot_write_into_source(tmp_path):
    with pytest.raises(ValueError, match="outside the source"):
        inputs.prepare(tmp_path, tmp_path / "new-analysis")
