# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Small synthetic checks for paper analyses, without experiment data."""

import csv
import importlib.util
import json
from pathlib import Path

import pytest


PLOTS = Path(__file__).resolve().parents[1] / "analysis_scripts" / "paper_plots"


def module(name):
    spec = importlib.util.spec_from_file_location(name, PLOTS / f"{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


@pytest.fixture(scope="module")
def training():
    return module("plot_training_budget")


@pytest.fixture(scope="module")
def proposal():
    return module("plot_case3_proposal_comparison")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def training_rows():
    return [
        dict(
            candidate_id=cid,
            budget=budget,
            rmse_total=error,
            energy_mj=energy,
            source="external/checkpoint.json",
        )
        for budget in (15, 300)
        for cid, error, energy in [("small", 0.8, 1.0), ("large", 0.3, 4.0)]
    ]


def test_front_retains_tradeoffs_and_equal_objectives(training):
    points = {
        "small": (0.8, 1.0),
        "large": (0.3, 4.0),
        "tied": (0.3, 4.0),
        "dominated": (0.8, 4.0),
        "same_energy_worse": (0.9, 1.0),
    }
    assert training.front(points) == ["small", "large", "tied"]


def test_training_matched_panel_and_orientation(training, tmp_path):
    points = training.load_csv(
        write_csv(tmp_path / "panel.csv", training_rows()), [15, 300]
    )
    assert set(points[15]) == set(points[300]) == {"small", "large"}
    assert points[15]["small"] == (0.8, 1.0)  # RMSE first; energy second.


@pytest.mark.parametrize(
    "defect", ["missing", "duplicate", "nan", "infinite_energy", "changing_energy"]
)
def test_training_csv_rejects_invalid_scientific_panels(training, tmp_path, defect):
    rows = training_rows()
    if defect == "missing":
        rows.pop()
    elif defect == "duplicate":
        rows.append(rows[0].copy())
    elif defect == "nan":
        rows[0]["rmse_total"] = "nan"
    elif defect == "infinite_energy":
        rows[0]["energy_mj"] = "inf"
    else:
        rows[-1]["energy_mj"] = 5.0
    with pytest.raises(ValueError):
        training.load_csv(write_csv(tmp_path / "panel.csv", rows), [15, 300])


def json_rows():
    return [
        dict(
            candidate_id=r["candidate_id"],
            seed=17,
            budget=r["budget"],
            policy="budget_only",
            status="ok",
            energy_policy="fixed_historical_measurement",
            energy_mj_per_inference=r["energy_mj"],
            metrics={"rmse_total": r["rmse_total"]},
        )
        for r in training_rows()
    ]


@pytest.mark.parametrize(
    "defect", [None, "missing", "duplicate", "nan", "changing_energy", "wrong_policy"]
)
def test_training_json_protocol_validation(training, tmp_path, defect):
    rows = json_rows()
    if defect == "missing":
        rows.pop()
    elif defect == "duplicate":
        rows.append(rows[0].copy())
    elif defect == "nan":
        rows[0]["metrics"]["rmse_total"] = float("nan")
    elif defect == "changing_energy":
        rows[-1]["energy_mj_per_inference"] = 5.0
    elif defect == "wrong_policy":
        rows[0]["energy_policy"] = "new_measurement"
    path = tmp_path / "results.json"
    path.write_text(json.dumps(rows))
    if defect is None:
        assert training.load([path], [15, 300], 17)[300]["large"] == (0.3, 4.0)
    else:
        with pytest.raises(ValueError):
            training.load([path], [15, 300], 17)


def proposal_tables():
    # Highest F1 and lowest energy are deliberately different from score winner.
    rows = []
    endpoints = []
    for board in ("STM32", "Portenta"):
        for method in ("Optuna", "Memory", "History"):
            for trial, f1, energy in [(0, 0.9, 1000.0), (1, 0.8, 100.0)]:
                rows.append(
                    dict(
                        board=board,
                        method=method,
                        trial=trial,
                        f1=f1,
                        energy_mj=energy,
                        score=f1 - energy / 4000,
                        latency_ms=2.0,
                    )
                )
            endpoints.append(
                dict(study=f"{board.lower()}_{method.lower()}", best_individual=0.775)
            )
    return rows, endpoints


def test_proposal_selects_score_maximum_without_dropping_repeats(proposal, tmp_path):
    rows, endpoints = proposal_tables()
    observations, winners = proposal.load(
        write_csv(tmp_path / "points.csv", rows),
        write_csv(tmp_path / "endpoints.csv", endpoints),
        2,
    )
    assert len(observations) == 12
    assert len(winners) == 6
    assert all(
        winner["trial"] == 1 and winner["score"] == pytest.approx(0.775)
        for winner in winners.values()
    )


@pytest.mark.parametrize(
    "defect",
    ["wrong_score", "missing_trial", "duplicate_trial", "nonfinite", "wrong_endpoint"],
)
def test_proposal_rejects_invalid_scores_and_counts(proposal, tmp_path, defect):
    rows, endpoints = proposal_tables()
    if defect == "wrong_score":
        rows[0]["score"] += 0.01
    elif defect == "missing_trial":
        rows.pop()
    elif defect == "duplicate_trial":
        rows[1]["trial"] = rows[0]["trial"]
    elif defect == "nonfinite":
        rows[0]["energy_mj"] = "nan"
    else:
        endpoints[0]["best_individual"] = 0.9
    with pytest.raises(ValueError):
        proposal.load(
            write_csv(tmp_path / "points.csv", rows),
            write_csv(tmp_path / "endpoints.csv", endpoints),
            2,
        )


@pytest.fixture(scope="module")
def cs2():
    return module("plot_case2_proxy_replay_fronts")


def cs2_write_rows(path, rows):
    """Write heterogeneous saved-artifact rows with all alias columns present."""
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_cs2_loaders_preserve_alias_precedence_and_exclusions(cs2, tmp_path):
    import replay_front_support as support

    crest = dict(
        state="complete",
        error_code=1,
        number="first",
        values_energy_mj_per_inference=2,
        energy_mj_per_inference=99,
        values_rmse_total=0.4,
        metric__rmse_total=9,
        values_latency_ms=220,
        values_latency_budget_ms=200,
    )
    crest_rows = [
        crest,
        dict(
            crest,
            number="fallback",
            values_energy_mj_per_inference="nan",
            values_rmse_total="bad",
            metric__rmse_total=0.3,
        ),
        dict(crest, state="failed"),
        dict(crest, error_code=2),
        dict(crest, values_energy_mj_per_inference=0),
        dict(crest, values_rmse_total="nan", metric__rmse_total="inf"),
    ]
    points = support.load_crest_points(
        cs2_write_rows(tmp_path / "crest.csv", crest_rows)
    )
    assert [
        (p.payload_key, p.energy_mj, p.rmse, p.latency_feasible) for p in points
    ] == [("first", 2, 0.4, False), ("fallback", 99, 0.3, False)]
    replay = dict(
        replay_status="completed",
        target__error_code=1,
        replay_payload_key="primary",
        payload_key="secondary",
        target__energy_mj_per_inference=3,
        target__cadenced_energy_mj_per_window=7,
        source__metric__rmse_total=0.2,
        source__rmse_total=0.8,
        target__metric__rmse_total=0.9,
        target__latency_ms=100,
        target__latency_budget_ms=200,
    )
    rows = [
        replay,
        dict(
            replay,
            replay_payload_key="",
            target__energy_mj_per_inference="nan",
            source__metric__rmse_total="",
            target__latency_ms="",
            target__cadenced_active_inference_latency_ms=250,
        ),
        dict(replay, replay_status="failed"),
        dict(replay, target__error_code=""),
        dict(replay, target__error_code=2),
        dict(replay, target__energy_mj_per_inference=-1),
        dict(
            replay,
            source__metric__rmse_total="nan",
            source__rmse_total="nan",
            target__metric__rmse_total="nan",
        ),
    ]
    points = support.load_replay_points(cs2_write_rows(tmp_path / "replay.csv", rows))
    assert [
        (p.payload_key, p.energy_mj, p.rmse, p.latency_feasible) for p in points
    ] == [("primary", 3, 0.2, True), ("secondary", 7, 0.8, False)]


def cs2_pairs(cs2):
    import replay_front_support as support

    points = [
        support.FrontPoint(
            key,
            energy,
            rmse,
            feasible,
            dict(
                latency_ms=latency, latency_budget_ms=200, cpu_clock_mhz_requested=clock
            ),
        )
        for key, energy, rmse, feasible, latency, clock in [
            ("fast", 1, 0.7, True, 100, 200),
            ("slow", 2, 0.4, False, 250, 400),
            ("tie", 2, 0.4, None, "", 800),
            ("dominated", 3, 0.8, False, 300, 300),
        ]
    ]
    replay = [
        support.FrontPoint(
            "proxy",
            4,
            0.5,
            True,
            dict(target__latency_ms=150, target__latency_budget_ms=200),
        )
    ]
    return [
        support.LoadedPair(
            label,
            points,
            replay,
            support.pareto_front(points),
            replay,
            [],
            support.summarize_point_outcomes(support.pareto_front(points)),
            support.CandidateOutcomeSummary(3, 1, 0, 1, 1, 0),
        )
        for label in ("Replay target", "Measured only")
    ]


def test_cs2_pareto_ties_and_latency_marking_are_independent(cs2):
    import replay_front_support as support
    import matplotlib.pyplot as plt

    pair = cs2_pairs(cs2)[0]
    assert [p.payload_key for p in pair.crest_front] == ["fast", "slow", "tie"]
    assert [
        p.payload_key for p in support.filter_latency_feasible(pair.crest_points, True)
    ] == ["fast"]
    fig, ax = plt.subplots()
    cs2.scatter_latency_points(
        ax,
        pair.crest_points,
        color="blue",
        marker="o",
        size=12,
        alpha=0.5,
        linewidth=0.2,
        zorder=2,
    )
    assert [c.get_offsets().tolist() for c in ax.collections] == [
        [[1, 0.7], [2, 0.4]],
        [[2, 0.4], [3, 0.8]],
    ]
    assert ax.collections[1].get_facecolors()[0, :3].tolist() == [1, 1, 1]
    assert ax.collections[1].get_alpha() == 0.9
    plt.close(fig)


def test_cs2_sidecars_preserve_numeric_and_measured_only_semantics(cs2, tmp_path):
    pairs = cs2_pairs(cs2)
    omitted = frozenset({"Measured only"})
    points = tmp_path / "points.csv"
    summary = tmp_path / "summary.csv"
    text = tmp_path / "summary.txt"
    cs2.write_plotted_points(points, pairs, pairs, omitted)
    cs2.write_summary_csv(summary, pairs, omitted)
    cs2.write_summary(text, pairs, pairs, omitted)
    with points.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 10
    assert {r["series"] for r in rows if r["target"] == "Measured only"} == {
        "measured_energy_nas"
    }
    unknown = next(r for r in rows if r["payload_key"] == "tie")
    assert unknown["latency_feasible"] == unknown["latency_ms"] == ""
    assert unknown["is_front"] == "True"
    assert float(unknown["energy_mj"]) == 2 and int(unknown["cpu_clock_mhz"]) == 800
    with summary.open() as stream:
        summaries = list(csv.DictReader(stream))
    assert summaries[0]["replay_total_candidates"] == "3"
    assert all(
        value == "" for key, value in summaries[1].items() if key.startswith("replay_")
    )
    assert "Scheduled replay target measurements: 3" in text.read_text()
    assert (
        "proxy replays omitted from this figure (not plotted, excluded from totals)"
        in text.read_text()
    )


def timing_inputs(directory):
    """Write reconciled synthetic timing evidence with known per-candidate costs."""
    directory.mkdir()
    conditions = {}
    for condition in ("continuous", "cadenced", "flops", "memory"):
        conditions[condition] = {
            "trained_candidates": 10,
            "epochs": 550,
            "categories_s": {
                "loading": 60,
                "training_fit": 600,
                "post_fit_evaluation": 120,
                "hil_request": 360,
            },
            "client_process_s": 1140,
            "hil_detail_s": {
                "runtime_protocol_measurement_wait": 120,
                "programming_and_boot_loading": 60,
                "firmware_build_signing": 120,
            },
        }
    (directory / "breakdown.json").write_text(json.dumps({"conditions": conditions}))
    (directory / "arena_search.json").write_text(
        json.dumps(
            {
                "conditions": {
                    condition: {"failed_probe_measurement_s": 30, "candidates": 10}
                    for condition in ("continuous", "cadenced")
                }
            }
        )
    )
    write_csv(
        directory / "proxy_times.csv",
        [
            {"stage": stage, "inclusive_s": duration}
            for stage, duration in (("proxy.flops", 1), ("proxy.memory_estimate", 2))
            for _ in range(40)
        ],
    )
    return directory


def test_timing_attribution_preserves_units_and_timeout_partition(tmp_path):
    """Timeouts partition measurement time without adding to the HIL total."""
    panel = module("plot_appendix_timing_analysis").load_panel(
        timing_inputs(tmp_path / "timing")
    )
    assert panel["host_min"] == pytest.approx(1.3)
    assert panel["training_min"] == pytest.approx(1)
    assert panel["setup_min"] == pytest.approx(0.1)
    assert panel["evaluation_min"] == pytest.approx(0.2)
    assert panel["flops_s"] == pytest.approx(1)
    assert panel["memory_traffic_s"] == pytest.approx(2)
    for condition in ("continuous", "cadenced"):
        segments = panel["hil_segments_min"][condition]
        assert list(segments) == [
            "compile_prepare",
            "flash_boot",
            "arena_timeouts",
            "measurement_handshake",
            "other",
        ]
        assert list(segments.values()) == pytest.approx([0.2, 0.1, 0.05, 0.15, 0.1])
        assert sum(segments.values()) == pytest.approx(panel[f"{condition}_hil_min"])
        assert panel[f"{condition}_hil_min"] == pytest.approx(0.6)


@pytest.mark.parametrize(
    "defect, message",
    [
        ("count", "campaign size"),
        ("reconciliation", "categories do not reconcile"),
        ("proxy", "finite samples"),
        ("timeout", "timeout time exceeds"),
        ("overhead", "segments do not reconcile"),
    ],
)
def test_timing_rejects_inconsistent_attribution(tmp_path, defect, message):
    """Malformed evidence cannot silently change timing denominators or segments."""
    directory = timing_inputs(tmp_path / "timing")
    if defect == "proxy":
        write_csv(
            directory / "proxy_times.csv",
            [
                {"stage": "proxy.flops", "inclusive_s": "nan"},
            ],
        )
    elif defect == "timeout":
        path = directory / "arena_search.json"
        data = json.loads(path.read_text())
        data["conditions"]["continuous"]["failed_probe_measurement_s"] = 121
        path.write_text(json.dumps(data))
    else:
        path = directory / "breakdown.json"
        data = json.loads(path.read_text())
        record = data["conditions"]["continuous"]
        if defect == "count":
            record["trained_candidates"] = 9
        elif defect == "reconciliation":
            record["client_process_s"] += 1
        else:
            record["hil_detail_s"]["firmware_build_signing"] = 1000
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=message):
        module("plot_appendix_timing_analysis").load_panel(directory)
