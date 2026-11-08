# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Scientific regression checks for the three standalone CS1 paper figures."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "analysis_scripts" / "paper_plots"


def load_plot(name: str) -> ModuleType:
    """Import one plot module without invoking its command line interface.

    Parameters
    ----------
    name : str
        Figure suffix.

    Returns
    -------
    types.ModuleType
        Imported plotting module.
    """
    spec = importlib.util.spec_from_file_location(
        name, SCRIPT_DIR / f"plot_case1_{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clock_rounds() -> pd.DataFrame:
    """Create rounds with distinct endpoint deltas to expose positional pairing.

    Returns
    -------
    pandas.DataFrame
        Complete two-period, three-round policy measurements.
    """
    rows = []
    for candidate, clock in (
        ("continuous_203", 800),
        ("continuous_203", 300),
        ("continuous_137", 300),
    ):
        for period in (100, 500):
            for round_id in (1, 2, 3):
                reference = 20 + period / 10 + round_id
                delta = 0 if clock == 800 else round_id * (1 if period == 100 else -2)
                rows.append(
                    dict(
                        candidate=candidate,
                        clock_mhz=clock,
                        period_ms=period,
                        round=round_id,
                        E_stop_mj=reference + delta,
                        E_inf_mj=10 + clock / 100,
                        C_s=0.01,
                    )
                )
    return pd.DataFrame(rows)


def test_shuffled_rounds_preserve_interpolated_uncertainty(tmp_path: Path) -> None:
    """Round order must not affect paired intermediate confidence intervals."""
    module = load_plot("clock_reversal")
    source = tmp_path / "rounds.csv"
    frame = clock_rounds()
    frame.to_csv(source, index=False)
    module.main(["--input-csv", str(source), "--output-dir", str(tmp_path / "ordered")])
    frame.sample(frac=1, random_state=27).to_csv(source, index=False)
    module.main(
        ["--input-csv", str(source), "--output-dir", str(tmp_path / "shuffled")]
    )
    name = "cs1_clock_reversal_plotted_points.csv"
    first = pd.read_csv(tmp_path / "ordered" / name)
    second = pd.read_csv(tmp_path / "shuffled" / name)
    pd.testing.assert_frame_equal(first, second, rtol=1e-12, atol=1e-12)
    midpoint = first[(first.kind == "interpolated_mean") & (first.period_ms == 300)]
    # Per-round midpoint differences are [-.5, -1, -1.5], not independent endpoints.
    assert np.allclose(midpoint.difference_mj, -1)
    from scipy.stats import t

    half = t.ppf(0.975, 2) * 0.5 / np.sqrt(3)
    assert np.allclose(midpoint.ci95_low_mj, -1 - half)
    assert np.allclose(midpoint.ci95_high_mj, -1 + half)


@pytest.mark.parametrize(
    "missing_index, message",
    [(0, "complete paired rounds"), (6, "round IDs must match")],
)
def test_missing_round_pair_is_rejected(
    tmp_path: Path, missing_index: int, message: str
) -> None:
    """A missing endpoint cannot silently shrink or misalign paired rounds."""
    source = tmp_path / "incomplete.csv"
    clock_rounds().drop(index=missing_index).to_csv(source, index=False)
    with pytest.raises(ValueError, match=message):
        load_plot("clock_reversal").main(
            ["--input-csv", str(source), "--output-dir", str(tmp_path / "out")]
        )


def test_energy_ranks_average_ties_within_search_pool(tmp_path: Path) -> None:
    """Ties use average ranks, independently in each period and search pool."""
    rows = []
    for period in (100, 200, 500):
        for study in ("continuous", "cadenced"):
            for trial, (continuous, scheduled) in enumerate(((1, 3), (1, 2), (3, 2))):
                rows.append(
                    dict(
                        pool=f"{period}ms_{study}",
                        period_ms=period,
                        study=study,
                        trial=trial,
                        continuous_mj=continuous,
                        scheduled_mj=scheduled,
                    )
                )
    source = tmp_path / "candidates.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    load_plot("energy_ranks").main(
        ["--input-csv", str(source), "--output-dir", str(tmp_path / "out")]
    )
    points = pd.read_csv(tmp_path / "out" / "cs1_energy_ranks_plotted_points.csv")
    assert len(points) == 18
    for _, group in points.groupby(["period_ms", "study"]):
        assert group.continuous_rank.tolist() == [1.5, 1.5, 3]
        assert group.periodic_rank.tolist() == [3, 1.5, 1.5]
    summary = json.loads(
        (tmp_path / "out" / "cs1_energy_ranks_summary.json").read_text()
    )
    assert [group["n"] for group in summary["groups"]] == [3] * 6


def test_model_transfer_population_and_signed_residuals(tmp_path: Path) -> None:
    """Only the training pool is excluded; residuals are measured minus predicted."""
    rows = []
    for period in (100, 200, 500):
        for study in ("continuous", "cadenced"):
            for trial, residual in enumerate((-3.0, 5.0)):
                rows.append(
                    dict(
                        pool=f"{period}ms_{study}",
                        trial=trial,
                        study=study,
                        period_ms=period,
                        continuous_mj=30,
                        scheduled_mj=20 + trial,
                        E_pred=20 + trial - residual,
                    )
                )
    frame = pd.DataFrame(rows)
    source = tmp_path / "transfer.csv"
    frame.to_csv(source, index=False)
    output = tmp_path / "out"
    load_plot("model_transfer").main(
        ["--input-csv", str(source), "--output-dir", str(output)]
    )
    points = pd.read_csv(output / "cs1_model_transfer_plotted_points.csv")
    expected = frame.loc[frame.pool != "200ms_cadenced"].copy()
    expected["error_mj"] = [-3.0, 5.0] * 5
    pd.testing.assert_frame_equal(
        points, expected.reset_index(drop=True), check_dtype=False
    )
    summary = json.loads((output / "cs1_model_transfer_summary.json").read_text())
    assert "excludes 200ms_cadenced" in summary["population"]
    assert [(g["period_ms"], g["study"]) for g in summary["groups"]] == [
        (100, "continuous"),
        (100, "cadenced"),
        (200, "continuous"),
        (500, "continuous"),
        (500, "cadenced"),
    ]
    for group in summary["groups"]:
        assert group["n"] == 2
        assert group["mean_error_mj"] == 1.0
        assert group["rmse_mj"] == pytest.approx(np.sqrt(17))
